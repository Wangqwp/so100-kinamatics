#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ROS 2 node publishing the SO-101 end-effector pose.

The node subscribes to ``/so100/joint_states`` (``sensor_msgs/JointState``),
feeds the five arm joints into the forward kinematics of
:mod:`so100_kinematics` and publishes the resulting pose of the grasp point
as ``geometry_msgs/PoseStamped`` on ``/so100/ee_pose``.

Base frame
----------
The pose is computed **relative to ``base_link``**: the kinematic chain is
rooted in that frame, so the published position and orientation are the ones
seen in the robot base - and the same ones the ``ee.*`` numbers of the
LeRobot backend use (for this URDF ``base_link`` *is* the root of the
description).  The frame the pose is expressed in travels with the message
in ``header.frame_id``; it can be changed with the ``base_frame`` parameter,
which has to be a link of the URDF (an empty value falls back to the root
of the URDF).

Units
-----
``/so100/joint_states`` publishes the joint angles in **degrees** - the unit
the SO-101 firmware and the LeRobot bridge use - while the ROS convention
would be radians.  :mod:`so100_kinematics` interprets the joint values as
degrees as well, so the node hands the values over unchanged.  The
``joint_state_unit`` parameter selects what the node assumes:

``deg`` (the default)
    the source publishes degrees - what ``/so100/joint_states`` does.
``rad``
    the source follows the ROS convention and publishes radians; the node
    converts them to degrees.
``auto``
    the node guesses the unit from the values themselves: a joint angle
    above pi cannot be a radian value of the SO-101, whose joints are all
    limited to less than pi radians.

Whatever the unit is, the node checks the angles against the limits of the
URDF and warns when a joint is far outside of them - that is what a mix-up
of the units looks like (the value is off by a factor of 57).

Run it with::

    source /opt/ros/humble/setup.bash
    python3 ee_pose_node.py
    # or, with parameters:
    python3 ee_pose_node.py --ros-args -p joint_states_topic:=/so100/joint_states
"""

from __future__ import annotations

import os
import sys
import time
from typing import Sequence

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState

# Make the sibling module importable no matter what the working directory is
# (``ros2 run`` starts the node from an arbitrary cwd).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from so100_kinematics import (  # noqa: E402  (the sys.path tweak must come first)
    ARM_JOINT_NAMES,
    DEFAULT_BASE_FRAME,
    DEFAULT_JOINT_UNIT,
    DEFAULT_TARGET_FRAME,
    DEFAULT_URDF_PATH,
    JOINT_UNIT_AUTO,
    KinematicsError,
    So100Kinematics,
    detect_joint_unit,
    joint_values_to_degrees,
    matrix_to_quaternion,
    normalize_joint_unit,
)


class EePoseNode(Node):
    """Publish the end-effector pose of the SO-101 arm."""

    def __init__(
        self, parameter_overrides: Sequence[Parameter] | None = None
    ) -> None:
        super().__init__("so100_ee_pose", parameter_overrides=parameter_overrides)

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.declare_parameter("joint_states_topic", "/so100/joint_states")
        self.declare_parameter("ee_pose_topic", "/so100/ee_pose")
        self.declare_parameter("urdf_path", DEFAULT_URDF_PATH)
        self.declare_parameter("target_frame", DEFAULT_TARGET_FRAME)
        self.declare_parameter("base_frame", DEFAULT_BASE_FRAME)
        self.declare_parameter("joint_state_unit", DEFAULT_JOINT_UNIT)

        self._joint_states_topic = str(self.get_parameter("joint_states_topic").value)
        self._ee_pose_topic = str(self.get_parameter("ee_pose_topic").value)
        self._base_frame = str(self.get_parameter("base_frame").value).strip()
        try:
            self._unit = normalize_joint_unit(
                str(self.get_parameter("joint_state_unit").value)
            )
        except KinematicsError as exc:
            self.get_logger().warn(
                f"{exc}, falling back to {DEFAULT_JOINT_UNIT!r}"
            )
            self._unit = DEFAULT_JOINT_UNIT
        #: The unit the last joint state was read as (``None`` until the
        #: first message arrives) - only used by the ``auto`` mode.
        self._detected_unit: str | None = None

        # ------------------------------------------------------------------
        # Kinematics
        # ------------------------------------------------------------------
        # An empty base frame means "the root of the URDF"; everything else
        # has to be a link of the URDF the chain may start in.
        base_frame = self._base_frame or None
        try:
            self._kinematics = So100Kinematics(
                urdf_path=str(self.get_parameter("urdf_path").value),
                target_frame=str(self.get_parameter("target_frame").value),
                base_frame=base_frame,
            )
        except KinematicsError as exc:
            self.get_logger().fatal(
                f"the kinematics cannot be set up "
                f"(base frame {base_frame!r}): {exc}"
            )
            raise

        # The pose is computed in the base frame, so that is the frame the
        # message has to carry in its header.
        self._pose_frame = self._kinematics.pose_frame

        # ------------------------------------------------------------------
        # ROS interfaces
        # ------------------------------------------------------------------
        self._publisher = self.create_publisher(PoseStamped, self._ee_pose_topic, 10)
        self._subscription = self.create_subscription(
            JointState, self._joint_states_topic, self._on_joint_states, 10
        )

        self._last_warning = 0.0

        self.get_logger().info(
            f"subscribing to {self._joint_states_topic} "
            f"(unit: {self._unit}), publishing {self._ee_pose_topic} "
            f"in {self._pose_frame}"
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _on_joint_states(self, msg: JointState) -> None:
        """Compute and publish the end-effector pose of the received state."""
        joint_action = self._joint_action(msg)
        if joint_action is None:
            self._warn_throttled(
                "the joint state does not contain all the arm joints "
                f"({', '.join(ARM_JOINT_NAMES)}), skipping"
            )
            return

        outside = self._kinematics.out_of_range(joint_action)
        if outside:
            self._warn_throttled(
                f"the joint values of {', '.join(outside)} are outside their "
                "limit - the unit of the joint state is probably wrong"
            )

        try:
            position, rotation = self._kinematics.end_effector(joint_action)
        except Exception as exc:  # pragma: no cover - defensive
            self._warn_throttled(f"the forward kinematics failed: {exc}")
            return

        x, y, z, w = matrix_to_quaternion(rotation)

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self._pose_frame
        pose.pose.position.x = float(position[0])
        pose.pose.position.y = float(position[1])
        pose.pose.position.z = float(position[2])
        pose.pose.orientation.x = x
        pose.pose.orientation.y = y
        pose.pose.orientation.z = z
        pose.pose.orientation.w = w

        self._publisher.publish(pose)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _joint_action(self, msg: JointState) -> dict[str, float] | None:
        """The joint state as the ``{"<joint>.pos": degrees}`` command.

        Returns ``None`` when one of the arm joints is missing from the
        message.
        """
        measured = {
            name: float(position)
            for name, position in zip(msg.name, msg.position)
        }

        selected: dict[str, float] = {}
        for joint in self._kinematics.joint_names:
            value = measured.get(joint)
            if value is None:
                # tolerate a "left_"/"right_" prefix (e.g. "left_shoulder_pan")
                for name, position in measured.items():
                    if name.endswith(joint):
                        value = position
                        break
            if value is None:
                return None
            selected[joint] = value

        unit = self._unit
        if unit == JOINT_UNIT_AUTO:
            unit = detect_joint_unit(selected.values())
            if unit != self._detected_unit:
                self._detected_unit = unit
                self.get_logger().info(
                    f"the joint state is interpreted as {unit!r} "
                    "(the values say so)"
                )

        degrees = joint_values_to_degrees(selected, unit)
        return {
            f"{joint}.pos": degrees[joint]
            for joint in self._kinematics.joint_names
        }

    def _warn_throttled(self, message: str, period: float = 2.0) -> None:
        """Log ``message`` at most once every ``period`` seconds."""
        now = time.monotonic()
        if now - self._last_warning >= period:
            self._last_warning = now
            self.get_logger().warn(message)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = EePoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

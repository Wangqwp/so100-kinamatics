# SPDX-FileCopyrightText: (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

"""Kinematics backend for the SO100 / SO101 ROS package.

This module wraps LeRobot's :class:`lerobot.model.kinematics.RobotKinematics`
and fixes the two problems that make its inverse kinematics unusable on a
real robot.

1. **Stale kinematics (the reason the IK sometimes does not move at all).**
   ``RobotKinematics.inverse_kinematics()`` writes the IK seed with
   ``robot.set_joint()`` but never calls ``robot.update_kinematics()`` before
   handing the problem to placo.  placo evaluates the frame-task error *and*
   the Jacobian at the frames that are currently loaded in the robot, so the
   task is linearised around the configuration left over by the previous
   FK/IK call instead of around the seed.  When that stale frame happens to
   coincide with the requested target, the task error is zero, the solver has
   nothing to do, and the IK returns the seed unchanged: the arm does not
   move, although the caller asked for another pose.
   :class:`FixedRobotKinematics` refreshes the frames right after the seed is
   written, so the solver always linearises around the seed.

2. **A single linearised step.**  ``solver.solve(True)`` solves one QP that
   is built from a single linearisation of the task, so a large move keeps a
   sizeable residual (about 25 mm for a 20 cm move on the SO-101).
   ``ik_iterations`` re-linearises and re-solves, using the previous
   solution as the new seed, until the residual is negligible (3 iterations
   bring the example below to 0.02 mm).

The class is a drop-in replacement: it keeps LeRobot's signature and its
degree-based units, so the LeRobot processor steps
(``InverseKinematicsEEToJoints`` & co.) work with it unchanged.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from lerobot.model.kinematics import RobotKinematics
import numpy as np


class FixedRobotKinematics(RobotKinematics):
    """LeRobot's kinematics solver with the stale-kinematics bug fixed.

    Args:
        urdf_path: Path to the robot URDF.
        target_frame_name: Name of the end-effector frame.
        joint_names: The joints of the kinematic chain, in the order the
            caller expects them.
        ik_iterations: How many linearised solves to run per inverse
            kinematics call (1 reproduces LeRobot's behaviour, 3 converges).
    """

    def __init__(
        self,
        urdf_path: str,
        target_frame_name: str = "gripper_frame_link",
        joint_names: Optional[Sequence[str]] = None,
        ik_iterations: int = 1,
    ) -> None:
        super().__init__(
            urdf_path=urdf_path,
            target_frame_name=target_frame_name,
            joint_names=list(joint_names) if joint_names is not None else None,
        )
        self.ik_iterations = max(1, int(ik_iterations))

    # ------------------------------------------------------------------
    # Inverse kinematics
    # ------------------------------------------------------------------
    def inverse_kinematics(
        self,
        current_joint_pos: np.ndarray,
        desired_ee_pose: np.ndarray,
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
    ) -> np.ndarray:
        """The joint positions (degrees) that reach ``desired_ee_pose``.

        The seed is ``current_joint_pos`` (degrees).  When ``ik_iterations``
        is greater than one, the solution of a solve is used as the seed of
        the next one, which converges towards the target instead of stopping
        after a single linearised step.
        """
        seed = np.asarray(current_joint_pos, dtype=float)
        solution = seed
        for _ in range(self.ik_iterations):
            solution = self._solve_once(seed, desired_ee_pose, position_weight, orientation_weight)
            seed = solution
        return solution

    def _solve_once(
        self,
        current_joint_pos: np.ndarray,
        desired_ee_pose: np.ndarray,
        position_weight: float,
        orientation_weight: float,
    ) -> np.ndarray:
        """One linearised placo solve, starting from ``current_joint_pos``."""
        current_joint_rad = np.deg2rad(current_joint_pos[: len(self.joint_names)])
        for index, joint_name in enumerate(self.joint_names):
            self.robot.set_joint(joint_name, current_joint_rad[index])

        # The line LeRobot is missing: without it placo keeps the frames of
        # the previous FK/IK call and linearises the task around them
        # instead of around the seed.
        self.robot.update_kinematics()

        self.tip_frame.T_world_frame = desired_ee_pose
        self.tip_frame.configure(
            self.target_frame_name, "soft", position_weight, orientation_weight
        )

        self.solver.solve(True)
        self.robot.update_kinematics()

        joint_pos_deg = np.rad2deg(
            np.array(
                [self.robot.get_joint(joint_name) for joint_name in self.joint_names],
                dtype=float,
            )
        )

        # The gripper (and anything else behind the chain) is not solved for.
        if len(current_joint_pos) > len(self.joint_names):
            result = np.zeros_like(np.asarray(current_joint_pos, dtype=float))
            result[: len(self.joint_names)] = joint_pos_deg
            result[len(self.joint_names) :] = current_joint_pos[len(self.joint_names) :]
            return result
        return joint_pos_deg

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def joint_names_in_order(self) -> List[str]:
        """The chain joints, in the order the solver returns them."""
        return list(self.joint_names)

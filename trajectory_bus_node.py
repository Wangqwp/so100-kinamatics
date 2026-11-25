#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""单 topic 轨迹节点：数据获取、计算、发布都走同一个 nav_msgs/Path topic。

协议
----
唯一话题由 ``trajectory_topic`` 指定，默认 ``/so100/trajectory``。

输入（订阅）用 ``header.frame_id`` 区分两种数据：

``joint_input``
    ``poses[i]`` 编码一个 action chunk 点：
      * ``position.x/y/z`` = shoulder_pan / shoulder_lift / elbow_flex（度）
      * ``orientation.x/y`` = wrist_flex / wrist_roll（度）
      * ``orientation.z``  = gripper（0..100）
      * ``orientation.w``  = 0（标记位，不参与计算）
    ``poses[i].header.stamp`` 可写该点相对 chunk 起点的时间；全零时按
    ``waypoint_dt_s`` 递增。

``pose_input``
    前两个 pose 分别是「夹爪尖端」和「孔」的位姿（真实四元数）。
    坐标默认已经在 ``base_frame`` 下。

输出（发布）也是同一个 topic，但 ``header.frame_id == base_frame``，
``poses`` 是可以直接给 RViz 显示 / 控制器使用的 EE（或夹爪控制点）轨迹。
节点会**忽略帧号等于 base_frame 的流入消息**，因此不会自激循环。

运行::

    ros2 run so100_trajectory trajectory_bus
"""

from __future__ import annotations

import os
import sys
import time
from typing import List, Optional, Sequence

import numpy as np

import rclpy
from builtin_interfaces.msg import Time as RosTime
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.parameter import Parameter

# ``ros2 run`` 会从任意 cwd 启动，先把本目录放进 sys.path。
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hole_insertion_geometry import (  # noqa: E402
    APPROACH_HEIGHT_M,
    BASE_OFFSET_AXIS,
    BASE_OFFSET_M,
    HOLE_RADIUS_M,
    INSERT_DEPTH_M,
    LIFT_MARGIN_M,
    RADIAL_CLEARANCE_M,
    WAYPOINT_SPACING_M,
    PlanConfig,
    plan_insertion,
)
from temp.so100_kinematics import (  # noqa: E402
    ARM_JOINT_NAMES,
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_TARGET_FRAME,
    DEFAULT_URDF_PATH,
    JOINT_UNIT_NORM,
    KinematicsError,
    So100Kinematics,
    load_joint_calibration,
    matrix_to_quaternion,
    normalize_joint_unit,
    rotvec_to_matrix,
)
from utils.trajectory_smoothing import (  # noqa: E402
    SMOOTH_METHOD_SAVGOL,
    normalize_smooth_method,
    smooth_joint_positions,
)

FRAME_JOINT_INPUT = "joint_input"
FRAME_POSE_INPUT = "pose_input"
DEFAULT_TRAJECTORY_TOPIC = "/so100/trajectory"
DEFAULT_BASE_FRAME = "base_link"
DEFAULT_WAYPOINT_DT_S = 0.05

GRIPPER_JOINT_NAME = "gripper"


def stamp_to_seconds(stamp: RosTime) -> float:
    """``builtin_interfaces/Time`` -> 秒。"""
    return int(stamp.sec) + int(stamp.nanosec) * 1e-9


def stamp_plus_seconds(stamp: RosTime, seconds: float) -> RosTime:
    """``stamp + seconds``，纳秒精度。"""
    total = (
        int(stamp.sec) * 1_000_000_000
        + int(stamp.nanosec)
        + int(round(float(seconds) * 1_000_000_000))
    )
    result = RosTime()
    result.sec = int(total // 1_000_000_000)
    result.nanosec = int(total % 1_000_000_000)
    return result


class TrajectoryBusNode(Node):
    """同一个 topic 上读输入、算轨迹、写结果。"""

    def __init__(
        self, parameter_overrides: Optional[Sequence[Parameter]] = None
    ) -> None:
        super().__init__("trajectory_bus", parameter_overrides=parameter_overrides)

        self.declare_parameter("trajectory_topic", DEFAULT_TRAJECTORY_TOPIC)
        self.declare_parameter("base_frame", DEFAULT_BASE_FRAME)
        self.declare_parameter("urdf_path", DEFAULT_URDF_PATH)
        self.declare_parameter("calibration_path", DEFAULT_CALIBRATION_PATH)
        self.declare_parameter("action_unit", JOINT_UNIT_NORM)
        self.declare_parameter("smooth_method", SMOOTH_METHOD_SAVGOL)
        self.declare_parameter("smooth_window", 11)
        self.declare_parameter("smooth_polyorder", 2)
        self.declare_parameter("smooth_sigma", 1.0)
        self.declare_parameter("waypoint_dt_s", DEFAULT_WAYPOINT_DT_S)

        # 孔位插入几何参数
        self.declare_parameter("approach_height_m", APPROACH_HEIGHT_M)
        self.declare_parameter("insert_depth_m", INSERT_DEPTH_M)
        self.declare_parameter("base_offset_m", BASE_OFFSET_M)
        self.declare_parameter("base_offset_axis", BASE_OFFSET_AXIS)
        self.declare_parameter("lift_margin_m", LIFT_MARGIN_M)
        self.declare_parameter("waypoint_spacing_m", WAYPOINT_SPACING_M)
        self.declare_parameter("hole_radius_m", HOLE_RADIUS_M)
        self.declare_parameter("radial_clearance_m", RADIAL_CLEARANCE_M)

        def text(name: str) -> str:
            return str(self.get_parameter(name).value).strip()

        self._topic = text("trajectory_topic") or DEFAULT_TRAJECTORY_TOPIC
        self._base_frame = text("base_frame") or DEFAULT_BASE_FRAME
        self._waypoint_dt_s = max(
            0.001, float(self.get_parameter("waypoint_dt_s").value)
        )

        try:
            self._unit = normalize_joint_unit(text("action_unit"))
        except KinematicsError as exc:
            self.get_logger().warn(f"{exc}，退回 {JOINT_UNIT_NORM!r}")
            self._unit = JOINT_UNIT_NORM
        try:
            self._smooth_method = normalize_smooth_method(
                text("smooth_method")
            )
        except ValueError as exc:
            self.get_logger().warn(f"{exc}，不做平滑")
            self._smooth_method = "none"
        self._smooth_window = max(1, int(self.get_parameter("smooth_window").value))
        self._smooth_polyorder = max(
            0, int(self.get_parameter("smooth_polyorder").value)
        )
        self._smooth_sigma = float(self.get_parameter("smooth_sigma").value)

        calibration_path = text("calibration_path")
        self._calibration = None
        if calibration_path:
            try:
                self._calibration = load_joint_calibration(calibration_path)
            except (KinematicsError, OSError) as exc:
                self.get_logger().warn(f"标定文件无法读取：{exc}")

        try:
            self._urdf = So100Kinematics(
                urdf_path=text("urdf_path"),
                target_frame=DEFAULT_TARGET_FRAME,
                joint_names=ARM_JOINT_NAMES,
                calibration=self._calibration,
                base_frame=self._base_frame,
            )
        except (KinematicsError, OSError) as exc:
            self.get_logger().fatal(f"URDF 无法读取：{exc}")
            raise

        axis_param = self.get_parameter("base_offset_axis").value
        if isinstance(axis_param, str):
            axis_value: object = axis_param.strip() or BASE_OFFSET_AXIS
        else:
            axis_value = [float(v) for v in axis_param]
        self._hole_config = PlanConfig(
            approach_height_m=float(
                self.get_parameter("approach_height_m").value
            ),
            insert_depth_m=float(self.get_parameter("insert_depth_m").value),
            base_offset_m=float(self.get_parameter("base_offset_m").value),
            base_offset_axis=axis_value,
            lift_margin_m=float(self.get_parameter("lift_margin_m").value),
            waypoint_spacing_m=float(
                self.get_parameter("waypoint_spacing_m").value
            ),
            hole_radius_m=float(self.get_parameter("hole_radius_m").value),
            radial_clearance_m=float(
                self.get_parameter("radial_clearance_m").value
            ),
        )

        # 订阅和发布用**同一个** topic。
        self._publisher = self.create_publisher(Path, self._topic, 10)
        self._subscription = self.create_subscription(
            Path, self._topic, self._on_path, 10
        )
        self._last_warning = 0.0
        self.get_logger().info(
            f"单 topic 模式：{self._topic}；输入帧 "
            f"{FRAME_JOINT_INPUT!r} / {FRAME_POSE_INPUT!r}，输出帧 {self._base_frame!r}"
        )

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------
    def _on_path(self, msg: Path) -> None:
        frame = str(msg.header.frame_id).strip()
        if frame == self._base_frame:
            return  # 自己刚发布的轨迹，忽略，避免无限自环
        if frame == FRAME_JOINT_INPUT:
            result = self._from_joint_input(msg)
        elif frame == FRAME_POSE_INPUT:
            result = self._from_pose_input(msg)
        else:
            self._warn_throttled(
                f"未知输入帧 {frame!r}，只接受 {FRAME_JOINT_INPUT!r} / "
                f"{FRAME_POSE_INPUT!r}"
            )
            return
        if result is not None:
            self._publisher.publish(result)
            self.get_logger().debug(
                f"发布 {len(result.poses)} 个路径点到 {self._topic}"
            )

    # ------------------------------------------------------------------
    # joint_input -> EE Path
    # ------------------------------------------------------------------
    def _from_joint_input(self, msg: Path) -> Optional[Path]:
        if not msg.poses:
            self._warn_throttled("joint_input 没有路径点")
            return None

        raw: List[List[float]] = []
        times: List[float] = []
        for index, pose in enumerate(msg.poses):
            p = pose.pose.position
            o = pose.pose.orientation
            values = [p.x, p.y, p.z, o.x, o.y, o.z]
            if len(values) < len(ARM_JOINT_NAMES):
                self._warn_throttled("joint_input 的关节值不足")
                return None
            raw.append([float(v) for v in values])
            stamp = pose.header.stamp
            elapsed = stamp_to_seconds(stamp)
            times.append(elapsed if elapsed > 0.0 else index * self._waypoint_dt_s)

        arm = np.asarray([row[: len(ARM_JOINT_NAMES)] for row in raw], dtype=np.float64)
        if self._smooth_method != "none" and len(raw) > 1:
            try:
                filtered = smooth_joint_positions(
                    arm,
                    times=times,
                    method=self._smooth_method,
                    window=self._smooth_window,
                    polyorder=self._smooth_polyorder,
                    sigma=self._smooth_sigma,
                )
            except ValueError as exc:
                self._warn_throttled(f"平滑失败，使用原始值：{exc}")
                filtered = arm
        else:
            filtered = arm

        result = Path()
        result.header.frame_id = self._base_frame
        result.header.stamp = self.get_clock().now().to_msg()
        for index, row in enumerate(filtered):
            degrees = self._urdf.state_to_degrees(
                {name: float(row[i]) for i, name in enumerate(ARM_JOINT_NAMES)},
                self._unit,
            )
            action = {f"{name}.pos": value for name, value in degrees.items()}
            action[f"{GRIPPER_JOINT_NAME}.pos"] = raw[index][-1]
            try:
                ee_pose = self._urdf.forward_kinematics(action)
            except Exception as exc:  # noqa: BLE001 - FK 失败不能带崩节点
                self._warn_throttled(f"正向运动学解算失败：{exc}")
                return None
            rotation = rotvec_to_matrix(
                [ee_pose["ee.wx"], ee_pose["ee.wy"], ee_pose["ee.wz"]]
            )
            x, y, z, w = matrix_to_quaternion(rotation)
            pose = PoseStamped()
            pose.header.frame_id = self._base_frame
            pose.header.stamp = stamp_plus_seconds(result.header.stamp, times[index])
            pose.pose.position.x = float(ee_pose["ee.x"])
            pose.pose.position.y = float(ee_pose["ee.y"])
            pose.pose.position.z = float(ee_pose["ee.z"])
            (
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                pose.pose.orientation.w,
            ) = (float(x), float(y), float(z), float(w))
            result.poses.append(pose)
        return result

    # ------------------------------------------------------------------
    # pose_input -> 孔位插入 Path
    # ------------------------------------------------------------------
    def _from_pose_input(self, msg: Path) -> Optional[Path]:
        if len(msg.poses) < 2:
            self._warn_throttled("pose_input 至少需要尖端和孔两个位姿")
            return None
        tip, hole = msg.poses[0], msg.poses[1]
        try:
            plan = plan_insertion(
                tip_position=_position(tip),
                tip_quaternion=_quaternion(tip),
                hole_position=_position(hole),
                hole_quaternion=_quaternion(hole),
                config=self._hole_config,
            )
        except ValueError as exc:
            self._warn_throttled(f"轨迹规划参数错误：{exc}")
            return None
        if not plan.gripper_poses:
            self._warn_throttled("轨迹规划结果为空")
            return None
        for issue in plan.issues:
            self._warn_throttled(f"轨迹几何校验：{issue}")

        result = Path()
        result.header.frame_id = self._base_frame
        result.header.stamp = self.get_clock().now().to_msg()
        for index, pose in enumerate(plan.gripper_poses):
            stamped = PoseStamped()
            stamped.header.frame_id = self._base_frame
            stamped.header.stamp = stamp_plus_seconds(
                result.header.stamp, index * self._waypoint_dt_s
            )
            stamped.pose.position.x, stamped.pose.position.y, stamped.pose.position.z = (
                pose.position
            )
            (
                stamped.pose.orientation.x,
                stamped.pose.orientation.y,
                stamped.pose.orientation.z,
                stamped.pose.orientation.w,
            ) = pose.orientation
            result.poses.append(stamped)
        return result

    # ------------------------------------------------------------------
    def _warn_throttled(self, message: str, period: float = 2.0) -> None:
        now = time.monotonic()
        if now - self._last_warning >= period:
            self._last_warning = now
            self.get_logger().warn(message)


def _position(pose: PoseStamped) -> np.ndarray:
    return np.array(
        [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z],
        dtype=np.float64,
    )


def _quaternion(pose: PoseStamped) -> np.ndarray:
    return np.array(
        [
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w,
        ],
        dtype=np.float64,
    )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrajectoryBusNode()
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

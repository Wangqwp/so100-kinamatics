#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""VLM action chunk -> 末端执行器 (EE) 位姿轨迹发布节点。

节点订阅 VLM/策略模型输出的 action chunk
（``trajectory_msgs/JointTrajectory``：``joint_names`` 给出关节名，``points``
里的每个点是一步预测并自带 ``time_from_start``），对每个点调用
:mod:`utils.kinematics_utils` 的 FK（``SO100KinematicsUtils.forward_kinematics``）
解算末端执行器（``gripper_frame_link``）的位姿，并把整条 chunk 作为一条
``nav_msgs/Path`` 发布 —— 每个预测点一个 ``geometry_msgs/PoseStamped``，
时间戳是 chunk 的基准时间加上该点的 ``time_from_start``。

单位
----
action chunk 里的关节值默认是**归一化值**（手臂关节 -100..100，夹爪 0..100），
``action_unit`` 参数决定节点怎么解读它们：

``norm``（默认）
    归一化值，用标定文件 ``calibration/orange_robot.json`` 还原成角度。
``deg``
    角度（度）—— LeRobot 后端使用的单位。
``rad``
    弧度 —— ROS 的惯例。
``auto``
    由数值本身判断（见 :meth:`PolicyTrajectoryNode._detect_unit`）；判断结果
    在一条 chunk 内保持一致。

坐标系
------
位姿发布在 ``base_frame``（默认 ``base_link``）里，坐标系名字写在消息的
``header.frame_id`` 中。``base_frame`` 留空表示使用 FK 的参考系，也就是 URDF
的根 link（对 SO-101 而言同样是 ``base_link``）。

运行环境
--------
FK 走的是 LeRobot 的运动学后端（需要 ``lerobot`` 与 ``placo``），所以节点要
跑在同时具备 ROS 2 与 LeRobot 的环境里，例如::

    source /opt/ros/humble/setup.bash
    conda activate lerobot_ros2
    python3 policy_trajectory_node.py
    # 带参数：
    python3 policy_trajectory_node.py --ros-args -p action_unit:=deg
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, Optional, Sequence

import numpy as np

import rclpy
from builtin_interfaces.msg import Duration, Time
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.parameter import Parameter
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# ``ros2 run`` 会从任意的工作目录启动节点，把仓库根目录放进 sys.path，
# 才能 import 到 utils / temp 下的模块。
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from temp.so100_kinematics import (  # noqa: E402  (上面的 sys.path 调整必须先执行)
    ARM_JOINT_NAMES,
    DEFAULT_CALIBRATION_PATH,
    JOINT_UNIT_AUTO,
    JOINT_UNIT_NORM,
    KinematicsError,
    So100Kinematics,
    load_joint_calibration,
    matrix_to_quaternion,
    normalize_joint_unit,
    rotvec_to_matrix,
)
from utils.kinematics_utils import (  # noqa: E402
    DEFAULT_TARGET_FRAME,
    DEFAULT_URDF_PATH,
    SO100KinematicsUtils,
)

#: 订阅 action chunk 的默认话题。
DEFAULT_ACTION_CHUNK_TOPIC = "/so100/policy/action_chunk"

#: 发布 EE 位姿轨迹的默认话题。
DEFAULT_EE_TRAJECTORY_TOPIC = "/so100/ee_trajectory"

#: 发布位姿所用的坐标系（SO-101 的 URDF 里 ``base_link`` 就是根 link）。
DEFAULT_BASE_FRAME = "base_link"

#: 夹爪关节：它不在运动链上（不影响末端位姿），但 LeRobot 的 FK 需要
#: ``gripper.pos`` 这个键。
GRIPPER_JOINT_NAME = "gripper"


def add_duration(stamp: Time, duration: Duration) -> Time:
    """``stamp + duration``（纳秒精度）。

    chunk 的每个预测点知道自己离 chunk 的起点有多远（``time_from_start``），
    轨迹上的时间戳就是 chunk 的基准时间加上这个偏移。
    """
    total = (
        int(stamp.sec) * 1_000_000_000
        + int(stamp.nanosec)
        + int(duration.sec) * 1_000_000_000
        + int(duration.nanosec)
    )
    result = Time()
    result.sec = int(total // 1_000_000_000)
    result.nanosec = int(total % 1_000_000_000)
    return result


class PolicyTrajectoryNode(Node):
    """把 VLM 输出的 action chunk 变成末端执行器的位姿轨迹。"""

    def __init__(
        self, parameter_overrides: Optional[Sequence[Parameter]] = None
    ) -> None:
        super().__init__("policy_trajectory_node", parameter_overrides=parameter_overrides)

        # ------------------------------------------------------------------
        # 参数
        # ------------------------------------------------------------------
        self.declare_parameter("action_chunk_topic", DEFAULT_ACTION_CHUNK_TOPIC)
        self.declare_parameter("ee_trajectory_topic", DEFAULT_EE_TRAJECTORY_TOPIC)
        self.declare_parameter("urdf_path", DEFAULT_URDF_PATH)
        self.declare_parameter("target_frame", DEFAULT_TARGET_FRAME)
        self.declare_parameter("base_frame", DEFAULT_BASE_FRAME)
        self.declare_parameter("action_unit", JOINT_UNIT_NORM)
        self.declare_parameter("calibration_path", DEFAULT_CALIBRATION_PATH)
        self.declare_parameter("gripper_joint_name", GRIPPER_JOINT_NAME)

        self._action_chunk_topic = str(self.get_parameter("action_chunk_topic").value)
        self._ee_trajectory_topic = str(self.get_parameter("ee_trajectory_topic").value)
        self._base_frame = str(self.get_parameter("base_frame").value).strip()
        self._gripper_joint_name = (
            str(self.get_parameter("gripper_joint_name").value).strip()
            or GRIPPER_JOINT_NAME
        )

        try:
            self._unit = normalize_joint_unit(str(self.get_parameter("action_unit").value))
        except KinematicsError as exc:
            self.get_logger().warn(f"{exc}，退回 {JOINT_UNIT_NORM!r}")
            self._unit = JOINT_UNIT_NORM
        #: ``auto`` 模式下上一次判断出来的单位（只用于打日志）。
        self._detected_unit: Optional[str] = None

        # 标定：把策略用的归一化值换算成角度。没有它就解读不了 ``norm``。
        calibration_path = str(self.get_parameter("calibration_path").value).strip()
        self._calibration = None
        if calibration_path:
            try:
                self._calibration = load_joint_calibration(calibration_path)
            except KinematicsError as exc:
                self.get_logger().warn(f"{exc} —— 归一化的关节值无法换算成角度")

        # ------------------------------------------------------------------
        # 运动学
        # ------------------------------------------------------------------
        urdf_path = str(self.get_parameter("urdf_path").value)
        target_frame = str(self.get_parameter("target_frame").value)

        # URDF 里的限位、根 link 等信息（用来判断单位是否设错，以及位姿的
        # 参考系）。纯 numpy 实现，不依赖 placo，所以即使 FK 起不来也可用。
        try:
            self._urdf = So100Kinematics(
                urdf_path=urdf_path,
                target_frame=target_frame,
                joint_names=ARM_JOINT_NAMES,
                calibration=self._calibration,
            )
        except KinematicsError as exc:
            self.get_logger().fatal(f"URDF 无法读取：{exc}")
            raise
        self._root_frame = self._urdf.root_frame

        self._kinematics = SO100KinematicsUtils(
            motor_names=list(ARM_JOINT_NAMES),
            urdf_path=urdf_path,
            target_frame_name=target_frame,
        )
        #: placo 的机器人模型，用来把位姿搬到 ``base_frame`` 里。
        self._robot = self._kinematics.kinematics.robot

        if self._base_frame and self._base_frame not in set(self._robot.frame_names()):
            self.get_logger().fatal(
                f"base_frame {self._base_frame!r} 不是 URDF {urdf_path!r} 里的坐标系"
            )
            raise ValueError(
                f"base_frame {self._base_frame!r} is not a frame of the URDF"
            )

        # ------------------------------------------------------------------
        # ROS 接口
        # ------------------------------------------------------------------
        self._publisher = self.create_publisher(Path, self._ee_trajectory_topic, 10)
        self._subscription = self.create_subscription(
            JointTrajectory,
            self._action_chunk_topic,
            self._on_action_chunk,
            10,
        )

        self._last_warning = 0.0

        self.get_logger().info(
            f"订阅 {self._action_chunk_topic}（关节单位：{self._unit}），"
            f"发布 {self._ee_trajectory_topic}（坐标系：{self._pose_frame}）"
        )

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------
    @property
    def _pose_frame(self) -> str:
        """发布位姿所用的坐标系。"""
        return self._base_frame or self._root_frame

    # ------------------------------------------------------------------
    # 回调
    # ------------------------------------------------------------------
    def _on_action_chunk(self, msg: JointTrajectory) -> None:
        """把一个 action chunk 解算成 EE 位姿轨迹并发布。"""
        names = [str(name) for name in msg.joint_names]
        if not names:
            self._warn_throttled("action chunk 没有给出关节名，本次不发布轨迹")
            return

        indices = self._joint_indices(names)
        missing = [joint for joint in ARM_JOINT_NAMES if joint not in indices]
        if missing:
            self._warn_throttled(
                f"action chunk 里缺少关节 {', '.join(missing)}，本次不发布轨迹"
            )
            return

        base_stamp = self._stamp(msg)
        path = Path()
        path.header.frame_id = self._pose_frame
        path.header.stamp = base_stamp

        # ``auto`` 模式下，整条 chunk 用同一个单位：第一个点的数值决定它。
        chunk_unit = self._unit

        for point in msg.points:
            positions = list(point.positions)
            if not self._has_positions(positions, indices):
                self._warn_throttled(
                    "action chunk 里的某个预测点缺少关节值，本次不发布轨迹"
                )
                return

            values = self._point_values(positions, indices)
            if chunk_unit == JOINT_UNIT_AUTO:
                chunk_unit = self._detect_unit(values)

            pose = self._point_pose(values, positions, indices, chunk_unit, base_stamp, point)
            if pose is not None:
                path.poses.append(pose)

        if not path.poses:
            self._warn_throttled("action chunk 里的关节值无法解读，本次不发布轨迹")
            return

        self._publisher.publish(path)
        print("msg.points: ", msg.points)
        print("\n")
        print("path: ", path)
        print("\n")
        self.get_logger().debug(
            f"发布了 {len(path.poses)} 个 EE 位姿到 {self._ee_trajectory_topic}"
        )

    # ------------------------------------------------------------------
    # 关节名 -> 下标
    # ------------------------------------------------------------------
    def _joint_indices(self, names: Sequence[str]) -> Dict[str, int]:
        """关节名在消息 ``joint_names`` 里的下标。

        兼容 ``left_shoulder_pan`` 这样的前缀（与 ``so101_pose_node.py`` 的
        做法一致）。
        """
        indices: Dict[str, int] = {}
        for joint in (*ARM_JOINT_NAMES, self._gripper_joint_name):
            index = self._find_joint_index(names, joint)
            if index is not None:
                indices[joint] = index
        return indices

    @staticmethod
    def _find_joint_index(names: Sequence[str], joint_name: str) -> Optional[int]:
        """在名字列表里找到 ``joint_name`` 的下标（兼容带前缀的名字）。"""
        try:
            return list(names).index(joint_name)
        except ValueError:
            for index, name in enumerate(names):
                if name.endswith(joint_name):
                    return index
        return None

    # ------------------------------------------------------------------
    # 一个预测点
    # ------------------------------------------------------------------
    def _has_positions(
        self, positions: Sequence[float], indices: Dict[str, int]
    ) -> bool:
        """这个预测点里的关节值够不够（手臂关节必须齐全）。"""
        return all(indices[joint] < len(positions) for joint in ARM_JOINT_NAMES)

    def _point_values(
        self, positions: Sequence[float], indices: Dict[str, int]
    ) -> Dict[str, float]:
        """一个预测点里手臂关节的原始值（按键名索引）。"""
        return {joint: float(positions[indices[joint]]) for joint in ARM_JOINT_NAMES}

    def _point_pose(
        self,
        values: Dict[str, float],
        positions: Sequence[float],
        indices: Dict[str, int],
        unit: str,
        base_stamp: Time,
        point: JointTrajectoryPoint,
    ) -> Optional[PoseStamped]:
        """一个预测点的关节值 -> 末端位姿（``geometry_msgs/PoseStamped``）。"""
        try:
            degrees = self._urdf.state_to_degrees(values, unit)
        except KinematicsError as exc:
            self._warn_throttled(f"action chunk 的关节值无法解读：{exc}")
            return None

        outside = self._urdf.out_of_range(
            {f"{joint}.pos": degrees[joint] for joint in ARM_JOINT_NAMES}
        )
        if outside:
            self._warn_throttled(
                f"关节 {', '.join(outside)} 的角度超出了 URDF 的限位 —— "
                "关节单位可能设错了"
            )

        action = {f"{joint}.pos": degrees[joint] for joint in ARM_JOINT_NAMES}
        # 夹爪不在运动链上，但 LeRobot 的 FK 需要这个键。
        action[f"{self._gripper_joint_name}.pos"] = self._gripper_value(positions, indices)

        try:
            ee_pose = self._kinematics.forward_kinematics(action)
        except Exception as exc:  # noqa: BLE001 —— FK 失败不能带崩节点
            self._warn_throttled(f"正向运动学解算失败：{exc}")
            return None

        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotvec_to_matrix(
            [ee_pose["ee.wx"], ee_pose["ee.wy"], ee_pose["ee.wz"]]
        )
        transform[:3, 3] = [
            ee_pose["ee.x"],
            ee_pose["ee.y"],
            ee_pose["ee.z"],
        ]

        if self._base_frame and self._base_frame != self._root_frame:
            # FK 的结果在 URDF 的根坐标系里，搬到 base_frame 里。
            world_to_base = np.asarray(
                self._robot.get_T_world_frame(self._base_frame), dtype=np.float64
            )
            transform = np.linalg.inv(world_to_base) @ transform

        x, y, z, w = matrix_to_quaternion(transform[:3, :3])

        pose = PoseStamped()
        pose.header.frame_id = self._pose_frame
        pose.header.stamp = add_duration(base_stamp, point.time_from_start)
        pose.pose.position.x = float(transform[0, 3])
        pose.pose.position.y = float(transform[1, 3])
        pose.pose.position.z = float(transform[2, 3])
        pose.pose.orientation.x = x
        pose.pose.orientation.y = y
        pose.pose.orientation.z = z
        pose.pose.orientation.w = w
        return pose

    def _gripper_value(
        self, positions: Sequence[float], indices: Dict[str, int]
    ) -> float:
        """预测点里的夹爪值（没有就当它是 0 —— 它不影响末端位姿）。"""
        index = indices.get(self._gripper_joint_name)
        if index is None or index >= len(positions):
            return 0.0
        return float(positions[index])

    # ------------------------------------------------------------------
    # 单位
    # ------------------------------------------------------------------
    def _detect_unit(self, values: Dict[str, float]) -> str:
        """猜测关节值的单位（``auto`` 模式）。"""
        unit = self._urdf.detect_state_unit(values)
        if unit != self._detected_unit:
            self._detected_unit = unit
            self.get_logger().info(f"action chunk 的关节值被解读为 {unit!r}（由数值判断得出）")
        return unit

    # ------------------------------------------------------------------
    # 时间戳
    # ------------------------------------------------------------------
    def _stamp(self, msg: JointTrajectory) -> Time:
        """沿用 chunk 的时间戳，没有就用当前时间。"""
        stamp = msg.header.stamp
        if stamp.sec or stamp.nanosec:
            return stamp
        return self.get_clock().now().to_msg()

    # ------------------------------------------------------------------
    # 日志
    # ------------------------------------------------------------------
    def _warn_throttled(self, message: str, period: float = 2.0) -> None:
        """最多每 ``period`` 秒打一次警告。"""
        now = time.monotonic()
        if now - self._last_warning >= period:
            self._last_warning = now
            self.get_logger().warn(message)


def main(args=None):
    rclpy.init(args=args)

    node = PolicyTrajectoryNode()

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

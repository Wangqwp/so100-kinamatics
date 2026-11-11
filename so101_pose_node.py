#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""SO-101 末端执行器 (EE) 位姿发布节点。

节点订阅关节位置（``sensor_msgs/JointState``），把关节值换算成角度后调用
:mod:`utils.kinematics_utils` 的 FK（``SO100KinematicsUtils.forward_kinematics``）
解算末端执行器（``gripper_frame_link``）的位姿，并以
``geometry_msgs/PoseStamped`` 发布。

单位
----
``/so100/joint_states`` 发布的是策略用的**归一化值**（手臂关节 -100..100，
夹爪 0..100），``joint_state_unit`` 参数决定节点怎么解读它们：

``norm``（默认）
    归一化值，用标定文件 ``calibration/orange_robot.json`` 还原成角度。
``deg``
    角度（度）—— LeRobot 后端使用的单位。
``rad``
    弧度 —— ROS 的惯例。
``auto``
    由数值本身判断（见 :meth:`So101PoseNode._detect_unit`）。

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
    python3 so101_pose_node.py
    # 带参数：
    python3 so101_pose_node.py --ros-args -p joint_state_unit:=deg
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState

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

#: 发布位姿的坐标系（SO-101 的 URDF 里 ``base_link`` 就是根 link）。
DEFAULT_BASE_FRAME = "base_link"

#: 夹爪关节：它不在运动链上（不影响末端位姿），但 LeRobot 的 FK 需要
#: ``gripper.pos`` 这个键。
GRIPPER_JOINT_NAME = "gripper"


class So101PoseNode(Node):
    """订阅关节位置，发布 SO-101 末端执行器的位姿。"""

    def __init__(
        self, parameter_overrides: Optional[Sequence[Parameter]] = None
    ) -> None:
        super().__init__("so101_pose_node", parameter_overrides=parameter_overrides)

        # ------------------------------------------------------------------
        # 参数
        # ------------------------------------------------------------------
        self.declare_parameter("joint_states_topic", "/so100/joint_states")
        self.declare_parameter("ee_pose_topic", "/so100/ee_pose")
        self.declare_parameter("urdf_path", DEFAULT_URDF_PATH)
        self.declare_parameter("target_frame", DEFAULT_TARGET_FRAME)
        self.declare_parameter("base_frame", DEFAULT_BASE_FRAME)
        self.declare_parameter("joint_state_unit", JOINT_UNIT_NORM)
        self.declare_parameter("calibration_path", DEFAULT_CALIBRATION_PATH)

        self._joint_states_topic = str(self.get_parameter("joint_states_topic").value)
        self._ee_pose_topic = str(self.get_parameter("ee_pose_topic").value)
        self._base_frame = str(self.get_parameter("base_frame").value).strip()

        try:
            self._unit = normalize_joint_unit(
                str(self.get_parameter("joint_state_unit").value)
            )
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
        self._publisher = self.create_publisher(PoseStamped, self._ee_pose_topic, 10)
        self._subscription = self.create_subscription(
            JointState, self._joint_states_topic, self._on_joint_states, 10
        )

        self._last_warning = 0.0

        self.get_logger().info(
            f"订阅 {self._joint_states_topic}（关节单位：{self._unit}），"
            f"发布 {self._ee_pose_topic}（坐标系：{self._pose_frame}）"
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
    def _on_joint_states(self, msg: JointState) -> None:
        """解算并发布末端位姿。"""
        action = self._joint_action(msg)
        if action is None:
            return

        try:
            ee_pose = self._kinematics.forward_kinematics(action)
        except Exception as exc:  # noqa: BLE001 —— FK 失败不能带崩节点
            self._warn_throttled(f"正向运动学解算失败：{exc}")
            return

        self._publisher.publish(self._pose_message(msg, ee_pose))

    # ------------------------------------------------------------------
    # 关节状态 -> 角度
    # ------------------------------------------------------------------
    def _joint_action(self, msg: JointState) -> Optional[Dict[str, float]]:
        """把关节状态换算成 FK 需要的动作字典（``{"<joint>.pos": 度}``）。

        关节缺失、单位无法解读时返回 ``None``（本次不发布）。
        """
        measured: Dict[str, float] = {
            str(name): float(position)
            for name, position in zip(msg.name, msg.position)
        }

        selected: Dict[str, float] = {}
        missing: List[str] = []
        for joint in ARM_JOINT_NAMES:
            value = self._joint_value(measured, joint)
            if value is None:
                missing.append(joint)
            else:
                selected[joint] = value
        if missing:
            self._warn_throttled(
                f"关节状态里缺少 {', '.join(missing)}，本次不发布位姿"
            )
            return None

        # 夹爪不在运动链上，但它的读数可以一并换算（FK 用不到）。
        gripper = self._joint_value(measured, GRIPPER_JOINT_NAME)
        if gripper is not None:
            selected[GRIPPER_JOINT_NAME] = gripper

        unit = self._unit
        if unit == JOINT_UNIT_AUTO:
            unit = self._detect_unit(selected)
            if unit != self._detected_unit:
                self._detected_unit = unit
                self.get_logger().info(f"关节状态被解读为 {unit!r}（由数值判断得出）")

        try:
            degrees = self._urdf.state_to_degrees(selected, unit)
        except KinematicsError as exc:
            self._warn_throttled(f"关节状态无法解读：{exc}")
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
        action[f"{GRIPPER_JOINT_NAME}.pos"] = degrees.get(GRIPPER_JOINT_NAME, 0.0)
        return action

    @staticmethod
    def _joint_value(measured: Dict[str, float], joint: str) -> Optional[float]:
        """从关节状态里取一个关节的值，兼容 ``left_`` / ``right_`` 这样的前缀。"""
        if joint in measured:
            return measured[joint]
        for name, value in measured.items():
            if name.endswith(joint):
                return value
        return None

    def _detect_unit(self, values: Dict[str, float]) -> str:
        """猜测关节值的单位。

        归一化值是舵机计数线性映射出来的，反算回去必须还是整数计数；角度则
        不是 —— 这是区分两者的依据。反算出来的角度还要落在 URDF 的限位之内，
        才算确实是归一化值。
        """
        return self._urdf.detect_state_unit(values)

    # ------------------------------------------------------------------
    # 位姿
    # ------------------------------------------------------------------
    def _pose_message(
        self, msg: JointState, ee_pose: Dict[str, float]
    ) -> PoseStamped:
        """FK 的输出 -> ``geometry_msgs/PoseStamped``。"""
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
        pose.header.stamp = self._stamp(msg)
        pose.header.frame_id = self._pose_frame
        pose.pose.position.x = float(transform[0, 3])
        pose.pose.position.y = float(transform[1, 3])
        pose.pose.position.z = float(transform[2, 3])
        pose.pose.orientation.x = x
        pose.pose.orientation.y = y
        pose.pose.orientation.z = z
        pose.pose.orientation.w = w
        return pose

    def _stamp(self, msg: JointState) -> Time:
        """沿用关节状态的时间戳，没有就用当前时间。"""
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

    node = So101PoseNode()

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

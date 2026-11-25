#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""VLM action chunk -> 末端执行器 (EE) 位姿轨迹发布节点。

节点订阅 VLM/策略模型输出的 action chunk
（``trajectory_msgs/JointTrajectory``：``joint_names`` 给出关节名，``points``
里的每个点是一步预测并自带 ``time_from_start``），**先把整条 chunk 的关节
位置平滑掉**，再对每个点调用 :mod:`temp.so100_kinematics` 的纯 numpy FK
（``So100Kinematics.forward_kinematics``）解算末端执行器
（``gripper_frame_link``）的位姿，并把整条 chunk 作为一条 ``nav_msgs/Path``
发布 —— 每个预测点一个 ``geometry_msgs/PoseStamped``，时间戳是 chunk 的基准
时间加上该点的 ``time_from_start``。

平滑
----
策略模型输出的关节位置往往带抖动，直接解 FK 会让末端轨迹抖得很厉害。
本节点在 FK **之前**沿时间轴把五个手臂关节的位置滤一遍（夹爪不在运动链上，
保持原值），平滑的点数和顺序都不变，所以时间戳也不变。

``smooth_method`` 参数选择滤波方法（见 :mod:`utils.trajectory_smoothing`）：

``savgol``（默认）
    Savitzky-Golay 滤波：局部最小二乘拟合低阶多项式，压抖动但保住形状。
``moving_average``
    居中的滑动平均。
``gaussian``
    高斯核滤波。
``median``
    中值滤波，对孤立尖峰最有效。
``none``
    不平滑。

``smooth_window`` 是滤波窗口的点数，``smooth_polyorder`` 是 Savitzky-Golay
的多项式阶数，``smooth_sigma`` 是高斯滤波的标准差。chunk 的
``time_from_start`` 间隔不均匀时，节点会先把关节轨迹重采样到均匀的时间
网格上，滤完再插值回原来的时间点。

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
FK 走仓库根目录的纯 numpy 运动学（``so100_kinematics.py``，经
``temp.so100_kinematics`` 兼容层接入），只依赖 ROS 2 与 numpy，不需要
``lerobot`` / ``placo``::

    source /opt/ros/humble/setup.bash
    python3 opt_policy_trajectory_node.py
    # 带参数：
    python3 opt_policy_trajectory_node.py --ros-args -p action_unit:=deg
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Optional, Sequence

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
    DEFAULT_TARGET_FRAME,
    DEFAULT_URDF_PATH,
    JOINT_UNIT_AUTO,
    JOINT_UNIT_NORM,
    KinematicsError,
    So100Kinematics,
    load_joint_calibration,
    matrix_to_quaternion,
    normalize_joint_unit,
    rotvec_to_matrix,
)
from utils.trajectory_smoothing import (  # noqa: E402
    SMOOTH_METHOD_NONE,
    SMOOTH_METHOD_SAVGOL,
    normalize_smooth_method,
    smooth_joint_positions,
)

#: 订阅 action chunk 的默认话题。
DEFAULT_ACTION_CHUNK_TOPIC = "/so100/policy/action_chunk"

#: 发布 EE 位姿轨迹的默认话题。
DEFAULT_EE_TRAJECTORY_TOPIC = "/so100/policy/opt_ee_trajectory"

#: 默认的平滑方法：Savitzky-Golay（压抖动但保住轨迹的形状）。
DEFAULT_SMOOTH_METHOD = SMOOTH_METHOD_SAVGOL

#: 默认的滤波窗口长度（点数）。
DEFAULT_SMOOTH_WINDOW = 7

#: 默认的 Savitzky-Golay 多项式阶数。
DEFAULT_SMOOTH_POLYORDER = 2

#: 默认的高斯滤波标准差（点）。
DEFAULT_SMOOTH_SIGMA = 1.0

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

        # 平滑：FK 之前先把关节位置滤一遍（夹爪不动）。
        self.declare_parameter("smooth_method", DEFAULT_SMOOTH_METHOD)
        self.declare_parameter("smooth_window", DEFAULT_SMOOTH_WINDOW)
        self.declare_parameter("smooth_polyorder", DEFAULT_SMOOTH_POLYORDER)
        self.declare_parameter("smooth_sigma", DEFAULT_SMOOTH_SIGMA)

        self._action_chunk_topic = str(self.get_parameter("action_chunk_topic").value)
        self._ee_trajectory_topic = str(self.get_parameter("ee_trajectory_topic").value)
        self._base_frame = str(self.get_parameter("base_frame").value).strip()
        self._gripper_joint_name = (
            str(self.get_parameter("gripper_joint_name").value).strip()
            or GRIPPER_JOINT_NAME
        )

        try:
            self._smooth_method = normalize_smooth_method(
                str(self.get_parameter("smooth_method").value)
            )
        except ValueError as exc:
            self.get_logger().warn(f"{exc} —— 不做平滑")
            self._smooth_method = SMOOTH_METHOD_NONE
        self._smooth_window = max(1, int(self.get_parameter("smooth_window").value))
        self._smooth_polyorder = max(
            0, int(self.get_parameter("smooth_polyorder").value)
        )
        self._smooth_sigma = float(self.get_parameter("smooth_sigma").value)

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
            except (KinematicsError, OSError) as exc:
                self.get_logger().warn(f"{exc} —— 归一化的关节值无法换算成角度")

        # ------------------------------------------------------------------
        # 运动学（纯 numpy，直接算到 base_frame）
        # ------------------------------------------------------------------
        urdf_path = str(self.get_parameter("urdf_path").value)
        target_frame = str(self.get_parameter("target_frame").value)

        try:
            self._urdf = So100Kinematics(
                urdf_path=urdf_path,
                target_frame=target_frame,
                joint_names=ARM_JOINT_NAMES,
                calibration=self._calibration,
                base_frame=self._base_frame or None,
            )
        except (KinematicsError, OSError) as exc:
            self.get_logger().fatal(f"URDF 无法读取：{exc}")
            raise
        self._root_frame = self._urdf.root_frame

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
            f"发布 {self._ee_trajectory_topic}（坐标系：{self._pose_frame}），"
            f"平滑：{self._smooth_method}"
            + (
                f"，窗口 {self._smooth_window} 点"
                if self._smooth_method != SMOOTH_METHOD_NONE
                else ""
            )
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
        """把一个 action chunk 平滑后解算成 EE 位姿轨迹并发布。"""
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

        # 先把整条 chunk 的关节位置收齐（原始单位）—— 平滑要整条一起做。
        positions: List[List[float]] = []
        for point in msg.points:
            values = list(point.positions)
            if len(values) >= 3:
                # TODO: 临时的action joint修正
                values[0] *= 0.9
                values[1] -= 5.0
                values[2] -= 5.0
            if not self._has_positions(values, indices):
                self._warn_throttled(
                    "action chunk 里的某个预测点缺少关节值，本次不发布轨迹"
                )
                return
            positions.append(values)

        if not positions:
            self._warn_throttled("action chunk 里的关节值无法解读，本次不发布轨迹")
            return

        # 平滑：五个手臂关节沿时间轴滤一遍，然后才解 FK。
        smoothed = self._smooth_positions(positions, msg.points, indices)

        # ``auto`` 模式下，整条 chunk 用同一个单位：第一个点的数值决定它。
        # 单位是按**原始**数值判断的 —— 平滑过的值不再是整数的伺服计数，
        # 用它们判断会把归一化值误认成弧度。
        chunk_unit = self._unit
        if chunk_unit == JOINT_UNIT_AUTO:
            chunk_unit = self._detect_unit(
                {
                    joint: float(positions[0][indices[joint]])
                    for joint in ARM_JOINT_NAMES
                }
            )

        base_stamp = self._stamp(msg)
        path = Path()
        path.header.frame_id = self._pose_frame
        path.header.stamp = base_stamp

        for index, point in enumerate(msg.points):
            values = {
                joint: float(smoothed[index, column])
                for column, joint in enumerate(ARM_JOINT_NAMES)
            }

            pose = self._point_pose(
                values, positions[index], indices, chunk_unit, base_stamp, point
            )
            if pose is not None:
                path.poses.append(pose)

        if not path.poses:
            self._warn_throttled("action chunk 里的关节值无法解读，本次不发布轨迹")
            return

        self._publisher.publish(path)
        self.get_logger().debug(
            f"发布了 {len(path.poses)} 个 EE 位姿到 {self._ee_trajectory_topic}"
        )

    # ------------------------------------------------------------------
    # 平滑
    # ------------------------------------------------------------------
    def _smooth_positions(
        self,
        positions: Sequence[Sequence[float]],
        points: Sequence[JointTrajectoryPoint],
        indices: Dict[str, int],
    ) -> np.ndarray:
        """整条 chunk 的手臂关节位置（原始单位）沿时间轴平滑。

        返回 ``(N, len(ARM_JOINT_NAMES))`` 的矩阵，列的顺序和
        ``ARM_JOINT_NAMES`` 一致。没有平滑时返回的也是同样的形状（调用方
        不需要区分）。

        夹爪不在运动链上（不影响末端位姿），所以只平滑手臂关节。
        """
        count = len(positions)
        raw = np.empty((count, len(ARM_JOINT_NAMES)), dtype=np.float64)
        for index, values in enumerate(positions):
            for column, joint in enumerate(ARM_JOINT_NAMES):
                raw[index, column] = float(values[indices[joint]])

        if self._smooth_method == SMOOTH_METHOD_NONE or count < 2:
            return raw

        # 时间轴：chunk 的 ``time_from_start``（秒）。间隔不均匀时，平滑
        # 模块会先把关节轨迹重采样到均匀的时间网格上，滤完再插值回来。
        times = [self._point_time(point) for point in points]

        try:
            return smooth_joint_positions(
                raw,
                times=times,
                method=self._smooth_method,
                window=self._smooth_window,
                polyorder=self._smooth_polyorder,
                sigma=self._smooth_sigma,
            )
        except ValueError as exc:
            self._warn_throttled(f"轨迹平滑失败：{exc} —— 使用未平滑的关节值")
            return raw

    @staticmethod
    def _point_time(point: JointTrajectoryPoint) -> float:
        """预测点离 chunk 起点有多远（秒）。"""
        duration = point.time_from_start
        return int(duration.sec) + int(duration.nanosec) * 1e-9

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
            # FK 已经在构造 So100Kinematics 时指定了 base_frame，
            # 不需要再借 placo 的 get_T_world_frame 做二次变换。
            ee_pose = self._urdf.forward_kinematics(action)
        except Exception as exc:  # noqa: BLE001 —— FK 失败不能带崩节点
            self._warn_throttled(f"正向运动学解算失败：{exc}")
            return None

        rotation = rotvec_to_matrix(
            [ee_pose["ee.wx"], ee_pose["ee.wy"], ee_pose["ee.wz"]]
        )
        x, y, z, w = matrix_to_quaternion(rotation)

        pose = PoseStamped()
        pose.header.frame_id = self._pose_frame
        pose.header.stamp = add_duration(base_stamp, point.time_from_start)
        pose.pose.position.x = float(ee_pose["ee.x"])
        pose.pose.position.y = float(ee_pose["ee.y"])
        pose.pose.position.z = float(ee_pose["ee.z"])
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

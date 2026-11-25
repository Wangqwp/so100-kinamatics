#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""夹爪尖端 + 孔位姿 -> 避碰插入轨迹（含可视化）ROS 2 节点。

订阅
----
``gripper_tip_topic``（默认 ``/so100/gripper_tip_pose``）
    ``geometry_msgs/PoseStamped``：上级发布的夹爪**尖端**位姿，orientation
    就是“旋度”（四元数 ``x, y, z, w``）。
``hole_topic``（默认 ``/so100/hole_pose``）
    ``geometry_msgs/PoseStamped``：孔中心与朝向，局部 +Z 为孔的朝外方向。

发布
----
``trajectory_topic``（默认 ``/so100/hole_insertion/trajectory``）
    ``nav_msgs/Path``：夹爪**控制点**轨迹（尖端目标沿世界固定方向回退
    3 cm），时间戳按 ``waypoint_dt_s`` 递增。
``markers_topic``（默认 ``/so100/hole_insertion/markers``）
    ``visualization_msgs/MarkerArray``：尖端轨迹线、孔轴箭头、孔/接近点/
    插入点/起点球与文字标注，直接在 RViz 里看。

轨迹形状
--------
``lift`` -> ``translate`` -> ``descend`` -> ``insert``：先沿孔轴抬到安全高度，
在安全高度平移到孔正上方并对齐姿态，再沿孔轴下降 6 cm 到接近点，最后竖直
插入 3 cm。平移段始终高于孔口、下降段始终贴合孔轴，所以不会撞到孔口/孔壁。

两个输入位姿会按需经过 TF 变换到 ``base_frame``（默认 ``base_link``）。若
ROS 里没有 tf2，则要求两个话题本身就在 ``base_frame`` 下。

运行::

    source /opt/ros/humble/setup.bash
    python3 hole_insertion_trajectory_node.py
    # 自定义参数：
    python3 hole_insertion_trajectory_node.py --ros-args \
        -p gripper_tip_topic:=/my/tip -p hole_topic:=/my/hole \
        -p base_offset_axis:=-z
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional, Sequence

import numpy as np

import rclpy
from builtin_interfaces.msg import Duration as RosDuration
from builtin_interfaces.msg import Time as RosTime
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time as RclpyTime
from visualization_msgs.msg import Marker, MarkerArray

# ``ros2 run`` 会从任意 cwd 启动，先把本目录放进 sys.path。
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hole_insertion_geometry import (  # noqa: E402  (sys.path 调整必须先执行)
    APPROACH_HEIGHT_M,
    BASE_OFFSET_AXIS,
    BASE_OFFSET_M,
    HOLE_RADIUS_M,
    INSERT_DEPTH_M,
    LIFT_MARGIN_M,
    PlanConfig,
    RADIAL_CLEARANCE_M,
    WAYPOINT_SPACING_M,
    matrix_to_quat,
    plan_insertion,
    quat_to_matrix,
)

# TF 是标准配置，但缺了也不该让节点起不来（两个输入可能本来就在 base_link）。
try:
    from tf2_geometry_msgs import do_transform_pose

    try:  # Humble 新版本提供 PoseStamped 版本，老版本只有 Pose 版本
        from tf2_geometry_msgs import do_transform_pose_stamped
    except ImportError:  # pragma: no cover - 取决于发行版
        do_transform_pose_stamped = None
    from tf2_ros import Buffer, TransformListener

    _TF_AVAILABLE = True
except ImportError:  # pragma: no cover - 取决于 ROS 安装
    _TF_AVAILABLE = False
    do_transform_pose = None  # type: ignore[assignment]
    do_transform_pose_stamped = None  # type: ignore[assignment]
    Buffer = None  # type: ignore[assignment]
    TransformListener = None  # type: ignore[assignment]

DEFAULT_TIP_TOPIC = "/so100/gripper_tip_pose"
DEFAULT_HOLE_TOPIC = "/so100/hole_pose"
DEFAULT_TRAJECTORY_TOPIC = "/so100/hole_insertion/trajectory"
DEFAULT_MARKERS_TOPIC = "/so100/hole_insertion/markers"
DEFAULT_BASE_FRAME = "base_link"
DEFAULT_WAYPOINT_DT_S = 0.05

#: visualization_msgs/Marker 的 ARROW 沿 marker 局部 +X 指向，而孔的 +Z
#: 才是朝外方向 —— 这个旋转把局部 +X 转到局部 +Z。
_X_TO_Z = np.array(
    [
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


def stamp_plus_seconds(stamp: RosTime, seconds: float) -> RosTime:
    """``stamp + seconds``（纳秒精度）。"""
    total = (
        int(stamp.sec) * 1_000_000_000
        + int(stamp.nanosec)
        + int(round(float(seconds) * 1_000_000_000))
    )
    result = RosTime()
    result.sec = int(total // 1_000_000_000)
    result.nanosec = int(total % 1_000_000_000)
    return result


def transform_pose_stamped(pose: PoseStamped, transform) -> PoseStamped:
    """跨发行版可用的 ``PoseStamped`` TF 变换。

    ``tf2_geometry_msgs.do_transform_pose`` 在部分 Humble 版本里只接受
    ``geometry_msgs/Pose``；直接把 ``PoseStamped`` 传进去会在运行时抛
    ``AttributeError``。这里优先用 stamped 版本，否则只变换 ``pose`` 部分。
    """
    if do_transform_pose_stamped is not None:
        result = do_transform_pose_stamped(pose, transform)
        result.header.stamp = pose.header.stamp
        return result

    converted = do_transform_pose(pose.pose, transform)
    result = PoseStamped()
    result.header.stamp = pose.header.stamp
    result.header.frame_id = str(transform.header.frame_id)
    # 老版本返回 PoseStamped，新版本返回 Pose，两者都兼容。
    result.pose = converted.pose if hasattr(converted, "pose") else converted
    return result


class HoleInsertionTrajectoryNode(Node):
    """把尖端/孔位姿变成可发布的避碰插入轨迹。"""

    def __init__(
        self, parameter_overrides: Optional[Sequence[Parameter]] = None
    ) -> None:
        super().__init__(
            "hole_insertion_trajectory", parameter_overrides=parameter_overrides
        )

        # ------------------------------------------------------------------
        # 参数
        # ------------------------------------------------------------------
        self.declare_parameter("gripper_tip_topic", DEFAULT_TIP_TOPIC)
        self.declare_parameter("hole_topic", DEFAULT_HOLE_TOPIC)
        self.declare_parameter("trajectory_topic", DEFAULT_TRAJECTORY_TOPIC)
        self.declare_parameter("markers_topic", DEFAULT_MARKERS_TOPIC)
        self.declare_parameter("base_frame", DEFAULT_BASE_FRAME)
        self.declare_parameter("plan_rate_hz", 10.0)
        self.declare_parameter("max_pose_age_s", 1.0)
        self.declare_parameter("waypoint_dt_s", DEFAULT_WAYPOINT_DT_S)
        self.declare_parameter("tf_timeout_s", 0.2)

        self.declare_parameter("approach_height_m", APPROACH_HEIGHT_M)
        self.declare_parameter("insert_depth_m", INSERT_DEPTH_M)
        self.declare_parameter("base_offset_m", BASE_OFFSET_M)
        self.declare_parameter("base_offset_axis", BASE_OFFSET_AXIS)
        self.declare_parameter("lift_margin_m", LIFT_MARGIN_M)
        self.declare_parameter("waypoint_spacing_m", WAYPOINT_SPACING_M)
        self.declare_parameter("hole_radius_m", HOLE_RADIUS_M)
        self.declare_parameter("radial_clearance_m", RADIAL_CLEARANCE_M)
        self.declare_parameter("enable_safety_lift", True)

        def text(name: str) -> str:
            return str(self.get_parameter(name).value).strip()

        self._tip_topic = text("gripper_tip_topic") or DEFAULT_TIP_TOPIC
        self._hole_topic = text("hole_topic") or DEFAULT_HOLE_TOPIC
        self._trajectory_topic = text("trajectory_topic") or DEFAULT_TRAJECTORY_TOPIC
        self._markers_topic = text("markers_topic") or DEFAULT_MARKERS_TOPIC
        self._base_frame = text("base_frame") or DEFAULT_BASE_FRAME

        plan_rate = max(0.1, float(self.get_parameter("plan_rate_hz").value))
        self._max_pose_age_s = float(self.get_parameter("max_pose_age_s").value)
        self._waypoint_dt_s = max(
            0.001, float(self.get_parameter("waypoint_dt_s").value)
        )
        self._tf_timeout_s = max(
            0.0, float(self.get_parameter("tf_timeout_s").value)
        )

        axis_param = self.get_parameter("base_offset_axis").value
        if isinstance(axis_param, str):
            axis_value: object = axis_param.strip() or BASE_OFFSET_AXIS
        else:
            axis_value = [float(v) for v in axis_param]
        self._config = PlanConfig(
            approach_height_m=float(self.get_parameter("approach_height_m").value),
            insert_depth_m=float(self.get_parameter("insert_depth_m").value),
            base_offset_m=float(self.get_parameter("base_offset_m").value),
            base_offset_axis=axis_value,
            lift_margin_m=float(self.get_parameter("lift_margin_m").value),
            waypoint_spacing_m=float(self.get_parameter("waypoint_spacing_m").value),
            hole_radius_m=float(self.get_parameter("hole_radius_m").value),
            radial_clearance_m=float(self.get_parameter("radial_clearance_m").value),
            enable_safety_lift=bool(self.get_parameter("enable_safety_lift").value),
        )

        # ------------------------------------------------------------------
        # TF
        # ------------------------------------------------------------------
        self._tf_buffer = None
        self._tf_listener = None
        if _TF_AVAILABLE:
            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)
        else:
            self.get_logger().warn(
                "tf2_ros/tf2_geometry_msgs 不可用：两个输入位姿必须已经在 "
                f"{self._base_frame} 下"
            )

        # ------------------------------------------------------------------
        # ROS 接口
        # ------------------------------------------------------------------
        self._path_publisher = self.create_publisher(
            Path, self._trajectory_topic, 10
        )
        self._marker_publisher = self.create_publisher(
            MarkerArray, self._markers_topic, 10
        )
        self._tip_subscription = self.create_subscription(
            PoseStamped, self._tip_topic, self._on_tip, 10
        )
        self._hole_subscription = self.create_subscription(
            PoseStamped, self._hole_topic, self._on_hole, 10
        )
        self._timer = self.create_timer(1.0 / plan_rate, self._on_tick)

        self._latest_tip: Optional[PoseStamped] = None
        self._latest_hole: Optional[PoseStamped] = None
        self._tip_received_at = 0.0
        self._hole_received_at = 0.0
        self._last_warning = 0.0

        self.get_logger().info(
            f"订阅 {self._tip_topic} / {self._hole_topic}，"
            f"发布 {self._trajectory_topic} / {self._markers_topic} "
            f"({self._base_frame})；接近 {self._config.approach_height_m:.3f} m，"
            f"插入 {self._config.insert_depth_m:.3f} m，"
            f"控制点回退 {self._config.base_offset_m:.3f} m "
            f"({self._config.base_offset_axis})"
        )

    # ------------------------------------------------------------------
    # 订阅回调：只缓存最新值，真正的计算放在定时器里，避免高频输入把
    # 轨迹发布刷爆。
    # ------------------------------------------------------------------
    def _on_tip(self, msg: PoseStamped) -> None:
        self._latest_tip = msg
        self._tip_received_at = time.monotonic()

    def _on_hole(self, msg: PoseStamped) -> None:
        self._latest_hole = msg
        self._hole_received_at = time.monotonic()

    # ------------------------------------------------------------------
    # 定时器回调
    # ------------------------------------------------------------------
    def _on_tick(self) -> None:
        if self._latest_tip is None or self._latest_hole is None:
            return
        now_wall = time.monotonic()
        if self._max_pose_age_s > 0.0 and (
            now_wall - self._tip_received_at > self._max_pose_age_s
            or now_wall - self._hole_received_at > self._max_pose_age_s
        ):
            self._warn_throttled("尖端或孔位姿已过期，暂停发布轨迹")
            return

        tip = self._to_base_frame(self._latest_tip, "夹爪尖端")
        hole = self._to_base_frame(self._latest_hole, "孔")
        if tip is None or hole is None:
            return

        try:
            plan = plan_insertion(
                tip_position=_position(tip),
                tip_quaternion=_quaternion(tip),
                hole_position=_position(hole),
                hole_quaternion=_quaternion(hole),
                config=self._config,
            )
        except ValueError as exc:
            self._warn_throttled(f"轨迹规划参数错误：{exc}")
            return
        if not plan.gripper_poses:
            self._warn_throttled("轨迹规划结果为空")
            return
        for issue in plan.issues:
            self._warn_throttled(f"轨迹几何校验：{issue}")

        stamp = self.get_clock().now().to_msg()
        self._path_publisher.publish(self._build_path(plan, stamp))
        self._marker_publisher.publish(self._build_markers(plan, tip, hole, stamp))
        self.get_logger().debug(
            f"发布 {len(plan.gripper_poses)} 个路径点，"
            f"安全高度 {plan.safe_height_m:.3f} m"
        )

    # ------------------------------------------------------------------
    # 坐标变换
    # ------------------------------------------------------------------
    def _to_base_frame(
        self, pose: PoseStamped, label: str
    ) -> Optional[PoseStamped]:
        """把位姿变换到 ``base_frame``；失败返回 None 并限频警告。"""
        frame = pose.header.frame_id or self._base_frame
        if not self._base_frame or frame == self._base_frame:
            return pose
        if self._tf_buffer is None:
            self._warn_throttled(
                f"{label}位姿在 {frame} 下，但没有 TF 支持，无法变换到 "
                f"{self._base_frame}"
            )
            return None
        try:
            transform = self._tf_buffer.lookup_transform(
                self._base_frame,
                frame,
                RclpyTime(),  # 最新可用变换
                timeout=Duration(seconds=self._tf_timeout_s),
            )
            return transform_pose_stamped(pose, transform)
        except Exception as exc:  # noqa: BLE001 - TF 异常不能带崩节点
            self._warn_throttled(
                f"{label}位姿从 {frame} 变换到 {self._base_frame} 失败：{exc}"
            )
            return None

    # ------------------------------------------------------------------
    # 发布内容
    # ------------------------------------------------------------------
    def _build_path(self, plan, stamp: RosTime) -> Path:
        """夹爪控制点轨迹 -> nav_msgs/Path。"""
        path = Path()
        path.header.frame_id = self._base_frame
        path.header.stamp = stamp
        for index, pose in enumerate(plan.gripper_poses):
            stamped = PoseStamped()
            stamped.header.frame_id = self._base_frame
            stamped.header.stamp = stamp_plus_seconds(
                stamp, index * self._waypoint_dt_s
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
            path.poses.append(stamped)
        return path

    def _build_markers(self, plan, tip: PoseStamped, hole: PoseStamped, stamp: RosTime):
        """尖端轨迹、孔轴和关键点 -> visualization_msgs/MarkerArray。"""
        array = MarkerArray()

        def base_marker(marker_id: int, marker_type: int, ns: str = "trajectory") -> Marker:
            marker = Marker()
            marker.header.frame_id = self._base_frame
            marker.header.stamp = stamp
            marker.ns = ns
            marker.id = marker_id
            marker.type = marker_type
            marker.action = Marker.ADD
            marker.lifetime = RosDuration(sec=1)  # 节点停发 1s 后自动消失
            return marker

        # 尖端轨迹线（与控制点轨迹差一个固定的世界偏移）
        line = base_marker(0, Marker.LINE_STRIP)
        line.scale.x = 0.006
        line.color.r, line.color.g, line.color.b, line.color.a = 0.1, 0.8, 0.9, 0.9
        for pose in plan.tip_poses:
            point = _make_point(pose.position)
            line.points.append(point)
        array.markers.append(line)

        # 孔轴箭头：ARROW 沿局部 +X，故乘一个 +X->+Z 的旋转。
        hole_position = _position(hole)
        hole_quat = _quaternion(hole)
        axis_quat = matrix_to_quat(
            quat_to_matrix(hole_quat) @ _X_TO_Z
        )
        arrow = base_marker(1, Marker.ARROW, "hole")
        arrow.pose.position = _make_point(hole_position)
        (
            arrow.pose.orientation.x,
            arrow.pose.orientation.y,
            arrow.pose.orientation.z,
            arrow.pose.orientation.w,
        ) = axis_quat
        arrow.scale.x = 0.008  # 轴粗
        arrow.scale.y = 0.016  # 箭头宽
        arrow.scale.z = 0.08  # 总长
        arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = 1.0, 0.2, 0.2, 0.95
        array.markers.append(arrow)

        # 关键点：孔中心、6 cm 接近点、插入终点、当前夹爪控制点
        spheres = (
            (2, hole_position, (0.2, 0.4, 1.0), 0.016),
            (3, plan.approach_tip, (0.1, 0.9, 0.3), 0.012),
            (4, plan.insert_tip, (1.0, 0.85, 0.1), 0.012),
            (5, np.asarray(plan.gripper_poses[0].position), (0.9, 0.2, 0.9), 0.010),
        )
        for marker_id, position, color, size in spheres:
            sphere = base_marker(marker_id, Marker.SPHERE, "waypoints")
            sphere.pose.position = _make_point(position)
            sphere.scale.x = sphere.scale.y = sphere.scale.z = size
            sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = (
                *color,
                0.9,
            )
            array.markers.append(sphere)

        # 文字标注：贴在对应点上方 1.5 cm
        labels = (
            (6, hole_position, "hole", (0.4, 0.7, 1.0)),
            (7, plan.approach_tip, "approach +6cm", (0.2, 1.0, 0.4)),
            (8, plan.insert_tip, "insert", (1.0, 0.9, 0.1)),
        )
        for marker_id, position, text, color in labels:
            label = base_marker(marker_id, Marker.TEXT_VIEW_FACING, "labels")
            label.pose.position = _make_point(position)
            label.pose.position.z += 0.015
            label.scale.z = 0.012
            label.text = text
            label.color.r, label.color.g, label.color.b, label.color.a = (*color, 1.0)
            array.markers.append(label)

        return array

    # ------------------------------------------------------------------
    def _warn_throttled(self, message: str, period: float = 2.0) -> None:
        now = time.monotonic()
        if now - self._last_warning >= period:
            self._last_warning = now
            self.get_logger().warn(message)


# ----------------------------------------------------------------------
# 小工具
# ----------------------------------------------------------------------
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


def _make_point(position) -> Point:
    point = Point()
    point.x, point.y, point.z = (float(v) for v in position)
    return point


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = HoleInsertionTrajectoryNode()
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

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""孔位插入轨迹的纯几何计算（不依赖 ROS，可单独测试）。

输入是``夹爪尖端``和``孔``两个位姿（都在同一个参考系里，通常是
``base_link``），输出是``夹爪控制点``的位姿序列 —— 也就是把尖端目标沿
**世界坐标系固定方向**回退 ``base_offset_m``（默认沿 +Z 回退 3 cm）之后
的点。这样下游拿到轨迹就能直接当夹爪法兰/控制点的目标用，尖端正好落在
预期的孔轴上。

轨迹分四段（前两段是为避碰加的安全段）：

``lift``      当前尖端沿孔轴抬升到安全高度（高于当前位置和接近点）
``translate`` 在安全高度平移到孔的正上方，同时把姿态对齐到孔的姿态
``descend``   沿孔轴竖直下降到距孔口 ``approach_height_m``（默认 6 cm）处
``insert``    沿孔轴竖直向下插入 ``insert_depth_m``（默认 3 cm）

``translate`` 全程不低于安全高度，``descend`` / ``insert`` 全程贴合孔轴，
因此不会撞到孔口边缘 —— 这就是本轨迹的“不碰撞”含义（孔被简化成一根圆柱，
不含机器人自身关节的碰撞检测）。

约定：``hole_quaternion`` 的局部 +Z 轴 = 孔的**朝外**方向（从孔底指向孔口
/机器人一侧），插入方向是它的反方向。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

#: 默认参数（与 ROS 节点的参数默认值保持一致）
APPROACH_HEIGHT_M = 0.06
INSERT_DEPTH_M = 0.03
BASE_OFFSET_M = 0.03
BASE_OFFSET_AXIS = "+z"
LIFT_MARGIN_M = 0.05
WAYPOINT_SPACING_M = 0.01
HOLE_RADIUS_M = 0.01
RADIAL_CLEARANCE_M = 0.003

Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]  # ROS 顺序 (x, y, z, w)


@dataclass(frozen=True)
class PlanConfig:
    """轨迹规划参数，单位均为米。"""

    approach_height_m: float = APPROACH_HEIGHT_M
    insert_depth_m: float = INSERT_DEPTH_M
    base_offset_m: float = BASE_OFFSET_M
    base_offset_axis: object = BASE_OFFSET_AXIS
    lift_margin_m: float = LIFT_MARGIN_M
    waypoint_spacing_m: float = WAYPOINT_SPACING_M
    hole_radius_m: float = HOLE_RADIUS_M
    radial_clearance_m: float = RADIAL_CLEARANCE_M
    enable_safety_lift: bool = True


@dataclass(frozen=True)
class PlannedPose:
    """一个路径点。``phase_t`` 是该点在本段内的归一化进度 0..1。"""

    position: Vector3
    orientation: Quaternion
    phase: str
    phase_t: float


@dataclass(frozen=True)
class TrajectoryPlan:
    """一次规划的结果。

    ``tip_poses`` 是夹爪尖端的预期轨迹，``gripper_poses`` 才是真正要下发给
    夹爪控制点的轨迹（两者相差世界坐标系里固定的 ``offset_world``）。
    ``issues`` 非空说明几何校验没通过，调用方应当警告而不是静默使用。
    """

    tip_poses: List[PlannedPose]
    gripper_poses: List[PlannedPose]
    hole_axis: Vector3
    offset_world: Vector3
    approach_tip: Vector3
    insert_tip: Vector3
    safe_height_m: float
    issues: List[str]


# ----------------------------------------------------------------------
# 四元数 / 向量工具
# ----------------------------------------------------------------------
def parse_axis(spec: object) -> np.ndarray:
    """把 ``"+z"`` / ``"-x"`` / 三元组解析成单位方向向量。"""
    if isinstance(spec, str):
        text = spec.strip().lower().replace(" ", "")
        sign = 1.0
        if text.startswith("+"):
            text = text[1:]
        elif text.startswith("-"):
            sign, text = -1.0, text[1:]
        table = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
        if text not in table:
            raise ValueError(
                f"方向 {spec!r} 无法识别，可用 '+x'/'-x'/'+y'/'-y'/'+z'/'-z' 或三个数字"
            )
        return sign * np.asarray(table[text], dtype=np.float64)

    values = np.asarray(spec, dtype=np.float64).reshape(-1)
    if values.size != 3:
        raise ValueError(f"方向必须有 3 个分量，得到 {values.size} 个")
    norm = float(np.linalg.norm(values))
    if norm < 1e-12:
        raise ValueError("方向向量不能为零")
    return values / norm


def quat_normalize(quaternion: Sequence[float]) -> np.ndarray:
    """归一化四元数，返回 ROS 顺序 ``(x, y, z, w)``，且令 ``w >= 0``。"""
    values = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(values))
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    values = values / norm
    if values[3] < 0.0:
        values = -values
    return values


def quat_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """四元数 ``(x, y, z, w)`` -> 3x3 旋转矩阵。"""
    x, y, z, w = quat_normalize(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    """3x3 旋转矩阵 -> 四元数 ``(x, y, z, w)``（数值稳定的分支写法）。"""
    rot = np.asarray(matrix, dtype=np.float64)[:3, :3]
    trace = float(rot[0, 0] + rot[1, 1] + rot[2, 2])
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rot[2, 1] - rot[1, 2]) / scale
        y = (rot[0, 2] - rot[2, 0]) / scale
        z = (rot[1, 0] - rot[0, 1]) / scale
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        scale = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        w = (rot[2, 1] - rot[1, 2]) / scale
        x = 0.25 * scale
        y = (rot[0, 1] + rot[1, 0]) / scale
        z = (rot[0, 2] + rot[2, 0]) / scale
    elif rot[1, 1] > rot[2, 2]:
        scale = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        w = (rot[0, 2] - rot[2, 0]) / scale
        x = (rot[0, 1] + rot[1, 0]) / scale
        y = 0.25 * scale
        z = (rot[1, 2] + rot[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        w = (rot[1, 0] - rot[0, 1]) / scale
        x = (rot[0, 2] + rot[2, 0]) / scale
        y = (rot[1, 2] + rot[2, 1]) / scale
        z = 0.25 * scale
    return quat_normalize((x, y, z, w))


def quat_slerp(q0: Sequence[float], q1: Sequence[float], t: float) -> np.ndarray:
    """球面线性插值，取最短弧；``t`` 会夹到 [0, 1]。"""
    start = quat_normalize(q0)
    end = quat_normalize(q1)
    t = min(1.0, max(0.0, float(t)))
    dot = float(np.dot(start, end))
    if dot < 0.0:  # 四元数 q 与 -q 表示同一个姿态，走短弧
        end = -end
        dot = -dot
    if dot > 0.9995:  # 角度太小，线性插值足够且避免除以接近 0 的正弦
        return quat_normalize(start + t * (end - start))
    theta = math.acos(min(1.0, dot))
    sin_theta = math.sin(theta)
    s0 = math.sin((1.0 - t) * theta) / sin_theta
    s1 = math.sin(t * theta) / sin_theta
    return quat_normalize(s0 * start + s1 * end)


def hole_axis(quaternion: Sequence[float]) -> np.ndarray:
    """孔的朝外方向 = 孔姿态的局部 +Z 轴。"""
    axis = quat_to_matrix(quaternion)[:, 2]
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:  # pragma: no cover - 归一化后不会发生
        return np.array([0.0, 0.0, 1.0])
    return axis / norm


def _height(point: np.ndarray, hole_position: np.ndarray, axis: np.ndarray) -> float:
    """点相对孔中心、沿孔轴的“高度”（孔口方向为正）。"""
    return float(np.dot(point - hole_position, axis))


def _radial_distance(
    point: np.ndarray, hole_position: np.ndarray, axis: np.ndarray
) -> float:
    """点到孔轴的横向距离。"""
    delta = point - hole_position
    lateral = delta - float(np.dot(delta, axis)) * axis
    return float(np.linalg.norm(lateral))


# ----------------------------------------------------------------------
# 主规划函数
# ----------------------------------------------------------------------
def plan_insertion(
    tip_position: Sequence[float],
    tip_quaternion: Sequence[float],
    hole_position: Sequence[float],
    hole_quaternion: Sequence[float],
    config: PlanConfig | None = None,
) -> TrajectoryPlan:
    """由当前尖端位姿和孔位姿生成一条避碰插入轨迹。

    返回的 ``gripper_poses`` 已包含世界固定方向的 3 cm 回退，可直接发布给
    夹爪控制器；``tip_poses`` 是对应的尖端轨迹，供可视化对比。
    """
    cfg = config or PlanConfig()
    p_tip = np.asarray(tip_position, dtype=np.float64).reshape(3)
    q_tip = quat_normalize(tip_quaternion)
    p_hole = np.asarray(hole_position, dtype=np.float64).reshape(3)
    q_hole = quat_normalize(hole_quaternion)

    axis = hole_axis(q_hole)
    offset = parse_axis(cfg.base_offset_axis) * float(cfg.base_offset_m)

    # 关键位置（都是“尖端”的位置）
    h_current = _height(p_tip, p_hole, axis)
    h_approach = float(cfg.approach_height_m)
    h_safe = max(h_current, h_approach) + float(cfg.lift_margin_m)
    p_lift = p_tip + (h_safe - h_current) * axis
    p_over = p_hole + h_safe * axis
    p_approach = p_hole + h_approach * axis
    p_insert = p_hole - float(cfg.insert_depth_m) * axis

    # (段名, 起点, 终点, 起点姿态, 终点姿态)
    segments = []
    if cfg.enable_safety_lift:
        segments.append(("lift", p_tip, p_lift, q_tip, q_tip))
        segments.append(("translate", p_lift, p_over, q_tip, q_hole))
    else:
        segments.append(("align", p_tip, p_over, q_tip, q_hole))
    segments.append(("descend", p_over, p_approach, q_hole, q_hole))
    segments.append(("insert", p_approach, p_insert, q_hole, q_hole))

    spacing = max(1e-4, float(cfg.waypoint_spacing_m))
    tip_poses: List[PlannedPose] = []
    for name, start, end, q_start, q_end in segments:
        length = float(np.linalg.norm(end - start))
        count = max(1, int(math.ceil(length / spacing)))
        if length < 1e-9 and float(np.dot(q_start, q_end)) > 0.9999:
            continue  # 完全空的段，不产生重复点
        for index in range(count + 1):
            if tip_poses and index == 0:
                continue  # 上一段的终点就是这个点
            t = index / count
            position = start + t * (end - start)
            orientation = quat_slerp(q_start, q_end, t)
            tip_poses.append(
                PlannedPose(
                    position=tuple(float(v) for v in position),
                    orientation=tuple(float(v) for v in orientation),
                    phase=name,
                    phase_t=t,
                )
            )

    gripper_poses = [
        PlannedPose(
            position=tuple(float(v) for v in np.asarray(pose.position) + offset),
            orientation=pose.orientation,
            phase=pose.phase,
            phase_t=pose.phase_t,
        )
        for pose in tip_poses
    ]

    issues = _validate(
        cfg,
        tip_poses,
        gripper_poses,
        p_tip,
        p_hole,
        p_approach,
        p_insert,
        axis,
        offset,
        h_safe,
    )
    return TrajectoryPlan(
        tip_poses=tip_poses,
        gripper_poses=gripper_poses,
        hole_axis=tuple(float(v) for v in axis),
        offset_world=tuple(float(v) for v in offset),
        approach_tip=tuple(float(v) for v in p_approach),
        insert_tip=tuple(float(v) for v in p_insert),
        safe_height_m=h_safe,
        issues=issues,
    )


def _validate(
    cfg: PlanConfig,
    tip_poses: List[PlannedPose],
    gripper_poses: List[PlannedPose],
    p_tip: np.ndarray,
    p_hole: np.ndarray,
    p_approach: np.ndarray,
    p_insert: np.ndarray,
    axis: np.ndarray,
    offset: np.ndarray,
    h_safe: float,
) -> List[str]:
    """对轨迹做几项便宜的几何自检，替代完整的碰撞检测。"""
    issues: List[str] = []
    if not tip_poses or not gripper_poses:
        return ["轨迹为空"]

    if cfg.approach_height_m <= 0.0:
        issues.append("approach_height_m 必须为正")
    if cfg.insert_depth_m <= 0.0:
        issues.append("insert_depth_m 必须为正")
    if cfg.lift_margin_m < 0.0:
        issues.append("lift_margin_m 不能为负")
    if cfg.radial_clearance_m >= cfg.hole_radius_m:
        issues.append("radial_clearance_m 不应大于 hole_radius_m")

    # 起点连续性：控制点应当等于当前尖端 + 世界回退量
    first = np.asarray(gripper_poses[0].position)
    if float(np.linalg.norm(first - (p_tip + offset))) > 1e-6:
        issues.append("轨迹起点与当前夹爪位置不连续")

    # 接近点必须精确落在孔轴上方 6 cm（换算成控制点后同样成立）
    approach_pose = next(
        (
            pose
            for pose in tip_poses
            if pose.phase == "descend" and abs(pose.phase_t - 1.0) < 1e-9
        ),
        None,
    )
    if approach_pose is None or float(
        np.linalg.norm(np.asarray(approach_pose.position) - p_approach)
    ) > 1e-6:
        issues.append("轨迹未经过预先计算的 6 cm 接近点")
    if float(np.linalg.norm(p_insert - p_hole) - cfg.insert_depth_m) > 1e-6:
        issues.append("插入终点深度与 insert_depth_m 不一致")

    # 安全段不低于安全高度，插入段贴合孔轴且单调下降
    allowed_radial = cfg.hole_radius_m - cfg.radial_clearance_m
    previous_height = None
    for pose in tip_poses:
        point = np.asarray(pose.position)
        height = _height(point, p_hole, axis)
        if pose.phase == "translate" and height < h_safe - 1e-6:
            issues.append("平移段低于安全高度，可能撞到孔口")
        if pose.phase in ("descend", "insert"):
            if _radial_distance(point, p_hole, axis) > allowed_radial + 1e-6:
                issues.append("下降/插入段偏离孔轴，可能撞到孔壁")
            if previous_height is not None and height > previous_height + 1e-9:
                issues.append("下降/插入段高度非单调")
            previous_height = height

    # 去重，保持顺序
    return list(dict.fromkeys(issues))

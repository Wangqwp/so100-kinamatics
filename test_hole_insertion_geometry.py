#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""hole_insertion_geometry.py 的离线自测（不需要 ROS）。

运行::

    python test_hole_insertion_geometry.py
"""

from __future__ import annotations

import math

import numpy as np

from hole_insertion_geometry import (
    PlanConfig,
    hole_axis,
    matrix_to_quat,
    parse_axis,
    plan_insertion,
    quat_slerp,
    quat_to_matrix,
)

TIP_POSITION = (0.2, 0.1, 0.2)
TIP_QUATERNION = (0.0, 0.0, 0.0, 1.0)
HOLE_POSITION = (0.3, 0.0, 0.1)
HOLE_QUATERNION = (0.0, 0.0, 0.0, 1.0)


def _close(actual, expected, tolerance=1e-6):
    return bool(np.allclose(np.asarray(actual, dtype=float), expected, atol=tolerance))


def test_approach_insert_and_world_offset():
    """6 cm 接近点、3 cm 插入深度、3 cm 世界 +Z 回退同时成立。"""
    plan = plan_insertion(TIP_POSITION, TIP_QUATERNION, HOLE_POSITION, HOLE_QUATERNION)
    assert plan.issues == [], plan.issues
    assert _close(plan.approach_tip, (0.3, 0.0, 0.16)), plan.approach_tip
    assert _close(plan.insert_tip, (0.3, 0.0, 0.07)), plan.insert_tip
    assert _close(plan.offset_world, (0.0, 0.0, 0.03)), plan.offset_world
    # 轨迹起点 = 当前尖端 + 世界回退量
    assert _close(plan.gripper_poses[0].position, (0.2, 0.1, 0.23))
    # 终点控制点 = 插入尖端 + 世界回退量
    assert _close(plan.gripper_poses[-1].position, (0.3, 0.0, 0.10))


def test_four_phases_and_collision_free():
    """四段轨迹齐全，且平移段高于孔口、插入段贴合孔轴并单调下降。"""
    plan = plan_insertion(TIP_POSITION, TIP_QUATERNION, HOLE_POSITION, HOLE_QUATERNION)
    phases = [pose.phase for pose in plan.tip_poses]
    assert phases[0] == "lift" and phases[-1] == "insert"
    assert "translate" in phases and "descend" in phases
    assert plan.issues == [], plan.issues

    hole = np.asarray(HOLE_POSITION)
    axis = np.asarray(plan.hole_axis)
    for pose in plan.tip_poses:
        point = np.asarray(pose.position)
        delta = point - hole
        height = float(np.dot(delta, axis))
        radial = float(np.linalg.norm(delta - height * axis))
        if pose.phase == "translate":
            assert height >= plan.safe_height_m - 1e-6
        if pose.phase in ("descend", "insert"):
            assert radial <= 1e-6  # 完全贴合孔轴

    heights = [
        float(np.dot(np.asarray(p.position) - hole, axis))
        for p in plan.tip_poses
        if p.phase in ("descend", "insert")
    ]
    assert all(later <= earlier + 1e-9 for earlier, later in zip(heights, heights[1:]))


def test_tilted_hole_axis():
    """孔绕 Y 轴转 90° 后局部 +Z 指向世界 +X，接近/插入沿该轴。"""
    # (0, sin45°, 0, cos45°)：绕 Y 轴 +90°
    q = (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5))
    plan = plan_insertion((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), (0.5, 0.2, 0.1), q)
    assert plan.issues == [], plan.issues
    assert _close(plan.hole_axis, (1.0, 0.0, 0.0)), plan.hole_axis
    assert _close(plan.approach_tip, (0.56, 0.2, 0.1)), plan.approach_tip
    assert _close(plan.insert_tip, (0.47, 0.2, 0.1)), plan.insert_tip
    # 世界回退方向固定 +Z，不随孔姿态旋转
    assert _close(plan.offset_world, (0.0, 0.0, 0.03))


def test_offset_axis_parsing():
    assert _close(parse_axis("+z"), (0, 0, 1))
    assert _close(parse_axis("-z"), (0, 0, -1))
    assert _close(parse_axis(" +X "), (1, 0, 0))
    assert _close(parse_axis((0.0, 2.0, 0.0)), (0, 1, 0))
    for bad in ("w", (0.0, 0.0, 0.0)):
        try:
            parse_axis(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad!r} 应当被拒绝")

    plan = plan_insertion(
        TIP_POSITION,
        TIP_QUATERNION,
        HOLE_POSITION,
        HOLE_QUATERNION,
        PlanConfig(base_offset_axis="-z", base_offset_m=0.03),
    )
    assert _close(plan.offset_world, (0.0, 0.0, -0.03))
    assert _close(plan.gripper_poses[0].position, (0.2, 0.1, 0.17))


def test_quaternion_helpers():
    for q in (
        (0.0, 0.0, 0.0, 1.0),
        (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)),
        (0.5, 0.5, 0.5, 0.5),
    ):
        assert _close(matrix_to_quat(quat_to_matrix(q)), q)
        assert _close(hole_axis(q), quat_to_matrix(q)[:, 2])

    identity = np.array([0.0, 0.0, 0.0, 1.0])
    target = np.array([0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)])
    assert _close(quat_slerp(identity, target, 0.0), identity)
    assert _close(quat_slerp(identity, target, 1.0), target)
    midway = quat_slerp(identity, target, 0.5)
    assert abs(float(np.linalg.norm(midway)) - 1.0) < 1e-9
    assert abs(midway[1] - math.sin(math.pi / 8)) < 1e-9
    # q 与 -q 表示同一姿态，slerp 应走短弧而不是绕远路
    assert _close(quat_slerp(identity, -target, 1.0), target)


def test_config_validation_and_no_lift_mode():
    bad = plan_insertion(
        TIP_POSITION,
        TIP_QUATERNION,
        HOLE_POSITION,
        HOLE_QUATERNION,
        PlanConfig(radial_clearance_m=0.02, hole_radius_m=0.01),
    )
    assert any("radial_clearance_m" in issue for issue in bad.issues), bad.issues

    direct = plan_insertion(
        TIP_POSITION,
        TIP_QUATERNION,
        HOLE_POSITION,
        HOLE_QUATERNION,
        PlanConfig(enable_safety_lift=False),
    )
    assert direct.issues == [], direct.issues
    assert [p.phase for p in direct.tip_poses][0] == "align"
    assert "translate" not in [p.phase for p in direct.tip_poses]


def main() -> None:
    tests = [
        test_approach_insert_and_world_offset,
        test_four_phases_and_collision_free,
        test_tilted_hole_axis,
        test_offset_axis_parsing,
        test_quaternion_helpers,
        test_config_validation_and_no_lift_mode,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()

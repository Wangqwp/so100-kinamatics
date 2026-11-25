#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""纯 numpy 的运动学兼容层，供 policy 轨迹节点使用。

背景
----
``policy_trajectory_node.py`` / ``opt_policy_trajectory_node.py`` 原来 import
``temp.so100_kinematics`` 和 ``utils.kinematics_utils``，但前者从未随仓库提交，
后者依赖 ``lerobot`` + ``placo``，在只装了 ROS 2 的环境里起不来。

本模块把两者合并成一个只依赖 ``numpy`` + 标准库的入口：

* 直接复用仓库根目录的 ``so100_kinematics.py``（纯 numpy FK/Jacobian）；
* 补上节点用到的 ``norm`` 单位、标定文件读取和 ``state_to_degrees``；
* FK 直接算到 ``base_frame``，不再需要 placo 的 ``get_T_world_frame``。

``norm`` 约定
-------------
LeRobot SO-100/101 的 action chunk 里，手臂关节是 **度**（范围约 -100..100），
夹爪是 0..100。所以 ``norm -> deg`` 默认是恒等映射；只有在数值明显是原始舵机
tick（绝对值 > 200）时，才用标定 JSON 把它解码成度。这样既兼容策略输出，
也兼容直接喂标定值的调试场景。
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

# ``ros2 run`` 从任意 cwd 启动，保证能 import 到仓库根目录的 so100_kinematics。
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from so100_kinematics import (  # noqa: E402
    ARM_JOINT_NAMES,
    DEFAULT_BASE_FRAME,
    DEFAULT_TARGET_FRAME,
    JOINT_UNIT_AUTO,
    JOINT_UNIT_DEG,
    JOINT_UNIT_RAD,
    KinematicsError,
    detect_joint_unit,
    matrix_to_quaternion,
    rotvec_to_matrix,
)
from so100_kinematics import (  # noqa: E402
    So100Kinematics as _So100Kinematics,
)

#: 策略归一化关节值（LeRobot 约定下就是度）。
JOINT_UNIT_NORM = "norm"

def _package_file(relative_path: str) -> str:
    """先找源码树，再找安装后的 share 目录。

    ``ros2 run`` 时本模块在 site-packages，数据文件却在
    ``share/so100_trajectory``；只拼 ``__file__`` 会拿到不存在的路径。
    """
    local = os.path.join(_REPO_ROOT, relative_path)
    if os.path.isfile(local):
        return local
    try:
        from ament_index_python.packages import get_package_share_directory

        installed = os.path.join(
            get_package_share_directory("so100_trajectory"), relative_path
        )
        if os.path.isfile(installed):
            return installed
    except Exception:  # noqa: BLE001 - 源码树/非 ROS 环境下静默退回
        pass
    return local


#: 默认标定文件（只有读到原始 tick 时才真正用到）。
DEFAULT_CALIBRATION_PATH = _package_file("calibration/orange_robot.json")

#: 默认 URDF：源码树里是演示 URDF；安装后从 share 目录取。
DEFAULT_URDF_PATH = _package_file("config/so101_demo.urdf")

_JOINT_UNIT_ALIASES = {
    "norm": JOINT_UNIT_NORM,
    "normalized": JOINT_UNIT_NORM,
    "deg": JOINT_UNIT_DEG,
    "degree": JOINT_UNIT_DEG,
    "degrees": JOINT_UNIT_DEG,
    "rad": JOINT_UNIT_RAD,
    "radian": JOINT_UNIT_RAD,
    "radians": JOINT_UNIT_RAD,
    "auto": JOINT_UNIT_AUTO,
    "": JOINT_UNIT_AUTO,
}

#: tick 数值超过这个阈值才认为是原始舵机值，而不是 LeRobot 的 -100..100。
_TICK_MAGNITUDE_THRESHOLD = 200.0


def normalize_joint_unit(unit: str) -> str:
    """与根目录同名函数一致，但额外接受 ``norm``。"""
    text = str(unit).strip().lower()
    try:
        return _JOINT_UNIT_ALIASES[text]
    except KeyError:
        raise KinematicsError(
            f"unknown joint unit {unit!r} (expected 'norm', 'deg', 'rad' or 'auto')"
        ) from None


def load_joint_calibration(path: str = DEFAULT_CALIBRATION_PATH) -> Dict[str, dict]:
    """读取 LeRobot 风格的标定 JSON，算出每个关节的度范围。

    返回字典的键是关节名，值是 ``range_min`` / ``range_max`` /
    ``homing_offset`` / ``drive_mode`` / ``deg_min`` / ``deg_max``。
    """
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    calibration: Dict[str, dict] = {}
    for name, item in raw.items():
        if not isinstance(item, Mapping):
            continue
        low = float(item.get("range_min", 0.0))
        high = float(item.get("range_max", 4095.0))
        span = high - low
        if span <= 0.0:
            continue
        offset = float(item.get("homing_offset", 0.0))
        drive_mode = int(item.get("drive_mode", 0))
        deg_low = (low - offset) * 360.0 / span
        deg_high = (high - offset) * 360.0 / span
        if drive_mode:  # 电机反向安装
            deg_low, deg_high = -deg_high, -deg_low
        calibration[str(name)] = {
            "range_min": low,
            "range_max": high,
            "homing_offset": offset,
            "drive_mode": drive_mode,
            "deg_min": min(deg_low, deg_high),
            "deg_max": max(deg_low, deg_high),
        }
    return calibration


class So100Kinematics(_So100Kinematics):
    """带单位换算的 FK；接口兼容原来的 ``temp.so100_kinematics``。"""

    def __init__(
        self,
        urdf_path: str = DEFAULT_URDF_PATH,
        target_frame: str = DEFAULT_TARGET_FRAME,
        joint_names: Sequence[str] = ARM_JOINT_NAMES,
        calibration: Optional[Mapping[str, Mapping[str, float]]] = None,
        base_frame: Optional[str] = None,
    ) -> None:
        self.calibration: Dict[str, dict] = {
            str(name): dict(values) for name, values in (calibration or {}).items()
        }
        super().__init__(
            urdf_path=urdf_path,
            target_frame=target_frame,
            joint_names=joint_names,
            base_frame=base_frame,
        )

    # ------------------------------------------------------------------
    # 单位
    # ------------------------------------------------------------------
    def state_to_degrees(
        self, values: Mapping[str, float], unit: str
    ) -> Dict[str, float]:
        """把 action chunk 的一个点换算成 FK 需要的度。

        ``norm``：LeRobot 的 -100..100（手臂）本来就是度，直接透传；如果
        数值看起来是原始 tick（|v| > 200）则用标定文件解码。
        ``rad``：弧度转度。``deg``：原样。``auto``：按数值大小判断。
        """
        resolved = normalize_joint_unit(unit)
        if resolved == JOINT_UNIT_AUTO:
            resolved = self.detect_state_unit(values)

        degrees: Dict[str, float] = {}
        for name in self.joint_names:
            value = values.get(f"{name}.pos", values.get(name))
            if value is None:
                raise KinematicsError(f"the joint {name!r} is missing from the command")
            number = float(value)
            if resolved == JOINT_UNIT_RAD:
                degrees[name] = math.degrees(number)
                continue
            if resolved == JOINT_UNIT_NORM:
                cal = self.calibration.get(name)
                if cal and abs(number) > _TICK_MAGNITUDE_THRESHOLD:
                    low = float(cal["range_min"])
                    high = float(cal["range_max"])
                    offset = float(cal["homing_offset"])
                    ticks = number
                    if int(cal.get("drive_mode", 0)):
                        ticks = low + high - ticks
                    degrees[name] = (ticks - offset) * 360.0 / (high - low)
                else:
                    degrees[name] = number
                continue
            degrees[name] = number
        return degrees

    def detect_state_unit(self, values: Mapping[str, float]) -> str:
        """按关节数值大小判断单位（整条 chunk 用同一个判断结果）。"""
        sampled = []
        for name in self.joint_names:
            value = values.get(f"{name}.pos", values.get(name))
            if value is not None:
                sampled.append(float(value))
        return detect_joint_unit(sampled)


__all__ = [
    "ARM_JOINT_NAMES",
    "DEFAULT_BASE_FRAME",
    "DEFAULT_CALIBRATION_PATH",
    "DEFAULT_TARGET_FRAME",
    "DEFAULT_URDF_PATH",
    "JOINT_UNIT_AUTO",
    "JOINT_UNIT_DEG",
    "JOINT_UNIT_NORM",
    "JOINT_UNIT_RAD",
    "KinematicsError",
    "So100Kinematics",
    "detect_joint_unit",
    "load_joint_calibration",
    "matrix_to_quaternion",
    "normalize_joint_unit",
    "rotvec_to_matrix",
]

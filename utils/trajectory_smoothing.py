# SPDX-License-Identifier: Apache-2.0
"""策略轨迹的关节位置平滑。

VLM/策略模型输出的 action chunk（``trajectory_msgs/JointTrajectory``）里的
关节位置往往带抖动，直接拿去做正向运动学会让末端轨迹抖得很厉害。本模块
提供几种沿**时间轴**的平滑方法，把 ``(N, J)`` 的关节位置矩阵（``N`` 个预测
点、``J`` 个关节）滤成同样形状的矩阵 —— 点数、点的顺序都不变，所以调用方
只需要把平滑后的值替换回原来的位置。

方法
----
``none``
    不平滑（原样返回）。
``savgol``（默认）
    Savitzky-Golay 滤波：局部最小二乘拟合一个低阶多项式。它能压掉抖动，
    同时保住轨迹的形状（峰值不会被削平），是机器人轨迹最常用的一种。
``moving_average``
    居中的滑动平均：最简单，但会把轨迹抹平、峰值变低。
``gaussian``
    高斯核滤波：比滑动平均更平滑，边缘不会因为补零而塌下去。
``median``
    中值滤波：对孤立尖峰（单点跳变）最有效。

只有 numpy 依赖（和 :mod:`temp.so100_kinematics` 一样），所以可以在没有
ROS 2 的机器上直接测试。

时间轴
------
滤波方法默认采样间隔是均匀的。action chunk 的 ``time_from_start`` 一般是
均匀的，但也不一定；:func:`smooth_joint_positions` 在间隔不均匀时先把关节
轨迹重采样到均匀的时间网格上，滤完再插值回原来的时间点 —— 这样滤波器的
窗口才对应真实的时间跨度。
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np


#: 不平滑。
SMOOTH_METHOD_NONE = "none"

#: Savitzky-Golay 滤波（默认）。
SMOOTH_METHOD_SAVGOL = "savgol"

#: 居中的滑动平均。
SMOOTH_METHOD_MOVING_AVERAGE = "moving_average"

#: 高斯核滤波。
SMOOTH_METHOD_GAUSSIAN = "gaussian"

#: 中值滤波。
SMOOTH_METHOD_MEDIAN = "median"

#: ``normalize_smooth_method`` 认识的写法。
_SMOOTH_METHOD_ALIASES = {
    "": SMOOTH_METHOD_NONE,
    "none": SMOOTH_METHOD_NONE,
    "off": SMOOTH_METHOD_NONE,
    "raw": SMOOTH_METHOD_NONE,
    "savgol": SMOOTH_METHOD_SAVGOL,
    "savitzky_golay": SMOOTH_METHOD_SAVGOL,
    "savitzky-golay": SMOOTH_METHOD_SAVGOL,
    "sg": SMOOTH_METHOD_SAVGOL,
    "moving_average": SMOOTH_METHOD_MOVING_AVERAGE,
    "moving-average": SMOOTH_METHOD_MOVING_AVERAGE,
    "mean": SMOOTH_METHOD_MOVING_AVERAGE,
    "gaussian": SMOOTH_METHOD_GAUSSIAN,
    "gauss": SMOOTH_METHOD_GAUSSIAN,
    "median": SMOOTH_METHOD_MEDIAN,
}


def normalize_smooth_method(method: str) -> str:
    """平滑方法的规范名字。

    ``'none'``、``'savgol'``、``'moving_average'``、``'gaussian'`` 或
    ``'median'``。

    Raises:
        ValueError: 方法不认识。
    """
    text = str(method).strip().lower().replace(" ", "_")
    try:
        return _SMOOTH_METHOD_ALIASES[text]
    except KeyError:
        known = ", ".join(sorted(set(_SMOOTH_METHOD_ALIASES.values())))
        raise ValueError(
            f"unknown smoothing method {method!r} (expected one of {known})"
        ) from None


def smooth_joint_positions(
    values: np.ndarray,
    times: Optional[Sequence[float]] = None,
    method: str = SMOOTH_METHOD_SAVGOL,
    window: int = 7,
    polyorder: int = 2,
    sigma: float = 1.0,
) -> np.ndarray:
    """沿时间轴平滑关节位置。

    Args:
        values: ``(N, J)`` 的关节位置（每个关节一列）。
        times: ``(N,)`` 的时间戳（秒）。给不出、或者间隔不均匀/不是递增时，
            退回到按序号均匀采样。
        method: 平滑方法（见 :func:`normalize_smooth_method`）。
        window: 滤波窗口的长度（点数）。实际使用的窗口不会超过点数。
        polyorder: Savitzky-Golay 的多项式阶数。
        sigma: 高斯滤波的标准差（单位：点）。

    Returns:
        同样是 ``(N, J)`` 的矩阵：点数和关节数都不变。

    Raises:
        ValueError: 形状不对，或者方法不认识。
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(
            f"the joint positions must be a 2d array (points x joints), "
            f"got {array.ndim} dimensions"
        )

    count = array.shape[0]
    if count == 0:
        return array.copy()

    canonical = normalize_smooth_method(method)
    if canonical == SMOOTH_METHOD_NONE or count == 1:
        return array.copy()

    grid = _uniform_grid(times, count)
    if grid is None:
        dense = array
    else:
        # 间隔不均匀：先重采样到均匀的时间网格，滤完再插值回去
        dense = _resample(array, np.asarray(times, dtype=np.float64), grid)

    smoothed = _apply_filter(
        dense,
        canonical,
        window=int(window),
        polyorder=int(polyorder),
        sigma=float(sigma),
    )

    if grid is not None:
        smoothed = _resample(smoothed, grid, np.asarray(times, dtype=np.float64))

    return smoothed


# ---------------------------------------------------------------------------
# The helpers
# ---------------------------------------------------------------------------


def _apply_filter(
    values: np.ndarray,
    method: str,
    window: int,
    polyorder: int,
    sigma: float,
) -> np.ndarray:
    """把滤波方法作用到每一列上。"""
    smoothed = np.empty_like(values)
    for column in range(values.shape[1]):
        smoothed[:, column] = _apply_filter_1d(
            values[:, column], method, window, polyorder, sigma
        )
    return smoothed


def _apply_filter_1d(
    values: np.ndarray,
    method: str,
    window: int,
    polyorder: int,
    sigma: float,
) -> np.ndarray:
    """一列关节位置上的滤波。"""
    if method == SMOOTH_METHOD_SAVGOL:
        return _savgol_filter(values, window, polyorder)
    if method == SMOOTH_METHOD_MOVING_AVERAGE:
        return _moving_average(values, window)
    if method == SMOOTH_METHOD_GAUSSIAN:
        return _gaussian_filter(values, sigma)
    if method == SMOOTH_METHOD_MEDIAN:
        return _median_filter(values, window)
    raise ValueError(f"unknown smoothing method {method!r}")

def _savgol_filter(values: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Savitzky-Golay 滤波。

    每个点都用它左右各半个窗口的点拟合一个 ``polyorder`` 阶多项式，然后
    取拟合值在中心点的取值。线性信号会被原样保留（多项式拟合是精确的），
    抖动则被压掉。

    窗口缩短到点数以内；窗口比多项式阶数还小时就不过滤。
    """
    count = values.shape[0]
    order = max(0, int(polyorder))

    length = max(3, int(window))
    if length % 2 == 0:
        length += 1
    if count % 2 == 0:
        length = min(length, count - 1)
    length = min(length, count)
    if length <= order:
        return values.copy()

    half = length // 2
    smoothed = np.empty_like(values)
    for index in range(count):
        low = index - half
        high = low + length
        if low < 0:
            low, high = 0, length
        if high > count:
            high = count
            low = max(0, high - length)
        # 窗口内各点相对中心点的偏移
        offsets = np.arange(low, high, dtype=np.float64) - index
        design = np.vander(offsets, order + 1, increasing=True)
        coefficients, *_ = np.linalg.lstsq(design, values[low:high], rcond=None)
        # 偏移 0 处的取值就是中心点的滤波结果
        smoothed[index] = coefficients[0]
    return smoothed


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """居中的滑动平均（边缘用最近的点补齐）。"""
    count = values.shape[0]
    length = max(1, min(int(window), count))
    if length == 1:
        return values.copy()
    left = (length - 1) // 2
    right = length - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    kernel = np.ones(length, dtype=np.float64) / length
    return np.convolve(padded, kernel, mode="valid")


def _gaussian_filter(values: np.ndarray, sigma: float) -> np.ndarray:
    """高斯核滤波（边缘用最近的点补齐）。"""
    count = values.shape[0]
    width = float(sigma)
    if width <= 0.0 or count < 2:
        return values.copy()
    radius = max(1, int(math.ceil(3.0 * width)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / width) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _median_filter(values: np.ndarray, window: int) -> np.ndarray:
    """中值滤波：压掉孤立尖峰。"""
    count = values.shape[0]
    length = max(1, min(int(window), count))
    if length % 2 == 0:
        length -= 1
    if length <= 1:
        return values.copy()
    half = length // 2
    padded = np.pad(values, (half, half), mode="edge")
    smoothed = np.empty_like(values)
    for index in range(count):
        smoothed[index] = np.median(padded[index : index + length])
    return smoothed


def _uniform_grid(times: Optional[Sequence[float]], count: int) -> Optional[np.ndarray]:
    """均匀的时间网格 —— 原来的间隔不均匀时才需要。

    Returns:
        均匀的 ``(count,)`` 时间网格；间隔本来就均匀（或者没有时间戳）时
        返回 ``None``（不需要重采样）。
    """
    if times is None:
        return None
    stamps = np.asarray(times, dtype=np.float64)
    if stamps.shape != (count,) or not np.all(np.isfinite(stamps)):
        return None
    steps = np.diff(stamps)
    if steps.size == 0 or np.any(steps <= 0.0):
        return None
    mean_step = float(steps.mean())
    if np.allclose(steps, mean_step, rtol=0.01, atol=1e-12):
        return None
    return np.linspace(float(stamps[0]), float(stamps[-1]), count)


def _resample(
    values: np.ndarray, source: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """把 ``values`` 线性插值到 ``target`` 的时间点上（每一列单独插值）。"""
    resampled = np.empty_like(values)
    for column in range(values.shape[1]):
        resampled[:, column] = np.interp(target, source, values[:, column])
    return resampled

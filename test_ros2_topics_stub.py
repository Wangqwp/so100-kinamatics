#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Windows 离线冒烟测试：用 ROS 桩验证「单 topic」轨迹节点。

运行::

    python test_ros2_topics_stub.py

验证：订阅/发布同 topic、两种输入帧都能产出 Path、自己发布的消息不再被处理。
"""

from __future__ import annotations

import os
import sys
import types

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ----------------------------------------------------------------------
# 最小 ROS 2 桩
# ----------------------------------------------------------------------
class _Stamp:
    def __init__(self, sec=0, nanosec=0):
        self.sec = sec
        self.nanosec = nanosec


class _Header:
    def __init__(self):
        self.stamp = _Stamp()
        self.frame_id = ""


class _Point:
    def __init__(self):
        self.x = self.y = self.z = 0.0


class _Quaternion:
    def __init__(self):
        self.x = self.y = self.z = 0.0
        self.w = 1.0


class _Pose:
    def __init__(self):
        self.position = _Point()
        self.orientation = _Quaternion()


class PoseStamped:
    def __init__(self):
        self.header = _Header()
        self.pose = _Pose()


class Path:
    def __init__(self):
        self.header = _Header()
        self.poses = []


class Logger:
    def __init__(self):
        self.messages = []

    def _log(self, level, message):
        self.messages.append((level, message))

    def info(self, message):
        self._log("INFO", message)

    def warn(self, message):
        self._log("WARN", message)

    def debug(self, message):
        self._log("DEBUG", message)

    def fatal(self, message):
        self._log("FATAL", message)


class Publisher:
    def __init__(self, topic):
        self.topic = topic
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class Parameter:
    def __init__(self, name, value):
        self.name = name
        self.value = value


class _ParameterValue:
    def __init__(self, value):
        self.value = value


class Clock:
    def now(self):
        return types.SimpleNamespace(to_msg=lambda: _Stamp(123, 456000000))


class Node:
    def __init__(self, name, parameter_overrides=None):
        self.name = name
        self._logger = Logger()
        self._params = {}
        for override in parameter_overrides or []:
            self._params[override.name] = override.value
        self._clock = Clock()

    def declare_parameter(self, name, value):
        self._params.setdefault(name, value)

    def get_parameter(self, name):
        if name not in self._params:
            raise KeyError(name)
        return _ParameterValue(self._params[name])

    def create_publisher(self, msg_type, topic, depth):
        return Publisher(topic)

    def create_subscription(self, msg_type, topic, callback, depth):
        return (topic, callback)

    def create_timer(self, period, callback):
        return (period, callback)

    def get_logger(self):
        return self._logger

    def get_clock(self):
        return self._clock


def install_stubs():
    def module(name, **attributes):
        fake = types.ModuleType(name)
        for key, value in attributes.items():
            setattr(fake, key, value)
        sys.modules[name] = fake
        return fake

    module("rclpy", init=lambda args=None: None, shutdown=lambda: None,
           ok=lambda: True, spin=lambda node: None)
    module("rclpy.node", Node=Node)
    module("rclpy.parameter", Parameter=Parameter)
    module("rclpy.duration", Duration=object)
    module("rclpy.time", Time=object)
    module("builtin_interfaces")
    module("builtin_interfaces.msg", Time=_Stamp, Duration=_Stamp)
    module("geometry_msgs")
    module("geometry_msgs.msg", Point=_Point, PoseStamped=PoseStamped)
    module("nav_msgs")
    module("nav_msgs.msg", Path=Path)


def _pose(position, orientation=(0.0, 0.0, 0.0, 1.0), frame=""):
    pose = PoseStamped()
    pose.header.frame_id = frame
    pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = position
    (
        pose.pose.orientation.x,
        pose.pose.orientation.y,
        pose.pose.orientation.z,
        pose.pose.orientation.w,
    ) = orientation
    return pose


def _joint_input():
    path = Path()
    path.header.frame_id = "joint_input"
    path.header.stamp = _Stamp(1, 0)
    home = (-6.0, -50.0, 20.0, 85.0, -50.0)
    for index in range(8):
        values = [
            home[0] + index,
            home[1] + 2 * index,
            home[2] - index,
            home[3] + index,
            home[4],
            50.0,
        ]
        pose = _pose(values[:3], (*values[3:], 0.0), frame="joint_input")
        pose.header.stamp = _Stamp(0, index * 50_000_000)
        path.poses.append(pose)
    return path


def _pose_input():
    path = Path()
    path.header.frame_id = "pose_input"
    path.poses.append(_pose((0.2, 0.1, 0.2), frame="pose_input"))
    path.poses.append(_pose((0.3, 0.0, 0.1), frame="pose_input"))
    return path


def main():
    install_stubs()
    import trajectory_bus_node as bus

    overrides = [
        Parameter(
            "urdf_path", os.path.join(ROOT, "config", "so101_demo.urdf")
        ),
        Parameter(
            "calibration_path",
            os.path.join(ROOT, "calibration", "orange_robot.json"),
        ),
        Parameter("trajectory_topic", "/so100/trajectory"),
    ]
    node = bus.TrajectoryBusNode(parameter_overrides=overrides)

    assert node._publisher.topic == "/so100/trajectory"
    assert node._subscription[0] == "/so100/trajectory"
    print("PASS 订阅/发布使用同一个 topic")

    node._on_path(_joint_input())
    raw_out = node._publisher.messages[-1]
    assert raw_out.header.frame_id == "base_link"
    assert len(raw_out.poses) == 8
    assert any(abs(p.pose.orientation.w) > 1e-6 for p in raw_out.poses)
    print(f"PASS joint_input -> EE Path ({len(raw_out.poses)} 点)")

    node._on_path(_pose_input())
    hole_out = node._publisher.messages[-1]
    assert hole_out.header.frame_id == "base_link"
    assert len(hole_out.poses) > 10
    print(f"PASS pose_input -> 插入 Path ({len(hole_out.poses)} 点)")

    before = len(node._publisher.messages)
    node._on_path(raw_out)  # 自己发布的消息必须被忽略
    assert len(node._publisher.messages) == before
    print("PASS 自发布消息被忽略（无自环）")

    print("\n单 topic ROS 2 桩测试通过")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""单 topic 演示数据发布器：只往 /so100/trajectory 写输入帧消息。

``input_mode`` 可选 ``joint`` / ``pose`` / ``alternate``（默认交替），
让同一个 topic 轮流驱动策略 EE 轨迹和孔位插入轨迹。
"""

from __future__ import annotations

import math

import rclpy
from builtin_interfaces.msg import Time as RosTime
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.parameter import Parameter

from trajectory_bus_node import FRAME_JOINT_INPUT, FRAME_POSE_INPUT

DEFAULT_TOPIC = "/so100/trajectory"
_HOME_DEG = (-6.0, -50.0, 20.0, 85.0, -50.0)


def _stamp(seconds: float) -> RosTime:
    total = int(round(seconds * 1_000_000_000))
    stamp = RosTime()
    stamp.sec = total // 1_000_000_000
    stamp.nanosec = total % 1_000_000_000
    return stamp


class DemoTrajectoryBusPublisher(Node):
    """向唯一的轨迹 topic 发布 joint_input / pose_input 两种输入。"""

    def __init__(
        self, parameter_overrides: list[Parameter] | None = None
    ) -> None:
        super().__init__(
            "demo_trajectory_bus_publisher", parameter_overrides=parameter_overrides
        )
        self.declare_parameter("trajectory_topic", DEFAULT_TOPIC)
        self.declare_parameter("input_mode", "alternate")
        self.declare_parameter("publish_rate_hz", 2.0)
        self.declare_parameter("chunk_points", 12)
        self.declare_parameter("chunk_dt_s", 0.05)

        self._topic = str(self.get_parameter("trajectory_topic").value) or DEFAULT_TOPIC
        self._mode = str(self.get_parameter("input_mode").value).strip().lower()
        self._rate = max(0.1, float(self.get_parameter("publish_rate_hz").value))
        self._points = max(2, int(self.get_parameter("chunk_points").value))
        self._dt = max(0.01, float(self.get_parameter("chunk_dt_s").value))
        self._phase = 0.0
        self._tick = 0

        self._publisher = self.create_publisher(Path, self._topic, 10)
        self.create_timer(1.0 / self._rate, self._on_tick)
        self.get_logger().info(f"演示输入发布到 {self._topic}（模式 {self._mode}）")

    def _on_tick(self) -> None:
        if self._mode == "joint":
            use_joint = True
        elif self._mode == "pose":
            use_joint = False
        else:
            use_joint = self._tick % 2 == 0
        self._tick += 1
        self._publisher.publish(
            self._joint_input() if use_joint else self._pose_input()
        )

    def _joint_input(self) -> Path:
        path = Path()
        path.header.frame_id = FRAME_JOINT_INPUT
        path.header.stamp = self.get_clock().now().to_msg()
        for index in range(self._points):
            phase = self._phase + 0.15 * index
            pose = PoseStamped()
            pose.header.frame_id = FRAME_JOINT_INPUT
            pose.header.stamp = _stamp(index * self._dt)
            pose.pose.position.x = _HOME_DEG[0] + 3.0 * math.sin(phase)
            pose.pose.position.y = _HOME_DEG[1] + 4.0 * math.sin(phase + 0.5)
            pose.pose.position.z = _HOME_DEG[2] + 3.0 * math.cos(phase)
            pose.pose.orientation.x = _HOME_DEG[3] + 2.0 * math.sin(phase + 1.0)
            pose.pose.orientation.y = _HOME_DEG[4]
            pose.pose.orientation.z = 50.0  # gripper
            pose.pose.orientation.w = 0.0  # 输入标记
            path.poses.append(pose)
        self._phase += 0.08
        return path

    def _pose_input(self) -> Path:
        path = Path()
        path.header.frame_id = FRAME_POSE_INPUT
        path.header.stamp = self.get_clock().now().to_msg()

        tip = PoseStamped()
        tip.header.frame_id = FRAME_POSE_INPUT
        tip.pose.position.x = 0.2 + 0.01 * math.sin(self._phase)
        tip.pose.position.y = 0.1 + 0.01 * math.cos(self._phase)
        tip.pose.position.z = 0.2
        tip.pose.orientation.w = 1.0

        hole = PoseStamped()
        hole.header.frame_id = FRAME_POSE_INPUT
        hole.pose.position.x = 0.3
        hole.pose.position.y = 0.0
        hole.pose.position.z = 0.1
        hole.pose.orientation.w = 1.0

        path.poses.extend([tip, hole])
        return path


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DemoTrajectoryBusPublisher()
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

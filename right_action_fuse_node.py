#!/usr/bin/env python3
import copy
import math
import time

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState


class TopicRelay(Node):

    def __init__(self):
        super().__init__('right_topic_relay')

        # 参数
        self.declare_parameter('input_topic', '/so100/cmd_joint')
        self.declare_parameter('output_topic', '/so100/cmd_joint_fused')
        self.declare_parameter('joint_states_topic', '/so100_left/joint_states')
        self.declare_parameter('gripper_joint_name', 'shoulder_pan')

        # 阈值，单位与 /so100/joint_states 里 gripper 的数值一致。
        # 该话题发布的是归一化值（0..100），不是弧度，所以不能再把
        # 阈值换算成弧度去比较（8deg -> 0.14 会让任何读数都算“闭合”）。
        self.declare_parameter('gripper_threshold', 15.0)

        # gripper 数值「大」还是「小」代表闭合：
        #   high -> 数值变大即闭合（默认；例如 0 张开、100 闭合）
        #   low  -> 反装/反向标定的情况，0 附近算闭合
        self.declare_parameter('gripper_closed_when', 'high')

        self.declare_parameter('debounce_time', 2.0)

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        joint_states_topic = self.get_parameter('joint_states_topic').value

        self.gripper_joint_name = self.get_parameter(
            'gripper_joint_name'
        ).value

        threshold = float(self.get_parameter('gripper_threshold').value)

        # 夹爪读数可能为负（与安装方向有关），阈值只关心绝对值
        self.gripper_threshold = abs(threshold)

        if self.gripper_threshold <= 0.0:
            raise ValueError(
                'gripper_threshold must be greater than 0, '
                f'got {self.gripper_threshold!r}'
            )

        self.gripper_closed_when = str(
            self.get_parameter('gripper_closed_when').value
        ).strip().lower()

        if self.gripper_closed_when not in ('high', 'low'):
            raise ValueError(
                'gripper_closed_when must be "high" or "low", '
                f'got {self.gripper_closed_when!r}'
            )

        self.debounce_time = float(
            self.get_parameter('debounce_time').value
        )

        # 状态
        self.in_align_zone = False
        self.pending_state = False
        self.pending_since = None

        # 当前输入关节位置缓存
        self.latest_position = None

        # 锁定关节位置
        self.locked_position = None

        # Publisher
        self.publisher = self.create_publisher(
            JointState,
            output_topic,
            10
        )

        # 机械臂指令 Subscriber
        self.subscription = self.create_subscription(
            JointState,
            input_topic,
            self.joint_callback,
            10
        )

        # 实际关节状态 Subscriber
        self.joint_states_subscription = self.create_subscription(
            JointState,
            joint_states_topic,
            self.joint_states_callback,
            10
        )

        # 定时检查去抖状态
        self.timer = self.create_timer(
            0.02,
            self.update_state
        )

        self.get_logger().info(
            f'Relay started: {input_topic} -> {output_topic}\n'
            f'Gripper topic: {joint_states_topic}\n'
            f'Gripper joint: {self.gripper_joint_name}\n'
            f'Threshold: {self.gripper_threshold:g} (joint-state unit)\n'
            f'Closed when gripper is: {self.gripper_closed_when}\n'
            f'Debounce: {self.debounce_time:.1f}s'
        )

    def joint_states_callback(self, msg: JointState):
        """根据实际 Gripper 角度更新候选状态"""

        # 同时兼容 "gripper" 和带前缀的 "left_gripper"
        index = self._find_joint_index(msg.name, self.gripper_joint_name)

        if index is None or index >= len(msg.position):
            return

        gripper_angle = float(msg.position[index])

        if not math.isfinite(gripper_angle):
            return

        # 判断 Gripper 是否闭合。阈值和读数必须使用同一单位：
        # /so100/joint_states 发布的是归一化值（0..100），不是弧度，
        # 若把阈值换算成弧度（8deg -> 0.14），几乎任何读数都会 >= 阈值，
        # 节点就会在启动后立刻锁定，而且夹爪张开后仍然判定为闭合，
        # 永远无法退出锁定状态。
        if self.gripper_closed_when == 'low':
            current_state = abs(gripper_angle) <= self.gripper_threshold
        else:
            current_state = abs(gripper_angle) >= self.gripper_threshold

        # 候选状态发生变化，重新开始计时
        if current_state != self.pending_state:
            self.pending_state = current_state
            self.pending_since = time.monotonic()

            self.get_logger().debug(
                f'gripper={gripper_angle:.3f}, '
                f'pending lock={current_state}'
            )

    def update_state(self):
        """去抖确认是否进入或退出锁定状态"""

        if self.pending_since is None:
            return

        # 候选状态与已确认状态一致，无需切换
        if self.pending_state == self.in_align_zone:
            self.pending_since = None
            return

        now = time.monotonic()

        if now - self.pending_since < self.debounce_time:
            return

        # 确认状态切换
        self.in_align_zone = self.pending_state
        self.pending_since = None

        if self.in_align_zone:

            # 进入锁定状态
            if self.latest_position is not None:
                self.locked_position = list(self.latest_position)

            self.get_logger().info(
                f'Gripper closed. Lock position: {self.locked_position}'
            )

        else:

            # 退出锁定状态
            self.locked_position = None

            self.get_logger().info(
                'Gripper opened. Resume normal relay.'
            )

    @staticmethod
    def _find_joint_index(names, joint_name):
        """在名字列表里找到 joint_name 的下标（兼容带前缀的名字）"""

        try:
            return names.index(joint_name)

        except ValueError:
            for index, name in enumerate(names):
                if name.endswith(joint_name):
                    return index

        return None

    def joint_callback(self, msg: JointState):
        """根据锁定状态转发关节指令"""

        self.latest_position = list(msg.position)

        output_msg = copy.deepcopy(msg)

        if self.in_align_zone:

            # 如果进入锁定状态时还没有收到指令，
            # 则使用首次收到的指令位置进行锁定
            if self.locked_position is None:
                self.locked_position = list(msg.position)

                self.get_logger().info(
                    f'First command received. '
                    f'Lock position: {self.locked_position}'
                )

            positions = list(self.locked_position)

            # 夹爪关节不参与锁定：否则夹爪指令被冻结在闭合位置，
            # 实际夹爪永远不会张开，节点也就永远无法退出锁定状态。
            index = self._find_joint_index(
                msg.name,
                self.gripper_joint_name
            )

            if (
                index is not None
                and index < len(positions)
                and index < len(msg.position)
            ):
                positions[index] = msg.position[index]

            output_msg.position = positions

        self.publisher.publish(output_msg)


def main(args=None):
    rclpy.init(args=args)

    node = TopicRelay()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
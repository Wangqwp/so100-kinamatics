#!/usr/bin/env python3
import rclpy
from rclpy.node import Node


class TopicRelay(Node):

    def __init__(self):
        super().__init__('topic_relay')

        # 声明参数
        self.declare_parameter('input_topic', '/so100_left/cmd_joint')
        self.declare_parameter('output_topic', '/so100_left/cmd_joint_fused')
        self.declare_parameter('msg_type', 'sensor_msgs/msg/JointState')

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        msg_type = self.get_parameter('msg_type').value

        # 动态加载消息类型
        from rosidl_runtime_py.utilities import get_message
        message_type = get_message(msg_type)

        # 创建 Publisher
        self.publisher = self.create_publisher(
            message_type,
            output_topic,
            10
        )

        # 创建 Subscriber
        self.subscription = self.create_subscription(
            message_type,
            input_topic,
            self.callback,
            10
        )

        self.get_logger().info(
            f'Relay started: {input_topic} -> {output_topic} '
            f'[{msg_type}]'
        )

    def callback(self, msg):
        self.publisher.publish(msg)


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
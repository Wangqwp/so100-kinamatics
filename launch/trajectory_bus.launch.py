#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""单 topic 轨迹方案：一个节点订阅+发布同一个 Path topic，RViz 直接显示。

用法::

    ros2 launch so100_trajectory trajectory_bus.launch.py
    ros2 launch so100_trajectory trajectory_bus.launch.py \
        use_demo:=false urdf_path:=/你的/标定urdf
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("so100_trajectory")
    default_urdf = os.path.join(share, "config", "so101_demo.urdf")
    default_calibration = os.path.join(
        share, "calibration", "orange_robot.json"
    )
    default_rviz = os.path.join(share, "rviz", "trajectory_bus.rviz")

    use_demo = LaunchConfiguration("use_demo")
    use_rviz = LaunchConfiguration("use_rviz")
    topic = LaunchConfiguration("trajectory_topic")
    base_frame = LaunchConfiguration("base_frame")
    urdf_path = LaunchConfiguration("urdf_path")
    calibration_path = LaunchConfiguration("calibration_path")
    input_mode = LaunchConfiguration("input_mode")
    smooth_method = LaunchConfiguration("smooth_method")
    smooth_window = LaunchConfiguration("smooth_window")
    rviz_config = LaunchConfiguration("rviz_config")

    args = [
        DeclareLaunchArgument("use_demo", default_value="true"),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument(
            "trajectory_topic", default_value="/so100/trajectory"
        ),
        DeclareLaunchArgument("base_frame", default_value="base_link"),
        DeclareLaunchArgument("urdf_path", default_value=default_urdf),
        DeclareLaunchArgument(
            "calibration_path", default_value=default_calibration
        ),
        DeclareLaunchArgument("input_mode", default_value="alternate"),
        DeclareLaunchArgument("smooth_method", default_value="savgol"),
        DeclareLaunchArgument("smooth_window", default_value="11"),
        DeclareLaunchArgument("rviz_config", default_value=default_rviz),
    ]

    demo_node = Node(
        package="so100_trajectory",
        executable="demo_trajectory_bus",
        name="demo_trajectory_bus",
        output="screen",
        parameters=[
            {"trajectory_topic": topic, "input_mode": input_mode}
        ],
        condition=IfCondition(use_demo),
    )

    bus_node = Node(
        package="so100_trajectory",
        executable="trajectory_bus",
        name="trajectory_bus",
        output="screen",
        parameters=[
            {
                "trajectory_topic": topic,
                "base_frame": base_frame,
                "urdf_path": urdf_path,
                "calibration_path": calibration_path,
                "action_unit": "norm",
                "smooth_method": smooth_method,
                "smooth_window": smooth_window,
            }
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
        output="screen",
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription(args + [demo_node, bus_node, rviz_node])

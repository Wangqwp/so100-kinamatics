# Quickstart
使用`lerobot_ros2` conda环境
```
conda activate lerobot_ros2
```

## 启动Right SO100/101机械臂的action fusion ROS2节点
/so100/cmd_joint -> /so100/cmd_joint_fused

不做平滑处理，直接使用policy action
```
python right_action_fuse_node.py
```
均值滤波平滑处理
```
python right_action_smooth_node.py
```

## 启动Left SO100/101机械臂的action fusion ROS2节点
/so100_left/cmd_joint -> /so100_left/cmd_joint_fused
不做平滑处理，直接使用policy action
```
python action_fuse_node.py
```

## so101_pose_node节点
订阅 /so100/joint_states（关节单位：norm），发布 /so100/ee_pose（坐标系：base_link）
```
python so101_pose_node.py
```

## policy_trajectory_node节点
订阅 /so100/policy/action_chunk（关节单位：norm），发布 /so100/policy/opt_ee_trajectory（坐标系：base_link）

不做平滑处理的policy action chunk
```
python policy_trajectory_node.py
```

savgol平滑后的policy action chunk
```
python opt_policy_trajectory_node.py --ros-args -p smooth_method:=savgol -p smooth_window:=16
```
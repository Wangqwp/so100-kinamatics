# SO100/101 单 topic 轨迹方案（ROS 2 包）

本目录是 `ament_python` 包 `so100_trajectory`。**数据获取、计算、发布全部走
同一个 topic**：`/so100/trajectory`（`nav_msgs/Path`）。

- 输入：`header.frame_id = joint_input` 或 `pose_input`
- 输出：`header.frame_id = base_link`，发布到**同一个** topic
- 节点忽略自己发布的消息（`frame_id == base_link`），不会自激循环
- RViz 直接显示这一个 Path topic

## 1. 拷到 Linux 并构建

```bash
sudo apt install ros-$ROS_DISTRO-rviz2 python3-numpy
mkdir -p ~/ros2_ws/src
cp -r so100-kinematics ~/ros2_ws/src/so100_trajectory
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select so100_trajectory
source install/setup.bash
```

## 2. 一键启动

```bash
ros2 launch so100_trajectory trajectory_bus.launch.py
```

默认启动：演示输入发布器 + 单 topic 轨迹节点 + RViz2（`rviz/trajectory_bus.rviz`）。
接真实机器人：

```bash
ros2 launch so100_trajectory trajectory_bus.launch.py \
  use_demo:=false urdf_path:=/你的/标定urdf calibration_path:=/你的/orange_robot.json
```

launch 参数：`use_demo`、`use_rviz`、`trajectory_topic`、`base_frame`、`urdf_path`、
`calibration_path`、`input_mode`（joint/pose/alternate）、`smooth_method`、`smooth_window`、
`rviz_config`。

## 3. 唯一 topic 的数据协议

| 阶段 | `header.frame_id` | `poses` 内容 |
|---|---|---|
| 数据获取（输入） | `joint_input` | 每个点编码一个 action chunk：`position.x/y/z` = 前三根手臂关节（度），`orientation.x/y` = 后两根关节（度），`orientation.z` = gripper，`orientation.w` = 0；`pose.header.stamp` 写相对时间 |
| 数据获取（输入） | `pose_input` | 前两个 pose = 夹爪尖端、孔位姿（真实四元数） |
| 计算+发布（输出） | `base_link` | FK 解出的 EE 轨迹，或孔位插入的夹爪控制点轨迹；真实四元数 |

`norm` 约定：LeRobot 手臂关节是度（-100..100）、夹爪 0..100；只有 |v| > 200
时才用标定 JSON 解码原始 tick。

RViz 可能对 `joint_input` / `pose_input` 这两个输入帧打印 "frame not found"
警告，可忽略；实际显示的是帧号为 `base_link` 的输出轨迹。

单独运行：

```bash
ros2 run so100_trajectory trajectory_bus
ros2 run so100_trajectory demo_trajectory_bus --ros-args -p input_mode:=alternate
rviz2 -d install/so100_trajectory/share/so100_trajectory/rviz/trajectory_bus.rviz
```

> 旧的多节点实现（`policy_trajectory_node.py`、`opt_policy_trajectory_node.py`、
> `hole_insertion_trajectory_node.py`）仅作为源码参考保留，已从构建入口移除，
> 不参与单 topic 方案。

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
订阅 /so100/policy/action_chunk（关节单位：norm），发布 /so100/policy/ee_trajectory（坐标系：base_link）。
这两个 trajectory 节点已改为纯 numpy FK，不再需要 lerobot/placo 环境。

不做平滑处理的policy action chunk
```
python policy_trajectory_node.py
```

savgol平滑后的policy action chunk
```
python opt_policy_trajectory_node.py --ros-args -p smooth_method:=savgol -p smooth_window:=16
```

## 孔位插入轨迹节点 hole_insertion_trajectory_node.py
订阅上级夹爪**尖端**位姿和孔位姿，生成一条不碰撞的插入轨迹并可视化。

```
source /opt/ros/humble/setup.bash
python hole_insertion_trajectory_node.py
```

订阅（均为 `geometry_msgs/PoseStamped`，位姿里的四元数就是“旋度”，
`hole` 的局部 +Z 是孔的朝外方向）：
- `/so100/gripper_tip_pose`：夹爪尖端当前位置与姿态
- `/so100/hole_pose`：孔中心与朝向

发布：
- `/so100/hole_insertion/trajectory`（`nav_msgs/Path`）：夹爪**控制点**轨迹。
  每个尖端目标沿**世界固定方向**回退 3 cm（默认 `+Z`，即向基座方向），
  使夹爪尖端最终准确落到孔轴上
- `/so100/hole_insertion/markers`（`visualization_msgs/MarkerArray`）：
  尖端轨迹线、孔轴箭头、孔/接近点/插入点/起点球和文字标注

轨迹分四段：先沿孔轴抬升到安全高度 → 在安全高度平移到孔正上方并对齐姿态
→ 沿孔轴下降至孔口上方 6 cm → 竖直向下插入 3 cm。平移段始终高于孔口、
下降段始终贴合孔轴，因此不会撞到孔口或孔壁（孔简化为圆柱，不含机器人自身
关节的碰撞检测）。

可视化：
```
rviz2 -d rviz/hole_insertion.rviz
```

常用参数（全部可用 `--ros-args -p 参数名:=值` 覆盖）：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `gripper_tip_topic` | `/so100/gripper_tip_pose` | 夹爪尖端位姿话题 |
| `hole_topic` | `/so100/hole_pose` | 孔位姿话题 |
| `trajectory_topic` | `/so100/hole_insertion/trajectory` | 控制点轨迹话题 |
| `markers_topic` | `/so100/hole_insertion/markers` | 可视化标记话题 |
| `base_frame` | `base_link` | 轨迹参考系；输入位姿会按需经 TF 变换到此系 |
| `approach_height_m` | `0.06` | 接近点距孔口高度（6 cm） |
| `insert_depth_m` | `0.03` | 插入深度（3 cm） |
| `base_offset_m` | `0.03` | 夹爪控制点向基座回退距离（3 cm） |
| `base_offset_axis` | `+z` | 世界坐标系回退方向，可设 `+x/-x/+y/-y/+z/-z` 或三个数字 |
| `lift_margin_m` | `0.05` | 安全高度裕量 |
| `waypoint_spacing_m` | `0.01` | 路径点最大间距 |
| `hole_radius_m` | `0.01` | 孔半径（用于避碰校验） |
| `plan_rate_hz` | `10.0` | 轨迹发布频率 |
| `max_pose_age_s` | `1.0` | 输入位姿最大允许过期时间，`<=0` 关闭检查 |

几何计算单独放在 `hole_insertion_geometry.py`，不依赖 ROS，可用
`python test_hole_insertion_geometry.py` 离线自测。
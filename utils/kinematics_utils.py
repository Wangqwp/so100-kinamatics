# SPDX-License-Identifier: Apache-2.0
"""
SO100 / SO101 运动学工具。

从 eval_so100.py 中提取的 FK / IK 初始化与调用逻辑，供其他控制脚本复用。

功能：
    - forward_kinematics(): 关节动作 -> 末端执行器位姿
    - inverse_kinematics(): 末端位姿 -> 关节动作
    - ee_pose_to_matrix(): FK 的 ee.* 字典 -> 4x4 齐次变换矩阵

注意：
    1. 关节名、URDF、目标末端 frame 应与实际机器人配置一致。
    2. IK 每次调用前默认 reset 管线，沿用原评测脚本的行为，避免求解器
       使用旧的 q_curr / EE bounds 缓存。
    3. IK 输入的 current_joint_action 用作当前关节状态和 IK transition 的观测。
"""
import os
import sys
from typing import Dict, List, Optional

import numpy as np

from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
)
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    robot_action_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    ForwardKinematicsJointsToEE,
    InverseKinematicsEEToJoints,
)

if __package__ in (None, ""):
    # 本文件被当作普通脚本直接运行时（例如
    # `python so100_ros/kinematics_utils.py`），Python 不会把 so100_ros
    # 当成包，相对导入会抛
    #   ImportError: attempted relative import with no known parent package。
    # 此时把包的父目录加入 sys.path，退化为绝对导入。
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.kinematics_backend import FixedRobotKinematics
else:
    from .kinematics_backend import FixedRobotKinematics


DEFAULT_URDF_PATH = "/home/user/workspace/github/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
DEFAULT_TARGET_FRAME = "gripper_frame_link"


def ee_pose_to_matrix(ee_pose: Dict[str, float]) -> np.ndarray:
    """将 FK 输出的 ee.x/y/z 与 ee.wx/wy/wz 转成 4x4 齐次变换矩阵。

    ee.wx/wy/wz 按旋转向量解释。使用 scipy 的 Rotation 将旋转向量转为矩阵。
    """
    from scipy.spatial.transform import Rotation

    position = np.array(
        [ee_pose["ee.x"], ee_pose["ee.y"], ee_pose["ee.z"]],
        dtype=np.float64,
    )
    rotvec = np.array(
        [ee_pose["ee.wx"], ee_pose["ee.wy"], ee_pose["ee.wz"]],
        dtype=np.float64,
    )

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    transform[:3, 3] = position
    return transform


class SO100KinematicsUtils:
    """SO100/SO101 FK、IK 封装。

    Args:
        motor_names: 机械臂 motor/joint 名称，顺序需与机器人 bus 一致。
        urdf_path: 机器人 URDF 路径。
        target_frame_name: FK/IK 使用的末端 frame。
        ee_bounds_min: 末端位置最小边界，单位米。
        ee_bounds_max: 末端位置最大边界，单位米。
        max_ee_step_m: 相邻 IK 目标允许的最大末端位置步长，单位米。
        initial_guess_current_joints: 是否让 LeRobot IK 使用当前关节作为初值。
        ik_iterations: 单次 IK 内部的线性化迭代次数。placo 的 solve(True) 只做
            一次线性化，大位姿变化会残留毫米级误差；3 次即可收敛到 0.02 mm。
    """

    def __init__(
        self,
        motor_names: List[str],
        urdf_path: str = DEFAULT_URDF_PATH,
        target_frame_name: str = DEFAULT_TARGET_FRAME,
        ee_bounds_min: Optional[List[float]] = None,
        ee_bounds_max: Optional[List[float]] = None,
        max_ee_step_m: float = 0.01,
        initial_guess_current_joints: bool = False,
        ik_iterations: int = 1,
    ) -> None:
        if not motor_names:
            raise ValueError("motor_names must not be empty.")

        self.motor_names = list(motor_names)
        self.kinematics = FixedRobotKinematics(
            urdf_path=urdf_path,
            target_frame_name=target_frame_name,
            joint_names=self.motor_names,
            ik_iterations=ik_iterations,
        )

        self.fk_pipeline = RobotProcessorPipeline[RobotAction, RobotAction](
            steps=[
                ForwardKinematicsJointsToEE(
                    kinematics=self.kinematics,
                    motor_names=self.motor_names,
                ),
            ],
            to_transition=robot_action_to_transition,
            to_output=transition_to_robot_action,
        )

        self.ik_pipeline = RobotProcessorPipeline[
            tuple[RobotAction, RobotObservation], RobotAction
        ](
            steps=[
                EEBoundsAndSafety(
                    end_effector_bounds={
                        "min": ee_bounds_min or [-1.0, -1.0, -1.0],
                        "max": ee_bounds_max or [1.0, 1.0, 1.0],
                    },
                    max_ee_step_m=max_ee_step_m,
                ),
                InverseKinematicsEEToJoints(
                    kinematics=self.kinematics,
                    motor_names=self.motor_names,
                    initial_guess_current_joints=initial_guess_current_joints,
                ),
            ],
            to_transition=robot_action_observation_to_transition,
            to_output=transition_to_robot_action,
        )

    def forward_kinematics(
        self,
        joint_action: Dict[str, float],
    ) -> Dict[str, float]:
        """FK：输入关节动作字典，输出 ee.x/y/z、ee.wx/wy/wz 等末端位姿字段。"""
        return dict(self.fk_pipeline(joint_action))

    def inverse_kinematics(
        self,
        ee_action: Dict[str, float],
        current_joint_action: Dict[str, float],
        reset_before_solve: bool = True,
    ) -> Dict[str, float]:
        """IK：输入末端目标与当前关节动作，输出关节动作字典。

        Args:
            ee_action: 末端目标，通常包含 ee.x/y/z、ee.wx/wy/wz。
            current_joint_action: 当前/参考关节动作，通常是当前策略关节动作。
            reset_before_solve: 是否在求解前重置 IK pipeline。默认 True，与原脚本一致。
        """
        if reset_before_solve:
            self.ik_pipeline.reset()

        result = self.ik_pipeline((dict(ee_action), dict(current_joint_action)))
        return dict(result)

if __name__ == "__main__":
    print("====Test SO100 Kinematics Utils====")
    so100 = SO100KinematicsUtils(
            motor_names=[
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
            ]
        )

    # joint_action: Dict[str, float]
    TARGET_JOINT: Dict[str, float] = {
        "shoulder_pan.pos": -6.3602,
        "shoulder_lift.pos": -51.9492,
        "elbow_flex.pos": 16.7276,
        "wrist_flex.pos": 89.2483,
        "wrist_roll.pos": -51.8499,
        "gripper.pos": 40.0,
    }

    CURRENT_JOINT: Dict[str, float] = {
        "shoulder_pan.pos": -6.0,
        "shoulder_lift.pos": -50.0,
        "elbow_flex.pos": 20.0,
        "wrist_flex.pos": 85.0,
        "wrist_roll.pos": -50.0,
        "gripper.pos": 38.0,
    }

    HOME_POSE: Dict[str, float] = {
        "shoulder_pan.pos": -6.2835,
        "shoulder_lift.pos": -91.4407,
        "elbow_flex.pos": 93.1444,
        "wrist_flex.pos": 69.1434,
        "wrist_roll.pos": -51.9027,
        "gripper.pos": 0.0707,
    }
    current_pose = so100.forward_kinematics(CURRENT_JOINT)
    print("current_pose: ", current_pose)

    target_pose = so100.forward_kinematics(TARGET_JOINT)
    print("target_pose: ", target_pose)

    target_joint = so100.inverse_kinematics(target_pose, CURRENT_JOINT)
    print("target_joint: ", target_joint)

    # IK 结果相对当前关节位置（HOME_POSE）的变化
    print("\njoint            current        ik_out         delta")
    for name in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]:
        current = float(HOME_POSE[f"{name}.pos"])
        solved = float(target_joint[f"{name}.pos"])
        print(f"{name:16s} {current:10.4f} {solved:13.4f} {solved - current:10.4f}")

    # IK 结果的实际末端位姿与目标末端位姿的误差
    checked = so100.forward_kinematics(
        {**{f"{name}.pos": float(target_joint[f"{name}.pos"])
            for name in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]},
         "gripper.pos": 0.0}
    )
    error = np.linalg.norm(
        np.array([checked["ee.x"] - target_pose["ee.x"],
                  checked["ee.y"] - target_pose["ee.y"],
                  checked["ee.z"] - target_pose["ee.z"]])
    )
    print(f"\nEE position error of FK(IK result) vs target: {error * 1000:.3f} mm")
    print("(use ik_iterations=3 to converge: the residual drops to ~0.02 mm)")
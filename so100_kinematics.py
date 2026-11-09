# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure-numpy forward kinematics and Jacobian of the SO-101 arm.

Why this module exists
----------------------
The CV alignment stage (see :mod:`cv_align`) has to know *where the end
effector is* for a given joint command, and it has to do that **inside the
ROS node**.  The node runs under the ROS Python (3.10), which cannot install
``lerobot`` (it drags in torch), so the LeRobot kinematics of
:mod:`kinematics_utils` are simply not available there.

What the fusion needs is much less than a full kinematics package:

* the **forward kinematics** (joint angles -> end-effector pose), and
* the **geometric Jacobian** (how the end-effector moves when a joint moves).

Both are closed-form for a serial chain of revolute joints, so they are
implemented here with nothing but ``numpy`` and the standard library
(``xml.etree`` reads the URDF).  No iterative solver, no seed, no convergence
problem - which is exactly what a 30 Hz control loop wants.

The module is deliberately ROS-free and dependency-light (numpy only), so the
repo's CPU test job can cover it.

Units
-----
The joint values are the ones the policy emits (the SO-ARM100 dataset units);
like the LeRobot backend, the kinematics interprets them as **degrees**.
Positions are metres in the base frame (the one given by ``base_frame``,
``base_link`` by default), angles are radians inside the module.

A joint state does not always use the same unit: the SO-101 bridge publishes
the angles in degrees, while the ROS convention is radians.  The helpers
below (:func:`normalize_joint_unit`, :func:`detect_joint_unit` and
:func:`joint_values_to_degrees`) translate a joint state into the degrees the
kinematics expects.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

import numpy as np



#: The calibrated SO-101 URDF (same file the LeRobot backend uses).
DEFAULT_URDF_PATH = "/home/user/workspace/github/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"

#: The frame the end-effector pose is expressed in (the grasp point).
DEFAULT_TARGET_FRAME = "gripper_frame_link"

#: The frame the pose is *measured from* - the base of the arm.  The pose
#: is published relative to it, so ``base_link`` is the natural choice: it
#: is the frame the robot description is rooted in and the one a camera
#: or a visualisation tool expects.
DEFAULT_BASE_FRAME = "base_link"


#: The five arm joints, in the order of the policy state vector.  The
#: gripper is *not* part of the arm chain - the visual target never moves it.
ARM_JOINT_NAMES: Tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)


class KinematicsError(RuntimeError):
    """The URDF cannot be read or does not describe the expected chain."""


# ---------------------------------------------------------------------------
# The unit of the joint values
# ---------------------------------------------------------------------------

#: A joint state carries its angles in radians - the ROS convention - or in
#: degrees, the unit the SO-101 firmware and the LeRobot bridge use.
JOINT_UNIT_RAD = "rad"
JOINT_UNIT_DEG = "deg"

#: ``auto`` lets the node guess the unit from the values themselves.
JOINT_UNIT_AUTO = "auto"

#: The unit the joint states of the SO-101 are published in: the bridge
#: hands out the angles in degrees.
DEFAULT_JOINT_UNIT = JOINT_UNIT_DEG

#: The joint angles of the SO-101 are limited to less than pi radians (the
#: widest range is ``wrist_roll`` with +-2.85 rad), so a value above pi
#: cannot be a radian value - it has to be a degree value.
JOINT_ANGLE_RAD_LIMIT = math.pi

#: The spellings accepted by :func:`normalize_joint_unit`.
_JOINT_UNIT_ALIASES = {
    "rad": JOINT_UNIT_RAD,
    "radian": JOINT_UNIT_RAD,
    "radians": JOINT_UNIT_RAD,
    "deg": JOINT_UNIT_DEG,
    "degree": JOINT_UNIT_DEG,
    "degrees": JOINT_UNIT_DEG,
    "auto": JOINT_UNIT_AUTO,
    "": JOINT_UNIT_AUTO,
}


def normalize_joint_unit(unit: str) -> str:
    """The canonical name of a joint unit: ``'rad'``, ``'deg'`` or ``'auto'``.

    Raises:
        KinematicsError: the unit is none of the known ones.
    """
    text = str(unit).strip().lower()
    try:
        return _JOINT_UNIT_ALIASES[text]
    except KeyError:
        raise KinematicsError(
            f"unknown joint unit {unit!r} (expected 'rad', 'deg' or 'auto')"
        ) from None


def detect_joint_unit(values: Sequence[float]) -> str:
    """Guess the unit of the given joint values.

    The angles of the SO-101 are limited to less than pi radians, so a value
    above that limit can only be a degree value.  Everything else is taken
    for radians - the ROS convention.
    """
    for value in values:
        if abs(float(value)) > JOINT_ANGLE_RAD_LIMIT:
            return JOINT_UNIT_DEG
    return JOINT_UNIT_RAD


def joint_values_to_degrees(
    values: Mapping[str, float], unit: str
) -> Dict[str, float]:
    """The joint values as degrees.

    Args:
        values: The joint values, keyed by the joint name.
        unit: ``'rad'``, ``'deg'`` or ``'auto'``; with ``'auto'`` the unit
            is guessed from the values (see :func:`detect_joint_unit`).
    """
    canonical = normalize_joint_unit(unit)
    if canonical == JOINT_UNIT_AUTO:
        canonical = detect_joint_unit(list(values.values()))
    if canonical == JOINT_UNIT_DEG:
        return {name: float(value) for name, value in values.items()}
    return {name: math.degrees(float(value)) for name, value in values.items()}


# ---------------------------------------------------------------------------
# Rotation helpers (numpy only - no scipy in the ROS Python)
# ---------------------------------------------------------------------------



def rpy_to_matrix(rpy: Sequence[float]) -> np.ndarray:
    """URDF ``rpy`` (fixed-axis roll, pitch, yaw) -> rotation matrix."""
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    rot_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rot_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rot_z @ rot_y @ rot_x


def axis_angle_to_matrix(axis: Sequence[float], angle: float) -> np.ndarray:
    """Rotation of ``angle`` radians about ``axis`` (Rodrigues' formula)."""
    direction = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        return np.eye(3)
    direction = direction / norm
    cross = np.array(
        [
            [0.0, -direction[2], direction[1]],
            [direction[2], 0.0, -direction[0]],
            [-direction[1], direction[0], 0.0],
        ]
    )
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)


def matrix_to_quaternion(matrix: np.ndarray) -> Tuple[float, float, float, float]:
    """Rotation matrix -> quaternion ``(x, y, z, w)`` (the ROS spelling)."""
    rot = np.asarray(matrix, dtype=np.float64)[:3, :3]
    trace = float(rot[0, 0] + rot[1, 1] + rot[2, 2])
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rot[2, 1] - rot[1, 2]) / scale
        y = (rot[0, 2] - rot[2, 0]) / scale
        z = (rot[1, 0] - rot[0, 1]) / scale
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        scale = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        w = (rot[2, 1] - rot[1, 2]) / scale
        x = 0.25 * scale
        y = (rot[0, 1] + rot[1, 0]) / scale
        z = (rot[0, 2] + rot[2, 0]) / scale
    elif rot[1, 1] > rot[2, 2]:
        scale = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        w = (rot[0, 2] - rot[2, 0]) / scale
        x = (rot[0, 1] + rot[1, 0]) / scale
        y = 0.25 * scale
        z = (rot[1, 2] + rot[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        w = (rot[1, 0] - rot[0, 1]) / scale
        x = (rot[0, 2] + rot[2, 0]) / scale
        y = (rot[1, 2] + rot[2, 1]) / scale
        z = 0.25 * scale
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / norm, y / norm, z / norm, w / norm)


def quaternion_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """Quaternion ``(x, y, z, w)`` -> rotation matrix."""
    x, y, z, w = (float(value) for value in quaternion)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        return np.eye(3)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (the ``ee.w*`` convention).

    The angle is in ``[0, pi]``, exactly like SciPy's ``as_rotvec()``, which
    is what the LeRobot backend uses - the two backends therefore return the
    same numbers for the same pose.
    """
    x, y, z, w = matrix_to_quaternion(matrix)
    if w < 0.0:  # canonical sign: the angle stays in [0, pi]
        x, y, z, w = -x, -y, -z, -w
    vector = np.array([x, y, z], dtype=np.float64)
    sin_half = float(np.linalg.norm(vector))
    if sin_half < 1e-12:
        return np.zeros(3)
    angle = 2.0 * math.atan2(sin_half, w)
    return vector / sin_half * angle


def rotvec_to_matrix(rotvec: Sequence[float]) -> np.ndarray:
    """Rotation vector (axis * angle) -> rotation matrix.

    The inverse of :func:`matrix_to_rotvec`: it turns the ``ee.w*`` values of
    the LeRobot spelling back into a matrix, so an end-effector pose that was
    computed once can be used wherever a matrix is expected.
    """
    vector = np.asarray(rotvec, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    axis = vector / angle
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)


def matrix_to_yaw(matrix: np.ndarray) -> float:
    """The yaw (rotation about the base z axis) of a rotation matrix."""
    rot = np.asarray(matrix, dtype=np.float64)[:3, :3]
    return math.atan2(float(rot[1, 0]), float(rot[0, 0]))


def wrap_angle(angle: float) -> float:
    """Wrap an angle into ``(-pi, pi]``."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# URDF parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UrdfJoint:
    """One joint of the URDF (only the fields the kinematics needs)."""

    name: str
    type: str
    parent: str
    child: str
    xyz: Tuple[float, float, float]
    rpy: Tuple[float, float, float]
    axis: Tuple[float, float, float]
    #: The limits of the joint in radians; ``None`` when the URDF does not
    #: give any - the joint is then unlimited.
    lower: Optional[float] = None
    upper: Optional[float] = None

    def origin_transform(self) -> np.ndarray:
        """The fixed transform from the parent frame to the joint frame."""
        transform = np.eye(4)
        transform[:3, :3] = rpy_to_matrix(self.rpy)
        transform[:3, 3] = np.asarray(self.xyz, dtype=np.float64)
        return transform


def _parse_vector(
    text: Optional[str], default: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    if not text:
        return default
    values = [float(part) for part in text.replace(",", " ").split()]
    if len(values) != 3:
        raise KinematicsError(f"expected three numbers, got {text!r}")
    return (values[0], values[1], values[2])


def _parse_optional_float(text: Optional[str]) -> Optional[float]:
    """A number of the URDF, or ``None`` when it is not given."""
    if text is None or not str(text).strip():
        return None
    try:
        return float(text)
    except ValueError:
        return None



def parse_urdf_joints(urdf_path: str) -> Dict[str, UrdfJoint]:
    """Read every joint of a URDF file into a ``{name: UrdfJoint}`` mapping."""
    if not os.path.isfile(urdf_path):
        raise KinematicsError(f"the URDF {urdf_path!r} does not exist")
    try:
        root = ET.parse(urdf_path).getroot()
    except ET.ParseError as exc:
        raise KinematicsError(f"the URDF {urdf_path!r} cannot be parsed: {exc}") from exc

    joints: Dict[str, UrdfJoint] = {}
    for element in root.findall("joint"):
        name = element.get("name")
        if not name:
            continue
        parent_element = element.find("parent")
        child_element = element.find("child")
        if parent_element is None or child_element is None:
            continue
        origin_element = element.find("origin")
        xyz = _parse_vector(
            origin_element.get("xyz") if origin_element is not None else None, (0.0, 0.0, 0.0)
        )
        rpy = _parse_vector(
            origin_element.get("rpy") if origin_element is not None else None, (0.0, 0.0, 0.0)
        )
        axis_element = element.find("axis")
        axis = _parse_vector(
            axis_element.get("xyz") if axis_element is not None else None, (1.0, 0.0, 0.0)
        )
        limit_element = element.find("limit")
        lower = upper = None
        if limit_element is not None:
            lower = _parse_optional_float(limit_element.get("lower"))
            upper = _parse_optional_float(limit_element.get("upper"))
        joints[name] = UrdfJoint(
            name=name,
            type=str(element.get("type") or "fixed"),
            parent=str(parent_element.get("link")),
            child=str(child_element.get("link")),
            xyz=xyz,
            rpy=rpy,
            axis=axis,
            lower=lower,
            upper=upper,
        )
    if not joints:
        raise KinematicsError(f"the URDF {urdf_path!r} holds no joint")
    return joints


def urdf_link_names(joints: Dict[str, UrdfJoint]) -> List[str]:
    """Every link name the URDF mentions (parents and children)."""
    names = set()
    for joint in joints.values():
        names.add(joint.parent)
        names.add(joint.child)
    return sorted(names)


def build_chain(
    joints: Dict[str, UrdfJoint],
    target_frame: str,
    base_frame: Optional[str] = None,
) -> List[UrdfJoint]:
    """The chain from the base to ``target_frame`` (base first).

    Only the joints on the path to the target frame are returned; the fixed
    ones among them do not carry an angle.

    ``base_frame`` selects the frame the chain starts in: the walk upwards
    stops as soon as ``base_frame`` is reached, so the resulting chain - and
    therefore the pose computed from it - is relative to that frame and not
    to the root of the URDF.  ``None`` keeps the legacy behaviour and starts
    at the root of the URDF.
    """
    if base_frame is not None and base_frame == target_frame:
        raise KinematicsError(
            f"the base frame {base_frame!r} and the target frame must differ"
        )

    by_child: Dict[str, UrdfJoint] = {joint.child: joint for joint in joints.values()}

    chain: List[UrdfJoint] = []
    frame = target_frame
    seen = set()
    while frame in by_child:
        joint = by_child[frame]
        if joint.name in seen:
            raise KinematicsError(f"the chain to {target_frame!r} contains a loop")
        seen.add(joint.name)
        chain.append(joint)
        frame = joint.parent
        if base_frame is not None and frame == base_frame:
            break
    chain.reverse()
    if not chain:
        raise KinematicsError(f"no chain leads to {target_frame!r}")
    if base_frame is not None and frame != base_frame:
        links = urdf_link_names(joints)
        if base_frame not in links:
            raise KinematicsError(
                f"the base frame {base_frame!r} is not a link of the URDF "
                f"(known links: {', '.join(links)})"
            )
        raise KinematicsError(
            f"the base frame {base_frame!r} is not on the path from the root of "
            f"the URDF to {target_frame!r}"
        )
    return chain


# ---------------------------------------------------------------------------
# The kinematics
# ---------------------------------------------------------------------------


class So100Kinematics:
    """Forward kinematics and geometric Jacobian of the SO-101 arm.

    Args:
        urdf_path: The calibrated SO-101 URDF.
        target_frame: The frame the end-effector pose refers to.
        joint_names: The actuated joints, in the order of the policy state
            vector.  Their values are interpreted as degrees.
        base_frame: The frame the pose is measured from - the base of the
            arm.  ``"base_link"`` (see :data:`DEFAULT_BASE_FRAME`) expresses
            the pose in the robot base, which is what a ROS message should
            carry; ``None`` falls back to the root of the URDF (the frame
            the LeRobot backend uses).
    """

    def __init__(
        self,
        urdf_path: str = DEFAULT_URDF_PATH,
        target_frame: str = DEFAULT_TARGET_FRAME,
        joint_names: Sequence[str] = ARM_JOINT_NAMES,
        base_frame: Optional[str] = None,
    ) -> None:
        self.urdf_path = urdf_path
        self.target_frame = target_frame
        self.base_frame = base_frame
        self.joint_names = list(joint_names)

        joints = parse_urdf_joints(urdf_path)
        chain = build_chain(joints, target_frame, base_frame)
        # the frame the chain starts in: ``base_frame`` when one was asked
        # for, the root of the URDF otherwise
        self.root_frame = chain[0].parent

        self._chain: List[Tuple[UrdfJoint, int]] = []
        for joint in chain:
            index = self.joint_names.index(joint.name) if joint.name in self.joint_names else -1
            if joint.type in ("revolute", "continuous") and index < 0:
                raise KinematicsError(
                    f"the joint {joint.name!r} is part of the chain but not of {self.joint_names!r}"
                )
            self._chain.append((joint, index))

        # the limits of the actuated joints, in degrees - they are the safety
        # net against a unit mix-up (see :meth:`out_of_range`)
        self._limits_deg: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        for name in self.joint_names:
            joint = joints.get(name)
            if joint is None:
                continue
            lower = math.degrees(joint.lower) if joint.lower is not None else None
            upper = math.degrees(joint.upper) if joint.upper is not None else None
            self._limits_deg[name] = (lower, upper)

    @property
    def joint_limits_deg(self) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """The limits of the actuated joints in degrees (``None`` = unlimited)."""
        return dict(self._limits_deg)

    def out_of_range(
        self, joint_action: Dict[str, float], tolerance_deg: float = 10.0
    ) -> List[str]:
        """The joints whose angle is outside the limit of the URDF.

        The check is a safety net against a unit mix-up (degrees read as
        radians or the other way round): such a value is off by a factor of
        57 and therefore far outside of the limit.  The tolerance keeps the
        calibration offsets of the real URDF from tripping the check.
        """
        violations: List[str] = []
        for name in self.joint_names:
            limits = self._limits_deg.get(name)
            if limits is None:
                continue
            lower, upper = limits
            value = joint_action.get(f"{name}.pos", joint_action.get(name))
            if value is None:
                continue
            value = float(value)
            if lower is not None and value < lower - tolerance_deg:
                violations.append(name)
            elif upper is not None and value > upper + tolerance_deg:
                violations.append(name)
        return violations

    @property
    def pose_frame(self) -> str:
        """The frame the computed end-effector pose is expressed in.

        It is the ``base_frame`` when one was given (``base_link`` for the
        SO-101) and the root of the URDF otherwise.
        """
        return self.base_frame or self.root_frame

    # ------------------------------------------------------------------
    # Forward kinematics
    # ------------------------------------------------------------------
    def _transforms(
        self, angles_rad: np.ndarray
    ) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
        """The end-effector transform and the joint axes/points.

        Returns the 4x4 transform of the end-effector frame and, for every
        actuated joint, the axis direction and the origin point of the joint
        (both in the base frame) - the ingredients of the geometric Jacobian.
        """
        transform = np.eye(4)
        axes: List[Tuple[np.ndarray, np.ndarray]] = []
        for joint, index in self._chain:
            transform = transform @ joint.origin_transform()
            if joint.type in ("revolute", "continuous"):
                axis_world = transform[:3, :3] @ np.asarray(joint.axis, dtype=np.float64)
                axes.append((axis_world, transform[:3, 3].copy()))
                rotation = axis_angle_to_matrix(joint.axis, float(angles_rad[index]))
                transform = transform.copy()
                transform[:3, :3] = transform[:3, :3] @ rotation
        return transform, axes

    def end_effector(self, joint_action: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        """The end-effector pose of a joint command.

        Args:
            joint_action: ``{"shoulder_pan.pos": value, ...}`` (degrees).

        Returns:
            The position (metres) and the rotation matrix, both expressed in
            the base frame - see :attr:`pose_frame`.
        """
        angles = self._angles(joint_action)
        transform, _axes = self._transforms(angles)
        return transform[:3, 3].copy(), transform[:3, :3].copy()

    def forward_kinematics(self, joint_action: Dict[str, float]) -> Dict[str, float]:
        """The end-effector pose in the ``ee.*`` spelling of the LeRobot backend.

        The keys are ``ee.x`` / ``ee.y`` / ``ee.z`` (metres) and ``ee.wx`` /
        ``ee.wy`` / ``ee.wz`` (rotation vector, radians) - the same numbers
        :mod:`kinematics_utils` returns, so the two backends are
        interchangeable.

        The numbers are expressed in the base frame; with the default
        ``base_frame=None`` that is the root of the URDF, which is what the
        LeRobot backend uses.
        """
        position, rotation = self.end_effector(joint_action)
        rotvec = matrix_to_rotvec(rotation)
        return {
            "ee.x": float(position[0]),
            "ee.y": float(position[1]),
            "ee.z": float(position[2]),
            "ee.wx": float(rotvec[0]),
            "ee.wy": float(rotvec[1]),
            "ee.wz": float(rotvec[2]),
        }

    # ------------------------------------------------------------------
    # Jacobian
    # ------------------------------------------------------------------
    def jacobian(self, joint_action: Dict[str, float]) -> np.ndarray:
        """The geometric Jacobian of the end-effector frame.

        The matrix has six rows (the three linear ones first, the three
        angular ones after) and one column per actuated joint.  It is
        expressed **per degree**, so that ``dq`` comes out in the same units
        as the joint commands.
        """
        angles = self._angles(joint_action)
        transform, axes = self._transforms(angles)
        end = transform[:3, 3]

        jacobian = np.zeros((6, len(self.joint_names)), dtype=np.float64)
        for column, (axis_world, origin) in enumerate(axes):
            jacobian[:3, column] = np.cross(axis_world, end - origin)
            jacobian[3:, column] = axis_world
        # radians -> degrees: the joint commands are degrees, so the
        # derivative has to be taken with respect to a degree as well
        # (q_rad = q_deg * pi / 180  =>  d/dq_deg = d/dq_rad * pi / 180).
        return jacobian * (math.pi / 180.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _angles(self, joint_action: Dict[str, float]) -> np.ndarray:
        """The joint command (degrees) as radians, in the chain order."""
        angles = np.zeros(len(self.joint_names), dtype=np.float64)
        for index, name in enumerate(self.joint_names):
            value = joint_action.get(f"{name}.pos", joint_action.get(name))
            if value is None:
                raise KinematicsError(f"the joint {name!r} is missing from the command")
            angles[index] = math.radians(float(value))
        return angles

    def joint_position(self, joint_action: Dict[str, float]) -> np.ndarray:
        """The joint command as a plain array (degrees)."""
        return np.array(
            [float(joint_action[f"{name}.pos"]) for name in self.joint_names],
            dtype=np.float64,
        )

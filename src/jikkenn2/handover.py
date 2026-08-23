"""Pose geometry for the handover: where the hand goes, in what order.

Planning lives in cuRobo; this module only decides the poses cuRobo is asked to
reach, which is pure arithmetic and therefore testable without a GPU.

The sequence is: above the tool, down onto it, close, straight up, then out to
the person with the safe part leading.
"""

from __future__ import annotations

import numpy as np

from jikkenn2 import ground_truth as gt
from jikkenn2.reachability import PANDA_FINGERTIP_DEPTH_M
from jikkenn2.scene_spec import SceneSpec


#: Column of the hand rotation the fingers travel along.  The Franka's finger
#: joints are prismatic along the hand's y axis.  For the Phase 0 proxy tool the
#: head is square in cross-section (45 x 45 mm) and both horizontal spans fit
#: the 80 mm opening, so a trial succeeds either way; a real tool in Phase 2
#: will need this confirmed against the gripper description.
HAND_CLOSING_COLUMN = 1

DEFAULT_APPROACH_OFFSET_M = 0.12
DEFAULT_LIFT_M = 0.15


def _hand_rotation(approach, closing) -> np.ndarray:
    """Hand rotation with +z along ``approach`` and +y along ``closing``."""
    z_axis = np.asarray(approach, dtype=np.float64)
    z_axis = z_axis / np.linalg.norm(z_axis)
    y_axis = np.asarray(closing, dtype=np.float64)
    y_axis = y_axis - z_axis * float(np.dot(y_axis, z_axis))
    norm = float(np.linalg.norm(y_axis))
    if norm < 1e-9:
        raise ValueError("the closing direction is parallel to the approach")
    y_axis = y_axis / norm
    x_axis = np.cross(y_axis, z_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def _pose(rotation: np.ndarray, position) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = np.asarray(position, dtype=np.float64)
    return pose


def grasp_hand_pose(
    tool_pose: np.ndarray,
    scene: SceneSpec,
    *,
    fingertip_depth_m: float = PANDA_FINGERTIP_DEPTH_M,
) -> np.ndarray:
    """Where ``panda_hand`` must be to close on the part that gets grasped.

    Approach is straight down, and the fingers close across the tool rather
    than along it, so the grasp does not depend on the tool's length.
    """
    transform = np.asarray(tool_pose, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"tool_pose must be 4x4, got {transform.shape}")
    part = scene.part(scene.grasp_part_name)
    target = gt.part_pose(transform, part)

    approach = np.array([0.0, 0.0, -1.0])
    # Across the tool: the part's own lateral axis, flattened into the
    # horizontal plane so the wrist stays level.
    lateral = transform[:3, :3] @ np.array([0.0, 1.0, 0.0])
    lateral[2] = 0.0
    if float(np.linalg.norm(lateral)) < 1e-6:
        # The tool is standing on end; any horizontal direction will do.
        lateral = transform[:3, :3] @ np.array([1.0, 0.0, 0.0])
        lateral[2] = 0.0
    if float(np.linalg.norm(lateral)) < 1e-6:
        lateral = np.array([0.0, 1.0, 0.0])

    rotation = _hand_rotation(approach, lateral)
    position = np.asarray(target.center_m) - approach * float(fingertip_depth_m)
    return _pose(rotation, position)


def pregrasp_pose(
    grasp_pose: np.ndarray, *, approach_offset_m: float = DEFAULT_APPROACH_OFFSET_M
) -> np.ndarray:
    """Back off along the hand's own approach axis, so the descent is straight."""
    if approach_offset_m <= 0.0:
        raise ValueError("approach_offset_m must be positive")
    pose = np.asarray(grasp_pose, dtype=np.float64).copy()
    pose[:3, 3] -= pose[:3, 2] * float(approach_offset_m)
    return pose


def lift_pose(grasp_pose: np.ndarray, *, lift_m: float = DEFAULT_LIFT_M) -> np.ndarray:
    """Straight up in the world, so the tool clears the table before travelling."""
    if lift_m <= 0.0:
        raise ValueError("lift_m must be positive")
    pose = np.asarray(grasp_pose, dtype=np.float64).copy()
    pose[2, 3] += float(lift_m)
    return pose


def handover_tool_pose(scene: SceneSpec) -> np.ndarray:
    """Where the tool must end up: at the handover point, safe part leading.

    The tool stays level and is yawed so its safe axis points at the person.
    """
    handover = np.asarray(scene.handover_position_m, dtype=np.float64)
    to_human = np.asarray(scene.human_point_m, dtype=np.float64) - handover
    to_human[2] = 0.0
    norm = float(np.linalg.norm(to_human))
    if norm < 1e-9:
        raise ValueError("the handover point is directly below the person")
    to_human = to_human / norm

    safe_axis_local = np.asarray(scene.part(scene.safe_part_name).axis_local, dtype=np.float64)
    if abs(safe_axis_local[2]) > 1e-9 or float(np.linalg.norm(safe_axis_local[:2])) < 1e-9:
        raise ValueError("the safe part axis must be horizontal in the tool frame")
    # Yaw that carries the safe axis onto the direction of the person.
    yaw = np.arctan2(to_human[1], to_human[0]) - np.arctan2(
        safe_axis_local[1], safe_axis_local[0]
    )
    rotation = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return _pose(rotation, handover)


def hand_pose_for_tool_pose(
    target_tool_pose: np.ndarray,
    grasp_tool_pose: np.ndarray,
    grasp_hand_pose_: np.ndarray,
) -> np.ndarray:
    """Move the hand so the *tool* lands on a target pose.

    Once the tool is held, the hand and the tool are rigidly linked, so the
    hand's target follows from the tool's target and the grasp taken.
    """
    target = np.asarray(target_tool_pose, dtype=np.float64)
    at_grasp_tool = np.asarray(grasp_tool_pose, dtype=np.float64)
    at_grasp_hand = np.asarray(grasp_hand_pose_, dtype=np.float64)
    T_tool_hand = np.linalg.inv(at_grasp_tool) @ at_grasp_hand
    return target @ T_tool_hand


def plan_waypoints(
    tool_pose: np.ndarray,
    scene: SceneSpec,
    *,
    approach_offset_m: float = DEFAULT_APPROACH_OFFSET_M,
    lift_m: float = DEFAULT_LIFT_M,
) -> dict[str, np.ndarray]:
    """The whole ordered sequence of ``panda_hand`` poses for one trial."""
    grasp = grasp_hand_pose(tool_pose, scene)
    pregrasp = pregrasp_pose(grasp, approach_offset_m=approach_offset_m)
    lift = lift_pose(grasp, lift_m=lift_m)
    target_tool = handover_tool_pose(scene)
    handover = hand_pose_for_tool_pose(target_tool, np.asarray(tool_pose), grasp)
    return {
        "pregrasp": pregrasp,
        "grasp": grasp,
        "lift": lift,
        "handover": handover,
        "target_tool": target_tool,
    }


def describe_waypoints(waypoints: dict[str, np.ndarray]) -> dict:
    """JSON-friendly summary, for the plan report."""
    from jikkenn2.geometry import quaternion_wxyz_from_rotation_matrix

    described = {}
    for name, pose in waypoints.items():
        described[name] = {
            "position_m": [round(float(v), 5) for v in pose[:3, 3]],
            "orientation_wxyz": [
                round(float(v), 6)
                for v in quaternion_wxyz_from_rotation_matrix(pose[:3, :3])
            ],
        }
    return described

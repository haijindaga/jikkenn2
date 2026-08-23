"""Ground-truth geometry for evaluation only.

Everything here reads the simulator's true state.  **No perception module may
import this.**  Phase 0 uses it in place of perception on purpose; from Phase 2
onward it is the yardstick that the perception result is measured against, and
mixing the two would make every later comparison meaningless.

The claim being measured is "the robot holds the dangerous part and presents
the safe part to the person", so the functions here answer exactly that: which
part is being held, and where each part points relative to the person.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jikkenn2.geometry import matrix_from_pose
from jikkenn2.scene_spec import SceneSpec, ToolPart


@dataclass(frozen=True)
class PartPose:
    """Where one labelled part of the tool is, in world coordinates."""

    name: str
    center_m: tuple[float, float, float]
    axis: tuple[float, float, float]

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "center_m": [float(v) for v in self.center_m],
            "axis": [float(v) for v in self.axis],
        }


def tool_pose_matrix(position_m, orientation_wxyz) -> np.ndarray:
    """Build ``T_world_tool`` from a recorded pose."""
    return matrix_from_pose(
        np.asarray(position_m, dtype=np.float64),
        np.asarray(orientation_wxyz, dtype=np.float64),
    )


def part_pose(tool_pose: np.ndarray, part: ToolPart) -> PartPose:
    """Place one part in the world, given the tool's pose."""
    transform = np.asarray(tool_pose, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"tool_pose must be 4x4, got {transform.shape}")
    center = transform[:3, :3] @ np.asarray(part.center_m, dtype=np.float64)
    center = center + transform[:3, 3]
    axis = transform[:3, :3] @ np.asarray(part.axis_local, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        raise ValueError(f"part {part.name} has a degenerate axis")
    return PartPose(
        name=part.name,
        center_m=tuple(float(v) for v in center),
        axis=tuple(float(v) for v in axis / norm),
    )


def part_poses(tool_pose: np.ndarray, scene: SceneSpec) -> dict[str, PartPose]:
    return {part.name: part_pose(tool_pose, part) for part in scene.tool_parts}


def point_in_part(tool_pose: np.ndarray, part: ToolPart, point_world_m) -> bool:
    """Is a world point inside a part's box?

    The point is taken back into the tool's own frame, so this stays correct
    however the tool is rotated.
    """
    transform = np.asarray(tool_pose, dtype=np.float64)
    point = np.asarray(point_world_m, dtype=np.float64)
    if point.shape != (3,):
        raise ValueError(f"point_world_m must have shape (3,), got {point.shape}")
    local = transform[:3, :3].T @ (point - transform[:3, 3])
    minimum, maximum = part.aabb_local_m()
    return bool(
        np.all(local >= np.asarray(minimum) - 1e-9)
        and np.all(local <= np.asarray(maximum) + 1e-9)
    )


def part_containing(
    tool_pose: np.ndarray, scene: SceneSpec, point_world_m
) -> str | None:
    """Name the part a world point falls inside, or None."""
    for part in scene.tool_parts:
        if point_in_part(tool_pose, part, point_world_m):
            return part.name
    return None


def points_in_tool_mask(
    tool_pose: np.ndarray,
    scene: SceneSpec,
    points_world: np.ndarray,
    *,
    margin_m: float = 0.01,
) -> np.ndarray:
    """Boolean image marking pixels whose 3-D point lies on the tool.

    Phase 1 has no perception yet, so the tool is removed from the depth using
    its true pose.  Working from the pixel-aligned point map rather than a
    projected bounding box keeps the mask exact whatever the tool's rotation.
    """
    transform = np.asarray(tool_pose, dtype=np.float64)
    points = np.asarray(points_world, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"tool_pose must be 4x4, got {transform.shape}")
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError(f"points_world must be HxWx3, got {points.shape}")
    if margin_m < 0.0:
        raise ValueError("margin_m must not be negative")

    flat = points.reshape(-1, 3)
    finite = np.all(np.isfinite(flat), axis=1)
    local = np.full_like(flat, np.nan)
    local[finite] = (flat[finite] - transform[:3, 3]) @ transform[:3, :3]

    mask = np.zeros(flat.shape[0], dtype=bool)
    for part in scene.tool_parts:
        minimum, maximum = part.aabb_local_m()
        inside = finite & np.all(
            (local >= np.asarray(minimum) - margin_m)
            & (local <= np.asarray(maximum) + margin_m),
            axis=1,
        )
        mask |= inside
    return mask.reshape(points.shape[:2])


def angle_between_deg(first, second) -> float:
    """Angle between two directions, in degrees."""
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    norms = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
    if norms <= 1e-12:
        raise ValueError("cannot take the angle of a zero-length direction")
    cosine = float(np.clip(np.dot(a, b) / norms, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def handover_orientation(tool_pose: np.ndarray, scene: SceneSpec) -> dict:
    """How the tool is presented to the person.

    ``safe_axis_to_human_deg`` small means the handle points at them;
    ``danger_axis_to_human_deg`` large means the blade points away.
    """
    poses = part_poses(tool_pose, scene)
    safe = poses[scene.safe_part_name]
    danger = poses[scene.danger_part_name]
    tool_center = np.asarray(tool_pose, dtype=np.float64)[:3, 3]
    to_human = np.asarray(scene.human_point_m, dtype=np.float64) - tool_center
    distance = float(np.linalg.norm(to_human))
    if distance <= 1e-9:
        raise ValueError("the tool is at the person's position")
    return {
        "safe_part": scene.safe_part_name,
        "danger_part": scene.danger_part_name,
        "distance_to_human_m": round(distance, 4),
        "safe_axis_to_human_deg": round(angle_between_deg(safe.axis, to_human), 3),
        "danger_axis_to_human_deg": round(angle_between_deg(danger.axis, to_human), 3),
        "safe_part_center_m": [round(float(v), 4) for v in safe.center_m],
        "danger_part_center_m": [round(float(v), 4) for v in danger.center_m],
    }


def rotation_between_deg(first: np.ndarray, second: np.ndarray) -> float:
    """Angle of the rotation carrying one pose's orientation onto another's."""
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    relative = a[:3, :3].T @ b[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def pose_in_frame(frame_pose: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Express ``pose`` in the frame of ``frame_pose``."""
    return np.linalg.inv(np.asarray(frame_pose, dtype=np.float64)) @ np.asarray(
        pose, dtype=np.float64
    )


def rotation_in_grip_deg(
    hand_then: dict, tool_then: dict, hand_now: dict, tool_now: dict
) -> float:
    """How far the tool turned *inside the hand* between two moments.

    Measuring in world coordinates would count the rotation the arm applies on
    purpose -- the handover deliberately yaws the tool about 94 degrees to
    point the handle at the person -- and report a firm grasp as a slip.
    """
    def matrix(pose: dict) -> np.ndarray:
        return tool_pose_matrix(pose["position_m"], pose["orientation_wxyz"])

    return rotation_between_deg(
        pose_in_frame(matrix(hand_then), matrix(tool_then)),
        pose_in_frame(matrix(hand_now), matrix(tool_now)),
    )


def pose_difference(before: dict, after: dict) -> dict:
    """Translation and rotation between two recorded poses of one object."""
    first = tool_pose_matrix(before["position_m"], before["orientation_wxyz"])
    second = tool_pose_matrix(after["position_m"], after["orientation_wxyz"])
    return {
        "prim_path": after.get("prim_path", before.get("prim_path")),
        "translation_m": round(float(np.linalg.norm(second[:3, 3] - first[:3, 3])), 5),
        "rotation_deg": round(rotation_between_deg(first, second), 3),
    }


def settle_report(
    authored: list[dict],
    settled: list[dict],
    *,
    translation_warning_m: float = 0.02,
    rotation_warning_deg: float = 10.0,
) -> dict:
    """Compare hand-authored placements with where physics left them.

    Large drift means the object was placed floating or intersecting something,
    which is a placement problem rather than a simulation problem.
    """
    by_path = {entry["prim_path"]: entry for entry in authored}
    differences = []
    for entry in settled:
        path = entry["prim_path"]
        if path not in by_path:
            raise KeyError(f"settled pose has no authored counterpart: {path}")
        differences.append(pose_difference(by_path[path], entry))

    drifted = [
        difference["prim_path"]
        for difference in differences
        if difference["translation_m"] > translation_warning_m
        or difference["rotation_deg"] > rotation_warning_deg
    ]
    return {
        "thresholds": {
            "translation_warning_m": translation_warning_m,
            "rotation_warning_deg": rotation_warning_deg,
        },
        "differences": differences,
        "objects_that_moved": drifted,
        "status": "success" if not drifted else "settle_warning",
    }

"""The six criteria, applied to what a trial actually did.

Kept apart from the script that executes a trial: the thing that measures and
the thing that decides should be separable, so a criterion can change without
re-running the simulator.  Everything here is dict in, dict out, so the whole
rubric is tested against synthetic trials.

Criteria 1, 4 and 5 are the claim being made -- the robot holds the dangerous
part and presents the safe part to the person.  The rest are the conditions
under which that claim means anything.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from jikkenn2 import ground_truth as gt
from jikkenn2.reachability import PANDA_FINGERTIP_DEPTH_M
from jikkenn2.scene_spec import SceneSpec


@dataclass(frozen=True)
class Thresholds:
    """Where each criterion draws its line."""

    safe_axis_to_human_deg: float = 30.0
    danger_axis_to_human_deg: float = 90.0
    handover_position_error_m: float = 0.02
    handover_orientation_error_deg: float = 10.0
    minimum_carry_height_m: float = 0.05
    minimum_grip_width_m: float = 0.005
    obstacle_translation_m: float = 0.005
    obstacle_rotation_deg: float = 2.0
    joint_tracking_error_rad: float = 0.05


def _matrix(pose: dict) -> np.ndarray:
    return gt.tool_pose_matrix(pose["position_m"], pose["orientation_wxyz"])


def fingertip_world_m(hand_pose: dict, depth_m: float = PANDA_FINGERTIP_DEPTH_M):
    """Where the fingers close, given the hand's pose."""
    hand = _matrix(hand_pose)
    return hand[:3, 3] + hand[:3, 2] * float(depth_m)


def _sample(execution: dict, label: str) -> dict | None:
    for entry in execution.get("samples", []):
        if entry.get("label") == label:
            return entry
    return None


def _criterion(name: str, passed: bool, measured, threshold, detail: str) -> dict:
    return {
        "criterion": name,
        "passed": bool(passed),
        "measured": measured,
        "threshold": threshold,
        "detail": detail,
    }


def grasped_part(scene: SceneSpec, execution: dict) -> tuple[str | None, str]:
    """Which part the fingers actually closed on, from the execution.

    Read at the moment of closing rather than from the plan, because the plan
    says what was intended and this says what happened.
    """
    closed = _sample(execution, "after_close")
    if closed is None:
        return None, "the execution recorded no after_close sample"
    hand = closed.get("hand")
    if hand is None:
        return None, "the execution recorded no hand pose; cannot locate the fingertips"
    fingertip = fingertip_world_m(hand)
    tool_pose = _matrix(closed["tool"])
    return gt.part_containing(tool_pose, scene, fingertip), "measured at the close"


def score(
    scene: SceneSpec,
    execution: dict,
    *,
    target_tool_pose: np.ndarray,
    thresholds: Thresholds | None = None,
) -> dict:
    """Apply all six criteria to one executed trial."""
    limits = thresholds or Thresholds()
    criteria = []

    # 1 -- the robot grasped the part it was supposed to grasp.
    part, how = grasped_part(scene, execution)
    criteria.append(
        _criterion(
            "grasped_the_intended_part",
            part == scene.grasp_part_name,
            part,
            scene.grasp_part_name,
            how,
        )
    )

    # 2 -- and did not drop it.
    carry = execution.get("carry", {})
    final_height = float(execution["final_tool"]["position_m"][2])
    held = (
        carry.get("grip_lost_at") is None
        and final_height > scene.table_top_z_m + limits.minimum_carry_height_m
        and float(execution["gripper"]["final_width_m"]) > limits.minimum_grip_width_m
    )
    criteria.append(
        _criterion(
            "still_holding_the_tool",
            held,
            {
                "grip_lost_at": carry.get("grip_lost_at"),
                "final_tool_height_m": round(final_height, 5),
                "final_grip_width_m": execution["gripper"]["final_width_m"],
            },
            {
                "grip_lost_at": None,
                "minimum_height_m": round(
                    scene.table_top_z_m + limits.minimum_carry_height_m, 5
                ),
                "minimum_width_m": limits.minimum_grip_width_m,
            },
            "the tool is still off the table and still between the fingers",
        )
    )

    # 3 -- nothing was knocked, and nothing blocked the arm.
    moved = [
        entry["prim_path"]
        for entry in execution.get("obstacle_motion", [])
        if entry["translation_m"] > limits.obstacle_translation_m
        or entry["rotation_deg"] > limits.obstacle_rotation_deg
    ]
    tracking = float(execution["robot"]["max_joint_tracking_error_rad"])
    criteria.append(
        _criterion(
            "no_collision",
            not moved and tracking <= limits.joint_tracking_error_rad,
            {"obstacles_that_moved": moved, "max_joint_tracking_error_rad": tracking},
            {
                "obstacles_that_moved": [],
                "max_joint_tracking_error_rad": limits.joint_tracking_error_rad,
            },
            "inferred from obstacle displacement and joint tracking; contact "
            "reporting is not enabled, so this is evidence rather than proof",
        )
    )

    # 4 and 5 -- the claim itself.
    orientation = execution["final_handover_orientation"]
    safe_angle = float(orientation["safe_axis_to_human_deg"])
    danger_angle = float(orientation["danger_axis_to_human_deg"])
    criteria.append(
        _criterion(
            "safe_part_faces_the_person",
            safe_angle < limits.safe_axis_to_human_deg,
            round(safe_angle, 3),
            f"< {limits.safe_axis_to_human_deg}",
            f"the {orientation['safe_part']} points at the person",
        )
    )
    criteria.append(
        _criterion(
            "danger_part_faces_away",
            danger_angle > limits.danger_axis_to_human_deg,
            round(danger_angle, 3),
            f"> {limits.danger_axis_to_human_deg}",
            f"the {orientation['danger_part']} points away from the person",
        )
    )

    # 6 -- and it got there.
    final_pose = _matrix(execution["final_tool"])
    target = np.asarray(target_tool_pose, dtype=np.float64)
    position_error = float(np.linalg.norm(final_pose[:3, 3] - target[:3, 3]))
    orientation_error = gt.rotation_between_deg(target, final_pose)
    criteria.append(
        _criterion(
            "reached_the_handover_pose",
            position_error < limits.handover_position_error_m
            and orientation_error < limits.handover_orientation_error_deg,
            {
                "position_error_m": round(position_error, 5),
                "orientation_error_deg": round(orientation_error, 3),
            },
            {
                "position_error_m": limits.handover_position_error_m,
                "orientation_error_deg": limits.handover_orientation_error_deg,
            },
            "the tool ended where the plan aimed it",
        )
    )

    passed = [entry["criterion"] for entry in criteria if entry["passed"]]
    failed = [entry["criterion"] for entry in criteria if not entry["passed"]]
    return {
        "status": "success" if not failed else "failed_criteria",
        "trial_passed": not failed,
        "passed_count": len(passed),
        "total_count": len(criteria),
        "failed": failed,
        "criteria": criteria,
        "thresholds": asdict(limits),
        "claim": (
            "the robot holds the dangerous part and presents the safe part to "
            "the person; criteria 1, 4 and 5 are the claim, the rest are the "
            "conditions under which it means anything"
        ),
    }


def summarize(scores: list[dict]) -> dict:
    """Aggregate many trials into a success rate, per criterion as well."""
    if not scores:
        return {"trials": 0}
    names = [entry["criterion"] for entry in scores[0]["criteria"]]
    per_criterion = {
        name: sum(
            1
            for result in scores
            for entry in result["criteria"]
            if entry["criterion"] == name and entry["passed"]
        )
        for name in names
    }
    passed = sum(1 for result in scores if result["trial_passed"])
    return {
        "trials": len(scores),
        "passed": passed,
        "success_rate": round(passed / len(scores), 4),
        "passed_per_criterion": per_criterion,
        "success_rate_per_criterion": {
            name: round(count / len(scores), 4) for name, count in per_criterion.items()
        },
    }

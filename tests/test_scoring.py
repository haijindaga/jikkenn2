"""Tests for the six criteria.

Built around one passing trial, then broken one way at a time: each test
asserts that exactly the criterion under test fails.  A rubric that cannot
fail, or that fails everything at once, is not measuring anything.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from jikkenn2 import ground_truth as gt
from jikkenn2 import handover as hv
from jikkenn2 import scoring
from jikkenn2.reachability import PANDA_FINGERTIP_DEPTH_M
from jikkenn2.scene_spec import DEFAULT_SCENE


def pose_dict(matrix) -> dict:
    from jikkenn2.geometry import quaternion_wxyz_from_rotation_matrix

    return {
        "position_m": [float(v) for v in matrix[:3, 3]],
        "orientation_wxyz": [
            float(v) for v in quaternion_wxyz_from_rotation_matrix(matrix[:3, :3])
        ],
    }


@pytest.fixture
def target_tool_pose():
    return hv.handover_tool_pose(DEFAULT_SCENE)


@pytest.fixture
def passing_execution(target_tool_pose):
    """A trial that satisfies every criterion, built from the real geometry."""
    resting = gt.tool_pose_matrix(
        (0.45, 0.15, DEFAULT_SCENE.table_top_z_m + 0.0225), (1.0, 0.0, 0.0, 0.0)
    )
    grasp_hand = hv.grasp_hand_pose(resting, DEFAULT_SCENE)
    hand_at_handover = hv.hand_pose_for_tool_pose(target_tool_pose, resting, grasp_hand)

    return {
        "robot": {"max_joint_tracking_error_rad": 0.025},
        "gripper": {"final_width_m": 0.045},
        "samples": [
            {"label": "after_close", "tool": pose_dict(resting), "hand": pose_dict(grasp_hand)},
            {
                "label": "after_hold",
                "tool": pose_dict(target_tool_pose),
                "hand": pose_dict(hand_at_handover),
            },
        ],
        "obstacle_motion": [
            {"prim_path": "/World/Obstacles/obstacle_a", "translation_m": 0.0, "rotation_deg": 0.0}
        ],
        "carry": {"grip_lost_at": None},
        "final_tool": pose_dict(target_tool_pose),
        "final_handover_orientation": gt.handover_orientation(
            target_tool_pose, DEFAULT_SCENE
        ),
    }


def run(execution, target):
    return scoring.score(DEFAULT_SCENE, execution, target_tool_pose=target)


def test_a_good_trial_passes_every_criterion(passing_execution, target_tool_pose):
    result = run(passing_execution, target_tool_pose)
    assert result["failed"] == [], result["failed"]
    assert result["trial_passed"] is True
    assert result["passed_count"] == result["total_count"] == 6


def test_the_criteria_are_the_six_in_the_design(passing_execution, target_tool_pose):
    names = [entry["criterion"] for entry in run(passing_execution, target_tool_pose)["criteria"]]
    assert names == [
        "grasped_the_intended_part",
        "still_holding_the_tool",
        "no_collision",
        "safe_part_faces_the_person",
        "danger_part_faces_away",
        "reached_the_handover_pose",
    ]


def test_grasping_the_handle_fails_only_that_criterion(passing_execution, target_tool_pose):
    """Move the hand onto the handle: the wrong part is held."""
    broken = copy.deepcopy(passing_execution)
    closed = broken["samples"][0]
    tool = gt.tool_pose_matrix(closed["tool"]["position_m"], closed["tool"]["orientation_wxyz"])
    handle_centre = gt.part_pose(tool, DEFAULT_SCENE.part("handle")).center_m
    hand = gt.tool_pose_matrix(
        closed["hand"]["position_m"], closed["hand"]["orientation_wxyz"]
    )
    hand[:3, 3] = np.asarray(handle_centre) - hand[:3, 2] * PANDA_FINGERTIP_DEPTH_M
    closed["hand"] = pose_dict(hand)

    result = run(broken, target_tool_pose)
    assert result["failed"] == ["grasped_the_intended_part"]
    assert result["criteria"][0]["measured"] == "handle"


def test_a_dropped_tool_fails_only_the_holding_criterion(passing_execution, target_tool_pose):
    broken = copy.deepcopy(passing_execution)
    broken["carry"]["grip_lost_at"] = {"step": 210, "leg": "lift"}
    result = run(broken, target_tool_pose)
    assert result["failed"] == ["still_holding_the_tool"]


def test_a_tool_left_on_the_table_fails_the_holding_criterion(
    passing_execution, target_tool_pose
):
    broken = copy.deepcopy(passing_execution)
    broken["final_tool"]["position_m"][2] = DEFAULT_SCENE.table_top_z_m + 0.01
    result = run(broken, target_tool_pose)
    assert "still_holding_the_tool" in result["failed"]


def test_empty_fingers_fail_the_holding_criterion(passing_execution, target_tool_pose):
    broken = copy.deepcopy(passing_execution)
    broken["gripper"]["final_width_m"] = 0.0
    assert "still_holding_the_tool" in run(broken, target_tool_pose)["failed"]


def test_a_knocked_obstacle_fails_only_the_collision_criterion(
    passing_execution, target_tool_pose
):
    broken = copy.deepcopy(passing_execution)
    broken["obstacle_motion"][0]["translation_m"] = 0.08
    result = run(broken, target_tool_pose)
    assert result["failed"] == ["no_collision"]
    assert result["criteria"][2]["measured"]["obstacles_that_moved"] == [
        "/World/Obstacles/obstacle_a"
    ]


def test_a_rotated_obstacle_also_counts_as_a_collision(passing_execution, target_tool_pose):
    broken = copy.deepcopy(passing_execution)
    broken["obstacle_motion"][0]["rotation_deg"] = 15.0
    assert "no_collision" in run(broken, target_tool_pose)["failed"]


def test_a_blocked_arm_fails_the_collision_criterion(passing_execution, target_tool_pose):
    broken = copy.deepcopy(passing_execution)
    broken["robot"]["max_joint_tracking_error_rad"] = 0.4
    assert "no_collision" in run(broken, target_tool_pose)["failed"]


def test_presenting_the_blade_fails_both_orientation_criteria(
    passing_execution, target_tool_pose
):
    """Turn the tool around: the person is offered the head."""
    turned = target_tool_pose.copy()
    turned[:3, :3] = turned[:3, :3] @ np.array(
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    broken = copy.deepcopy(passing_execution)
    broken["final_tool"] = pose_dict(turned)
    broken["final_handover_orientation"] = gt.handover_orientation(turned, DEFAULT_SCENE)

    result = run(broken, target_tool_pose)
    assert "safe_part_faces_the_person" in result["failed"]
    assert "danger_part_faces_away" in result["failed"]
    assert "grasped_the_intended_part" not in result["failed"]


def test_stopping_short_fails_only_the_reached_criterion(
    passing_execution, target_tool_pose
):
    broken = copy.deepcopy(passing_execution)
    broken["final_tool"]["position_m"][1] += 0.10
    result = run(broken, target_tool_pose)
    assert result["failed"] == ["reached_the_handover_pose"]
    assert result["criteria"][5]["measured"]["position_error_m"] == pytest.approx(0.10)


def test_a_centimetre_of_error_is_still_a_pass(passing_execution, target_tool_pose):
    nearly = copy.deepcopy(passing_execution)
    nearly["final_tool"]["position_m"][0] += 0.012
    assert run(nearly, target_tool_pose)["trial_passed"] is True


def test_an_execution_without_hand_poses_cannot_judge_the_grasp(
    passing_execution, target_tool_pose
):
    broken = copy.deepcopy(passing_execution)
    for sample in broken["samples"]:
        sample["hand"] = None
    result = run(broken, target_tool_pose)
    assert result["failed"] == ["grasped_the_intended_part"]
    assert "cannot locate the fingertips" in result["criteria"][0]["detail"]


def test_thresholds_can_be_tightened(passing_execution, target_tool_pose):
    """A criterion that passes at the default limit fails at a stricter one."""
    default = run(passing_execution, target_tool_pose)
    assert "no_collision" not in default["failed"]

    # The trial's tracking error is 0.025 rad; demand better than that.
    strict = scoring.Thresholds(joint_tracking_error_rad=0.01)
    result = scoring.score(
        DEFAULT_SCENE, passing_execution, target_tool_pose=target_tool_pose, thresholds=strict
    )
    assert result["failed"] == ["no_collision"]
    assert result["thresholds"]["joint_tracking_error_rad"] == 0.01


def test_the_planner_aims_the_handover_exactly(passing_execution, target_tool_pose):
    """The orientation criteria have room to spare, by construction."""
    result = run(passing_execution, target_tool_pose)
    safe = next(
        entry for entry in result["criteria"] if entry["criterion"] == "safe_part_faces_the_person"
    )
    assert safe["measured"] == pytest.approx(0.0, abs=1e-6)


def test_summary_of_no_trials():
    assert scoring.summarize([]) == {"trials": 0}


def test_summary_counts_trials_and_criteria(passing_execution, target_tool_pose):
    good = run(passing_execution, target_tool_pose)
    dropped = copy.deepcopy(passing_execution)
    dropped["carry"]["grip_lost_at"] = {"step": 1, "leg": "lift"}
    bad = run(dropped, target_tool_pose)

    summary = scoring.summarize([good, bad, good])
    assert summary["trials"] == 3
    assert summary["passed"] == 2
    assert summary["success_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert summary["passed_per_criterion"]["still_holding_the_tool"] == 2
    assert summary["passed_per_criterion"]["grasped_the_intended_part"] == 3

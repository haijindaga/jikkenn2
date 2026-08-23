"""Tests for choosing which proposed grasp to use.

The decision under test is the claim itself: out of a pile of poses a grasp
proposer thought were fine, take the ones that close on the dangerous part.
"""

from __future__ import annotations

import numpy as np
import pytest

from jikkenn2 import grasp_selection as gs
from jikkenn2 import ground_truth as gt
from jikkenn2 import handover as hv
from jikkenn2.reachability import PANDA_FINGERTIP_DEPTH_M
from jikkenn2.scene_spec import DEFAULT_SCENE


@pytest.fixture
def tool_pose():
    return gt.tool_pose_matrix(
        (0.45, 0.15, DEFAULT_SCENE.table_top_z_m + 0.0225), (1.0, 0.0, 0.0, 0.0)
    )


def grasp_at(point, *, approach=(0.0, 0.0, -1.0), closing=(0.0, 1.0, 0.0)):
    """A hand pose whose fingers close on ``point`` from ``approach``."""
    rotation = hv._hand_rotation(approach, closing)
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = np.asarray(point, dtype=np.float64) - rotation[:, 2] * (
        PANDA_FINGERTIP_DEPTH_M
    )
    return pose


def part_centre(tool_pose, name):
    return np.asarray(gt.part_pose(tool_pose, DEFAULT_SCENE.part(name)).center_m)


def test_a_grasp_on_the_head_is_kept(tool_pose):
    candidates = gs.candidates_from_arrays(
        [grasp_at(part_centre(tool_pose, "head"))], [0.9]
    )
    ranking = gs.rank_candidates(candidates, tool_pose, DEFAULT_SCENE)
    assert ranking["counts"]["on_the_intended_part"] == 1
    assert ranking["kept"][0]["part"] == "head"


def test_a_grasp_on_the_handle_is_rejected_with_a_reason(tool_pose):
    candidates = gs.candidates_from_arrays(
        [grasp_at(part_centre(tool_pose, "handle"))], [0.99]
    )
    ranking = gs.rank_candidates(candidates, tool_pose, DEFAULT_SCENE)
    assert ranking["counts"]["on_the_intended_part"] == 0
    assert "'handle'" in ranking["rejected"][0]["rejected_because"]


def test_a_high_score_does_not_rescue_the_wrong_part(tool_pose):
    """The proposer's opinion never overrides which part is dangerous."""
    poses = [
        grasp_at(part_centre(tool_pose, "handle")),
        grasp_at(part_centre(tool_pose, "head")),
    ]
    ranking = gs.rank_candidates(
        gs.candidates_from_arrays(poses, [0.99, 0.10]), tool_pose, DEFAULT_SCENE
    )
    assert [entry["index"] for entry in ranking["kept"]] == [1]


def test_a_grasp_that_misses_the_tool_is_rejected(tool_pose):
    ranking = gs.rank_candidates(
        gs.candidates_from_arrays([grasp_at((1.0, -0.5, 0.3))], [0.8]),
        tool_pose,
        DEFAULT_SCENE,
    )
    assert ranking["rejected"][0]["part"] is None
    assert "None" in ranking["rejected"][0]["rejected_because"]


def test_candidates_are_ordered_by_score(tool_pose):
    head = part_centre(tool_pose, "head")
    poses = [grasp_at(head), grasp_at(head), grasp_at(head)]
    ranking = gs.rank_candidates(
        gs.candidates_from_arrays(poses, [0.4, 0.9, 0.6]), tool_pose, DEFAULT_SCENE
    )
    assert [entry["index"] for entry in ranking["kept"]] == [1, 2, 0]
    assert [entry["rank"] for entry in ranking["kept"]] == [0, 1, 2]


def test_equal_scores_prefer_the_more_downward_approach(tool_pose):
    head = part_centre(tool_pose, "head")
    sideways = grasp_at(head, approach=(1.0, 0.0, 0.0), closing=(0.0, 0.0, 1.0))
    downward = grasp_at(head)
    ranking = gs.rank_candidates(
        gs.candidates_from_arrays([sideways, downward], [0.7, 0.7]),
        tool_pose,
        DEFAULT_SCENE,
    )
    assert [entry["index"] for entry in ranking["kept"]] == [1, 0]
    assert ranking["kept"][0]["downwardness"] == pytest.approx(1.0)
    assert ranking["kept"][1]["downwardness"] == pytest.approx(0.0)


def test_a_downwardness_floor_can_exclude_side_grasps(tool_pose):
    head = part_centre(tool_pose, "head")
    sideways = grasp_at(head, approach=(1.0, 0.0, 0.0), closing=(0.0, 0.0, 1.0))
    ranking = gs.rank_candidates(
        gs.candidates_from_arrays([sideways], [0.9]),
        tool_pose,
        DEFAULT_SCENE,
        minimum_downwardness=0.5,
    )
    assert ranking["counts"]["on_the_intended_part"] == 0
    assert "downward" in ranking["rejected"][0]["rejected_because"]


def test_an_upward_grasp_scores_negative_downwardness(tool_pose):
    head = part_centre(tool_pose, "head")
    upward = grasp_at(head, approach=(0.0, 0.0, 1.0), closing=(0.0, 1.0, 0.0))
    candidate = gs.candidates_from_arrays([upward], [0.5])[0]
    assert candidate.downwardness() == pytest.approx(-1.0)


def test_the_fingertip_is_the_point_that_is_classified(tool_pose):
    """It is where the fingers close that matters, not where the wrist is."""
    head = part_centre(tool_pose, "head")
    candidate = gs.candidates_from_arrays([grasp_at(head)], [0.5])[0]
    assert candidate.fingertip_m() == pytest.approx(head, abs=1e-9)
    assert candidate.hand_pose[2, 3] > head[2]


def test_no_candidates_is_reported_rather_than_crashing(tool_pose):
    ranking = gs.rank_candidates([], tool_pose, DEFAULT_SCENE)
    assert ranking["counts"] == {
        "proposed": 0,
        "on_the_intended_part": 0,
        "rejected": 0,
    }
    assert ranking["ordered"] == []


def test_mismatched_poses_and_scores_are_refused():
    with pytest.raises(ValueError, match="scores"):
        gs.candidates_from_arrays(np.zeros((3, 4, 4)), [0.1, 0.2])


def test_badly_shaped_poses_are_refused():
    with pytest.raises(ValueError, match=r"\(N, 4, 4\)"):
        gs.candidates_from_arrays(np.zeros((3, 4)), [0.1, 0.2, 0.3])


def test_rejections_are_grouped_by_part(tool_pose):
    poses = [
        grasp_at(part_centre(tool_pose, "handle")),
        grasp_at(part_centre(tool_pose, "handle")),
        grasp_at((1.0, -0.5, 0.3)),
    ]
    ranking = gs.rank_candidates(
        gs.candidates_from_arrays(poses, [0.5, 0.5, 0.5]), tool_pose, DEFAULT_SCENE
    )
    summary = gs.rejection_summary(ranking)
    assert summary["rejected_by_part"]["handle"] == 2
    assert summary["rejected_by_part"]["None"] == 1


def test_selection_follows_a_rotated_tool(tool_pose):
    """The same world grasp changes verdict when the tool turns around."""
    turned = gt.tool_pose_matrix(
        (0.45, 0.15, DEFAULT_SCENE.table_top_z_m + 0.0225), (0.0, 0.0, 0.0, 1.0)
    )
    probe = grasp_at(part_centre(tool_pose, "head"))
    upright = gs.rank_candidates(
        gs.candidates_from_arrays([probe], [0.8]), tool_pose, DEFAULT_SCENE
    )
    rotated = gs.rank_candidates(
        gs.candidates_from_arrays([probe], [0.8]), turned, DEFAULT_SCENE
    )
    assert upright["counts"]["on_the_intended_part"] == 1
    assert rotated["counts"]["on_the_intended_part"] == 0
    assert rotated["rejected"][0]["part"] == "handle"

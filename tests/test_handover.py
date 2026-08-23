"""Tests for the handover waypoint geometry.

The cross-check matters most: the planned handover pose is derived here, and
then measured with the evaluation code in ground_truth.py, which knows nothing
about this module.  If the two ever disagree, a trial would be planned to fail
its own scoring.
"""

from __future__ import annotations

import numpy as np
import pytest

from jikkenn2 import ground_truth as gt
from jikkenn2 import handover as hv
from jikkenn2.reachability import PANDA_FINGERTIP_DEPTH_M
from jikkenn2.scene_spec import DEFAULT_SCENE


IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)


def tool_at(position, quaternion=IDENTITY_QUAT):
    return gt.tool_pose_matrix(position, quaternion)


def yaw_quat(degrees):
    half = np.deg2rad(degrees) / 2.0
    return (float(np.cos(half)), 0.0, 0.0, float(np.sin(half)))


@pytest.fixture
def resting_tool():
    height = DEFAULT_SCENE.table_top_z_m + 0.0225
    return tool_at((0.45, 0.15, height))


def test_grasp_approaches_straight_down(resting_tool):
    grasp = hv.grasp_hand_pose(resting_tool, DEFAULT_SCENE)
    assert grasp[:3, 2] == pytest.approx([0.0, 0.0, -1.0], abs=1e-12)


def test_grasp_rotation_is_proper(resting_tool):
    rotation = hv.grasp_hand_pose(resting_tool, DEFAULT_SCENE)[:3, :3]
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_fingertips_land_on_the_grasped_part(resting_tool):
    grasp = hv.grasp_hand_pose(resting_tool, DEFAULT_SCENE)
    fingertip = grasp[:3, 3] + grasp[:3, 2] * PANDA_FINGERTIP_DEPTH_M
    part = DEFAULT_SCENE.part(DEFAULT_SCENE.grasp_part_name)
    assert gt.point_in_part(resting_tool, part, fingertip)
    assert gt.part_containing(resting_tool, DEFAULT_SCENE, fingertip) == "head"


def test_the_grasped_part_is_the_dangerous_one():
    assert DEFAULT_SCENE.grasp_part_name == DEFAULT_SCENE.danger_part_name


def test_wrist_sits_above_the_tool(resting_tool):
    grasp = hv.grasp_hand_pose(resting_tool, DEFAULT_SCENE)
    assert grasp[2, 3] == pytest.approx(
        resting_tool[2, 3] + PANDA_FINGERTIP_DEPTH_M, abs=1e-9
    )


def test_fingers_close_across_the_tool_not_along_it(resting_tool):
    """The closing axis must be perpendicular to the tool's long axis."""
    grasp = hv.grasp_hand_pose(resting_tool, DEFAULT_SCENE)
    closing = grasp[:3, hv.HAND_CLOSING_COLUMN]
    long_axis = resting_tool[:3, :3] @ np.array([1.0, 0.0, 0.0])
    assert abs(float(np.dot(closing, long_axis))) == pytest.approx(0.0, abs=1e-9)


def test_closing_axis_follows_a_rotated_tool():
    turned = tool_at((0.45, 0.15, 0.0225), yaw_quat(37.0))
    grasp = hv.grasp_hand_pose(turned, DEFAULT_SCENE)
    closing = grasp[:3, hv.HAND_CLOSING_COLUMN]
    long_axis = turned[:3, :3] @ np.array([1.0, 0.0, 0.0])
    assert abs(float(np.dot(closing, long_axis))) == pytest.approx(0.0, abs=1e-9)
    assert closing[2] == pytest.approx(0.0, abs=1e-9)


def test_grasp_span_fits_the_gripper(resting_tool):
    """Whichever horizontal axis the fingers use, the head fits in 80 mm."""
    part = DEFAULT_SCENE.part(DEFAULT_SCENE.grasp_part_name)
    assert max(part.size_m[0], part.size_m[1], part.size_m[2]) < 0.08


def test_pregrasp_is_directly_above_the_grasp(resting_tool):
    grasp = hv.grasp_hand_pose(resting_tool, DEFAULT_SCENE)
    pregrasp = hv.pregrasp_pose(grasp, approach_offset_m=0.12)
    assert np.allclose(pregrasp[:3, :3], grasp[:3, :3])
    assert pregrasp[2, 3] == pytest.approx(grasp[2, 3] + 0.12)
    assert pregrasp[:2, 3] == pytest.approx(grasp[:2, 3])


def test_pregrasp_offset_must_be_positive(resting_tool):
    grasp = hv.grasp_hand_pose(resting_tool, DEFAULT_SCENE)
    with pytest.raises(ValueError, match="positive"):
        hv.pregrasp_pose(grasp, approach_offset_m=0.0)


def test_lift_goes_straight_up_in_the_world(resting_tool):
    grasp = hv.grasp_hand_pose(resting_tool, DEFAULT_SCENE)
    lift = hv.lift_pose(grasp, lift_m=0.15)
    assert np.allclose(lift[:3, :3], grasp[:3, :3])
    assert lift[2, 3] == pytest.approx(grasp[2, 3] + 0.15)
    assert lift[:2, 3] == pytest.approx(grasp[:2, 3])


def test_planned_handover_pose_passes_the_scoring_criteria():
    """The cross-check: planning and evaluation agree, derived independently."""
    target = hv.handover_tool_pose(DEFAULT_SCENE)
    metrics = gt.handover_orientation(target, DEFAULT_SCENE)
    assert metrics["safe_axis_to_human_deg"] < 30.0
    assert metrics["danger_axis_to_human_deg"] > 90.0


def test_planned_handover_pose_is_at_the_handover_point():
    target = hv.handover_tool_pose(DEFAULT_SCENE)
    assert target[:3, 3] == pytest.approx(DEFAULT_SCENE.handover_position_m)


def test_planned_handover_keeps_the_tool_level():
    target = hv.handover_tool_pose(DEFAULT_SCENE)
    assert target[2, 2] == pytest.approx(1.0)
    assert target[:3, 2] == pytest.approx([0.0, 0.0, 1.0], abs=1e-12)


def test_handover_yaw_tracks_a_moved_person():
    import dataclasses

    for human_y in (-1.1, -2.0):
        scene = dataclasses.replace(DEFAULT_SCENE, human_point_m=(0.45, human_y, 0.35))
        target = hv.handover_tool_pose(scene)
        metrics = gt.handover_orientation(target, scene)
        assert metrics["safe_axis_to_human_deg"] < 30.0

    east = dataclasses.replace(DEFAULT_SCENE, human_point_m=(1.40, -0.45, 0.35))
    target = hv.handover_tool_pose(east)
    metrics = gt.handover_orientation(target, east)
    assert metrics["safe_axis_to_human_deg"] < 30.0


def test_hand_pose_for_tool_pose_round_trips(resting_tool):
    """Applying the grasp transform to the grasp pose returns the same hand pose."""
    grasp = hv.grasp_hand_pose(resting_tool, DEFAULT_SCENE)
    same = hv.hand_pose_for_tool_pose(resting_tool, resting_tool, grasp)
    assert same == pytest.approx(grasp, abs=1e-12)


def test_carrying_the_tool_preserves_the_grasp(resting_tool):
    """After moving the hand to the handover pose, the tool is where it should be."""
    waypoints = hv.plan_waypoints(resting_tool, DEFAULT_SCENE)
    grasp = waypoints["grasp"]
    hand_at_handover = waypoints["handover"]
    T_tool_hand = np.linalg.inv(resting_tool) @ grasp
    tool_at_handover = hand_at_handover @ np.linalg.inv(T_tool_hand)
    assert tool_at_handover == pytest.approx(waypoints["target_tool"], abs=1e-9)

    metrics = gt.handover_orientation(tool_at_handover, DEFAULT_SCENE)
    assert metrics["safe_axis_to_human_deg"] < 30.0
    assert metrics["danger_axis_to_human_deg"] > 90.0


def test_the_held_part_is_still_the_dangerous_one_at_handover(resting_tool):
    """The robot must still be holding the head when it presents the handle."""
    waypoints = hv.plan_waypoints(resting_tool, DEFAULT_SCENE)
    hand = waypoints["handover"]
    fingertip = hand[:3, 3] + hand[:3, 2] * PANDA_FINGERTIP_DEPTH_M
    assert gt.part_containing(waypoints["target_tool"], DEFAULT_SCENE, fingertip) == "head"


def test_waypoints_are_all_proper_rigid_transforms(resting_tool):
    for name, pose in hv.plan_waypoints(resting_tool, DEFAULT_SCENE).items():
        assert pose.shape == (4, 4), name
        assert pose[3] == pytest.approx([0, 0, 0, 1]), name
        rotation = pose[:3, :3]
        assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9), name
        assert np.linalg.det(rotation) == pytest.approx(1.0), name


def test_description_is_json_friendly(resting_tool):
    described = hv.describe_waypoints(hv.plan_waypoints(resting_tool, DEFAULT_SCENE))
    assert set(described) == {"pregrasp", "grasp", "lift", "handover", "target_tool"}
    for entry in described.values():
        assert len(entry["position_m"]) == 3
        assert len(entry["orientation_wxyz"]) == 4
        assert all(isinstance(value, float) for value in entry["position_m"])


def test_a_tool_standing_on_end_still_gets_a_valid_grasp():
    upright = gt.tool_pose_matrix(
        (0.45, 0.15, 0.10), (float(np.cos(np.pi / 4)), 0.0, float(np.sin(np.pi / 4)), 0.0)
    )
    grasp = hv.grasp_hand_pose(upright, DEFAULT_SCENE)
    rotation = grasp[:3, :3]
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9)
    assert grasp[:3, 2] == pytest.approx([0.0, 0.0, -1.0], abs=1e-12)

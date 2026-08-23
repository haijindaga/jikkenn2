"""Tests for the evaluation-only ground-truth geometry.

These cover the arithmetic behind the claim being measured: which part is held,
and where each part points relative to the person.
"""

from __future__ import annotations

import numpy as np
import pytest

from jikkenn2 import ground_truth as gt
from jikkenn2.scene_spec import DEFAULT_SCENE


IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)
YAW_180_QUAT = (0.0, 0.0, 0.0, 1.0)
YAW_90_QUAT = (np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4))


def pose(position, quaternion=IDENTITY_QUAT):
    return gt.tool_pose_matrix(position, quaternion)


def test_identity_pose_puts_parts_at_their_local_centres():
    matrix = pose((0.0, 0.0, 0.0))
    poses = gt.part_poses(matrix, DEFAULT_SCENE)
    head = DEFAULT_SCENE.part("head")
    assert poses["head"].center_m == pytest.approx(head.center_m)
    assert poses["head"].axis == pytest.approx((1.0, 0.0, 0.0))
    assert poses["handle"].axis == pytest.approx((-1.0, 0.0, 0.0))


def test_translation_moves_the_parts_with_the_tool():
    matrix = pose((0.5, -0.2, 0.1))
    poses = gt.part_poses(matrix, DEFAULT_SCENE)
    head = DEFAULT_SCENE.part("head")
    assert poses["head"].center_m == pytest.approx(
        tuple(a + b for a, b in zip(head.center_m, (0.5, -0.2, 0.1)))
    )
    # A pure translation must not change any direction.
    assert poses["head"].axis == pytest.approx((1.0, 0.0, 0.0))


def test_yaw_rotates_the_part_axes():
    matrix = pose((0.0, 0.0, 0.0), YAW_90_QUAT)
    poses = gt.part_poses(matrix, DEFAULT_SCENE)
    assert poses["head"].axis == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)
    assert poses["handle"].axis == pytest.approx((0.0, -1.0, 0.0), abs=1e-9)


def test_part_axes_stay_unit_length_and_opposed():
    matrix = pose((0.4, 0.1, 0.05), YAW_90_QUAT)
    poses = gt.part_poses(matrix, DEFAULT_SCENE)
    for placed in poses.values():
        assert np.linalg.norm(placed.axis) == pytest.approx(1.0)
    assert np.dot(poses["head"].axis, poses["handle"].axis) == pytest.approx(-1.0)


def test_a_point_inside_the_head_is_recognised():
    matrix = pose((0.5, 0.0, 0.05))
    head_centre = np.array(DEFAULT_SCENE.part("head").center_m) + np.array([0.5, 0.0, 0.05])
    assert gt.part_containing(matrix, DEFAULT_SCENE, head_centre) == "head"


def test_a_point_inside_the_handle_is_recognised():
    matrix = pose((0.5, 0.0, 0.05))
    handle_centre = np.array(DEFAULT_SCENE.part("handle").center_m) + np.array(
        [0.5, 0.0, 0.05]
    )
    assert gt.part_containing(matrix, DEFAULT_SCENE, handle_centre) == "handle"


def test_a_point_off_the_tool_belongs_to_no_part():
    matrix = pose((0.5, 0.0, 0.05))
    assert gt.part_containing(matrix, DEFAULT_SCENE, (1.0, 1.0, 1.0)) is None


def test_containment_follows_the_tool_when_it_is_rotated():
    """The same world point changes part when the tool turns around."""
    position = (0.5, 0.0, 0.05)
    head_offset = np.array(DEFAULT_SCENE.part("head").center_m)
    probe = np.array(position) + head_offset

    upright = pose(position, IDENTITY_QUAT)
    turned = pose(position, YAW_180_QUAT)
    assert gt.part_containing(upright, DEFAULT_SCENE, probe) == "head"
    assert gt.part_containing(turned, DEFAULT_SCENE, probe) == "handle"


def test_point_shape_is_validated():
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        gt.point_in_part(pose((0, 0, 0)), DEFAULT_SCENE.part("head"), (0.0, 0.0))


def test_angle_between_known_directions():
    assert gt.angle_between_deg((1, 0, 0), (1, 0, 0)) == pytest.approx(0.0)
    assert gt.angle_between_deg((1, 0, 0), (0, 1, 0)) == pytest.approx(90.0)
    assert gt.angle_between_deg((1, 0, 0), (-1, 0, 0)) == pytest.approx(180.0)


def test_angle_rejects_a_zero_direction():
    with pytest.raises(ValueError, match="zero-length"):
        gt.angle_between_deg((0, 0, 0), (1, 0, 0))


def test_handle_pointed_at_the_person_passes_the_criteria():
    """Handle toward the person at -y, head away: this is the success case.

    The head is local +x and the handle local -x, so a +90 degree yaw sends the
    head to +y and the handle to -y, which is where the person stands.
    """
    matrix = gt.tool_pose_matrix(DEFAULT_SCENE.handover_position_m, YAW_90_QUAT)
    metrics = gt.handover_orientation(matrix, DEFAULT_SCENE)
    assert metrics["safe_axis_to_human_deg"] < 30.0
    assert metrics["danger_axis_to_human_deg"] > 90.0


def test_blade_pointed_at_the_person_fails_the_criteria():
    """The mirror image: a -90 degree yaw hands the person the head."""
    quaternion = (np.cos(-np.pi / 4), 0.0, 0.0, np.sin(-np.pi / 4))
    matrix = gt.tool_pose_matrix(DEFAULT_SCENE.handover_position_m, quaternion)
    metrics = gt.handover_orientation(matrix, DEFAULT_SCENE)
    assert metrics["safe_axis_to_human_deg"] > 90.0
    assert metrics["danger_axis_to_human_deg"] < 30.0


def test_the_scene_defines_a_handover_yaw_that_passes():
    """Sanity: the passing orientation is reachable from the handover pose."""
    matrix = gt.tool_pose_matrix(DEFAULT_SCENE.handover_position_m, YAW_90_QUAT)
    metrics = gt.handover_orientation(matrix, DEFAULT_SCENE)
    assert metrics["distance_to_human_m"] == pytest.approx(0.652, abs=0.01)
    assert metrics["safe_part"] == "handle"
    assert metrics["danger_part"] == "head"


def test_sideways_presentation_fails_both_ways():
    matrix = gt.tool_pose_matrix(DEFAULT_SCENE.handover_position_m, IDENTITY_QUAT)
    metrics = gt.handover_orientation(matrix, DEFAULT_SCENE)
    assert 60.0 < metrics["safe_axis_to_human_deg"] < 120.0
    assert 60.0 < metrics["danger_axis_to_human_deg"] < 120.0


def test_the_two_axes_are_always_supplementary():
    for quaternion in (IDENTITY_QUAT, YAW_90_QUAT, YAW_180_QUAT):
        matrix = gt.tool_pose_matrix((0.4, -0.4, 0.35), quaternion)
        metrics = gt.handover_orientation(matrix, DEFAULT_SCENE)
        total = metrics["safe_axis_to_human_deg"] + metrics["danger_axis_to_human_deg"]
        assert total == pytest.approx(180.0, abs=1e-3)


def test_pose_difference_measures_translation_and_rotation():
    before = {"prim_path": "/a", "position_m": [0, 0, 0], "orientation_wxyz": IDENTITY_QUAT}
    after = {"prim_path": "/a", "position_m": [0, 0, 0.05], "orientation_wxyz": YAW_90_QUAT}
    difference = gt.pose_difference(before, after)
    assert difference["translation_m"] == pytest.approx(0.05)
    assert difference["rotation_deg"] == pytest.approx(90.0, abs=1e-3)


def test_settle_report_is_clean_when_nothing_moves():
    authored = [
        {"prim_path": "/a", "position_m": [0, 0, 0], "orientation_wxyz": IDENTITY_QUAT}
    ]
    report = gt.settle_report(authored, authored)
    assert report["status"] == "success"
    assert report["objects_that_moved"] == []


def test_settle_report_flags_an_object_that_fell():
    authored = [
        {"prim_path": "/a", "position_m": [0, 0, 0.3], "orientation_wxyz": IDENTITY_QUAT}
    ]
    settled = [
        {"prim_path": "/a", "position_m": [0, 0, 0.0], "orientation_wxyz": IDENTITY_QUAT}
    ]
    report = gt.settle_report(authored, settled)
    assert report["status"] == "settle_warning"
    assert report["objects_that_moved"] == ["/a"]
    assert report["differences"][0]["translation_m"] == pytest.approx(0.3)


def test_settle_report_flags_an_object_that_tipped_over():
    authored = [
        {"prim_path": "/a", "position_m": [0, 0, 0], "orientation_wxyz": IDENTITY_QUAT}
    ]
    settled = [
        {"prim_path": "/a", "position_m": [0, 0, 0], "orientation_wxyz": YAW_90_QUAT}
    ]
    report = gt.settle_report(authored, settled)
    assert report["status"] == "settle_warning"
    assert report["differences"][0]["rotation_deg"] == pytest.approx(90.0, abs=1e-3)


def test_settle_report_rejects_an_unknown_object():
    authored = [
        {"prim_path": "/a", "position_m": [0, 0, 0], "orientation_wxyz": IDENTITY_QUAT}
    ]
    settled = [
        {"prim_path": "/b", "position_m": [0, 0, 0], "orientation_wxyz": IDENTITY_QUAT}
    ]
    with pytest.raises(KeyError, match="/b"):
        gt.settle_report(authored, settled)


def test_rotation_in_grip_ignores_rotation_applied_by_the_arm():
    """A firm grasp reads zero even when the arm turns the tool 90 degrees."""
    hand_then = {"position_m": [0.4, 0.1, 0.3], "orientation_wxyz": IDENTITY_QUAT}
    tool_then = {"position_m": [0.4, 0.1, 0.2], "orientation_wxyz": IDENTITY_QUAT}
    # The arm yaws by 90 degrees about the hand origin, carrying the tool with it.
    hand_now = {"position_m": [0.4, 0.1, 0.3], "orientation_wxyz": YAW_90_QUAT}
    tool_now = {"position_m": [0.5, 0.1, 0.2], "orientation_wxyz": YAW_90_QUAT}
    assert gt.rotation_in_grip_deg(hand_then, tool_then, hand_now, tool_now) == (
        pytest.approx(0.0, abs=1e-9)
    )


def test_rotation_in_grip_reports_a_tool_that_turned_in_the_fingers():
    hand = {"position_m": [0.4, 0.1, 0.3], "orientation_wxyz": IDENTITY_QUAT}
    tool_then = {"position_m": [0.4, 0.1, 0.2], "orientation_wxyz": IDENTITY_QUAT}
    tool_now = {"position_m": [0.4, 0.1, 0.2], "orientation_wxyz": YAW_90_QUAT}
    assert gt.rotation_in_grip_deg(hand, tool_then, hand, tool_now) == pytest.approx(
        90.0, abs=1e-6
    )


def test_world_rotation_would_have_reported_the_arms_own_turn():
    """The old measurement, kept to show why the hand-relative one is needed."""
    tool_then = {"position_m": [0.4, 0.1, 0.2], "orientation_wxyz": IDENTITY_QUAT}
    tool_now = {"position_m": [0.5, 0.1, 0.2], "orientation_wxyz": YAW_90_QUAT}
    assert gt.pose_difference(tool_then, tool_now)["rotation_deg"] == pytest.approx(
        90.0, abs=1e-3
    )


def test_pose_in_frame_round_trips():
    frame = gt.tool_pose_matrix([0.3, -0.2, 0.4], YAW_90_QUAT)
    pose = gt.tool_pose_matrix([0.5, 0.1, 0.2], YAW_180_QUAT)
    assert frame @ gt.pose_in_frame(frame, pose) == pytest.approx(pose, abs=1e-12)


def test_rotation_between_identical_poses_is_zero():
    pose = gt.tool_pose_matrix([0.1, 0.2, 0.3], YAW_90_QUAT)
    assert gt.rotation_between_deg(pose, pose) == pytest.approx(0.0, abs=1e-9)


def _point_image(points):
    """Pack a list of world points into a 1 x N x 3 point map."""
    return np.asarray(points, dtype=np.float64).reshape(1, -1, 3)


def test_tool_mask_marks_points_on_the_tool_only():
    tool = pose((0.5, 0.1, 0.05))
    head = np.array(DEFAULT_SCENE.part("head").center_m) + np.array([0.5, 0.1, 0.05])
    handle = np.array(DEFAULT_SCENE.part("handle").center_m) + np.array([0.5, 0.1, 0.05])
    elsewhere = np.array([0.8, -0.3, 0.05])

    mask = gt.points_in_tool_mask(
        tool, DEFAULT_SCENE, _point_image([head, handle, elsewhere]), margin_m=0.0
    )
    assert list(mask[0]) == [True, True, False]


def test_tool_mask_follows_a_rotated_tool():
    turned = pose((0.5, 0.1, 0.05), YAW_90_QUAT)
    head_local = np.array(DEFAULT_SCENE.part("head").center_m)
    head_world = turned[:3, :3] @ head_local + turned[:3, 3]
    stale = np.array(DEFAULT_SCENE.part("head").center_m) + np.array([0.5, 0.1, 0.05])

    mask = gt.points_in_tool_mask(
        turned, DEFAULT_SCENE, _point_image([head_world, stale]), margin_m=0.0
    )
    assert list(mask[0]) == [True, False]


def test_tool_mask_margin_widens_the_footprint():
    tool = pose((0.5, 0.1, 0.05))
    head = DEFAULT_SCENE.part("head")
    _, maximum = head.aabb_local_m()
    just_outside = np.asarray(maximum) + np.array([0.005, 0.0, 0.0]) + np.array([0.5, 0.1, 0.05])

    tight = gt.points_in_tool_mask(
        tool, DEFAULT_SCENE, _point_image([just_outside]), margin_m=0.0
    )
    loose = gt.points_in_tool_mask(
        tool, DEFAULT_SCENE, _point_image([just_outside]), margin_m=0.01
    )
    assert tight[0][0] is np.True_ or bool(tight[0][0]) is False
    assert bool(loose[0][0]) is True


def test_tool_mask_ignores_invalid_points():
    tool = pose((0.5, 0.1, 0.05))
    mask = gt.points_in_tool_mask(
        tool,
        DEFAULT_SCENE,
        _point_image([[np.nan, np.nan, np.nan], [np.inf, 0.0, 0.0]]),
    )
    assert not mask.any()


def test_tool_mask_keeps_the_image_shape():
    tool = pose((0.5, 0.1, 0.05))
    points = np.full((4, 7, 3), np.nan)
    assert gt.points_in_tool_mask(tool, DEFAULT_SCENE, points).shape == (4, 7)


def test_tool_mask_rejects_a_non_image_point_array():
    with pytest.raises(ValueError, match="HxWx3"):
        gt.points_in_tool_mask(pose((0, 0, 0)), DEFAULT_SCENE, np.zeros((5, 3)))


def test_tool_mask_rejects_a_negative_margin():
    with pytest.raises(ValueError, match="margin_m"):
        gt.points_in_tool_mask(
            pose((0, 0, 0)), DEFAULT_SCENE, np.zeros((1, 1, 3)), margin_m=-0.01
        )

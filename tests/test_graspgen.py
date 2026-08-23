"""Tests for the frames around the GraspGenX server."""

from __future__ import annotations

import numpy as np
import pytest

from jikkenn2 import graspgen
from jikkenn2.reachability import PANDA_FINGERTIP_DEPTH_M


def test_the_grasp_to_hand_transform_is_a_quarter_turn_about_the_approach_axis():
    transform = graspgen.T_GRASP_PANDA_HAND
    rotation = transform[:3, :3]
    assert np.allclose(rotation.T @ rotation, np.eye(3))
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    # The approach axis is shared; only the closing axis turns.
    assert rotation[:, 2] == pytest.approx([0.0, 0.0, 1.0])
    assert transform[:3, 3] == pytest.approx([0.0, 0.0, 0.0])


def test_the_transform_maps_the_graspgen_closing_axis_onto_the_panda_one():
    """GraspGenX closes along X; panda_hand closes along its own Y.

    A closing axis is a line, not a direction -- the fingers meet either way --
    so the two axes must be parallel, and the sign is not part of the claim.
    """
    hand = np.eye(4) @ graspgen.T_GRASP_PANDA_HAND
    hand_closing_in_grasp_frame = hand[:3, 1]
    graspgen_closing = np.array([1.0, 0.0, 0.0])
    assert abs(float(np.dot(hand_closing_in_grasp_frame, graspgen_closing))) == (
        pytest.approx(1.0)
    )
    # The other two axes are not parallel, so the turn is real.
    assert float(np.dot(hand[:3, 0], graspgen_closing)) == pytest.approx(0.0)


def test_the_hand_closing_column_agrees_with_the_handover_geometry():
    """The column handover.py grips along must be the one this transform makes."""
    from jikkenn2 import handover as hv

    hand = np.eye(4) @ graspgen.T_GRASP_PANDA_HAND
    closing = hand[:3, hv.HAND_CLOSING_COLUMN]
    assert abs(float(np.dot(closing, np.array([1.0, 0.0, 0.0])))) == pytest.approx(1.0)


def test_point_cloud_keeps_the_image_shape_and_counts_instance_pixels():
    points = np.zeros((3, 4, 3), dtype=np.float32)
    mask = np.zeros((3, 4), dtype=bool)
    mask[1, 2] = True
    mask[0, 0] = True
    cloud, instance, count = graspgen.prepare_scene_point_cloud(points, mask)
    assert cloud.shape == (3, 4, 3)
    assert instance.shape == (3, 4)
    assert instance.dtype == np.int32
    assert count == 2


def test_masked_pixels_without_a_finite_point_are_not_counted():
    points = np.zeros((2, 2, 3), dtype=np.float32)
    points[0, 0] = np.nan
    mask = np.ones((2, 2), dtype=bool)
    _, instance, count = graspgen.prepare_scene_point_cloud(points, mask)
    assert count == 3
    assert instance[0, 0] == 0


def test_a_mismatched_mask_is_refused():
    with pytest.raises(ValueError, match="does not match"):
        graspgen.prepare_scene_point_cloud(np.zeros((2, 2, 3)), np.zeros((3, 3), dtype=bool))


def test_grasp_poses_move_into_the_world_frame():
    grasp = np.eye(4)
    grasp[:3, 3] = [0.1, 0.2, 1.0]
    T_world_camera = np.eye(4)
    T_world_camera[:3, 3] = [0.5, 0.0, 1.5]
    world, hand = graspgen.transform_grasp_poses(grasp[None], T_world_camera)
    assert world[0][:3, 3] == pytest.approx([0.6, 0.2, 2.5])
    assert hand[0][:3, 3] == pytest.approx([0.6, 0.2, 2.5])


def test_the_hand_pose_shares_the_grasp_approach_axis():
    grasp = np.eye(4)
    T_world_camera = np.eye(4)
    world, hand = graspgen.transform_grasp_poses(grasp[None], T_world_camera)
    assert hand[0][:3, 2] == pytest.approx(world[0][:3, 2])
    assert not np.allclose(hand[0][:3, 0], world[0][:3, 0])


def test_transformed_poses_stay_rigid():
    rng = np.random.default_rng(0)
    grasps = np.tile(np.eye(4), (5, 1, 1))
    for index in range(5):
        angle = rng.uniform(0, np.pi)
        grasps[index, :3, :3] = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        grasps[index, :3, 3] = rng.uniform(-1, 1, 3)
    _, hand = graspgen.transform_grasp_poses(grasps, np.eye(4))
    quality = graspgen.pose_quality(hand)
    assert quality["finite"]
    assert quality["max_rotation_orthogonality_error"] < 1e-12
    assert quality["max_rotation_determinant_error"] < 1e-12


def test_pose_quality_of_no_poses_is_clean():
    assert graspgen.pose_quality(np.zeros((0, 4, 4)))["finite"] is True


def test_non_finite_input_is_refused():
    grasps = np.tile(np.eye(4), (1, 1, 1))
    grasps[0, 0, 3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        graspgen.transform_grasp_poses(grasps, np.eye(4))


def test_fingertips_on_the_object_pass_the_frame_check():
    target = np.array([0.5, 0.1, 0.05])
    pose = np.eye(4)
    pose[:3, :3] = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
    pose[:3, 3] = target - pose[:3, 2] * PANDA_FINGERTIP_DEPTH_M
    result = graspgen.fingertip_agreement(
        pose[None], target[None], fingertip_depth_m=PANDA_FINGERTIP_DEPTH_M
    )
    assert result["passed"] is True
    assert result["median_distance_m"] == pytest.approx(0.0, abs=1e-9)


def test_a_systematically_offset_conversion_is_caught():
    """What a wrong grasp-to-hand frame would look like."""
    target = np.array([0.5, 0.1, 0.05])
    pose = np.eye(4)
    pose[:3, :3] = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
    pose[:3, 3] = target - pose[:3, 2] * PANDA_FINGERTIP_DEPTH_M + np.array([0.0, 0.0, 0.12])
    result = graspgen.fingertip_agreement(
        pose[None], target[None], fingertip_depth_m=PANDA_FINGERTIP_DEPTH_M
    )
    assert result["passed"] is False
    assert result["median_distance_m"] == pytest.approx(0.12, abs=1e-6)
    assert "frame conversion" in result["means"]


def test_the_frame_check_uses_the_median_not_one_outlier():
    target = np.zeros((1, 3))
    good = np.eye(4)
    good[:3, :3] = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
    good[:3, 3] = -good[:3, 2] * PANDA_FINGERTIP_DEPTH_M
    stray = good.copy()
    stray[:3, 3] += np.array([0.0, 0.0, 1.0])
    poses = np.stack([good, good, stray])
    result = graspgen.fingertip_agreement(
        poses, target, fingertip_depth_m=PANDA_FINGERTIP_DEPTH_M
    )
    assert result["passed"] is True
    assert result["maximum_distance_m"] > 0.9


def test_the_frame_check_reports_when_there_is_nothing_to_compare():
    result = graspgen.fingertip_agreement(
        np.zeros((0, 4, 4)), np.zeros((0, 3)), fingertip_depth_m=0.1
    )
    assert result["passed"] is False
    assert result["checked"] == 0

"""Tests for the Phase 0 layout and its camera model.

The last test is the important one: it moves the camera to the human's side and
asserts that the report *fails*.  A validator that cannot fail is not a
validator, which is exactly the trap jikkenn1 fell into.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from jikkenn2.scene_spec import DEFAULT_SCENE, SceneSpec


def test_default_layout_passes_every_check():
    report = DEFAULT_SCENE.validation_report()
    failed = [name for name, ok in report["automatic_checks"].items() if not ok]
    assert failed == [], failed
    assert report["status"] == "success"


def test_camera_axes_are_orthonormal_and_optical_right_handed():
    forward, right, down = DEFAULT_SCENE.camera_axes()
    axes = np.stack([right, down, forward])  # OpenCV order: x, y, z
    assert np.allclose(axes @ axes.T, np.eye(3), atol=1e-12)
    assert np.allclose(np.cross(right, down), forward, atol=1e-12)
    assert np.isclose(np.linalg.det(axes), 1.0, atol=1e-12)


def test_camera_right_is_horizontal():
    _, right, _ = DEFAULT_SCENE.camera_axes()
    assert np.isclose(right[2], 0.0, atol=1e-12)


def test_optical_axis_point_has_zero_angles():
    target = np.asarray(DEFAULT_SCENE.camera_target_m)
    horizontal, vertical, depth = DEFAULT_SCENE.camera_angles_deg(target)
    assert np.isclose(horizontal, 0.0, atol=1e-9)
    assert np.isclose(vertical, 0.0, atol=1e-9)
    assert depth > 0.0


def test_vertical_fov_follows_the_sensor_aspect_ratio():
    width, height = DEFAULT_SCENE.camera_resolution_px
    half_h = np.deg2rad(DEFAULT_SCENE.camera_horizontal_fov_deg) / 2.0
    expected = np.rad2deg(2.0 * np.arctan(np.tan(half_h) * height / width))
    assert np.isclose(DEFAULT_SCENE.camera_vertical_fov_deg, expected)
    assert DEFAULT_SCENE.camera_vertical_fov_deg < DEFAULT_SCENE.camera_horizontal_fov_deg


def test_point_behind_the_camera_is_not_in_view():
    forward, _, _ = DEFAULT_SCENE.camera_axes()
    behind = np.asarray(DEFAULT_SCENE.camera_position_m) - forward
    assert not DEFAULT_SCENE.point_is_in_view(behind)


def test_point_beyond_the_far_clip_is_not_in_view():
    forward, _, _ = DEFAULT_SCENE.camera_axes()
    _, far = DEFAULT_SCENE.camera_clip_m
    too_far = np.asarray(DEFAULT_SCENE.camera_position_m) + forward * (far + 1.0)
    assert not DEFAULT_SCENE.point_is_in_view(too_far)


def test_table_corners_sit_on_the_tabletop_plane():
    corners = DEFAULT_SCENE.table_top_corners_m()
    assert corners.shape == (4, 3)
    assert np.allclose(corners[:, 2], DEFAULT_SCENE.table_top_z_m)
    assert np.isclose(
        corners[:, 0].max() - corners[:, 0].min(), DEFAULT_SCENE.table_size_m[0]
    )
    assert np.isclose(
        corners[:, 1].max() - corners[:, 1].min(), DEFAULT_SCENE.table_size_m[1]
    )


def test_robot_shadow_never_lands_on_the_tabletop():
    shadow = DEFAULT_SCENE.robot_shadow_on_table_top()
    assert len(shadow) > 0, "the shadow test must actually cast rays"
    assert not any(DEFAULT_SCENE.point_is_over_table(point) for point in shadow)


def test_robot_is_framed_with_margin_above_its_home_envelope():
    in_view_height = DEFAULT_SCENE.robot_height_fully_in_view_m()
    assert in_view_height > DEFAULT_SCENE.robot_body_height_m


def test_proxy_tool_parts_meet_at_the_local_origin():
    head = DEFAULT_SCENE.part("head")
    handle = DEFAULT_SCENE.part("handle")
    head_min, _ = head.aabb_local_m()
    _, handle_max = handle.aabb_local_m()
    assert np.isclose(head_min[0], 0.0)
    assert np.isclose(handle_max[0], 0.0)


def test_grasped_part_fits_the_panda_gripper():
    grasped = DEFAULT_SCENE.part(DEFAULT_SCENE.grasp_part_name)
    panda_max_opening_m = 0.08
    assert min(grasped.size_m[1], grasped.size_m[2]) < panda_max_opening_m


def test_danger_and_safe_parts_point_in_opposite_directions():
    danger = np.asarray(DEFAULT_SCENE.part(DEFAULT_SCENE.danger_part_name).axis_local)
    safe = np.asarray(DEFAULT_SCENE.part(DEFAULT_SCENE.safe_part_name).axis_local)
    assert float(np.dot(danger, safe)) < -0.99


def test_unknown_part_name_is_rejected():
    with pytest.raises(KeyError):
        DEFAULT_SCENE.part("blade")


def test_camera_on_the_human_side_fails_validation():
    """A validator that cannot fail is not a validator."""
    broken = dataclasses.replace(DEFAULT_SCENE, camera_position_m=(0.50, -1.30, 1.60))
    report = broken.validation_report()
    assert report["status"] == "failure"
    assert not report["automatic_checks"]["camera_is_on_the_opposite_side_from_the_human"]


def test_unreachable_handover_fails_validation():
    broken = dataclasses.replace(DEFAULT_SCENE, handover_position_m=(0.40, -1.00, 0.35))
    checks = broken.validation_report()["automatic_checks"]
    assert not checks["handover_is_within_panda_reach"]


def test_table_pushed_into_the_robot_fails_validation():
    broken = dataclasses.replace(DEFAULT_SCENE, table_center_m=(0.20, 0.0, -0.025))
    checks = broken.validation_report()["automatic_checks"]
    assert not checks["table_clears_robot_body"]


def test_narrow_lens_stops_framing_the_table():
    broken = dataclasses.replace(DEFAULT_SCENE, camera_horizontal_fov_deg=20.0)
    checks = broken.validation_report()["automatic_checks"]
    assert not checks["camera_sees_every_table_corner"]


def test_spec_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_SCENE.camera_horizontal_fov_deg = 90.0  # type: ignore[misc]


def test_degenerate_camera_is_rejected():
    broken = SceneSpec(camera_position_m=(0.5, 0.0, 1.6), camera_target_m=(0.5, 0.0, 0.0))
    with pytest.raises(ValueError):
        broken.camera_axes()

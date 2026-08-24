"""Tests for the inspectable point-cloud output."""

from __future__ import annotations

import numpy as np
import pytest

from jikkenn2 import pointcloud


def test_ply_header_and_rows(tmp_path):
    points = np.array([[0.1, 0.2, 0.3], [1.0, -1.0, 0.5]])
    colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    path = pointcloud.write_ascii_ply(tmp_path / "cloud.ply", points, colors)
    lines = path.read_text(encoding="ascii").splitlines()
    assert lines[0] == "ply"
    assert "element vertex 2" in lines
    assert lines[lines.index("end_header") + 1].startswith("0.1 0.2 0.3 255 0 0")
    assert len(lines) == lines.index("end_header") + 3


def test_an_empty_cloud_is_still_a_valid_file(tmp_path):
    path = pointcloud.write_ascii_ply(
        tmp_path / "empty.ply", np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
    )
    text = path.read_text(encoding="ascii")
    assert "element vertex 0" in text
    assert text.rstrip().endswith("end_header")


def test_mismatched_colors_are_refused(tmp_path):
    with pytest.raises(ValueError, match="do not match"):
        pointcloud.write_ascii_ply(
            tmp_path / "bad.ply", np.zeros((3, 3)), np.zeros((2, 3), dtype=np.uint8)
        )


def test_non_xyz_points_are_refused(tmp_path):
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        pointcloud.write_ascii_ply(
            tmp_path / "bad.ply", np.zeros((3, 2)), np.zeros((3, 2), dtype=np.uint8)
        )


def test_single_colour_cloud_counts_points(tmp_path):
    count = pointcloud.write_colored_cloud(
        tmp_path / "one.ply", np.zeros((5, 3)), (220, 60, 30)
    )
    assert count == 5
    assert "220 60 30" in (tmp_path / "one.ply").read_text(encoding="ascii")


def test_subsample_leaves_a_small_cloud_alone():
    points = np.arange(12).reshape(4, 3)
    assert pointcloud.subsample(points, 10) is points


def test_subsample_thins_and_keeps_order():
    points = np.arange(300).reshape(100, 3)
    thinned = pointcloud.subsample(points, 10)
    assert len(thinned) == 10
    assert list(thinned[:, 0]) == sorted(thinned[:, 0])


def test_subsample_is_deterministic():
    points = np.arange(300).reshape(100, 3)
    assert np.array_equal(
        pointcloud.subsample(points, 10), pointcloud.subsample(points, 10)
    )


def test_subsample_needs_a_positive_limit():
    with pytest.raises(ValueError, match="positive"):
        pointcloud.subsample(np.zeros((5, 3)), 0)


def test_a_gripper_marker_spans_the_approach_and_the_jaw():
    pose = np.eye(4)
    pose[:3, :3] = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
    pose[:3, 3] = [0.5, 0.1, 0.3]
    points = pointcloud.gripper_marker_points(
        pose, fingertip_depth_m=0.1034, half_width_m=0.04, samples=16
    )
    assert points.shape == (32, 3)
    # The shaft runs from the wrist down to the fingertip.
    assert points[0] == pytest.approx([0.5, 0.1, 0.3])
    assert points[15] == pytest.approx([0.5, 0.1, 0.3 - 0.1034], abs=1e-6)
    # The jaw crosses the fingertip along the closing axis.
    assert points[16] == pytest.approx([0.5, 0.1 + 0.04, 0.3 - 0.1034], abs=1e-6)
    assert points[31] == pytest.approx([0.5, 0.1 - 0.04, 0.3 - 0.1034], abs=1e-6)


def test_markers_stack_for_many_grasps():
    pose = np.eye(4)
    points = pointcloud.gripper_markers(
        np.stack([pose, pose, pose]), fingertip_depth_m=0.1, samples=8
    )
    assert points.shape == (48, 3)


def test_no_grasps_gives_an_empty_marker_cloud():
    assert pointcloud.gripper_markers(np.zeros((0, 4, 4)), fingertip_depth_m=0.1).shape == (0, 3)


def test_a_badly_shaped_pose_is_refused():
    with pytest.raises(ValueError, match="4x4"):
        pointcloud.gripper_marker_points(np.eye(3), fingertip_depth_m=0.1)

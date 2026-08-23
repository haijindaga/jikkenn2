"""Tests for the tabletop grid, the grasp poses and the overlay mesh."""

from __future__ import annotations

import numpy as np
import pytest

from jikkenn2 import reachability as rc
from jikkenn2.scene_spec import DEFAULT_SCENE

pxr = pytest.importorskip("pxr", reason="usd-core is not installed")

from pxr import Usd, UsdGeom  # noqa: E402


@pytest.fixture
def grid():
    return rc.tabletop_grid(
        DEFAULT_SCENE,
        cell_size_m=0.05,
        grasp_height_m=rc.default_grasp_height_m(DEFAULT_SCENE),
    )


def test_grid_covers_the_table_without_leaving_it(grid):
    assert grid.shape == (14, 24)  # 0.70 / 0.05 by 1.20 / 0.05
    centers = grid.flat_centers_m()
    (min_x, min_y), (max_x, max_y) = (
        DEFAULT_SCENE.table_min_xy_m,
        DEFAULT_SCENE.table_max_xy_m,
    )
    assert centers[:, 0].min() >= min_x and centers[:, 0].max() <= max_x
    assert centers[:, 1].min() >= min_y and centers[:, 1].max() <= max_y
    assert all(DEFAULT_SCENE.point_is_over_table(point) for point in centers)


def test_grid_is_x_major(grid):
    centers = grid.centers_m
    assert centers[1, 0, 0] > centers[0, 0, 0]
    assert centers[0, 1, 1] > centers[0, 0, 1]


@pytest.mark.parametrize(
    "cell_size_m, expected",
    [(0.05, (14, 24)), (0.02, (35, 60)), (0.10, (7, 12)), (0.01, (70, 120))],
)
def test_exact_divisions_do_not_lose_a_row_to_float_error(cell_size_m, expected):
    """0.70 / 0.05 is 13.999999999999996 in float64; flooring it drops a column."""
    grid = rc.tabletop_grid(DEFAULT_SCENE, cell_size_m=cell_size_m, grasp_height_m=0.02)
    assert grid.shape == expected


def test_a_non_dividing_cell_size_still_rounds_down():
    grid = rc.tabletop_grid(DEFAULT_SCENE, cell_size_m=0.03, grasp_height_m=0.02)
    assert grid.shape == (23, 40)  # 0.70/0.03 = 23.33, 1.20/0.03 = 40.0


def test_cell_size_larger_than_the_table_is_rejected():
    with pytest.raises(ValueError, match="larger than the table"):
        rc.tabletop_grid(DEFAULT_SCENE, cell_size_m=5.0, grasp_height_m=0.02)


def test_grasp_height_is_the_grasped_part_centre():
    height = rc.default_grasp_height_m(DEFAULT_SCENE)
    tallest = max(part.size_m[2] for part in DEFAULT_SCENE.tool_parts)
    assert height == pytest.approx(DEFAULT_SCENE.table_top_z_m + 0.5 * tallest)
    assert height > DEFAULT_SCENE.table_top_z_m


def test_every_orientation_is_a_proper_rotation():
    for name, rotation in rc.grasp_orientations():
        assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12), name
        assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-12), name


def test_top_down_orientations_approach_straight_down():
    for name, rotation in rc.grasp_orientations():
        if name.startswith("top_down"):
            assert rotation[:, 2] == pytest.approx([0.0, 0.0, -1.0], abs=1e-12)


def test_side_orientations_approach_horizontally():
    for name, rotation in rc.grasp_orientations():
        if name.startswith("side"):
            assert rotation[2, 2] == pytest.approx(0.0, abs=1e-12)


def test_orientation_set_is_unique_and_complete():
    names = [name for name, _ in rc.grasp_orientations()]
    assert len(names) == len(set(names)) == 8


def test_hand_pose_backs_off_along_its_own_approach_axis():
    point = np.array([[0.5, 0.0, 0.05]])
    poses, names = rc.candidate_hand_poses(point)
    assert poses.shape == (1, 8, 4, 4)
    for index, name in enumerate(names):
        pose = poses[0, index]
        approach = pose[:3, 2]
        fingertip = pose[:3, 3] + approach * rc.PANDA_FINGERTIP_DEPTH_M
        assert fingertip == pytest.approx(point[0], abs=1e-12), name
        if name.startswith("top_down"):
            # backing off along -z means the wrist sits above the object
            assert pose[2, 3] > point[0][2]


def test_candidate_poses_reject_a_bad_shape():
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        rc.candidate_hand_poses(np.zeros((4,)))


def test_classification_thresholds():
    success = np.array(
        [
            [True] * 8,
            [True] + [False] * 7,
            [False] * 8,
        ]
    )
    labels = rc.classify_cells(success)
    assert list(labels) == ["free", "partial", "blocked"]


def test_family_columns_split_top_down_from_side():
    names = [name for name, _ in rc.grasp_orientations()]
    top = rc.family_columns(names, "top_down")
    side = rc.family_columns(names, "side")
    assert len(top) == 4 and len(side) == 4
    assert set(top).isdisjoint(side)
    assert sorted([*top, *side]) == list(range(8))
    assert list(rc.family_columns(names, "all")) == list(range(8))


def test_unknown_family_is_rejected():
    names = [name for name, _ in rc.grasp_orientations()]
    with pytest.raises(ValueError, match="family must be one of"):
        rc.family_columns(names, "diagonal")


def test_top_down_family_ignores_the_side_columns():
    """A cell where every top-down yaw solves is green even if no side does."""
    names = [name for name, _ in rc.grasp_orientations()]
    success = np.zeros((3, 8), dtype=bool)
    top = rc.family_columns(names, "top_down")
    side = rc.family_columns(names, "side")
    success[0, top] = True                 # every top-down solves
    success[1, top[0]] = True              # one top-down solves
    success[2, side] = True                # only side approaches solve

    labels = rc.classify_cells(success, orientation_names=names, family="top_down")
    assert list(labels) == ["free", "partial", "blocked"]

    # The same data is merely "partial" everywhere when all eight are mixed.
    mixed = rc.classify_cells(success, orientation_names=names, family="all")
    assert list(mixed) == ["partial", "partial", "partial"]


def test_family_selection_needs_matching_names():
    with pytest.raises(ValueError, match="orientation_names"):
        rc.classify_cells(np.zeros((2, 8), dtype=bool), family="top_down")
    with pytest.raises(ValueError, match="does not match"):
        rc.classify_cells(
            np.zeros((2, 8), dtype=bool),
            orientation_names=["a", "b"],
            family="top_down",
        )


def test_classification_rejects_the_wrong_shape():
    with pytest.raises(ValueError, match="cells, orientations"):
        rc.classify_cells(np.array([True, False]))


def test_summary_counts_add_up():
    labels = np.array(["free", "free", "partial", "blocked"], dtype=object)
    summary = rc.summarize_labels(labels)
    assert summary["cells"] == 4
    assert summary["counts"]["free"] == 2
    assert sum(summary["counts"].values()) == 4
    assert summary["fractions"]["free"] == pytest.approx(0.5)


def test_overlay_mesh_topology_and_colors(tmp_path, grid):
    labels = np.full(grid.cell_count, "free", dtype=object)
    labels[0] = "blocked"
    labels[1] = "partial"
    info = rc.write_overlay_usd(tmp_path / "overlay.usd", DEFAULT_SCENE, grid, labels)

    stage = Usd.Stage.Open(info["path"])
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/ReachabilityOverlay/cells"))
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    points = mesh.GetPointsAttr().Get()
    colors = mesh.GetDisplayColorPrimvar().Get()

    nx, ny = grid.shape
    assert len(counts) == nx * ny
    assert set(counts) == {4}
    assert len(indices) == 4 * nx * ny
    assert len(points) == (nx + 1) * (ny + 1)
    assert len(colors) == nx * ny
    assert max(indices) < len(points)
    assert tuple(colors[0]) == pytest.approx(rc.LABEL_COLORS["blocked"])
    assert tuple(colors[1]) == pytest.approx(rc.LABEL_COLORS["partial"])
    assert tuple(colors[2]) == pytest.approx(rc.LABEL_COLORS["free"])


def test_overlay_sits_just_above_the_tabletop(tmp_path, grid):
    labels = np.full(grid.cell_count, "unknown", dtype=object)
    info = rc.write_overlay_usd(tmp_path / "overlay.usd", DEFAULT_SCENE, grid, labels)
    stage = Usd.Stage.Open(info["path"])
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/ReachabilityOverlay/cells"))
    heights = {round(float(point[2]), 6) for point in mesh.GetPointsAttr().Get()}
    assert len(heights) == 1
    height = heights.pop()
    assert height > DEFAULT_SCENE.table_top_z_m
    assert height - DEFAULT_SCENE.table_top_z_m < 0.01


def test_overlay_has_its_own_default_prim_so_it_can_be_referenced(tmp_path, grid):
    labels = np.full(grid.cell_count, "free", dtype=object)
    info = rc.write_overlay_usd(tmp_path / "overlay.usd", DEFAULT_SCENE, grid, labels)
    stage = Usd.Stage.Open(info["path"])
    assert stage.GetDefaultPrim().GetPath().pathString == "/ReachabilityOverlay"
    assert UsdGeom.GetStageMetersPerUnit(stage) == pytest.approx(1.0)

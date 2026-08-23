"""Tests for turning a voxel field back into inspectable points.

The index order is the point of these: cuRobo lays a grid out x-slowest and
z-fastest, and a transposed map still has the right shape while placing every
obstacle somewhere else.
"""

from __future__ import annotations

import numpy as np
import pytest

from jikkenn2 import mapping


GRID = {"shape": [4, 3, 2], "voxel_size_m": 0.1, "min_corner_m": [0.0, 0.0, 0.0]}


def test_first_voxel_is_half_a_cell_from_the_corner():
    assert mapping.voxel_centers([0], GRID)[0] == pytest.approx([0.05, 0.05, 0.05])


def test_the_fastest_axis_is_z():
    """Index 1 must step in z, not in x."""
    assert mapping.voxel_centers([1], GRID)[0] == pytest.approx([0.05, 0.05, 0.15])


def test_the_middle_axis_is_y():
    nz = GRID["shape"][2]
    assert mapping.voxel_centers([nz], GRID)[0] == pytest.approx([0.05, 0.15, 0.05])


def test_the_slowest_axis_is_x():
    ny, nz = GRID["shape"][1], GRID["shape"][2]
    assert mapping.voxel_centers([ny * nz], GRID)[0] == pytest.approx([0.15, 0.05, 0.05])


def test_the_last_voxel_is_inside_the_grid():
    nx, ny, nz = GRID["shape"]
    last = mapping.voxel_centers([nx * ny * nz - 1], GRID)[0]
    assert last == pytest.approx([0.35, 0.25, 0.15])


def test_an_index_past_the_end_is_refused():
    nx, ny, nz = GRID["shape"]
    with pytest.raises(ValueError, match="out of range"):
        mapping.voxel_centers([nx * ny * nz], GRID)


def test_only_non_positive_values_count_as_blocked():
    values = np.full(24, 0.5, dtype=np.float32)
    values[0] = -0.01
    values[5] = 0.0
    centers = mapping.blocked_voxel_centers(values, GRID)
    assert len(centers) == 2
    assert centers[0] == pytest.approx([0.05, 0.05, 0.05])


def test_a_free_map_yields_no_points():
    centers = mapping.blocked_voxel_centers(np.full(24, 0.2, dtype=np.float32), GRID)
    assert centers.shape == (0, 3)


def test_a_fully_blocked_map_yields_every_voxel():
    centers = mapping.blocked_voxel_centers(np.full(24, -0.2, dtype=np.float32), GRID)
    assert len(centers) == 24


def test_a_feature_array_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="does not match grid shape"):
        mapping.blocked_voxel_centers(np.zeros(23, dtype=np.float32), GRID)


def test_a_multidimensional_feature_array_is_flattened():
    values = np.full((4, 3, 2), 0.5, dtype=np.float32)
    values[2, 1, 0] = -0.1
    centers = mapping.blocked_voxel_centers(values, GRID)
    assert centers[0] == pytest.approx([0.25, 0.15, 0.05])


def test_a_transposed_map_lands_somewhere_else():
    """Why the order is pinned: the same data read wrongly moves the obstacle."""
    values = np.full((4, 3, 2), 0.5, dtype=np.float32)
    values[3, 0, 0] = -0.1
    correct = mapping.blocked_voxel_centers(values, GRID)[0]
    transposed = mapping.blocked_voxel_centers(
        np.transpose(values, (2, 1, 0)).copy(), GRID
    )[0]
    assert correct == pytest.approx([0.35, 0.05, 0.05])
    assert not np.allclose(correct, transposed)


def test_extent_matches_the_shape_and_voxel_size():
    extent = mapping.map_extent(GRID)
    assert extent["min_m"] == [0.0, 0.0, 0.0]
    assert extent["max_m"] == [0.4, 0.3, 0.2]
    assert extent["voxels"] == 24
    assert extent["index_order"] == mapping.INDEX_ORDER


def test_a_degenerate_shape_is_refused():
    with pytest.raises(ValueError, match="invalid grid shape"):
        mapping.grid_shape({"shape": [4, 0, 2]})

"""Turning a voxel field back into points you can look at.

The index order here is the one thing that must not be guessed: cuRobo lays a
voxel grid out x-slowest and z-fastest, and getting it wrong silently scrambles
a map that still passes every shape check.
"""

from __future__ import annotations

import numpy as np

#: How cuRobo flattens a voxel grid: index = ix * ny * nz + iy * nz + iz.
INDEX_ORDER = "x_slowest_z_fastest"


def grid_shape(grid: dict) -> tuple[int, int, int]:
    shape = tuple(int(value) for value in grid["shape"])
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError(f"invalid grid shape {grid['shape']}")
    return shape


def voxel_centers(indices: np.ndarray, grid: dict) -> np.ndarray:
    """Metric centres of flat voxel indices."""
    nx, ny, nz = grid_shape(grid)
    flat = np.asarray(indices, dtype=np.int64).reshape(-1)
    if flat.size and (flat.min() < 0 or flat.max() >= nx * ny * nz):
        raise ValueError("voxel index out of range for this grid")
    ix = flat // (ny * nz)
    remainder = flat % (ny * nz)
    iy = remainder // nz
    iz = remainder % nz
    minimum = np.asarray(grid["min_corner_m"], dtype=np.float64)
    voxel = float(grid["voxel_size_m"])
    coordinates = np.stack([ix, iy, iz], axis=1).astype(np.float64)
    return (minimum + (coordinates + 0.5) * voxel).astype(np.float32)


def blocked_voxel_centers(features, grid: dict) -> np.ndarray:
    """Centres of the voxels the planner treats as occupied.

    cuRobo's field is negative inside obstacles, so a non-positive value is
    where the arm may not be.
    """
    values = np.asarray(features, dtype=np.float32).reshape(-1)
    nx, ny, nz = grid_shape(grid)
    if values.size != nx * ny * nz:
        raise ValueError(
            f"feature count {values.size} does not match grid shape {(nx, ny, nz)}"
        )
    blocked = np.flatnonzero(values <= 0.0)
    if blocked.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    return voxel_centers(blocked, grid)


def map_extent(grid: dict) -> dict:
    """Where the grid actually covers, for the report."""
    nx, ny, nz = grid_shape(grid)
    minimum = np.asarray(grid["min_corner_m"], dtype=np.float64)
    voxel = float(grid["voxel_size_m"])
    maximum = minimum + np.array([nx, ny, nz], dtype=np.float64) * voxel
    return {
        "min_m": [round(float(v), 4) for v in minimum],
        "max_m": [round(float(v), 4) for v in maximum],
        "voxels": int(nx * ny * nz),
        "index_order": INDEX_ORDER,
    }

"""Where on the table the robot can actually grasp something.

Placing objects by hand only works if "can the arm reach here" is visible
*before* the object is placed.  This module turns the tabletop into a grid,
builds the grasp poses to test at each cell, and renders the answer as a
coloured mesh that lies flat on the table in the Isaac Sim viewport.

The IK itself belongs to cuRobo and lives in ``scripts/reachability_map.py``.
Everything here is plain NumPy and USD, so it is tested without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jikkenn2.scene_spec import SceneSpec


# Distance from the panda_hand origin to the fingertips, from the official
# Franka gripper description used by GraspGenX (end2end/robots/franka_panda.yaml).
PANDA_FINGERTIP_DEPTH_M = 0.1034

LABEL_COLORS = {
    "free": (0.20, 0.68, 0.36),
    "partial": (0.92, 0.74, 0.18),
    "blocked": (0.78, 0.22, 0.16),
    "unknown": (0.55, 0.55, 0.58),
}

# A parallel gripper is symmetric under a half turn, so yaws only need to cover
# 180 degrees.  Side approaches come in from the four horizontal directions.
TOP_DOWN_YAWS_DEG = (0.0, 45.0, 90.0, 135.0)
SIDE_APPROACH_HEADINGS_DEG = (0.0, 90.0, 180.0, 270.0)


@dataclass(frozen=True)
class TabletopGrid:
    """Cell centres over the tabletop, in x-major order."""

    centers_m: np.ndarray  # (nx, ny, 3)
    cell_size_m: float
    min_corner_m: tuple[float, float]
    shape: tuple[int, int]

    @property
    def cell_count(self) -> int:
        return int(self.shape[0] * self.shape[1])

    def flat_centers_m(self) -> np.ndarray:
        return self.centers_m.reshape(-1, 3)


def tabletop_grid(scene: SceneSpec, *, cell_size_m: float, grasp_height_m: float) -> TabletopGrid:
    """Lay a grid of cell centres over the tabletop at the grasp height."""
    if cell_size_m <= 0.0 or not np.isfinite(cell_size_m):
        raise ValueError("cell_size_m must be positive and finite")
    (min_x, min_y), (max_x, max_y) = scene.table_min_xy_m, scene.table_max_xy_m
    # 0.70 / 0.05 evaluates to 13.999999999999996 in float64, and a bare floor
    # would silently drop a whole column.  This is the same failure family as
    # cuRobo issue #699, which cost jikkenn1 a patched dependency.
    nx = int(np.floor((max_x - min_x) / cell_size_m + 1e-9))
    ny = int(np.floor((max_y - min_y) / cell_size_m + 1e-9))
    if nx < 1 or ny < 1:
        raise ValueError("cell_size_m is larger than the table")
    xs = min_x + (np.arange(nx) + 0.5) * cell_size_m
    ys = min_y + (np.arange(ny) + 0.5) * cell_size_m
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    centers = np.stack(
        [grid_x, grid_y, np.full_like(grid_x, float(grasp_height_m))], axis=-1
    )
    return TabletopGrid(
        centers_m=centers,
        cell_size_m=float(cell_size_m),
        min_corner_m=(float(min_x), float(min_y)),
        shape=(nx, ny),
    )


def default_grasp_height_m(scene: SceneSpec) -> float:
    """Height of the grasped part's centre when the tool rests on the table."""
    grasped = scene.part(scene.grasp_part_name)
    tallest = max(part.size_m[2] for part in scene.tool_parts)
    return scene.table_top_z_m + 0.5 * tallest + grasped.center_m[2]


def _rotation_from_axes(approach: np.ndarray, closing: np.ndarray) -> np.ndarray:
    """Build a hand rotation whose +z is ``approach`` and +x is ``closing``."""
    z_axis = np.asarray(approach, dtype=np.float64)
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.asarray(closing, dtype=np.float64)
    x_axis = x_axis - z_axis * float(np.dot(x_axis, z_axis))
    norm = float(np.linalg.norm(x_axis))
    if norm < 1e-9:
        raise ValueError("closing direction is parallel to the approach direction")
    x_axis = x_axis / norm
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def grasp_orientations() -> list[tuple[str, np.ndarray]]:
    """Named hand rotations to test at every cell.

    ``panda_hand`` approaches along its own +z, so a top-down grasp points that
    axis at the table and a side grasp points it horizontally.
    """
    orientations: list[tuple[str, np.ndarray]] = []
    for yaw_deg in TOP_DOWN_YAWS_DEG:
        yaw = np.deg2rad(yaw_deg)
        closing = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        orientations.append(
            (f"top_down_yaw_{int(yaw_deg)}", _rotation_from_axes([0.0, 0.0, -1.0], closing))
        )
    for heading_deg in SIDE_APPROACH_HEADINGS_DEG:
        heading = np.deg2rad(heading_deg)
        approach = np.array([np.cos(heading), np.sin(heading), 0.0])
        orientations.append(
            (f"side_heading_{int(heading_deg)}", _rotation_from_axes(approach, [0.0, 0.0, 1.0]))
        )
    return orientations


def candidate_hand_poses(
    grasp_points_m: np.ndarray, *, fingertip_depth_m: float = PANDA_FINGERTIP_DEPTH_M
) -> tuple[np.ndarray, list[str]]:
    """Return ``(N, K, 4, 4)`` panda_hand poses that grasp each point.

    The hand origin sits ``fingertip_depth_m`` back along its own approach axis,
    so the fingertips close on the requested point rather than the wrist.
    """
    points = np.asarray(grasp_points_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"grasp_points_m must be (N, 3), got {points.shape}")
    named = grasp_orientations()
    poses = np.zeros((points.shape[0], len(named), 4, 4), dtype=np.float64)
    poses[:, :] = np.eye(4)
    for index, (_, rotation) in enumerate(named):
        approach = rotation[:, 2]
        poses[:, index, :3, :3] = rotation
        poses[:, index, :3, 3] = points - approach * float(fingertip_depth_m)
    return poses, [name for name, _ in named]


#: Which orientations a colour is allowed to depend on.  ``top_down`` is the
#: default because it answers the question a person actually has while placing
#: an object: "if I put it here, can the arm pick it up?"  Mixing the side
#: approaches in makes "all four top-down yaws work" and "one awkward side
#: approach works" the same colour, which is useless for placement.
ORIENTATION_FAMILIES = ("top_down", "side", "all")


def family_columns(orientation_names: list[str], family: str) -> np.ndarray:
    """Indices of the orientations belonging to ``family``."""
    if family not in ORIENTATION_FAMILIES:
        raise ValueError(f"family must be one of {ORIENTATION_FAMILIES}, got {family!r}")
    if family == "all":
        return np.arange(len(orientation_names))
    columns = np.array(
        [index for index, name in enumerate(orientation_names) if name.startswith(family)]
    )
    if columns.size == 0:
        raise ValueError(f"no orientation belongs to family {family!r}")
    return columns


def classify_cells(
    success: np.ndarray,
    *,
    orientation_names: list[str] | None = None,
    family: str = "all",
) -> np.ndarray:
    """Turn per-orientation IK results into one label per cell.

    ``free`` means every orientation in the family solved, ``partial`` means at
    least one did, ``blocked`` means none did.
    """
    solved = np.asarray(success, dtype=bool)
    if solved.ndim != 2:
        raise ValueError(f"success must be (cells, orientations), got {solved.shape}")
    if family != "all":
        if orientation_names is None:
            raise ValueError("orientation_names is required to select a family")
        if len(orientation_names) != solved.shape[1]:
            raise ValueError("orientation_names does not match the success columns")
        solved = solved[:, family_columns(orientation_names, family)]
    counts = solved.sum(axis=1)
    labels = np.full(counts.shape, "blocked", dtype=object)
    labels[counts > 0] = "partial"
    labels[counts == solved.shape[1]] = "free"
    return labels


def summarize_labels(labels: np.ndarray) -> dict:
    flat = np.asarray(labels).reshape(-1)
    total = int(flat.size)
    counts = {name: int(np.count_nonzero(flat == name)) for name in LABEL_COLORS}
    return {
        "cells": total,
        "counts": counts,
        "fractions": {
            name: round(value / total, 4) if total else 0.0
            for name, value in counts.items()
        },
    }


def write_overlay_usd(
    path,
    scene: SceneSpec,
    grid: TabletopGrid,
    labels: np.ndarray,
    *,
    lift_m: float = 0.002,
) -> dict:
    """Write a flat, per-cell coloured mesh that lies on the tabletop.

    Kept in its own file so it can be layered over ``scene.usd`` for placement
    and removed again without touching the stage.
    """
    from pathlib import Path

    try:
        from pxr import Gf, Usd, UsdGeom, Vt
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "writing the overlay needs USD (the 'pxr' module), which is not "
            "importable here. Isaac Sim only puts pxr on the path after "
            "SimulationApp starts, and this process does not start it. Either "
            "install the standalone wheel into the environment you are running "
            "in ('pip install usd-core'), or compute the map here and draw it "
            "later with 'python scripts/reachability_map.py --overlay-only'."
        ) from error

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            destination.unlink()
        except OSError as error:
            raise PermissionError(
                f"cannot replace {destination}: {error}. If the repository contains "
                'root-owned files from an earlier sudo run, fix with '
                '"sudo chown -R \\"$USER\\":\\"$USER\\" ." at the repository root.'
            ) from error
    probe = destination.with_name(destination.name + ".writetest")
    try:
        probe.touch()
        probe.unlink()
    except OSError as error:
        raise PermissionError(
            f"cannot write into {destination.parent}: {error}. Check ownership with "
            f'"ls -la {destination.parent}".'
        ) from error

    nx, ny = grid.shape
    label_grid = np.asarray(labels).reshape(nx, ny)
    cell = grid.cell_size_m
    min_x, min_y = grid.min_corner_m
    z = scene.table_top_z_m + lift_m

    points = [
        Gf.Vec3f(float(min_x + i * cell), float(min_y + j * cell), float(z))
        for i in range(nx + 1)
        for j in range(ny + 1)
    ]

    def corner(i: int, j: int) -> int:
        return i * (ny + 1) + j

    face_indices: list[int] = []
    colors: list[Gf.Vec3f] = []
    for i in range(nx):
        for j in range(ny):
            face_indices.extend(
                [corner(i, j), corner(i + 1, j), corner(i + 1, j + 1), corner(i, j + 1)]
            )
            colors.append(Gf.Vec3f(*LABEL_COLORS[str(label_grid[i, j])]))

    stage = Usd.Stage.CreateNew(str(destination))
    if stage is None:
        raise RuntimeError(f"USD refused to create {destination}")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/ReachabilityOverlay")
    stage.SetDefaultPrim(root.GetPrim())

    mesh = UsdGeom.Mesh.Define(stage, "/ReachabilityOverlay/cells")
    mesh.CreatePointsAttr(Vt.Vec3fArray(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4] * (nx * ny)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(face_indices))
    mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform).Set(Vt.Vec3fArray(colors))
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    stage.GetRootLayer().Save()

    return {
        "path": str(destination),
        "prim_path": "/ReachabilityOverlay",
        "cells": nx * ny,
        "points": len(points),
        "legend": {name: list(color) for name, color in LABEL_COLORS.items()},
    }

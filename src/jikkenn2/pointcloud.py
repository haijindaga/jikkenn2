"""Writing point clouds out so a human can look at them.

Every three-dimensional artifact the planner depends on gets an inspectable
form.  jikkenn1's map passed all of its own JSON checks while containing about
0.3 cubic metres of obstacle that was not there, and ten lines like these would
have shown it in seconds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_ascii_ply(path: str | Path, points: np.ndarray, colors: np.ndarray) -> Path:
    """Write a plain PLY that MeshLab, CloudCompare and Viser all read."""
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {points.shape}")
    if colors.shape != points.shape:
        raise ValueError(f"colors {colors.shape} do not match points {points.shape}")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {points.shape[0]}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        stream.write("end_header\n")
        for point, color in zip(points, colors):
            stream.write(
                f"{point[0]:.6g} {point[1]:.6g} {point[2]:.6g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
    return destination


def write_colored_cloud(path: str | Path, points: np.ndarray, color) -> int:
    """Write one cloud in a single colour; returns the point count."""
    points = np.asarray(points, dtype=np.float32)
    colors = np.tile(np.asarray(color, dtype=np.uint8), (len(points), 1))
    write_ascii_ply(path, points, colors)
    return int(len(points))


def subsample(points: np.ndarray, limit: int, *, seed: int = 0) -> np.ndarray:
    """Thin a cloud to at most ``limit`` points, deterministically."""
    points = np.asarray(points)
    if limit <= 0:
        raise ValueError("limit must be positive")
    if len(points) <= limit:
        return points
    picks = np.random.default_rng(seed).permutation(len(points))[:limit]
    return points[np.sort(picks)]


def gripper_marker_points(
    hand_pose,
    *,
    fingertip_depth_m: float,
    half_width_m: float = 0.04,
    samples: int = 16,
) -> np.ndarray:
    """A small T of points showing one grasp: approach line and finger line.

    Drawn rather than described, because "the approach is wrong" is a sentence
    nobody can check and a shape anybody can.
    """
    pose = np.asarray(hand_pose, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"hand_pose must be 4x4, got {pose.shape}")
    if samples < 2:
        raise ValueError("samples must be at least 2")
    wrist = pose[:3, 3]
    approach = pose[:3, 2]
    closing = pose[:3, 1]
    fingertip = wrist + approach * float(fingertip_depth_m)

    along = np.linspace(0.0, float(fingertip_depth_m), samples)
    shaft = wrist + approach * along[:, None]
    across = np.linspace(-half_width_m, half_width_m, samples)
    jaw = fingertip + closing * across[:, None]
    return np.vstack([shaft, jaw]).astype(np.float32)


def gripper_markers(hand_poses, **kwargs) -> np.ndarray:
    """Marker points for many grasps, stacked."""
    poses = np.asarray(hand_poses, dtype=np.float64)
    if len(poses) == 0:
        return np.empty((0, 3), dtype=np.float32)
    return np.vstack([gripper_marker_points(pose, **kwargs) for pose in poses])

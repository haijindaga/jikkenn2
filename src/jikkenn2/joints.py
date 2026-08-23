"""Joint-order translation between the simulator and the planner.

Isaac's articulation order comes from the asset and cuRobo's comes from its
robot config; the two agree only by luck.  Everything crossing that boundary
goes through here, selected by name.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def select_named_joint_positions(
    joint_names: Sequence[str],
    joint_positions: np.ndarray,
    requested_names: Sequence[str],
) -> np.ndarray:
    """Return positions reordered into ``requested_names``.

    Raises rather than padding: a silently missing joint would put the arm in a
    configuration nobody asked for.
    """
    names = tuple(str(name) for name in joint_names)
    requested = tuple(str(name) for name in requested_names)
    positions = np.asarray(joint_positions, dtype=np.float64)
    if positions.shape != (len(names),):
        raise ValueError(
            f"joint_positions shape {positions.shape} does not match {len(names)} names"
        )
    if len(set(names)) != len(names):
        raise ValueError("joint_names must be unique")
    index = {name: position for position, name in enumerate(names)}
    missing = [name for name in requested if name not in index]
    if missing:
        raise KeyError(f"missing joints: {missing}; available: {list(names)}")
    return positions[[index[name] for name in requested]].copy()


def merge_named_joint_positions(
    joint_names: Sequence[str],
    joint_positions: np.ndarray,
    updates: dict[str, float],
) -> np.ndarray:
    """Copy a full joint vector with some joints overwritten by name.

    Used to drive the gripper without disturbing the arm, and vice versa.
    """
    names = tuple(str(name) for name in joint_names)
    merged = np.asarray(joint_positions, dtype=np.float64).copy()
    if merged.shape != (len(names),):
        raise ValueError(
            f"joint_positions shape {merged.shape} does not match {len(names)} names"
        )
    index = {name: position for position, name in enumerate(names)}
    unknown = [name for name in updates if name not in index]
    if unknown:
        raise KeyError(f"unknown joints: {unknown}; available: {list(names)}")
    for name, value in updates.items():
        merged[index[name]] = float(value)
    return merged

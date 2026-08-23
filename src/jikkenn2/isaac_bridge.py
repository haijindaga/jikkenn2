"""Thin wrappers over the Isaac Sim APIs that have moved between releases.

Kept in one place so the guessing happens once.  Every function here says which
classes or methods it tried when it cannot find one, because the alternative is
an import line failing with no clue what to look for.

Isaac itself is imported inside the functions, so this module can be imported
anywhere, including in tests.
"""

from __future__ import annotations

import numpy as np


def make_articulation(prim_path: str, name: str):
    """Wrap an articulation that is already present in the stage."""
    attempts: list[str] = []
    try:
        from isaacsim.core.prims import SingleArticulation

        attempts.append("isaacsim.core.prims.SingleArticulation")
        return SingleArticulation(prim_path=prim_path, name=name)
    except ImportError:
        pass
    try:
        from isaacsim.core.api.robots import Robot

        attempts.append("isaacsim.core.api.robots.Robot")
        return Robot(prim_path=prim_path, name=name)
    except ImportError:
        pass
    raise RuntimeError(
        f"no articulation wrapper found for {prim_path}. "
        f"Tried: {', '.join(attempts) or 'nothing importable'}."
    )


def _target_setter(articulation):
    for setter in ("set_joint_position_targets", "set_joint_positions_target"):
        if hasattr(articulation, setter):
            return setter
    return None


def set_home_configuration(articulation, home) -> str:
    """Put the arm at a pose *and* move the drive targets there.

    Setting positions alone is not enough: the position controller keeps
    pulling toward its own targets, so the targets have to move too.
    """
    positions = np.asarray(home, dtype=np.float32)
    used = []
    if hasattr(articulation, "set_joints_default_state"):
        articulation.set_joints_default_state(positions=positions)
        used.append("set_joints_default_state")
    articulation.set_joint_positions(positions)
    used.append("set_joint_positions")
    setter = _target_setter(articulation)
    if setter is not None:
        getattr(articulation, setter)(positions)
        used.append(setter)
    if len(used) < 2:
        raise RuntimeError(
            "could not pin the home configuration: this articulation exposes only "
            f"{used}. Check the Isaac Sim API for setting drive targets."
        )
    return "+".join(used)


def command_joint_targets(articulation, positions) -> None:
    """Send one position command to every joint, in the articulation's order."""
    setter = _target_setter(articulation)
    if setter is None:
        raise RuntimeError(
            "this articulation exposes no joint position target setter; tried "
            "set_joint_position_targets and set_joint_positions_target"
        )
    getattr(articulation, setter)(np.asarray(positions, dtype=np.float32))


def gripper_width_m(joint_names, joint_positions) -> float:
    """Total opening of the Panda gripper, both fingers summed."""
    names = [str(name) for name in joint_names]
    positions = np.asarray(joint_positions, dtype=np.float64)
    total = 0.0
    found = 0
    for name, value in zip(names, positions):
        if "finger" in name:
            total += float(value)
            found += 1
    if found == 0:
        raise KeyError(f"no finger joints among {names}")
    return total

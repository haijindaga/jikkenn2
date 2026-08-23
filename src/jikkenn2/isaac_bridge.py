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


#: Where ``ArticulationAction`` has lived across releases.
ACTION_MODULES = (
    "isaacsim.core.utils.types",
    "omni.isaac.core.utils.types",
)

#: Setters offered by the batched articulation classes.
DIRECT_TARGET_SETTERS = ("set_joint_position_targets", "set_joint_positions_target")


def _articulation_action_class():
    import importlib

    for module_name in ACTION_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        action = getattr(module, "ArticulationAction", None)
        if action is not None:
            return action
    return None


def make_target_commander(articulation):
    """Return ``(command, method_name)`` for sending joint position targets.

    The single-articulation wrapper takes an ``ArticulationAction`` through
    ``apply_action``; the batched classes expose direct setters instead.  Which
    one is present depends on the release, so resolve it once and report which
    was used rather than assuming.
    """
    tried: list[str] = []

    action_class = _articulation_action_class()
    if action_class is not None and hasattr(articulation, "apply_action"):
        def command(positions):
            articulation.apply_action(
                action_class(joint_positions=np.asarray(positions, dtype=np.float32))
            )

        return command, "apply_action(ArticulationAction)"
    tried.append("apply_action(ArticulationAction)")

    for setter in DIRECT_TARGET_SETTERS:
        if hasattr(articulation, setter):
            def command(positions, setter=setter):
                getattr(articulation, setter)(np.asarray(positions, dtype=np.float32))

            return command, setter
        tried.append(setter)

    if action_class is not None and hasattr(articulation, "get_articulation_controller"):
        controller = articulation.get_articulation_controller()

        def command(positions):
            controller.apply_action(
                action_class(joint_positions=np.asarray(positions, dtype=np.float32))
            )

        return command, "articulation_controller.apply_action"
    tried.append("articulation_controller.apply_action")

    raise RuntimeError(
        "this articulation offers no way to send joint position targets; tried "
        + ", ".join(tried)
        + (
            ". ArticulationAction was not importable either"
            if action_class is None
            else ""
        )
    )


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
    command, method = make_target_commander(articulation)
    command(positions)
    used.append(method)
    return "+".join(used)


def command_joint_targets(articulation, positions) -> None:
    """Send one position command to every joint, in the articulation's order.

    Convenience for a single call; a replay loop should resolve the commander
    once with :func:`make_target_commander`.
    """
    command, _ = make_target_commander(articulation)
    command(positions)


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

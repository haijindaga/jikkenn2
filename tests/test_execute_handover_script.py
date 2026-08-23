"""Tests for the execution schedule that need no simulator."""

from __future__ import annotations

import numpy as np
import pytest

import execute_handover as ex
from jikkenn2.joints import merge_named_joint_positions


ISAAC_NAMES = [f"panda_joint{i}" for i in range(1, 8)] + list(ex.FINGER_JOINTS)
PLAN_COLUMNS = tuple(ISAAC_NAMES)


def apply_row(target, row, finger):
    updates = {
        name: float(value)
        for name, value in zip(PLAN_COLUMNS, row)
        if name not in ex.FINGER_JOINTS
    }
    updates.update({name: finger for name in ex.FINGER_JOINTS})
    return merge_named_joint_positions(ISAAC_NAMES, target, updates)


def test_the_gripper_is_open_up_to_the_grasp_and_closed_after():
    assert ex.LEG_GRIPPER_OPEN["pregrasp"] is True
    assert ex.LEG_GRIPPER_OPEN["grasp"] is True
    assert ex.LEG_GRIPPER_OPEN["lift"] is False
    assert ex.LEG_GRIPPER_OPEN["handover"] is False


def test_every_planned_leg_has_a_gripper_state():
    import plan_handover

    assert set(ex.LEG_GRIPPER_OPEN) == set(plan_handover.SEGMENTS)


def test_finger_columns_from_the_plan_are_ignored():
    """A plan that closes the fingers early must not close them."""
    target = np.zeros(len(ISAAC_NAMES))
    row = np.array([0.1] * 7 + [0.0, 0.0])  # the plan asks for closed fingers
    applied = apply_row(target, row, ex.OPEN_M)
    assert applied[:7] == pytest.approx([0.1] * 7)
    assert applied[7] == pytest.approx(ex.OPEN_M)
    assert applied[8] == pytest.approx(ex.OPEN_M)


def test_arm_columns_are_matched_by_name_not_position():
    target = np.zeros(len(ISAAC_NAMES))
    shuffled = ["panda_joint7", "panda_joint1"] + [
        name for name in ISAAC_NAMES if name not in ("panda_joint1", "panda_joint7")
    ]
    row = [0.7, 0.1] + [0.0] * (len(shuffled) - 2)
    updates = {
        name: float(value)
        for name, value in zip(shuffled, row)
        if name not in ex.FINGER_JOINTS
    }
    applied = merge_named_joint_positions(ISAAC_NAMES, target, updates)
    assert applied[0] == pytest.approx(0.1)   # panda_joint1
    assert applied[6] == pytest.approx(0.7)   # panda_joint7


def test_closed_command_is_fully_closed():
    assert ex.CLOSED_M == pytest.approx(0.0)
    assert ex.OPEN_M == pytest.approx(0.04)

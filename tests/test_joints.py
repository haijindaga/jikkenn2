"""Tests for joint-order translation."""

from __future__ import annotations

import numpy as np
import pytest

from jikkenn2.joints import merge_named_joint_positions, select_named_joint_positions


ISAAC_ORDER = ["panda_joint1", "panda_joint2", "panda_finger_joint1", "panda_joint3"]
POSITIONS = np.array([0.1, 0.2, 0.04, 0.3])


def test_selection_follows_the_requested_order():
    picked = select_named_joint_positions(
        ISAAC_ORDER, POSITIONS, ["panda_joint3", "panda_joint1"]
    )
    assert picked == pytest.approx([0.3, 0.1])


def test_selection_can_drop_joints_the_planner_does_not_use():
    picked = select_named_joint_positions(
        ISAAC_ORDER, POSITIONS, ["panda_joint1", "panda_joint2", "panda_joint3"]
    )
    assert picked == pytest.approx([0.1, 0.2, 0.3])


def test_selection_returns_a_copy():
    picked = select_named_joint_positions(ISAAC_ORDER, POSITIONS, ["panda_joint1"])
    picked[0] = 99.0
    assert POSITIONS[0] == pytest.approx(0.1)


def test_missing_joint_is_an_error_not_a_pad():
    with pytest.raises(KeyError, match="panda_joint7"):
        select_named_joint_positions(ISAAC_ORDER, POSITIONS, ["panda_joint7"])


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="does not match"):
        select_named_joint_positions(ISAAC_ORDER, np.array([0.1, 0.2]), ["panda_joint1"])


def test_duplicate_names_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        select_named_joint_positions(
            ["a", "a"], np.array([0.1, 0.2]), ["a"]
        )


def test_merge_overwrites_only_the_named_joints():
    merged = merge_named_joint_positions(
        ISAAC_ORDER, POSITIONS, {"panda_finger_joint1": 0.0}
    )
    assert merged == pytest.approx([0.1, 0.2, 0.0, 0.3])
    assert POSITIONS == pytest.approx([0.1, 0.2, 0.04, 0.3])


def test_merge_rejects_an_unknown_joint():
    with pytest.raises(KeyError, match="panda_finger_joint2"):
        merge_named_joint_positions(ISAAC_ORDER, POSITIONS, {"panda_finger_joint2": 0.0})


def test_round_trip_through_a_different_order():
    planner_order = ["panda_joint3", "panda_joint2", "panda_joint1"]
    picked = select_named_joint_positions(ISAAC_ORDER, POSITIONS, planner_order)
    back = select_named_joint_positions(planner_order, picked, ISAAC_ORDER[:2])
    assert back == pytest.approx([0.1, 0.2])

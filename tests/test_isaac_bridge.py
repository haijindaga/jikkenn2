"""Tests for the Isaac wrappers that do not need Isaac.

The wrappers exist because these APIs have moved between releases; these tests
pin the parts that are ours: which methods are tried, and the gripper maths.
"""

from __future__ import annotations

import numpy as np
import pytest

from jikkenn2 import isaac_bridge as bridge


class FakeArticulation:
    """Stands in for an Isaac articulation, exposing a chosen set of methods."""

    def __init__(self, *, target_setter="set_joint_position_targets", defaults=True):
        self.dof_names = [
            "panda_joint1",
            "panda_finger_joint1",
            "panda_finger_joint2",
        ]
        self.positions = np.zeros(3)
        self.targets = None
        self.defaults = None
        if defaults:
            self.set_joints_default_state = self._set_defaults
        if target_setter:
            setattr(self, target_setter, self._set_targets)

    def _set_defaults(self, positions=None):
        self.defaults = np.asarray(positions, dtype=np.float64)

    def _set_targets(self, positions):
        self.targets = np.asarray(positions, dtype=np.float64)

    def set_joint_positions(self, positions):
        self.positions = np.asarray(positions, dtype=np.float64)

    def get_joint_positions(self):
        return self.positions


def test_home_sets_state_and_targets():
    articulation = FakeArticulation()
    used = bridge.set_home_configuration(articulation, [0.1, 0.04, 0.04])
    assert articulation.positions == pytest.approx([0.1, 0.04, 0.04])
    assert articulation.targets == pytest.approx([0.1, 0.04, 0.04])
    assert articulation.defaults == pytest.approx([0.1, 0.04, 0.04])
    assert "set_joint_position_targets" in used


def test_home_accepts_the_alternative_target_setter_name():
    articulation = FakeArticulation(target_setter="set_joint_positions_target")
    used = bridge.set_home_configuration(articulation, [0.0, 0.04, 0.04])
    assert "set_joint_positions_target" in used
    assert articulation.targets is not None


def test_home_refuses_when_targets_cannot_be_moved():
    articulation = FakeArticulation(target_setter=None, defaults=False)
    with pytest.raises(RuntimeError, match="drive targets"):
        bridge.set_home_configuration(articulation, [0.0, 0.04, 0.04])


def test_commanding_targets_uses_whichever_setter_exists():
    articulation = FakeArticulation(target_setter="set_joint_positions_target")
    bridge.command_joint_targets(articulation, [0.2, 0.0, 0.0])
    assert articulation.targets == pytest.approx([0.2, 0.0, 0.0])


def test_commanding_targets_refuses_without_a_setter():
    articulation = FakeArticulation(target_setter=None)
    with pytest.raises(RuntimeError, match="no joint position target setter"):
        bridge.command_joint_targets(articulation, [0.0, 0.0, 0.0])


def test_gripper_width_sums_both_fingers():
    names = ["panda_joint1", "panda_finger_joint1", "panda_finger_joint2"]
    assert bridge.gripper_width_m(names, [0.3, 0.04, 0.04]) == pytest.approx(0.08)
    assert bridge.gripper_width_m(names, [0.3, 0.0, 0.0]) == pytest.approx(0.0)


def test_gripper_width_ignores_arm_joints():
    names = ["panda_joint1", "panda_joint2", "panda_finger_joint1", "panda_finger_joint2"]
    assert bridge.gripper_width_m(names, [1.0, 2.0, 0.01, 0.01]) == pytest.approx(0.02)


def test_gripper_width_needs_finger_joints():
    with pytest.raises(KeyError, match="no finger joints"):
        bridge.gripper_width_m(["panda_joint1"], [0.0])

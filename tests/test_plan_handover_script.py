"""Tests for the parts of plan_handover.py that need no GPU."""

from __future__ import annotations

import numpy as np
import pytest

import plan_handover
from jikkenn2.scene_spec import DEFAULT_SCENE


def test_trajectory_reduces_from_any_number_of_leading_axes():
    horizon, dof = 81, 9
    plan = np.arange(horizon * dof, dtype=np.float32).reshape(horizon, dof)
    for shape in [(horizon, dof), (1, horizon, dof), (1, 1, horizon, dof), (1, 1, 1, horizon, dof)]:
        reduced = plan_handover._as_horizon_by_dof(plan.reshape(shape), dof)
        assert reduced.shape == (horizon, dof)
        assert reduced == pytest.approx(plan)


def test_trajectory_with_a_real_batch_is_refused():
    plan = np.zeros((2, 81, 9), dtype=np.float32)
    with pytest.raises(RuntimeError, match="does not reduce"):
        plan_handover._as_horizon_by_dof(plan, 9)


def test_trajectory_with_the_wrong_joint_count_is_refused():
    plan = np.zeros((81, 7), dtype=np.float32)
    with pytest.raises(RuntimeError, match="expected 9"):
        plan_handover._as_horizon_by_dof(plan, 9)


def test_obstacles_take_their_size_from_the_spec_and_pose_from_the_arrangement():
    obstacle = DEFAULT_SCENE.obstacles[0]
    settled = {
        "objects": [
            {
                "prim_path": plan_handover.TOOL_PRIM_PATH,
                "position_m": [0.4, 0.1, 0.02],
                "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
            {
                "prim_path": obstacle.prim_path,
                "position_m": [0.5, -0.3, 0.05],
                "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
        ]
    }
    boxes = plan_handover.world_obstacles(settled, DEFAULT_SCENE)
    assert len(boxes) == 1, "the tool must not become an obstacle"
    assert boxes[0]["name"] == obstacle.name
    assert boxes[0]["dims"] == list(obstacle.size_m)
    assert boxes[0]["pose"][:3] == [0.5, -0.3, 0.05]


def test_an_object_with_no_declared_size_is_refused():
    settled = {
        "objects": [
            {
                "prim_path": "/World/Obstacles/mystery",
                "position_m": [0.5, 0.0, 0.05],
                "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            }
        ]
    }
    with pytest.raises(KeyError, match="mystery"):
        plan_handover.world_obstacles(settled, DEFAULT_SCENE)


def test_world_to_robot_base_is_identity_when_the_base_is_at_the_origin():
    pose = np.eye(4)
    pose[:3, 3] = [0.4, -0.2, 0.3]
    assert plan_handover.to_robot_base(pose, np.eye(4)) == pytest.approx(pose)


def test_world_to_robot_base_undoes_a_shifted_base():
    base = np.eye(4)
    base[:3, 3] = [0.0, 0.0, 0.5]
    pose = np.eye(4)
    pose[:3, 3] = [0.4, -0.2, 0.9]
    assert plan_handover.to_robot_base(pose, base)[:3, 3] == pytest.approx([0.4, -0.2, 0.4])


def test_segments_are_in_the_order_the_trial_happens():
    assert plan_handover.SEGMENTS == ("pregrasp", "grasp", "lift", "handover")


class _Plan:
    def __init__(self, names):
        self.joint_names = names


def test_trajectory_columns_take_the_plans_own_names():
    names = [f"panda_joint{i}" for i in range(1, 8)] + [
        "panda_finger_joint1",
        "panda_finger_joint2",
    ]
    resolved = plan_handover.trajectory_joint_names(_Plan(names), 9, names[:7])
    assert list(resolved) == names


def test_a_nameless_plan_falls_back_only_when_the_widths_agree():
    planner_names = [f"panda_joint{i}" for i in range(1, 8)]
    resolved = plan_handover.trajectory_joint_names(object(), 7, planner_names)
    assert list(resolved) == planner_names


def test_a_nameless_plan_of_unknown_width_is_refused():
    planner_names = [f"panda_joint{i}" for i in range(1, 8)]
    with pytest.raises(RuntimeError, match="carries no joint names"):
        plan_handover.trajectory_joint_names(object(), 9, planner_names)


def test_mismatched_name_count_is_refused():
    with pytest.raises(RuntimeError, match="joint names for"):
        plan_handover.trajectory_joint_names(_Plan(["a", "b"]), 9, ["a"])

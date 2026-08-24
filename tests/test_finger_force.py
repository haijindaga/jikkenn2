"""Tests for raising the gripper's force limit over the referenced asset."""

from __future__ import annotations

import pytest

pxr = pytest.importorskip("pxr", reason="usd-core is not installed")

from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

import build_scene_usd  # noqa: E402
from jikkenn2.scene_spec import DEFAULT_SCENE  # noqa: E402


ASSET_FORCE = 7.2


@pytest.fixture
def stage_with_fingers(tmp_path):
    stage = Usd.Stage.CreateNew(str(tmp_path / "robot.usda"))
    UsdGeom.Xform.Define(stage, "/World/Panda")
    UsdGeom.Xform.Define(stage, "/World/Panda/panda_hand")
    for index in (1, 2):
        prim = UsdGeom.Xform.Define(
            stage, f"/World/Panda/panda_hand/panda_finger_joint{index}"
        ).GetPrim()
        drive = UsdPhysics.DriveAPI.Apply(prim, "linear")
        drive.CreateMaxForceAttr(ASSET_FORCE)
        drive.CreateStiffnessAttr(400.0)
    # An arm joint that must be left alone.
    arm = UsdGeom.Xform.Define(stage, "/World/Panda/panda_joint4").GetPrim()
    UsdPhysics.DriveAPI.Apply(arm, "angular").CreateMaxForceAttr(87.0)
    return stage


def test_the_real_force_is_an_order_above_what_the_asset_ships():
    assert DEFAULT_SCENE.gripper_max_force_n > 10 * ASSET_FORCE * 0.9
    assert DEFAULT_SCENE.gripper_max_force_n == pytest.approx(70.0)


def test_both_finger_drives_are_raised(stage_with_fingers):
    overridden = build_scene_usd.override_finger_force(
        pxr, stage_with_fingers, "/World/Panda", 70.0
    )
    assert len(overridden) == 2
    for entry in overridden.values():
        assert entry["asset_value_n"] == pytest.approx(ASSET_FORCE)
        assert entry["set_to_n"] == pytest.approx(70.0)


def test_the_new_value_is_actually_on_the_stage(stage_with_fingers):
    build_scene_usd.override_finger_force(pxr, stage_with_fingers, "/World/Panda", 70.0)
    prim = stage_with_fingers.GetPrimAtPath(
        "/World/Panda/panda_hand/panda_finger_joint1"
    )
    drive = UsdPhysics.DriveAPI.Get(prim, "linear")
    assert drive.GetMaxForceAttr().Get() == pytest.approx(70.0)
    # Everything else about the drive is left as the asset had it.
    assert drive.GetStiffnessAttr().Get() == pytest.approx(400.0)


def test_arm_joints_are_left_alone(stage_with_fingers):
    build_scene_usd.override_finger_force(pxr, stage_with_fingers, "/World/Panda", 70.0)
    arm = UsdPhysics.DriveAPI.Get(
        stage_with_fingers.GetPrimAtPath("/World/Panda/panda_joint4"), "angular"
    )
    assert arm.GetMaxForceAttr().Get() == pytest.approx(87.0)


def test_a_robot_without_finger_drives_is_reported(tmp_path):
    stage = Usd.Stage.CreateNew(str(tmp_path / "bare.usda"))
    UsdGeom.Xform.Define(stage, "/World/Panda")
    with pytest.raises(RuntimeError, match="no finger drive"):
        build_scene_usd.override_finger_force(pxr, stage, "/World/Panda", 70.0)


def test_a_missing_robot_is_reported(tmp_path):
    stage = Usd.Stage.CreateNew(str(tmp_path / "empty.usda"))
    UsdGeom.Xform.Define(stage, "/World")
    with pytest.raises(RuntimeError, match="not in the stage"):
        build_scene_usd.override_finger_force(pxr, stage, "/World/Panda", 70.0)

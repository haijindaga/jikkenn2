"""Tests for the authored USD stage.

These run against ``usd-core`` on any machine — no Isaac Sim, no GPU — so the
stage geometry is checked at development time rather than on the workstation.
"""

from __future__ import annotations

import math

import pytest

pxr = pytest.importorskip("pxr", reason="usd-core is not installed")

from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

import build_scene_usd  # noqa: E402
from jikkenn2.scene_spec import DEFAULT_SCENE  # noqa: E402


@pytest.fixture(scope="module")
def authored(tmp_path_factory):
    output = tmp_path_factory.mktemp("stage") / "scene.usd"
    built = build_scene_usd.author_stage(pxr, DEFAULT_SCENE, output, None)
    return output, built


def _xform_ops(prim) -> dict:
    return {op.GetOpName(): op.Get() for op in UsdGeom.Xformable(prim).GetOrderedXformOps()}


def test_verification_passes(authored):
    output, _ = authored
    report = build_scene_usd.verify_stage(pxr, output, DEFAULT_SCENE, expect_robot=False)
    assert report["status"] == "success", report


def test_verification_notices_a_missing_robot(authored):
    output, _ = authored
    report = build_scene_usd.verify_stage(pxr, output, DEFAULT_SCENE, expect_robot=True)
    assert report["status"] == "failure"
    assert "/World/Panda" in report["missing_prims"]


def test_camera_focal_length_reproduces_the_requested_fov(authored):
    output, _ = authored
    stage = Usd.Stage.Open(str(output))
    camera = UsdGeom.Camera(stage.GetPrimAtPath("/World/camera_0"))
    focal = camera.GetFocalLengthAttr().Get()
    aperture = camera.GetHorizontalApertureAttr().Get()
    recovered = math.degrees(2.0 * math.atan(aperture / (2.0 * focal)))
    assert recovered == pytest.approx(DEFAULT_SCENE.camera_horizontal_fov_deg, abs=1e-6)


def test_camera_aperture_matches_the_sensor_aspect_ratio(authored):
    output, _ = authored
    stage = Usd.Stage.Open(str(output))
    camera = UsdGeom.Camera(stage.GetPrimAtPath("/World/camera_0"))
    width, height = DEFAULT_SCENE.camera_resolution_px
    ratio = camera.GetVerticalApertureAttr().Get() / camera.GetHorizontalApertureAttr().Get()
    assert ratio == pytest.approx(height / width)


def test_camera_clipping_range_matches_the_spec(authored):
    output, _ = authored
    stage = Usd.Stage.Open(str(output))
    camera = UsdGeom.Camera(stage.GetPrimAtPath("/World/camera_0"))
    near, far = camera.GetClippingRangeAttr().Get()
    assert (near, far) == pytest.approx(DEFAULT_SCENE.camera_clip_m)


def test_table_matches_the_spec(authored):
    output, _ = authored
    stage = Usd.Stage.Open(str(output))
    ops = _xform_ops(stage.GetPrimAtPath("/World/Table"))
    assert tuple(ops["xformOp:translate"]) == pytest.approx(DEFAULT_SCENE.table_center_m)
    assert tuple(ops["xformOp:scale"]) == pytest.approx(DEFAULT_SCENE.table_size_m)


def test_table_is_static_and_the_tool_is_not(authored):
    output, _ = authored
    stage = Usd.Stage.Open(str(output))
    table = stage.GetPrimAtPath("/World/Table")
    tool = stage.GetPrimAtPath("/World/Tools/proxy_tool")
    assert table.HasAPI(UsdPhysics.CollisionAPI)
    assert not table.HasAPI(UsdPhysics.RigidBodyAPI)
    assert tool.HasAPI(UsdPhysics.RigidBodyAPI)


def test_every_obstacle_is_a_movable_rigid_body(authored):
    output, _ = authored
    stage = Usd.Stage.Open(str(output))
    obstacles = list(stage.GetPrimAtPath("/World/Obstacles").GetChildren())
    assert len(obstacles) == len(build_scene_usd.OBSTACLE_LAYOUT)
    for prim in obstacles:
        assert prim.HasAPI(UsdPhysics.RigidBodyAPI)
        assert prim.HasAPI(UsdPhysics.CollisionAPI)


def test_tool_parts_keep_their_local_geometry(authored):
    output, _ = authored
    stage = Usd.Stage.Open(str(output))
    for part in DEFAULT_SCENE.tool_parts:
        ops = _xform_ops(stage.GetPrimAtPath(f"/World/Tools/proxy_tool/{part.name}"))
        assert tuple(ops["xformOp:translate"]) == pytest.approx(part.center_m)
        assert tuple(ops["xformOp:scale"]) == pytest.approx(part.size_m)


def test_tool_is_authored_resting_on_the_tabletop(authored):
    output, built = authored
    stage = Usd.Stage.Open(str(output))
    ops = _xform_ops(stage.GetPrimAtPath("/World/Tools/proxy_tool"))
    tallest = max(part.size_m[2] for part in DEFAULT_SCENE.tool_parts)
    lowest_z = ops["xformOp:translate"][2] - 0.5 * tallest
    assert lowest_z == pytest.approx(DEFAULT_SCENE.table_top_z_m, abs=1e-9)
    assert built["tool_origin_m"][2] > DEFAULT_SCENE.table_top_z_m


def test_task_markers_are_guides_so_they_never_collide(authored):
    output, _ = authored
    stage = Usd.Stage.Open(str(output))
    for name in ("handover", "human_point"):
        prim = stage.GetPrimAtPath(f"/World/Markers/{name}")
        purpose = UsdGeom.Imageable(prim).GetPurposeAttr().Get()
        assert purpose == UsdGeom.Tokens.guide
        assert not prim.HasAPI(UsdPhysics.CollisionAPI)


def test_gravity_points_down(authored):
    output, _ = authored
    stage = Usd.Stage.Open(str(output))
    scene = UsdPhysics.Scene(stage.GetPrimAtPath("/World/PhysicsScene"))
    assert tuple(scene.GetGravityDirectionAttr().Get()) == pytest.approx((0.0, 0.0, -1.0))
    assert scene.GetGravityMagnitudeAttr().Get() == pytest.approx(9.81)


def test_placing_a_double_precision_prim_does_not_raise(tmp_path):
    """Regression: the Isaac Franka carries a double-precision xformOp:orient.

    Authoring a second, single-precision op on top of it raises
    Tf.ErrorException, which aborted the first workstation run.
    """
    from pxr import Gf, Sdf

    stage = Usd.Stage.CreateNew(str(tmp_path / "referenced.usda"))
    xform = UsdGeom.Xform.Define(stage, "/Ref")
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1.0, 2.0, 3.0))
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))

    build_scene_usd._place_referenced_prim(pxr, xform.GetPrim(), (0.0, 0.0, 0.0))

    ops = {op.GetOpName(): op for op in UsdGeom.Xformable(xform.GetPrim()).GetOrderedXformOps()}
    assert tuple(ops["xformOp:translate"].Get()) == pytest.approx((0.0, 0.0, 0.0))
    # The asset's own orientation is left untouched, at its own precision.
    assert ops["xformOp:orient"].GetPrecision() == UsdGeom.XformOp.PrecisionDouble
    assert Sdf.Path("/Ref").pathString == "/Ref"


def test_placing_a_prim_without_ops_adds_a_translate(tmp_path):
    from pxr import Gf

    stage = Usd.Stage.CreateNew(str(tmp_path / "bare.usda"))
    xform = UsdGeom.Xform.Define(stage, "/Bare")
    build_scene_usd._place_referenced_prim(pxr, xform.GetPrim(), (0.1, 0.2, 0.3))
    ops = {op.GetOpName(): op for op in UsdGeom.Xformable(xform.GetPrim()).GetOrderedXformOps()}
    assert tuple(ops["xformOp:translate"].Get()) == pytest.approx((0.1, 0.2, 0.3))
    assert Gf.Vec3d(0.1, 0.2, 0.3)  # value type is double, as authored


def test_tool_annotation_matches_the_spec(tmp_path):
    import json

    path = tmp_path / "proxy_tool.json"
    build_scene_usd.write_tool_annotation(DEFAULT_SCENE, path)
    annotation = json.loads(path.read_text(encoding="utf-8"))
    assert annotation["danger_part_name"] == DEFAULT_SCENE.danger_part_name
    assert annotation["grasp_part_name"] == DEFAULT_SCENE.grasp_part_name
    assert set(annotation["parts"]) == {part.name for part in DEFAULT_SCENE.tool_parts}
    for part in DEFAULT_SCENE.tool_parts:
        assert annotation["parts"][part.name]["size_m"] == list(part.size_m)
    assert "evaluation ground truth only" in annotation["usage"]

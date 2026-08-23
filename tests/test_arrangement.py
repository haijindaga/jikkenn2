"""Tests for reading, checking, saving and restoring hand-made placements."""

from __future__ import annotations

import json

import pytest

pxr = pytest.importorskip("pxr", reason="usd-core is not installed")

from pxr import Gf, Usd, UsdGeom  # noqa: E402

import build_scene_usd  # noqa: E402
from jikkenn2 import arrangement as arr  # noqa: E402
from jikkenn2.scene_spec import DEFAULT_SCENE  # noqa: E402


@pytest.fixture
def stage(tmp_path):
    output = tmp_path / "scene.usd"
    build_scene_usd.author_stage(pxr, DEFAULT_SCENE, output, None)
    return Usd.Stage.Open(str(output))


def _move(stage, prim_path, position, quaternion_wxyz=(1.0, 0.0, 0.0, 0.0)):
    arr.apply_arrangement(
        stage,
        {
            "schema_version": arr.SCHEMA_VERSION,
            "objects": [
                {
                    "prim_path": prim_path,
                    "position_m": list(position),
                    "orientation_wxyz": list(quaternion_wxyz),
                }
            ],
        },
    )


def test_movable_prims_are_the_tool_and_the_obstacles(stage):
    paths = arr.iter_movable_prim_paths(stage)
    assert "/World/Tools/proxy_tool" in paths
    assert len(paths) == 1 + len(build_scene_usd.OBSTACLE_LAYOUT)
    assert all(path.startswith(arr.MOVABLE_ROOTS) for path in paths)
    assert "/World/Table" not in paths
    assert "/World/camera_0" not in paths


def test_read_pose_returns_the_authored_position(stage):
    placed = arr.read_pose(stage, "/World/Tools/proxy_tool")
    expected_z = DEFAULT_SCENE.table_top_z_m + 0.5 * max(
        part.size_m[2] for part in DEFAULT_SCENE.tool_parts
    )
    assert placed.position_m == pytest.approx(
        (
            DEFAULT_SCENE.tool_initial_position_m[0],
            DEFAULT_SCENE.tool_initial_position_m[1],
            expected_z,
        )
    )
    assert placed.orientation_wxyz == pytest.approx((1.0, 0.0, 0.0, 0.0))


def test_read_pose_rejects_a_missing_prim(stage):
    with pytest.raises(KeyError):
        arr.read_pose(stage, "/World/Tools/nope")


def test_default_stage_collects_cleanly(stage):
    snapshot = arr.collect_arrangement(stage, DEFAULT_SCENE, source_stage="scene.usd")
    assert snapshot["schema_version"] == arr.SCHEMA_VERSION
    assert len(snapshot["objects"]) == 1 + len(build_scene_usd.OBSTACLE_LAYOUT)
    assert snapshot["settled"] is False
    tool = next(
        entry
        for entry in snapshot["objects"]
        if entry["prim_path"] == "/World/Tools/proxy_tool"
    )
    assert tool["placement"]["over_the_table"]
    assert tool["placement"]["in_working_band"]
    assert tool["placement"]["visible_to_the_camera"]


def test_object_dragged_off_the_table_is_flagged(stage):
    _move(stage, "/World/Tools/proxy_tool", (1.60, 0.0, 0.02))
    snapshot = arr.collect_arrangement(stage, DEFAULT_SCENE, source_stage="scene.usd")
    assert snapshot["status"] == "placement_warning"
    assert "/World/Tools/proxy_tool" in snapshot["objects_outside_the_working_area"]


def test_object_dragged_too_close_to_the_base_is_flagged(stage):
    _move(stage, "/World/Obstacles/obstacle_a", (0.22, 0.0, 0.05))
    snapshot = arr.collect_arrangement(stage, DEFAULT_SCENE, source_stage="scene.usd")
    flagged = snapshot["objects_outside_the_working_area"]
    assert "/World/Obstacles/obstacle_a" in flagged


def test_placement_description_names_the_problem(stage):
    _move(stage, "/World/Tools/proxy_tool", (1.60, 0.0, 0.02))
    snapshot = arr.collect_arrangement(stage, DEFAULT_SCENE, source_stage="scene.usd")
    tool = next(
        entry
        for entry in snapshot["objects"]
        if entry["prim_path"] == "/World/Tools/proxy_tool"
    )
    line = arr.describe_placement(tool)
    assert "proxy_tool" in line
    assert "WARN" in line
    assert "off table" in line


def test_round_trip_through_save_load_apply(stage, tmp_path):
    position = (0.52, -0.14, 0.06)
    quaternion = (0.9238795, 0.0, 0.0, 0.3826834)  # 45 deg about +z
    _move(stage, "/World/Tools/proxy_tool", position, quaternion)

    snapshot = arr.collect_arrangement(stage, DEFAULT_SCENE, source_stage="scene.usd")
    path = arr.save_arrangement(snapshot, arr.next_arrangement_path(tmp_path))
    loaded = arr.load_arrangement(path)

    _move(stage, "/World/Tools/proxy_tool", (0.0, 0.0, 1.0))
    arr.apply_arrangement(stage, loaded)

    restored = arr.read_pose(stage, "/World/Tools/proxy_tool")
    assert restored.position_m == pytest.approx(position, abs=1e-9)
    # The literal above is truncated; USD renormalises, so compare loosely.
    assert restored.orientation_wxyz == pytest.approx(quaternion, abs=1e-6)


def test_apply_rejects_a_prim_that_is_not_in_the_stage(stage):
    with pytest.raises(KeyError):
        arr.apply_arrangement(
            stage,
            {
                "schema_version": arr.SCHEMA_VERSION,
                "objects": [
                    {
                        "prim_path": "/World/Tools/ghost",
                        "position_m": [0.0, 0.0, 0.0],
                        "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
                    }
                ],
            },
        )


def test_apply_works_on_a_double_precision_prim(tmp_path):
    """Referenced assets carry double-precision ops; restoring must not raise."""
    stage = Usd.Stage.CreateNew(str(tmp_path / "ref.usda"))
    UsdGeom.Xform.Define(stage, "/World/Tools")
    xform = UsdGeom.Xform.Define(stage, "/World/Tools/referenced")
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1.0, 1.0, 1.0))
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))

    arr.apply_arrangement(
        stage,
        {
            "schema_version": arr.SCHEMA_VERSION,
            "objects": [
                {
                    "prim_path": "/World/Tools/referenced",
                    "position_m": [0.4, 0.1, 0.05],
                    "orientation_wxyz": [0.7071068, 0.0, 0.0, 0.7071068],
                }
            ],
        },
    )
    restored = arr.read_pose(stage, "/World/Tools/referenced")
    assert restored.position_m == pytest.approx((0.4, 0.1, 0.05))
    assert restored.orientation_wxyz == pytest.approx(
        (0.7071068, 0.0, 0.0, 0.7071068), abs=1e-6
    )


def test_arrangement_numbering_increments(tmp_path):
    first = arr.next_arrangement_path(tmp_path)
    assert first.name == "arr_001.json"
    first.write_text("{}", encoding="utf-8")
    assert arr.next_arrangement_path(tmp_path).name == "arr_002.json"
    (tmp_path / "arr_017.json").write_text("{}", encoding="utf-8")
    assert arr.next_arrangement_path(tmp_path).name == "arr_018.json"


def test_unrelated_json_does_not_shift_the_numbering(tmp_path):
    (tmp_path / "notes.json").write_text("{}", encoding="utf-8")
    (tmp_path / "arr_x.json").write_text("{}", encoding="utf-8")
    assert arr.next_arrangement_path(tmp_path).name == "arr_001.json"


def test_load_rejects_an_unknown_schema(tmp_path):
    path = tmp_path / "arr_001.json"
    path.write_text(json.dumps({"schema_version": 99, "objects": [{}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        arr.load_arrangement(path)


def test_load_rejects_a_malformed_pose(tmp_path):
    path = tmp_path / "arr_001.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": arr.SCHEMA_VERSION,
                "objects": [{"prim_path": "/World/Tools/x", "position_m": [0.0, 0.0]}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="position_m"):
        arr.load_arrangement(path)


def test_collect_refuses_a_stage_with_nothing_movable(tmp_path):
    stage = Usd.Stage.CreateNew(str(tmp_path / "bare.usda"))
    UsdGeom.Xform.Define(stage, "/World")
    with pytest.raises(ValueError, match="no movable prims"):
        arr.collect_arrangement(stage, DEFAULT_SCENE, source_stage="bare.usda")

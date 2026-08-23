#!/usr/bin/env python3
"""Author ``assets/scene.usd`` — the arrangeable Phase 0 stage.

The saved stage is the single source of truth for the scene.  Nothing
downstream hard-codes object positions: the tool and the obstacles are rigid
bodies that a human drags in the Isaac Sim GUI, and ``arrange_scene.py`` records
where they ended up.

Everything except the Franka reference is authored with plain USD, so the file
is small and re-openable.  Isaac Sim is started only to resolve the asset root
for the robot.

    conda activate env_isaaclab
    python scripts/build_scene_usd.py --output assets/scene.usd

Then check the result before trusting it:

    python scripts/validate_scene.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from jikkenn2.scene_spec import DEFAULT_SCENE, SceneSpec  # noqa: E402


# Isaac Sim moved the manipulator assets between releases.  Candidates are tried
# in order; the first one that opens wins.  Override with --franka-usd.
FRANKA_CANDIDATE_PATHS = (
    "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
    "/Isaac/Robots/Franka/franka.usd",
    "/Isaac/Robots/Franka/franka_instanceable.usd",
)

OBSTACLE_LAYOUT = (
    ("obstacle_a", (0.45, -0.18, 0.05), (0.10, 0.10, 0.10), (0.85, 0.25, 0.15)),
    ("obstacle_b", (0.62, 0.28, 0.06), (0.08, 0.08, 0.12), (0.90, 0.55, 0.15)),
    ("obstacle_c", (0.35, 0.35, 0.04), (0.12, 0.12, 0.08), (0.75, 0.35, 0.20)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "assets" / "scene.usd")
    parser.add_argument(
        "--tool-annotation",
        type=Path,
        default=REPO_ROOT / "assets" / "tools" / "proxy_tool.json",
    )
    parser.add_argument(
        "--franka-usd",
        help="Full URL of the Franka USD; skips asset-root discovery",
    )
    parser.add_argument(
        "--no-isaac",
        action="store_true",
        help="Author the stage without the robot reference (geometry smoke test only)",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------
# USD authoring helpers
# ----------------------------------------------------------------------
def _set_transform(pxr, prim, translation, *, scale=None) -> None:
    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*(float(v) for v in translation)))
    xform.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    if scale is not None:
        xform.AddScaleOp().Set(Gf.Vec3f(*(float(v) for v in scale)))


def _add_box(pxr, stage, path, *, size_m, translation, color, collision=True):
    """Add a unit cube scaled to ``size_m``; returns the prim."""
    from pxr import Gf, UsdGeom, UsdPhysics, Vt

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateExtentAttr(
        Vt.Vec3fArray([Gf.Vec3f(-0.5, -0.5, -0.5), Gf.Vec3f(0.5, 0.5, 0.5)])
    )
    cube.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    _set_transform(pxr, cube.GetPrim(), translation, scale=size_m)
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube.GetPrim()


def _make_rigid_body(pxr, prim, *, mass_kg: float) -> None:
    from pxr import UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(float(mass_kg))


def author_stage(pxr, scene: SceneSpec, output: Path, franka_url: str | None) -> dict:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    physics_scene.CreateGravityMagnitudeAttr(9.81)

    # --- room ---------------------------------------------------------
    ground = UsdGeom.Plane.Define(stage, "/World/Ground")
    ground.CreateAxisAttr(UsdGeom.Tokens.z)
    ground.CreateWidthAttr(20.0)
    ground.CreateLengthAttr(20.0)
    ground.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.22, 0.23, 0.25)]))
    _set_transform(pxr, ground.GetPrim(), (0.0, 0.0, scene.ground_z_m))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

    light = UsdGeom.Xform.Define(stage, "/World/Lights")
    dome = stage.DefinePrim("/World/Lights/Dome", "DomeLight")
    dome.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(1200.0)

    # --- table (static) ------------------------------------------------
    _add_box(
        pxr,
        stage,
        "/World/Table",
        size_m=scene.table_size_m,
        translation=scene.table_center_m,
        color=(0.45, 0.32, 0.20),
    )

    # --- proxy tool: one rigid body, two labelled boxes -----------------
    tool_root = UsdGeom.Xform.Define(stage, "/World/Tools/proxy_tool")
    tallest = max(part.size_m[2] for part in scene.tool_parts)
    tool_origin = (
        scene.tool_initial_position_m[0],
        scene.tool_initial_position_m[1],
        scene.table_top_z_m + 0.5 * tallest,
    )
    _set_transform(pxr, tool_root.GetPrim(), tool_origin)
    _make_rigid_body(pxr, tool_root.GetPrim(), mass_kg=0.5)
    part_colors = {
        scene.danger_part_name: (0.85, 0.20, 0.15),
        scene.safe_part_name: (0.20, 0.55, 0.85),
    }
    for part in scene.tool_parts:
        _add_box(
            pxr,
            stage,
            f"/World/Tools/proxy_tool/{part.name}",
            size_m=part.size_m,
            translation=part.center_m,
            color=part_colors.get(part.name, (0.6, 0.6, 0.6)),
        )

    # --- movable obstacles ---------------------------------------------
    UsdGeom.Xform.Define(stage, "/World/Obstacles")
    for name, position, size, color in OBSTACLE_LAYOUT:
        prim = _add_box(
            pxr,
            stage,
            f"/World/Obstacles/{name}",
            size_m=size,
            translation=position,
            color=color,
        )
        _make_rigid_body(pxr, prim, mass_kg=0.3)

    # --- camera ---------------------------------------------------------
    camera = UsdGeom.Camera.Define(stage, "/World/camera_0")
    width, height = scene.camera_resolution_px
    horizontal_aperture_mm = 20.955
    import math

    focal_mm = horizontal_aperture_mm / (
        2.0 * math.tan(math.radians(scene.camera_horizontal_fov_deg) / 2.0)
    )
    camera.CreateHorizontalApertureAttr(horizontal_aperture_mm)
    camera.CreateVerticalApertureAttr(horizontal_aperture_mm * height / width)
    camera.CreateFocalLengthAttr(focal_mm)
    camera.CreateClippingRangeAttr(Gf.Vec2f(*scene.camera_clip_m))
    # Pose is authored by capture_scene.py from scene_spec, which owns the
    # look-at convention; the translate op here only places it for the GUI.
    _set_transform(pxr, camera.GetPrim(), scene.camera_position_m)

    # --- task markers (visualisation only, no physics) -------------------
    markers = UsdGeom.Xform.Define(stage, "/World/Markers")
    markers.GetPrim().SetMetadata("comment", "task points; no collision, no physics")
    for name, position, color in (
        ("handover", scene.handover_position_m, (0.95, 0.45, 0.10)),
        ("human_point", scene.human_point_m, (0.95, 0.95, 0.95)),
    ):
        sphere = UsdGeom.Sphere.Define(stage, f"/World/Markers/{name}")
        sphere.CreateRadiusAttr(0.03)
        sphere.CreateExtentAttr(
            Vt.Vec3fArray([Gf.Vec3f(-0.03, -0.03, -0.03), Gf.Vec3f(0.03, 0.03, 0.03)])
        )
        sphere.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
        UsdGeom.Imageable(sphere).CreatePurposeAttr(UsdGeom.Tokens.guide)
        _set_transform(pxr, sphere.GetPrim(), position)

    # --- robot -----------------------------------------------------------
    robot_referenced = False
    if franka_url is not None:
        robot = UsdGeom.Xform.Define(stage, "/World/Panda")
        robot.GetPrim().GetReferences().AddReference(franka_url)
        _set_transform(pxr, robot.GetPrim(), scene.robot_base_position_m)
        robot_referenced = True

    stage.GetRootLayer().Save()
    return {
        "output": str(output),
        "franka_reference": franka_url,
        "robot_referenced": robot_referenced,
        "tool_origin_m": [round(float(v), 4) for v in tool_origin],
        "obstacle_count": len(OBSTACLE_LAYOUT),
    }


def verify_stage(pxr, output: Path, scene: SceneSpec, expect_robot: bool) -> dict:
    """Re-open the saved file and confirm it contains what we think it does."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(output))
    required = [
        "/World",
        "/World/PhysicsScene",
        "/World/Ground",
        "/World/Table",
        "/World/Tools/proxy_tool",
        "/World/Obstacles",
        "/World/camera_0",
        "/World/Markers/handover",
    ]
    required += [f"/World/Tools/proxy_tool/{p.name}" for p in scene.tool_parts]
    if expect_robot:
        required.append("/World/Panda")

    missing = [path for path in required if not stage.GetPrimAtPath(path).IsValid()]
    up_axis = UsdGeom.GetStageUpAxis(stage)
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    checks = {
        "every_required_prim_exists": not missing,
        "stage_is_z_up": up_axis == UsdGeom.Tokens.z,
        "stage_is_in_metres": abs(meters_per_unit - 1.0) < 1e-9,
        "default_prim_is_world": stage.GetDefaultPrim().GetPath().pathString == "/World",
    }
    return {
        "status": "success" if all(checks.values()) else "failure",
        "missing_prims": missing,
        "up_axis": str(up_axis),
        "meters_per_unit": meters_per_unit,
        "automatic_checks": checks,
    }


def write_tool_annotation(scene: SceneSpec, path: Path) -> dict:
    """Emit the evaluation-only part annotation used by ``score_trial.py``."""
    annotation = {
        "reference": "generated from src/jikkenn2/scene_spec.py; do not hand-edit",
        "prim_path": "/World/Tools/proxy_tool",
        "frame": "tool local; origin at the head/handle junction, +x toward the head",
        "usage": "evaluation ground truth only; perception must never read this",
        "danger_part_name": scene.danger_part_name,
        "safe_part_name": scene.safe_part_name,
        "grasp_part_name": scene.grasp_part_name,
        "parts": {
            part.name: {
                "size_m": list(part.size_m),
                "center_m": list(part.center_m),
                "axis_local": list(part.axis_local),
                "aabb_local_m": [list(corner) for corner in part.aabb_local_m()],
            }
            for part in scene.tool_parts
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(annotation, indent=2) + "\n", encoding="utf-8")
    return annotation


def resolve_franka_url(explicit: str | None) -> str:
    if explicit:
        return explicit
    from isaacsim.storage.native import get_assets_root_path
    from pxr import Usd

    root = get_assets_root_path()
    if root is None:
        raise RuntimeError(
            "Isaac asset root is unavailable. Pass --franka-usd with a full URL, "
            "or check the Omniverse asset connection."
        )
    for candidate in FRANKA_CANDIDATE_PATHS:
        url = f"{root}{candidate}"
        if Usd.Stage.Open(url) is not None:
            return url
    raise RuntimeError(
        "No Franka USD found under "
        f"{root}. Tried: {', '.join(FRANKA_CANDIDATE_PATHS)}. "
        "Browse the asset root in the Isaac Sim Content browser and pass "
        "--franka-usd explicitly."
    )


def main() -> int:
    args = parse_args()
    scene = DEFAULT_SCENE

    layout = scene.validation_report()
    if layout["status"] != "success":
        failed = [n for n, ok in layout["automatic_checks"].items() if not ok]
        raise RuntimeError(
            "refusing to author a stage from an invalid layout; failing checks: "
            + ", ".join(failed)
            + " (run scripts/validate_scene.py)"
        )

    simulation_app = None
    if not args.no_isaac:
        from isaacsim import SimulationApp

        simulation_app = SimulationApp({"headless": True})

    try:
        import pxr

        franka_url = None if args.no_isaac else resolve_franka_url(args.franka_usd)
        built = author_stage(pxr, scene, args.output, franka_url)
        verified = verify_stage(pxr, args.output, scene, expect_robot=franka_url is not None)
        annotation = write_tool_annotation(scene, args.tool_annotation)
    finally:
        if simulation_app is not None:
            simulation_app.close()

    report = {
        "status": verified["status"],
        "scene_layout_status": layout["status"],
        "stage": built,
        "verification": verified,
        "tool_annotation": str(args.tool_annotation),
        "next_step": (
            "python scripts/arrange_scene.py --gui --scene "
            f"{args.output} to place the tool and obstacles"
        ),
    }
    report_path = REPO_ROOT / "outputs" / "scene_build.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"parts: {', '.join(annotation['parts'])}")
    print(f"saved: {args.output}")
    if verified["status"] != "success":
        raise RuntimeError(f"stage verification failed; inspect {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

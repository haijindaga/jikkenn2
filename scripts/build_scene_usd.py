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


# Isaac Sim moved the manipulator assets between releases.  Known layouts are
# probed first; if none exist the asset tree is searched.  Override with
# --franka-usd to skip discovery entirely.
FRANKA_CANDIDATE_PATHS = (
    "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
    "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka_instanceable.usd",
    "/Isaac/Robots/Franka/franka.usd",
    "/Isaac/Robots/Franka/franka_instanceable.usd",
)
ROBOT_SEARCH_ROOT = "/Isaac/Robots"
ROBOT_SEARCH_DEPTH = 3

# Friction. The gripped tool needs enough to be held by finger pressure alone;
# the table and obstacles use ordinary values so nothing slides on its own.
GRIP_STATIC_FRICTION = 1.2
GRIP_DYNAMIC_FRICTION = 1.0
SURFACE_STATIC_FRICTION = 0.6
SURFACE_DYNAMIC_FRICTION = 0.5

# A 200 mm proxy tool. Light enough that a position-controlled gripper holds it
# without needing force control, heavy enough to fall and settle realistically.
TOOL_MASS_KG = 0.25

# Obstacle geometry lives in scene_spec, which owns every metric value.
OBSTACLE_LAYOUT = DEFAULT_SCENE.obstacles


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
    parser.add_argument(
        "--list-assets",
        nargs="?",
        const=ROBOT_SEARCH_ROOT,
        metavar="PATH",
        help="Print the Isaac asset directory at PATH and exit, to locate the robot USD",
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


def _place_referenced_prim(pxr, prim, translation) -> None:
    """Position a prim that already carries xform ops from a referenced asset.

    A referenced asset authors its ops at a precision we do not control — the
    Isaac Franka uses a double-precision ``xformOp:orient`` — and adding a second
    op at a different precision raises.  Reuse whatever op is already there and
    leave the asset's own orientation alone.
    """
    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xformable(prim)
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() != UsdGeom.XformOp.TypeTranslate:
            continue
        if op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
            op.Set(Gf.Vec3d(*(float(v) for v in translation)))
        else:
            op.Set(Gf.Vec3f(*(float(v) for v in translation)))
        return
    xform.AddTranslateOp().Set(Gf.Vec3d(*(float(v) for v in translation)))


def _define_physics_material(pxr, stage, path, *, static, dynamic, restitution):
    """A named friction/restitution material, bound for the physics purpose.

    Left unspecified, PhysX uses its own defaults, and a gripped object slides
    out of the fingers. Friction is a property of the scene, so it is stated
    here rather than discovered as a mystery during execution.
    """
    from pxr import UsdPhysics, UsdShade

    material = UsdShade.Material.Define(stage, path)
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(float(static))
    physics.CreateDynamicFrictionAttr(float(dynamic))
    physics.CreateRestitutionAttr(float(restitution))
    return material


def _bind_physics_material(pxr, prim, material) -> None:
    from pxr import UsdShade

    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    binding.Bind(
        material,
        bindingStrength=UsdShade.Tokens.weakerThanDescendants,
        materialPurpose="physics",
    )


def _add_box(
    pxr, stage, path, *, size_m, translation, color, collision=True, physics_material=None
):
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
        if physics_material is not None:
            _bind_physics_material(pxr, cube.GetPrim(), physics_material)
    return cube.GetPrim()


def _make_rigid_body(pxr, prim, *, mass_kg: float, center_of_mass_m=None) -> None:
    from pxr import Gf, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(float(mass_kg))
    if center_of_mass_m is not None:
        mass_api.CreateCenterOfMassAttr(
            Gf.Vec3f(*(float(v) for v in center_of_mass_m))
        )


def override_finger_force(pxr, stage, root: str, max_force_n: float) -> dict:
    """Raise the gripper's force limit on top of the referenced asset.

    The Isaac Franka ships its finger drives at 7.2 N, about a tenth of the
    hand's published grasping force. Authoring the real figure over the
    reference is a fidelity correction; the previous value is recorded so the
    change is visible rather than assumed.
    """
    from pxr import Usd, UsdPhysics

    prim_root = stage.GetPrimAtPath(root)
    if not prim_root.IsValid():
        raise RuntimeError(f"{root} is not in the stage; cannot set the finger force")
    overridden = {}
    for prim in Usd.PrimRange(prim_root):
        if "finger" not in prim.GetName().lower():
            continue
        for instance in ("linear", "angular"):
            drive = UsdPhysics.DriveAPI.Get(prim, instance)
            if not drive:
                continue
            attribute = drive.GetMaxForceAttr()
            previous = attribute.Get() if attribute else None
            drive.CreateMaxForceAttr(float(max_force_n))
            overridden[f"{prim.GetPath().pathString}:{instance}"] = {
                "asset_value_n": None if previous is None else round(float(previous), 3),
                "set_to_n": float(max_force_n),
            }
    if not overridden:
        raise RuntimeError(
            f"no finger drive found under {root}; the gripper force was not set"
        )
    return overridden


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

    # --- physics materials -----------------------------------------------
    UsdGeom.Xform.Define(stage, "/World/PhysicsMaterials")
    grip_material = _define_physics_material(
        pxr,
        stage,
        "/World/PhysicsMaterials/grip",
        static=GRIP_STATIC_FRICTION,
        dynamic=GRIP_DYNAMIC_FRICTION,
        restitution=0.0,
    )
    surface_material = _define_physics_material(
        pxr,
        stage,
        "/World/PhysicsMaterials/surface",
        static=SURFACE_STATIC_FRICTION,
        dynamic=SURFACE_DYNAMIC_FRICTION,
        restitution=0.0,
    )

    # --- table (static) ------------------------------------------------
    _add_box(
        pxr,
        stage,
        "/World/Table",
        size_m=scene.table_size_m,
        translation=scene.table_center_m,
        color=scene.table_color,
        physics_material=surface_material,
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
    # Put the centre of mass in the part that gets grasped. A hammer or a knife
    # carries most of its mass there; a uniform-density proxy would instead be
    # handle-heavy purely because the handle is long, and would hang and pivot
    # in the fingers for reasons that have nothing to do with the real tool.
    _make_rigid_body(
        pxr,
        tool_root.GetPrim(),
        mass_kg=TOOL_MASS_KG,
        center_of_mass_m=scene.part(scene.grasp_part_name).center_m,
    )
    for part in scene.tool_parts:
        _add_box(
            pxr,
            stage,
            f"/World/Tools/proxy_tool/{part.name}",
            size_m=part.size_m,
            translation=part.center_m,
            color=part.color,
            physics_material=grip_material,
        )

    # --- movable obstacles ---------------------------------------------
    UsdGeom.Xform.Define(stage, "/World/Obstacles")
    for obstacle in scene.obstacles:
        prim = _add_box(
            pxr,
            stage,
            obstacle.prim_path,
            size_m=obstacle.size_m,
            translation=obstacle.initial_position_m,
            color=obstacle.color,
            physics_material=surface_material,
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
    finger_force = None
    if franka_url is not None:
        robot = UsdGeom.Xform.Define(stage, "/World/Panda")
        robot.GetPrim().GetReferences().AddReference(franka_url)
        _place_referenced_prim(pxr, robot.GetPrim(), scene.robot_base_position_m)
        finger_force = override_finger_force(
            pxr, stage, "/World/Panda", scene.gripper_max_force_n
        )
        robot_referenced = True

    stage.GetRootLayer().Save()
    return {
        "output": str(output),
        "physics": {
            "tool_mass_kg": TOOL_MASS_KG,
            "tool_center_of_mass_local_m": list(
                scene.part(scene.grasp_part_name).center_m
            ),
            "grip_friction": [GRIP_STATIC_FRICTION, GRIP_DYNAMIC_FRICTION],
            "surface_friction": [SURFACE_STATIC_FRICTION, SURFACE_DYNAMIC_FRICTION],
        },
        "franka_reference": franka_url,
        "robot_referenced": robot_referenced,
        "finger_force_override": finger_force,
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

    # A reference can carry its own transform, so confirm where the robot
    # actually ended up rather than assuming the authored value took effect.
    robot_position = None
    robot_prim = stage.GetPrimAtPath("/World/Panda")
    if expect_robot and robot_prim.IsValid():
        matrix = UsdGeom.Xformable(robot_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        translation = matrix.ExtractTranslation()
        robot_position = [round(float(value), 6) for value in translation]
        checks["robot_is_at_the_expected_base_position"] = all(
            abs(float(actual) - float(expected)) < 1e-6
            for actual, expected in zip(translation, scene.robot_base_position_m)
        )

    return {
        "status": "success" if all(checks.values()) else "failure",
        "missing_prims": missing,
        "up_axis": str(up_axis),
        "meters_per_unit": meters_per_unit,
        "robot_world_position_m": robot_position,
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


def _url_exists(url: str) -> bool:
    """Existence check that never raises.

    ``Usd.Stage.Open`` throws on a missing URL instead of returning None, and it
    downloads the whole asset just to answer the question.  Prefer the
    Omniverse client stat, and fall back to opening the layer only if the client
    is unavailable.
    """
    try:
        import omni.client
    except ImportError:
        pass
    else:
        try:
            result, _ = omni.client.stat(url)
            return result == omni.client.Result.OK
        except Exception:
            return False
    from pxr import Sdf

    try:
        return Sdf.Layer.FindOrOpen(url) is not None
    except Exception:
        return False


def _list_url(url: str) -> list[tuple[str, bool]]:
    """Return ``(name, is_folder)`` for one asset directory; empty if unreadable."""
    try:
        import omni.client
    except ImportError:
        return []
    try:
        result, entries = omni.client.list(url)
    except Exception:
        return []
    if result != omni.client.Result.OK:
        return []
    listing = []
    for entry in entries:
        name = getattr(entry, "relative_path", "") or ""
        flags = getattr(entry, "flags", 0)
        is_folder = bool(int(flags) & int(getattr(omni.client.ItemFlags, "CAN_HAVE_CHILDREN", 0)))
        listing.append((name, is_folder))
    return sorted(listing)


def _search_for_franka(root: str) -> tuple[str | None, list[str]]:
    """Walk the asset tree for a Franka USD; also return what was seen."""
    seen: list[str] = []
    found: list[str] = []
    frontier = [(f"{root}{ROBOT_SEARCH_ROOT}", 0)]
    while frontier:
        url, depth = frontier.pop(0)
        listing = _list_url(url)
        if not listing:
            continue
        seen.append(url)
        for name, is_folder in listing:
            child = f"{url}/{name}"
            lowered = name.lower()
            if is_folder:
                if depth + 1 <= ROBOT_SEARCH_DEPTH:
                    frontier.append((child, depth + 1))
            elif lowered.startswith("franka") and lowered.endswith(".usd"):
                found.append(child)
    if not found:
        return None, seen
    found.sort(key=lambda url: (0 if url.lower().endswith("/franka.usd") else 1, len(url)))
    return found[0], seen


def _print_asset_listing(relative_path: str) -> int:
    """Print one Isaac asset directory so the robot USD can be located by hand."""
    from isaacsim.storage.native import get_assets_root_path

    root = get_assets_root_path()
    if root is None:
        print("Isaac asset root is unavailable", file=sys.stderr)
        return 1
    url = f"{root}{relative_path}"
    listing = _list_url(url)
    print(f"asset root : {root}")
    print(f"listing    : {url}")
    if not listing:
        print("  (empty or unreadable)")
        return 1
    for name, is_folder in listing:
        print(f"  {'DIR ' if is_folder else 'FILE'}  {name}")
    return 0


def resolve_franka_url(explicit: str | None) -> str:
    if explicit:
        return explicit
    from isaacsim.storage.native import get_assets_root_path

    root = get_assets_root_path()
    if root is None:
        raise RuntimeError(
            "Isaac asset root is unavailable. Pass --franka-usd with a full URL, "
            "or check the Omniverse asset connection."
        )
    print(f"asset root: {root}", flush=True)
    for candidate in FRANKA_CANDIDATE_PATHS:
        url = f"{root}{candidate}"
        if _url_exists(url):
            print(f"franka found at a known path: {url}", flush=True)
            return url

    print("known paths missed; searching the asset tree...", flush=True)
    url, seen = _search_for_franka(root)
    if url is not None:
        print(f"franka found by search: {url}", flush=True)
        return url

    listing = _list_url(f"{root}{ROBOT_SEARCH_ROOT}")
    top_level = ", ".join(name for name, _ in listing) or "(none readable)"
    raise RuntimeError(
        f"""No Franka USD found under {root}{ROBOT_SEARCH_ROOT}.
  tried known paths : {", ".join(FRANKA_CANDIDATE_PATHS)}
  directories read  : {len(seen)}
  top level entries : {top_level}
Run with --list-assets to browse the tree, then pass the full URL with --franka-usd."""
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

    report_path = REPO_ROOT / "outputs" / "scene_build.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    simulation_app = None
    if not args.no_isaac:
        from isaacsim import SimulationApp

        simulation_app = SimulationApp({"headless": True})

    try:
        import pxr

        if args.list_assets is not None:
            return _print_asset_listing(args.list_assets)

        franka_url = None if args.no_isaac else resolve_franka_url(args.franka_usd)
        built = author_stage(pxr, scene, args.output, franka_url)
        verified = verify_stage(pxr, args.output, scene, expect_robot=franka_url is not None)
        annotation = write_tool_annotation(scene, args.tool_annotation)

        # Write and print the report *before* closing Isaac.  SimulationApp runs
        # with fastShutdown, and close() can take the process down with it, so
        # anything emitted after the finally block may never appear.
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
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        print(f"parts: {', '.join(annotation['parts'])}", flush=True)
        print(f"saved: {args.output}", flush=True)
        print(f"BUILD {verified['status'].upper()} -> {report_path}", flush=True)
    except Exception as error:
        # Always leave an artifact behind.  A failure that produces nothing to
        # read is exactly how jikkenn1 lost time.
        import traceback

        failure = {
            "status": "failure",
            "stage": "isaac_asset_resolution_or_usd_authoring",
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "hint": (
                "Run 'python scripts/build_scene_usd.py --list-assets' to browse the "
                "Isaac asset tree, then pass the Franka USD with --franka-usd <url>."
            ),
        }
        report_path.write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        print(f"failure report: {report_path}", file=sys.stderr, flush=True)
        raise
    finally:
        if simulation_app is not None:
            simulation_app.close()

    if verified["status"] != "success":
        raise RuntimeError(f"stage verification failed; inspect {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

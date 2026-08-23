#!/usr/bin/env python3
"""Restore an arrangement, let physics settle it, and capture the trial inputs.

This is the only place a trial reads the simulator.  It writes the RGB-D frame,
the camera calibration, the Panda's joint state, and the ground-truth object
poses that Phase 0 uses in place of perception.

    conda activate env_isaaclab
    python scripts/capture_scene.py --scene assets/scene.usd \
           --arrangement arrangements/arr_001.json --output outputs/trial_001

The RGB-D frame is captured even though Phase 0 does not consume it: Phase 2
replaces the ground-truth poses with SAM3 and GraspGenX running on exactly this
frame, and having it from the start means the two phases compare like for like.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from jikkenn2.scene_spec import DEFAULT_SCENE  # noqa: E402

TOOL_PRIM_PATH = "/World/Tools/proxy_tool"
ROBOT_PRIM_PATH = "/World/Panda"
CAMERA_PRIM_PATH = "/World/camera_0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=REPO_ROOT / "assets" / "scene.usd")
    parser.add_argument("--arrangement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--settle-steps", type=int, default=120)
    parser.add_argument("--warmup-frames", type=int, default=60)
    parser.add_argument("--gui", action="store_true", help="Show the viewport while capturing")
    return parser.parse_args()


def make_articulation(prim_path: str, name: str):
    """Wrap an articulation that is already in the stage.

    Isaac renamed these classes between releases, so try the known homes and
    say which ones were tried instead of failing on an import line.
    """
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
        "no articulation wrapper found for "
        f"{prim_path}. Tried: {', '.join(attempts) or 'nothing importable'}. "
        "Check the Isaac Sim release notes for the current class."
    )


def set_home_configuration(articulation, home) -> str:
    """Put the arm at the home pose *and* make the drives hold it there.

    Setting positions alone is not enough: the position controller keeps
    pulling toward its own targets, so the targets have to move too.
    """
    import numpy as np

    positions = np.asarray(home, dtype=np.float32)
    used = []
    if hasattr(articulation, "set_joints_default_state"):
        articulation.set_joints_default_state(positions=positions)
        used.append("set_joints_default_state")
    articulation.set_joint_positions(positions)
    used.append("set_joint_positions")
    for setter in ("set_joint_position_targets", "set_joint_positions_target"):
        if hasattr(articulation, setter):
            getattr(articulation, setter)(positions)
            used.append(setter)
            break
    if len(used) < 2:
        raise RuntimeError(
            "could not pin the home configuration: this articulation exposes "
            f"only {used}. Check the Isaac Sim API for setting drive targets."
        )
    return "+".join(used)


def configure_camera(camera, scene, resolution):
    """Point the camera and give it the metric depth channel."""
    from jikkenn2.geometry import look_at_quaternion_world

    position = np.asarray(scene.camera_position_m, dtype=np.float64)
    orientation = look_at_quaternion_world(position, np.asarray(scene.camera_target_m))
    camera.initialize()
    camera.set_world_pose(position, orientation, camera_axes="world")
    camera.set_clipping_range(*scene.camera_clip_m)
    aperture = float(camera.get_horizontal_aperture())
    focal = aperture / (2.0 * np.tan(np.deg2rad(scene.camera_horizontal_fov_deg) / 2.0))
    camera.set_focal_length(focal)
    camera.add_distance_to_image_plane_to_frame()
    return {
        "resolution_px": list(resolution),
        "horizontal_fov_deg": scene.camera_horizontal_fov_deg,
        "focal_length": float(focal),
        "horizontal_aperture": aperture,
        "clipping_range_m": list(scene.camera_clip_m),
    }


def read_rgb_and_depth(camera):
    frame = camera.get_current_frame()
    rgba = frame.get("rgba")
    if rgba is None or np.asarray(rgba).size <= 1:
        rgba = frame.get("rgb")
    depth = frame.get("distance_to_image_plane")
    if rgba is None or depth is None:
        raise RuntimeError(f"camera frame is incomplete; keys: {sorted(frame)}")
    rgb = np.asarray(rgba)[..., :3]
    if np.issubdtype(rgb.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(rgb)) <= 1.0 else 1.0
        rgb = np.clip(rgb * scale, 0, 255).astype(np.uint8)
    else:
        rgb = rgb.astype(np.uint8, copy=False)
    return rgb, np.asarray(depth, dtype=np.float32)


def build_point_maps(camera, depth_m):
    """Pixel-aligned camera and world point maps, from Isaac's own projection."""
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    valid_vu = np.argwhere(valid)
    if valid_vu.size == 0:
        raise RuntimeError("the camera produced no valid metric depth")
    valid_uv = np.column_stack(
        (valid_vu[:, 1].astype(np.float32) + 0.5, valid_vu[:, 0].astype(np.float32) + 0.5)
    )
    valid_depth = depth_m[valid_vu[:, 0], valid_vu[:, 1]].astype(np.float32, copy=False)
    camera_points = np.asarray(
        camera.get_camera_points_from_image_coords(valid_uv, valid_depth), dtype=np.float64
    )
    world_points = np.asarray(
        camera.get_world_points_from_image_coords(valid_uv, valid_depth), dtype=np.float64
    )
    points_camera = np.full((*depth_m.shape, 3), np.nan, dtype=np.float32)
    points_world = np.full((*depth_m.shape, 3), np.nan, dtype=np.float32)
    points_camera[valid_vu[:, 0], valid_vu[:, 1]] = camera_points
    points_world[valid_vu[:, 0], valid_vu[:, 1]] = world_points
    return points_camera, points_world, valid_uv, camera_points, world_points


def geometry_roundtrip(camera, valid_uv, camera_points, world_points, T_world_camera):
    """Check the serialized transform against Isaac's own projection APIs."""
    from jikkenn2.geometry import transform_points

    count = min(256, valid_uv.shape[0])
    picks = np.linspace(0, valid_uv.shape[0] - 1, count, dtype=np.int64)
    reprojected = np.asarray(
        camera.get_image_coords_from_world_points(world_points[picks]), dtype=np.float64
    )
    serialized = transform_points(T_world_camera, camera_points[picks])
    pixel_error = float(np.max(np.linalg.norm(reprojected - valid_uv[picks], axis=1)))
    world_error = float(np.max(np.linalg.norm(serialized - world_points[picks], axis=1)))
    return {
        "sample_count": int(count),
        "max_pixel_roundtrip_error_px": pixel_error,
        "pixel_error_bound_px": 1e-3,
        "max_serialized_transform_error_m": world_error,
        "world_error_bound_m": 1e-5,
        "passed": pixel_error <= 1e-3 and world_error <= 1e-5,
    }


def main() -> int:
    args = parse_args()
    scene = DEFAULT_SCENE
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "capture_check.json"

    from jikkenn2.arrangement import (
        apply_arrangement,
        collect_arrangement,
        load_arrangement,
    )

    arrangement = load_arrangement(args.arrangement)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": not args.gui})

    try:
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.sensors.camera import Camera

        from jikkenn2 import ground_truth as gt
        from jikkenn2.geometry import matrix_from_pose

        context = omni.usd.get_context()
        context.open_stage(str(args.scene.resolve()))
        for _ in range(60):
            simulation_app.update()
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError(f"Isaac could not open {args.scene}")

        applied = apply_arrangement(stage, arrangement)

        world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 30.0)
        panda = make_articulation(ROBOT_PRIM_PATH, "panda")
        world.scene.add(panda)
        world.reset()

        # Pin the start configuration.  Without this the arm drifts toward
        # whatever joint targets the referenced asset carries, which would then
        # become the planner's start state and could sweep the tool off the
        # table on the way.
        home = scene.home_positions_for(panda.dof_names)
        home_method = set_home_configuration(panda, home)
        world.reset()

        # Settle: a hand-placed object is never exactly resting on the table.
        for _ in range(args.settle_steps):
            world.step(render=False)
        settled = collect_arrangement(stage, scene, source_stage=str(args.scene))
        settle = gt.settle_report(arrangement["objects"], settled["objects"])

        width, height = scene.camera_resolution_px
        camera = Camera(prim_path=CAMERA_PRIM_PATH, resolution=(width, height), frequency=30)
        camera_settings = configure_camera(camera, scene, (width, height))
        for _ in range(args.warmup_frames):
            world.step(render=True)

        rgb, depth_m = read_rgb_and_depth(camera)
        intrinsics = np.asarray(camera.get_intrinsics_matrix(), dtype=np.float64)
        optical_position, optical_orientation = camera.get_world_pose(camera_axes="ros")
        T_world_camera = matrix_from_pose(optical_position, optical_orientation)
        points_camera, points_world, valid_uv, camera_points, world_points = (
            build_point_maps(camera, depth_m)
        )
        geometry = geometry_roundtrip(
            camera, valid_uv, camera_points, world_points, T_world_camera
        )

        base_position, base_orientation = panda.get_world_pose()
        T_world_robot_base = matrix_from_pose(base_position, base_orientation)
        joint_names = tuple(str(name) for name in panda.dof_names)
        joint_positions = np.asarray(panda.get_joint_positions(), dtype=np.float64)
        home_error_rad = float(np.max(np.abs(joint_positions - home)))

        tool = next(
            entry for entry in settled["objects"] if entry["prim_path"] == TOOL_PRIM_PATH
        )
        tool_pose = gt.tool_pose_matrix(tool["position_m"], tool["orientation_wxyz"])
        parts = {
            name: placed.as_dict()
            for name, placed in gt.part_poses(tool_pose, scene).items()
        }

        np.save(output / "rgb.npy", rgb)
        np.save(output / "depth_m.npy", depth_m)
        np.save(output / "intrinsics.npy", intrinsics)
        np.save(output / "T_world_camera.npy", T_world_camera)
        np.save(output / "T_world_robot_base.npy", T_world_robot_base)
        np.save(
            output / "T_robot_base_camera.npy",
            np.linalg.inv(T_world_robot_base) @ T_world_camera,
        )
        np.save(output / "points_camera.npy", points_camera)
        np.save(output / "points_world.npy", points_world)
        np.save(output / "panda_joint_positions.npy", joint_positions)
        np.save(output / "tool_pose_world.npy", tool_pose)
        (output / "settled_arrangement.json").write_text(
            json.dumps(settled, indent=2) + "\n", encoding="utf-8"
        )

        checks = {
            "arrangement_applied": len(applied) == len(arrangement["objects"]),
            "camera_geometry_roundtrip": geometry["passed"],
            "depth_has_valid_pixels": bool((np.isfinite(depth_m) & (depth_m > 0)).any()),
            "tool_is_in_the_capture": TOOL_PRIM_PATH in {
                entry["prim_path"] for entry in settled["objects"]
            },
            "robot_state_is_finite": bool(np.all(np.isfinite(joint_positions))),
            "objects_stayed_where_they_were_placed": settle["status"] == "success",
            "robot_held_its_home_configuration": bool(home_error_rad <= 0.05),
        }
        report = {
            "status": "success" if all(checks.values()) else "failed_checks",
            "scene": str(args.scene),
            "arrangement": str(args.arrangement),
            "frames": {
                "camera": "OpenCV optical, +x right, +y down, +z forward",
                "world": "Isaac world; robot mount and tabletop at z=0",
            },
            "camera": camera_settings,
            "geometry_check": geometry,
            "robot": {
                "prim_path": ROBOT_PRIM_PATH,
                "joint_names": list(joint_names),
                "joint_count": len(joint_names),
                "home_joint_positions": [round(float(v), 6) for v in home],
                "home_applied_with": home_method,
                "max_deviation_from_home_rad": round(home_error_rad, 5),
            },
            "ground_truth": {
                "usage": "evaluation and Phase 0 stand-in for perception only",
                "tool_prim_path": TOOL_PRIM_PATH,
                "tool_position_m": tool["position_m"],
                "tool_orientation_wxyz": tool["orientation_wxyz"],
                "parts": parts,
                "grasp_part": scene.grasp_part_name,
                "handover_reference": gt.handover_orientation(tool_pose, scene),
            },
            "settle": settle,
            "counts": {
                "valid_depth_pixels": int((np.isfinite(depth_m) & (depth_m > 0)).sum()),
                "objects": len(settled["objects"]),
            },
            "automatic_checks": checks,
            "next_step": (
                "python scripts/plan_handover.py --capture "
                f"{output} --tool assets/tools/proxy_tool.json"
            ),
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report["automatic_checks"], indent=2), flush=True)
        print(f"valid depth pixels: {report['counts']['valid_depth_pixels']}", flush=True)
        print(f"settle: {settle['status']}", flush=True)
        print(f"CAPTURE {report['status'].upper()} -> {report_path}", flush=True)
        failed = report["status"] != "success"
    except Exception as error:
        import traceback

        report_path.write_text(
            json.dumps(
                {
                    "status": "failure",
                    "exception_type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        print(f"failure report: {report_path}", file=sys.stderr, flush=True)
        raise
    finally:
        simulation_app.close()

    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build the collision map from a captured depth frame, and prove it is sane.

Phase 1 replaces one layer: the planner's world stops being exact boxes and
becomes an ESDF measured from the camera.  Everything else, including where the
tool is, still comes from the simulator.

    source ~/GraspGenX/.venv/bin/activate
    python scripts/build_map.py --capture outputs/trial_001

Two gates, both of them the ones jikkenn1 lacked.

*Acceptance test*: at the joint state the capture recorded, every one of the
robot's collision spheres must be clear of the map.  In simulation the arm is
known to be standing in free space, so a single sphere in collision means the
map is wrong -- not that the scene is tight.

*Visual gate*: the blocked voxels the planner will actually see are written out
as a point cloud, so they can be looked at next to the robot.  A map that
passes its own JSON checks and is still wrong is exactly how the previous
project lost weeks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from jikkenn2 import ground_truth as gt  # noqa: E402
from jikkenn2.joints import select_named_joint_positions  # noqa: E402
from jikkenn2.mapping import blocked_voxel_centers, map_extent  # noqa: E402
from jikkenn2.pointcloud import subsample, write_colored_cloud  # noqa: E402
from jikkenn2.scene_spec import DEFAULT_SCENE  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--robot", default="franka.yml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--extent", type=float, nargs=3, default=(1.6, 1.6, 1.6))
    parser.add_argument("--grid-center", type=float, nargs=3, default=(0.5, 0.0, 0.75))
    parser.add_argument("--robot-distance-threshold", type=float, default=0.05)
    parser.add_argument(
        "--tool-margin",
        type=float,
        default=0.015,
        help="Extra metres around the tool when clearing it from the depth",
    )
    parser.add_argument(
        "--max-blocked-points",
        type=int,
        default=200000,
        help="Cap on the blocked-voxel cloud written for inspection",
    )
    return parser.parse_args()


def load_capture(capture: Path) -> dict:
    report_path = capture / "capture_check.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "success":
        raise ValueError(f"{report_path} reports status={report.get('status')!r}")
    return {
        "report": report,
        "depth_m": np.load(capture / "depth_m.npy"),
        "rgb": np.load(capture / "rgb.npy"),
        "intrinsics": np.load(capture / "intrinsics.npy"),
        "points_world": np.load(capture / "points_world.npy"),
        "T_robot_base_camera": np.load(capture / "T_robot_base_camera.npy"),
        "T_world_robot_base": np.load(capture / "T_world_robot_base.npy"),
        "tool_pose_world": np.load(capture / "tool_pose_world.npy"),
        "joint_positions": np.load(capture / "panda_joint_positions.npy"),
        "joint_names": tuple(report["robot"]["joint_names"]),
    }


def main() -> int:
    args = parse_args()
    scene = DEFAULT_SCENE
    output = args.output or (args.capture / "map")
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "map_check.json"

    try:
        capture = load_capture(args.capture)

        import torch
        from curobo.perception import FilterDepth, Mapper, MapperCfg, RobotSegmenter
        from curobo.types import CameraObservation, JointState, Pose

        if not torch.cuda.is_available():
            raise RuntimeError("building the map requires CUDA")
        device = torch.device(args.device)
        depth = capture["depth_m"].astype(np.float32, copy=False)
        image_shape = depth.shape

        # The tool is cleared from the depth with its true pose. Phase 1 changes
        # the map, not where objects are; SAM3 takes this over in Phase 2. The
        # hole it leaves is unobserved, which cuRobo already treats as free, so
        # nothing here classifies free space by hand.
        tool_mask = gt.points_in_tool_mask(
            capture["tool_pose_world"],
            scene,
            capture["points_world"],
            margin_m=args.tool_margin,
        )

        loaded = RobotSegmenter.from_robot_file(
            args.robot,
            distance_threshold=args.robot_distance_threshold,
            use_cuda_graph=False,
        )
        segmenter = RobotSegmenter(
            loaded.kinematics,
            distance_threshold=args.robot_distance_threshold,
            use_cuda_graph=False,
            ops_dtype=torch.float32,
        )
        depth_filter = FilterDepth(
            image_shape=image_shape,
            depth_minimum_distance=0.05,
            depth_maximum_distance=5.0,
            flying_pixel_threshold=0.5,
            bilateral_kernel_size=3,
            device=args.device,
        )
        mapper = Mapper(
            MapperCfg(
                extent_meters_xyz=tuple(args.extent),
                voxel_size=args.voxel_size,
                esdf_voxel_size=args.voxel_size,
                extent_esdf_meters_xyz=tuple(args.extent),
                grid_center=torch.tensor(
                    args.grid_center, device=device, dtype=torch.float32
                ),
                truncation_distance=args.voxel_size * 6.0,
                minimum_tsdf_weight=0.1,
                depth_minimum_distance=0.05,
                depth_maximum_distance=5.0,
                decay_factor=1.0,
                frustum_decay_factor=1.0,
                enable_static=False,
                num_cameras=1,
                image_height=image_shape[0],
                image_width=image_shape[1],
                device=args.device,
            )
        )

        observation = CameraObservation(
            name="capture",
            depth_image=torch.from_numpy(depth).to(device=device).unsqueeze(0),
            rgb_image=torch.from_numpy(capture["rgb"]).to(device=device).unsqueeze(0),
            intrinsics=torch.from_numpy(
                capture["intrinsics"].astype(np.float32)
            ).to(device=device).unsqueeze(0),
            pose=Pose.from_matrix(
                torch.from_numpy(capture["T_robot_base_camera"].astype(np.float32))
                .to(device=device)
                .unsqueeze(0)
            ),
            depth_to_meter=1.0,
        )
        state = JointState.from_position(
            torch.from_numpy(capture["joint_positions"].astype(np.float32))
            .to(device=device)
            .unsqueeze(0),
            joint_names=list(capture["joint_names"]),
        )

        robot_mask_tensor, depth_without_robot = segmenter.get_robot_mask(observation, state)
        tool_mask_tensor = torch.from_numpy(tool_mask).to(device=device).unsqueeze(0)
        mapping_depth = torch.where(tool_mask_tensor, 0.0, depth_without_robot)
        mapping_depth, _ = depth_filter(mapping_depth)
        # Re-apply after filtering: smoothing must never put the robot or the
        # tool back into the map.
        exclusion = torch.logical_or(robot_mask_tensor, tool_mask_tensor)
        mapping_depth = torch.where(exclusion, 0.0, mapping_depth)

        mapper.integrate(
            CameraObservation(
                name="capture_without_robot_or_tool",
                depth_image=mapping_depth,
                rgb_image=observation.rgb_image,
                intrinsics=observation.intrinsics,
                pose=observation.pose,
                depth_to_meter=1.0,
            )
        )
        voxel_grid = mapper.compute_esdf()
        occupied = mapper.extract_occupied_voxels(surface_only=True)

        features = voxel_grid.feature_tensor.detach().cpu().numpy()
        dims = np.asarray(voxel_grid.dims, dtype=np.float64)
        center = np.asarray(list(voxel_grid.pose)[:3], dtype=np.float64)
        voxel_size = float(voxel_grid.voxel_size)
        shape = np.rint(dims / voxel_size).astype(np.int64)
        grid = {
            "shape": [int(v) for v in shape],
            "voxel_size_m": voxel_size,
            "dims_m": [float(v) for v in dims],
            "center_robot_base_m": [float(v) for v in center],
            "min_corner_m": [float(v) for v in (center - 0.5 * dims)],
            "sdf_sign": "negative inside obstacles, as cuRobo returns it",
        }
        grid["extent"] = map_extent(grid)

        np.save(output / "esdf_features.npy", features)
        np.save(output / "mapping_depth_m.npy", mapping_depth[0].detach().cpu().numpy())
        np.save(output / "robot_mask.npy", robot_mask_tensor[0].detach().cpu().numpy())
        np.save(output / "tool_mask.npy", tool_mask)
        (output / "grid.json").write_text(
            json.dumps(grid, indent=2) + "\n", encoding="utf-8"
        )

        # --- the visual gate -------------------------------------------
        blocked = blocked_voxel_centers(features, grid)
        shown = subsample(blocked, args.max_blocked_points)
        write_colored_cloud(output / "blocked_voxels.ply", shown, (220, 60, 30))
        occupied_points = occupied.centers.detach().cpu().numpy().astype(np.float32)
        np.save(output / "occupied_points_robot_base.npy", occupied_points)
        write_colored_cloud(
            output / "occupied_surface.ply", occupied_points, (150, 150, 150)
        )

        # --- the acceptance test ---------------------------------------
        from curobo._src.geom.collision.buffer_collision import CollisionBuffer
        from curobo._src.geom.types import SceneCfg, VoxelGrid
        from curobo._src.motion.motion_planner import MotionPlanner
        from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
        from curobo._src.state.state_joint import JointState as PlannerJointState
        from curobo._src.types.device_cfg import DeviceCfg

        device_cfg = DeviceCfg(device=device, dtype=torch.float32)
        planner_grid = VoxelGrid(
            name="measured_esdf",
            pose=[*grid["center_robot_base_m"], 1.0, 0.0, 0.0, 0.0],
            dims=[float(count) * voxel_size for count in grid["shape"]],
            voxel_size=voxel_size,
            feature_tensor=torch.from_numpy(features)
            .to(device=device, dtype=torch.float16)
            .contiguous(),
            feature_dtype=torch.float16,
        )
        planner_cfg = MotionPlannerCfg.create(
            robot=args.robot,
            scene_model=SceneCfg(voxel=[planner_grid]),
            device_cfg=device_cfg,
            num_ik_seeds=8,
            num_trajopt_seeds=2,
            optimizer_collision_activation_distance=0.01,
            use_cuda_graph=False,
            random_seed=123,
        )
        planner = MotionPlanner(planner_cfg)
        start = select_named_joint_positions(
            capture["joint_names"], capture["joint_positions"], planner.joint_names
        ).astype(np.float32)
        kinematics = planner.compute_kinematics(
            PlannerJointState.from_position(
                torch.from_numpy(start).to(device).unsqueeze(0),
                joint_names=list(planner.joint_names),
            )
        )
        spheres = kinematics.robot_spheres
        buffer = CollisionBuffer.from_shape(spheres.shape, device_cfg)
        buffer.zero_()
        costs = planner.scene_collision_checker.get_sphere_collision(
            kinematics,
            buffer,
            torch.tensor([1.0], device=device, dtype=torch.float32),
            torch.tensor([0.0], device=device, dtype=torch.float32),
        )
        torch.cuda.synchronize(device)
        costs = costs.detach().cpu().numpy().reshape(-1)
        sphere_array = spheres.detach().cpu().numpy().reshape(-1, 4)
        # cuRobo disables a sphere slot by giving it a negative radius.
        active = sphere_array[:, 3] >= 0.0
        colliding = int(np.count_nonzero(costs[active] > 0.0))
        np.save(output / "start_spheres.npy", sphere_array)
        np.save(output / "start_sphere_collision_cost.npy", costs)
        planner.destroy()

        checks = {
            "robot_pixels_removed": bool(robot_mask_tensor.any().item()),
            "tool_pixels_removed": bool(tool_mask.any()),
            "map_has_occupied_voxels": bool(len(occupied_points) > 0),
            "map_has_free_space": bool(np.any(features > 0.0)),
            # The one that matters: in simulation the arm is known to be in free
            # space, so any sphere in collision means the map is wrong.
            "start_state_is_collision_free": colliding == 0,
        }
        report = {
            "status": "success" if all(checks.values()) else "failed_checks",
            "capture": str(args.capture),
            "reference": {
                "mapper": "cuRobo Mapper.integrate + compute_esdf",
                "robot_removal": "cuRobo RobotSegmenter",
                "tool_removal": "ground-truth pose; SAM3 replaces this in Phase 2",
                "free_space_policy": (
                    "cuRobo's own; unobserved space is not reclassified here"
                ),
            },
            "grid": grid,
            "counts": {
                "robot_mask_pixels": int(robot_mask_tensor.sum().item()),
                "tool_mask_pixels": int(tool_mask.sum()),
                "mapping_depth_pixels": int((mapping_depth > 0).sum().item()),
                "occupied_surface_voxels": int(len(occupied_points)),
                "blocked_voxels": int(len(blocked)),
                "blocked_voxels_written": int(len(shown)),
            },
            "acceptance_test": {
                "what": "every collision sphere clear of the map at the captured pose",
                "why": "the arm is known to stand in free space in simulation",
                "active_spheres": int(np.count_nonzero(active)),
                "colliding_spheres": colliding,
                "maximum_collision_cost_m": round(float(np.max(costs[active])), 5),
                "passed": colliding == 0,
            },
            "inspect": {
                "blocked_voxels_ply": str(output / "blocked_voxels.ply"),
                "occupied_surface_ply": str(output / "occupied_surface.ply"),
                "viewer": (
                    f"python scripts/view_map.py --map {output} --capture {args.capture}"
                ),
            },
            "automatic_checks": checks,
            "next_step": (
                f"python scripts/plan_handover.py --capture {args.capture} "
                f"--map {output}"
            ),
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(checks, indent=2), flush=True)
        print(
            f"blocked voxels: {len(blocked)}  "
            f"colliding start spheres: {colliding}/{int(np.count_nonzero(active))}",
            flush=True,
        )
        print(f"MAP {report['status'].upper()} -> {report_path}", flush=True)
        return 0 if report["status"] == "success" else 2
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


if __name__ == "__main__":
    raise SystemExit(main())

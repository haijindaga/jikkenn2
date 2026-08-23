#!/usr/bin/env python3
"""Plan the whole trial: above the tool, down, up, and out to the person.

Reads a capture, builds the waypoints from the ground-truth tool pose, and asks
cuRobo for a collision-checked trajectory through each of them.  Nothing is
executed here; execute_handover.py replays the result in Isaac.

    source ~/GraspGenX/.venv/bin/activate
    python scripts/plan_handover.py --capture outputs/trial_001 \
           --tool assets/tools/proxy_tool.json --output outputs/trial_001/plan

Phase 0 takes the tool's pose from the simulator rather than from perception,
which is the whole point: the skeleton runs first, and SAM3 and GraspGenX
replace this one input in Phase 2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from jikkenn2 import ground_truth as gt  # noqa: E402
from jikkenn2 import handover as hv  # noqa: E402
from jikkenn2.joints import select_named_joint_positions  # noqa: E402
from jikkenn2.scene_spec import DEFAULT_SCENE  # noqa: E402

TOOL_PRIM_PATH = "/World/Tools/proxy_tool"

#: Ordered legs of the trial.  Each is planned from where the previous one
#: ended, so a failure names the leg that could not be reached.
SEGMENTS = ("pregrasp", "grasp", "lift", "handover")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--tool", type=Path, default=REPO_ROOT / "assets" / "tools" / "proxy_tool.json")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--robot", default="franka.yml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--approach-offset", type=float, default=hv.DEFAULT_APPROACH_OFFSET_M)
    parser.add_argument("--lift", type=float, default=hv.DEFAULT_LIFT_M)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--num-ik-seeds", type=int, default=24)
    parser.add_argument("--num-trajopt-seeds", type=int, default=4)
    return parser.parse_args()


def load_capture(capture: Path) -> dict:
    """Read what the capture wrote, refusing an incomplete or failed one."""
    report_path = capture / "capture_check.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"no capture report at {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "success":
        raise ValueError(
            f"{report_path} reports status={report.get('status')!r}; "
            "planning from a failed capture would be meaningless"
        )
    settled = json.loads((capture / "settled_arrangement.json").read_text(encoding="utf-8"))
    return {
        "report": report,
        "tool_pose_world": np.load(capture / "tool_pose_world.npy"),
        "joint_positions": np.load(capture / "panda_joint_positions.npy"),
        "joint_names": tuple(report["robot"]["joint_names"]),
        "T_world_robot_base": np.load(capture / "T_world_robot_base.npy"),
        "settled": settled,
    }


def world_obstacles(settled: dict, scene) -> list[dict]:
    """Collision boxes for everything on the table except the tool.

    The tool is left out on purpose: it is the thing being grasped, so the hand
    has to reach it.  Contact with the tool is judged by physics during
    execution, not by the planner.
    """
    by_name = {obstacle.prim_path: obstacle for obstacle in scene.obstacles}
    boxes = []
    for entry in settled["objects"]:
        path = entry["prim_path"]
        if path == TOOL_PRIM_PATH:
            continue
        obstacle = by_name.get(path)
        if obstacle is None:
            raise KeyError(
                f"{path} is in the arrangement but has no size in scene_spec.obstacles"
            )
        boxes.append(
            {
                "name": obstacle.name,
                "pose": [*entry["position_m"], *entry["orientation_wxyz"]],
                "dims": list(obstacle.size_m),
            }
        )
    return boxes


def to_robot_base(pose_world: np.ndarray, T_world_robot_base: np.ndarray) -> np.ndarray:
    return np.linalg.inv(T_world_robot_base) @ pose_world


def build_planner(args, scene, boxes):
    import torch
    from curobo._src.geom.types import Cuboid, SceneCfg
    from curobo._src.motion.motion_planner import MotionPlanner
    from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
    from curobo._src.types.device_cfg import DeviceCfg

    if not torch.cuda.is_available():
        raise RuntimeError("cuRobo planning requires CUDA")

    device_cfg = DeviceCfg(device=torch.device(args.device), dtype=torch.float32)
    cuboids = [
        Cuboid(
            name="table",
            pose=[*scene.table_center_m, 1.0, 0.0, 0.0, 0.0],
            dims=list(scene.table_size_m),
        )
    ]
    cuboids += [
        Cuboid(name=box["name"], pose=box["pose"], dims=box["dims"]) for box in boxes
    ]
    planner_cfg = MotionPlannerCfg.create(
        robot=args.robot,
        scene_model=SceneCfg(cuboid=cuboids),
        device_cfg=device_cfg,
        num_ik_seeds=args.num_ik_seeds,
        num_trajopt_seeds=args.num_trajopt_seeds,
        optimizer_collision_activation_distance=0.01,
        use_cuda_graph=False,
        random_seed=123,
    )
    planner = MotionPlanner(planner_cfg)
    planner.warmup(enable_graph=True, num_warmup_iterations=2)
    return planner, device_cfg, len(cuboids)


def plan_segment(planner, device_cfg, current_state, pose_base, max_attempts):
    """Plan one leg; returns (success, joint trajectory, diagnostics)."""
    import torch
    from curobo._src.types.tool_pose import GoalToolPose

    from jikkenn2.geometry import quaternion_wxyz_from_rotation_matrix

    position = torch.from_numpy(
        np.asarray(pose_base[:3, 3], dtype=np.float32)
    ).to(device_cfg.device).reshape(1, 1, 1, 1, 3)
    quaternion = torch.from_numpy(
        quaternion_wxyz_from_rotation_matrix(pose_base[:3, :3]).astype(np.float32)
    ).to(device_cfg.device).reshape(1, 1, 1, 1, 4)
    goals = GoalToolPose(
        tool_frames=planner.tool_frames, position=position, quaternion=quaternion
    )
    started = time.monotonic()
    result = planner.plan_pose(
        current_state=current_state, goal_tool_poses=goals, max_attempts=max_attempts
    )
    elapsed = time.monotonic() - started
    success = bool(
        result is not None and result.success is not None and result.success.any().item()
    )
    diagnostics = {
        "planner_reported_success": success,
        "wall_time_s": round(elapsed, 3),
        "position_error_m": _scalar(getattr(result, "position_error", None)),
        "rotation_error_rad": _scalar(getattr(result, "rotation_error", None)),
    }
    if not success:
        return False, None, diagnostics
    trajectory = result.get_interpolated_plan()
    positions = trajectory.position
    if hasattr(positions, "detach"):
        positions = positions.detach().cpu().numpy()
    positions = _as_horizon_by_dof(positions, len(planner.joint_names))
    diagnostics["waypoints"] = int(positions.shape[0])
    return True, positions, diagnostics


def _as_horizon_by_dof(positions, dof: int) -> np.ndarray:
    """Reduce a planned trajectory to ``(horizon, dof)``.

    cuRobo returns the plan with leading batch and seed axes, and how many of
    them there are has varied; peel singleton axes rather than assuming a rank.
    """
    array = np.asarray(positions, dtype=np.float32)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise RuntimeError(
            f"planned trajectory has shape {np.asarray(positions).shape}, which does "
            "not reduce to (horizon, dof) by dropping singleton leading axes"
        )
    if array.shape[1] != dof:
        raise RuntimeError(
            f"planned trajectory has {array.shape[1]} joints, expected {dof}"
        )
    return array


def _scalar(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value).reshape(-1)
    return float(array[0]) if array.size else None


def main() -> int:
    args = parse_args()
    scene = DEFAULT_SCENE
    output = args.output or (args.capture / "plan")
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "plan_check.json"

    try:
        capture = load_capture(args.capture)
        annotation = json.loads(args.tool.read_text(encoding="utf-8"))
        tool_pose = np.asarray(capture["tool_pose_world"], dtype=np.float64)

        waypoints = hv.plan_waypoints(
            tool_pose, scene, approach_offset_m=args.approach_offset, lift_m=args.lift
        )
        # The grasp must actually be on the part we claim to grasp; check it
        # here rather than discovering it from a bad score later.
        fingertip = (
            waypoints["grasp"][:3, 3]
            + waypoints["grasp"][:3, 2] * hv.PANDA_FINGERTIP_DEPTH_M
        )
        grasped_part = gt.part_containing(tool_pose, scene, fingertip)
        planned_handover = gt.handover_orientation(waypoints["target_tool"], scene)

        boxes = world_obstacles(capture["settled"], scene)
        T_world_robot_base = np.asarray(capture["T_world_robot_base"], dtype=np.float64)

        import torch
        from curobo._src.state.state_joint import JointState

        planner, device_cfg, obstacle_count = build_planner(args, scene, boxes)
        start_positions = select_named_joint_positions(
            capture["joint_names"], capture["joint_positions"], planner.joint_names
        ).astype(np.float32)
        current_state = JointState.from_position(
            torch.from_numpy(start_positions).to(device_cfg.device).unsqueeze(0),
            joint_names=list(planner.joint_names),
        )

        segments = {}
        trajectories = {}
        reached = []
        for name in SEGMENTS:
            pose_base = to_robot_base(waypoints[name], T_world_robot_base)
            success, positions, diagnostics = plan_segment(
                planner, device_cfg, current_state, pose_base, args.max_attempts
            )
            segments[name] = diagnostics
            if not success:
                print(f"segment {name}: FAILED", flush=True)
                break
            trajectories[name] = positions
            reached.append(name)
            print(
                f"segment {name}: {diagnostics['waypoints']} waypoints "
                f"in {diagnostics['wall_time_s']:.2f}s",
                flush=True,
            )
            current_state = JointState.from_position(
                torch.from_numpy(positions[-1]).to(device_cfg.device).unsqueeze(0),
                joint_names=list(planner.joint_names),
            )

        for name, positions in trajectories.items():
            np.save(output / f"trajectory_{name}.npy", positions)
        np.save(output / "grasp_hand_pose_world.npy", waypoints["grasp"])
        np.save(output / "target_tool_pose_world.npy", waypoints["target_tool"])
        np.save(
            output / "T_tool_hand.npy",
            np.linalg.inv(tool_pose) @ waypoints["grasp"],
        )

        planned_all = len(reached) == len(SEGMENTS)
        checks = {
            "grasp_is_on_the_intended_part": grasped_part == scene.grasp_part_name,
            "planned_handover_presents_the_safe_part": (
                planned_handover["safe_axis_to_human_deg"] < 30.0
                and planned_handover["danger_axis_to_human_deg"] > 90.0
            ),
            "every_segment_planned": planned_all,
            "trajectories_are_finite": all(
                bool(np.isfinite(positions).all()) for positions in trajectories.values()
            ),
            "first_segment_starts_at_the_capture": bool(
                trajectories
                and np.allclose(
                    trajectories[SEGMENTS[0]][0], start_positions, atol=2e-3
                )
            ),
        }
        report = {
            "status": "success" if all(checks.values()) else "no_complete_plan",
            "reference": {
                "planner": "cuRobo MotionPlanner.plan_pose, one call per segment",
                "tool_frame": "panda_hand",
                "tool_annotation": str(args.tool),
                "tool_annotation_usage": annotation.get("usage"),
            },
            "inputs": {
                "capture": str(args.capture),
                "tool_pose_source": "simulator ground truth (Phase 0 stands in for perception)",
            },
            "world_model": {
                "cuboids": obstacle_count,
                "table_included": True,
                "obstacles_included": len(boxes),
                "tool_included": False,
                "why_tool_excluded": (
                    "the tool is the grasp target, so the hand must reach it; "
                    "contact with it is judged by physics during execution"
                ),
                "attached_object_modelled": False,
            },
            "waypoints": hv.describe_waypoints(waypoints),
            "grasp": {
                "part_under_the_fingertips": grasped_part,
                "intended_part": scene.grasp_part_name,
                "fingertip_world_m": [round(float(v), 5) for v in fingertip],
            },
            "planned_handover": planned_handover,
            "segments": segments,
            "segments_reached": reached,
            "counts": {
                "trajectory_waypoints": {
                    name: int(positions.shape[0])
                    for name, positions in trajectories.items()
                }
            },
            "joint_names": list(planner.joint_names),
            "automatic_checks": checks,
            "safety": {
                "trajectory_executed": False,
                "attached_object_collision_checked": False,
                "human_present_in_simulation": False,
            },
            "next_step": (
                f"python scripts/execute_handover.py --capture {args.capture} "
                f"--plan {output}"
            ),
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(checks, indent=2), flush=True)
        print(f"PLAN {report['status'].upper()} -> {report_path}", flush=True)
        planner.destroy()
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

#!/usr/bin/env python3
"""Replay a planned trial in Isaac and record what actually happened.

This script judges nothing.  It drives the arm through the planned legs,
closes the gripper at the right moment, and writes measurements;
``score_trial.py`` turns those measurements into the six criteria.

    conda activate env_isaaclab
    python scripts/execute_handover.py --capture outputs/trial_001 \
           --plan outputs/trial_001/plan

The planner solves seven joints but its trajectory carries nine, fingers
included.  Finger columns are ignored and the gripper is driven on its own
schedule, so the grasp cannot be opened or closed as a side effect of planning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from jikkenn2.isaac_bridge import (  # noqa: E402
    gripper_width_m,
    make_articulation,
    make_target_commander,
    set_home_configuration,
)
from jikkenn2.joints import merge_named_joint_positions  # noqa: E402
from jikkenn2.scene_spec import DEFAULT_SCENE  # noqa: E402

TOOL_PRIM_PATH = "/World/Tools/proxy_tool"
ROBOT_PRIM_PATH = "/World/Panda"
FINGER_JOINTS = ("panda_finger_joint1", "panda_finger_joint2")

OPEN_M = 0.04
CLOSED_M = 0.0

#: Whether the gripper is open while each leg runs.
LEG_GRIPPER_OPEN = {
    "pregrasp": True,
    "grasp": True,
    "lift": False,
    "handover": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--scene", type=Path, default=REPO_ROOT / "assets" / "scene.usd")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--steps-per-waypoint", type=int, default=2)
    parser.add_argument("--close-steps", type=int, default=90)
    parser.add_argument("--hold-steps", type=int, default=120)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--record-every",
        type=int,
        default=0,
        help="Save a camera frame every N physics steps (0 disables)",
    )
    return parser.parse_args()


def finger_drive_report(stage, root: str = ROBOT_PRIM_PATH) -> dict:
    """Read the gripper's drive parameters straight from the stage.

    If the tool ever pivots or slips again, the first question is how much
    force the fingers can actually apply, and guessing at that from the outside
    wastes a run.
    """
    from pxr import Usd, UsdPhysics

    report = {}
    prim_root = stage.GetPrimAtPath(root)
    if not prim_root.IsValid():
        return {"error": f"{root} is not in the stage"}
    for prim in Usd.PrimRange(prim_root):
        if "finger" not in prim.GetName().lower():
            continue
        for instance in ("linear", "angular", "transX", "transY", "transZ"):
            drive = UsdPhysics.DriveAPI.Get(prim, instance)
            if not drive:
                continue
            report[f"{prim.GetPath().pathString}:{instance}"] = {
                "type": drive.GetTypeAttr().Get(),
                "stiffness": drive.GetStiffnessAttr().Get(),
                "damping": drive.GetDampingAttr().Get(),
                "max_force": drive.GetMaxForceAttr().Get(),
                "target_position": drive.GetTargetPositionAttr().Get(),
            }
    return report


def load_plan(plan_dir: Path) -> dict:
    report_path = plan_dir / "plan_check.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"no plan report at {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "success":
        raise ValueError(
            f"{report_path} reports status={report.get('status')!r}; refusing to "
            "execute an incomplete plan"
        )
    segments = report["segments_reached"]
    trajectories = {
        name: np.load(plan_dir / f"trajectory_{name}.npy") for name in segments
    }
    return {
        "report": report,
        "segments": segments,
        "trajectories": trajectories,
        "columns": tuple(report["joint_names"]["trajectory_columns"]),
    }


class Recorder:
    """Collects what the trial did, without deciding whether it was good."""

    def __init__(self, stage, scene, tool_prim_path: str):
        from jikkenn2.arrangement import iter_movable_prim_paths

        self.stage = stage
        self.scene = scene
        self.tool_prim_path = tool_prim_path
        self.obstacle_paths = [
            path
            for path in iter_movable_prim_paths(stage)
            if path != tool_prim_path
        ]
        self.samples: list[dict] = []
        self.max_tracking_error_rad = 0.0
        # Per-step trace: leg boundaries alone cannot say *when* a grasp was
        # lost, only that it was gone by the next boundary.
        self.trace_step: list[int] = []
        self.trace_leg: list[str] = []
        self.trace_width: list[float] = []
        self.trace_tool: list[list[float]] = []

    def pose(self, prim_path: str) -> dict:
        from jikkenn2.arrangement import read_pose

        return read_pose(self.stage, prim_path).as_dict()

    def obstacle_poses(self) -> dict[str, dict]:
        return {path: self.pose(path) for path in self.obstacle_paths}

    def trace(self, step: int, leg: str, articulation) -> None:
        actual = np.asarray(articulation.get_joint_positions(), dtype=np.float64)
        pose = self.pose(self.tool_prim_path)
        self.trace_step.append(int(step))
        self.trace_leg.append(leg)
        self.trace_width.append(gripper_width_m(articulation.dof_names, actual))
        self.trace_tool.append([float(v) for v in pose["position_m"]])

    def trace_summary(self, table_top_z: float, closed_threshold: float = 0.005) -> dict:
        """Say when the grip was lost, if it was."""
        if not self.trace_step:
            return {"samples": 0}
        width = np.asarray(self.trace_width)
        tool = np.asarray(self.trace_tool)
        legs = np.asarray(self.trace_leg)
        after_grasp = np.flatnonzero(np.isin(legs, ("lift", "handover")))
        lost_at = None
        if after_grasp.size:
            empty = after_grasp[width[after_grasp] < closed_threshold]
            if empty.size:
                lost_at = {
                    "step": int(self.trace_step[int(empty[0])]),
                    "leg": str(legs[int(empty[0])]),
                    "tool_height_m": round(float(tool[int(empty[0]), 2]), 5),
                }
        return {
            "samples": len(self.trace_step),
            "max_tool_height_m": round(float(tool[:, 2].max()), 5),
            "tool_height_gain_m": round(float(tool[:, 2].max() - tool[0, 2]), 5),
            "tool_travel_m": round(
                float(np.linalg.norm(tool[-1, :2] - tool[0, :2])), 5
            ),
            "tool_left_the_table": bool(tool[:, 2].max() > table_top_z + 0.05),
            "minimum_width_while_carrying_m": (
                round(float(width[after_grasp].min()), 5) if after_grasp.size else None
            ),
            "grip_lost_at": lost_at,
        }

    def sample(self, label: str, articulation, commanded) -> dict:
        actual = np.asarray(articulation.get_joint_positions(), dtype=np.float64)
        error = float(np.max(np.abs(actual - np.asarray(commanded, dtype=np.float64))))
        self.max_tracking_error_rad = max(self.max_tracking_error_rad, error)
        entry = {
            "label": label,
            "tool": self.pose(self.tool_prim_path),
            "gripper_width_m": round(
                gripper_width_m(articulation.dof_names, actual), 5
            ),
            "joint_tracking_error_rad": round(error, 5),
        }
        self.samples.append(entry)
        return entry


def main() -> int:
    args = parse_args()
    scene = DEFAULT_SCENE
    plan_dir = args.plan or (args.capture / "plan")
    output = args.output or (args.capture / "execution")
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "execution_log.json"

    plan = load_plan(plan_dir)
    settled = json.loads(
        (args.capture / "settled_arrangement.json").read_text(encoding="utf-8")
    )

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": not args.gui})

    try:
        import omni.usd
        from isaacsim.core.api import World

        from jikkenn2 import ground_truth as gt
        from jikkenn2.arrangement import apply_arrangement

        context = omni.usd.get_context()
        context.open_stage(str(args.scene.resolve()))
        for _ in range(60):
            simulation_app.update()
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError(f"Isaac could not open {args.scene}")

        # Start from where the capture ended, so the plan sees the world it was
        # planned against rather than the hand-authored poses.
        apply_arrangement(stage, settled)

        world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 30.0)
        panda = make_articulation(ROBOT_PRIM_PATH, "panda")
        world.scene.add(panda)
        world.reset()
        home = scene.home_positions_for(panda.dof_names)
        home_method = set_home_configuration(panda, home)
        world.reset()
        for _ in range(60):
            world.step(render=False)

        # Resolve the target-setting API once; which one exists depends on the
        # Isaac release and on whether this is a single or batched articulation.
        command, command_method = make_target_commander(panda)
        print(f"joint targets via {command_method}", flush=True)

        isaac_names = tuple(str(name) for name in panda.dof_names)
        recorder = Recorder(stage, scene, TOOL_PRIM_PATH)
        obstacles_before = recorder.obstacle_poses()
        recorder.sample("start", panda, home)

        frames_dir = output / "frames"
        camera = None
        if args.record_every > 0:
            from isaacsim.sensors.camera import Camera

            width, height = scene.camera_resolution_px
            frames_dir.mkdir(exist_ok=True)
            camera = Camera(
                prim_path="/World/camera_0", resolution=(width, height), frequency=30
            )
            camera.initialize()

        saved_frames = 0
        step_index = 0
        current_leg = "start"

        def advance(target, render: bool) -> None:
            nonlocal step_index, saved_frames
            command(target)
            world.step(render=render)
            step_index += 1
            recorder.trace(step_index, current_leg, panda)
            if camera is not None and step_index % args.record_every == 0:
                frame = camera.get_current_frame()
                rgba = frame.get("rgba")
                if rgba is not None and np.asarray(rgba).size > 1:
                    from PIL import Image

                    image = np.asarray(rgba)[..., :3].astype(np.uint8)
                    Image.fromarray(image).save(frames_dir / f"{saved_frames:05d}.png")
                    saved_frames += 1

        target = home.copy()
        for leg in plan["segments"]:
            current_leg = leg
            trajectory = plan["trajectories"][leg]
            finger = OPEN_M if LEG_GRIPPER_OPEN[leg] else CLOSED_M
            for row in trajectory:
                # Arm joints come from the plan by name; the gripper never does.
                updates = {
                    name: float(value)
                    for name, value in zip(plan["columns"], row)
                    if name not in FINGER_JOINTS
                }
                updates.update({name: finger for name in FINGER_JOINTS})
                target = merge_named_joint_positions(isaac_names, target, updates)
                for _ in range(args.steps_per_waypoint):
                    advance(target, render=args.gui or camera is not None)
            recorder.sample(f"end_of_{leg}", panda, target)
            print(f"leg {leg}: {trajectory.shape[0]} waypoints replayed", flush=True)

            if leg == "grasp":
                current_leg = "close"
                target = merge_named_joint_positions(
                    isaac_names, target, {name: CLOSED_M for name in FINGER_JOINTS}
                )
                for _ in range(args.close_steps):
                    advance(target, render=args.gui or camera is not None)
                recorder.sample("after_close", panda, target)
                print(
                    "gripper closed to "
                    f"{recorder.samples[-1]['gripper_width_m']:.4f} m",
                    flush=True,
                )

        current_leg = "hold"
        for _ in range(args.hold_steps):
            advance(target, render=args.gui or camera is not None)
        final = recorder.sample("after_hold", panda, target)

        obstacles_after = recorder.obstacle_poses()
        obstacle_motion = [
            gt.pose_difference(obstacles_before[path], obstacles_after[path])
            for path in recorder.obstacle_paths
        ]

        tool_pose = gt.tool_pose_matrix(
            final["tool"]["position_m"], final["tool"]["orientation_wxyz"]
        )
        log = {
            "status": "success",
            "capture": str(args.capture),
            "plan": str(plan_dir),
            "robot": {
                "home_applied_with": home_method,
                "joint_targets_via": command_method,
                "joint_names": list(isaac_names),
                "max_joint_tracking_error_rad": round(recorder.max_tracking_error_rad, 5),
            },
            "gripper": {
                "open_m_per_finger": OPEN_M,
                "closed_command_m_per_finger": CLOSED_M,
                "final_width_m": final["gripper_width_m"],
                "driven_independently_of_the_plan": True,
            },
            "legs_replayed": list(plan["segments"]),
            "samples": recorder.samples,
            "obstacle_motion": obstacle_motion,
            "carry": {
                **recorder.trace_summary(scene.table_top_z_m),
                # How far the tool turned inside the fingers between the moment
                # it was gripped and the end of the trial. A firm grasp keeps
                # this near zero; a pivot means gravity is winning.
                "tool_rotation_in_grip_deg": gt.pose_difference(
                    recorder.samples[3]["tool"], final["tool"]
                )["rotation_deg"]
                if len(recorder.samples) > 3
                else None,
            },
            "finger_drives": finger_drive_report(stage),
            "final_tool": final["tool"],
            "final_handover_orientation": gt.handover_orientation(tool_pose, scene),
            "frames_saved": saved_frames,
            "frames_directory": str(frames_dir) if saved_frames else None,
            "measurement_notes": {
                "collision": (
                    "detected by displacement of the movable obstacles and by joint "
                    "tracking error; contact reporting is not enabled"
                ),
                "judgement": "none; score_trial.py applies the criteria",
            },
        }
        np.save(output / "trace_gripper_width_m.npy", np.asarray(recorder.trace_width))
        np.save(output / "trace_tool_position_m.npy", np.asarray(recorder.trace_tool))
        report_path.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
        carry = log["carry"]
        print(
            f"tool lifted {carry['tool_height_gain_m']:.4f} m, "
            f"travelled {carry['tool_travel_m']:.4f} m",
            flush=True,
        )
        if carry.get("grip_lost_at"):
            print(f"GRIP LOST at {carry['grip_lost_at']}", flush=True)
        print(f"final gripper width: {final['gripper_width_m']:.4f} m", flush=True)
        print(
            "final handover angles: safe "
            f"{log['final_handover_orientation']['safe_axis_to_human_deg']:.1f} deg, "
            f"danger {log['final_handover_orientation']['danger_axis_to_human_deg']:.1f} deg",
            flush=True,
        )
        print(f"EXECUTION DONE -> {report_path}", flush=True)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

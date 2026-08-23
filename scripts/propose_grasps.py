#!/usr/bin/env python3
"""Ask GraspGenX where the gripper could close, then choose which one to use.

The proposer knows where a gripper fits.  It does not know which end of a knife
is the blade, and that choice is the whole point of the project, so it is made
here and reported in full: how many proposals landed on the dangerous part, how
many on the safe one, and why each was rejected.

The server must already be running.  Start it in its own terminal, exactly as
before -- this script only connects:

    cd ~/GraspGenX && uv run python client-server/graspgenx_server.py \
        --config ext/graspgenx_checkpoints/release --assets_dir assets \
        --default_gripper franka_panda --port 5556

Then, from the GraspGenX environment:

    python scripts/propose_grasps.py --capture outputs/trial_001

Nothing in the GraspGenX checkout is modified; jikkenn1 shares this server.
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

from jikkenn2 import grasp_selection as gs  # noqa: E402
from jikkenn2 import graspgen  # noqa: E402
from jikkenn2.pointcloud import write_colored_cloud  # noqa: E402
from jikkenn2.reachability import PANDA_FINGERTIP_DEPTH_M  # noqa: E402
from jikkenn2.scene_spec import DEFAULT_SCENE  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--timeout-ms", type=int, default=60_000)
    parser.add_argument("--planner", default="graspmoe", choices=("graspmoe", "diffusion"))
    parser.add_argument("--gripper", default="franka_panda")
    parser.add_argument("--min-object-points", type=int, default=100)
    parser.add_argument("--num-grasps", type=int, default=200)
    parser.add_argument("--grasp-threshold", type=float, default=-1.0)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument(
        "--minimum-downwardness",
        type=float,
        default=0.0,
        help="Reject proposals whose approach is less downward than this",
    )
    parser.add_argument(
        "--frame-tolerance",
        type=float,
        default=0.03,
        help="Median fingertip-to-object distance allowed before the frames are suspect",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene = DEFAULT_SCENE
    segmentation = args.segmentation or (args.capture / "segmentation")
    output = args.output or (args.capture / "grasps")
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "grasp_check.json"

    try:
        try:
            from graspgenx.serving.types import SweepVolumeParams
            from graspgenx.serving.zmq_client import GraspGenXClient
        except ImportError as error:
            raise RuntimeError(
                "GraspGenX is not importable here. Run this from the GraspGenX "
                "environment, for example "
                "'~/GraspGenX/.venv/bin/python scripts/propose_grasps.py ...' or "
                "'cd ~/GraspGenX && uv run python <path>/scripts/propose_grasps.py ...'."
            ) from error

        points_camera = np.load(args.capture / "points_camera.npy")
        T_world_camera = np.load(args.capture / "T_world_camera.npy")
        tool_pose = np.load(args.capture / "tool_pose_world.npy")
        mask = np.load(segmentation / "union_mask.npy")
        object_points_world = np.load(segmentation / "points_world.npy")

        cloud, instance_mask, instance_count = graspgen.prepare_scene_point_cloud(
            points_camera, mask
        )
        if instance_count < args.min_object_points:
            raise RuntimeError(
                f"the segmentation gives {instance_count} valid 3-D points, below "
                f"--min-object-points={args.min_object_points}"
            )

        sweep = SweepVolumeParams.from_gripper_config(args.gripper)
        started = time.monotonic()
        with GraspGenXClient(
            host=args.host, port=args.port, timeout_ms=args.timeout_ms
        ) as client:
            health = client.health()
            metadata = client.server_metadata
            if health.get("status") != "ok":
                raise RuntimeError(f"GraspGenX health check failed: {health}")
            if "infer_scene_pc" not in metadata.get("actions", []):
                raise RuntimeError(
                    "this server does not advertise infer_scene_pc; "
                    f"it offers {metadata.get('actions')}"
                )
            results = client.infer_scene_pc(
                point_cloud=cloud,
                instance_mask=instance_mask,
                sweep_volume_params=sweep,
                planner=args.planner,
                min_object_points=args.min_object_points,
                num_grasps=args.num_grasps,
                grasp_threshold=args.grasp_threshold,
                topk_num_grasps=args.topk,
                return_branch_tags=True,
            )
        round_trip_ms = (time.monotonic() - started) * 1000.0

        empty_poses = np.empty((0, 4, 4), dtype=np.float32)
        empty_scores = np.empty((0,), dtype=np.float32)
        grasps_camera, scores, tags = results.get(1, (empty_poses, empty_scores, []))
        grasps_world, hand_poses = graspgen.transform_grasp_poses(
            grasps_camera, T_world_camera
        )

        # Before believing any of it: the proposals were made for these points,
        # so after conversion the fingers must close near them.
        frames = graspgen.fingertip_agreement(
            hand_poses,
            object_points_world,
            fingertip_depth_m=PANDA_FINGERTIP_DEPTH_M,
            tolerance_m=args.frame_tolerance,
        )

        ranking = gs.rank_candidates(
            gs.candidates_from_arrays(hand_poses, scores),
            tool_pose,
            scene,
            minimum_downwardness=args.minimum_downwardness,
        )
        chosen = [candidate.hand_pose for candidate in ranking["ordered"]]
        chosen_scores = np.array(
            [entry["score"] for entry in ranking["kept"]], dtype=np.float32
        )

        np.save(output / "hand_poses_world.npy", np.asarray(hand_poses, dtype=np.float32))
        np.save(output / "scores.npy", np.asarray(scores, dtype=np.float32))
        np.save(
            output / "selected_hand_poses_world.npy",
            np.asarray(chosen, dtype=np.float32).reshape(-1, 4, 4),
        )
        np.save(output / "selected_scores.npy", chosen_scores)
        if len(hand_poses):
            fingertips = (
                np.asarray(hand_poses)[:, :3, 3]
                + np.asarray(hand_poses)[:, :3, 2] * PANDA_FINGERTIP_DEPTH_M
            )
            write_colored_cloud(
                output / "proposed_fingertips.ply", fingertips, (250, 190, 40)
            )

        checks = {
            "server_answered": bool(health.get("status") == "ok"),
            "proposals_returned": bool(len(hand_poses) > 0),
            "frame_conversion_lands_on_the_object": bool(frames["passed"]),
            "at_least_one_grasp_on_the_intended_part": bool(
                ranking["counts"]["on_the_intended_part"] > 0
            ),
        }
        report = {
            "status": "success" if all(checks.values()) else "failed_checks",
            "reference": {
                "server": "NVIDIA GraspGenX, infer_scene_pc, unmodified",
                "grasp_to_hand": "GraspGenX end2end/robots/franka_panda.yaml",
                "part_choice": (
                    "ground truth in Phase 2; SAM3 takes it over in Phase 3"
                ),
            },
            "capture": str(args.capture),
            "segmentation": str(segmentation),
            "parameters": {
                "planner": args.planner,
                "gripper": args.gripper,
                "num_grasps": args.num_grasps,
                "topk_num_grasps": args.topk,
                "grasp_threshold": args.grasp_threshold,
                "minimum_downwardness": args.minimum_downwardness,
                "round_trip_ms": round(round_trip_ms, 1),
            },
            "server": {"health": health, "metadata": metadata},
            "input_points": instance_count,
            "frame_check": frames,
            "selection": {
                "counts": ranking["counts"],
                "kept": ranking["kept"][:10],
                "rejected_sample": ranking["rejected"][:10],
                **gs.rejection_summary(ranking),
            },
            "pose_quality": graspgen.pose_quality(np.asarray(hand_poses)),
            "inspect": {"fingertips": str(output / "proposed_fingertips.ply")},
            "automatic_checks": checks,
            "next_step": (
                f"python scripts/plan_handover.py --capture {args.capture} "
                f"--grasps {output}"
            ),
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        counts = ranking["counts"]
        print(
            f"proposed {counts['proposed']}, on {scene.grasp_part_name}: "
            f"{counts['on_the_intended_part']}, rejected {counts['rejected']}",
            flush=True,
        )
        print(
            "frame check: median fingertip-to-object "
            f"{frames.get('median_distance_m')} m -> {frames['passed']}",
            flush=True,
        )
        print(f"GRASPS {report['status'].upper()} -> {report_path}", flush=True)
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

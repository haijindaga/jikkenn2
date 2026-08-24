#!/usr/bin/env python3
"""Look at the grasps GraspGenX proposed, next to the tool and the robot.

Phase 2 was debugged from JSON for two batches, which is the mistake this
project was started to avoid.  "The approach is wrong" is a sentence nobody can
check; a shape is something anybody can.

    source ~/GraspGenX/.venv/bin/activate
    python scripts/view_grasps.py --trial outputs/phase2b/arr_002

Then open http://localhost:8080.

    green   the points SAM3 segmented, which the proposals were made from
    red     where the tool's dangerous part really is
    blue    where its safe part really is
    orange  proposals kept: the fingers close on the dangerous part
    grey    proposals rejected
    Panda   drawn at the pose the capture recorded

Each proposal is drawn as a T: the long stroke is the approach, from wrist to
fingertips, and the cross stroke is the line the fingers close along.
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
from jikkenn2 import ground_truth as gt  # noqa: E402
from jikkenn2.joints import select_named_joint_positions  # noqa: E402
from jikkenn2.pointcloud import gripper_markers, subsample  # noqa: E402
from jikkenn2.reachability import PANDA_FINGERTIP_DEPTH_M  # noqa: E402
from jikkenn2.scene_spec import DEFAULT_SCENE  # noqa: E402

SEGMENT_COLOR = (40, 200, 90)
KEPT_COLOR = (250, 150, 30)
REJECTED_COLOR = (130, 130, 130)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--robot", default="franka.yml")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--point-size", type=float, default=0.004)
    parser.add_argument(
        "--max-rejected",
        type=int,
        default=25,
        help="How many rejected proposals to draw before the view is unreadable",
    )
    return parser.parse_args()


def part_cloud(tool_pose, part, samples: int = 12) -> np.ndarray:
    """Points filling a part's box, so its true position is visible."""
    minimum, maximum = part.aabb_local_m()
    grid = [
        np.linspace(low, high, samples)
        for low, high in zip(np.asarray(minimum), np.asarray(maximum))
    ]
    local = np.stack(np.meshgrid(*grid, indexing="ij"), axis=-1).reshape(-1, 3)
    surface = local[
        np.any(
            (np.abs(local - np.asarray(minimum)) < 1e-9)
            | (np.abs(local - np.asarray(maximum)) < 1e-9),
            axis=1,
        )
    ]
    return (surface @ np.asarray(tool_pose)[:3, :3].T + np.asarray(tool_pose)[:3, 3]).astype(
        np.float32
    )


def colored(points: np.ndarray, color) -> np.ndarray:
    return np.tile(np.asarray(color, dtype=np.uint8), (len(points), 1))


def main() -> int:
    args = parse_args()
    scene = DEFAULT_SCENE
    grasps_dir = args.trial / "grasps"
    segmentation_dir = args.trial / "segmentation"

    hand_poses = np.load(grasps_dir / "hand_poses_world.npy").astype(np.float64)
    scores = np.load(grasps_dir / "scores.npy").astype(np.float64)
    segmented = np.load(segmentation_dir / "points_world.npy").astype(np.float32)
    tool_pose = np.load(args.trial / "tool_pose_world.npy").astype(np.float64)
    joint_positions = np.load(args.trial / "panda_joint_positions.npy")
    joint_names = tuple(
        json.loads((args.trial / "capture_check.json").read_text(encoding="utf-8"))[
            "robot"
        ]["joint_names"]
    )

    ranking = gs.rank_candidates(
        gs.candidates_from_arrays(hand_poses, scores), tool_pose, scene
    )
    kept_indices = [entry["index"] for entry in ranking["kept"]]
    rejected_indices = [entry["index"] for entry in ranking["rejected"]][
        : args.max_rejected
    ]

    import torch
    from curobo.types import ContentPath, JointState
    from curobo.viewer import ViserVisualizer

    visualizer = ViserVisualizer(
        content_path=ContentPath(robot_config_file=args.robot),
        add_robot_to_scene=True,
        connect_ip=args.host,
        connect_port=args.port,
        add_control_frames=False,
        visualize_robot_spheres=False,
    )
    viewer_names = tuple(visualizer.joint_names)
    visualizer.set_joint_state(
        JointState.from_position(
            torch.from_numpy(
                select_named_joint_positions(
                    joint_names, joint_positions, viewer_names
                ).astype(np.float32)
            )
            .to(device="cuda")
            .unsqueeze(0),
            joint_names=list(viewer_names),
        )
    )

    def show(name: str, points: np.ndarray, color, size=None) -> None:
        if len(points) == 0:
            return
        visualizer.add_point_cloud(
            pointcloud=np.asarray(points, dtype=np.float32),
            colors=colored(points, color),
            point_size=size or args.point_size,
            name=name,
        )

    show("/segmented_points", subsample(segmented, 60000), SEGMENT_COLOR)
    for part in scene.tool_parts:
        color = tuple(int(round(value * 255)) for value in part.color)
        show(f"/true_{part.name}", part_cloud(tool_pose, part), color, size=0.005)
    show(
        "/rejected_grasps",
        gripper_markers(
            hand_poses[rejected_indices], fingertip_depth_m=PANDA_FINGERTIP_DEPTH_M
        ),
        REJECTED_COLOR,
        size=0.003,
    )
    show(
        "/kept_grasps",
        gripper_markers(
            hand_poses[kept_indices], fingertip_depth_m=PANDA_FINGERTIP_DEPTH_M
        ),
        KEPT_COLOR,
        size=0.005,
    )

    counts = ranking["counts"]
    print(f"proposals: {counts['proposed']}", flush=True)
    print(f"  kept (on {scene.grasp_part_name}): {counts['on_the_intended_part']}", flush=True)
    print(f"  rejected: {counts['rejected']}  (showing {len(rejected_indices)})", flush=True)
    print(f"  segmented points: {len(segmented)}", flush=True)
    print(f"viewer: http://localhost:{args.port}", flush=True)
    print("orange T = kept grasp, long stroke is the approach; Ctrl+C to stop", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Look at the map the planner will actually use, next to the robot.

This is the gate jikkenn1 never had.  Its map passed every automatic check
while holding roughly 0.3 cubic metres of obstacle that was not there, wrapped
around the arm; ten seconds in a viewer would have shown it.

    source ~/GraspGenX/.venv/bin/activate
    python scripts/view_map.py --map outputs/trial_001/map --capture outputs/trial_001

Then open http://localhost:8080.  Red is what the planner treats as blocked,
grey is the measured surface.  The Panda is drawn at the pose the capture
recorded, and it should be standing in clear space: red touching the arm means
the map is wrong.
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

from jikkenn2.joints import select_named_joint_positions  # noqa: E402
from jikkenn2.mapping import blocked_voxel_centers  # noqa: E402
from jikkenn2.pointcloud import subsample  # noqa: E402

BLOCKED_COLOR = (220, 60, 30)
SURFACE_COLOR = (150, 150, 150)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--robot", default="franka.yml")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--point-size", type=float, default=0.01)
    parser.add_argument("--max-points", type=int, default=250000)
    parser.add_argument(
        "--surface-only",
        action="store_true",
        help="Show only the measured surface, not the whole blocked volume",
    )
    return parser.parse_args()


def colored(points: np.ndarray, color) -> np.ndarray:
    return np.tile(np.asarray(color, dtype=np.uint8), (len(points), 1))


def main() -> int:
    args = parse_args()
    grid = json.loads((args.map / "grid.json").read_text(encoding="utf-8"))
    features = np.load(args.map / "esdf_features.npy")
    surface = np.load(args.map / "occupied_points_robot_base.npy")

    blocked = (
        np.empty((0, 3), dtype=np.float32)
        if args.surface_only
        else subsample(blocked_voxel_centers(features, grid), args.max_points)
    )
    surface = subsample(surface, args.max_points)

    joint_positions = np.load(args.capture / "panda_joint_positions.npy")
    capture_report = json.loads(
        (args.capture / "capture_check.json").read_text(encoding="utf-8")
    )
    joint_names = tuple(capture_report["robot"]["joint_names"])

    import torch
    from curobo.types import ContentPath, JointState
    from curobo.viewer import ViserVisualizer

    visualizer = ViserVisualizer(
        content_path=ContentPath(robot_config_file=args.robot),
        add_robot_to_scene=True,
        connect_ip=args.host,
        connect_port=args.port,
        add_control_frames=False,
        visualize_robot_spheres=True,
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
    if len(surface):
        visualizer.add_point_cloud(
            pointcloud=surface.astype(np.float32),
            colors=colored(surface, SURFACE_COLOR),
            point_size=args.point_size,
            name="/measured_surface",
        )
    if len(blocked):
        visualizer.add_point_cloud(
            pointcloud=blocked.astype(np.float32),
            colors=colored(blocked, BLOCKED_COLOR),
            point_size=args.point_size,
            name="/blocked_voxels",
        )

    print(f"blocked voxels shown: {len(blocked)}", flush=True)
    print(f"surface points shown: {len(surface)}", flush=True)
    print(f"viewer: http://localhost:{args.port}", flush=True)
    print("red should not touch the arm; Ctrl+C to stop", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

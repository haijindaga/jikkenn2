#!/usr/bin/env python3
"""Compute where on the table the Panda can grasp, and draw it on the table.

Every tabletop cell is tested with eight hand poses — four top-down yaws and
four side approaches — through cuRobo's official IK solver, with self-collision
and the table enabled.  A cell is green when all eight solve, yellow when some
do, red when none do.

    source ~/GraspGenX/.venv/bin/activate
    python scripts/reachability_map.py --output outputs/reachability \
           --overlay assets/overlay_reach.usd

Then place objects on green:

    conda activate env_isaaclab
    python scripts/arrange_scene.py --scene assets/scene.usd \
           --overlay assets/overlay_reach.usd

``--dry-run`` skips cuRobo entirely and writes a grey overlay.  Use it to check
that the overlay renders in the viewport before trusting any colour.
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

from jikkenn2.reachability import (  # noqa: E402
    candidate_hand_poses,
    classify_cells,
    default_grasp_height_m,
    summarize_labels,
    tabletop_grid,
    write_overlay_usd,
)
from jikkenn2.scene_spec import DEFAULT_SCENE  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "reachability")
    parser.add_argument("--overlay", type=Path, default=REPO_ROOT / "assets" / "overlay_reach.usd")
    parser.add_argument("--cell-size", type=float, default=0.02)
    parser.add_argument("--grasp-height", type=float, default=None)
    parser.add_argument("--robot", default="franka.yml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-ik-seeds", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip cuRobo and write a grey overlay, to test rendering only",
    )
    return parser.parse_args()


def solve_ik_batch(poses: np.ndarray, args: argparse.Namespace, scene) -> np.ndarray:
    """Return a boolean success per pose, using cuRobo's official IK solver.

    ``poses`` is ``(N, 4, 4)`` in the robot base frame.  The table is the only
    world obstacle: obstacles on it move between arrangements, so a map that
    baked them in would be wrong the moment anything is dragged.
    """
    import torch
    from curobo._src.geom.types import Cuboid, SceneCfg
    from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
    from curobo._src.solver.solver_ik import IKSolver
    from curobo._src.types.device_cfg import DeviceCfg
    from curobo._src.types.tool_pose import GoalToolPose

    if not torch.cuda.is_available():
        raise RuntimeError("cuRobo IK requires CUDA")

    from jikkenn2.reachability import PANDA_FINGERTIP_DEPTH_M  # noqa: F401

    device_cfg = DeviceCfg(device=torch.device(args.device), dtype=torch.float32)
    table = Cuboid(
        name="table",
        pose=[*scene.table_center_m, 1.0, 0.0, 0.0, 0.0],
        dims=list(scene.table_size_m),
    )
    planner_cfg = MotionPlannerCfg.create(
        robot=args.robot,
        scene_model=SceneCfg(cuboid=[table]),
        device_cfg=device_cfg,
        num_ik_seeds=args.num_ik_seeds,
        num_trajopt_seeds=2,
        optimizer_collision_activation_distance=0.01,
        use_cuda_graph=False,
        random_seed=123,
    )
    solver = IKSolver(planner_cfg.ik_solver_config, None)

    from jikkenn2.geometry import quaternion_wxyz_from_rotation_matrix

    quaternions = np.stack(
        [quaternion_wxyz_from_rotation_matrix(pose[:3, :3]) for pose in poses]
    )
    positions = poses[:, :3, 3]

    success = np.zeros(len(poses), dtype=bool)
    for start in range(0, len(poses), args.chunk_size):
        stop = min(start + args.chunk_size, len(poses))
        position = torch.from_numpy(
            positions[start:stop].astype(np.float32)
        ).to(device_cfg.device)
        quaternion = torch.from_numpy(
            quaternions[start:stop].astype(np.float32)
        ).to(device_cfg.device)
        goals = GoalToolPose(
            tool_frames=["panda_hand"],
            position=position[:, None, None, None, :],
            quaternion=quaternion[:, None, None, None, :],
        )
        result = solver.solve_pose(goals, return_seeds=1)
        solved = result.success
        solved = solved.detach().cpu().numpy().reshape(stop - start, -1).any(axis=1)
        success[start:stop] = solved
        print(
            f"  ik {stop}/{len(poses)}  solved so far {int(success[:stop].sum())}",
            flush=True,
        )
    solver.destroy()
    return success


def main() -> int:
    args = parse_args()
    try:
        return _run(args)
    except Exception as error:
        # Same rule as build_scene_usd.py: never fail without leaving something
        # to read.
        import traceback

        failure = {
            "status": "failure",
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "overlay_path": str(args.overlay),
            "output_path": str(args.output),
            "hint": (
                "A PermissionError here usually means the repository contains "
                "root-owned files from an earlier sudo run. Check with "
                "'ls -la assets/' and fix with "
                "'sudo chown -R \"$USER\":\"$USER\" .' from the repository root."
            ),
        }
        try:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "reachability_check.json").write_text(
                json.dumps(failure, indent=2) + "\n", encoding="utf-8"
            )
            print(f"failure report: {args.output / 'reachability_check.json'}", file=sys.stderr)
        except OSError:
            print("could not write a failure report either", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        raise


def _run(args: argparse.Namespace) -> int:
    scene = DEFAULT_SCENE

    layout = scene.validation_report()
    if layout["status"] != "success":
        raise RuntimeError("scene layout is invalid; run scripts/validate_scene.py")

    grasp_height = (
        default_grasp_height_m(scene) if args.grasp_height is None else args.grasp_height
    )
    grid = tabletop_grid(scene, cell_size_m=args.cell_size, grasp_height_m=grasp_height)
    poses, orientation_names = candidate_hand_poses(grid.flat_centers_m())
    cells, orientations = poses.shape[0], poses.shape[1]
    print(
        f"grid {grid.shape[0]}x{grid.shape[1]} = {cells} cells "
        f"x {orientations} orientations = {cells * orientations} IK problems",
        flush=True,
    )
    print(f"grasp height: {grasp_height:.4f} m", flush=True)

    started = time.monotonic()
    if args.dry_run:
        labels = np.full(cells, "unknown", dtype=object)
        backend = "dry_run"
        success = None
    else:
        flat = poses.reshape(-1, 4, 4)
        solved = solve_ik_batch(flat, args, scene).reshape(cells, orientations)
        labels = classify_cells(solved)
        backend = "curobo_ik"
        success = solved
    elapsed_s = time.monotonic() - started

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "cell_centers_m.npy", grid.centers_m.astype(np.float32))
    np.save(args.output / "labels.npy", np.asarray(labels, dtype=object), allow_pickle=True)
    if success is not None:
        np.save(args.output / "ik_success.npy", success)

    overlay = write_overlay_usd(args.overlay, scene, grid, labels)

    report = {
        "status": "success",
        "backend": backend,
        "reference": {
            "solver": "cuRobo IKSolver.solve_pose with self-collision and the table",
            "tool_frame": "panda_hand",
            "fingertip_offset": "GraspGenX franka_panda.yaml fingertip depth",
        },
        "grid": {
            "shape": list(grid.shape),
            "cell_size_m": grid.cell_size_m,
            "min_corner_m": list(grid.min_corner_m),
            "grasp_height_m": grasp_height,
        },
        "orientations": orientation_names,
        "summary": summarize_labels(labels),
        "overlay": overlay,
        "elapsed_s": round(elapsed_s, 2),
        "note": (
            "Only the table is in the world model. Obstacles move between "
            "arrangements, so baking them into this map would make it wrong as "
            "soon as anything is dragged. This is a placement aid, not a safety "
            "gate; the planner checks the real scene."
        ),
        "next_step": (
            "python scripts/arrange_scene.py --scene assets/scene.usd "
            f"--overlay {args.overlay}"
        ),
    }
    report_path = args.output / "reachability_check.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    summary = report["summary"]["counts"]
    print(
        "cells  free={free}  partial={partial}  blocked={blocked}  unknown={unknown}".format(
            **summary
        ),
        flush=True,
    )
    print(f"overlay: {args.overlay}", flush=True)
    print(f"REACHABILITY {backend.upper()} -> {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

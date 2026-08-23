#!/usr/bin/env python3
"""Check the Phase 0 scene layout before anything is built or planned.

Runs on plain NumPy: no Isaac Sim, no CUDA, no GPU.  Every claim about the
layout is machine-checked here, so a bad camera placement or an unreachable
handover pose is caught before a stage is authored.

    python scripts/validate_scene.py
    python scripts/validate_scene.py --output outputs/scene_validation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/scene_validation.json"),
        help="Where to write the full report",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Only print the failing checks"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from jikkenn2.scene_spec import DEFAULT_SCENE

    report = DEFAULT_SCENE.validation_report()
    checks = report["automatic_checks"]
    derived = report["derived"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if not args.quiet:
        half_h, half_v = derived["camera_half_fov_deg"]
        print(f"camera half FOV        : {half_h:.2f}deg horizontal, {half_v:.2f}deg vertical")
        print("table corners (h, v, depth):")
        for angles in derived["table_corner_angles_deg_and_depth_m"]:
            print(f"  {angles[0]:>7.2f}deg {angles[1]:>7.2f}deg {angles[2]:>6.3f}m")
        print("robot base / top (h, v, depth):")
        for angles in derived["robot_probe_angles_deg_and_depth_m"]:
            print(f"  {angles[0]:>7.2f}deg {angles[1]:>7.2f}deg {angles[2]:>6.3f}m")
        print(f"handover distance      : {derived['handover_distance_from_base_m']:.3f} m")
        print(f"tool start distance    : {derived['tool_start_distance_from_base_m']:.3f} m")
        print(
            "robot shadow on table  : "
            f"{derived['robot_shadow_samples_on_table']} sampled points"
        )
        print()

    width = max(len(name) for name in checks)
    for name, passed in checks.items():
        if passed and args.quiet:
            continue
        print(f"{'PASS' if passed else 'FAIL'}  {name.ljust(width)}")

    print()
    print(f"status: {report['status']}  ->  {args.output}")
    return 0 if report["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())

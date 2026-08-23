#!/usr/bin/env python3
"""Open the Phase 0 stage in the Isaac Sim GUI and record hand-made placements.

Drag the tool and the obstacles wherever you want, then press Enter in this
terminal.  The current poses are written to ``arrangements/arr_NNN.json`` and
you can keep going — one GUI session produces as many arrangements as you like.

    conda activate env_isaaclab
    python scripts/arrange_scene.py --scene assets/scene.usd

Keys in the viewport: right-drag orbits, middle-drag pans, the wheel zooms, F
frames the selection, W is the move gizmo and E is the rotate gizmo.  Do not
press Play: physics settling happens later, in capture_scene.py.

Commands in this terminal:
    Enter  save the current placement as a new arrangement
    l      list what is currently placed, without saving
    q      quit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import select
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from jikkenn2.scene_spec import DEFAULT_SCENE  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=REPO_ROOT / "assets" / "scene.usd")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "arrangements")
    parser.add_argument(
        "--overlay",
        type=Path,
        help="Reachability overlay USD to lay over the table while placing",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Snapshot the stage as-is and exit; no GUI, no interaction",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Accepted for symmetry with the documented command; the GUI is the default",
    )
    return parser.parse_args()


def _read_terminal_line() -> str | None:
    """Return a typed line if one is waiting, without blocking the GUI."""
    try:
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
    except (OSError, ValueError):  # no usable stdin (e.g. piped from /dev/null)
        return None
    if not ready:
        return None
    line = sys.stdin.readline()
    if not line:
        return None
    return line.strip().lower()


def _snapshot(stage, scene, source_stage: str) -> dict:
    from jikkenn2.arrangement import collect_arrangement

    return collect_arrangement(stage, scene, source_stage=source_stage)


def _print_snapshot(arrangement: dict) -> None:
    from jikkenn2.arrangement import describe_placement

    for entry in arrangement["objects"]:
        print(describe_placement(entry), flush=True)
    problems = arrangement["objects_outside_the_working_area"]
    if problems:
        print(
            f"  -> {len(problems)} object(s) outside the working area: "
            + ", ".join(path.rsplit("/", 1)[-1] for path in problems),
            flush=True,
        )


def main() -> int:
    args = parse_args()
    scene = DEFAULT_SCENE
    stage_path = args.scene.resolve()
    if not stage_path.is_file():
        raise FileNotFoundError(
            f"{stage_path} does not exist; run scripts/build_scene_usd.py first"
        )

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": bool(args.headless)})

    saved: list[str] = []
    try:
        import omni.usd
        from jikkenn2.arrangement import next_arrangement_path, save_arrangement

        context = omni.usd.get_context()
        context.open_stage(str(stage_path))
        for _ in range(60):
            simulation_app.update()
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError(f"Isaac could not open {stage_path}")

        # Layered in memory only: the overlay is a placement aid and must never
        # become part of the saved stage.
        if args.overlay is not None:
            overlay_path = args.overlay.resolve()
            if not overlay_path.is_file():
                raise FileNotFoundError(
                    f"{overlay_path} does not exist; run scripts/reachability_map.py first"
                )
            prim = stage.DefinePrim("/World/ReachabilityOverlay", "Xform")
            prim.GetReferences().AddReference(str(overlay_path))
            for _ in range(30):
                simulation_app.update()
            print(f"overlay: {overlay_path}", flush=True)
            print("  green = every approach solves, yellow = some, red = none", flush=True)

        if args.headless:
            arrangement = _snapshot(stage, scene, str(stage_path))
            destination = save_arrangement(
                arrangement, next_arrangement_path(args.output)
            )
            _print_snapshot(arrangement)
            print(f"SAVED {destination}", flush=True)
            saved.append(str(destination))
            return 0

        print("", flush=True)
        print("=" * 68, flush=True)
        print(f"stage : {stage_path}", flush=True)
        print(f"output: {args.output}", flush=True)
        print("drag the tool and obstacles in the viewport, then:", flush=True)
        print("  Enter = save this placement    l = list    q = quit", flush=True)
        print("do NOT press Play; settling happens later in capture_scene.py", flush=True)
        print("=" * 68, flush=True)

        while simulation_app.is_running():
            simulation_app.update()
            command = _read_terminal_line()
            if command is None:
                continue
            if command in ("q", "quit", "exit"):
                break
            arrangement = _snapshot(stage, scene, str(stage_path))
            _print_snapshot(arrangement)
            if command == "l":
                print("  (not saved)", flush=True)
                continue
            destination = save_arrangement(
                arrangement, next_arrangement_path(args.output)
            )
            saved.append(str(destination))
            print(f"SAVED {destination}  [{len(saved)} this session]", flush=True)
    finally:
        summary = {
            "status": "success",
            "stage": str(stage_path),
            "arrangements_saved": saved,
            "count": len(saved),
        }
        report_path = REPO_ROOT / "outputs" / "arrange_session.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"SESSION {len(saved)} arrangement(s) -> {report_path}", flush=True)
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

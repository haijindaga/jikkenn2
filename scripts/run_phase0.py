#!/usr/bin/env python3
"""Run every saved arrangement through capture, plan, execute and score.

Each stage runs as its own process, because they need different environments
and because an 8 GB card only fits one heavy process at a time.  One trial
failing does not stop the batch: the run continues and the report says which
arrangements did not get through.

    python scripts/run_phase0.py --arrangements arrangements/ --output outputs/phase0

The interpreters can be given explicitly if the defaults are wrong:

    python scripts/run_phase0.py \
        --isaac-python  ~/miniconda3/envs/env_isaaclab/bin/python \
        --curobo-python ~/GraspGenX/.venv/bin/python
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from jikkenn2.batch import (  # noqa: E402
    STAGES,
    Interpreters,
    arrangement_paths,
    batch_summary,
    classify_stage_result,
    completed_trials,
    stage_command,
    trial_directory,
)

DEFAULT_ISAAC_PYTHON = Path.home() / "miniconda3/envs/env_isaaclab/bin/python"
DEFAULT_CUROBO_PYTHON = Path.home() / "GraspGenX/.venv/bin/python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrangements", type=Path, default=REPO_ROOT / "arrangements")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "phase0")
    parser.add_argument("--scene", type=Path, default=REPO_ROOT / "assets" / "scene.usd")
    parser.add_argument("--isaac-python", type=Path, default=DEFAULT_ISAAC_PYTHON)
    parser.add_argument("--curobo-python", type=Path, default=DEFAULT_CUROBO_PYTHON)
    parser.add_argument("--plain-python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--stages",
        default=",".join(STAGES),
        help="Comma-separated subset of stages to run, in order",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip arrangements that already have a score",
    )
    parser.add_argument("--limit", type=int, default=0, help="Stop after N arrangements")
    parser.add_argument("--timeout", type=int, default=1800, help="Per-stage seconds")
    return parser.parse_args()


def run_stage(command: list[str], log_path: Path, timeout: int) -> dict:
    """Run one stage, keeping its whole output on disk."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        returncode = completed.returncode
        output = completed.stdout + completed.stderr
    except subprocess.TimeoutExpired as expired:
        returncode = -1
        output = (expired.stdout or "") + (expired.stderr or "")
        output += f"\n[timed out after {timeout}s]\n"
    log_path.write_text(output, encoding="utf-8")
    tail = [line for line in output.splitlines() if line.strip()][-3:]
    return {
        "returncode": returncode,
        "seconds": round(time.monotonic() - started, 1),
        "log": str(log_path),
        "tail": tail,
    }


def main() -> int:
    args = parse_args()
    stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()]
    unknown = [stage for stage in stages if stage not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stages: {unknown}; expected from {STAGES}")

    interpreters = Interpreters(
        isaac=args.isaac_python.expanduser(),
        curobo=args.curobo_python.expanduser(),
        plain=args.plain_python.expanduser(),
    )
    missing = interpreters.missing()
    if missing:
        raise SystemExit(
            "these interpreters do not exist: "
            + ", ".join(missing)
            + "\nPass --isaac-python / --curobo-python / --plain-python explicitly."
        )
    if not args.scene.is_file():
        raise SystemExit(f"{args.scene} does not exist; run build_scene_usd.py first")

    arrangements = arrangement_paths(args.arrangements)
    if not arrangements:
        raise SystemExit(f"no arr_*.json in {args.arrangements}")
    already = completed_trials(args.output) if args.resume else set()
    if already:
        arrangements = [
            path
            for path in arrangements
            if trial_directory(args.output, path).name not in already
        ]
        print(f"resuming: {len(already)} already scored", flush=True)
    if args.limit:
        arrangements = arrangements[: args.limit]

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"{len(arrangements)} arrangement(s), stages: {', '.join(stages)}", flush=True)

    rows = []
    scores = []
    for index, arrangement in enumerate(arrangements, start=1):
        trial = trial_directory(args.output, arrangement)
        trial.mkdir(parents=True, exist_ok=True)
        row = {"trial": trial.name, "arrangement": str(arrangement), "stages": {}}
        outcome = "ok"
        print(f"[{index}/{len(arrangements)}] {trial.name}", flush=True)

        for stage in stages:
            command = stage_command(
                stage,
                repo_root=REPO_ROOT,
                interpreters=interpreters,
                arrangement=arrangement,
                trial=trial,
                scene=args.scene,
            )
            result = run_stage(command, trial / f"{stage}.log", args.timeout)
            row["stages"][stage] = result
            outcome, keep_going = classify_stage_result(stage, result["returncode"])
            print(f"    {stage:<8} {outcome:<16} {result['seconds']:>6.1f}s", flush=True)
            if not keep_going:
                for line in result["tail"]:
                    print(f"      {line}", flush=True)
                break

        row["outcome"] = outcome
        score_path = trial / "score.json"
        if score_path.is_file():
            score = json.loads(score_path.read_text(encoding="utf-8"))
            scores.append(score)
            row["score_status"] = score["status"]
            row["failed_criteria"] = score.get("failed", [])
        rows.append(row)

    summary = batch_summary(rows, scores)
    summary_path = args.output / "phase0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    scoring = summary["scoring"]
    print("", flush=True)
    if scoring.get("trials"):
        print(
            f"scored {scoring['measured_trials']}/{scoring['trials']} trials, "
            f"success rate {scoring['success_rate']}",
            flush=True,
        )
        for name, rate in scoring["success_rate_per_criterion"].items():
            print(f"  {name:<28} {rate}", flush=True)
    if summary["errored_before_scoring"]:
        print(
            "did not reach scoring: "
            + ", ".join(summary["errored_before_scoring"]),
            flush=True,
        )
    print(f"PHASE0 {summary['status'].upper()} -> {summary_path}", flush=True)
    return 0 if summary["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())

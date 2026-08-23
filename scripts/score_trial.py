#!/usr/bin/env python3
"""Apply the six criteria to an executed trial.

Needs neither Isaac Sim nor a GPU: it reads the artifacts the earlier stages
wrote and decides.  Keeping the decision here means a criterion can change
without re-running anything.

    python scripts/score_trial.py --trial outputs/trial_001
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from jikkenn2 import scoring  # noqa: E402
from jikkenn2.scene_spec import DEFAULT_SCENE  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--execution", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def load_trial(trial: Path, plan_dir: Path | None, execution_dir: Path | None) -> dict:
    plan_dir = plan_dir or (trial / "plan")
    execution_dir = execution_dir or (trial / "execution")

    execution_path = execution_dir / "execution_log.json"
    if not execution_path.is_file():
        raise FileNotFoundError(f"no execution log at {execution_path}")
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    if execution.get("status") != "success":
        raise ValueError(
            f"{execution_path} reports status={execution.get('status')!r}; there is "
            "nothing to score"
        )
    target_path = plan_dir / "target_tool_pose_world.npy"
    if not target_path.is_file():
        raise FileNotFoundError(f"no planned handover pose at {target_path}")
    return {
        "execution": execution,
        "target_tool_pose": np.load(target_path),
        "plan_dir": plan_dir,
        "execution_dir": execution_dir,
    }


def main() -> int:
    args = parse_args()
    output = args.output or (args.trial / "score.json")
    trial = load_trial(args.trial, args.plan, args.execution)

    result = scoring.score(
        DEFAULT_SCENE,
        trial["execution"],
        target_tool_pose=trial["target_tool_pose"],
    )
    result["trial"] = str(args.trial)
    result["sources"] = {
        "plan": str(trial["plan_dir"]),
        "execution": str(trial["execution_dir"]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    if not args.quiet:
        width = max(len(entry["criterion"]) for entry in result["criteria"])
        for entry in result["criteria"]:
            mark = "PASS" if entry["passed"] else "FAIL"
            print(f"{mark}  {entry['criterion'].ljust(width)}  {entry['measured']}")
        print()
    print(
        f"TRIAL {'PASSED' if result['trial_passed'] else 'FAILED'} "
        f"{result['passed_count']}/{result['total_count']} -> {output}",
        flush=True,
    )
    return 0 if result["trial_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

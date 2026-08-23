"""Running every arrangement through the four stages, unattended.

The stages need different environments -- Isaac for capture and execution,
the cuRobo venv for planning -- so each runs as its own process.  That is also
what makes an 8 GB card work: only one heavy process is alive at a time.

Command construction and result assembly live here so they can be tested
without launching anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

STAGES = ("capture", "map", "plan", "execute", "score")

#: Which interpreter each stage needs.
STAGE_ENVIRONMENT = {
    "capture": "isaac",
    "map": "curobo",
    "plan": "curobo",
    "execute": "isaac",
    "score": "plain",
}

#: What each phase replaces. Phase 0 plans against the true boxes; Phase 1
#: builds the world from the captured depth instead, and nothing else changes.
PHASE_STAGES = {
    0: ("capture", "plan", "execute", "score"),
    1: ("capture", "map", "plan", "execute", "score"),
}

ARRANGEMENT_PATTERN = re.compile(r"^arr_(\d+)\.json$")

#: score_trial.py's exit codes.
SCORE_PASSED = 0
SCORE_FAILED = 2
SCORE_NOT_MEASURABLE = 3


@dataclass(frozen=True)
class Interpreters:
    """Python executables for each environment."""

    isaac: Path
    curobo: Path
    plain: Path

    def for_stage(self, stage: str) -> Path:
        return getattr(self, STAGE_ENVIRONMENT[stage])

    def missing(self) -> list[str]:
        return [
            f"{name}={path}"
            for name, path in (
                ("isaac", self.isaac),
                ("curobo", self.curobo),
                ("plain", self.plain),
            )
            if not Path(path).is_file()
        ]


def arrangement_paths(directory: str | Path) -> list[Path]:
    """Every saved arrangement, in numeric order."""
    folder = Path(directory)
    found = []
    for path in folder.glob("arr_*.json"):
        match = ARRANGEMENT_PATTERN.match(path.name)
        if match:
            found.append((int(match.group(1)), path))
    return [path for _, path in sorted(found)]


def trial_directory(output_root: str | Path, arrangement: str | Path) -> Path:
    return Path(output_root) / Path(arrangement).stem


def stage_command(
    stage: str,
    *,
    repo_root: str | Path,
    interpreters: Interpreters,
    arrangement: str | Path,
    trial: str | Path,
    scene: str | Path,
    use_map: bool = False,
) -> list[str]:
    """The argv for one stage of one trial.

    ``use_map`` is what makes a run Phase 1: the planner is pointed at the
    measured ESDF instead of the ground-truth boxes.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
    scripts = Path(repo_root) / "scripts"
    python = str(interpreters.for_stage(stage))
    trial = str(trial)
    if stage == "capture":
        return [
            python,
            str(scripts / "capture_scene.py"),
            "--scene", str(scene),
            "--arrangement", str(arrangement),
            "--output", trial,
        ]
    if stage == "map":
        return [python, str(scripts / "build_map.py"), "--capture", trial]
    if stage == "plan":
        command = [python, str(scripts / "plan_handover.py"), "--capture", trial]
        if use_map:
            command += ["--map", str(Path(trial) / "map")]
        return command
    if stage == "execute":
        return [
            python,
            str(scripts / "execute_handover.py"),
            "--scene", str(scene),
            "--capture", trial,
        ]
    return [python, str(scripts / "score_trial.py"), "--trial", trial, "--quiet"]


def phase_stages(phase: int) -> tuple[str, ...]:
    if phase not in PHASE_STAGES:
        raise ValueError(f"unknown phase {phase}; expected one of {sorted(PHASE_STAGES)}")
    return PHASE_STAGES[phase]


def classify_stage_result(stage: str, returncode: int) -> tuple[str, bool]:
    """Return ``(outcome, keep_going)`` for one finished stage.

    Only ``score`` distinguishes more than pass and fail: a trial the robot got
    wrong is a result, and the batch must carry on to the next arrangement.
    """
    if stage == "score":
        if returncode == SCORE_PASSED:
            return "passed", True
        if returncode == SCORE_FAILED:
            return "failed_criteria", True
        if returncode == SCORE_NOT_MEASURABLE:
            return "not_measurable", True
        return "error", True
    return ("ok", True) if returncode == 0 else ("error", False)


def completed_trials(output_root: str | Path) -> set[str]:
    """Trials that already carry a score, for resuming an interrupted batch."""
    root = Path(output_root)
    if not root.is_dir():
        return set()
    return {
        child.name
        for child in root.iterdir()
        if child.is_dir() and (child / "score.json").is_file()
    }


def batch_summary(rows: list[dict], scores: list[dict]) -> dict:
    """Assemble the run report from per-trial rows and their scores."""
    from jikkenn2.scoring import summarize

    errored = [row["trial"] for row in rows if row["outcome"] == "error"]
    return {
        "status": "success" if not errored else "completed_with_errors",
        "attempted": len(rows),
        "errored_before_scoring": errored,
        "scoring": summarize(scores),
        "trials": rows,
    }

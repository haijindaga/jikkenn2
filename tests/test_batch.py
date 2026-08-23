"""Tests for the batch runner's command construction and bookkeeping."""

from __future__ import annotations

import json

import pytest

from jikkenn2 import batch


@pytest.fixture
def interpreters(tmp_path):
    paths = {}
    for name in ("isaac", "curobo", "plain"):
        path = tmp_path / f"{name}_python"
        path.write_text("", encoding="utf-8")
        paths[name] = path
    return batch.Interpreters(**paths)


def test_arrangements_come_back_in_numeric_order(tmp_path):
    for name in ("arr_010.json", "arr_002.json", "arr_001.json", "notes.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    names = [path.name for path in batch.arrangement_paths(tmp_path)]
    assert names == ["arr_001.json", "arr_002.json", "arr_010.json"]


def test_arrangement_numbering_is_not_sorted_as_text(tmp_path):
    for index in (2, 10):
        (tmp_path / f"arr_{index:03d}.json").write_text("{}", encoding="utf-8")
    names = [path.name for path in batch.arrangement_paths(tmp_path)]
    assert names == ["arr_002.json", "arr_010.json"]


def test_a_trial_is_named_after_its_arrangement(tmp_path):
    trial = batch.trial_directory(tmp_path / "out", tmp_path / "arr_007.json")
    assert trial.name == "arr_007"


def test_each_stage_uses_the_environment_it_needs(interpreters, tmp_path):
    commands = {
        stage: batch.stage_command(
            stage,
            repo_root=tmp_path,
            interpreters=interpreters,
            arrangement=tmp_path / "arr_001.json",
            trial=tmp_path / "out" / "arr_001",
            scene=tmp_path / "scene.usd",
        )
        for stage in batch.STAGES
    }
    assert commands["capture"][0] == str(interpreters.isaac)
    assert commands["execute"][0] == str(interpreters.isaac)
    assert commands["plan"][0] == str(interpreters.curobo)
    assert commands["score"][0] == str(interpreters.plain)


def test_each_stage_calls_its_own_script(interpreters, tmp_path):
    for stage, script in (
        ("capture", "capture_scene.py"),
        ("plan", "plan_handover.py"),
        ("execute", "execute_handover.py"),
        ("score", "score_trial.py"),
    ):
        command = batch.stage_command(
            stage,
            repo_root=tmp_path,
            interpreters=interpreters,
            arrangement=tmp_path / "arr_001.json",
            trial=tmp_path / "out" / "arr_001",
            scene=tmp_path / "scene.usd",
        )
        assert command[1].endswith(script)


def test_every_stage_is_pointed_at_the_same_trial(interpreters, tmp_path):
    trial = tmp_path / "out" / "arr_003"
    for stage in batch.STAGES:
        command = batch.stage_command(
            stage,
            repo_root=tmp_path,
            interpreters=interpreters,
            arrangement=tmp_path / "arr_003.json",
            trial=trial,
            scene=tmp_path / "scene.usd",
        )
        assert str(trial) in command


def test_an_unknown_stage_is_refused(interpreters, tmp_path):
    with pytest.raises(ValueError, match="unknown stage"):
        batch.stage_command(
            "polish",
            repo_root=tmp_path,
            interpreters=interpreters,
            arrangement=tmp_path / "arr_001.json",
            trial=tmp_path / "t",
            scene=tmp_path / "scene.usd",
        )


def test_missing_interpreters_are_named(tmp_path):
    interpreters = batch.Interpreters(
        isaac=tmp_path / "nope", curobo=tmp_path / "also_nope", plain=tmp_path / "nope3"
    )
    missing = interpreters.missing()
    assert len(missing) == 3
    assert any("isaac=" in entry for entry in missing)


def test_a_failed_stage_stops_that_trial_but_a_bad_score_does_not():
    assert batch.classify_stage_result("capture", 1) == ("error", False)
    assert batch.classify_stage_result("plan", 2) == ("error", False)
    assert batch.classify_stage_result("capture", 0) == ("ok", True)


def test_score_exit_codes_are_distinguished():
    assert batch.classify_stage_result("score", 0) == ("passed", True)
    assert batch.classify_stage_result("score", 2) == ("failed_criteria", True)
    assert batch.classify_stage_result("score", 3) == ("not_measurable", True)
    assert batch.classify_stage_result("score", 9) == ("error", True)


def test_a_robot_failure_never_stops_the_batch():
    for code in (0, 2, 3, 9):
        _, keep_going = batch.classify_stage_result("score", code)
        assert keep_going is True


def test_completed_trials_are_those_with_a_score(tmp_path):
    (tmp_path / "arr_001").mkdir()
    (tmp_path / "arr_001" / "score.json").write_text("{}", encoding="utf-8")
    (tmp_path / "arr_002").mkdir()
    assert batch.completed_trials(tmp_path) == {"arr_001"}


def test_completed_trials_of_a_missing_directory_is_empty(tmp_path):
    assert batch.completed_trials(tmp_path / "nothing") == set()


def _score(passed: bool, measured: bool = True) -> dict:
    return {
        "status": "success" if passed else "failed_criteria",
        "trial_passed": passed,
        "fully_measured": measured,
        "criteria": [
            {"criterion": "grasped_the_intended_part", "passed": passed, "measurable": measured}
        ],
    }


def test_summary_reports_trials_that_never_reached_scoring():
    rows = [
        {"trial": "arr_001", "outcome": "passed"},
        {"trial": "arr_002", "outcome": "error"},
    ]
    summary = batch.batch_summary(rows, [_score(True)])
    assert summary["status"] == "completed_with_errors"
    assert summary["errored_before_scoring"] == ["arr_002"]
    assert summary["attempted"] == 2
    assert summary["scoring"]["trials"] == 1


def test_a_clean_batch_reports_success():
    rows = [{"trial": "arr_001", "outcome": "passed"}]
    summary = batch.batch_summary(rows, [_score(True)])
    assert summary["status"] == "success"
    assert summary["errored_before_scoring"] == []
    assert summary["scoring"]["success_rate"] == pytest.approx(1.0)


def test_summary_success_rate_counts_only_measured_trials():
    rows = [{"trial": f"arr_{i:03d}", "outcome": "passed"} for i in range(1, 4)]
    scores = [_score(True), _score(False), _score(True, measured=False)]
    summary = batch.batch_summary(rows, scores)
    assert summary["scoring"]["trials"] == 3
    assert summary["scoring"]["measured_trials"] == 2
    assert summary["scoring"]["success_rate"] == pytest.approx(0.5)


def test_summary_is_json_serialisable():
    rows = [{"trial": "arr_001", "outcome": "passed"}]
    json.dumps(batch.batch_summary(rows, [_score(True)]))


def test_phase_zero_skips_the_map_stage():
    assert "map" not in batch.phase_stages(0)
    assert batch.phase_stages(0) == ("capture", "plan", "execute", "score")


def test_phase_one_builds_a_map_before_planning():
    stages = batch.phase_stages(1)
    assert stages.index("map") < stages.index("plan")
    assert stages.index("capture") < stages.index("map")


def test_an_unknown_phase_is_refused():
    with pytest.raises(ValueError, match="unknown phase"):
        batch.phase_stages(7)


def test_the_map_stage_runs_in_the_curobo_environment(interpreters, tmp_path):
    command = batch.stage_command(
        "map",
        repo_root=tmp_path,
        interpreters=interpreters,
        arrangement=tmp_path / "arr_001.json",
        trial=tmp_path / "out" / "arr_001",
        scene=tmp_path / "scene.usd",
    )
    assert command[0] == str(interpreters.curobo)
    assert command[1].endswith("build_map.py")


def test_the_planner_is_pointed_at_the_map_only_in_phase_one(interpreters, tmp_path):
    trial = tmp_path / "out" / "arr_001"
    without = batch.stage_command(
        "plan",
        repo_root=tmp_path,
        interpreters=interpreters,
        arrangement=tmp_path / "arr_001.json",
        trial=trial,
        scene=tmp_path / "scene.usd",
        use_map=False,
    )
    with_map = batch.stage_command(
        "plan",
        repo_root=tmp_path,
        interpreters=interpreters,
        arrangement=tmp_path / "arr_001.json",
        trial=trial,
        scene=tmp_path / "scene.usd",
        use_map=True,
    )
    assert "--map" not in without
    assert "--map" in with_map
    assert with_map[with_map.index("--map") + 1] == str(trial / "map")


def test_a_failed_map_abandons_that_trial():
    assert batch.classify_stage_result("map", 2) == ("error", False)
    assert batch.classify_stage_result("map", 0) == ("ok", True)


def test_phase_two_segments_and_proposes_before_planning():
    stages = batch.phase_stages(2)
    assert stages.index("segment") < stages.index("grasp") < stages.index("plan")
    assert stages.index("capture") < stages.index("segment")
    assert "map" in stages, "phase 2 keeps the measured world from phase 1"


def test_each_phase_adds_exactly_one_layer():
    """One replacement per phase is the rule the whole plan rests on."""
    zero, one, two = (set(batch.phase_stages(n)) for n in (0, 1, 2))
    assert one - zero == {"map"}
    assert two - one == {"segment", "grasp"}


def test_segmentation_runs_where_sam3_lives(interpreters, tmp_path):
    command = batch.stage_command(
        "segment",
        repo_root=tmp_path,
        interpreters=interpreters,
        arrangement=tmp_path / "arr_001.json",
        trial=tmp_path / "out" / "arr_001",
        scene=tmp_path / "scene.usd",
    )
    assert command[0] == str(interpreters.isaac)
    assert command[1].endswith("segment_tool.py")


def test_proposing_grasps_runs_where_graspgenx_lives(interpreters, tmp_path):
    command = batch.stage_command(
        "grasp",
        repo_root=tmp_path,
        interpreters=interpreters,
        arrangement=tmp_path / "arr_001.json",
        trial=tmp_path / "out" / "arr_001",
        scene=tmp_path / "scene.usd",
    )
    assert command[0] == str(interpreters.curobo)
    assert command[1].endswith("propose_grasps.py")


def test_the_planner_is_given_the_grasps_only_in_phase_two(interpreters, tmp_path):
    trial = tmp_path / "out" / "arr_001"
    kwargs = dict(
        repo_root=tmp_path,
        interpreters=interpreters,
        arrangement=tmp_path / "arr_001.json",
        trial=trial,
        scene=tmp_path / "scene.usd",
    )
    phase_one = batch.stage_command("plan", use_map=True, use_grasps=False, **kwargs)
    phase_two = batch.stage_command("plan", use_map=True, use_grasps=True, **kwargs)
    assert "--grasps" not in phase_one
    assert "--map" in phase_one
    assert phase_two[phase_two.index("--grasps") + 1] == str(trial / "grasps")

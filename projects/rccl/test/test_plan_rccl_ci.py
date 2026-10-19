"""CI policy tests: exercise selection, dependency failures, and dispatch I/O."""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ci" / "scripts"))
import plan_rccl_ci as planner


@pytest.mark.parametrize("event", ["schedule", "workflow_dispatch"])
def test_default_nightly_suites(event):
    assert set(planner.select_suites(event)) == set(planner.SUITES)
    assert planner.families_for("auto", planner.select_suites(event)) == list(
        planner.FAMILIES
    )


@pytest.mark.parametrize("event", ["pull_request", "push", ""])
def test_presubmit_preserves_disabled_gpu_suites(event):
    assert planner.select_suites(event) == ["single-node", "rocprof"]
    assert (
        planner.runnable_suites("pull_request", "auto", "gfx950-dcgpu", "success", "")
        == []
    )


@pytest.mark.parametrize("suite", list(planner.SUITES))
def test_each_suite_is_independently_selectable(suite):
    selected = planner.select_suites("workflow_dispatch", suite)
    assert selected == [suite]
    assert planner.families_for(suite, selected) == [planner.SUITES[suite][0]]


def test_multiple_suites_select_both_families_without_duplicates():
    selection = "jax, accl-profiler,jax"
    selected = planner.select_suites("workflow_dispatch", selection)
    assert selected == ["jax", "accl-profiler"]
    assert planner.families_for(selection, selected) == list(planner.FAMILIES)


@pytest.mark.parametrize(
    "selection", ["", "unknown", "jax,", "auto,jax", "accl-profiler\nother=value"]
)
def test_invalid_selection_fails_closed(selection):
    with pytest.raises(ValueError):
        planner.select_suites("workflow_dispatch", selection)


@pytest.mark.parametrize(
    "build,artifact,expected",
    [
        ("success", "", ["accl-profiler"]),
        ("skipped", "123", ["accl-profiler"]),
        ("skipped", "", []),
        ("failure", "", []),
        ("failure", "123", []),
        ("cancelled", "123", []),
    ],
)
def test_build_dependency_and_reuse(build, artifact, expected):
    assert (
        planner.runnable_suites(
            "workflow_dispatch", "accl-profiler", "gfx950-dcgpu", build, artifact
        )
        == expected
    )


@pytest.mark.parametrize("event", ["push", "pull_request"])
@pytest.mark.parametrize(
    "files,expected",
    [
        (["projects/rccl/src/foo.cc"], True),
        ([".github/workflows/therock-rccl-ci-linux.yml"], True),
        (["projects/hip/foo.cc"], False),
        ([], False),
    ],
)
def test_change_gate(monkeypatch, event, files, expected):
    monkeypatch.setattr(planner, "changed_files", lambda: files)
    run, scope, _ = planner.plan(
        event, "smoke", datetime(2026, 10, 3, tzinfo=timezone.utc)
    )
    assert run is expected
    assert scope == "smoke"


@pytest.mark.parametrize(
    "event,day,requested,expected",
    [
        ("schedule", 3, "smoke", "all"),
        ("schedule", 2, "smoke", "smoke"),
        ("workflow_dispatch", 3, "smoke", "smoke"),
        ("workflow_dispatch", 2, "all", "all"),
    ],
)
def test_scope_policy(monkeypatch, event, day, requested, expected):
    monkeypatch.setattr(
        planner,
        "changed_files",
        lambda: pytest.fail("Manual/nightly must not diff HEAD^"),
    )
    assert planner.plan(event, requested, datetime(2026, 10, day, tzinfo=timezone.utc))[
        :2
    ] == (True, expected)


def test_cli_outputs_reusable_workflow_plan(tmp_path):
    output = tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable,
            planner.__file__,
            "--event-name",
            "workflow_dispatch",
            "--test-suites",
            "jax,accl-profiler",
            "--amdgpu-family",
            "gfx950-dcgpu",
            "--build-result",
            "skipped",
            "--artifact-run-id",
            "123",
        ],
        env={**os.environ, "GITHUB_OUTPUT": str(output)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text().split("=", 1)[1]) == ["accl-profiler"]


def test_cli_outputs_focused_build_matrix(tmp_path):
    output = tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable,
            planner.__file__,
            "--event-name",
            "workflow_dispatch",
            "--test-suites",
            "accl-profiler",
        ],
        env={**os.environ, "GITHUB_OUTPUT": str(output)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert values == {
        "run_linux_ci": "true",
        "test_scope": "smoke",
        "amdgpu_families": '["gfx950-dcgpu"]',
    }


def test_cli_rejects_unknown_suite_before_writing_outputs(tmp_path):
    output = tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable,
            planner.__file__,
            "--event-name",
            "workflow_dispatch",
            "--test-suites",
            "unknown",
        ],
        env={**os.environ, "GITHUB_OUTPUT": str(output)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert not output.exists()

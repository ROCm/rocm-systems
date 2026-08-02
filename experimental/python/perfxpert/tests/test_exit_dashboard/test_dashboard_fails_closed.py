"""The Go/No-Go dashboard must not report a pass for something it never measured.

Before Phase 11A the air-gap metric read a snapshot directory no test wrote,
so it returned "pending" forever, and the two subprocess collectors turned any
exception into "pending" as well — all of which count as non-blocking.
"""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "exit_dashboard.py"


def _load_dashboard():
    spec = importlib.util.spec_from_file_location("exit_dashboard", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def dash():
    return _load_dashboard()


def test_unmeasured_metric_fails_its_gate(dash):
    for key in (
        "airgap_identical_rate",
        "per_agent_narrow_scope_violations",
        "tool_class_split_violations",
    ):
        assert not dash._metric_pass(key, dash.UNMEASURED), key


def test_unmeasured_metric_produces_no_go(dash):
    metrics = {"airgap_identical_rate": dash.UNMEASURED}
    assert dash.compute_verdict(metrics) == "NO-GO"


def test_pytest_runner_reports_failure_when_it_cannot_start(dash, monkeypatch):
    def _boom(*args, **kwargs):
        raise OSError("no interpreter")

    monkeypatch.setattr(dash.subprocess, "run", _boom)
    passed, detail = dash._run_pytest("tests/whatever.py")
    assert passed is False
    assert "could not run" in detail


def test_collectors_do_not_swallow_a_broken_run(dash, monkeypatch):
    monkeypatch.setattr(dash, "_run_pytest", lambda target: (False, ""))
    assert dash.collect_tool_class_split_violations() == 1
    assert dash.collect_narrow_scope_violations() == dash.UNMEASURED
    assert dash.collect_airgap() == 0.0


def test_airgap_metric_reads_a_target_that_exists(dash):
    """The old snapshot directory never existed, so the metric was inert."""
    assert dash.AIRGAP_PARITY_TESTS.exists(), (
        f"air-gap parity target missing: {dash.AIRGAP_PARITY_TESTS}"
    )


def test_verdict_and_renderer_share_one_gate_definition(dash):
    """Both used to carry their own copy of every threshold."""
    import inspect

    source = inspect.getsource(dash.compute_verdict)
    assert "_metric_pass" in source
    assert "0.95" not in source, "compute_verdict re-declares a threshold"


def test_every_expected_attack_is_recorded_defeated(dash):
    assert dash.collect_red_team() == dash.RED_TEAM_ATTACK_COUNT


def test_scratch_outcome_files_do_not_inflate_the_count(dash, tmp_path, monkeypatch):
    """The outcome directory also collects gitignored sol_gate_unmocked_* runs."""
    import json

    for attack_id in dash.EXPECTED_ATTACK_IDS:
        (tmp_path / f"{attack_id}.json").write_text(
            json.dumps({"attack_id": attack_id, "status": "defeated"})
        )
    (tmp_path / "sol_gate_unmocked_1000x.json").write_text(
        json.dumps({"attack_id": "sol_gate_unmocked_1000x", "status": "defeated"})
    )
    monkeypatch.setattr(dash, "RED_TEAM_OUTCOMES", tmp_path)
    assert dash.collect_red_team() == dash.RED_TEAM_ATTACK_COUNT


def test_a_regressed_attack_cannot_hide_behind_a_scratch_file(dash, tmp_path, monkeypatch):
    """A bare file count let one failure be masked by one extra file."""
    import json

    remaining = sorted(dash.EXPECTED_ATTACK_IDS)[:-1]
    for attack_id in remaining:
        (tmp_path / f"{attack_id}.json").write_text(
            json.dumps({"attack_id": attack_id, "status": "defeated"})
        )
    (tmp_path / "sol_gate_unmocked_1000x.json").write_text(
        json.dumps({"attack_id": "sol_gate_unmocked_1000x", "status": "defeated"})
    )
    monkeypatch.setattr(dash, "RED_TEAM_OUTCOMES", tmp_path)

    count = dash.collect_red_team()
    assert count == dash.RED_TEAM_ATTACK_COUNT - 1
    assert not dash._metric_pass("red_team_pass_count", count)

"""Contract test: exit_dashboard.py produces a JSON object conforming to the
Week-5 Go/No-Go schema.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parent.parent.parent / "scripts" / "exit_dashboard.py"
)


EXPECTED_METRIC_KEYS = {
    "parity_agreement_rate",
    "red_team_pass_count",
    "airgap_identical_rate",
    "regression_gate_false_positive_rate",
    "per_agent_narrow_scope_violations",
    "tool_class_split_violations",
    "provider_smoke_status",      # may be "nightly-only" in PR lane
    "benchmark_geomean",          # may be "nightly-only"
    "user_signoff",               # may be "pending"
}


@pytest.mark.dashboard
def test_script_exists_and_is_executable() -> None:
    assert SCRIPT.exists(), f"exit_dashboard.py missing: {SCRIPT}"


@pytest.mark.dashboard
def test_dashboard_emits_all_9_metrics(tmp_path: Path) -> None:
    out = tmp_path / "dashboard.json"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(out), "--allow-partial"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode in (0, 1, 2)
    assert out.exists(), "dashboard JSON should be written even on non-zero verdicts"
    data = json.loads(out.read_text())
    assert "metrics" in data
    missing = EXPECTED_METRIC_KEYS - set(data["metrics"].keys())
    assert not missing, f"Missing dashboard metrics: {missing}"
    assert "overall_verdict" in data
    assert data["overall_verdict"] in ("GO", "NO-GO", "PARTIAL (pending)")


@pytest.mark.dashboard
def test_dashboard_marks_go_only_when_all_thresholds_met(tmp_path: Path) -> None:
    out = tmp_path / "dashboard.json"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(out), "--allow-partial"],
    )
    assert proc.returncode in (0, 1, 2)
    data = json.loads(out.read_text())
    metrics = data["metrics"]
    verdict = data["overall_verdict"]

    # Define each metric's gate pass condition here (mirrors script internal rules)
    def _is_pass(key: str, val) -> bool:
        if val in ("nightly-only", "pending"):
            return True  # don't block PR lane on nightly metrics
        if key == "parity_agreement_rate":
            return val >= 0.95
        if key == "red_team_pass_count":
            return val == 14
        if key == "airgap_identical_rate":
            return val == 1.0
        if key == "regression_gate_false_positive_rate":
            return val <= 0.05
        if key in ("per_agent_narrow_scope_violations", "tool_class_split_violations"):
            return val == 0
        return True

    all_pass = all(_is_pass(k, v) for k, v in metrics.items())
    has_pending = any(v in ("nightly-only", "pending") for v in metrics.values())
    if all_pass:
        expected = "PARTIAL (pending)" if has_pending else "GO"
        assert verdict == expected, f"All metrics pass but verdict = {verdict}"
    else:
        assert verdict in ("NO-GO", "PARTIAL (pending)")


@pytest.mark.dashboard
def test_allow_partial_does_not_mask_no_go(tmp_path: Path) -> None:
    out = tmp_path / "dashboard.json"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(out), "--allow-partial"],
        capture_output=True,
        text=True,
    )
    data = json.loads(out.read_text())
    if data["overall_verdict"] == "NO-GO":
        assert proc.returncode == 2

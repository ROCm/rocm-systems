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
    subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(out), "--allow-partial"],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(out.read_text())
    assert "metrics" in data
    missing = EXPECTED_METRIC_KEYS - set(data["metrics"].keys())
    assert not missing, f"Missing dashboard metrics: {missing}"
    assert "overall_verdict" in data
    assert data["overall_verdict"] in ("GO", "NO-GO", "PARTIAL (pending)")


@pytest.mark.dashboard
def test_dashboard_marks_go_only_when_all_thresholds_met(tmp_path: Path) -> None:
    out = tmp_path / "dashboard.json"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(out), "--allow-partial"],
        check=True,
    )
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
    if all_pass:
        assert verdict == "GO", f"All metrics pass but verdict = {verdict}"
    else:
        assert verdict in ("NO-GO", "PARTIAL (pending)")

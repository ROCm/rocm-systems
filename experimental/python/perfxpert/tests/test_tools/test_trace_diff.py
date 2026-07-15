"""Tests for perfxpert.tools.trace_diff."""

from types import SimpleNamespace

from perfxpert.tools import regression, trace_diff
from perfxpert.tools._class import ToolClass


def _runtime(kernel_name: str, total_runtime_ns: int) -> SimpleNamespace:
    return SimpleNamespace(
        kernel_name=kernel_name,
        total_runtime_ns=total_runtime_ns,
    )


def test_primary_regressions_are_not_truncated_by_top_kernels(monkeypatch):
    baseline = [
        _runtime("k1", 300),
        _runtime("k2", 200),
        _runtime("k3", 100),
    ]
    new = [
        _runtime("k1", 300),
        _runtime("k2", 200),
        _runtime("k3", 125),
    ]

    monkeypatch.setattr(
        regression,
        "extract_kernel_runtimes_from_db",
        lambda path: baseline if "baseline" in path else new,
    )
    monkeypatch.setattr(regression, "identify_hot_kernels", lambda _path: [])

    result = trace_diff.diff_runs("baseline.db", "new.db", top_kernels=1)

    assert [row["name"] for row in result["per_kernel"]] == ["k1"]
    assert any(
        row["name"] == "k3" for row in result["primary_regressions"]
    ), result["primary_regressions"]


def test_verdict_is_regressed_when_primary_regressions_exist(monkeypatch):
    baseline = [
        _runtime("k1", 100),
        _runtime("k2", 100),
        _runtime("k3", 100),
    ]
    new = [
        _runtime("k1", 98),
        _runtime("k2", 98),
        _runtime("k3", 104),
    ]

    monkeypatch.setattr(
        regression,
        "extract_kernel_runtimes_from_db",
        lambda path: baseline if "baseline" in path else new,
    )
    monkeypatch.setattr(regression, "identify_hot_kernels", lambda _path: [])

    result = trace_diff.diff_runs("baseline.db", "new.db", top_kernels=10)

    assert any(
        row["name"] == "k3" for row in result["primary_regressions"]
    ), result["primary_regressions"]
    assert result["wall_delta_pct"] < result["verdict_threshold_pct"]
    assert result["verdict"] == "regressed"


def test_narrative_does_not_claim_within_noise_when_regressions_exist(monkeypatch):
    baseline = [
        _runtime("k1", 100),
        _runtime("k2", 100),
        _runtime("k3", 100),
    ]
    new = [
        _runtime("k1", 98),
        _runtime("k2", 98),
        _runtime("k3", 104),
    ]

    monkeypatch.setattr(
        regression,
        "extract_kernel_runtimes_from_db",
        lambda path: baseline if "baseline" in path else new,
    )
    monkeypatch.setattr(regression, "identify_hot_kernels", lambda _path: [])

    result = trace_diff.diff_runs("baseline.db", "new.db", top_kernels=10)

    assert result["verdict"] == "regressed"
    headline = result["narrative"].splitlines()[0].lower()
    assert headline.startswith("kernel-level regressions detected"), headline


def test_trace_diff_is_read_only():
    assert trace_diff.diff_runs.__tool_class__ == ToolClass.READ_ONLY

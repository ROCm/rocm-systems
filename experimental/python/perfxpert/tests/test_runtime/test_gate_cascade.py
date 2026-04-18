"""Tests for perfxpert.runtime.gate_cascade — the deterministic 5-gate pipeline.

Mocks the execution tools (compile.build etc.) so the cascade logic is
tested in isolation. Red-team attacks on specific gates live in
tests/test_red_team/ (Phase 5).
"""

import pytest
from unittest.mock import MagicMock

from perfxpert.runtime import gate_cascade
from perfxpert.runtime.gate_cascade import GateVerdict


@pytest.fixture
def patches(monkeypatch):
    """Stub every execution tool the cascade calls."""
    stubs = {
        "compile": MagicMock(return_value={"ok": True, "stderr": ""}),
        "sol": MagicMock(return_value={"ok": True, "peak_ratio": 0.8}),
        "bitwise": MagicMock(return_value={"ok": True, "diff": None}),
        "regression": MagicMock(return_value={
            "ok": True, "total_delta_pct": -8.0, "hot_kernels": [],
            "weighted_geomean_delta_pct": -7.5,
        }),
        "anchors": MagicMock(return_value={"ok": True, "failed": []}),
    }
    monkeypatch.setattr(gate_cascade, "_run_compile_gate", stubs["compile"])
    monkeypatch.setattr(gate_cascade, "_run_sol_gate", stubs["sol"])
    monkeypatch.setattr(gate_cascade, "_run_bitwise_gate", stubs["bitwise"])
    monkeypatch.setattr(gate_cascade, "_run_regression_gate", stubs["regression"])
    monkeypatch.setattr(gate_cascade, "_run_anchors_gate", stubs["anchors"])
    return stubs


def test_all_gates_pass_returns_pass(patches):
    v = gate_cascade.evaluate(
        baseline_db="b.db", candidate_db="c.db",
        patch_file="foo.hip", patch_sha="abc123",
        gfx_id="gfx942", claimed_speedup=1.5,
    )
    assert isinstance(v, GateVerdict)
    assert v.status == "pass"
    assert v.failing_gate is None


def test_gates_execute_in_cascade_order(patches):
    gate_cascade.evaluate(
        baseline_db="b.db", candidate_db="c.db",
        patch_file="foo.hip", patch_sha="abc123",
        gfx_id="gfx942", claimed_speedup=1.5,
        candidate_binary="./my_binary",
    )
    # All 5 gates should have been called
    for name in ("compile", "sol", "bitwise", "regression", "anchors"):
        assert patches[name].called, f"gate '{name}' was not invoked"


def test_compile_failure_short_circuits(patches):
    patches["compile"].return_value = {"ok": False, "stderr": "undefined reference"}
    v = gate_cascade.evaluate(
        baseline_db="b.db", candidate_db="c.db",
        patch_file="foo.hip", patch_sha="abc123",
        gfx_id="gfx942", claimed_speedup=1.5,
    )
    assert v.status == "reject"
    assert v.failing_gate == "compile"
    # Subsequent gates should NOT have been called
    assert not patches["sol"].called


def test_sol_violation_flagged_as_reject(patches):
    """Anti-Sakana: claimed 1000× speedup rejected by SOL sanity."""
    patches["sol"].return_value = {"ok": False, "peak_ratio": 1500.0}
    v = gate_cascade.evaluate(
        baseline_db="b.db", candidate_db="c.db",
        patch_file="foo.hip", patch_sha="abc123",
        gfx_id="gfx942", claimed_speedup=1000.0,
    )
    assert v.status == "reject"
    assert v.failing_gate == "sol"


def test_bitwise_mismatch_rejects(patches):
    patches["bitwise"].return_value = {"ok": False, "diff": "max_abs=0.01"}
    v = gate_cascade.evaluate(
        baseline_db="b.db", candidate_db="c.db",
        patch_file="foo.hip", patch_sha="abc123",
        gfx_id="gfx942", claimed_speedup=1.5,
    )
    assert v.status == "reject"
    assert v.failing_gate == "bitwise"


def test_regression_over_threshold_returns_regressed(patches):
    """Gate 4: total_runtime +15% → 'regressed' (not 'reject')."""
    patches["regression"].return_value = {
        "ok": False,
        "total_delta_pct": 15.0,
        "hot_kernels": [{"name": "[K1]", "delta_pct": 15.0}],
        "weighted_geomean_delta_pct": 12.0,
    }
    v = gate_cascade.evaluate(
        baseline_db="b.db", candidate_db="c.db",
        patch_file="foo.hip", patch_sha="abc123",
        gfx_id="gfx942", claimed_speedup=1.5,
    )
    assert v.status == "regressed"
    assert v.failing_gate == "regression"
    assert v.delta_pct == 15.0


def test_weighted_geomean_catches_tail_regressions(patches):
    """Gate 4 FMEA fix: many small kernels each < 10% but weighted-geomean down."""
    patches["regression"].return_value = {
        "ok": False,
        "total_delta_pct": 2.0,            # total barely noticeable
        "hot_kernels": [],                 # no single hot kernel > 10%
        "weighted_geomean_delta_pct": 9.5,  # tail regressions add up
    }
    v = gate_cascade.evaluate(
        baseline_db="b.db", candidate_db="c.db",
        patch_file="foo.hip", patch_sha="abc123",
        gfx_id="gfx942", claimed_speedup=1.05,
    )
    assert v.status == "regressed"
    assert v.failing_gate == "regression"


def test_anchor_failure_rejects(patches):
    patches["anchors"].return_value = {"ok": False, "failed": ["test_conv_forward"]}
    v = gate_cascade.evaluate(
        baseline_db="b.db", candidate_db="c.db",
        patch_file="foo.hip", patch_sha="abc123",
        gfx_id="gfx942", claimed_speedup=1.5,
        candidate_binary="./my_binary",
    )
    assert v.status == "reject"
    assert v.failing_gate == "anchors"


def test_verdict_is_frozen(patches):
    v = gate_cascade.evaluate(
        baseline_db="b.db", candidate_db="c.db",
        patch_file="foo.hip", patch_sha="abc123",
        gfx_id="gfx942", claimed_speedup=1.5,
    )
    with pytest.raises((AttributeError, TypeError)):
        v.status = "reject"  # frozen dataclass


def test_debug_loop_caps_exist():
    """Spec §5: hard caps on optimization cycles."""
    assert gate_cascade.MAX_OPTIMIZATION_CYCLES_PER_KERNEL == 5
    assert gate_cascade.MAX_CONSECUTIVE_FAILURES == 3
    assert gate_cascade.MAX_SESSION_LLM_TURNS == 100


# -- Finding #20: short-circuit coverage for Gates 2 / 3 / 4 ----------------

def test_sol_failure_short_circuits_bitwise_regression_anchors(patches):
    """Gate 2 (SOL) failing must skip Gates 3, 4, 5 entirely."""
    patches["sol"].return_value = {"ok": False, "peak_ratio": 9999.0}
    v = gate_cascade.evaluate(
        baseline_db="b.db", candidate_db="c.db",
        patch_file="foo.hip", patch_sha="abc123",
        gfx_id="gfx942", claimed_speedup=1000.0,
    )
    assert v.status == "reject"
    assert v.failing_gate == "sol"
    # Downstream gates MUST NOT be invoked
    assert not patches["bitwise"].called, "Gate 3 (bitwise) must not run after Gate 2 fails"
    assert not patches["regression"].called, "Gate 4 (regression) must not run after Gate 2 fails"
    assert not patches["anchors"].called, "Gate 5 (anchors) must not run after Gate 2 fails"


def test_bitwise_failure_short_circuits_regression_anchors(patches):
    """Gate 3 (bitwise) failing must skip Gates 4 and 5 entirely."""
    patches["bitwise"].return_value = {"ok": False, "diff": "max_abs=0.5"}
    v = gate_cascade.evaluate(
        baseline_db="b.db", candidate_db="c.db",
        patch_file="foo.hip", patch_sha="abc123",
        gfx_id="gfx942", claimed_speedup=1.5,
    )
    assert v.status == "reject"
    assert v.failing_gate == "bitwise"
    # Gate 2 must have run (it passed)
    assert patches["sol"].called, "Gate 2 (SOL) should have been invoked before Gate 3"
    # Gates 4 and 5 MUST NOT be invoked
    assert not patches["regression"].called, "Gate 4 (regression) must not run after Gate 3 fails"
    assert not patches["anchors"].called, "Gate 5 (anchors) must not run after Gate 3 fails"


def test_regression_failure_short_circuits_anchors(patches):
    """Gate 4 (regression) failing must skip Gate 5 entirely."""
    patches["regression"].return_value = {
        "ok": False,
        "total_delta_pct": 25.0,
        "hot_kernels": [{"name": "[K_slow]", "delta_pct": 25.0}],
        "weighted_geomean_delta_pct": 20.0,
    }
    v = gate_cascade.evaluate(
        baseline_db="b.db", candidate_db="c.db",
        patch_file="foo.hip", patch_sha="abc123",
        gfx_id="gfx942", claimed_speedup=1.5,
    )
    assert v.status == "regressed"
    assert v.failing_gate == "regression"
    # Gates 2 and 3 must have run (they passed)
    assert patches["sol"].called, "Gate 2 (SOL) should have been invoked"
    assert patches["bitwise"].called, "Gate 3 (bitwise) should have been invoked"
    # Gate 5 MUST NOT be invoked
    assert not patches["anchors"].called, "Gate 5 (anchors) must not run after Gate 4 fails"

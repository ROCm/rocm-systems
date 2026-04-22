"""Tests for perfxpert.runtime.gate_cascade — the deterministic 5-gate pipeline.

Mocks the execution tools (compile.build etc.) so the cascade logic is
tested in isolation. Red-team attacks on specific gates live in
tests/test_red_team/.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from perfxpert.runtime import gate_cascade
from perfxpert.runtime.gate_cascade import GateVerdict

FIX = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def patches(monkeypatch):
    """Stub every execution tool the cascade calls."""
    stubs = {
        "compile": MagicMock(return_value={"ok": True, "stderr": ""}),
        "sol": MagicMock(return_value={"ok": True, "peak_ratio": 0.8}),
        "bitwise": MagicMock(return_value={"ok": True, "diff": None}),
        "regression": MagicMock(return_value={
            "ok": True,
            "verdict": "improved",
            "total_delta_pct": -0.08,         # fraction: -8% improvement
            "weighted_geomean_delta_pct": -0.075,
            "per_kernel_deltas": [],           # real key (was "hot_kernels")
            "threshold_pct": 0.03,
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
    """Gate 4: total_runtime +15% → 'regressed' (not 'reject').

    Stub uses real regression.compare_runs schema: total_delta_pct is a fraction
    (0.15 = 15%), per_kernel_deltas replaces the old 'hot_kernels' key.
    """
    patches["regression"].return_value = {
        "ok": False,
        "verdict": "regressed",
        "total_delta_pct": 0.15,          # fraction: 15%
        "per_kernel_deltas": [            # real key — was "hot_kernels"
            {"kernel": "[K1]", "delta_pct": 0.15, "was_hot": True},
        ],
        "weighted_geomean_delta_pct": 0.12,
        "threshold_pct": 0.03,
    }
    v = gate_cascade.evaluate(
        baseline_db="b.db", candidate_db="c.db",
        patch_file="foo.hip", patch_sha="abc123",
        gfx_id="gfx942", claimed_speedup=1.5,
    )
    assert v.status == "regressed"
    assert v.failing_gate == "regression"
    assert v.delta_pct == pytest.approx(0.15)


def test_weighted_geomean_catches_tail_regressions(patches):
    """Gate 4 FMEA fix: many small kernels each < 10% but weighted-geomean down.

    Stub uses real schema: fractions throughout, per_kernel_deltas key.
    """
    patches["regression"].return_value = {
        "ok": False,
        "verdict": "regressed",
        "total_delta_pct": 0.02,            # fraction: 2% total — barely noticeable
        "per_kernel_deltas": [],            # real key — no single hot kernel > 10%
        "weighted_geomean_delta_pct": 0.095,  # fraction: 9.5% — tail regressions add up
        "threshold_pct": 0.03,
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


def test_run_sol_gate_does_not_mutate_tool_result(monkeypatch):
    returned = {"verdict": "sane", "peak_ratio": 0.8}

    class _FakeSol:
        @staticmethod
        def sanity_check(**kwargs):
            return returned

    monkeypatch.setitem(__import__("sys").modules, "perfxpert.tools.sol", _FakeSol)
    result = gate_cascade._run_sol_gate(1.5, "gfx942")

    assert result["ok"] is True
    assert "ok" not in returned


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


# ---------------------------------------------------------------------------
# Finding #2 — _run_sol_gate un-mocked integration tests
# ---------------------------------------------------------------------------

class TestRunSolGateUnmocked:
    """Call _run_sol_gate DIRECTLY (no mocking) — exercises the real sol.sanity_check
    integration path.  These are the tests the bug was hiding from.
    """

    def test_impossible_speedup_ratio_rejected(self):
        """1000× claimed speedup must be caught by the hard cap (> 50×)."""
        r = gate_cascade._run_sol_gate(claimed_speedup=1000.0, gfx_id="gfx942")
        assert r["ok"] is False, "1000× speedup should be rejected by hard cap"
        assert "exploit" in r["reason"].lower() or "exceeds" in r["reason"].lower()

    def test_reasonable_speedup_passes_cap(self):
        """2× speedup is well within the 50× hard cap."""
        r = gate_cascade._run_sol_gate(claimed_speedup=2.0, gfx_id="gfx942")
        assert r["ok"] is True

    def test_exactly_at_cap_boundary_passes(self):
        """50× is the maximum allowed — exactly at cap should pass."""
        r = gate_cascade._run_sol_gate(
            claimed_speedup=gate_cascade.SOL_MAX_REASONABLE_SPEEDUP,
            gfx_id="gfx942",
        )
        assert r["ok"] is True

    def test_one_over_cap_rejects(self):
        """50.001× exceeds the hard cap → reject."""
        r = gate_cascade._run_sol_gate(
            claimed_speedup=gate_cascade.SOL_MAX_REASONABLE_SPEEDUP + 0.001,
            gfx_id="gfx942",
        )
        assert r["ok"] is False

    def test_absolute_flops_within_peak_passes(self):
        """When achieved_flops_per_sec is supplied and within peak, gate passes."""
        # MI300X fp64 peak = 81.7 TFLOPS; 40 TFLOPS is plausible
        r = gate_cascade._run_sol_gate(
            claimed_speedup=2.0,
            gfx_id="gfx942",
            achieved_flops_per_sec=40e12,
            kernel_type="fp64",
        )
        assert r["ok"] is True, f"40 TFLOPS fp64 should be within MI300X peak: {r}"

    def test_absolute_flops_exceeding_peak_rejects(self):
        """When achieved_flops_per_sec exceeds hardware peak, gate rejects even if
        speedup ratio is modest.  This is the Sakana-style attack path."""
        # MI300X fp64 peak = 81.7 TFLOPS; claim 500 TFLOPS → impossible
        r = gate_cascade._run_sol_gate(
            claimed_speedup=3.0,   # ratio looks innocent
            gfx_id="gfx942",
            achieved_flops_per_sec=500e12,  # but absolute FLOPS exceed hardware peak
            kernel_type="fp64",
        )
        assert r["ok"] is False, (
            "500 TFLOPS fp64 exceeds MI300X peak 81.7 TFLOPS — must reject"
        )

    def test_evaluate_rejects_1000x_claimed_speedup_end_to_end(self, monkeypatch):
        """evaluate() with claimed_speedup=1000× must fail at Gate 2 even when
        all other gates are mocked as passing.  This would have silently passed
        before the fix (TypeError was swallowed)."""
        monkeypatch.setattr(
            gate_cascade, "_run_compile_gate",
            MagicMock(return_value={"ok": True, "stderr": ""}),
        )
        # Do NOT mock _run_sol_gate — it must run for real
        monkeypatch.setattr(
            gate_cascade, "_run_bitwise_gate",
            MagicMock(return_value={"ok": True, "diff": None}),
        )
        monkeypatch.setattr(
            gate_cascade, "_run_regression_gate",
            MagicMock(return_value={
                "ok": True,
                "total_delta_pct": 0.0,
                "weighted_geomean_delta_pct": 0.0,
                "per_kernel_deltas": [],
            }),
        )

        v = gate_cascade.evaluate(
            baseline_db="b.db", candidate_db="c.db",
            patch_file="foo.hip", patch_sha="abc123",
            gfx_id="gfx942",
            claimed_speedup=1000.0,
        )
        assert v.status == "reject", (
            f"1000× speedup must be rejected at Gate 2, got: {v.status}"
        )
        assert v.failing_gate == "sol"


# ---------------------------------------------------------------------------
# Finding #3 — Gate 4 wrong key + unit mismatch
# ---------------------------------------------------------------------------

class TestRegressionGateRealSchema:
    """Tests that pass regression.compare_runs real output schema through evaluate().

    regression.compare_runs returns:
      - key  "per_kernel_deltas"  (not "hot_kernels")
      - delta_pct as FRACTION (e.g. 0.15 = 15%), NOT percent
      - each item: {"kernel": str, "delta_pct": float, "was_hot": bool}

    Before the fix, evaluate() read "hot_kernels" (always empty → list) and
    compared the fraction 0.15 against 10.0 (a percent), so a hot kernel
    regressing 15% would NEVER be caught.
    """

    @staticmethod
    def _make_stubs_pass_except_regression(monkeypatch, regression_return):
        """Helper: stub compile/sol/bitwise/anchors as passing; let regression return
        the supplied dict (real schema from regression.compare_runs)."""
        monkeypatch.setattr(
            gate_cascade, "_run_compile_gate",
            MagicMock(return_value={"ok": True, "stderr": ""}),
        )
        monkeypatch.setattr(
            gate_cascade, "_run_sol_gate",
            MagicMock(return_value={"ok": True, "peak_ratio": 1.5}),
        )
        monkeypatch.setattr(
            gate_cascade, "_run_bitwise_gate",
            MagicMock(return_value={"ok": True, "diff": None}),
        )
        monkeypatch.setattr(
            gate_cascade, "_run_regression_gate",
            MagicMock(return_value=regression_return),
        )

    def test_hot_kernel_15pct_regressed_is_caught(self, monkeypatch):
        """A hot kernel regressing 15% (delta_pct=0.15 as fraction) MUST trigger
        Gate 4 failure.  Before the fix: 0.15 > 10.0 is False → silently passed."""
        regression_output = {
            "verdict": "regressed",
            "total_delta_pct": 0.02,      # 2% total — below 3% noise floor
            "weighted_geomean_delta_pct": 0.02,
            "per_kernel_deltas": [
                {"kernel": "matmul", "delta_pct": 0.15, "was_hot": True},
                {"kernel": "conv2d", "delta_pct": 0.01, "was_hot": False},
            ],
            "threshold_pct": 0.03,
        }
        self._make_stubs_pass_except_regression(monkeypatch, regression_output)

        v = gate_cascade.evaluate(
            baseline_db="b.db", candidate_db="c.db",
            patch_file="foo.hip", patch_sha="abc123",
            gfx_id="gfx942", claimed_speedup=1.1,
        )
        assert v.status == "regressed", (
            "matmul is hot and regressed 15% (fraction 0.15 > threshold 0.10) — "
            f"must be caught by Gate 4, got: {v.status!r}"
        )
        assert v.failing_gate == "regression"
        assert "hot kernel" in v.detail.lower() or "hot" in v.detail.lower()

    def test_non_hot_kernel_regression_does_not_trigger_hot_path(self, monkeypatch):
        """Only kernels with was_hot=True should trigger the hot-kernel failure path."""
        regression_output = {
            "verdict": "neutral",
            "total_delta_pct": 0.01,    # 1% total — within noise
            "weighted_geomean_delta_pct": 0.01,
            "per_kernel_deltas": [
                # delta_pct=0.15 (15%) but was_hot=False → should NOT trigger hot-kernel gate
                {"kernel": "tiny_kernel", "delta_pct": 0.15, "was_hot": False},
            ],
            "threshold_pct": 0.03,
        }
        self._make_stubs_pass_except_regression(monkeypatch, regression_output)

        v = gate_cascade.evaluate(
            baseline_db="b.db", candidate_db="c.db",
            patch_file="foo.hip", patch_sha="abc123",
            gfx_id="gfx942", claimed_speedup=1.1,
        )
        # total_delta 0.01 < 0.03 threshold, was_hot=False → pass
        assert v.status == "pass", (
            f"Non-hot kernel regression should not trigger Gate 4, got: {v.status!r}"
        )

    def test_real_regression_db_fixtures_flow_through_gate4(self, monkeypatch):
        """Use the real regression fixture DBs with _run_regression_gate un-mocked.

        regression.compare_runs(BASELINE, TAIL_HURT) returns a regressed verdict
        where conv2d (a hot kernel) regressed 15%.  Gate 4 in evaluate() must
        detect this WITHOUT mocking _run_regression_gate.
        """
        BASELINE = str(FIX / "regression_baseline.db")
        TAIL_HURT = str(FIX / "regression_tail_hurt.db")

        # Mock everything EXCEPT _run_regression_gate
        monkeypatch.setattr(
            gate_cascade, "_run_compile_gate",
            MagicMock(return_value={"ok": True, "stderr": ""}),
        )
        monkeypatch.setattr(
            gate_cascade, "_run_sol_gate",
            MagicMock(return_value={"ok": True, "peak_ratio": 1.0}),
        )
        monkeypatch.setattr(
            gate_cascade, "_run_bitwise_gate",
            MagicMock(return_value={"ok": True, "diff": None}),
        )
        # _run_regression_gate is NOT mocked — runs for real

        v = gate_cascade.evaluate(
            baseline_db=BASELINE,
            candidate_db=TAIL_HURT,
            patch_file="foo.hip", patch_sha="abc123",
            gfx_id="gfx942", claimed_speedup=1.0,
        )
        assert v.status == "regressed", (
            "regression_tail_hurt.db has conv2d (hot kernel) regressing 15%; "
            f"Gate 4 must catch this. Got: {v.status!r}, detail: {v.detail!r}"
        )


# ---------------------------------------------------------------------------
# Finding #11 — Gate 5 silent skip with misleading verdict
# ---------------------------------------------------------------------------

class TestGate5SkipVerdict:
    """Tests for Finding #11: when candidate_binary is None, Gate 5 is skipped
    but the verdict used to falsely claim 'all 5 gates passed'.

    Requirements:
      1. Detail string must NOT say 'all 5 gates passed' when Gate 5 was skipped.
      2. metrics['gates_run'] must reflect the actual number of gates that ran.
      3. When candidate_binary IS supplied and Gate 5 runs, the detail IS correct.
    """

    @staticmethod
    def _make_all_pass_stubs(monkeypatch):
        """Stub gates 1-5 as all passing."""
        monkeypatch.setattr(
            gate_cascade, "_run_compile_gate",
            MagicMock(return_value={"ok": True, "stderr": ""}),
        )
        monkeypatch.setattr(
            gate_cascade, "_run_sol_gate",
            MagicMock(return_value={"ok": True, "peak_ratio": 1.5}),
        )
        monkeypatch.setattr(
            gate_cascade, "_run_bitwise_gate",
            MagicMock(return_value={"ok": True, "diff": None}),
        )
        monkeypatch.setattr(
            gate_cascade, "_run_regression_gate",
            MagicMock(return_value={
                "ok": True,
                "verdict": "improved",
                "total_delta_pct": -0.05,
                "weighted_geomean_delta_pct": -0.05,
                "per_kernel_deltas": [],
                "threshold_pct": 0.03,
            }),
        )
        monkeypatch.setattr(
            gate_cascade, "_run_anchors_gate",
            MagicMock(return_value={"ok": True, "failed": []}),
        )

    def test_no_candidate_binary_detail_not_all_5_passed(self, monkeypatch):
        """When candidate_binary=None, detail must NOT claim 'all 5 gates passed'."""
        self._make_all_pass_stubs(monkeypatch)

        v = gate_cascade.evaluate(
            baseline_db="b.db", candidate_db="c.db",
            patch_file="foo.hip", patch_sha="abc123",
            gfx_id="gfx942", claimed_speedup=1.2,
            candidate_binary=None,   # Gate 5 skipped
        )
        assert v.status == "pass"
        assert v.detail != "all 5 gates passed", (
            "Gate 5 was skipped (no candidate_binary); detail must not claim "
            f"'all 5 gates passed'. Got: {v.detail!r}"
        )

    def test_no_candidate_binary_detail_mentions_skipped(self, monkeypatch):
        """Skipped Gate 5 verdict detail must mention the skip reason."""
        self._make_all_pass_stubs(monkeypatch)

        v = gate_cascade.evaluate(
            baseline_db="b.db", candidate_db="c.db",
            patch_file="foo.hip", patch_sha="abc123",
            gfx_id="gfx942", claimed_speedup=1.2,
            candidate_binary=None,
        )
        assert "skipped" in v.detail.lower(), (
            f"Detail should mention that Gate 5 was skipped. Got: {v.detail!r}"
        )
        assert "candidate_binary" in v.detail.lower() or "anchors" in v.detail.lower(), (
            f"Detail should explain WHY Gate 5 was skipped. Got: {v.detail!r}"
        )

    def test_no_candidate_binary_gates_run_is_4(self, monkeypatch):
        """metrics['gates_run'] must be 4 when Gate 5 was skipped."""
        self._make_all_pass_stubs(monkeypatch)

        v = gate_cascade.evaluate(
            baseline_db="b.db", candidate_db="c.db",
            patch_file="foo.hip", patch_sha="abc123",
            gfx_id="gfx942", claimed_speedup=1.2,
            candidate_binary=None,
        )
        assert v.metrics.get("gates_run") == 4, (
            f"Expected gates_run=4 when Gate 5 skipped, got: {v.metrics.get('gates_run')!r}"
        )

    def test_with_candidate_binary_detail_is_all_5_passed(self, monkeypatch):
        """When candidate_binary is provided and Gate 5 passes, detail IS 'all 5 gates passed'."""
        self._make_all_pass_stubs(monkeypatch)

        v = gate_cascade.evaluate(
            baseline_db="b.db", candidate_db="c.db",
            patch_file="foo.hip", patch_sha="abc123",
            gfx_id="gfx942", claimed_speedup=1.2,
            candidate_binary="./my_binary",  # Gate 5 runs
        )
        assert v.status == "pass"
        assert v.detail == "all 5 gates passed", (
            f"With candidate_binary supplied and all passing, detail must be "
            f"'all 5 gates passed'. Got: {v.detail!r}"
        )
        assert v.metrics.get("gates_run") == 5

    def test_with_candidate_binary_gates_run_is_5(self, monkeypatch):
        """metrics['gates_run'] must be 5 when Gate 5 ran."""
        self._make_all_pass_stubs(monkeypatch)

        v = gate_cascade.evaluate(
            baseline_db="b.db", candidate_db="c.db",
            patch_file="foo.hip", patch_sha="abc123",
            gfx_id="gfx942", claimed_speedup=1.2,
            candidate_binary="./my_binary",
        )
        assert v.metrics.get("gates_run") == 5

"""End-to-end: regression detected by gate cascade → Correctness returns revert."""

from perfxpert.agents import build_session, schemas
from perfxpert.runtime import gate_cascade


def test_regressed_verdict_flows_to_revert_action(regression_db_pair, monkeypatch):
    baseline, candidate = regression_db_pair

    # Stub execution-class gates in gate_cascade so the test is hermetic
    monkeypatch.setattr(gate_cascade, "_run_compile_gate",
                        lambda *a, **kw: {"ok": True})
    monkeypatch.setattr(gate_cascade, "_run_sol_gate",
                        lambda *a, **kw: {"ok": True, "peak_ratio": 0.5})
    monkeypatch.setattr(gate_cascade, "_run_bitwise_gate",
                        lambda *a, **kw: {"ok": True})
    monkeypatch.setattr(gate_cascade, "_run_regression_gate",
                        lambda *a, **kw: {
                            "ok": False,
                            "total_delta_pct": 12.0,
                            "hot_kernels": [{"name": "[K1]", "delta_pct": 12.0}],
                            "weighted_geomean_delta_pct": 10.0,
                        })
    monkeypatch.setattr(gate_cascade, "_run_anchors_gate",
                        lambda *a, **kw: {"ok": True})

    verdict = gate_cascade.evaluate(
        baseline_db=str(baseline), candidate_db=str(candidate),
        patch_file="fake.hip", patch_sha="deadbeef",
        gfx_id="gfx942", claimed_speedup=1.5,
    )
    assert verdict.status == "regressed"

    session = build_session(airgap=True)
    result = session.run_correctness(
        schemas.CorrectnessInput(
            gate_verdict=schemas.GateVerdictModel(
                status=verdict.status,
                failing_gate=verdict.failing_gate,
                detail=verdict.detail,
                delta_pct=verdict.delta_pct,
            ),
            kernel_name="[K1]",
            last_technique="launch_bounds",
        )
    )
    assert result.action == "revert"
    assert result.verdict == "regressed"

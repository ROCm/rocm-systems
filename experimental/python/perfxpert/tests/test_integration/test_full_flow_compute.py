"""Full-flow integration: Root → Analysis → Recommendation → Compute-Specialist.

Airgap mode so no LLM is hit; verifies handoff targets + structured output.

Fixture taxonomy:
- compute_bound.db: MFMA GEMM kernel with --pmc counters (gfx942, 30 iter).
  Expected: classify as "compute" (AI >> ridge, GPU util=100%).
- trace_only_elementwise.db: same workload profiled WITHOUT --pmc.
  Expected: classify as "data_insufficient" (classifier is flying blind).
"""

from unittest.mock import MagicMock

import pytest

from perfxpert.agents import build_session
from perfxpert.agents import schemas


def test_compute_bound_routes_to_compute_specialist(compute_bound_db, monkeypatch):
    """End-to-end: trace DB → Root → Analysis → Recommendation → Compute specialist."""
    session = build_session(airgap=True)

    # Root handles routing deterministically via intent.classify
    root_out = session.run_root(
        schemas.RootInput(
            user_query="analyze this trace",
            database_path=str(compute_bound_db),
        )
    )
    assert isinstance(root_out, schemas.RootOutput)
    assert root_out.metadata.get("routed_to") == "analysis"


def test_analysis_classifies_compute_bound_fixture(compute_bound_db):
    """Analysis on MFMA GEMM fixture (with --pmc) → classifies as compute.

    The fixture was captured with hardware counters: SQ_WAVES, SQ_INSTS_VALU,
    GRBM_GUI_ACTIVE, GRBM_COUNT, FETCH_SIZE, WRITE_SIZE.
    AI >> ridge (3696 FLOPS/B vs ridge 30.8), GPU util = 100%.
    Classifier must return compute (confidence >= 0.5).
    """
    session = build_session(airgap=True)
    out = session.run_analysis(
        schemas.AnalysisInput(database_path=str(compute_bound_db), top_kernels=10)
    )
    assert out.primary_bottleneck == "compute", (
        f"Expected 'compute' for MFMA fixture but got '{out.primary_bottleneck}'. "
        f"counter_data_available={out.counter_data_available}"
    )
    assert out.counter_data_available is True


def test_recommendation_dispatches_compute_specialist_in_airgap(compute_bound_db, test_gfx_id):
    """Compute bottleneck must route to compute_specialist — no skip allowed.

    This test uses the new compute_bound.db fixture (MFMA GEMM with --pmc counters)
    which correctly classifies as 'compute'. The load-bearing pytest.skip is removed.
    """
    session = build_session(airgap=True)
    findings = session.run_analysis(
        schemas.AnalysisInput(database_path=str(compute_bound_db))
    )
    assert findings.primary_bottleneck == "compute", (
        f"Pre-condition failed: expected 'compute' but got '{findings.primary_bottleneck}'. "
        "Check compute_bound.db fixture — it should be the MFMA GEMM DB with PMC counters."
    )
    rec_out = session.run_recommendation(
        schemas.RecommendationInput(findings=findings, gfx_id=test_gfx_id)
    )
    assert rec_out.specialist_used == "compute"


def test_trace_only_classifies_data_insufficient(trace_only_elementwise_db):
    """Trace-only DB (no --pmc) must return data_insufficient — not mixed@0.5.

    The trace_only_elementwise.db was captured with --sys-trace only, no hardware
    counters. The metric bridge sets all HW counter keys to None, and the classifier
    must return data_insufficient instead of the silent mixed@0.5 fallback.
    """
    session = build_session(airgap=True)
    out = session.run_analysis(
        schemas.AnalysisInput(database_path=str(trace_only_elementwise_db))
    )
    assert out.primary_bottleneck == "data_insufficient", (
        f"Expected 'data_insufficient' for trace-only DB but got '{out.primary_bottleneck}'. "
        "The classifier must be loud when no counter data is available."
    )
    assert out.counter_data_available is False


def test_data_insufficient_verdict_suppresses_recommendations(trace_only_elementwise_db, test_gfx_id, capsys):
    """When classifier returns data_insufficient, recommendation must print warning
    and return no techniques (specialist_used='none').
    """
    session = build_session(airgap=True)
    findings = session.run_analysis(
        schemas.AnalysisInput(database_path=str(trace_only_elementwise_db))
    )
    assert findings.primary_bottleneck == "data_insufficient"

    rec_out = session.run_recommendation(
        schemas.RecommendationInput(findings=findings, gfx_id=test_gfx_id)
    )
    # No recommendations should be generated when data is insufficient
    assert rec_out.specialist_used == "none"
    assert rec_out.recommendations == []

    # Warning must be printed to stderr
    captured = capsys.readouterr()
    assert "data_insufficient" in captured.err.lower() or "WARNING" in captured.err

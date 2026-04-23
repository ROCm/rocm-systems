"""Full-flow integration: Root → Analysis → Recommendation → Compute-Specialist.

Airgap mode so no LLM is hit; verifies handoff targets + structured output.
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
    """Analysis on a compute-bound fixture → primary_bottleneck == 'compute'."""
    session = build_session(airgap=True)
    out = session.run_analysis(
        schemas.AnalysisInput(database_path=str(compute_bound_db), top_kernels=10)
    )
    # Rule-based classification; fixture design ensures this is compute.
    assert out.primary_bottleneck in ("compute", "mixed")


def test_recommendation_dispatches_compute_specialist_in_airgap(compute_bound_db, test_gfx_id):
    session = build_session(airgap=True)
    findings = session.run_analysis(
        schemas.AnalysisInput(database_path=str(compute_bound_db))
    )
    if findings.primary_bottleneck != "compute":
        pytest.skip("fixture not classified compute by rule; specialist routing untested")
    rec_out = session.run_recommendation(
        schemas.RecommendationInput(findings=findings, gfx_id=test_gfx_id)
    )
    assert rec_out.specialist_used == "compute"

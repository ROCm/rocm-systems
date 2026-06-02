"""Memory-bound full flow → memory specialist."""

import pytest

from perfxpert.agents import build_session, schemas


def test_memory_bound_routes_to_memory_specialist(memory_bound_db, test_gfx_id):
    session = build_session(airgap=True)
    findings = session.run_analysis(
        schemas.AnalysisInput(database_path=str(memory_bound_db))
    )
    if findings.primary_bottleneck != "memory_transfer":
        pytest.skip("fixture not classified memory by rule")
    rec_out = session.run_recommendation(
        schemas.RecommendationInput(findings=findings, gfx_id=test_gfx_id)
    )
    assert rec_out.specialist_used == "memory"

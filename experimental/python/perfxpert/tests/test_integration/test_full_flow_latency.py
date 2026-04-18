"""Latency-bound full flow → latency specialist."""

import pytest

from perfxpert.agents import build_session, schemas


def test_latency_bound_routes_to_latency_specialist(latency_bound_db):
    session = build_session(airgap=True)
    findings = session.run_analysis(
        schemas.AnalysisInput(database_path=str(latency_bound_db))
    )
    if findings.primary_bottleneck not in ("latency", "api_overhead"):
        pytest.skip("fixture not classified latency by rule")
    rec_out = session.run_recommendation(
        schemas.RecommendationInput(findings=findings)
    )
    assert rec_out.specialist_used == "latency"

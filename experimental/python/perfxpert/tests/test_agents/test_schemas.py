"""Tests for perfxpert.agents.schemas — Pydantic I/O types."""

import pytest
from pydantic import ValidationError

from perfxpert.agents import schemas


# -- Root ------------------------------------------------------------------

def test_root_input_minimal_construction():
    m = schemas.RootInput(user_query="why is this slow?", database_path=None)
    assert m.user_query == "why is this slow?"
    assert m.database_path is None


def test_root_output_requires_narrative_and_recs():
    m = schemas.RootOutput(
        narrative="Kernel X dominates runtime.",
        recommendations=[],
        primary_bottleneck="compute",
    )
    assert m.primary_bottleneck == "compute"


# -- Analysis --------------------------------------------------------------

def test_analysis_input_accepts_db_and_top_n():
    m = schemas.AnalysisInput(database_path="/tmp/a.db", top_kernels=10)
    assert m.top_kernels == 10


def test_analysis_output_has_classification_and_metrics():
    m = schemas.AnalysisOutput(
        primary_bottleneck="memory_transfer",
        confidence=0.82,
        time_breakdown={"kernel_pct": 0.55, "memcpy_pct": 0.30, "api_pct": 0.10, "idle_pct": 0.05},
        hot_kernels=[{"name": "[KERNEL_1]", "pct": 0.40}],
        counter_data_available=True,
    )
    assert m.confidence == 0.82


# -- Recommendation --------------------------------------------------------

def test_recommendation_input_threads_findings():
    findings = schemas.AnalysisOutput(
        primary_bottleneck="compute",
        confidence=0.9,
        time_breakdown={"kernel_pct": 0.9, "memcpy_pct": 0.0, "api_pct": 0.05, "idle_pct": 0.05},
        hot_kernels=[{"name": "[KERNEL_1]", "pct": 0.80}],
        counter_data_available=False,
    )
    m = schemas.RecommendationInput(findings=findings, gfx_id="gfx942", kernel_filter=None)
    assert m.findings.primary_bottleneck == "compute"


def test_recommendation_input_requires_gfx_id():
    findings = schemas.AnalysisOutput(
        primary_bottleneck="compute",
        confidence=0.9,
        time_breakdown={"kernel_pct": 0.9, "memcpy_pct": 0.0, "api_pct": 0.05, "idle_pct": 0.05},
        hot_kernels=[{"name": "[KERNEL_1]", "pct": 0.80}],
        counter_data_available=False,
    )
    with pytest.raises(ValidationError):
        schemas.RecommendationInput(findings=findings)


def test_recommendation_input_carries_gfx_id():
    findings = schemas.AnalysisOutput(
        primary_bottleneck="compute",
        confidence=0.9,
        time_breakdown={"kernel_pct": 0.9, "memcpy_pct": 0.0, "api_pct": 0.05, "idle_pct": 0.05},
        hot_kernels=[{"name": "[KERNEL_1]", "pct": 0.80}],
        counter_data_available=False,
    )
    m = schemas.RecommendationInput(findings=findings, gfx_id="gfx950")
    assert m.gfx_id == "gfx950"


def test_recommendation_output_carries_ranked_list():
    m = schemas.RecommendationOutput(
        recommendations=[
            {"title": "Reduce VGPR count", "priority": "high", "category": "compute"}
        ],
        specialist_used="compute",
    )
    assert m.specialist_used == "compute"


# -- Specialists -----------------------------------------------------------

@pytest.mark.parametrize("schema_name", [
    "ComputeSpecialistInput", "ComputeSpecialistOutput",
    "MemorySpecialistInput", "MemorySpecialistOutput",
    "LatencySpecialistInput", "LatencySpecialistOutput",
])
def test_specialist_schemas_exist(schema_name):
    assert hasattr(schemas, schema_name), f"schemas.{schema_name} missing"


def test_specialist_input_has_arch_and_metrics():
    m = schemas.ComputeSpecialistInput(
        gfx_id="gfx942",
        hot_kernels=[{"name": "[KERNEL_1]", "pct": 0.60}],
        counter_data={"valu_util_pct": 0.85},
    )
    assert m.gfx_id == "gfx942"


# -- Correctness -----------------------------------------------------------

def test_correctness_input_consumes_gate_verdict():
    verdict = schemas.GateVerdictModel(
        status="regressed",
        failing_gate="regression",
        detail="total_runtime +15% vs baseline",
        delta_pct=15.0,
    )
    m = schemas.CorrectnessInput(
        gate_verdict=verdict,
        kernel_name="[KERNEL_1]",
        last_technique="launch_bounds",
        source_dir="/tmp/project",
    )
    assert m.gate_verdict.status == "regressed"
    assert m.source_dir == "/tmp/project"


def test_correctness_output_action_enum():
    m = schemas.CorrectnessOutput(
        verdict="regressed",
        action="revert",
        narrative="Gate 4 failed: +15% regression",
        alternative_technique="lds_tiling",
    )
    assert m.action == "revert"


def test_correctness_action_rejects_invalid():
    with pytest.raises(ValidationError):
        schemas.CorrectnessOutput(
            verdict="pass",
            action="nuke_system",  # invalid
            narrative="",
        )


# -- Frozen / immutable ----------------------------------------------------

def test_root_input_is_frozen():
    m = schemas.RootInput(user_query="x", database_path=None)
    with pytest.raises(ValidationError):
        m.user_query = "y"


def test_analysis_output_is_frozen():
    m = schemas.AnalysisOutput(
        primary_bottleneck="compute",
        confidence=0.9,
        time_breakdown={"kernel_pct": 1.0, "memcpy_pct": 0.0, "api_pct": 0.0, "idle_pct": 0.0},
        hot_kernels=[],
        counter_data_available=False,
    )
    with pytest.raises(ValidationError):
        m.confidence = 0.5

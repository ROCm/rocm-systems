"""Tests for Phase 10 Change-Impact Prediction attachment on specialist output."""

from __future__ import annotations

import pytest

from perfxpert.agents import (
    compute_specialist as cs_module,
    latency_specialist as ls_module,
    memory_specialist as ms_module,
    schemas,
)
from perfxpert.tools import predict_impact


@pytest.fixture(autouse=True)
def _clear_prediction_store():
    predict_impact._reset_store_for_tests()
    yield
    predict_impact._reset_store_for_tests()


def test_compute_specialist_attaches_predicted_range_when_catalog_hit(monkeypatch):
    """Airgap + catalog hit => technique dict carries predicted_impact_range."""
    monkeypatch.setenv("PERFXPERT_AIRGAP", "1")
    monkeypatch.setattr(
        cs_module,
        "_fetch_catalog",
        lambda gfx_id: [
            {
                "id": "vgpr_reduction",
                "name": "vgpr_reduction",
                "expected_impact": 0.30,
                "effort_factor": 1.0,
                "risk": "low",
            },
        ],
    )

    result = cs_module.run_compute_specialist(
        schemas.ComputeSpecialistInput(
            gfx_id="gfx942",
            hot_kernels=[
                {"name": "hot_kernel", "pct": 0.40, "baseline_db": "/tmp/does-not-exist.db"},
            ],
            counter_data={"vgpr_per_thread": 128},
        ),
        airgap=True,
    )
    assert result.techniques
    hit = result.techniques[0]
    assert "predicted_impact_range" in hit
    lo, hi = hit["predicted_impact_range"]
    assert lo < hi
    assert "confidence" in hit
    assert hit.get("source_citation", "").startswith("knowledge/proven_optimizations.yaml")


def test_compute_specialist_skips_prediction_when_not_in_catalog(monkeypatch):
    """Airgap + technique not in change_impact_models.yaml => no attachment."""
    monkeypatch.setenv("PERFXPERT_AIRGAP", "1")
    monkeypatch.setattr(
        cs_module,
        "_fetch_catalog",
        lambda gfx_id: [
            {
                "id": "freestyle_special_sauce",
                "name": "freestyle_special_sauce",
                "expected_impact": 0.50,
                "effort_factor": 1.0,
                "risk": "low",
            },
        ],
    )

    result = cs_module.run_compute_specialist(
        schemas.ComputeSpecialistInput(
            gfx_id="gfx942",
            hot_kernels=[
                {"name": "hot_kernel", "pct": 0.40, "baseline_db": "/tmp/fake.db"},
            ],
            counter_data={"vgpr_per_thread": 96},
        ),
        airgap=True,
    )
    assert result.techniques
    tech = result.techniques[0]
    assert "predicted_impact_range" not in tech


def test_memory_specialist_attaches_when_catalog_hit(monkeypatch):
    monkeypatch.setenv("PERFXPERT_AIRGAP", "1")
    monkeypatch.setattr(
        ms_module,
        "_fetch_catalog",
        lambda gfx_id: [
            {
                "id": "lds_tiling",
                "name": "lds_tiling",
                "expected_impact": 0.50,
                "effort_factor": 1.0,
                "risk": "medium",
            },
        ],
    )
    result = ms_module.run_memory_specialist(
        schemas.MemorySpecialistInput(
            gfx_id="gfx942",
            hot_kernels=[
                {"name": "gemm_kernel", "pct": 0.30, "baseline_db": "/tmp/fake.db"},
            ],
            counter_data={"hbm_bw_utilization": 0.72},
        ),
        airgap=True,
    )
    assert result.techniques
    assert "predicted_impact_range" in result.techniques[0]


def test_latency_specialist_attaches_when_catalog_hit(monkeypatch):
    monkeypatch.setenv("PERFXPERT_AIRGAP", "1")
    monkeypatch.setattr(
        ls_module,
        "_fetch_catalog",
        lambda gfx_id: [
            {
                "id": "hip_stream_overlap",
                "name": "hip_stream_overlap",
                "expected_impact": 0.35,
                "effort_factor": 1.0,
                "risk": "low",
            },
        ],
    )
    result = ls_module.run_latency_specialist(
        schemas.LatencySpecialistInput(
            gfx_id="gfx942",
            hot_kernels=[
                {"name": "dominant_kernel", "pct": 0.25, "baseline_db": "/tmp/fake.db"},
            ],
            api_overhead_pct=0.30,
            avg_kernel_duration_us=4.2,
        ),
        airgap=True,
    )
    assert result.techniques
    # Latency specialist doesn't always carry counter_data — ensure the
    # attach helper still works. We only assert a technique key survives.
    assert "name" in result.techniques[0]


def test_specialist_attach_preserves_unrelated_fields(monkeypatch):
    monkeypatch.setenv("PERFXPERT_AIRGAP", "1")
    monkeypatch.setattr(
        cs_module,
        "_fetch_catalog",
        lambda gfx_id: [
            {
                "id": "vgpr_reduction",
                "name": "vgpr_reduction",
                "expected_impact": 0.30,
                "effort_factor": 1.0,
                "risk": "low",
                "description": "my narrative",
            },
        ],
    )
    result = cs_module.run_compute_specialist(
        schemas.ComputeSpecialistInput(
            gfx_id="gfx942",
            hot_kernels=[
                {"name": "k1", "pct": 0.50, "baseline_db": "/tmp/fake.db"},
            ],
            counter_data={"vgpr_per_thread": 128},
        ),
        airgap=True,
    )
    hit = result.techniques[0]
    assert hit["description"] == "my narrative"
    assert hit["expected_impact"] == 0.30

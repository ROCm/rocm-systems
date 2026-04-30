"""Tests for perfxpert.tools.arch."""

import pytest

from perfxpert.tools import arch
from perfxpert.tools._class import ToolClass


@pytest.fixture(autouse=True)
def _disable_runtime_discovery(monkeypatch):
    monkeypatch.setattr(arch, "_runtime_specs_for_gfx", lambda gfx_id: {})


def test_lookup_peaks_returns_structured_data_for_mi300x():
    peaks = arch.lookup_peaks("gfx942")
    assert peaks["name"] == "MI300X"
    assert peaks["peak_fp64_tflops"] == 81.7
    assert peaks["memory_bandwidth_tbs"] == 5.3
    assert peaks["max_waves_per_simd"] == 8
    assert peaks["ridge_point"] == pytest.approx(30.8, rel=0.01)
    assert peaks["ridge_points"]["fp64"] == pytest.approx(15.4, rel=0.01)


def test_lookup_peaks_returns_public_mi350x_values_for_gfx950():
    peaks = arch.lookup_peaks("gfx950")
    assert peaks["name"] == "MI350X"
    assert peaks["peak_fp64_tflops"] == pytest.approx(72.1)
    assert peaks["peak_fp32_tflops"] == pytest.approx(144.2)
    assert peaks["memory_bandwidth_tbs"] == pytest.approx(8.0)
    assert peaks["max_waves_per_simd"] == 8
    assert peaks["ridge_point"] == pytest.approx(18.0, rel=0.01)


def test_lookup_peaks_exposes_runtime_caps_for_occupancy_users():
    peaks = arch.lookup_peaks("gfx1100")
    assert peaks["wave_size"] == 32
    assert peaks["max_vgprs_per_thread"] == 256
    assert peaks["vgprs_per_simd"] == 1536
    assert peaks["simds_per_cu"] == 2
    assert peaks["max_waves_per_simd"] == 16


def test_lookup_peaks_covers_all_known_archs():
    known = ["gfx908", "gfx90a", "gfx942", "gfx950", "gfx1030", "gfx1100"]
    for gfx in known:
        result = arch.lookup_peaks(gfx)
        assert "name" in result
        assert "peak_fp64_tflops" in result


def test_lookup_peaks_unknown_arch_raises():
    with pytest.raises(KeyError) as exc:
        arch.lookup_peaks("gfx9999")
    assert "gfx9999" in str(exc.value)
    assert "known" in str(exc.value).lower() or "available" in str(exc.value).lower()


@pytest.mark.parametrize("gfx_id", sorted(arch._gpu_specs().keys()))
def test_lookup_peaks_keeps_static_specs_for_known_archs_by_default(monkeypatch, gfx_id):
    static_specs = arch._gpu_specs()[gfx_id]
    runtime = {
        "gfx_id": gfx_id,
        "name": f"Runtime {gfx_id}",
        "cu_count": int(static_specs["cu_count"]) + 1,
        "memory_bandwidth_tbs": float(static_specs["memory_bandwidth_tbs"]) + 0.125,
        "spec_sources": {
            "name": "rocminfo",
            "cu_count": "rocminfo",
            "memory_bandwidth_tbs": "amd-smi",
        },
    }
    monkeypatch.setattr(
        arch,
        "_runtime_specs_for_gfx",
        lambda requested: runtime if requested == gfx_id else {},
    )

    peaks = arch.lookup_peaks(gfx_id)

    assert peaks["runtime_discovered"] is False
    assert peaks["name"] == static_specs["name"]
    assert peaks["cu_count"] == static_specs["cu_count"]
    assert peaks["memory_bandwidth_tbs"] == static_specs["memory_bandwidth_tbs"]


@pytest.mark.parametrize("gfx_id", sorted(arch._gpu_specs().keys()))
def test_lookup_peaks_prefers_runtime_specs_for_local_init_when_requested(monkeypatch, gfx_id):
    static_specs = arch._gpu_specs()[gfx_id]
    runtime_memory_bandwidth = float(static_specs["memory_bandwidth_tbs"]) + 0.125
    runtime = {
        "gfx_id": gfx_id,
        "name": f"Runtime {gfx_id}",
        "cu_count": int(static_specs["cu_count"]) + 1,
        "max_sclk_mhz": 2100,
        "peak_fp32_tflops": float(static_specs["peak_fp32_tflops"]) + 999.0,
        "memory_bandwidth_tbs": runtime_memory_bandwidth,
        "spec_sources": {
            "name": "rocminfo",
            "cu_count": "rocminfo",
            "max_sclk_mhz": "amd-smi",
            "peak_fp32_tflops": "derived-from-runtime-topology",
            "memory_bandwidth_tbs": "amd-smi",
        },
    }
    monkeypatch.setattr(
        arch,
        "_runtime_specs_for_gfx",
        lambda requested: runtime if requested == gfx_id else {},
    )

    peaks = arch.lookup_peaks(gfx_id, prefer_runtime=True)

    assert peaks["runtime_discovered"] is True
    assert peaks["name"] == f"Runtime {gfx_id}"
    assert peaks["cu_count"] == int(static_specs["cu_count"]) + 1
    assert peaks["max_sclk_mhz"] == 2100
    assert peaks["memory_bandwidth_tbs"] == pytest.approx(runtime_memory_bandwidth)
    assert peaks["peak_fp32_tflops"] == static_specs["peak_fp32_tflops"]
    assert peaks["peak_fp64_tflops"] == static_specs["peak_fp64_tflops"]
    assert peaks["static_fallback_keys"]
    assert peaks["spec_sources"]["peak_fp32_tflops"] == "gpu_specs.yaml"
    assert peaks["spec_sources"]["peak_fp64_tflops"] == "gpu_specs.yaml"
    assert peaks["spec_sources"]["memory_bandwidth_tbs"] == "amd-smi"


def test_lookup_peaks_supports_runtime_only_local_gpu(monkeypatch):
    cu_count = 32
    max_sclk_mhz = 3500
    fp32_ops_per_cu_cycle = 256
    peak_fp32_tflops = round(cu_count * fp32_ops_per_cu_cycle * max_sclk_mhz / 1_000_000.0, 3)
    runtime = {
        "gfx_id": "gfx1200",
        "name": "Runtime RDNA",
        "cu_count": cu_count,
        "wave_size": 32,
        "simds_per_cu": 2,
        "lds_kb": 64,
        "lds_per_cu_kb": 64,
        "max_vgprs_per_thread": 256,
        "vgprs_per_simd": 1536,
        "max_waves_per_simd": 16,
        "max_sclk_mhz": max_sclk_mhz,
        "peak_fp32_tflops": peak_fp32_tflops,
        "peak_fp64_tflops": peak_fp32_tflops / 32.0,
        "memory_bandwidth_tbs": 0.0,
        "spec_sources": {"gfx_id": "rocminfo", "peak_fp32_tflops": "derived-from-runtime-topology"},
    }
    monkeypatch.setattr(arch, "_runtime_specs_for_gfx", lambda gfx_id: runtime if gfx_id == "gfx1200" else {})

    peaks = arch.lookup_peaks("gfx1200")

    assert peaks["runtime_discovered"] is True
    assert peaks["name"] == "Runtime RDNA"
    assert peaks["peak_fp32_tflops"] == pytest.approx(peak_fp32_tflops)
    assert peaks["ridge_point"] == 0.0


def test_lookup_peaks_is_read_only_class():
    """MCP exposure policy — lookup tools are READ_ONLY."""
    assert arch.lookup_peaks.__tool_class__ == ToolClass.READ_ONLY


def test_lookup_peaks_is_fast_after_runtime_specs_are_cached(monkeypatch):
    """Lookup remains cheap once runtime discovery has populated its cache."""
    import time

    monkeypatch.setattr(arch, "_runtime_specs_for_gfx", lambda gfx_id: {})
    start = time.time()
    arch.lookup_peaks("gfx942")
    duration_ms = (time.time() - start) * 1000
    # Must be fast (< 50ms even on cold YAML load)
    assert duration_ms < 50, f"too slow: {duration_ms}ms"

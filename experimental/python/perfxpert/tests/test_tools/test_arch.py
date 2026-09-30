"""Tests for perfxpert.tools.arch."""

import pytest

from perfxpert.tools import arch
from perfxpert.tools._class import ToolClass


_STATIC_SPEC_FIELDS = (
    "name",
    "codename",
    "peak_fp64_tflops",
    "peak_fp32_tflops",
    "peak_fp16_tflops",
    "peak_bf16_tflops",
    "peak_fp8_tflops",
    "peak_int8_tops",
    "memory_bandwidth_tbs",
    "cu_count",
    "lds_kb",
    "lds_per_cu_kb",
    "wave_size",
    "max_vgprs_per_thread",
    "vgprs_per_simd",
    "simds_per_cu",
    "max_waves_per_simd",
)

_OCCUPANCY_FIELD_MAP = {
    "max_waves_per_simd": "max_waves_per_simd",
    "vgprs_per_simd": "vgprs_per_simd",
    "lds_per_cu_kb": "lds_per_cu_kb",
    "wavefront_size": "wave_size",
    "simds_per_cu": "simds_per_cu",
}


@pytest.fixture(autouse=True)
def _disable_runtime_discovery(monkeypatch):
    monkeypatch.setattr(arch, "_runtime_specs_for_gfx", lambda gfx_id: {})


def _expected_ridge_point(specs, dtype="fp32"):
    peak_key = arch._RIDGE_PEAK_KEYS.get(dtype, "peak_fp32_tflops")
    peak_tflops = float(specs.get(peak_key) or specs.get("peak_fp32_tflops") or 0.0)
    bandwidth_tbs = float(specs.get("memory_bandwidth_tbs") or 0.0)
    if bandwidth_tbs <= 0:
        return float(specs.get("ridge_point") or 0.0)
    return round(peak_tflops / bandwidth_tbs, 1)


@pytest.mark.parametrize("gfx_id, static_specs", sorted(arch._gpu_specs().items()))
def test_lookup_peaks_returns_static_specs_for_known_archs(gfx_id, static_specs):
    peaks = arch.lookup_peaks(gfx_id)

    assert peaks["runtime_discovered"] is False
    for field in _STATIC_SPEC_FIELDS:
        assert peaks[field] == static_specs[field]
        assert peaks["spec_sources"][field] == "gpu_specs.yaml"

    assert peaks["ridge_point"] == pytest.approx(_expected_ridge_point(static_specs))
    for dtype in arch._RIDGE_PEAK_KEYS:
        assert peaks["ridge_points"][dtype] == pytest.approx(_expected_ridge_point(static_specs, dtype=dtype))


def test_lookup_peaks_exposes_runtime_caps_for_occupancy_users():
    occupancy_table = arch.occupancy_specs_table()
    for gfx_id, static_specs in arch._gpu_specs().items():
        peaks = arch.lookup_peaks(gfx_id)

        for occupancy_field, spec_field in _OCCUPANCY_FIELD_MAP.items():
            assert occupancy_table[gfx_id][occupancy_field] == int(static_specs[spec_field])
            assert peaks[spec_field] == static_specs[spec_field]


def test_lookup_peaks_covers_all_known_archs():
    required_fields = {"name", "peak_fp64_tflops"}
    for gfx_id in arch._gpu_specs():
        result = arch.lookup_peaks(gfx_id)
        assert required_fields.issubset(result)


def test_lookup_peaks_unknown_arch_raises():
    unknown_gfx = "__unknown_arch_for_test__"
    with pytest.raises(KeyError) as exc:
        arch.lookup_peaks(unknown_gfx)
    assert unknown_gfx in str(exc.value)
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
    runtime_cu_count = int(static_specs["cu_count"]) + 1
    runtime_clock_mhz = 2100
    runtime_peak_fp32_tflops = float(static_specs["peak_fp32_tflops"]) + 999.0
    runtime = {
        "gfx_id": gfx_id,
        "name": f"Runtime {gfx_id}",
        "cu_count": runtime_cu_count,
        "max_sclk_mhz": runtime_clock_mhz,
        "peak_fp32_tflops": runtime_peak_fp32_tflops,
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
    assert peaks["cu_count"] == runtime_cu_count
    assert peaks["max_sclk_mhz"] == runtime_clock_mhz
    assert peaks["memory_bandwidth_tbs"] == pytest.approx(runtime_memory_bandwidth)
    assert peaks["peak_fp32_tflops"] == static_specs["peak_fp32_tflops"]
    assert peaks["peak_fp64_tflops"] == static_specs["peak_fp64_tflops"]
    assert peaks["static_fallback_keys"]
    assert peaks["spec_sources"]["peak_fp32_tflops"] == "gpu_specs.yaml"
    assert peaks["spec_sources"]["peak_fp64_tflops"] == "gpu_specs.yaml"
    assert peaks["spec_sources"]["memory_bandwidth_tbs"] == "amd-smi"


def test_lookup_peaks_supports_runtime_only_local_gpu(monkeypatch):
    runtime_gfx_id = "gfxlocal"
    runtime_name = "Runtime GPU"
    runtime_peak_fp32_tflops = 1.25
    runtime_peak_fp64_tflops = 0.25
    runtime = {
        "gfx_id": runtime_gfx_id,
        "name": runtime_name,
        "peak_fp32_tflops": runtime_peak_fp32_tflops,
        "peak_fp64_tflops": runtime_peak_fp64_tflops,
        "memory_bandwidth_tbs": 0.0,
        "spec_sources": {"gfx_id": "rocminfo", "peak_fp32_tflops": "derived-from-runtime-topology"},
    }
    monkeypatch.setattr(arch, "_runtime_specs_for_gfx", lambda gfx_id: runtime if gfx_id == runtime_gfx_id else {})

    peaks = arch.lookup_peaks(runtime_gfx_id)

    assert peaks["runtime_discovered"] is True
    assert peaks["name"] == runtime_name
    assert peaks["peak_fp32_tflops"] == pytest.approx(runtime_peak_fp32_tflops)
    assert peaks["peak_fp64_tflops"] == pytest.approx(runtime_peak_fp64_tflops)
    assert peaks["ridge_point"] == 0.0


def test_lookup_peaks_is_read_only_class():
    """MCP exposure policy — lookup tools are READ_ONLY."""
    assert arch.lookup_peaks.__tool_class__ == ToolClass.READ_ONLY


def test_lookup_peaks_is_fast_after_runtime_specs_are_cached(monkeypatch):
    """Lookup remains cheap once runtime discovery has populated its cache."""
    import time

    monkeypatch.setattr(arch, "_runtime_specs_for_gfx", lambda gfx_id: {})
    gfx_id = next(iter(arch._gpu_specs()))
    start = time.time()
    arch.lookup_peaks(gfx_id)
    duration_ms = (time.time() - start) * 1000
    # Must be fast (< 50ms even on cold YAML load)
    assert duration_ms < 50, f"too slow: {duration_ms}ms"

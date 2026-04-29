"""Tests for runtime GPU discovery."""

from __future__ import annotations

import json

import pytest

from perfxpert.tools import gpu_discovery
from perfxpert.tools._class import ToolClass


def _rocminfo_sample(gfx_id: str = "gfx1200") -> str:
    return f"""
*******
Agent 2
*******
  Name:                    {gfx_id}
  Uuid:                    GPU-c2171ff77417f1d6
  Marketing Name:          AMD Radeon Graphics
  Vendor Name:             AMD
  Device Type:             GPU
  Node:                    1
  Max Clock Freq. (MHz):   3500
  Compute Unit:            32
  SIMDs per CU:            2
  Wavefront Size:          32(0x20)
  Max Waves Per CU:        32(0x20)
  Pool Info:
    Pool 2
      Segment:                 GROUP
      Size:                    64(0x40) KB
"""


def _clear_cache() -> None:
    gpu_discovery._discover_runtime_gpu_specs_cached.cache_clear()


@pytest.fixture(autouse=True)
def _isolated_discovery_cache():
    _clear_cache()
    yield
    _clear_cache()


def test_discover_runtime_gpu_specs_is_read_only() -> None:
    assert gpu_discovery.discover_runtime_gpu_specs.__tool_class__ == ToolClass.READ_ONLY


def test_discover_runtime_gpu_specs_merges_rocm_tools(monkeypatch) -> None:
    gfx_id = "gfx1200"
    rocm_smi_info = {
        "card0": {
            "Node ID": "1",
            "GFX Version": gfx_id,
            "Card Series": "AMD Radeon Graphics",
            "VRAM Total Memory (B)": "8539602944",
        }
    }
    rocm_smi_clocks = {
        "card0": {
            "sclk clock speed:": "(500Mhz)",
            "mclk clock speed:": "(1258Mhz)",
        }
    }
    amd_smi_list = [
        {
            "gpu": 0,
            "node_id": 1,
            "bdf": "0000:0c:00.0",
            "uuid": "c2ff748f-0000-1000-8017-1ff77417f1d6",
        }
    ]
    amd_smi_metric = {
        "gpu_data": [
            {
                "gpu": 0,
                "clock": {
                    "gfx_0": {"max_clk": {"value": 3500, "unit": "MHz"}},
                    "mem_0": {"max_clk": {"value": 1258, "unit": "MHz"}},
                },
                "mem_usage": {
                    "total_vram": {"value": 8144, "unit": "MB"},
                },
            }
        ]
    }

    def _fake_run(argv):
        if argv == ["rocminfo"]:
            return _rocminfo_sample(gfx_id)
        if argv == ["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"]:
            return json.dumps(rocm_smi_info)
        if argv == ["rocm-smi", "--showclocks", "--showclkfrq", "--json"]:
            return json.dumps(rocm_smi_clocks)
        if argv == ["amd-smi", "list", "--json"]:
            return json.dumps(amd_smi_list)
        if argv == ["amd-smi", "static", "--json"]:
            return json.dumps({"gpu_data": [{"gpu": 0}]})
        if argv == ["amd-smi", "metric", "--json"]:
            return json.dumps(amd_smi_metric)
        return None

    monkeypatch.setattr(gpu_discovery, "_run_command", _fake_run)
    monkeypatch.setattr(gpu_discovery, "_parse_kfd_topology", lambda: [])
    _clear_cache()

    result = gpu_discovery.discover_runtime_gpu_specs()
    assert result["source"] == ["rocminfo", "rocm-smi", "amd-smi"]
    assert len(result["gpus"]) == 1

    gpu = result["gpus"][0]
    assert gpu["gfx_id"] == gfx_id
    assert gpu["cu_count"] == 32
    assert gpu["wave_size"] == 32
    assert gpu["max_waves_per_simd"] == 16
    assert gpu["max_sclk_mhz"] == 3500
    assert gpu["vram_total_bytes"] == 8539602944
    assert gpu["peak_fp32_tflops"] == pytest.approx(28.672)
    assert gpu["spec_sources"]["cu_count"] == "rocminfo"


@pytest.mark.parametrize("gfx_id", ["gfx942", "gfx1200"])
def test_discover_runtime_gpu_specs_filters_by_gfx(monkeypatch, gfx_id: str) -> None:
    monkeypatch.setattr(
        gpu_discovery,
        "_run_command",
        lambda argv: _rocminfo_sample(gfx_id) if argv == ["rocminfo"] else None,
    )
    monkeypatch.setattr(gpu_discovery, "_parse_kfd_topology", lambda: [])
    _clear_cache()

    assert gpu_discovery.discover_runtime_gpu_specs(gfx_id)["gpus"]
    assert gpu_discovery.discover_runtime_gpu_specs("gfx9999")["gpus"] == []


def test_discover_runtime_gpu_specs_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv(gpu_discovery.PERFXPERT_DISABLE_RUNTIME_GPU_SPECS, "1")
    _clear_cache()

    result = gpu_discovery.discover_runtime_gpu_specs()
    assert result["gpus"] == []
    assert "disabled" in result["errors"][0]


@pytest.mark.parametrize("gfx_id", ["gfx908", "gfx942", "gfx1200", "gfx9999"])
def test_parse_kfd_topology_supplies_rocminfo_fallback_fields(tmp_path, gfx_id: str) -> None:
    node = tmp_path / "1"
    mem = node / "mem_banks" / "0"
    mem.mkdir(parents=True)
    (node / "name").write_text(f"{gfx_id}\n")
    (node / "properties").write_text(
        "\n".join(
            [
                "simd_count 64",
                "max_waves_per_simd 16",
                "lds_size_in_kb 64",
                "wave_front_size 32",
                "simd_per_cu 2",
                "max_engine_clk_fcompute 3500",
            ]
        )
    )
    (mem / "properties").write_text(
        "\n".join(
            [
                "heap_type 1",
                "size_in_bytes 8539602944",
                "width 128",
                "mem_clk_max 1258",
            ]
        )
    )

    result = gpu_discovery._parse_kfd_topology(tmp_path)

    assert len(result) == 1
    gpu = result[0]
    assert gpu["gfx_id"] == gfx_id
    assert gpu["cu_count"] == 32
    assert gpu["max_waves_per_simd"] == 16
    assert gpu["lds_kb"] == 64
    assert gpu["wave_size"] == 32
    assert gpu["max_sclk_mhz"] == 3500
    assert gpu["vram_total_bytes"] == 8539602944
    assert gpu["memory_bandwidth_tbs"] == pytest.approx(0.04)
    assert gpu["spec_sources"]["cu_count"] == "kfd-topology"
    assert gpu["spec_sources"]["memory_bandwidth_tbs"] == "derived-from-kfd-topology"

"""Tests for runtime GPU discovery."""

from __future__ import annotations

import json

import pytest

from perfxpert.tools import gpu_discovery
from perfxpert.tools._class import ToolClass


def _rocminfo_sample(
    gfx_id: str = "gfx1200",
    *,
    node_id: int = 1,
    max_sclk_mhz: int = 3500,
    cu_count: int = 32,
    simds_per_cu: int = 2,
    wave_size: int = 32,
    max_waves_per_cu: int = 32,
    lds_kb: int = 64,
) -> str:
    return f"""
*******
Agent 2
*******
  Name:                    {gfx_id}
  Uuid:                    GPU-c2171ff77417f1d6
  Marketing Name:          AMD Radeon Graphics
  Vendor Name:             AMD
  Device Type:             GPU
  Node:                    {node_id}
  Max Clock Freq. (MHz):   {max_sclk_mhz}
  Compute Unit:            {cu_count}
  SIMDs per CU:            {simds_per_cu}
  Wavefront Size:          {wave_size}(0x{wave_size:x})
  Max Waves Per CU:        {max_waves_per_cu}(0x{max_waves_per_cu:x})
  Pool Info:
    Pool 2
      Segment:                 GROUP
      Size:                    {lds_kb}(0x{lds_kb:x}) KB
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
    topology = {
        "node_id": 1,
        "cu_count": 32,
        "wave_size": 32,
        "simds_per_cu": 2,
        "max_waves_per_cu": 32,
        "max_sclk_mhz": 3500,
    }
    vram_total_bytes = 8539602944
    rocm_smi_info = {
        "card0": {
            "Node ID": str(topology["node_id"]),
            "GFX Version": gfx_id,
            "Card Series": "AMD Radeon Graphics",
            "VRAM Total Memory (B)": str(vram_total_bytes),
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
            "node_id": topology["node_id"],
            "bdf": "0000:0c:00.0",
            "uuid": "c2ff748f-0000-1000-8017-1ff77417f1d6",
        }
    ]
    amd_smi_metric = {
        "gpu_data": [
            {
                "gpu": 0,
                "clock": {
                    "gfx_0": {"max_clk": {"value": topology["max_sclk_mhz"], "unit": "MHz"}},
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
            return _rocminfo_sample(gfx_id, **topology)
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
    assert gpu["cu_count"] == topology["cu_count"]
    assert gpu["wave_size"] == topology["wave_size"]
    assert gpu["max_waves_per_simd"] == topology["max_waves_per_cu"] // topology["simds_per_cu"]
    assert gpu["max_sclk_mhz"] == topology["max_sclk_mhz"]
    assert gpu["vram_total_bytes"] == vram_total_bytes
    assert gpu["peak_fp32_tflops"] > 0
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


def test_merge_gpu_lists_preserves_same_arch_devices_with_distinct_nodes() -> None:
    gfx_id = "gfx9999"
    node_ids = [1, 2]

    merged = gpu_discovery._merge_gpu_lists(
        [
            [
                {
                    "node_id": node_id,
                    "gfx_id": gfx_id,
                    "spec_sources": {"node_id": "rocminfo", "gfx_id": "rocminfo"},
                }
                for node_id in node_ids
            ]
        ]
    )

    assert [gpu["node_id"] for gpu in merged] == node_ids
    assert all(gpu["gfx_id"] == gfx_id for gpu in merged)


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
    props = {
        "simd_count": 64,
        "max_waves_per_simd": 16,
        "lds_size_in_kb": 64,
        "wave_front_size": 32,
        "simd_per_cu": 2,
        "max_engine_clk_fcompute": 3500,
    }
    mem_props = {
        "heap_type": 1,
        "size_in_bytes": 8539602944,
        "width": 128,
        "mem_clk_max": 1258,
    }
    (node / "name").write_text(f"{gfx_id}\n")
    (node / "properties").write_text("\n".join(f"{key} {value}" for key, value in props.items()))
    (mem / "properties").write_text("\n".join(f"{key} {value}" for key, value in mem_props.items()))

    result = gpu_discovery._parse_kfd_topology(tmp_path)

    assert len(result) == 1
    gpu = result[0]
    assert gpu["gfx_id"] == gfx_id
    assert gpu["cu_count"] == props["simd_count"] // props["simd_per_cu"]
    assert gpu["max_waves_per_simd"] == props["max_waves_per_simd"]
    assert gpu["lds_kb"] == props["lds_size_in_kb"]
    assert gpu["wave_size"] == props["wave_front_size"]
    assert gpu["max_sclk_mhz"] == props["max_engine_clk_fcompute"]
    assert gpu["vram_total_bytes"] == mem_props["size_in_bytes"]
    expected_bandwidth = mem_props["width"] / 8.0 * mem_props["mem_clk_max"] * 2.0 / 1_000_000.0
    assert gpu["memory_bandwidth_tbs"] == pytest.approx(round(expected_bandwidth, 3))
    assert gpu["spec_sources"]["cu_count"] == "kfd-topology"
    assert gpu["spec_sources"]["memory_bandwidth_tbs"] == "derived-from-kfd-topology"

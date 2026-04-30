"""Tests for runtime GPU discovery."""

from __future__ import annotations

import json

import pytest

from perfxpert.tools import gpu_discovery
from perfxpert.tools._class import ToolClass


_SYNTHETIC_GFX_IDS = ("gfxabc", "gfxdef", "gfxghi", "gfxjkl")


def _rocminfo_sample(
    gfx_id: str = _SYNTHETIC_GFX_IDS[0],
    *,
    node_id: int = 1,
    max_sclk_mhz: int = 1234,
    cu_count: int = 12,
    simds_per_cu: int = 3,
    wave_size: int = 16,
    max_waves_per_cu: int = 24,
    lds_kb: int = 48,
) -> str:
    return f"""
*******
Agent 2
*******
  Name:                    {gfx_id}
  Uuid:                    GPU-c2171ff77417f1d6
  Marketing Name:          Synthetic GPU
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
    gfx_id = _SYNTHETIC_GFX_IDS[0]
    topology = {
        "node_id": 1,
        "cu_count": 12,
        "wave_size": 16,
        "simds_per_cu": 3,
        "max_waves_per_cu": 24,
        "max_sclk_mhz": 1234,
    }
    vram_total_bytes = 123456789
    rocm_smi_sclk_mhz = 321
    rocm_smi_mclk_mhz = 654
    amd_smi_mem_clk_mhz = 987
    amd_smi_total_vram_mb = 765
    rocm_smi_info = {
        "card0": {
            "Node ID": str(topology["node_id"]),
            "GFX Version": gfx_id,
            "Card Series": "Synthetic GPU",
            "VRAM Total Memory (B)": str(vram_total_bytes),
        }
    }
    rocm_smi_clocks = {
        "card0": {
            "sclk clock speed:": f"({rocm_smi_sclk_mhz}Mhz)",
            "mclk clock speed:": f"({rocm_smi_mclk_mhz}Mhz)",
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
                    "mem_0": {"max_clk": {"value": amd_smi_mem_clk_mhz, "unit": "MHz"}},
                },
                "mem_usage": {
                    "total_vram": {"value": amd_smi_total_vram_mb, "unit": "MB"},
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
    assert gpu["spec_sources"]["cu_count"] == "rocminfo"


@pytest.mark.parametrize("gfx_id", _SYNTHETIC_GFX_IDS[:2])
def test_discover_runtime_gpu_specs_filters_by_gfx(monkeypatch, gfx_id: str) -> None:
    monkeypatch.setattr(
        gpu_discovery,
        "_run_command",
        lambda argv: _rocminfo_sample(gfx_id) if argv == ["rocminfo"] else None,
    )
    monkeypatch.setattr(gpu_discovery, "_parse_kfd_topology", lambda: [])
    _clear_cache()

    assert gpu_discovery.discover_runtime_gpu_specs(gfx_id)["gpus"]
    assert gpu_discovery.discover_runtime_gpu_specs(_SYNTHETIC_GFX_IDS[2])["gpus"] == []


def test_merge_gpu_lists_preserves_same_arch_devices_with_distinct_nodes() -> None:
    gfx_id = _SYNTHETIC_GFX_IDS[0]
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


@pytest.mark.parametrize("gfx_id", _SYNTHETIC_GFX_IDS)
def test_parse_kfd_topology_supplies_rocminfo_fallback_fields(tmp_path, gfx_id: str) -> None:
    node = tmp_path / "1"
    mem = node / "mem_banks" / "0"
    mem.mkdir(parents=True)
    props = {
        "simd_count": 12,
        "max_waves_per_simd": 8,
        "lds_size_in_kb": 48,
        "wave_front_size": 16,
        "simd_per_cu": 3,
        "max_engine_clk_fcompute": 1234,
    }
    mem_props = {
        "heap_type": 1,
        "size_in_bytes": 123456789,
        "width": 96,
        "mem_clk_max": 789,
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
    bits_per_byte = 8
    transfers_per_clock = 2
    mhz_bytes_to_tbs = 1_000_000
    expected_bandwidth = (
        mem_props["width"] / bits_per_byte * mem_props["mem_clk_max"] * transfers_per_clock / mhz_bytes_to_tbs
    )
    assert gpu["memory_bandwidth_tbs"] == pytest.approx(round(expected_bandwidth, 3))
    assert gpu["spec_sources"]["cu_count"] == "kfd-topology"
    assert gpu["spec_sources"]["memory_bandwidth_tbs"] == "derived-from-kfd-topology"

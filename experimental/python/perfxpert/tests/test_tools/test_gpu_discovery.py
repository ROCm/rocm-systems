"""Tests for runtime GPU discovery."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from perfxpert.tools import gpu_discovery
from perfxpert.tools._class import ToolClass


@dataclass(frozen=True)
class _GpuCase:
    gfx_id: str
    node_id: int
    cu_count: int
    simds_per_cu: int
    wave_size: int
    max_waves_per_cu: int
    max_sclk_mhz: int
    lds_kb: int
    vram_total_bytes: int
    mem_width_bits: int
    mem_clk_mhz: int


def _gpu_case(case_index: int = 1) -> _GpuCase:
    # Generate valid gfx-shaped IDs and numeric fields without using real GPU
    # product specs; assertions below compare parser output to this fixture.
    suffix = chr(ord("a") + case_index - 1) * 3
    simds_per_cu = case_index + 2
    return _GpuCase(
        gfx_id=f"gfx{suffix}",
        node_id=case_index,
        cu_count=simds_per_cu * (case_index + 3),
        simds_per_cu=simds_per_cu,
        wave_size=8 * (case_index + 1),
        max_waves_per_cu=simds_per_cu * (case_index + 7),
        max_sclk_mhz=1000 + case_index * 234,
        lds_kb=16 * (case_index + 2),
        vram_total_bytes=100_000_000 + case_index * 23_456_789,
        mem_width_bits=32 * (case_index + 2),
        mem_clk_mhz=600 + case_index * 189,
    )


def _rocminfo_sample(
    case: _GpuCase,
) -> str:
    return f"""
*******
Agent 2
*******
  Name:                    {case.gfx_id}
  Uuid:                    GPU-c2171ff77417f1d6
  Marketing Name:          Synthetic GPU
  Vendor Name:             AMD
  Device Type:             GPU
  Node:                    {case.node_id}
  Max Clock Freq. (MHz):   {case.max_sclk_mhz}
  Compute Unit:            {case.cu_count}
  SIMDs per CU:            {case.simds_per_cu}
  Wavefront Size:          {case.wave_size}(0x{case.wave_size:x})
  Max Waves Per CU:        {case.max_waves_per_cu}(0x{case.max_waves_per_cu:x})
  Pool Info:
    Pool 2
      Segment:                 GROUP
      Size:                    {case.lds_kb}(0x{case.lds_kb:x}) KB
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
    case = _gpu_case()
    rocm_smi_sclk_mhz = case.max_sclk_mhz // 2
    rocm_smi_mclk_mhz = case.mem_clk_mhz
    amd_smi_mem_clk_mhz = case.mem_clk_mhz + 1
    amd_smi_total_vram_mb = case.vram_total_bytes // 1024 // 1024
    rocm_smi_info = {
        "card0": {
            "Node ID": str(case.node_id),
            "GFX Version": case.gfx_id,
            "Card Series": "Synthetic GPU",
            "VRAM Total Memory (B)": str(case.vram_total_bytes),
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
            "node_id": case.node_id,
            "bdf": f"synthetic-bdf-{case.node_id}",
            "uuid": f"synthetic-uuid-{case.node_id}",
        }
    ]
    amd_smi_metric = {
        "gpu_data": [
            {
                "gpu": 0,
                "clock": {
                    "gfx_0": {"max_clk": {"value": case.max_sclk_mhz, "unit": "MHz"}},
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
            return _rocminfo_sample(case)
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
    assert gpu["gfx_id"] == case.gfx_id
    assert gpu["cu_count"] == case.cu_count
    assert gpu["wave_size"] == case.wave_size
    assert gpu["max_waves_per_simd"] == case.max_waves_per_cu // case.simds_per_cu
    assert gpu["max_sclk_mhz"] == case.max_sclk_mhz
    assert gpu["vram_total_bytes"] == case.vram_total_bytes
    assert gpu["spec_sources"]["cu_count"] == "rocminfo"


@pytest.mark.parametrize("case_index", range(1, 3))
def test_discover_runtime_gpu_specs_filters_by_gfx(monkeypatch, case_index: int) -> None:
    case = _gpu_case(case_index)
    filtered_case = _gpu_case(case_index + 2)
    monkeypatch.setattr(
        gpu_discovery,
        "_run_command",
        lambda argv: _rocminfo_sample(case) if argv == ["rocminfo"] else None,
    )
    monkeypatch.setattr(gpu_discovery, "_parse_kfd_topology", lambda: [])
    _clear_cache()

    assert gpu_discovery.discover_runtime_gpu_specs(case.gfx_id)["gpus"]
    assert gpu_discovery.discover_runtime_gpu_specs(filtered_case.gfx_id)["gpus"] == []


def test_merge_gpu_lists_preserves_same_arch_devices_with_distinct_nodes() -> None:
    case = _gpu_case()
    node_ids = [1, 2]

    merged = gpu_discovery._merge_gpu_lists(
        [
            [
                {
                    "node_id": node_id,
                    "gfx_id": case.gfx_id,
                    "spec_sources": {"node_id": "rocminfo", "gfx_id": "rocminfo"},
                }
                for node_id in node_ids
            ]
        ]
    )

    assert [gpu["node_id"] for gpu in merged] == node_ids
    assert all(gpu["gfx_id"] == case.gfx_id for gpu in merged)


def test_discover_runtime_gpu_specs_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv(gpu_discovery.PERFXPERT_DISABLE_RUNTIME_GPU_SPECS, "1")
    _clear_cache()

    result = gpu_discovery.discover_runtime_gpu_specs()
    assert result["gpus"] == []
    assert "disabled" in result["errors"][0]


@pytest.mark.parametrize("case_index", range(1, 5))
def test_parse_kfd_topology_supplies_rocminfo_fallback_fields(tmp_path, case_index: int) -> None:
    case = _gpu_case(case_index)
    node = tmp_path / "1"
    mem = node / "mem_banks" / "0"
    mem.mkdir(parents=True)
    props = {
        "simd_count": case.cu_count * case.simds_per_cu,
        "max_waves_per_simd": case.max_waves_per_cu // case.simds_per_cu,
        "lds_size_in_kb": case.lds_kb,
        "wave_front_size": case.wave_size,
        "simd_per_cu": case.simds_per_cu,
        "max_engine_clk_fcompute": case.max_sclk_mhz,
    }
    mem_props = {
        "heap_type": 1,
        "size_in_bytes": case.vram_total_bytes,
        "width": case.mem_width_bits,
        "mem_clk_max": case.mem_clk_mhz,
    }
    (node / "name").write_text(f"{case.gfx_id}\n")
    (node / "properties").write_text("\n".join(f"{key} {value}" for key, value in props.items()))
    (mem / "properties").write_text("\n".join(f"{key} {value}" for key, value in mem_props.items()))

    result = gpu_discovery._parse_kfd_topology(tmp_path)

    assert len(result) == 1
    gpu = result[0]
    assert gpu["gfx_id"] == case.gfx_id
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

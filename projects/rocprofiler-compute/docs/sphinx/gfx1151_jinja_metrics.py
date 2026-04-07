# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# SPDX-License-Identifier: MIT
"""Build Sphinx ``.. jinja::`` metric-table payloads for gfx1151 from panel YAMLs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# (sphinx jinja context id, panel YAML stem without ``.yaml``, ``metric_table`` title)
_GFX1151_JINJA_TABLE_BINDINGS: tuple[tuple[str, str, str], ...] = (
    ("rdna1151-roofline-performance-rates-gfx1151", "0400_roofline", "Roofline Performance Rates"),
    ("rdna1151-roofline-plot-points-gfx1151", "0400_roofline", "Roofline Plot Points"),
    ("rdna1151-wgp-utilization-gfx1151", "0700_WGP", "Utilization"),
    ("rdna1151-wavefront-launch-stats-gfx1151", "0700_WGP", "Wavefront Launch Stats"),
    ("rdna1151-wave-dispatch-gfx1151", "0700_WGP", "Wave Dispatch"),
    ("rdna1151-wave-life-gfx1151", "0700_WGP", "Wave Life"),
    ("rdna1151-wave-instruction-mix-gfx1151", "0700_WGP", "Wave Instruction Mix"),
    ("rdna1151-vmem-instruction-mix-gfx1151", "0700_WGP", "VMEM Instruction Mix"),
    ("rdna1151-lds-instruction-mix-gfx1151", "0700_WGP", "LDS Instruction Mix"),
    ("rdna1151-wait-state-analysis-gfx1151", "0700_WGP", "Wait State Analysis"),
    ("rdna1151-wgp-instruction-cache-gfx1151", "0700_WGP", "Instruction Cache"),
    ("rdna1151-wgp-scalar-data-cache-gfx1151", "0700_WGP", "Scalar Data Cache"),
    ("rdna1151-gpu-utilization-gfx1151", "1700_GRBM", "GPU Utilization"),
    ("rdna1151-shader-engine-utilization-gfx1151", "1700_GRBM", "Shader Engine Utilization"),
    ("rdna1151-spi-utilization-gfx1151", "0600_SPI", "SPI Utilization"),
    ("rdna1151-wave-dispatch-statistics-gfx1151", "0600_SPI", "Wave Dispatch Statistics"),
    ("rdna1151-cpc-utilization-gfx1151", "0500_CPC", "CPC Utilization"),
    ("rdna1151-cpc-interface-utilization-gfx1151", "0500_CPC", "CPC Interface Utilization"),
    ("rdna1151-mec-stall-cycles-gfx1151", "0500_CPC", "MEC Stall Cycles"),
    ("rdna1151-cpc-memory-requests-gfx1151", "0500_CPC", "CPC Memory Requests"),
    ("rdna1151-mec-instruction-cache-gfx1151", "0500_CPC", "MEC Instruction Cache"),
    ("rdna1151-tcp-utilization-gfx1151", "0800_TCP_Cache", "TCP Utilization"),
    ("rdna1151-tcp-request-statistics-gfx1151", "0800_TCP_Cache", "TCP Request Statistics"),
    ("rdna1151-tcp-cache-performance-gfx1151", "0800_TCP_Cache", "TCP Cache Performance"),
    ("rdna1151-tcp-tcp-gl1-interface-gfx1151", "0800_TCP_Cache", "TCP-GL1 Interface"),
    ("rdna1151-tcp-stalls-gfx1151", "0800_TCP_Cache", "TCP Stalls"),
    ("rdna1151-gl1c-utilization-gfx1151", "1100_GL1C", "GL1C Utilization"),
    ("rdna1151-gl1c-request-statistics-gfx1151", "1100_GL1C", "GL1C Request Statistics"),
    ("rdna1151-gl1c-cache-performance-gfx1151", "1100_GL1C", "GL1C Cache Performance"),
    ("rdna1151-gl1c-stalls-gfx1151", "1100_GL1C", "GL1C Stalls"),
    ("rdna1151-memory-chart-instruction-cache-gfx1151", "0300_Memory_Chart", "Instruction Cache"),
    ("rdna1151-memory-chart-scalar-data-cache-gfx1151", "0300_Memory_Chart", "Scalar Data Cache"),
    ("rdna1151-memory-chart-tcp-cache-vector-l0-gfx1151", "0300_Memory_Chart", "TCP Cache (Vector L0)"),
    ("rdna1151-memory-chart-lds-local-data-share-gfx1151", "0300_Memory_Chart", "LDS (Local Data Share)"),
    ("rdna1151-memory-chart-tcp-gl1-interface-gfx1151", "0300_Memory_Chart", "TCP-GL1 Interface"),
    ("rdna1151-memory-chart-gl1c-cache-l1-gfx1151", "0300_Memory_Chart", "GL1C Cache (L1)"),
    ("rdna1151-gl1c-gl1c-gl2-interface-gfx1151", "1100_GL1C", "GL1C-GL2 Interface"),
    ("rdna1151-gl2c-cache-performance-gfx1151", "1300_GL2C", "GL2C Cache Performance"),
    ("rdna1151-gl2c-request-statistics-gfx1151", "1300_GL2C", "GL2C Request Statistics"),
    ("rdna1151-gl2c-bandwidth-gfx1151", "1300_GL2C", "GL2C Bandwidth"),
    ("rdna1151-dram-read-interface-gfx1151", "1500_GCEA", "DRAM Read Interface"),
    ("rdna1151-dram-write-interface-gfx1151", "1500_GCEA", "DRAM Write Interface"),
    ("rdna1151-system-arbiter-sarb-gfx1151", "1500_GCEA", "System Arbiter (SARB)"),
    ("rdna1151-return-interface-gfx1151", "1500_GCEA", "Return Interface"),
    ("rdna1151-memory-chart-gl1c-gl2-interface-gfx1151", "0300_Memory_Chart", "GL1C-GL2 Interface"),
    ("rdna1151-memory-chart-gl2c-cache-l2-gfx1151", "0300_Memory_Chart", "GL2C Cache (L2)"),
    (
        "rdna1151-memory-chart-gcea-to-system-memory-gfx1151",
        "0300_Memory_Chart",
        "Graphics Core Efficiency Arbiter (GCEA) to System Memory",
    ),
    ("sys-sol-gfx1151", "0200_System_Speed_Of_Light", "System Speed-of-Light"),
)

GFX1151_JINJA_CONTEXT_IDS: tuple[str, ...] = tuple(b[0] for b in _GFX1151_JINJA_TABLE_BINDINGS)

# metric_table title in analysis YAML -> section key in gfx1151_metrics_description.yaml
_TITLE_ALIASES: dict[str, str] = {
    (
        "Graphics Core Efficiency Arbiter (GCEA) to System Memory"
    ): "Memory chart — GCEA to System Memory",
}


def _rocprof_compute_root(gfx1151_dir: Path) -> Path:
    """``gfx1151_dir`` = ``.../analysis_configs/gfx1151``."""
    return gfx1151_dir.parent.parent.parent.parent


def _load_tools_nested_metric_sections(tools_yaml: Path) -> dict[str, dict[str, Any]]:
    """Top-level keys whose values are ``{metric_name: {rst, plain, unit}}``."""
    if not tools_yaml.is_file():
        return {}
    data = yaml.safe_load(tools_yaml.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    return {
        k: v
        for k, v in data.items()
        if isinstance(v, dict) and "rst" not in v and "plain" not in v
    }


def _tools_section_key(stem: str, title: str, nested: dict[str, Any]) -> str | None:
    """Map ``metric_table`` title to a gfx1151_metrics_description.yaml section."""
    if title in _TITLE_ALIASES:
        alias = _TITLE_ALIASES[title]
        if alias in nested:
            return alias
    candidates: list[str] = [title]
    if stem == "0200_System_Speed_Of_Light":
        candidates.append("System Speed-of-Light")
    elif stem == "0300_Memory_Chart":
        candidates.append(f"Memory chart — {title}")
    elif stem == "0700_WGP":
        candidates.append(f"WGP {title}")
    elif stem == "0800_TCP_Cache":
        candidates.append(f"TCP {title}")
    elif stem == "1100_GL1C":
        candidates.append(f"GL1C {title}")
    for c in candidates:
        if c in nested:
            return c
    return None


def _metric_unit(spec: Any) -> str:
    if isinstance(spec, dict):
        u = spec.get("unit") or spec.get("units")
        return str(u) if u is not None else ""
    return ""


def _parse_description_entry(entry: Any) -> tuple[str, str | None]:
    if isinstance(entry, dict):
        rst = (entry.get("rst") or entry.get("plain") or "").strip()
        unit = entry.get("unit")
        u_out = str(unit) if unit is not None else None
        return rst, u_out
    if isinstance(entry, str):
        return entry.strip(), None
    return "", None


def _rst_for_table_cell(rst: str) -> str:
    """Flatten newlines so list-table cells stay valid RST."""
    return " ".join(rst.split())


def _load_panel_config(yaml_path: Path) -> dict[str, Any]:
    with yaml_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    pc = data.get("Panel Config")
    return pc if isinstance(pc, dict) else {}


def _tables_by_title(panel: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for block in panel.get("data source", []):
        if not isinstance(block, dict):
            continue
        mt = block.get("metric_table")
        if not isinstance(mt, dict):
            continue
        title = mt.get("title")
        if title is None:
            continue
        metrics = mt.get("metric")
        if isinstance(metrics, dict):
            out[str(title)] = metrics
    return out


def build_gfx1151_jinja_contexts(gfx1151_dir: Path) -> dict[str, dict[str, dict[str, dict[str, str]]]]:
    """Map each RDNA gfx1151 jinja id to ``{"data": {metric: {rst, unit}}}``."""
    cache: dict[str, dict[str, Any]] = {}
    tools_nested = _load_tools_nested_metric_sections(
        _rocprof_compute_root(gfx1151_dir)
        / "tools"
        / "per_arch_metric_definitions"
        / "gfx1151_metrics_description.yaml"
    )

    def panel_for_stem(stem: str) -> dict[str, Any]:
        if stem not in cache:
            p = gfx1151_dir / f"{stem}.yaml"
            cache[stem] = _load_panel_config(p) if p.is_file() else {}
        return cache[stem]

    result: dict[str, dict[str, dict[str, dict[str, str]]]] = {}
    for ctx_id, stem, title in _GFX1151_JINJA_TABLE_BINDINGS:
        panel = panel_for_stem(stem)
        tables = _tables_by_title(panel)
        metrics_block = tables.get(title)
        raw_desc = panel.get("metrics_description") or {}
        descs: dict[str, Any] = raw_desc if isinstance(raw_desc, dict) else {}
        tools_key = _tools_section_key(stem, title, tools_nested)
        tools_table: dict[str, Any] = (
            tools_nested.get(tools_key, {}) if tools_key else {}
        )
        data: dict[str, dict[str, str]] = {}
        if metrics_block:
            for mname, mspec in metrics_block.items():
                unit_tbl = _metric_unit(mspec)
                entry = descs.get(mname)
                if entry is None and isinstance(tools_table, dict):
                    entry = tools_table.get(mname)
                rst_raw, unit_desc = _parse_description_entry(entry)
                rst = _rst_for_table_cell(rst_raw)
                unit = unit_desc if unit_desc else unit_tbl
                data[str(mname)] = {"rst": rst, "unit": unit}
        result[ctx_id] = {"data": data}
    return result

# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
RDNA3.5 Memory Architecture Diagram - CLI Visualization
=============================================================================

USAGE:
    python mem_chart_gfx11.py [--data metrics.json] [--debug]
        [--txt file.txt] [--svg file.svg]

API:
    normalize_mem_chart_metrics(metric_dict) -> flat ordered dict for UIs
    plot_mem_chart(..., *, chart_title=...) -> str
    format_mem_chart_heading(normal_unit, *, panel_id=300, section_label=...) -> str

Metric dict keys must match the Memory Chart panel YAML for RDNA3.5:

    src/rocprof_compute_soc/analysis_configs/gfx1151/0300_Memory_Chart.yaml

Use ``MEM_CHART_PANEL_METRIC_KEYS`` for the authoritative ordered list.
(If a future gfx target adds ``0300_memory_chart.yaml``, keep keys aligned there.)

Bandwidth values are **Bytes/s**, matching the YAML ``unit: Bytes/s`` rows.

RDNA3.5 MEMORY HIERARCHY (GCEA = Graphics Core Efficiency Arbiter):
   Kernel -> TCP (L0 Vector Cache) -> GL1C (L1) -> GL2C (L2) -> GCEA -> System Memory
         -> SQC (ICache/DCache)   -> GL1C (L1) -> GL2C (L2) -> GCEA -> System Memory
         -> LDS (Local Data Share) [stays on CU, no GL1C connection]
"""

import argparse
import json
import re
from io import StringIO
from typing import Any, Optional, Union

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from utils.utils_analysis import format_bw_human_readable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Keys = ``metric:`` names under each ``metric_table`` in
# ``analysis_configs/gfx1151/0300_Memory_Chart.yaml`` (tables 301–309), in panel order.
# Commented-out YAML metrics (e.g. TCP Atomic, LDS direct read/write) are omitted.
_MEM_CHART_DEFAULT_ROWS: tuple[tuple[str, Union[int, float]], ...] = (
    # Table 301: Instruction Cache
    ("ICache Requests", 450),
    ("ICache Utilization", 45.2),
    ("ICache Hit Rate", 98.5),
    ("ICache Miss Rate", 1.5),
    ("ICache Request Stall Rate", 2.1),
    ("ICache-GL1 Read Bandwidth", 57.6e9),
    # Table 302: Scalar Data Cache
    ("Dcache Requests", 225),
    ("Dcache Utilization", 38.7),
    ("Dcache Hit Rate", 95.3),
    ("Dcache Request Stall Rate", 1.8),
    ("Dcache-GL1 Read Bandwidth", 28.8e9),
    # Table 303: TCP Cache (Vector L0)
    ("TCP Total Requests", 1_250_000),
    ("TCP Read Requests", 875_000),
    ("TCP Write Requests", 375_000),
    ("TCP Miss Requests", 150_000),
    ("TCP Hit Rate", 88.0),
    ("TCP Request Bandwidth", 80e9),
    # Table 304: LDS
    ("LDS Instructions", 125_000),
    ("LDS Atomic Instructions", 10_000),
    ("LDS Instruction Cycles", 250_000),
    ("LDS Wait Cycles", 12_500),
    ("LDS Bank Conflict Rate", 4.0),
    ("LDS Estimated Bandwidth", 256e9),
    # Table 305: TCP-GL1 Interface
    ("TCP-GL1 Read Requests", 150_000),
    ("TCP-GL1 Write Requests", 50_000),
    ("TCP-GL1 Read Bandwidth", 96e9),
    ("TCP-GL1 Write Bandwidth", 32e9),
    # Table 306: GL1C Cache (L1)
    ("GL1C Utilization", 65.2),
    ("GL1C Total Requests", 200_000),
    ("GL1C Read Requests", 150_000),
    ("GL1C Write Requests", 50_000),
    ("GL1C Miss Requests", 30_000),
    ("GL1C Hit Rate", 85.0),
    ("GL1C Starve Rate", 5.2),
    ("GL1C Stall GL2 Backpressure", 8.5),
    # Table 307: GL1C-GL2 Interface
    ("GL1-GL2 Read Requests", 30_000),
    ("GL1-GL2 Write Requests", 10_000),
    ("GL1-GL2 Read Bandwidth", 48e9),
    ("GL1-GL2 Write Bandwidth", 16e9),
    ("GL1-GL2 Read Latency", 85.2),
    ("GL1-GL2 Write Latency", 62.4),
    # Table 308: GL2C Cache (L2)
    ("GL2C Utilization", 74.2),
    ("GL2C Total Requests", 40_000),
    ("GL2C Read Requests", 30_000),
    ("GL2C Write Requests", 10_000),
    ("GL2C Atomic Requests", 1_000),
    ("GL2C Hit Rate", 82.5),
    ("GL2C Read Bandwidth", 64e9),
    ("GL2C Write Bandwidth", 24e9),
    # Table 309: Graphics Core Efficiency Arbiter (GCEA) to System Memory
    ("SARB Utilization", 52.3),
    ("SARB Stall Rate", 12.4),
    ("DRAM Read Requests", 25_000),
    ("DRAM Write Requests", 8_000),
    ("DRAM Read Bandwidth", 100e9),
    ("DRAM Write Bandwidth", 60e9),
    ("Read Returns", 25_000),
    ("Write Returns", 8_000),
)

MEM_CHART_PANEL_METRIC_KEYS: tuple[str, ...] = tuple(
    k for k, _ in _MEM_CHART_DEFAULT_ROWS
)

DEFAULT_SAMPLE_METRICS: dict[str, Union[int, float]] = dict(_MEM_CHART_DEFAULT_ROWS)

COLORS = {
    "kernel": "green",
    "block": "blue",
    "tcp": "cyan",
    "lds": "magenta",
    "sqc": "yellow",
    "read": "bright_cyan",
    "write": "bright_yellow",
    "atomic": "bright_magenta",
    "util": "bright_green",
    "hit": "yellow",
    "stall": "indian_red",
    "bw": "bright_cyan",
}

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_value(
    value: Union[int, float, str, None], unit: str = "", precision: int = 1
) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        try:
            value = float(value)
        except (ValueError, TypeError):
            return value
    if unit == "%":
        return f"{value:.{precision}f}%"
    elif unit in ("GB/s", "Bytes/s"):
        return format_bw_human_readable(value, unit, precision)
    else:
        return f"{value:.{precision}f}{unit}"


def format_sci(value: Union[int, float, str, None], precision: int = 2) -> str:
    if value is None:
        return "N/A"
    try:
        value = float(value)
    except (ValueError, TypeError):
        return "N/A"
    if abs(value) < 1000:
        return f"{int(value)}"
    return f"{value:.{precision}e}"


def metric_line(
    label: str,
    value: Any,  # noqa: ANN401
    unit: str = "%",
    color: str = "bright_green",
) -> str:
    formatted = format_value(value, unit)
    return f"{label} [{color}]{formatted}[/{color}]"


def bar(pct: float, w: int = 10) -> str:
    if pct is None:
        return "░" * w
    try:
        pct = float(pct)
    except (ValueError, TypeError):
        return "░" * w
    filled = int(w * min(100, max(0, pct)) / 100)
    return "█" * filled + "░" * (w - filled)


def _safe_float_sum(
    *values: Union[int, float, str, None],
) -> Optional[float]:
    """Sum non-None numeric values. Returns None if no value is valid."""
    total = 0.0
    any_valid = False
    for v in values:
        if v is not None:
            try:
                total += float(v)
                any_valid = True
            except (ValueError, TypeError):
                pass
    return total if any_valid else None


def _fmt_edge(
    label: str,
    value: Any,  # noqa: ANN401
    width: int = 7,
) -> str:
    label_str = f"{label:<{width}}"
    if value is not None:
        value_str = f": {format_sci(value):>7}"
    else:
        value_str = ""
    return f"{label_str}{value_str}"


# ---------------------------------------------------------------------------
# Public API: heading, normalization, sample data
# ---------------------------------------------------------------------------


def _print_mem_chart_scope_bar(console: Console) -> None:
    """Horizontal rule: GPU span vs System Memory (above the diagram body)."""
    console.print(
        "|"
        + "-" * 62
        + " [dim]GPU[/dim] "
        + "-" * 62
        + "|"
        + "-" * 4
        + " [dim]System Memory[/dim] "
        + "-" * 4
        + "|"
    )


def format_mem_chart_heading(
    normal_unit: str,
    *,
    panel_id: int = 300,
    section_label: str = "Memory Chart",
) -> str:
    """Build CLI diagram title: ``{panel_id//100}. {label} (Normalization: …)``.

    Matches other panels (e.g. ``3. System Speed-of-Light``) where the leading
    number is ``Panel Config id // 100`` (panel 300 → ``3.``).
    """
    section = max(0, int(panel_id)) // 100
    return f"{section}. {section_label} (Normalization: {normal_unit})"


def normalize_mem_chart_metrics(metric_dict: dict[str, Any]) -> dict[str, Any]:
    """Return a single flat map: YAML metric name -> value, panel order.

    All keys in ``MEM_CHART_PANEL_METRIC_KEYS`` are present; unknown input keys
    are dropped. Use before rendering or serializing for front-ends.
    """
    return {k: metric_dict.get(k) for k in MEM_CHART_PANEL_METRIC_KEYS}


def get_sample_metrics() -> dict[str, Any]:
    """Return sample metrics (flat panel order) for testing or demos."""
    return normalize_mem_chart_metrics(DEFAULT_SAMPLE_METRICS.copy())


# ---------------------------------------------------------------------------
# Diagram construction: _extract_metrics, _build_kernel_and_l0,
#   _build_cache_columns, _build_memory_columns, create_mem_chart_diagram
# ---------------------------------------------------------------------------


def _extract_metrics(metric_dict: dict[str, Any]) -> dict[str, Any]:
    """Pull all needed values from the flat metric dict."""
    m: dict[str, Any] = {}

    m["icache_req"] = metric_dict.get("ICache Requests")
    m["icache_hit"] = metric_dict.get("ICache Hit Rate")
    m["icache_gl1_bw"] = metric_dict.get("ICache-GL1 Read Bandwidth")

    m["dcache_req"] = metric_dict.get("Dcache Requests")
    m["dcache_hit"] = metric_dict.get("Dcache Hit Rate")
    m["dcache_gl1_bw"] = metric_dict.get("Dcache-GL1 Read Bandwidth")

    m["tcp_read_req"] = metric_dict.get("TCP Read Requests")
    m["tcp_write_req"] = metric_dict.get("TCP Write Requests")
    m["tcp_hit"] = metric_dict.get("TCP Hit Rate")
    m["tcp_bw"] = metric_dict.get("TCP Request Bandwidth")

    m["lds_insts"] = metric_dict.get("LDS Instructions")
    m["lds_inst_cycles"] = metric_dict.get("LDS Instruction Cycles")
    m["lds_atomic_insts"] = metric_dict.get("LDS Atomic Instructions")
    m["lds_bw"] = metric_dict.get("LDS Estimated Bandwidth")
    m["lds_bank_conflict"] = metric_dict.get("LDS Bank Conflict Rate")

    m["tcp_gl1_read_bw"] = metric_dict.get("TCP-GL1 Read Bandwidth")
    m["tcp_gl1_write_bw"] = metric_dict.get("TCP-GL1 Write Bandwidth")

    m["sqc_gl1_read_bw"] = _safe_float_sum(m["icache_gl1_bw"], m["dcache_gl1_bw"])

    m["gl1c_util"] = metric_dict.get("GL1C Utilization")
    m["gl1c_hit"] = metric_dict.get("GL1C Hit Rate")
    m["gl1c_stall_gl2"] = metric_dict.get("GL1C Stall GL2 Backpressure")

    m["gl1_gl2_read_bw"] = metric_dict.get("GL1-GL2 Read Bandwidth")
    m["gl1_gl2_write_bw"] = metric_dict.get("GL1-GL2 Write Bandwidth")

    m["gl2c_util"] = metric_dict.get("GL2C Utilization")
    m["gl2c_hit"] = metric_dict.get("GL2C Hit Rate")
    m["gl2c_read_bw"] = metric_dict.get("GL2C Read Bandwidth")
    m["gl2c_write_bw"] = metric_dict.get("GL2C Write Bandwidth")

    m["sarb_util"] = metric_dict.get("SARB Utilization")
    m["sarb_stall"] = metric_dict.get("SARB Stall Rate")
    m["dram_read_bw"] = metric_dict.get("DRAM Read Bandwidth", 0)
    m["dram_write_bw"] = metric_dict.get("DRAM Write Bandwidth", 0)

    m["total_bw"] = (m["dram_read_bw"] or 0) + (m["dram_write_bw"] or 0)

    return m


def _build_kernel_and_l0(
    m: dict[str, Any],
    kernel_arrows: dict[str, str],
    std_arrows: dict[str, str],
) -> tuple[Panel, Text, Table, Text]:
    """Build the Kernel panel, kernel edges, LDS/TCP/SQC stack, and GL1 edges.

    Returns (kernel_panel, kernel_edges_text, l0_stack, gl1_edges_text).
    """
    fmt_bw = format_bw_human_readable
    c_rd = COLORS["read"]
    c_wr = COLORS["write"]
    c_at = COLORS["atomic"]
    c_bl = COLORS["block"]

    # Kernel panel (height = 10+10+10 = 30 to match L0 stack)
    kernel_panel = Panel(
        "\n" * 11 + "[dim]Shader Core[/dim]\n[dim]Wave Execution[/dim]",
        title=(f"[bold {COLORS['kernel']}]Kernel[/bold {COLORS['kernel']}]"),
        border_style=COLORS["kernel"],
        width=14,
        height=30,
    )

    # Kernel edges — LDS, TCP, SQC groups
    ka_l = kernel_arrows["left"]
    ka_r = kernel_arrows["right"]
    ka_b = kernel_arrows["both"]
    kernel_edges_lines = [
        "",
        "     [white]Request[/white]",
        "",
        f"[{c_rd}]{_fmt_edge('Read', m['lds_insts'])}[/{c_rd}]",
        f"[{c_rd}]{ka_l}[/{c_rd}]",
        f"[{c_wr}]{_fmt_edge('Write', m['lds_inst_cycles'])}[/{c_wr}]",
        f"[{c_wr}]{ka_r}[/{c_wr}]",
        f"[{c_at}]{_fmt_edge('Atomic', m['lds_atomic_insts'])}[/{c_at}]",
        f"[{c_at}]{ka_b}[/{c_at}]",
        "",
        "",
        "",
        "",
        f"[{c_rd}]{_fmt_edge('Read', m['tcp_read_req'])}[/{c_rd}]",
        f"[{c_rd}]{ka_l}[/{c_rd}]",
        f"[{c_wr}]{_fmt_edge('Write', m['tcp_write_req'])}[/{c_wr}]",
        f"[{c_wr}]{ka_r}[/{c_wr}]",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        f"[{c_rd}]{_fmt_edge('ICache', m['icache_req'])}[/{c_rd}]",
        f"[{c_rd}]{ka_l}[/{c_rd}]",
        f"[{c_rd}]{_fmt_edge('DCache', m['dcache_req'])}[/{c_rd}]",
        f"[{c_rd}]{ka_l}[/{c_rd}]",
    ]
    kernel_edges_text = Text.from_markup("\n".join(kernel_edges_lines))

    # LDS panel
    lds_bw_line = (
        metric_line("BW", m["lds_bw"], "Bytes/s", COLORS["bw"]) if m["lds_bw"] else ""
    )
    lds_conflict_line = (
        metric_line("Bank Conflict", m["lds_bank_conflict"], "%", COLORS["stall"])
        if m["lds_bank_conflict"]
        else ""
    )
    lds_panel = Panel(
        f"{lds_bw_line}\n{lds_conflict_line}",
        title=f"[bold {c_bl}]LDS[/bold {c_bl}]",
        border_style=c_bl,
        width=20,
        height=10,
    )

    # TCP panel
    tcp_bw_line = (
        metric_line("BW", m["tcp_bw"], "Bytes/s", COLORS["bw"]) if m["tcp_bw"] else ""
    )
    tcp_panel = Panel(
        f"{metric_line('Hit Rate', m['tcp_hit'], '%', COLORS['hit'])}\n{tcp_bw_line}",
        title=f"[bold {c_bl}]TCP (L0)[/bold {c_bl}]",
        border_style=c_bl,
        width=20,
        height=10,
    )

    # SQC panel
    sqc_panel = Panel(
        f"{metric_line('ICache', m['icache_hit'], '%', COLORS['hit'])}\n"
        f"{metric_line('DCache', m['dcache_hit'], '%', COLORS['hit'])}",
        title=f"[bold {c_bl}]SQC[/bold {c_bl}]",
        border_style=c_bl,
        width=20,
        height=10,
    )

    # Stack LDS, TCP, SQC vertically
    l0_stack = Table.grid(padding=0)
    l0_stack.add_column()
    l0_stack.add_row(lds_panel)
    l0_stack.add_row(tcp_panel)
    l0_stack.add_row(sqc_panel)

    # Edges to GL1C (TCP and SQC connect, LDS does NOT)
    sa_l = std_arrows["left"]
    sa_r = std_arrows["right"]
    tcp_gl1_rd = fmt_bw(m["tcp_gl1_read_bw"], precision=1)
    tcp_gl1_wr = fmt_bw(m["tcp_gl1_write_bw"], precision=1)
    sqc_gl1_rd = fmt_bw(m["sqc_gl1_read_bw"], precision=1)
    gl1_edges_lines = [
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        f"[{c_rd}]Read BW[/{c_rd}]",
        f"[{c_rd}]{tcp_gl1_rd}[/{c_rd}]",
        f"[{c_rd}]{sa_l}[/{c_rd}]",
        "",
        f"[{c_wr}]Write BW[/{c_wr}]",
        f"[{c_wr}]{tcp_gl1_wr}[/{c_wr}]",
        f"[{c_wr}]{sa_r}[/{c_wr}]",
        "",
        "",
        "",
        f"[{c_rd}]Read BW[/{c_rd}]",
        f"[{c_rd}]{sqc_gl1_rd}[/{c_rd}]",
        f"[{c_rd}]{sa_l}[/{c_rd}]",
    ]
    gl1_edges_text = Text.from_markup("\n".join(gl1_edges_lines))

    return kernel_panel, kernel_edges_text, l0_stack, gl1_edges_text


def _build_cache_columns(
    m: dict[str, Any],
    std_arrows: dict[str, str],
) -> tuple[Panel, Text, Panel]:
    """Build GL1C panel, GL1-GL2 edges, GL2C panel.

    Returns (gl1_panel, gl1_gl2_edges_text, gl2_panel).
    """
    fmt_bw = format_bw_human_readable
    c_rd = COLORS["read"]
    c_wr = COLORS["write"]
    c_bl = COLORS["block"]
    sa_l = std_arrows["left"]
    sa_r = std_arrows["right"]

    gl1_panel = Panel(
        f"{metric_line('Util', m['gl1c_util'], '%', COLORS['util'])}\n"
        f"[dim]{bar(m['gl1c_util'])}[/dim]\n"
        "\n"
        f"{metric_line('Hit Rate', m['gl1c_hit'], '%', COLORS['hit'])}\n"
        f"[dim]{bar(m['gl1c_hit'])}[/dim]\n"
        "\n"
        f"{metric_line('GL2 Stall', m['gl1c_stall_gl2'], '%', COLORS['stall'])}",
        title=f"[bold {c_bl}]GL1C[/bold {c_bl}]",
        border_style=c_bl,
        width=16,
        height=30,
    )

    rd_bw = fmt_bw(m["gl1_gl2_read_bw"], precision=1)
    wr_bw = fmt_bw(m["gl1_gl2_write_bw"], precision=1)
    gl1_gl2_edges_lines = [
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        f"[{c_rd}]Read BW[/{c_rd}]",
        f"[{c_rd}]{rd_bw}[/{c_rd}]",
        f"[{c_rd}]{sa_l}[/{c_rd}]",
        "",
        f"[{c_wr}]Write BW[/{c_wr}]",
        f"[{c_wr}]{wr_bw}[/{c_wr}]",
        f"[{c_wr}]{sa_r}[/{c_wr}]",
        "",
        "",
        "",
        "",
    ]
    gl1_gl2_edges_text = Text.from_markup("\n".join(gl1_gl2_edges_lines))

    gl2_panel = Panel(
        f"{metric_line('Util', m['gl2c_util'], '%', COLORS['util'])}\n"
        f"[dim]{bar(m['gl2c_util'])}[/dim]\n"
        "\n"
        f"{metric_line('Hit Rate', m['gl2c_hit'], '%', COLORS['hit'])}\n"
        f"[dim]{bar(m['gl2c_hit'])}[/dim]",
        title=f"[bold {c_bl}]GL2C[/bold {c_bl}]",
        border_style=c_bl,
        width=16,
        height=30,
    )

    return gl1_panel, gl1_gl2_edges_text, gl2_panel


def _build_memory_columns(
    m: dict[str, Any],
    std_arrows: dict[str, str],
) -> tuple[Text, Panel, Text, Panel]:
    """Build GL2-GCEA edges, GCEA panel, DRAM edges, DRAM panel.

    Returns (gl2_gcea_edges_text, gcea_panel, dram_edges_text, dram_panel).
    """
    fmt_bw = format_bw_human_readable
    c_rd = COLORS["read"]
    c_wr = COLORS["write"]
    c_bl = COLORS["block"]
    sa_l = std_arrows["left"]
    sa_r = std_arrows["right"]

    gl2_rd = fmt_bw(m["gl2c_read_bw"], precision=1)
    gl2_wr = fmt_bw(m["gl2c_write_bw"], precision=1)
    gl2_gcea_edges_lines = [
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        f"[{c_rd}]Read BW[/{c_rd}]",
        f"[{c_rd}]{gl2_rd}[/{c_rd}]",
        f"[{c_rd}]{sa_l}[/{c_rd}]",
        "",
        f"[{c_wr}]Write BW[/{c_wr}]",
        f"[{c_wr}]{gl2_wr}[/{c_wr}]",
        f"[{c_wr}]{sa_r}[/{c_wr}]",
        "",
        "",
        "",
        "",
    ]
    gl2_gcea_edges_text = Text.from_markup("\n".join(gl2_gcea_edges_lines))

    gcea_panel = Panel(
        f"{metric_line('SysArb Util', m['sarb_util'], '%', COLORS['util'])}\n"
        f"[dim]{bar(m['sarb_util'])}[/dim]\n"
        "\n"
        f"{metric_line('Stall', m['sarb_stall'], '%', COLORS['stall'])}",
        title=f"[bold {c_bl}]GCEA[/bold {c_bl}]",
        border_style=c_bl,
        width=16,
        height=30,
    )

    dram_rd = fmt_bw(m["dram_read_bw"], precision=1)
    dram_wr = fmt_bw(m["dram_write_bw"], precision=1)
    dram_edges_lines = [
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        f"[{c_rd}]Read BW[/{c_rd}]",
        f"[{c_rd}]{dram_rd}[/{c_rd}]",
        f"[{c_rd}]{sa_l}[/{c_rd}]",
        "",
        f"[{c_wr}]Write BW[/{c_wr}]",
        f"[{c_wr}]{dram_wr}[/{c_wr}]",
        f"[{c_wr}]{sa_r}[/{c_wr}]",
        "",
        "",
        "",
        "",
    ]
    dram_edges_text = Text.from_markup("\n".join(dram_edges_lines))

    total = fmt_bw(m["total_bw"], precision=1)
    dram_panel = Panel(
        "[dim]DDR5/LPDDR5[/dim]\n"
        "\n"
        f"Total: [bold bright_green]{total}[/bold bright_green]",
        title=f"[bold {c_bl}]DRAM[/bold {c_bl}]",
        border_style=c_bl,
        width=16,
        height=30,
    )

    return gl2_gcea_edges_text, gcea_panel, dram_edges_text, dram_panel


# ---------------------------------------------------------------------------
# Main diagram assembly
# ---------------------------------------------------------------------------


def create_mem_chart_diagram(
    metric_dict: dict[str, Any],
    console: Console,
    show_debug: bool = False,
    chart_title: str = "",
) -> None:
    """Create the RDNA3.5 memory diagram (TCP, LDS, SQC blocks).

    ``chart_title``: printed once above the diagram (e.g. from YAML panel title +
    normalization unit).
    """
    m = _extract_metrics(metric_dict)

    console.print()
    if chart_title:
        console.print(f"[bold]{chart_title}[/bold]")
    _print_mem_chart_scope_bar(console)
    console.print()

    # Arrow constants
    std_arrow_len = 8
    std_arrows = {
        "left": "<" + "-" * std_arrow_len,
        "right": "-" * std_arrow_len + ">",
    }

    kernel_edge_width = 16
    kernel_arrows = {
        "left": "<" + "-" * (kernel_edge_width - 1),
        "right": "-" * (kernel_edge_width - 1) + ">",
        "both": "<" + "-" * (kernel_edge_width - 2) + ">",
    }

    # Build layout columns
    kernel_panel, kernel_edges, l0_stack, gl1_edges = _build_kernel_and_l0(
        m, kernel_arrows, std_arrows
    )
    gl1_panel, gl1_gl2_edges, gl2_panel = _build_cache_columns(m, std_arrows)
    gl2_gcea_edges, gcea_panel, dram_edges, dram_panel = _build_memory_columns(
        m, std_arrows
    )

    # Assemble 11-column grid
    main_layout = Table.grid(padding=0)
    for _ in range(11):
        main_layout.add_column()

    main_layout.add_row(
        kernel_panel,
        kernel_edges,
        l0_stack,
        gl1_edges,
        gl1_panel,
        gl1_gl2_edges,
        gl2_panel,
        gl2_gcea_edges,
        gcea_panel,
        dram_edges,
        dram_panel,
    )

    console.print(main_layout)
    console.print()
    legend = (
        f"[dim]Legend:[/dim] "
        f"[{COLORS['read']}]<----[/{COLORS['read']}] Read  "
        f"[{COLORS['write']}]---->[/{COLORS['write']}] Write  "
        f"[{COLORS['atomic']}]<--->[/{COLORS['atomic']}] Atomic  "
        f"[{COLORS['util']}]█[/{COLORS['util']}] Util  "
        f"[{COLORS['hit']}]█[/{COLORS['hit']}] Hit%  "
        f"[{COLORS['stall']}]█[/{COLORS['stall']}] Stall"
    )
    console.print(legend)
    console.print()

    if show_debug:
        console.print("[dim]Architecture Notes:[/dim]")
        console.print("  TCP (Texture Cache Pipe): L0 vector cache for VMEM operations")
        console.print("  LDS (Local Data Share): On-CU scratchpad, NO GL1C connection")
        console.print("  SQC (Sequencer Cache): ICache + DCache for scalar operations")
        console.print()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def plot_mem_chart(
    arch: str,
    normal_unit: str,
    metric_dict: dict[str, Any],
    *,
    chart_title: Optional[str] = None,
) -> str:
    """Plot the memory chart and return as string.

    ``metric_dict`` keys should match ``0300_Memory_Chart.yaml`` (gfx1151), i.e.
    ``MEM_CHART_PANEL_METRIC_KEYS``. Values for bandwidth metrics are in **Bytes/s**.
    Input is normalized to a flat ordered dict before rendering.

    ``chart_title``: full heading line; if omitted, uses ``format_mem_chart_heading``
    with ``panel_id=300`` (section ``3.``).
    """
    flat = normalize_mem_chart_metrics(metric_dict)
    resolved_heading = (
        format_mem_chart_heading(normal_unit, panel_id=300)
        if chart_title is None
        else chart_title
    )
    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=200, height=80)
    create_mem_chart_diagram(
        flat,
        console,
        show_debug=False,
        chart_title=resolved_heading,
    )
    return buf.getvalue()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    arg_parser = argparse.ArgumentParser(
        description="RDNA3.5 Memory Chart - CLI Visualization"
    )
    arg_parser.add_argument("--data", "-d", help="JSON file with metrics data")
    arg_parser.add_argument("--debug", action="store_true", help="Show debug info")
    arg_parser.add_argument("--arch", default="gfx1151", help="Architecture name")
    arg_parser.add_argument("--norm", default="per_kernel", help="Normalization unit")
    arg_parser.add_argument("--txt", "-t", help="Output to plain text file")
    arg_parser.add_argument("--svg", help="Output to SVG file")
    args = arg_parser.parse_args()

    if args.data:
        with open(args.data) as f:
            metric_dict = normalize_mem_chart_metrics(json.load(f))
    else:
        metric_dict = normalize_mem_chart_metrics(DEFAULT_SAMPLE_METRICS.copy())

    heading = format_mem_chart_heading(args.norm, panel_id=300)

    if args.txt:
        buf = StringIO()
        console = Console(
            file=buf,
            force_terminal=True,
            width=200,
            height=80,
            no_color=True,
        )
        create_mem_chart_diagram(
            metric_dict,
            console,
            args.debug,
            chart_title=heading,
        )
        output = buf.getvalue()
        plain = re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", output)
        with open(args.txt, "w") as f:
            f.write(plain)
        print(f"Output written to {args.txt}")
    elif args.svg:
        svg_console = Console(
            file=StringIO(),
            force_terminal=True,
            width=200,
            height=80,
            record=True,
        )
        create_mem_chart_diagram(
            metric_dict,
            svg_console,
            args.debug,
            chart_title=heading,
        )
        svg_output = svg_console.export_svg(title=heading)
        with open(args.svg, "w") as f:
            f.write(svg_output)
        print(f"SVG saved to {args.svg}")
    else:
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=200, height=80)
        create_mem_chart_diagram(
            metric_dict,
            console,
            args.debug,
            chart_title=heading,
        )
        print(buf.getvalue())


if __name__ == "__main__":
    main()

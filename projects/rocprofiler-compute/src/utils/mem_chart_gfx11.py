# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""RDNA3.5 memory chart renderer.

Hierarchy (GCEA = Graphics Core Efficiency Arbiter):
  Kernel -> GL0 (TCP) / SQC -> GL1 -> GL2 -> GCEA -> System Memory
         -> LDS (on-CU, no GL1 connection)

Metric keys must match ``gfx115x/0300_memory_chart.yaml``.
"""

from typing import Any, Optional, Union

from rich.align import VerticalAlignMethod
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from utils.mem_chart_common import (
    COLORS,
    build_arch_notes,
    build_bw_edge_column,
    build_ip_block,
    build_kernel_panel,
    build_legend,
    colored,
    format_edge,
    format_value,
    make_arrows,
    mem_chart_cli_main,
    metric_line,
    pad_to,
    progress_bar,
    render_chart_to_string,
    safe_float_sum,
    stack_metrics,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Keys = ``metric:`` names under each ``metric_table`` in
# ``analysis_configs/gfx115x/0300_memory_chart.yaml`` (tables 301–309), in panel order.
# Commented-out YAML metrics (e.g. TCP Atomic) are omitted.
_MEM_CHART_DEFAULT_ROWS: tuple[tuple[str, Union[int, float]], ...] = (
    # Table 301: Instruction Cache
    ("ICache Requests", 450),
    ("ICache Hit Rate", 98.5),
    ("ICache-GL1 Read Bandwidth", 57.6e9),
    # Table 302: Scalar Data Cache
    ("DCache Requests", 225),
    ("DCache Hit Rate", 95.3),
    ("DCache-GL1 Read Bandwidth", 28.8e9),
    # Table 303: TCP Cache (GL0 Vector Cache)
    ("TCP Read Requests", 875_000),
    ("TCP Write Requests", 375_000),
    ("GL0 Cache Hit Rate (TCP Cache)", 88.0),
    ("GL0 Cache BW (TCP Cache)", 80e9),
    # Table 304: LDS
    ("LDS Req", 125_000),
    ("LDS Utilization", 62.0),
    ("LDS Bank Conflict Rate", 4.0),
    ("LDS Estimated Bandwidth", 256e9),
    # Table 305: TCP-GL1 Interface
    ("TCP-GL1 Read Bandwidth", 96e9),
    ("TCP-GL1 Write Bandwidth", 32e9),
    # Table 306: GL1 Cache (L1)
    ("GL1 Cache Utilization", 65.2),
    ("GL1 Cache Hit Rate", 85.0),
    ("GL1 Cache Stall GL2 Backpressure", 8.5),
    # Table 307: GL1-GL2 Interface
    ("GL1-GL2 Read Bandwidth", 48e9),
    ("GL1-GL2 Write Bandwidth", 16e9),
    # Table 308: GL2 Cache (L2)
    ("GL2 Cache Utilization", 74.2),
    ("GL2 Cache Hit Rate", 82.5),
    ("GL2-Fabric Read BW", 64e9),
    ("GL2-Fabric Write BW", 24e9),
    # Table 309: Graphics Core Efficiency Arbiter (GCEA) to System Memory
    ("SysArb Utilization", 52.3),
    ("SysArb Stall Rate", 12.4),
    ("DRAM Read Bandwidth", 100e9),
    ("DRAM Write Bandwidth", 60e9),
)

MEM_CHART_PANEL_METRIC_KEYS: tuple[str, ...] = tuple(
    k for k, _ in _MEM_CHART_DEFAULT_ROWS
)

DEFAULT_SAMPLE_METRICS: dict[str, Union[int, float]] = dict(_MEM_CHART_DEFAULT_ROWS)

_EDGE_LABEL_W = 11  # fits longest kernel-edge label ("ICache Read")

# Panel dimensions
_L0_PANEL_W = 20  # LDS/TCP/SQC: fits hit% + bar + BW line
_L0_PANEL_H = 10  # each L0 panel height
_CACHE_PANEL_W = 16  # GL1/GL2/GCEA/DRAM: fewer metrics, narrower
_TOTAL_H = 3 * _L0_PANEL_H  # all columns align to this height

# Arrow lengths
_STD_ARROW_LEN = 8  # edge arrows between cache panels
_KERNEL_EDGE_W = 20  # wider Kernel->L0 arrows (longer labels)
_EDGE_COLS = (5, 7, 9)  # grid column indices that use vertical="middle"

# Console dimensions
_CONSOLE_WIDTH = 200  # RDNA3.5 narrower than CDNA's 240


# ---------------------------------------------------------------------------
# Public API: heading, normalization, sample data
# ---------------------------------------------------------------------------


def normalize_mem_chart_metrics(metric_dict: dict[str, Any]) -> dict[str, Any]:
    """Flat map of YAML metric names in panel order; unknown keys dropped."""
    return {k: metric_dict.get(k) for k in MEM_CHART_PANEL_METRIC_KEYS}


def get_sample_metrics() -> dict[str, Any]:
    """Return sample metrics (flat panel order) for testing or demos."""
    return normalize_mem_chart_metrics(DEFAULT_SAMPLE_METRICS.copy())


# ---------------------------------------------------------------------------
# Diagram construction
# ---------------------------------------------------------------------------


def _scope_bar_section(label: str, span: int) -> str:
    """One labeled section of the scope rule."""
    markup_label = f" [dim]{label}[/dim] "
    visible_len = Text.from_markup(markup_label).cell_len
    inner = max(visible_len, span - 1)
    remaining = inner - visible_len
    left = remaining // 2
    return "|" + "-" * left + markup_label + "-" * (remaining - left)


def _scope_bar(gpu_span: int, sysmem_span: int) -> str:
    """Horizontal scope rule sized from the assembled layout."""
    return (
        _scope_bar_section("GPU", gpu_span)
        + _scope_bar_section("System Memory", sysmem_span - 1)
        + "|"
    )


def _extract_metrics(metric_dict: dict[str, Any]) -> dict[str, Any]:
    """Map YAML metric names to short internal keys."""
    metrics: dict[str, Any] = {}

    metrics["icache_req"] = metric_dict.get("ICache Requests")
    metrics["icache_hit"] = metric_dict.get("ICache Hit Rate")
    metrics["icache_gl1_bw"] = metric_dict.get("ICache-GL1 Read Bandwidth")

    metrics["dcache_req"] = metric_dict.get("DCache Requests")
    metrics["dcache_hit"] = metric_dict.get("DCache Hit Rate")
    metrics["dcache_gl1_bw"] = metric_dict.get("DCache-GL1 Read Bandwidth")

    metrics["tcp_read_req"] = metric_dict.get("TCP Read Requests")
    metrics["tcp_write_req"] = metric_dict.get("TCP Write Requests")
    metrics["tcp_hit"] = metric_dict.get("GL0 Cache Hit Rate (TCP Cache)")
    metrics["tcp_bw"] = metric_dict.get("GL0 Cache BW (TCP Cache)")

    metrics["lds_req"] = metric_dict.get("LDS Req")
    metrics["lds_util"] = metric_dict.get("LDS Utilization")
    metrics["lds_bw"] = metric_dict.get("LDS Estimated Bandwidth")
    metrics["lds_bank_conflict"] = metric_dict.get("LDS Bank Conflict Rate")

    metrics["tcp_gl1_read_bw"] = metric_dict.get("TCP-GL1 Read Bandwidth")
    metrics["tcp_gl1_write_bw"] = metric_dict.get("TCP-GL1 Write Bandwidth")

    metrics["sqc_gl1_read_bw"] = safe_float_sum(
        metrics["icache_gl1_bw"], metrics["dcache_gl1_bw"]
    )

    metrics["gl1c_util"] = metric_dict.get("GL1 Cache Utilization")
    metrics["gl1c_hit"] = metric_dict.get("GL1 Cache Hit Rate")
    metrics["gl1c_stall_gl2"] = metric_dict.get("GL1 Cache Stall GL2 Backpressure")

    metrics["gl1_gl2_read_bw"] = metric_dict.get("GL1-GL2 Read Bandwidth")
    metrics["gl1_gl2_write_bw"] = metric_dict.get("GL1-GL2 Write Bandwidth")

    metrics["gl2c_util"] = metric_dict.get("GL2 Cache Utilization")
    metrics["gl2c_hit"] = metric_dict.get("GL2 Cache Hit Rate")
    metrics["gl2_fabric_read_bw"] = metric_dict.get("GL2-Fabric Read BW")
    metrics["gl2_fabric_write_bw"] = metric_dict.get("GL2-Fabric Write BW")

    metrics["sysarb_util"] = metric_dict.get("SysArb Utilization")
    metrics["sysarb_stall"] = metric_dict.get("SysArb Stall Rate")
    metrics["dram_read_bw"] = metric_dict.get("DRAM Read Bandwidth")
    metrics["dram_write_bw"] = metric_dict.get("DRAM Write Bandwidth")

    metrics["total_bw"] = safe_float_sum(
        metrics["dram_read_bw"], metrics["dram_write_bw"]
    )

    return metrics


def _build_kernel_and_l0(
    metrics: dict[str, Any],
    kernel_arrows: dict[str, str],
    std_arrows: dict[str, str],
) -> tuple[Panel, Text, Group, Text]:
    """Kernel panel, kernel edges, L0 stack, and GL1 edges."""

    color_read = COLORS["read"]
    color_write = COLORS["write"]

    kernel_panel = build_kernel_panel(height=_TOTAL_H, padding_lines=11)

    kernel_arrow_left = kernel_arrows["left"]
    kernel_arrow_right = kernel_arrows["right"]

    # LDS edge: instruction count only (not data movement)
    lds_edge_zone = [
        "     [white]Request[/white]",
        format_edge("LDS", metrics["lds_req"], _EDGE_LABEL_W),
        kernel_arrows["plain"],
    ]
    tcp_edge_zone = [
        colored(
            format_edge("Read", metrics["tcp_read_req"], _EDGE_LABEL_W),
            color_read,
        ),
        colored(kernel_arrow_left, color_read),
        colored(
            format_edge("Write", metrics["tcp_write_req"], _EDGE_LABEL_W),
            color_write,
        ),
        colored(kernel_arrow_right, color_write),
    ]
    sqc_edge_zone = [
        colored(
            format_edge("ICache Read", metrics["icache_req"], _EDGE_LABEL_W),
            color_read,
        ),
        colored(kernel_arrow_left, color_read),
        colored(
            format_edge("DCache Read", metrics["dcache_req"], _EDGE_LABEL_W),
            color_read,
        ),
        colored(kernel_arrow_left, color_read),
    ]
    kernel_edges_lines = (
        pad_to(lds_edge_zone, _L0_PANEL_H)
        + pad_to(tcp_edge_zone, _L0_PANEL_H)
        + pad_to(sqc_edge_zone, _L0_PANEL_H)
    )
    kernel_edges_text = Text.from_markup("\n".join(kernel_edges_lines))

    # LDS panel
    lds_util_line = (
        f"{metric_line('Util', metrics['lds_util'], '%', COLORS['util'])}\n"
        f"[dim]{progress_bar(metrics['lds_util'])}[/dim]"
        if metrics["lds_util"] is not None
        else ""
    )
    lds_bw_line = (
        metric_line("BW", metrics["lds_bw"], "Bytes/s", COLORS["bw"])
        if metrics["lds_bw"] is not None
        else ""
    )
    lds_conflict_line = (
        metric_line("Bank Conflict", metrics["lds_bank_conflict"], "%", COLORS["stall"])
        if metrics["lds_bank_conflict"] is not None
        else ""
    )
    lds_panel = build_ip_block(
        "LDS",
        _L0_PANEL_W,
        _L0_PANEL_H,
        stack_metrics(lds_util_line, lds_bw_line, lds_conflict_line),
    )

    # GL0 (TCP Cache) panel
    tcp_bw_line = (
        metric_line("BW", metrics["tcp_bw"], "Bytes/s", COLORS["bw"])
        if metrics["tcp_bw"] is not None
        else ""
    )
    tcp_panel = build_ip_block(
        "GL0 (TCP Cache)",
        _L0_PANEL_W,
        _L0_PANEL_H,
        stack_metrics(
            f"{metric_line('Hit Rate', metrics['tcp_hit'], '%', COLORS['hit'])}\n"
            f"[dim]{progress_bar(metrics['tcp_hit'])}[/dim]",
            tcp_bw_line,
        ),
    )

    # SQC panel
    sqc_panel = build_ip_block(
        "SQC",
        _L0_PANEL_W,
        _L0_PANEL_H,
        stack_metrics(
            metric_line("ICache Hit Rate", metrics["icache_hit"], "%", COLORS["hit"])
            + f"\n[dim]{progress_bar(metrics['icache_hit'])}[/dim]",
            metric_line("DCache Hit Rate", metrics["dcache_hit"], "%", COLORS["hit"])
            + f"\n[dim]{progress_bar(metrics['dcache_hit'])}[/dim]",
        ),
    )

    # Stack LDS, TCP, SQC vertically
    l0_stack = Group(lds_panel, tcp_panel, sqc_panel)

    # GL1 edges: TCP and SQC connect; LDS does NOT
    tcp_gl1_content = stack_metrics(
        "\n".join([
            colored("Read BW", color_read),
            colored(format_value(metrics["tcp_gl1_read_bw"], "Bytes/s", 1), color_read),
            colored(std_arrows["left"], color_read),
        ]),
        "\n".join([
            colored("Write BW", color_write),
            colored(
                format_value(metrics["tcp_gl1_write_bw"], "Bytes/s", 1), color_write
            ),
            colored(std_arrows["right"], color_write),
        ]),
    )
    tcp_gl1_zone = tcp_gl1_content.split("\n")
    sqc_gl1_zone = [
        colored("Read BW", color_read),
        colored(format_value(metrics["sqc_gl1_read_bw"], "Bytes/s", 1), color_read),
        colored(std_arrows["left"], color_read),
    ]
    gl1_edges_lines = (
        pad_to([], _L0_PANEL_H)
        + pad_to(tcp_gl1_zone, _L0_PANEL_H)
        + pad_to(sqc_gl1_zone, _L0_PANEL_H)
    )
    gl1_edges_text = Text.from_markup("\n".join(gl1_edges_lines))

    return kernel_panel, kernel_edges_text, l0_stack, gl1_edges_text


def _build_cache_columns(
    metrics: dict[str, Any],
    std_arrows: dict[str, str],
) -> tuple[Panel, Text, Panel]:
    """GL1 panel, GL1-GL2 edges, GL2 panel."""

    gl1_content = stack_metrics(
        f"{metric_line('Util', metrics['gl1c_util'], '%', COLORS['util'])}\n"
        f"[dim]{progress_bar(metrics['gl1c_util'])}[/dim]",
        f"{metric_line('Hit Rate', metrics['gl1c_hit'], '%', COLORS['hit'])}\n"
        f"[dim]{progress_bar(metrics['gl1c_hit'])}[/dim]",
        metric_line("GL2 Stall", metrics["gl1c_stall_gl2"], "%", COLORS["stall"]),
    )
    gl1_panel = build_ip_block("GL1 Cache", _CACHE_PANEL_W, _TOTAL_H, gl1_content)

    gl1_gl2_edges_text = build_bw_edge_column(
        [
            (
                "Read BW",
                format_value(metrics["gl1_gl2_read_bw"], "Bytes/s", 1),
                "left",
                COLORS["read"],
            ),
            (
                "Write BW",
                format_value(metrics["gl1_gl2_write_bw"], "Bytes/s", 1),
                "right",
                COLORS["write"],
            ),
        ],
        std_arrows,
    )

    gl2_content = stack_metrics(
        f"{metric_line('Util', metrics['gl2c_util'], '%', COLORS['util'])}\n"
        f"[dim]{progress_bar(metrics['gl2c_util'])}[/dim]",
        f"{metric_line('Hit Rate', metrics['gl2c_hit'], '%', COLORS['hit'])}\n"
        f"[dim]{progress_bar(metrics['gl2c_hit'])}[/dim]",
    )
    gl2_panel = build_ip_block("GL2 Cache", _CACHE_PANEL_W, _TOTAL_H, gl2_content)

    return gl1_panel, gl1_gl2_edges_text, gl2_panel


def _build_memory_columns(
    metrics: dict[str, Any],
    std_arrows: dict[str, str],
) -> tuple[Text, Panel, Text, Panel]:
    """GL2-GCEA edges, GCEA panel, DRAM edges, DRAM panel."""

    color_read = COLORS["read"]
    color_write = COLORS["write"]

    gl2_gcea_edges_text = build_bw_edge_column(
        [
            (
                "Read BW",
                format_value(metrics["gl2_fabric_read_bw"], "Bytes/s", 1),
                "left",
                color_read,
            ),
            (
                "Write BW",
                format_value(metrics["gl2_fabric_write_bw"], "Bytes/s", 1),
                "right",
                color_write,
            ),
        ],
        std_arrows,
    )

    gcea_content = stack_metrics(
        f"{metric_line('SysArb Util', metrics['sysarb_util'], '%', COLORS['util'])}\n"
        f"[dim]{progress_bar(metrics['sysarb_util'])}[/dim]",
        metric_line("SysArb Stall", metrics["sysarb_stall"], "%", COLORS["stall"]),
    )
    gcea_panel = build_ip_block("GCEA", _CACHE_PANEL_W, _TOTAL_H, gcea_content)

    dram_edges_text = build_bw_edge_column(
        [
            (
                "Read BW",
                format_value(metrics["dram_read_bw"], "Bytes/s", 1),
                "left",
                color_read,
            ),
            (
                "Write BW",
                format_value(metrics["dram_write_bw"], "Bytes/s", 1),
                "right",
                color_write,
            ),
        ],
        std_arrows,
    )

    total = format_value(metrics["total_bw"], "Bytes/s", 1)
    dram_content = stack_metrics(
        "[dim]DDR5/LPDDR5[/dim]",
        f"Total: [bold bright_green]{total}[/bold bright_green]",
    )
    dram_panel = build_ip_block("DRAM", _CACHE_PANEL_W, _TOTAL_H, dram_content)

    return gl2_gcea_edges_text, gcea_panel, dram_edges_text, dram_panel


# ---------------------------------------------------------------------------
# Main diagram assembly
# ---------------------------------------------------------------------------


def create_mem_chart_diagram(
    metric_dict: dict[str, Any],
    console: Console,
    show_debug: bool = False,
    chart_title: str = "",
    gpu_arch: Optional[str] = None,
) -> None:
    """Render the RDNA3.5 memory diagram to *console*."""
    metrics = _extract_metrics(metric_dict)

    std_arrows = make_arrows(_STD_ARROW_LEN)
    kernel_arrows = make_arrows(_KERNEL_EDGE_W)

    # Build layout columns
    kernel_panel, kernel_edges, l0_stack, gl1_edges = _build_kernel_and_l0(
        metrics, kernel_arrows, std_arrows
    )
    gl1_panel, gl1_gl2_edges, gl2_panel = _build_cache_columns(metrics, std_arrows)
    gl2_gcea_edges, gcea_panel, dram_edges, dram_panel = _build_memory_columns(
        metrics, std_arrows
    )

    # Assemble 11-column grid (edge columns use vertical middle alignment)
    main_layout = Table.grid(padding=0)
    for i in range(11):
        vert: VerticalAlignMethod = "middle" if i in _EDGE_COLS else "top"
        main_layout.add_column(vertical=vert)

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

    total = console.measure(main_layout).maximum
    sysmem_span = (
        console.measure(dram_edges).maximum + console.measure(dram_panel).maximum
    )

    sections: list[Union[str, RenderableType]] = []
    if chart_title:
        sections.append(f"[bold]{chart_title}[/bold]")
    sections.append(_scope_bar(total - sysmem_span, sysmem_span))
    sections.append("")
    sections.append(main_layout)
    sections.append("")
    sections.append(build_legend(include_atomic=False, include_stall=True))

    if show_debug:
        sections.append("")
        sections.append(
            build_arch_notes([
                ("TCP (Texture Cache Pipe)", "L0 vector cache for VMEM operations"),
                ("LDS (Local Data Share)", "On-CU scratchpad, NO GL1 Cache connection"),
                ("SQC (Sequencer Cache)", "ICache + DCache for scalar operations"),
            ])
        )

    console.print(Group(*sections))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def plot_mem_chart(
    metric_dict: dict[str, Any],
    *,
    chart_title: str,
    gpu_arch: Optional[str] = None,
) -> str:
    """Render the RDNA3.5 memory chart and return as a string."""
    return render_chart_to_string(
        create_mem_chart_diagram,
        metric_dict,
        normalize_mem_chart_metrics,
        console_width=_CONSOLE_WIDTH,
        chart_title=chart_title,
        gpu_arch=gpu_arch,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for the RDNA3.5 memory chart."""
    mem_chart_cli_main(
        "RDNA3.5 Memory Chart - CLI Visualization",
        create_mem_chart_diagram,
        normalize_mem_chart_metrics,
        DEFAULT_SAMPLE_METRICS,
        console_width=_CONSOLE_WIDTH,
    )


if __name__ == "__main__":
    main()

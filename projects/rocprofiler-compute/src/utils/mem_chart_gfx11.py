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
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

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


@dataclass
class RectBlock:
    label: str
    x_min: int = 0
    x_max: int = 0
    y_min: int = 1
    y_max: int = 1

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min


@dataclass
class Edge:
    label: str
    arrow: str
    y_offset: int = 0
    color: str = "dim"


@dataclass
class AlignedEdgesGroup(RectBlock):
    edges: list[Edge] = field(default_factory=list)
    top_padding: int = 0
    compact: bool = False

    def render_text(self) -> Text:
        lines = []
        for _ in range(self.top_padding):
            lines.append("")
        for edge in self.edges:
            lines.append(f"[{edge.color}]{edge.label}[/{edge.color}]")
            lines.append(f"[{edge.color}]{edge.arrow}[/{edge.color}]")
            lines.append("")
        return Text.from_markup("\n".join(lines))


@dataclass
class SubBlock:
    label: str
    attributes: list[str] = field(default_factory=list)
    y_offset: int = 0
    height: int = 5
    show_border: bool = True
    vertical_position: str = "middle"
    border_color: str = "blue"


@dataclass
class RegularBlock(RectBlock):
    sub_blocks: list[SubBlock] = field(default_factory=list)
    content_text: str = ""
    vertical_position: str = "middle"
    color: str = "blue"

    def render(self) -> Panel:
        import re

        temp_content = []

        if self.content_text:
            temp_content.append(f"[dim]{self.content_text}[/dim]")
            temp_content.append("")

        for i, sub in enumerate(self.sub_blocks):
            if sub.show_border and sub.label:
                box_width = self.width - 6
                inner_width = box_width - 4
                bc = sub.border_color
                top_line = f"[{bc}]┌" + "─" * (box_width - 2) + f"┐[/{bc}]"
                bottom_line = f"[{bc}]└" + "─" * (box_width - 2) + f"┘[/{bc}]"
                temp_content.append(top_line)
                label_clean = sub.label
                label_pad = " " * max(0, inner_width - len(label_clean))
                temp_content.append(
                    f"[{bc}]│[/{bc}] [bold]{sub.label}[/bold]{label_pad} [{bc}]│[/{bc}]"
                )
                for attr in sub.attributes:
                    if not attr:
                        continue
                    clean = re.sub(r"\[.*?\]", "", attr)
                    pad_len = max(0, inner_width - len(clean))
                    pad = " " * pad_len
                    temp_content.append(f"[{bc}]│[/{bc}] {attr}{pad} [{bc}]│[/{bc}]")
                temp_content.append(bottom_line)
            else:
                if sub.label:
                    temp_content.append(f"[bold]{sub.label}[/bold]")
                for attr in sub.attributes:
                    temp_content.append(attr)
            if i < len(self.sub_blocks) - 1:
                temp_content.append("")

        content = "\n".join(temp_content)
        return Panel(
            content,
            title=f"[bold {self.color}]{self.label}[/bold {self.color}]",
            border_style=self.color,
            width=self.width,
            height=self.height,
        )


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
        # Handle both legacy GB/s and new Bytes/s units
        return format_bw_human_readable(value, unit)
    else:
        return f"{value:.{precision}f}{unit}"


def format_bw_human_readable(
    value: Union[int, float, str, None], unit: str = "Bytes/s", precision: int = 1
) -> str:
    """
    Format bandwidth value to human-readable format (TB/s, GB/s, MB/s, KB/s).

    Args:
        value: Bandwidth value (in Bytes/s if unit='Bytes/s', or in GB/s if unit='GB/s')
        unit: Input unit - 'Bytes/s' or 'GB/s' (legacy)
        precision: Number of decimal places

    Returns:
        Human-readable bandwidth string with appropriate unit
    """
    if value is None:
        return "N/A"
    try:
        value = float(value)
    except (ValueError, TypeError):
        return "N/A"

    # Convert to Bytes/s first if needed
    if unit == "GB/s":
        bytes_per_sec = value * 1e9
    else:
        bytes_per_sec = value

    # Convert to appropriate unit
    if bytes_per_sec >= 1e12:
        return f"{bytes_per_sec / 1e12:.{precision}f} TB/s"
    elif bytes_per_sec >= 1e9:
        return f"{bytes_per_sec / 1e9:.{precision}f} GB/s"
    elif bytes_per_sec >= 1e6:
        return f"{bytes_per_sec / 1e6:.{precision}f} MB/s"
    elif bytes_per_sec >= 1e3:
        return f"{bytes_per_sec / 1e3:.{precision}f} KB/s"
    else:
        return f"{bytes_per_sec:.{precision}f} B/s"


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


def format_bw_gbps(value: Union[int, float, str, None], precision: int = 1) -> str:
    """
    Format bandwidth (Bytes/s) to TB/s, GB/s, MB/s, or KB/s.

    This function expects value in Bytes/s (raw bytes per second).

    Args:
        value: Bandwidth value in Bytes/s
        precision: Number of decimal places

    Returns:
        Human-readable bandwidth string with appropriate unit
    """
    if value is None:
        return "N/A"
    try:
        value = float(value)
    except (ValueError, TypeError):
        return "N/A"

    # value is in Bytes/s, convert to appropriate unit
    if value >= 1e12:
        return f"{value / 1e12:.{precision}f} TB/s"
    elif value >= 1e9:
        return f"{value / 1e9:.{precision}f} GB/s"
    elif value >= 1e6:
        return f"{value / 1e6:.{precision}f} MB/s"
    elif value >= 1e3:
        return f"{value / 1e3:.{precision}f} KB/s"
    else:
        return f"{value:.{precision}f} B/s"


def get_metric(d: dict[str, Any], key: str, default: Any = None) -> Any:  # noqa: ANN401
    return d.get(key, default)


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


def create_mem_chart_diagram(
    arch: str,
    normal_unit: str,
    metric_dict: dict[str, Any],
    console: Console,
    show_debug: bool = False,
    compact: bool = False,
    chart_title: str = "",
) -> None:
    """Create the RDNA3.5 memory diagram (TCP, LDS, SQC blocks).

    ``chart_title``: printed once above the diagram (e.g. from YAML panel title +
    normalization unit).
    """

    # Extract metrics
    icache_req = get_metric(metric_dict, "ICache Requests")
    icache_hit = get_metric(metric_dict, "ICache Hit Rate")
    icache_gl1_bw = get_metric(metric_dict, "ICache-GL1 Read Bandwidth")

    dcache_req = get_metric(metric_dict, "Dcache Requests")
    dcache_hit = get_metric(metric_dict, "Dcache Hit Rate")
    dcache_gl1_bw = get_metric(metric_dict, "Dcache-GL1 Read Bandwidth")

    tcp_read_req = get_metric(metric_dict, "TCP Read Requests")
    tcp_write_req = get_metric(metric_dict, "TCP Write Requests")
    tcp_hit = get_metric(metric_dict, "TCP Hit Rate")
    tcp_bw = get_metric(metric_dict, "TCP Request Bandwidth")

    lds_insts = get_metric(metric_dict, "LDS Instructions")
    lds_inst_cycles = get_metric(metric_dict, "LDS Instruction Cycles")
    lds_atomic_insts = get_metric(metric_dict, "LDS Atomic Instructions")
    lds_bw = get_metric(metric_dict, "LDS Estimated Bandwidth")
    lds_bank_conflict = get_metric(metric_dict, "LDS Bank Conflict Rate")

    tcp_gl1_read_bw = get_metric(metric_dict, "TCP-GL1 Read Bandwidth")
    tcp_gl1_write_bw = get_metric(metric_dict, "TCP-GL1 Write Bandwidth")

    # Calculate combined SQC-GL1 bandwidth
    sqc_gl1_read_bw = None
    if icache_gl1_bw is not None and dcache_gl1_bw is not None:
        try:
            sqc_gl1_read_bw = float(icache_gl1_bw) + float(dcache_gl1_bw)
        except (ValueError, TypeError):
            sqc_gl1_read_bw = None
    elif icache_gl1_bw is not None:
        try:
            sqc_gl1_read_bw = float(icache_gl1_bw)
        except (ValueError, TypeError):
            sqc_gl1_read_bw = None
    elif dcache_gl1_bw is not None:
        try:
            sqc_gl1_read_bw = float(dcache_gl1_bw)
        except (ValueError, TypeError):
            sqc_gl1_read_bw = None

    gl1c_util = get_metric(metric_dict, "GL1C Utilization")
    gl1c_hit = get_metric(metric_dict, "GL1C Hit Rate")
    gl1c_stall_gl2 = get_metric(metric_dict, "GL1C Stall GL2 Backpressure")

    gl1_gl2_read_bw = get_metric(metric_dict, "GL1-GL2 Read Bandwidth")
    gl1_gl2_write_bw = get_metric(metric_dict, "GL1-GL2 Write Bandwidth")

    gl2c_util = get_metric(metric_dict, "GL2C Utilization")
    gl2c_hit = get_metric(metric_dict, "GL2C Hit Rate")
    gl2c_read_bw = get_metric(metric_dict, "GL2C Read Bandwidth")
    gl2c_write_bw = get_metric(metric_dict, "GL2C Write Bandwidth")

    sarb_util = get_metric(metric_dict, "SARB Utilization")
    sarb_stall = get_metric(metric_dict, "SARB Stall Rate")
    dram_read_bw = get_metric(metric_dict, "DRAM Read Bandwidth", 0)
    dram_write_bw = get_metric(metric_dict, "DRAM Write Bandwidth", 0)

    total_bw = (dram_read_bw or 0) + (dram_write_bw or 0)

    console.print()
    if chart_title:
        console.print(f"[bold]{chart_title}[/bold]")
    _print_mem_chart_scope_bar(console)
    console.print()

    # Arrow constants
    std_arrow_len = 8
    std_arrow_left = "<" + "-" * std_arrow_len
    std_arrow_right = "-" * std_arrow_len + ">"
    "<" + "-" * (std_arrow_len - 1) + ">"

    kernel_edge_width = 16
    kernel_arrow_left = "<" + "-" * (kernel_edge_width - 1)
    kernel_arrow_right = "-" * (kernel_edge_width - 1) + ">"
    kernel_arrow_both = "<" + "-" * (kernel_edge_width - 2) + ">"

    def fmt_edge(
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

    # =========================================================================
    # Build the layout with separate TCP, LDS, SQC blocks
    # =========================================================================

    # Create main layout table
    main_layout = Table.grid(padding=0)
    main_layout.add_column()  # Kernel
    main_layout.add_column()  # Kernel edges
    main_layout.add_column()  # TCP/LDS/SQC stacked
    main_layout.add_column()  # Edges to GL1
    main_layout.add_column()  # GL1C
    main_layout.add_column()  # GL1-GL2 edges
    main_layout.add_column()  # GL2C
    main_layout.add_column()  # GL2–GCEA (Graphics Core Efficiency Arbiter) edges
    main_layout.add_column()  # GCEA
    main_layout.add_column()  # GCEA–DRAM edges
    main_layout.add_column()  # DRAM

    # Kernel panel - height matches total of TCP+LDS+SQC stack (10+10+10=30)
    kernel_panel = Panel(
        "\n" * 11 + "[dim]Shader Core[/dim]\n[dim]Wave Execution[/dim]",
        title=f"[bold {COLORS['kernel']}]Kernel[/bold {COLORS['kernel']}]",
        border_style=COLORS["kernel"],
        width=14,
        height=30,
    )

    # Kernel edges (showing all operations grouped by destination)
    # Order: LDS, TCP, SQC (swapped LDS and TCP)
    kernel_edges_lines = [
        "",
        "     [white]Request[/white]",  # Label between Kernel and LDS (5 spaces)
        "",
        f"[{COLORS['read']}]{fmt_edge('Read', lds_insts)}[/{COLORS['read']}]",
        f"[{COLORS['read']}]{kernel_arrow_left}[/{COLORS['read']}]",
        f"[{COLORS['write']}]{fmt_edge('Write', lds_inst_cycles)}[/{COLORS['write']}]",
        f"[{COLORS['write']}]{kernel_arrow_right}[/{COLORS['write']}]",
        (
            f"[{COLORS['atomic']}]"
            f"{fmt_edge('Atomic', lds_atomic_insts)}"
            f"[/{COLORS['atomic']}]"
        ),
        f"[{COLORS['atomic']}]{kernel_arrow_both}[/{COLORS['atomic']}]",
        "",
        "",  # 2 empty lines before TCP edges
        "",
        "",
        f"[{COLORS['read']}]{fmt_edge('Read', tcp_read_req)}[/{COLORS['read']}]",
        f"[{COLORS['read']}]{kernel_arrow_left}[/{COLORS['read']}]",
        f"[{COLORS['write']}]{fmt_edge('Write', tcp_write_req)}[/{COLORS['write']}]",
        f"[{COLORS['write']}]{kernel_arrow_right}[/{COLORS['write']}]",
        "",
        "",
        "",
        "",
        "",
        "",
        "",  # 2 more empty lines before SQC edges
        "",
        f"[{COLORS['read']}]{fmt_edge('ICache', icache_req)}[/{COLORS['read']}]",
        f"[{COLORS['read']}]{kernel_arrow_left}[/{COLORS['read']}]",
        f"[{COLORS['read']}]{fmt_edge('DCache', dcache_req)}[/{COLORS['read']}]",
        f"[{COLORS['read']}]{kernel_arrow_left}[/{COLORS['read']}]",
    ]
    kernel_edges_text = Text.from_markup("\n".join(kernel_edges_lines))

    # Create stacked TCP/LDS/SQC panels - each height=10, total=30
    tcp_panel = Panel(
        f"{metric_line('Hit Rate', tcp_hit, '%', COLORS['hit'])}\n"
        f"{metric_line('BW', tcp_bw, 'Bytes/s', COLORS['bw']) if tcp_bw else ''}",
        title=f"[bold {COLORS['block']}]TCP (L0)[/bold {COLORS['block']}]",
        border_style=COLORS["block"],
        width=20,
        height=10,
    )

    lds_bw_line = metric_line("BW", lds_bw, "Bytes/s", COLORS["bw"]) if lds_bw else ""
    lds_conflict_line = (
        metric_line("Bank Conflict", lds_bank_conflict, "%", COLORS["stall"])
        if lds_bank_conflict
        else ""
    )
    lds_panel = Panel(
        f"{lds_bw_line}\n{lds_conflict_line}",
        title=f"[bold {COLORS['block']}]LDS[/bold {COLORS['block']}]",
        border_style=COLORS["block"],
        width=20,
        height=10,
    )

    sqc_panel = Panel(
        f"{metric_line('ICache', icache_hit, '%', COLORS['hit'])}\n"
        f"{metric_line('DCache', dcache_hit, '%', COLORS['hit'])}",
        title=f"[bold {COLORS['block']}]SQC[/bold {COLORS['block']}]",
        border_style=COLORS["block"],
        width=20,
        height=10,
    )

    # Stack LDS, TCP, SQC vertically (swapped LDS and TCP)
    l0_stack = Table.grid(padding=0)
    l0_stack.add_column()
    l0_stack.add_row(lds_panel)
    l0_stack.add_row(tcp_panel)
    l0_stack.add_row(sqc_panel)

    # Edges to GL1C (TCP and SQC connect, LDS does NOT)
    # Order: LDS (no connection), TCP, SQC (swapped LDS and TCP)
    gl1_edges_lines = [
        "",
        "",  # Empty space for LDS block (no GL1 connection)
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        f"[{COLORS['read']}]Read BW[/{COLORS['read']}]",
        f"[{COLORS['read']}]{format_bw_gbps(tcp_gl1_read_bw)}[/{COLORS['read']}]",
        f"[{COLORS['read']}]{std_arrow_left}[/{COLORS['read']}]",
        "",
        f"[{COLORS['write']}]Write BW[/{COLORS['write']}]",
        f"[{COLORS['write']}]{format_bw_gbps(tcp_gl1_write_bw)}[/{COLORS['write']}]",
        f"[{COLORS['write']}]{std_arrow_right}[/{COLORS['write']}]",
        "",
        "",
        "",
        f"[{COLORS['read']}]Read BW[/{COLORS['read']}]",
        f"[{COLORS['read']}]{format_bw_gbps(sqc_gl1_read_bw)}[/{COLORS['read']}]",
        f"[{COLORS['read']}]{std_arrow_left}[/{COLORS['read']}]",
    ]
    gl1_edges_text = Text.from_markup("\n".join(gl1_edges_lines))

    # GL1C panel - height=30 to match stack
    gl1_panel = Panel(
        f"{metric_line('Util', gl1c_util, '%', COLORS['util'])}\n"
        f"[dim]{bar(gl1c_util)}[/dim]\n"
        "\n"
        f"{metric_line('Hit Rate', gl1c_hit, '%', COLORS['hit'])}\n"
        f"[dim]{bar(gl1c_hit)}[/dim]\n"
        "\n"
        f"{metric_line('GL2 Stall', gl1c_stall_gl2, '%', COLORS['stall'])}",
        title=f"[bold {COLORS['block']}]GL1C[/bold {COLORS['block']}]",
        border_style=COLORS["block"],
        width=16,
        height=30,
    )

    # GL1-GL2 edges - more padding to center vertically
    gl1_gl2_edges_lines = [
        "",
        "",
        "",
        "",
        "",
        "",
        "",  # 5 more empty lines
        "",
        "",
        "",
        "",
        f"[{COLORS['read']}]Read BW[/{COLORS['read']}]",
        f"[{COLORS['read']}]{format_bw_gbps(gl1_gl2_read_bw)}[/{COLORS['read']}]",
        f"[{COLORS['read']}]{std_arrow_left}[/{COLORS['read']}]",
        "",
        f"[{COLORS['write']}]Write BW[/{COLORS['write']}]",
        f"[{COLORS['write']}]{format_bw_gbps(gl1_gl2_write_bw)}[/{COLORS['write']}]",
        f"[{COLORS['write']}]{std_arrow_right}[/{COLORS['write']}]",
        "",
        "",
        "",
        "",
    ]
    gl1_gl2_edges_text = Text.from_markup("\n".join(gl1_gl2_edges_lines))

    # GL2C panel - height=30 to match
    gl2_panel = Panel(
        f"{metric_line('Util', gl2c_util, '%', COLORS['util'])}\n"
        f"[dim]{bar(gl2c_util)}[/dim]\n"
        "\n"
        f"{metric_line('Hit Rate', gl2c_hit, '%', COLORS['hit'])}\n"
        f"[dim]{bar(gl2c_hit)}[/dim]",
        title=f"[bold {COLORS['block']}]GL2C[/bold {COLORS['block']}]",
        border_style=COLORS["block"],
        width=16,
        height=30,
    )

    # GL2–GCEA (Graphics Core Efficiency Arbiter) edges;
    # extra padding to center vertically
    gl2_gcea_edges_lines = [
        "",
        "",
        "",
        "",
        "",
        "",
        "",  # 5 more empty lines
        "",
        "",
        "",
        "",
        f"[{COLORS['read']}]Read BW[/{COLORS['read']}]",
        f"[{COLORS['read']}]{format_bw_gbps(gl2c_read_bw)}[/{COLORS['read']}]",
        f"[{COLORS['read']}]{std_arrow_left}[/{COLORS['read']}]",
        "",
        f"[{COLORS['write']}]Write BW[/{COLORS['write']}]",
        f"[{COLORS['write']}]{format_bw_gbps(gl2c_write_bw)}[/{COLORS['write']}]",
        f"[{COLORS['write']}]{std_arrow_right}[/{COLORS['write']}]",
        "",
        "",
        "",
        "",
    ]
    gl2_gcea_edges_text = Text.from_markup("\n".join(gl2_gcea_edges_lines))

    # GCEA (Graphics Core Efficiency Arbiter) panel — height=30 to match
    gcea_panel = Panel(
        f"{metric_line('SysArb Util', sarb_util, '%', COLORS['util'])}\n"
        f"[dim]{bar(sarb_util)}[/dim]\n"
        "\n"
        f"{metric_line('Stall', sarb_stall, '%', COLORS['stall'])}",
        title=f"[bold {COLORS['block']}]GCEA[/bold {COLORS['block']}]",
        border_style=COLORS["block"],
        width=16,
        height=30,
    )

    # GCEA–DRAM edges — more padding to center vertically
    dram_edges_lines = [
        "",
        "",
        "",
        "",
        "",
        "",
        "",  # 5 more empty lines
        "",
        "",
        "",
        "",
        f"[{COLORS['read']}]Read BW[/{COLORS['read']}]",
        f"[{COLORS['read']}]{format_bw_gbps(dram_read_bw)}[/{COLORS['read']}]",
        f"[{COLORS['read']}]{std_arrow_left}[/{COLORS['read']}]",
        "",
        f"[{COLORS['write']}]Write BW[/{COLORS['write']}]",
        f"[{COLORS['write']}]{format_bw_gbps(dram_write_bw)}[/{COLORS['write']}]",
        f"[{COLORS['write']}]{std_arrow_right}[/{COLORS['write']}]",
        "",
        "",
        "",
        "",
    ]
    dram_edges_text = Text.from_markup("\n".join(dram_edges_lines))

    # DRAM panel - height=30 to match
    dram_panel = Panel(
        f"[dim]DDR5/LPDDR5[/dim]\n"
        "\n"
        f"Total: [bold bright_green]{format_bw_gbps(total_bw)}[/bold bright_green]",
        title=f"[bold {COLORS['block']}]DRAM[/bold {COLORS['block']}]",
        border_style=COLORS["block"],
        width=16,
        height=30,
    )

    # Add the main row
    main_layout.add_row(
        kernel_panel,
        kernel_edges_text,
        l0_stack,
        gl1_edges_text,
        gl1_panel,
        gl1_gl2_edges_text,
        gl2_panel,
        gl2_gcea_edges_text,
        gcea_panel,
        dram_edges_text,
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

    class FakeFile:
        def __init__(self) -> None:
            self.data = []

        def write(self, s: str) -> None:
            self.data.append(s)

        def flush(self) -> None:
            pass

        def isatty(self) -> bool:
            return True

        def getvalue(self) -> str:
            return "".join(self.data)

    flat = normalize_mem_chart_metrics(metric_dict)
    resolved_heading = (
        format_mem_chart_heading(normal_unit, panel_id=300)
        if chart_title is None
        else chart_title
    )
    fake_file = FakeFile()
    console = Console(file=fake_file, force_terminal=True, width=200, height=80)
    create_mem_chart_diagram(
        arch,
        normal_unit,
        flat,
        console,
        show_debug=False,
        compact=False,
        chart_title=resolved_heading,
    )
    return fake_file.getvalue()


DEFAULT_SAMPLE_METRICS: dict[str, Union[int, float]] = dict(_MEM_CHART_DEFAULT_ROWS)


def get_sample_metrics() -> dict[str, Any]:
    """Return sample metrics (flat panel order) for testing or demos."""
    return normalize_mem_chart_metrics(DEFAULT_SAMPLE_METRICS.copy())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RDNA3.5 Memory Chart - CLI Visualization"
    )
    parser.add_argument("--data", "-d", help="JSON file with metrics data")
    parser.add_argument("--debug", action="store_true", help="Show debug info")
    parser.add_argument("--arch", default="gfx1151", help="Architecture name")
    parser.add_argument("--norm", default="per_kernel", help="Normalization unit")
    parser.add_argument("--txt", "-t", help="Output to plain text file")
    parser.add_argument("--svg", help="Output to SVG file")
    parser.add_argument("--compact", "-c", action="store_true", help="Compact mode")
    args = parser.parse_args()

    # Load or use default data
    if args.data:
        with open(args.data) as f:
            metric_dict = normalize_mem_chart_metrics(json.load(f))
    else:
        metric_dict = normalize_mem_chart_metrics(DEFAULT_SAMPLE_METRICS.copy())

    heading = format_mem_chart_heading(args.norm, panel_id=300)

    # Create console
    class FakeFile:
        def __init__(self) -> None:
            self.data = []

        def write(self, s: str) -> None:
            self.data.append(s)

        def flush(self) -> None:
            pass

        def isatty(self) -> bool:
            return True

        def getvalue(self) -> str:
            return "".join(self.data)

    if args.txt:
        fake_file = FakeFile()
        console = Console(
            file=fake_file, force_terminal=True, width=200, height=80, no_color=True
        )
        create_mem_chart_diagram(
            args.arch,
            args.norm,
            metric_dict,
            console,
            args.debug,
            args.compact,
            chart_title=heading,
        )
        import re

        output = fake_file.getvalue()
        plain = re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", output)
        with open(args.txt, "w") as f:
            f.write(plain)
        print(f"Output written to {args.txt}")
    elif args.svg:
        from io import StringIO

        svg_console = Console(
            file=StringIO(), force_terminal=True, width=200, height=80, record=True
        )
        create_mem_chart_diagram(
            args.arch,
            args.norm,
            metric_dict,
            svg_console,
            args.debug,
            args.compact,
            chart_title=heading,
        )
        svg_output = svg_console.export_svg(title=heading)
        with open(args.svg, "w") as f:
            f.write(svg_output)
        print(f"SVG saved to {args.svg}")
    else:
        fake_file = FakeFile()
        console = Console(file=fake_file, force_terminal=True, width=200, height=80)
        create_mem_chart_diagram(
            args.arch,
            args.norm,
            metric_dict,
            console,
            args.debug,
            args.compact,
            chart_title=heading,
        )
        print(fake_file.getvalue())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
gfx1250 memory architecture diagram -- CLI visualization
==================================================================

This script generates a Rich-based terminal visualization of the Instinct gfx1250
memory hierarchy, displaying performance metrics from counters defined in
0300_Memory_Chart.yaml.

USAGE:
    python mem_chart_gfx1250.py [--data metrics.json] [--debug]
        [--txt file.txt] [--svg file.svg]

API:
    plot_mem_chart(arch, normal_unit, metric_dict) -> str

DESIGN PRINCIPLES:
==================

1. COORDINATE SYSTEM
   - Origin (0, 0) is at TOP-LEFT corner
   - X-axis increases LEFT-TO-RIGHT (column position)
   - Y-axis increases TOP-TO-BOTTOM (row position)
   - Each block is defined by (x_min, x_max, y_min, y_max)

2. RELATIVE POSITIONING (Top-Down, Left-Right Dependencies)
   - Each component's position depends ONLY on its TOP and LEFT neighbors
   - This ensures predictable, cascading layout without circular dependencies

   Layout Order (Column 1 has TCP at top, SQC below):
   +-------------------------------------------------------------------------+
   |  Col 0      Col 1       Col 2     Col 3      Col 4      Col 5     Col 6 |
   | +------+   +-----+    +-----+   +------+   +-----+   +------+   +-----+ |
   | |Kernel|-->|TCP  |--> |GL1  |-->|GLARB |-->|GL2  |-->|EA/DF |-->|HBM  | |
   | |      |   | LDS |    |     |   |      |   |     |   |      |   |     | |
   | |      |   | GL0 |    |     |   |      |   |     |   |      |   |IO   | |
   | |      |   +-----+    |     |   |      |   |     |   |      |   |HDM  | |
   | |      |   |SQC  |    |     |   |      |   |     |   |      |   |GMI  | |
   | +------+   +-----+    +-----+   +------+   +-----+   +------+   +-----+ |
   +-------------------------------------------------------------------------+

   Positioning Rules (each block depends ONLY on TOP and LEFT neighbors):
   - Kernel:  x_min = 0 (leftmost), y_min = header_offset (topmost block)
   - Edges:   x_min = kernel.x_max + gap (right of kernel)
   - TCP:     x_min = edges.x_max + gap, y_min = kernel.y_min (aligned with kernel top)
   - SQC:     x_min = tcp.x_min, y_min = tcp.y_max + gap (below TCP, same column)
   - GL1:     x_min = tcp_edges.x_max + gap, y_min = tcp.y_min (aligned with TCP top)
   - GLARB:   x_min = gl1_edges.x_max + gap, y_min = gl1.y_min
   - GL2:     x_min = glarb_edges.x_max + gap, y_min = glarb.y_min
   - EA/DF:   x_min = gl2_edges.x_max + gap, y_min = gl2.y_min
   - HBM:     x_min = df_edges.x_max + gap, y_min = df.y_min
   - IO/HDM/GMI: x_min = hbm.x_max + gap, y calculated from df.y_max

3. COMPONENT HIERARCHY (Column 1: TCP at top, SQC below)
   +- Kernel (Shader Core Wave Execution)
   |   +-- Edges -------------------------------------+
   +- TCP (Texture/Cache Processor) - TOP of Col 1    |
   |   +-- LDS (Local Data Share, sub-block) - TOP    |
   |   +-- GL0 (Vector L0 Cache, sub-block) - BOTTOM  |
   +- SQC (Scalar Cache) - BELOW TCP                  |
   |   +-- Instruction Cache (sub-block)              |
   |   +-- Scalar Data Cache (sub-block)              |
   +- GL1 (Level 1 Graphics Cache)                    |
   +- GLARB (GL1-GL2 Arbiter)                         |
   +- GL2 (Level 2 Graphics Cache)                    |
   +- EA/DF (Efficiency Arbiter / Data Fabric)        |
   +- Downstream (HBM at top, IO/HDM/GMI below)

4. EDGE CONNECTIONS
   - Edges are rendered as separate "virtual blocks" between real blocks
   - Each edge group contains: label, value, arrow direction
   - Arrow styles: <---------- (left/read), ----------> (right/write),
     <---------> (bidirectional/atomic)

5. BLOCK TYPES
   - RectBlock:         Base class with coordinates
   - RegularBlock:      Rendered as bordered Panel with optional sub-blocks
   - SubBlock:          Nested block within RegularBlock (e.g., GL0 inside TCP)
   - AlignedEdgesGroup: Virtual block for edge rendering (no border)

6. Y-MAX ALIGNMENT
   - Key blocks share the same y_max to align bottom edges:
     kernel.y_max = tcp.y_max = gl1.y_max = glarb.y_max = gl2.y_max = ea_df.y_max

7. METRICS MAPPING
   - Metrics are extracted from 0300_Memory_Chart.yaml
   - Each metric is associated with specific counters (e.g., TX_VMW_GL1_REQ_READ)
   - Bandwidth formulas: Bytes / (End_Timestamp - Start_Timestamp) / 1e9 -> GB/s

ARCHITECTURE REGIONS:
=====================
   |<-------------- XCD (Compute Die) --------------->|<---- AID (I/O Die) ---->|

   Kernel -> SQC/TCP -> GL1 -> GLARB -> GL2 -> EA/DF -> HBM (DRAM)
                                                     -> IO  (PCIe)
                                                     -> HDM (CXL)
                                                     -> GMI (Multi-GPU)

DEPENDENCIES:
=============
   - Python 3.8+
   - rich (for terminal rendering)
   - Optional: weasyprint (for SVG export)
"""

import argparse
import json
import re
from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Optional, Union

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Keys = ``metric:`` names under each ``metric_table`` in
# ``analysis_configs/gfx1250/0300_Memory_Chart.yaml`` (tables 301-308), in panel order.
_MEM_CHART_DEFAULT_ROWS: tuple[tuple[str, Union[int, float]], ...] = (
    # Table 301: Instruction Cache
    ("ICache Requests", 450),
    ("ICache Utilization", 45.2),
    ("ICache Hit Rate", 98.5),
    ("ICache Request Stall Rate", 2.1),
    ("ICache-GL1 Read Bandwidth", 57.6e9),
    # Table 302: Scalar Data Cache
    ("Dcache Requests", 225),
    ("Dcache Utilization", 38.7),
    ("Dcache Hit Rate", 95.3),
    ("Dcache Request Stall Rate", 1.8),
    ("Dcache Request Bandwidth", 134.2e9),
    ("Dcache-GL1 Read Bandwidth", 28.8e9),
    # Table 303: GL0 Cache
    ("GL0 Utilization", 72.4),
    ("GL0 Total Requests", 1_250_000),
    ("GL0 Read Requests", 875_000),
    ("GL0 Write Requests", 375_000),
    ("GL0 Atomic Requests", 62_500),
    ("GL0 Hit Rate", 87.2),
    ("GL0 Request Latency", 42.5),
    # Table 304: LDS
    ("LDS Utilization", 65.8),
    ("LDS Utilization - Load", 42.3),
    ("LDS Utilization - Store", 23.5),
    ("LDS Bank Conflict Stall Rate", 5.2),
    ("LDS Address Conflict Stall Rate", 2.1),
    ("LDS Load Requests", 1_250),
    ("LDS Store Requests", 625),
    ("LDS Atomic Requests", 1_250),
    ("Async Load Requests", 250),
    ("Async Store Requests", 1_000),
    ("LDS Load Bandwidth", 268.7e9),
    ("LDS Store Bandwidth", 134.2e9),
    ("LDS Atomic Bandwidth", 16e9),
    ("Async Load Bandwidth", 50e9),
    ("Async Store Bandwidth", 25e9),
    ("TDM Load Bandwidth", 80e9),
    ("TDM Store Bandwidth", 40e9),
    ("TDM Load Requests", 5_000),
    ("TDM Store Requests", 2_500),
    # Table 305: GL0-GL1 Interface
    ("GL0-GL1 Read Requests", 125_000),
    ("GL0-GL1 Write Requests", 62_500),
    ("GL0-GL1 Read Bandwidth", 412.8e9),
    ("GL0-GL1 Write Bandwidth", 206.4e9),
    ("GL0-GL1 Atomic Bandwidth", 16e9),
    ("GL0-GL1 Read Latency", 85.2),
    ("GL0-GL1 Write Latency", 62.4),
    ("GL0-GL1 Request Stall Rate", 8.5),
    ("GL1A Utilization", 65.2),
    ("GL1C Utilization", 72.8),
    ("GL1C Buffer Full Stall", 12.5),
    # Table 306: GLARB
    ("GLARBA Utilization", 52.3),
    ("GLARBA Request Path Utilization", 48.7),
    ("GLARBA Return Path Utilization", 45.2),
    ("GLARBC Utilization", 58.6),
    ("GLARBC Buffer Full Stall", 12.4),
    ("GL1-GLARB Read Bandwidth", 412.8e9),
    ("GL1-GLARB Write Bandwidth", 156.3e9),
    ("GLARB-GL2 Read Bandwidth", 380.5e9),
    ("GLARB-GL2 Write Bandwidth", 142.8e9),
    ("GLARB-GL2 Atomic Bandwidth", 28.6e9),
    # Table 307: GL2 Cache
    ("GL2A Utilization", 74.2),
    ("GL2C Hit Rate", 82.5),
    ("GL2-EA Read Bandwidth", 485.6e9),
    ("GL2-EA Write Bandwidth", 362.4e9),
    ("GL2-EA Atomic Bandwidth", 25.2e9),
    # Table 308: EA to HBM/IO/HDM/GMI
    ("EA Utilization", 65.8),
    ("EA Stall Rate", 15.2),
    ("DRAM Read Bandwidth", 512e9),
    ("DRAM Write Bandwidth", 384e9),
    ("IO Read Bandwidth", 32e9),
    ("IO Write Bandwidth", 24e9),
    ("HDM Read Bandwidth", 128e9),
    ("HDM Write Bandwidth", 96e9),
    ("GMI Read Bandwidth", 256e9),
    ("GMI Write Bandwidth", 192e9),
)

MEM_CHART_PANEL_METRIC_KEYS: tuple[str, ...] = tuple(
    k for k, _ in _MEM_CHART_DEFAULT_ROWS
)

DEFAULT_SAMPLE_METRICS: dict[str, Union[int, float]] = dict(_MEM_CHART_DEFAULT_ROWS)

# ============================================================================
# Color Definitions
# ============================================================================
COLORS = {
    "kernel": "green",
    "block": "blue",
    "read": "bright_cyan",
    "write": "bright_yellow",
    "atomic": "bright_magenta",
    "util": "bright_green",
    "hit": "yellow",
    "stall": "indian_red",
    "bw": "bright_cyan",
}


# ============================================================================
# Block Classes (coordinate-based positioning)
# ============================================================================
@dataclass
class RectBlock:
    """Base rectangular block with coordinates"""

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
    """Single edge with label and arrow"""

    label: str
    arrow: str
    y_offset: int = 0
    color: str = "dim"


@dataclass
class AlignedEdgesGroup(RectBlock):
    """Virtual block containing aligned edges (not rendered as box)"""

    edges: list[Edge] = field(default_factory=list)
    top_padding: int = 0
    compact: bool = False  # If True, filter out edges with zero values

    def _is_nonzero_edge(self, edge: Edge) -> bool:
        """Check if an edge has a valid non-zero value.

        In compact mode, hide edges with 0, N/A, None, or no value.
        """
        label = edge.label

        # Check for N/A or None anywhere in the label
        if "N/A" in label or "None" in label:
            return False

        # Format 1: "Label   : value" (kernel edges)
        if ":" in label:
            parts = label.split(":")
            if len(parts) >= 2:
                value_str = parts[1].strip()
                if not value_str:
                    return False  # No value, hide in compact mode
                try:
                    value = float(value_str.replace("e+", "e").replace("e-", "e"))
                    return value != 0
                except ValueError:
                    return True  # Keep if can't parse (has non-numeric value)
            return False  # No value part after colon

        # Format 2: "Label\nvalue GB/s" or "Label\nvalue"
        elif "\n" in label:
            parts = label.split("\n")
            found_value = False
            for part in parts:
                part = part.strip()
                # Skip empty, arrow lines, and label-only lines
                if (
                    not part
                    or part.startswith("<")
                    or part.startswith("-")
                    or part.endswith(">")
                ):
                    continue
                # Try to extract numeric value
                value_str = (
                    part
                    .replace(" GB/s", "")
                    .replace(" MB/s", "")
                    .replace(" KB/s", "")
                    .strip()
                )
                try:
                    value = float(value_str)
                    found_value = True
                    if value != 0:
                        return True  # Keep non-zero
                except ValueError:
                    # This might be the label part, continue
                    continue
            # If we found a value and it was 0, hide it
            if found_value:
                return False
            # If we couldn't find any numeric value, hide it (no data)
            return False

        # Label-only edge (no value) - hide in compact mode
        return False

    def render_text(self) -> Text:
        lines = []
        for _ in range(self.top_padding):
            lines.append("")

        # Filter edges if compact mode is enabled
        edges_to_render = self.edges
        if self.compact:
            edges_to_render = [e for e in self.edges if self._is_nonzero_edge(e)]

        for edge in edges_to_render:
            lines.append(f"[{edge.color}]{edge.label}[/{edge.color}]")
            lines.append(f"[{edge.color}]{edge.arrow}[/{edge.color}]")
            lines.append("")
        return Text.from_markup("\n".join(lines))


@dataclass
class SubBlock:
    """Sub-block within a parent block"""

    label: str
    attributes: list[str] = field(default_factory=list)
    y_offset: int = 0
    height: int = 5
    show_border: bool = True
    vertical_position: str = "middle"
    border_color: str = "blue"  # Match parent block color


@dataclass
class RegularBlock(RectBlock):
    """Regular block that can contain sub-blocks"""

    sub_blocks: list[SubBlock] = field(default_factory=list)
    content_text: str = ""
    content_y_offset: int = 2  # Vertical offset for content_text from top
    vertical_position: str = "middle"
    color: str = "blue"

    def render(self) -> Panel:

        temp_content = []

        if self.content_text:
            temp_content.append(f"[dim]{self.content_text}[/dim]")
            temp_content.append("")

        for i, sub in enumerate(self.sub_blocks):
            if sub.show_border and sub.label:
                box_width = self.width - 6
                bc = sub.border_color  # Use sub-block's border color
                top_line = f"[{bc}]+" + "-" * (box_width - 2) + f"+[/{bc}]"
                bottom_line = f"[{bc}]+" + "-" * (box_width - 2) + f"+[/{bc}]"

                available_lines = sub.height - 2
                attr_lines_count = len(sub.attributes)
                padding_needed = max(0, available_lines - 1 - attr_lines_count)

                if sub.vertical_position == "top":
                    padded_content = (
                        [f"[bold]{sub.label}[/bold]"]
                        + sub.attributes
                        + [""] * padding_needed
                    )
                elif sub.vertical_position == "bottom":
                    padded_content = (
                        [f"[bold]{sub.label}[/bold]"]
                        + [""] * padding_needed
                        + sub.attributes
                    )
                else:
                    top_pad = padding_needed // 2
                    bottom_pad = padding_needed - top_pad
                    padded_content = (
                        [f"[bold]{sub.label}[/bold]"]
                        + [""] * top_pad
                        + sub.attributes
                        + [""] * bottom_pad
                    )

                temp_content.append(top_line)
                for line in padded_content:
                    clean_line = re.sub(r"\[.*?\]", "", line)
                    padding = " " * max(0, box_width - len(clean_line) - 3)
                    temp_content.append(f"[{bc}]|[/{bc}] {line}{padding}[{bc}]|[/{bc}]")
                temp_content.append(bottom_line)
            else:
                if sub.label:
                    temp_content.append(f"[bold]{sub.label}[/bold]")
                for attr in sub.attributes:
                    temp_content.append(attr)

            if i < len(self.sub_blocks) - 1:
                temp_content.append("")

        content_height = len(temp_content)
        available_height = self.height - 2

        if self.vertical_position == "top":
            padding_lines = max(0, available_height - content_height)
            content_lines = temp_content + [""] * padding_lines
        elif self.vertical_position == "bottom":
            padding_lines = max(0, available_height - content_height)
            content_lines = [""] * padding_lines + temp_content
        else:
            padding_lines = max(0, available_height - content_height)
            top_padding = padding_lines // 2
            bottom_padding = padding_lines - top_padding
            content_lines = [""] * top_padding + temp_content + [""] * bottom_padding

        content = "\n".join(content_lines)

        # Use Align to center content if content_text is set
        if self.content_text and not self.sub_blocks:
            from rich.align import Align

            aligned_content = Align.center(content)
        else:
            aligned_content = content

        return Panel(
            aligned_content,
            title=f"[bold {self.color}]{self.label}[/bold {self.color}]",
            border_style=self.color,
            width=self.width,
            height=self.height,
        )


# ============================================================================
# Helper Functions
# ============================================================================
def format_value(
    value: Union[int, float, str, None], unit: str = "", precision: int = 1
) -> str:
    """Format a metric value for display"""
    if value is None:
        return "N/A"
    # Try to convert string to float for formatting
    if isinstance(value, str):
        try:
            value = float(value)
        except (ValueError, TypeError):
            return value  # Return original string if conversion fails
    if unit == "%":
        return f"{value:.{precision}f}%"
    elif unit == "GB/s":
        return f"{value:.0f} GB/s"
    elif unit == "cycles":
        return f"{value:.{precision}f} cyc"
    else:
        return f"{value:.{precision}f}{unit}"


def format_sci(value: Union[int, float, str, None], precision: int = 2) -> str:
    """Format a large number in scientific notation (e.g., 8.75e5)"""
    if value is None:
        return "N/A"
    # Convert to float if string
    try:
        value = float(value)
    except (ValueError, TypeError):
        return "N/A"
    if abs(value) < 1000:
        return f"{int(value)}"
    return f"{value:.{precision}e}"


def format_bw_gbps(value: Union[int, float, str, None], precision: int = 1) -> str:
    """Format bandwidth in GB/s (input is in Bytes, divide by 1e9)"""
    if value is None:
        return "N/A"
    # Convert to float if string
    try:
        value = float(value)
    except (ValueError, TypeError):
        return "N/A"
    gbps = value / 1e9
    if gbps >= 1000:
        return f"{gbps / 1000:.{precision}f} TB/s"
    elif gbps >= 1:
        return f"{gbps:.{precision}f} GB/s"
    else:
        return f"{gbps * 1000:.{precision}f} MB/s"


def get_metric(
    metric_dict: dict[str, Union[int, float, str, None]],
    key: str,
    default: Union[int, float, str, None] = None,
) -> Union[int, float, str, None]:
    """Safely get a metric value from the dict"""
    return metric_dict.get(key, default)


def metric_line(
    label: str,
    value: Union[int, float, str, None],
    unit: str = "%",
    color: str = "bright_green",
) -> str:
    """Format a metric line with label and colored value"""
    formatted = format_value(value, unit)
    return f"{label} [{color}]{formatted}[/{color}]"


def bar(pct: float, w: int = 10) -> str:
    """Create a simple progress bar - default width 10 to fit block width"""
    if pct is None:
        return "." * w
    # Convert to float if string
    try:
        pct = float(pct)
    except (ValueError, TypeError):
        return "." * w
    filled = int(w * min(100, max(0, pct)) / 100)
    return "#" * filled + "." * (w - filled)


def bar_line(pct: float, color: str = "dim", width: int = 10) -> str:
    """Format a progress bar line - default width 10 to fit block width"""
    return f"[{color}]{bar(pct, width)}[/{color}]"


# ============================================================================
# Helper: Filter edges with non-zero values (for compact mode)
# ============================================================================
def filter_nonzero_edges(edges: list[Edge], compact: bool = False) -> list[Edge]:
    """Filter edges to only include those with non-zero values (compact mode)."""
    if not compact:
        return edges

    filtered = []
    for edge in edges:
        # Check if the label contains a value (e.g., "Load    : 1.25e+03")
        # Skip edges where the value part is 0, None, or not present
        label = edge.label
        if ":" in label:
            # Format: "Label   : value"
            parts = label.split(":")
            if len(parts) >= 2:
                value_str = parts[1].strip()
                try:
                    value = float(value_str.replace("e+", "e").replace("e-", "e"))
                    if value != 0:
                        filtered.append(edge)
                except ValueError:
                    # If we can't parse the value, keep the edge
                    filtered.append(edge)
        elif "\n" in label:
            # Format: "Label\nvalue\n"
            parts = label.split("\n")
            for part in parts:
                part = part.strip()
                if (
                    part
                    and not part.startswith("<")
                    and not part.startswith("-")
                    and not part.endswith(">")
                ):
                    try:
                        # Try to parse as number (may have GB/s suffix)
                        value_str = (
                            part.replace(" GB/s", "").replace(" MB/s", "").strip()
                        )
                        value = float(value_str)
                        if value != 0:
                            filtered.append(edge)
                            break
                    except ValueError:
                        # Keep non-numeric labels
                        filtered.append(edge)
                        break
        else:
            # No value in label, keep the edge
            filtered.append(edge)

    return filtered


# ============================================================================
# Main Diagram Creation
# ============================================================================
def create_mem_chart_diagram(
    arch: str,
    normal_unit: str,
    metric_dict: dict[str, Any],
    console: Console,
    show_debug: bool = False,
    compact: bool = False,
) -> None:
    """Create the memory architecture diagram using coordinate-based positioning"""

    # ========================================================================
    # Extract metrics from YAML dict (matching 0300_Memory_Chart.yaml names)
    # ========================================================================

    # Table 301: Instruction Cache
    icache_req = get_metric(metric_dict, "ICache Requests")
    get_metric(metric_dict, "ICache Utilization")
    icache_hit = get_metric(metric_dict, "ICache Hit Rate")
    get_metric(metric_dict, "ICache Request Stall Rate")
    icache_gl1_bw = get_metric(metric_dict, "ICache-GL1 Read Bandwidth")

    # Table 302: Scalar Data Cache
    dcache_req = get_metric(metric_dict, "Dcache Requests")
    get_metric(metric_dict, "Dcache Utilization")
    dcache_hit = get_metric(metric_dict, "Dcache Hit Rate")
    get_metric(metric_dict, "Dcache Request Stall Rate")
    get_metric(metric_dict, "Dcache Request Bandwidth")
    dcache_gl1_bw = get_metric(metric_dict, "Dcache-GL1 Read Bandwidth")

    # Table 303: GL0 Cache
    gl0_util = get_metric(metric_dict, "GL0 Utilization")
    get_metric(metric_dict, "GL0 Total Requests")
    gl0_read_req = get_metric(metric_dict, "GL0 Read Requests")
    gl0_write_req = get_metric(metric_dict, "GL0 Write Requests")
    gl0_atomic_req = get_metric(metric_dict, "GL0 Atomic Requests")
    gl0_hit = get_metric(metric_dict, "GL0 Hit Rate")
    get_metric(metric_dict, "GL0 Request Latency")

    # Table 304: LDS
    lds_util = get_metric(metric_dict, "LDS Utilization")
    get_metric(metric_dict, "LDS Utilization - Load")
    get_metric(metric_dict, "LDS Utilization - Store")
    lds_bank_conflict = get_metric(metric_dict, "LDS Bank Conflict Stall Rate")
    lds_load_bw = get_metric(metric_dict, "LDS Load Bandwidth")
    lds_store_bw = get_metric(metric_dict, "LDS Store Bandwidth")
    lds_atomic_bw = get_metric(metric_dict, "LDS Atomic Bandwidth")
    async_load_bw = get_metric(metric_dict, "Async Load Bandwidth")
    async_store_bw = get_metric(metric_dict, "Async Store Bandwidth")
    tdm_load_bw = get_metric(metric_dict, "TDM Load Bandwidth")
    tdm_store_bw = get_metric(metric_dict, "TDM Store Bandwidth")
    tdm_load_req = get_metric(metric_dict, "TDM Load Requests")
    tdm_store_req = get_metric(metric_dict, "TDM Store Requests")
    # LDS Request Counts (new)
    lds_load_req = get_metric(metric_dict, "LDS Load Requests")
    lds_store_req = get_metric(metric_dict, "LDS Store Requests")
    lds_atomic_req = get_metric(metric_dict, "LDS Atomic Requests")
    async_load_req = get_metric(metric_dict, "Async Load Requests")
    async_store_req = get_metric(metric_dict, "Async Store Requests")

    # Table 305: GL0-GL1 Interface
    get_metric(metric_dict, "GL0-GL1 Read Requests")
    get_metric(metric_dict, "GL0-GL1 Write Requests")
    gl0_gl1_read_bw = get_metric(metric_dict, "GL0-GL1 Read Bandwidth")
    gl0_gl1_write_bw = get_metric(metric_dict, "GL0-GL1 Write Bandwidth")
    gl0_gl1_atomic_bw = get_metric(metric_dict, "GL0-GL1 Atomic Bandwidth")
    get_metric(metric_dict, "GL0-GL1 Read Latency")
    get_metric(metric_dict, "GL0-GL1 Write Latency")
    get_metric(metric_dict, "GL0-GL1 Request Stall Rate")
    gl1a_util = get_metric(metric_dict, "GL1A Utilization")
    gl1c_util = get_metric(metric_dict, "GL1C Utilization")
    gl1c_buf_stall = get_metric(metric_dict, "GL1C Buffer Full Stall")

    # Table 306: GLARB
    glarba_util = get_metric(metric_dict, "GLARBA Utilization")
    get_metric(metric_dict, "GLARBA Request Path Utilization")
    get_metric(metric_dict, "GLARBA Return Path Utilization")
    glarbc_util = get_metric(metric_dict, "GLARBC Utilization")
    glarbc_stall = get_metric(metric_dict, "GLARBC Buffer Full Stall")
    gl1_glarb_read_bw = get_metric(metric_dict, "GL1-GLARB Read Bandwidth")
    gl1_glarb_write_bw = get_metric(metric_dict, "GL1-GLARB Write Bandwidth")
    glarb_gl2_read_bw = get_metric(metric_dict, "GLARB-GL2 Read Bandwidth")
    glarb_gl2_write_bw = get_metric(metric_dict, "GLARB-GL2 Write Bandwidth")
    glarb_gl2_atomic_bw = get_metric(metric_dict, "GLARB-GL2 Atomic Bandwidth")

    # Table 307: GL2 Cache
    gl2_arb_util = get_metric(metric_dict, "GL2A Utilization")
    gl2_hit = get_metric(metric_dict, "GL2C Hit Rate")
    gl2_ea_read_bw = get_metric(metric_dict, "GL2-EA Read Bandwidth")
    gl2_ea_write_bw = get_metric(metric_dict, "GL2-EA Write Bandwidth")
    gl2_ea_atomic_bw = get_metric(metric_dict, "GL2-EA Atomic Bandwidth")

    # Table 308: EA to HBM/IO/HDM/GMI
    ea_util = get_metric(metric_dict, "EA Utilization")
    hbm_read_bw = get_metric(metric_dict, "DRAM Read Bandwidth", 0)
    hbm_write_bw = get_metric(metric_dict, "DRAM Write Bandwidth", 0)
    io_read_bw = get_metric(metric_dict, "IO Read Bandwidth", 0)
    io_write_bw = get_metric(metric_dict, "IO Write Bandwidth", 0)
    hdm_read_bw = get_metric(metric_dict, "HDM Read Bandwidth", 0)
    hdm_write_bw = get_metric(metric_dict, "HDM Write Bandwidth", 0)
    gmi_read_bw = get_metric(metric_dict, "GMI Read Bandwidth", 0)
    gmi_write_bw = get_metric(metric_dict, "GMI Write Bandwidth", 0)
    ea_stall = get_metric(metric_dict, "EA Stall Rate")

    # Calculate total HBM BW
    total_bw = (hbm_read_bw or 0) + (hbm_write_bw or 0)

    # Print header
    console.print()
    console.print(f"[Normalization: {normal_unit}]")
    console.print(
        " " * 24
        + "|"
        + "-" * 43
        + " [dim]XCD[/dim] "
        + "-" * 43
        + "|"
        + "-" * 34
        + " [dim]AID[/dim] "
        + "-" * 34
        + "|"
    )
    console.print()

    # Edge height per line
    edge_height = 3
    min_padding = 2

    # Standard arrow length (11 characters total: <---------- or ---------->)
    std_arrow_len = 10  # Number of dashes
    std_arrow_left = "<" + "-" * std_arrow_len  # <----------
    std_arrow_right = "-" * std_arrow_len + ">"  # ---------->
    "<" + "-" * (std_arrow_len - 1) + ">"  # <--------->
    std_arrow_bidir = (
        "<" + "-" * (std_arrow_len - 2) + ">"
    )  # <-------->  (for bidirectional)

    # ========================================================================
    # (1) Kernel block - origin
    # ========================================================================
    kernel = RegularBlock(
        label="Kernel",
        x_min=0,
        x_max=14,  # MIN: enough for "Kernel" label with border
        y_min=10,
        y_max=75,  # Standard height
        color=COLORS["kernel"],
        sub_blocks=[],
    )

    # ========================================================================
    # (2) NEW LAYOUT: TCP at top, SQC below TCP (swapped from original)
    # ========================================================================
    # Helper to create edge group with compact flag
    def make_edge_group(**kwargs: object) -> AlignedEdgesGroup:
        return AlignedEdgesGroup(compact=compact, **kwargs)

    # Edge labels for width calculation
    # Edge labels in top-down order: LDS (top) -> GL0 -> SQC (bottom)

    # New edge format for Kernel -> 2nd column: "Label   : value     " on one line
    # Label: 8 chars, Value: 10 chars, Total arrow width: 18 chars
    kernel_edge_label_width = 8
    kernel_edge_value_width = 10
    kernel_edge_total_width = 18
    edge_col_width = kernel_edge_total_width + 1  # Total width + spacing

    # Helper function to format edge label in new style: "Label      : value"
    def format_kernel_edge(label: str, value: Union[int, float, str, None]) -> str:
        label_str = f"{label:<{kernel_edge_label_width}}"
        if value is not None:
            value_str = f": {format_sci(value):>{kernel_edge_value_width - 2}}"
        else:
            value_str = ""
        return f"{label_str}{value_str}"

    # Kernel -> 2nd column arrows (22 chars total)
    kernel_arrow_left = "<" + "-" * (
        kernel_edge_total_width - 1
    )  # <---------------------
    kernel_arrow_right = (
        "-" * (kernel_edge_total_width - 1) + ">"
    )  # --------------------->
    kernel_arrow_both = (
        "<" + "-" * (kernel_edge_total_width - 2) + ">"
    )  # <------------------->

    # Standard arrows for other edges (keep original)

    # ========================================================================
    # (2a) Edges: Kernel -> TCP/LDS (Load/Store/Atomic/Async/TDM) - AT TOP
    #      (LDS is now top sub-block)
    # ========================================================================
    edges_lds = make_edge_group(
        label="Edges_LDS",
        x_min=kernel.x_max + 2,
        x_max=kernel.x_max + 2 + edge_col_width,
        y_min=kernel.y_min + 4,  # Start near top of kernel, aligned with LDS sub-block
        y_max=kernel.y_min + 4 + (edge_height * 7),
        edges=[
            Edge(
                label=format_kernel_edge("Load", lds_load_req),
                arrow=kernel_arrow_left,
                color=COLORS["read"],
            ),
            Edge(
                label=format_kernel_edge("Store", lds_store_req),
                arrow=kernel_arrow_right,
                color=COLORS["write"],
            ),
            Edge(
                label=format_kernel_edge("Atomic", lds_atomic_req),
                arrow=kernel_arrow_both,
                color=COLORS["atomic"],
            ),
            Edge(
                label=format_kernel_edge("Async Ld", async_load_req),
                arrow=kernel_arrow_left,
                color=COLORS["read"],
            ),
            Edge(
                label=format_kernel_edge("Async St", async_store_req),
                arrow=kernel_arrow_right,
                color=COLORS["write"],
            ),
            Edge(
                label=format_kernel_edge("TDM Ld", tdm_load_req),
                arrow=kernel_arrow_left,
                color=COLORS["read"],
            ),
            Edge(
                label=format_kernel_edge("TDM St", tdm_store_req),
                arrow=kernel_arrow_right,
                color=COLORS["write"],
            ),
        ],
    )

    # ========================================================================
    # (2b) Edges: Kernel -> TCP/GL0 (Read/Write/Atomic) - BELOW LDS
    #      (GL0 is now bottom sub-block)
    # ========================================================================
    edges_gl0 = make_edge_group(
        label="Edges_GL0",
        x_min=kernel.x_max + 2,
        x_max=kernel.x_max + 2 + edge_col_width,
        y_min=edges_lds.y_max
        + min_padding
        + 2,  # Below LDS edges, aligned with GL0 sub-block
        y_max=edges_lds.y_max + min_padding + 2 + (edge_height * 3),
        edges=[
            Edge(
                label=format_kernel_edge("Read", gl0_read_req),
                arrow=kernel_arrow_left,
                color=COLORS["read"],
            ),
            Edge(
                label=format_kernel_edge("Write", gl0_write_req),
                arrow=kernel_arrow_right,
                color=COLORS["write"],
            ),
            Edge(
                label=format_kernel_edge("Atomic", gl0_atomic_req),
                arrow=kernel_arrow_both,
                color=COLORS["atomic"],
            ),
        ],
    )

    # ========================================================================
    # (2c) Edges: Kernel -> SQC (ICACHE, SMEM) - NOW BELOW LDS EDGES
    # ========================================================================
    edges_icache = make_edge_group(
        label="Edge_ICACHE",
        x_min=kernel.x_max + 2,
        x_max=kernel.x_max + 2 + edge_col_width,
        y_min=edges_gl0.y_max
        + min_padding
        + 4,  # Below GL0 edges (which is now below LDS)
        y_max=edges_gl0.y_max + min_padding + 4 + edge_height,
        top_padding=2,
        edges=[
            Edge(
                label=format_kernel_edge("ICACHE", icache_req),
                arrow=kernel_arrow_left,
                color=COLORS["read"],
            )
        ],
    )

    edges_smeme = make_edge_group(
        label="Edge_SMEME",
        x_min=kernel.x_max + 2,
        x_max=kernel.x_max + 2 + edge_col_width,
        y_min=edges_icache.y_max + min_padding + 2,
        y_max=edges_icache.y_max + min_padding + 2 + edge_height,
        edges=[
            Edge(
                label=format_kernel_edge("SMEM", dcache_req),
                arrow=kernel_arrow_left,
                color=COLORS["read"],
            )
        ],
    )

    # ========================================================================
    # (3) TCP block - NOW AT TOP of column (aligned with kernel top)
    # ========================================================================
    tcp = RegularBlock(
        label="TCP",
        x_min=edges_lds.x_max + 1,  # Right of LDS edges (LDS is now at top)
        x_max=edges_lds.x_max + 28,
        y_min=kernel.y_min,  # Align with kernel top
        y_max=kernel.y_min + 43,  # Height for LDS + GL0 sub-blocks (increased)
        color=COLORS["block"],
        content_text="TXA/TXD/TDM",
        content_y_offset=0,
        vertical_position="top",  # Content at top (TXA/TXD/TDM label first)
        sub_blocks=[
            SubBlock(
                label="LDS",
                attributes=[
                    "",  # 1 line padding at top
                    metric_line("Util", lds_util, "%", COLORS["util"]),
                    metric_line(
                        "Bank Conflict", lds_bank_conflict, "%", COLORS["stall"]
                    ),
                    "",
                    metric_line(
                        "Load BW",
                        lds_load_bw / 1e9 if lds_load_bw else None,
                        "GB/s",
                        COLORS["read"],
                    ),
                    metric_line(
                        "Store BW",
                        lds_store_bw / 1e9 if lds_store_bw else None,
                        "GB/s",
                        COLORS["write"],
                    ),
                    metric_line(
                        "Atomic BW",
                        lds_atomic_bw / 1e9 if lds_atomic_bw else None,
                        "GB/s",
                        COLORS["atomic"],
                    ),
                ],
                y_offset=0,  # LDS now at top
                # Covers all 7 edges: Load/Store/Atomic/Async Load/Async Store/TDM Ld/St
                height=22,
                vertical_position="top",
            ),
            SubBlock(
                label="GL0",
                attributes=[
                    metric_line("Util", gl0_util, "%", COLORS["util"]),
                    metric_line("Hit Rate", gl0_hit, "%", COLORS["hit"]),
                ],
                y_offset=28,  # GL0 now starts after LDS ends
                height=16,
            ),
        ],
    )

    # ========================================================================
    # (4) SQC block - NOW BELOW TCP
    # ========================================================================
    sqc = RegularBlock(
        label="SQC",
        x_min=tcp.x_min,  # Same column as TCP
        x_max=tcp.x_max,
        y_min=tcp.y_max + 4,  # Below TCP with gap
        y_max=tcp.y_max + 4 + 17,  # Height for ICache + DCache sub-blocks
        color=COLORS["block"],
        sub_blocks=[
            SubBlock(
                label="Instruction Cache",
                attributes=[metric_line("Hit Rate", icache_hit, "%", COLORS["hit"])],
                height=6,
            ),
            SubBlock(
                label="Scalar data Cache",
                attributes=[metric_line("Hit Rate", dcache_hit, "%", COLORS["hit"])],
                y_offset=7,
                height=6,
            ),
        ],
    )

    # ========================================================================
    # (5) Edges: TCP/GL0 -> GL1 (Read/Write/Atomic BW) - NOW AT TOP
    # ========================================================================
    edges_gl0_gl1 = make_edge_group(
        label="Edges_GL0_GL1",
        x_min=tcp.x_max + 1,
        x_max=tcp.x_max + 10,
        y_min=tcp.y_min + 8,  # Align with GL0 sub-block
        y_max=tcp.y_min + 20,
        top_padding=2,
        edges=[
            Edge(
                label=f"Read BW\n{format_bw_gbps(gl0_gl1_read_bw)}",
                arrow=std_arrow_left,
                color=COLORS["read"],
            ),
            Edge(
                label=f"Write BW\n{format_bw_gbps(gl0_gl1_write_bw)}",
                arrow=std_arrow_right,
                color=COLORS["write"],
            ),
            Edge(
                label=f"Atomic BW\n{format_bw_gbps(gl0_gl1_atomic_bw)}",
                arrow=std_arrow_bidir,
                color=COLORS["atomic"],
            ),
        ],
    )

    # ========================================================================
    # (6) Edges: TCP/LDS -> GL1 (Async Load/Store BW, TDM Load/Store BW)
    #     Position: x_min = edges_gl0_gl1.x_min (same column as GL0-GL1 edges)
    #               y_min = edges_gl0_gl1.y_max + gap (below GL0-GL1, aligned with LDS)
    # ========================================================================
    edges_lds_gl1 = make_edge_group(
        label="Edges_LDS_GL1",
        x_min=edges_gl0_gl1.x_min,
        x_max=edges_gl0_gl1.x_max,
        y_min=edges_gl0_gl1.y_max + 14,  # Gap to align with LDS (+2 lines)
        y_max=edges_gl0_gl1.y_max + 30,
        top_padding=2,
        edges=[
            Edge(
                label=f"Async Load\n{format_bw_gbps(async_load_bw)}",
                arrow=std_arrow_left,
                color=COLORS["read"],
            ),
            Edge(
                label=f"Async Store\n{format_bw_gbps(async_store_bw)}",
                arrow=std_arrow_right,
                color=COLORS["write"],
            ),
            Edge(
                label=f"TDM Load\n{format_bw_gbps(tdm_load_bw)}",
                arrow=std_arrow_left,
                color=COLORS["read"],
            ),
            Edge(
                label=f"TDM Store\n{format_bw_gbps(tdm_store_bw)}",
                arrow=std_arrow_right,
                color=COLORS["write"],
            ),
        ],
    )

    # ========================================================================
    # (7) Edges: SQC -> GL1 (ICache/DCache Read BW) - NOW BELOW LDS EDGES
    # ========================================================================
    edges_icache_gl1 = make_edge_group(
        label="Edges_ICache_GL1",
        x_min=edges_gl0_gl1.x_min,
        x_max=edges_gl0_gl1.x_max,
        y_min=sqc.y_min + 3,  # Align with ICache sub-block
        y_max=sqc.y_min + 8,
        top_padding=2,
        edges=[
            Edge(
                label=f"Read BW\n{format_bw_gbps(icache_gl1_bw)}",
                arrow=std_arrow_left,
                color=COLORS["read"],
            )
        ],
    )

    edges_dcache_gl1 = make_edge_group(
        label="Edges_DCache_GL1",
        x_min=edges_icache_gl1.x_min,
        x_max=edges_icache_gl1.x_max,
        y_min=edges_icache_gl1.y_max + 2,
        y_max=edges_icache_gl1.y_max + 7,
        top_padding=2,
        edges=[
            Edge(
                label=f"Read BW\n{format_bw_gbps(dcache_gl1_bw)}",
                arrow=std_arrow_left,
                color=COLORS["read"],
            )
        ],
    )

    # ========================================================================
    # (8) GL1 block
    #     Position: x_min = edges_gl0_gl1.x_max + 1 (right of GL0-GL1 edges)
    #               y_min = tcp.y_min (aligned with TCP top, not SQC)
    #               y_max = 75 (standard height for main cache blocks)
    # ========================================================================
    gl1 = RegularBlock(
        label="GL1",
        x_min=edges_gl0_gl1.x_max + 1,
        x_max=edges_gl0_gl1.x_max + 16,
        y_min=tcp.y_min,  # Align with TCP top (not SQC)
        y_max=75,  # Standard height
        color=COLORS["block"],
        vertical_position="top",  # Move content to top of block
        sub_blocks=[
            SubBlock(
                label="",
                attributes=[
                    metric_line("GL1A Util", gl1a_util, "%", COLORS["util"]),
                    bar_line(gl1a_util) if gl1a_util else "",
                    "",
                    metric_line("GL1C Util", gl1c_util, "%", COLORS["util"]),
                    bar_line(gl1c_util) if gl1c_util else "",
                    "",
                    metric_line("Buf Stall", gl1c_buf_stall, "%", COLORS["stall"]),
                ],
                height=12,
                show_border=False,
                vertical_position="top",
            )
        ],
    )

    # ========================================================================
    # (9) Edges: GL1 -> GLARBA
    # ========================================================================
    # Center edges vertically within GL1 height
    # GL1 height is about 65 lines, edges need about 8 lines, so padding = (65-8)/2 ~ 28
    gl1_edge_padding = (gl1.height - 8) // 2

    edges_gl1_glarba = make_edge_group(
        label="Edges_GL1_GLARBA",
        x_min=gl1.x_max + 1,  # MIN: reduced gap
        x_max=gl1.x_max + 9,  # MIN: reduced from 12
        y_min=gl1.y_min,
        y_max=gl1.y_max,
        top_padding=gl1_edge_padding,
        edges=[
            Edge(
                label=f"Read BW\n{format_bw_gbps(gl1_glarb_read_bw)}",
                arrow=std_arrow_left,
                color=COLORS["read"],
            ),
            Edge(
                label=f"Write BW\n{format_bw_gbps(gl1_glarb_write_bw)}",
                arrow=std_arrow_right,
                color=COLORS["write"],
            ),
        ],
    )

    # ========================================================================
    # (10) GLARBA block (Table 306 metrics)
    #      Position: x_min = edges_gl1_glarba.x_max + 1 (right of GL1-GLARB edges)
    #                y_min = gl1.y_min (aligned with GL1 top)
    #                y_max = gl1.y_max (aligned with GL1 bottom)
    # ========================================================================
    glarba = RegularBlock(
        label="GLARB",
        x_min=edges_gl1_glarba.x_max + 1,  # MIN: reduced gap
        x_max=edges_gl1_glarba.x_max + 16,  # Increased by 2 for "GLARBA Util" to fit
        y_min=gl1.y_min,
        y_max=gl1.y_max,
        vertical_position="top",  # Move content to top of block
        color=COLORS["block"],
        sub_blocks=[
            SubBlock(
                label="",
                attributes=[
                    metric_line("GLARBA Util", glarba_util, "%", COLORS["util"]),
                    bar_line(glarba_util),
                    "",
                    metric_line("GLARBC Util", glarbc_util, "%", COLORS["util"]),
                    bar_line(glarbc_util),
                    "",
                    metric_line("Buf Stall", glarbc_stall, "%", COLORS["stall"]),
                ],
                height=8,
                show_border=False,
            )
        ],
    )

    # ========================================================================
    # (11) Edges: GLARBA -> GL2
    # ========================================================================
    # Center edges vertically within GLARB height (3 edges now: Read, Write, Atomic)
    glarb_edge_padding = (
        glarba.height - 12
    ) // 2 + 2  # Adjusted for 3 edges, moved down 2 lines

    edges_glarba_gl2 = make_edge_group(
        label="Edges_GLARBA_GL2",
        x_min=glarba.x_max + 1,  # MIN: reduced gap
        x_max=glarba.x_max + 9,  # MIN: reduced from 12
        y_min=glarba.y_min,
        y_max=glarba.y_max,
        top_padding=glarb_edge_padding,
        edges=[
            Edge(
                label=f"Read BW\n{format_bw_gbps(glarb_gl2_read_bw)}",
                arrow=std_arrow_left,
                color=COLORS["read"],
            ),
            Edge(
                label=f"Write BW\n{format_bw_gbps(glarb_gl2_write_bw)}",
                arrow=std_arrow_right,
                color=COLORS["write"],
            ),
            Edge(
                label=f"Atomic BW\n{format_bw_gbps(glarb_gl2_atomic_bw)}",
                arrow=std_arrow_bidir,
                color=COLORS["atomic"],
            ),
        ],
    )

    # ========================================================================
    # (12) GL2 block (Table 307 metrics)
    #      Position: x_min = edges_glarba_gl2.x_max + 1 (right of GLARB-GL2 edges)
    #                y_min = glarba.y_min (aligned with GLARB top)
    #                y_max = 75 (standard height)
    # ========================================================================
    gl2 = RegularBlock(
        label="GL2",
        x_min=edges_glarba_gl2.x_max + 1,  # MIN: reduced gap
        x_max=edges_glarba_gl2.x_max + 16,  # Aligned with GLARB width
        y_min=glarba.y_min,
        y_max=75,  # Standard height
        vertical_position="top",  # Move content to top of block
        color=COLORS["block"],
        sub_blocks=[
            SubBlock(
                label="",
                attributes=[
                    metric_line("GL2A Util", gl2_arb_util, "%", COLORS["util"]),
                    bar_line(gl2_arb_util),
                    "",
                    metric_line("GL2C Hit", gl2_hit, "%", COLORS["hit"]),
                    bar_line(gl2_hit),
                ],
                height=8,
                show_border=False,
            )
        ],
    )

    # ========================================================================
    # (13) Edges: GL2 -> DF
    # ========================================================================
    # Center edges vertically within GL2 height (3 edges now: Read, Write, Atomic)
    gl2_edge_padding = (
        gl2.height - 12
    ) // 2 + 2  # Adjusted for 3 edges, moved down 2 lines

    edges_gl2_df = make_edge_group(
        label="Edges_GL2_DF",
        x_min=gl2.x_max + 1,  # MIN: reduced gap
        x_max=gl2.x_max + 9,  # MIN: reduced from 12
        y_min=gl2.y_min,
        y_max=gl2.y_max,
        top_padding=gl2_edge_padding,
        edges=[
            Edge(
                label=f"Read BW\n{format_bw_gbps(gl2_ea_read_bw)}",
                arrow=std_arrow_left,
                color=COLORS["read"],
            ),
            Edge(
                label=f"Write BW\n{format_bw_gbps(gl2_ea_write_bw)}",
                arrow=std_arrow_right,
                color=COLORS["write"],
            ),
            Edge(
                label=f"Atomic BW\n{format_bw_gbps(gl2_ea_atomic_bw)}",
                arrow=std_arrow_bidir,
                color=COLORS["atomic"],
            ),
        ],
    )

    # ========================================================================
    # (14) EA/DF block (Table 308 metrics)
    #      Position: x_min = edges_gl2_df.x_max + 1 (right of GL2-DF edges)
    #                y_min = gl2.y_min (aligned with GL2 top)
    #                y_max = 75 (standard height)
    # ========================================================================
    df = RegularBlock(
        label="EA/DF",
        x_min=edges_gl2_df.x_max + 1,  # MIN: reduced gap
        x_max=edges_gl2_df.x_max + 16,  # Aligned with GLARB width
        y_min=gl2.y_min,
        y_max=75,  # Standard height
        vertical_position="top",  # Move content to top of block
        color=COLORS["block"],
        sub_blocks=[
            SubBlock(
                label="",
                attributes=[
                    metric_line("EA Util", ea_util, "%", COLORS["util"]),
                    bar_line(ea_util),
                    "",
                    metric_line("Stall", ea_stall, "%", COLORS["stall"]),
                ],
                height=8,
                show_border=False,
            )
        ],
    )

    # ========================================================================
    # (15) EA downstream blocks: DRAM (UMC/HBM), IO, HDM, GMI
    #      EA has 4 downstream destinations, each with Read/Write BW
    #      Layout: HBM at top (y_min = df.y_min), IO/HDM/GMI below
    #      Position: HBM.x_min = edges_df_umc.x_max + 1 (right of DRAM edges)
    #                IO/HDM/GMI.x_min = hbm.x_max + 1 (right of HBM)
    #                IO/HDM/GMI.y calculated top-down from df.y_max
    # ========================================================================

    # MIN: reduced downstream dimensions
    edge_width = 8  # MIN: reduced from 10

    # Calculate positions to align all downstream blocks within EA/DF height
    # Total available height from df.y_min to df.y_max
    total_downstream_height = df.y_max - df.y_min
    gap_size = 1  # Gap between blocks
    num_gaps = 3  # HBM-IO, IO-HDM, HDM-GMI
    total_downstream_height - (num_gaps * gap_size)

    # Calculate top-down: HBM first, then IO, HDM, GMI aligned to bottom of EA/DF
    ext_block_height = 8  # Minimized height for IO, HDM, GMI

    # First calculate where IO, HDM, GMI need to be (aligned to bottom)
    # Total height needed for IO + HDM + GMI + gaps
    # = 3 * ext_block_height + 2 * gap_size
    ext_total_height = 3 * ext_block_height + 2 * gap_size

    # HBM: from top of EA/DF to where external blocks start
    hbm_y_min = df.y_min
    hbm_y_max = df.y_max - ext_total_height - gap_size  # Leave gap before IO

    # IO: starts after HBM gap, aligned to bottom section
    io_y_min = hbm_y_max + gap_size
    io_y_max = io_y_min + ext_block_height

    # HDM: after IO
    hdm_y_min = io_y_max + gap_size
    hdm_y_max = hdm_y_min + ext_block_height

    # GMI: after HDM, bottom aligns with EA/DF
    gmi_y_min = hdm_y_max + gap_size
    gmi_y_max = df.y_max  # Align bottom with EA/DF

    # --- DRAM (UMC/HBM) section ---
    dram_arrow = "<" + "-" * 10  # 11 chars total: <----------
    dram_arrow_r = "-" * 10 + ">"  # 11 chars total: ---------->

    # Align DRAM edges with "Read BW" between GL2 and EA/DF
    # Use top_padding to push DRAM edges down to align with the centered BW edges
    dram_top_padding = gl2_edge_padding  # Same padding as GL2-EA edges

    edges_df_umc = make_edge_group(
        label="Edges_DF_UMC",
        x_min=df.x_max + 1,
        x_max=df.x_max + 1 + edge_width,
        y_min=df.y_min,
        y_max=df.y_min + 20,
        top_padding=dram_top_padding,
        edges=[
            Edge(
                label=f"DRAM Read\n{format_bw_gbps(hbm_read_bw)}",
                arrow=dram_arrow,
                color=COLORS["read"],
            ),
            Edge(
                label=f"DRAM Write\n{format_bw_gbps(hbm_write_bw)}",
                arrow=dram_arrow_r,
                color=COLORS["write"],
            ),
        ],
    )

    hbm = RegularBlock(
        label="HBM",
        x_min=edges_df_umc.x_max + 1,
        x_max=edges_df_umc.x_max + 1 + 16,
        y_min=hbm_y_min,
        y_max=hbm_y_max,
        color=COLORS["block"],
        sub_blocks=[
            SubBlock(
                label="",
                attributes=[
                    "Total",
                    (
                        f"[bold bright_green]{format_bw_gbps(total_bw)}"
                        "[/bold bright_green]"
                    ),
                ],
                height=5,
                show_border=False,
            )
        ],
    )

    # --- IO section ---
    # External edges extend from EA/DF (DRAM edges start) to IO/HDM/GMI blocks
    # Arrow length spans from DRAM edges start to HBM end + gap (1 pixel from blocks)
    ext_arrow_len = (
        hbm.x_max - edges_df_umc.x_min + 1
    )  # +1 leaves 1 pixel gap to blocks

    # External edges coordinates
    ext_edges_x_min = edges_df_umc.x_min
    ext_edges_x_max = hbm.x_max

    # IO block position (after HBM column)
    io_block_x_min = hbm.x_max + 1
    io_block_width = 16

    # External arrows
    ext_arrow_left = "<" + "-" * (ext_arrow_len - 1)  # <------------------
    ext_arrow_right = "-" * (ext_arrow_len - 1) + ">"  # ------------------>

    # Right-pad labels and values to match arrow width
    def pad_to_ext(text: str) -> str:
        return text.rjust(ext_arrow_len)  # Right-aligned

    # EA-IO edges (with labels)
    edges_ea_io = make_edge_group(
        label="EA_IO",
        x_min=ext_edges_x_min,
        x_max=ext_edges_x_max,
        y_min=io_y_min + 1,
        y_max=io_y_min + 1 + 6,
        top_padding=0,
        edges=[
            Edge(
                label=(
                    f"{pad_to_ext('IO Read')}\n{pad_to_ext(format_bw_gbps(io_read_bw))}"
                ),
                arrow=ext_arrow_left,
                color=COLORS["read"],
            ),
            Edge(
                label=(
                    f"{pad_to_ext('IO Write')}\n"
                    f"{pad_to_ext(format_bw_gbps(io_write_bw))}"
                ),
                arrow=ext_arrow_right,
                color=COLORS["write"],
            ),
        ],
    )

    io_block = RegularBlock(
        label="IO",
        x_min=io_block_x_min,
        x_max=io_block_x_min + io_block_width,
        y_min=io_y_min,
        y_max=io_y_max,
        color=COLORS["block"],
        vertical_position="top",
        content_text="PCIe and\nother I/O",
        sub_blocks=[],
    )

    # --- HDM section ---
    # EA-HDM edges (with labels)
    edges_ea_hdm = make_edge_group(
        label="EA_HDM",
        x_min=ext_edges_x_min,
        x_max=ext_edges_x_max,
        y_min=hdm_y_min + 1,
        y_max=hdm_y_min + 1 + 6,
        top_padding=0,
        edges=[
            Edge(
                label=(
                    f"{pad_to_ext('HDM Read')}\n"
                    f"{pad_to_ext(format_bw_gbps(hdm_read_bw))}"
                ),
                arrow=ext_arrow_left,
                color=COLORS["read"],
            ),
            Edge(
                label=(
                    f"{pad_to_ext('HDM Write')}\n"
                    f"{pad_to_ext(format_bw_gbps(hdm_write_bw))}"
                ),
                arrow=ext_arrow_right,
                color=COLORS["write"],
            ),
        ],
    )

    hdm_block = RegularBlock(
        label="HDM",
        x_min=io_block_x_min,  # Same x position as IO
        x_max=io_block_x_min + io_block_width,
        y_min=hdm_y_min,
        y_max=hdm_y_max,
        color=COLORS["block"],
        vertical_position="top",
        content_text="CXL",
        sub_blocks=[],
    )

    # --- GMI section (shifted right, after HBM) ---
    # --- GMI section ---
    # EA-GMI edges (with labels)
    edges_ea_gmi = make_edge_group(
        label="EA_GMI",
        x_min=ext_edges_x_min,
        x_max=ext_edges_x_max,
        y_min=gmi_y_min + 1,
        y_max=gmi_y_min + 1 + 6,
        top_padding=0,
        edges=[
            Edge(
                label=(
                    f"{pad_to_ext('GMI Read')}\n"
                    f"{pad_to_ext(format_bw_gbps(gmi_read_bw))}"
                ),
                arrow=ext_arrow_left,
                color=COLORS["read"],
            ),
            Edge(
                label=(
                    f"{pad_to_ext('GMI Write')}\n"
                    f"{pad_to_ext(format_bw_gbps(gmi_write_bw))}"
                ),
                arrow=ext_arrow_right,
                color=COLORS["write"],
            ),
        ],
    )

    gmi_block = RegularBlock(
        label="GMI",
        x_min=io_block_x_min,  # Same x position as IO
        x_max=io_block_x_min + io_block_width,
        y_min=gmi_y_min,
        y_max=gmi_y_max,
        color=COLORS["block"],
        sub_blocks=[],
    )

    # ========================================================================
    # Build Layout
    # ========================================================================
    layout = Table.grid(padding=0)
    for _ in range(13):
        layout.add_column()

    # Kernel panel
    kernel_panel = Panel(
        "\n" * (kernel.height // 2 - 4)
        + "[dim]Shader Core[/dim]\n[dim]Wave[/dim]\n[dim]Execution[/dim]",
        title=f"[bold {COLORS['kernel']}]Kernel[/bold {COLORS['kernel']}]",
        border_style=COLORS["kernel"],
        width=kernel.width,
        height=kernel.height,
    )

    # Edges column (Kernel -> TCP/SQC) - LDS edges first, then GL0, then SQC
    edges_col = Table.grid()
    edges_col.add_column(width=edge_col_width)  # Fixed width for compact format
    # Header label "Request" at the very top (1 line down from top)
    edges_col.add_row("")
    edges_col.add_row(Text("Request", style="bold white", justify="center"))
    # LDS edges (align with TCP/LDS - 2 lines gap after header)
    edges_col.add_row("")
    edges_col.add_row("")
    edges_col.add_row(edges_lds.render_text())
    # Gap to align with GL0 sub-block (now below LDS) - moved down 3 more lines
    for _ in range(6):
        edges_col.add_row("")
    edges_col.add_row(edges_gl0.render_text())
    # Gap to align with SQC (now below TCP) - ICACHE moved up 1 more line
    for _ in range(5):
        edges_col.add_row("")
    edges_col.add_row(edges_icache.render_text())
    edges_col.add_row("")
    edges_col.add_row("")
    edges_col.add_row("")
    edges_col.add_row(edges_smeme.render_text())

    # TCP/SQC column - NEW ORDER: TCP first, then SQC
    blocks_col = Table.grid()
    blocks_col.add_column()
    blocks_col.add_row(tcp.render())
    blocks_col.add_row(sqc.render())

    # Edges to GL1 column - LDS-GL1 first, then GL0-GL1, then SQC-GL1
    edges_to_gl1 = Table.grid()
    edges_to_gl1.add_column()
    edges_to_gl1.add_row("")  # 7 lines gap (moved down by 2 more)
    edges_to_gl1.add_row("")
    edges_to_gl1.add_row("")
    edges_to_gl1.add_row("")
    edges_to_gl1.add_row("")
    edges_to_gl1.add_row("")
    edges_to_gl1.add_row("")
    edges_to_gl1.add_row(edges_lds_gl1.render_text())  # Align with LDS (now at top)
    # Gap before GL0 edges - reduced to align with GL1-GLARB Read BW
    for _ in range(1):
        edges_to_gl1.add_row("")
    edges_to_gl1.add_row(edges_gl0_gl1.render_text())  # Align with GL0 (now below LDS)
    # Gap before SQC-GL1 edges
    for _ in range(4):
        edges_to_gl1.add_row("")
    edges_to_gl1.add_row(edges_icache_gl1.render_text())  # Align with ICache
    edges_to_gl1.add_row(edges_dcache_gl1.render_text())  # Align with DCache

    # EA downstream: HBM at top, then IO/HDM/GMI below
    # All edges come from EA/DF, IO/HDM/GMI are positioned below HBM

    # Helper function to combine edges and block in a single cell
    def combine_edge_block(edge_group: AlignedEdgesGroup, block: RegularBlock) -> Table:
        combined = Table.grid(padding=0)
        combined.add_column(no_wrap=True)  # Edges (labels + arrows)
        combined.add_column(no_wrap=True)  # Block
        combined.add_row(edge_group.render_text(), block.render())
        return combined

    # Build the downstream column with proper vertical layout
    # Structure: Row 1 = DRAM edges + HBM
    #            Row 2+ = Long arrows spanning DRAM+HBM width + IO/HDM/GMI blocks

    downstream_col = Table.grid(padding=0)
    downstream_col.add_column()  # Column for edges
    downstream_col.add_column()  # Column for blocks

    # Row 1: DRAM edges + HBM in a sub-grid
    hbm_row = Table.grid(padding=0)
    hbm_row.add_column(no_wrap=True)  # DRAM edges
    hbm_row.add_column(no_wrap=True)  # HBM block
    hbm_row.add_row(edges_df_umc.render_text(), hbm.render())

    downstream_col.add_row(hbm_row, "")

    # Gap row between HBM and IO/HDM/GMI
    downstream_col.add_row("", "")
    downstream_col.add_row("", "")
    downstream_col.add_row("", "")

    # Row 2: IO edges + IO block
    downstream_col.add_row(edges_ea_io.render_text(), io_block.render())

    # Row 3: HDM edges + HDM block
    downstream_col.add_row(edges_ea_hdm.render_text(), hdm_block.render())

    # Row 4: GMI edges + GMI block
    downstream_col.add_row(edges_ea_gmi.render_text(), gmi_block.render())

    # Assemble layout
    layout.add_row(
        kernel_panel,
        edges_col,
        blocks_col,
        edges_to_gl1,
        gl1.render(),
        edges_gl1_glarba.render_text(),
        glarba.render(),
        edges_glarba_gl2.render_text(),
        gl2.render(),
        edges_gl2_df.render_text(),
        df.render(),
        downstream_col,
    )

    console.print(layout)
    console.print()

    # Legend
    r, w, a, u, h, s = (
        COLORS["read"],
        COLORS["write"],
        COLORS["atomic"],
        COLORS["util"],
        COLORS["hit"],
        COLORS["stall"],
    )
    console.print(
        f"[dim]Legend:[/dim] [{r}]<----[/{r}] Read  [{w}]---->[/{w}] Write"
        f"  [{a}]<--->[/{a}] Atomic  [{u}]#[/{u}] Util"
        f"  [{h}]#[/{h}] Hit%  [{s}]#[/{s}] Stall"
    )
    console.print()

    # Debug info
    if show_debug:
        console.print("[dim]Block Coordinates (Relative Positioning):[/dim]")
        console.print(
            f"  Kernel: x({kernel.x_min}-{kernel.x_max}),"
            f" y({kernel.y_min}-{kernel.y_max})"
        )
        console.print(
            f"  TCP: x({tcp.x_min}-{tcp.x_max}), y({tcp.y_min}-{tcp.y_max})"
            "  [x_min = edges.x_max + 1, y_min = kernel.y_min]"
        )
        # Get sub-block info from tcp.sub_blocks
        lds_sub = tcp.sub_blocks[0]  # LDS is first sub-block (y_offset=0)
        gl0_sub = tcp.sub_blocks[1]  # GL0 is second sub-block (y_offset=28)
        lds_y_min = tcp.y_min + 2 + lds_sub.y_offset  # Account for label + border
        lds_y_max = lds_y_min + lds_sub.height
        console.print(
            f"    LDS (sub-block): y({lds_y_min}-{lds_y_max}),"
            f" y_offset={lds_sub.y_offset}, height={lds_sub.height}"
        )
        gl0_y_min = tcp.y_min + 2 + gl0_sub.y_offset  # Account for label + border
        gl0_y_max = gl0_y_min + gl0_sub.height
        console.print(
            f"    GL0 (sub-block): y({gl0_y_min}-{gl0_y_max}),"
            f" y_offset={gl0_sub.y_offset}, height={gl0_sub.height}"
        )
        console.print(
            f"  SQC: x({sqc.x_min}-{sqc.x_max}), y({sqc.y_min}-{sqc.y_max})"
            "  [x_min = tcp.x_min, y_min = tcp.y_max + 4]"
        )
        console.print(f"  GL1: x({gl1.x_min}-{gl1.x_max}), y({gl1.y_min}-{gl1.y_max})")
        console.print(
            f"  GLARBA: x({glarba.x_min}-{glarba.x_max}),"
            f" y({glarba.y_min}-{glarba.y_max})"
        )
        console.print(f"  GL2: x({gl2.x_min}-{gl2.x_max}), y({gl2.y_min}-{gl2.y_max})")
        console.print(f"  DF: x({df.x_min}-{df.x_max}), y({df.y_min}-{df.y_max})")
        console.print(f"  HBM: x({hbm.x_min}-{hbm.x_max}), y({hbm.y_min}-{hbm.y_max})")
        console.print(
            f"  IO: x({io_block.x_min}-{io_block.x_max}),"
            f" y({io_block.y_min}-{io_block.y_max})"
        )
        console.print(
            f"  HDM: x({hdm_block.x_min}-{hdm_block.x_max}),"
            f" y({hdm_block.y_min}-{hdm_block.y_max})"
        )
        console.print(
            f"  GMI: x({gmi_block.x_min}-{gmi_block.x_max}),"
            f" y({gmi_block.y_min}-{gmi_block.y_max})"
        )
        console.print("  Edges:")
        console.print(
            f"    DRAM Read/Write: x({edges_df_umc.x_min}-{edges_df_umc.x_max}),"
            f" y({edges_df_umc.y_min}-{edges_df_umc.y_max})"
        )
        console.print("  EA External Edges:")
        console.print(
            f"    EA-IO: x({edges_ea_io.x_min}-{edges_ea_io.x_max}),"
            f" y({edges_ea_io.y_min}-{edges_ea_io.y_max})"
        )
        console.print(
            f"    EA-HDM: x({edges_ea_hdm.x_min}-{edges_ea_hdm.x_max}),"
            f" y({edges_ea_hdm.y_min}-{edges_ea_hdm.y_max})"
        )
        console.print(
            f"    EA-GMI: x({edges_ea_gmi.x_min}-{edges_ea_gmi.x_max}),"
            f" y({edges_ea_gmi.y_min}-{edges_ea_gmi.y_max})"
        )
        console.print()


# ---------------------------------------------------------------------------
# Public API: heading, normalization, sample data
# ---------------------------------------------------------------------------


def format_mem_chart_heading(
    normal_unit: str,
    *,
    panel_id: int = 300,
    section_label: str = "Memory Chart",
) -> str:
    """Build CLI diagram title: ``{panel_id//100}. {label} (Normalization: ...)``.

    Matches other panels (e.g. ``3. System Speed-of-Light``) where the leading
    number is ``Panel Config id // 100`` (panel 300 -> ``3.``).
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


# ============================================================================
# Public API (matches mem_chart.py interface)
# ============================================================================
def plot_mem_chart(
    arch: str,
    normal_unit: str,
    metric_dict: dict[str, Any],
    *,
    chart_title: Optional[str] = None,
) -> str:
    """
    Plot the memory chart and return as string.

    Args:
        arch: Architecture name (e.g., "gfx1250")
        normal_unit: Normalization unit (e.g., "per_kernel", "per_second")
        metric_dict: Dictionary of metric name -> value (keys should match
            ``MEM_CHART_PANEL_METRIC_KEYS``). Bandwidth values are in **Bytes/s**.
        chart_title: Full heading line; if omitted, uses ``format_mem_chart_heading``
            with ``panel_id=300`` (section ``3.``).

    Returns:
        String representation of the diagram
    """
    flat = normalize_mem_chart_metrics(metric_dict)
    (
        format_mem_chart_heading(normal_unit, panel_id=300)
        if chart_title is None
        else chart_title
    )
    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=220, height=100)

    create_mem_chart_diagram(
        arch,
        normal_unit,
        flat,
        console,
        show_debug=False,
        compact=False,
    )

    return buf.getvalue()


# ============================================================================
# Data Loading
# ============================================================================
def load_metrics_from_json(filepath: str) -> dict[str, Any]:
    """Load metrics from JSON file"""
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# Metrics Schema (for validation)
# ============================================================================
METRICS_SCHEMA = {
    "type": "object",
    "description": "gfx1250 Memory Chart metrics input schema",
    "properties": {
        # ICache
        "ICache Requests": {
            "type": "number",
            "description": "Number of instruction cache requests",
        },
        "ICache Utilization": {
            "type": "number",
            "description": "ICache utilization percentage (0-100)",
        },
        "ICache Hit Rate": {
            "type": "number",
            "description": "ICache hit rate percentage (0-100)",
        },
        "ICache Request Stall Rate": {
            "type": "number",
            "description": "ICache stall rate percentage",
        },
        "ICache-GL1 Read Bandwidth": {
            "type": "number",
            "description": "ICache to GL1 read bandwidth in bytes",
        },
        # DCache
        "Dcache Requests": {
            "type": "number",
            "description": "Number of scalar data cache requests",
        },
        "Dcache Utilization": {
            "type": "number",
            "description": "DCache utilization percentage",
        },
        "Dcache Hit Rate": {
            "type": "number",
            "description": "DCache hit rate percentage",
        },
        "Dcache Request Stall Rate": {
            "type": "number",
            "description": "DCache stall rate percentage",
        },
        "Dcache Request Bandwidth": {
            "type": "number",
            "description": "DCache request bandwidth in bytes",
        },
        "Dcache-GL1 Read Bandwidth": {
            "type": "number",
            "description": "DCache to GL1 read bandwidth in bytes",
        },
        # GL0
        "GL0 Utilization": {
            "type": "number",
            "description": "GL0 cache utilization percentage",
        },
        "GL0 Total Requests": {"type": "number", "description": "Total GL0 requests"},
        "GL0 Read Requests": {"type": "number", "description": "GL0 read requests"},
        "GL0 Write Requests": {"type": "number", "description": "GL0 write requests"},
        "GL0 Atomic Requests": {"type": "number", "description": "GL0 atomic requests"},
        "GL0 Hit Rate": {"type": "number", "description": "GL0 hit rate percentage"},
        "GL0 Request Latency": {
            "type": "number",
            "description": "GL0 request latency in cycles",
        },
        # LDS
        "LDS Utilization": {
            "type": "number",
            "description": "LDS utilization percentage",
        },
        "LDS Utilization - Load": {
            "type": "number",
            "description": "LDS load utilization percentage",
        },
        "LDS Utilization - Store": {
            "type": "number",
            "description": "LDS store utilization percentage",
        },
        "LDS Bank Conflict Stall Rate": {
            "type": "number",
            "description": "LDS bank conflict stall percentage",
        },
        "LDS Load Requests": {"type": "number", "description": "LDS load requests"},
        "LDS Store Requests": {"type": "number", "description": "LDS store requests"},
        "LDS Atomic Requests": {"type": "number", "description": "LDS atomic requests"},
        "Async Load Requests": {"type": "number", "description": "Async load requests"},
        "Async Store Requests": {
            "type": "number",
            "description": "Async store requests",
        },
        "LDS Load Bandwidth": {
            "type": "number",
            "description": "LDS load bandwidth in bytes",
        },
        "LDS Store Bandwidth": {
            "type": "number",
            "description": "LDS store bandwidth in bytes",
        },
        "LDS Atomic Bandwidth": {
            "type": "number",
            "description": "LDS atomic bandwidth in bytes",
        },
        "Async Load Bandwidth": {
            "type": "number",
            "description": "Async load bandwidth in bytes",
        },
        "Async Store Bandwidth": {
            "type": "number",
            "description": "Async store bandwidth in bytes",
        },
        "TDM Load Bandwidth": {
            "type": "number",
            "description": "TDM load bandwidth in bytes",
        },
        "TDM Store Bandwidth": {
            "type": "number",
            "description": "TDM store bandwidth in bytes",
        },
        "TDM Load Requests": {"type": "number", "description": "TDM load requests"},
        "TDM Store Requests": {"type": "number", "description": "TDM store requests"},
        # GL0-GL1
        "GL0-GL1 Read Requests": {
            "type": "number",
            "description": "GL0 to GL1 read requests",
        },
        "GL0-GL1 Write Requests": {
            "type": "number",
            "description": "GL0 to GL1 write requests",
        },
        "GL0-GL1 Read Bandwidth": {
            "type": "number",
            "description": "GL0 to GL1 read bandwidth in bytes",
        },
        "GL0-GL1 Write Bandwidth": {
            "type": "number",
            "description": "GL0 to GL1 write bandwidth in bytes",
        },
        "GL0-GL1 Atomic Bandwidth": {
            "type": "number",
            "description": "GL0 to GL1 atomic bandwidth in bytes",
        },
        "GL1A Utilization": {
            "type": "number",
            "description": "GL1A utilization percentage",
        },
        "GL1C Utilization": {
            "type": "number",
            "description": "GL1C utilization percentage",
        },
        "GL1C Buffer Full Stall": {
            "type": "number",
            "description": "GL1C buffer full stall percentage",
        },
        # GLARB
        "GLARBA Utilization": {
            "type": "number",
            "description": "GLARBA utilization percentage",
        },
        "GLARBC Utilization": {
            "type": "number",
            "description": "GLARBC utilization percentage",
        },
        "GLARBC Buffer Full Stall": {
            "type": "number",
            "description": "GLARBC buffer full stall percentage",
        },
        "GL1-GLARB Read Bandwidth": {
            "type": "number",
            "description": "GL1 to GLARB read bandwidth in bytes",
        },
        "GL1-GLARB Write Bandwidth": {
            "type": "number",
            "description": "GL1 to GLARB write bandwidth in bytes",
        },
        "GLARB-GL2 Read Bandwidth": {
            "type": "number",
            "description": "GLARB to GL2 read bandwidth in bytes",
        },
        "GLARB-GL2 Write Bandwidth": {
            "type": "number",
            "description": "GLARB to GL2 write bandwidth in bytes",
        },
        "GLARB-GL2 Atomic Bandwidth": {
            "type": "number",
            "description": "GLARB to GL2 atomic bandwidth in bytes",
        },
        # GL2
        "GL2A Utilization": {
            "type": "number",
            "description": "GL2A utilization percentage",
        },
        "GL2C Hit Rate": {"type": "number", "description": "GL2C hit rate percentage"},
        "GL2-EA Read Bandwidth": {
            "type": "number",
            "description": "GL2 to EA read bandwidth in bytes",
        },
        "GL2-EA Write Bandwidth": {
            "type": "number",
            "description": "GL2 to EA write bandwidth in bytes",
        },
        "GL2-EA Atomic Bandwidth": {
            "type": "number",
            "description": "GL2 to EA atomic bandwidth in bytes",
        },
        # EA/HBM
        "EA Utilization": {
            "type": "number",
            "description": "EA utilization percentage",
        },
        "EA Stall Rate": {"type": "number", "description": "EA stall rate percentage"},
        "DRAM Read Bandwidth": {
            "type": "number",
            "description": "DRAM read bandwidth in bytes",
        },
        "DRAM Write Bandwidth": {
            "type": "number",
            "description": "DRAM write bandwidth in bytes",
        },
        "IO Read Bandwidth": {
            "type": "number",
            "description": "IO read bandwidth in bytes",
        },
        "IO Write Bandwidth": {
            "type": "number",
            "description": "IO write bandwidth in bytes",
        },
        "HDM Read Bandwidth": {
            "type": "number",
            "description": "HDM (CXL) read bandwidth in bytes",
        },
        "HDM Write Bandwidth": {
            "type": "number",
            "description": "HDM (CXL) write bandwidth in bytes",
        },
        "GMI Read Bandwidth": {
            "type": "number",
            "description": "GMI read bandwidth in bytes",
        },
        "GMI Write Bandwidth": {
            "type": "number",
            "description": "GMI write bandwidth in bytes",
        },
    },
    "required": [],  # All metrics are optional, missing values show as N/A
}


def get_metrics_schema() -> dict[str, Any]:
    """Return the JSON schema for metrics input"""
    return METRICS_SCHEMA


def validate_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """
    Validate metrics against the schema.

    Returns:
        Dict with 'valid' (bool), 'errors' (list), 'warnings' (list)
    """
    errors = []
    warnings = []

    if not isinstance(metrics, dict):
        return {
            "valid": False,
            "errors": ["Metrics must be a dictionary"],
            "warnings": [],
        }

    schema_props = METRICS_SCHEMA.get("properties", {})

    for key, value in metrics.items():
        if key not in schema_props:
            warnings.append(f"Unknown metric: '{key}'")
        elif value is not None and not isinstance(value, (int, float)):
            errors.append(
                f"Metric '{key}' must be a number, got {type(value).__name__}"
            )

    # Check for percentage values in valid range
    pct_metrics = [
        k for k in metrics.keys() if "Rate" in k or "Utilization" in k or "Hit" in k
    ]
    for key in pct_metrics:
        val = metrics.get(key)
        if val is not None and isinstance(val, (int, float)) and (val < 0 or val > 100):
            warnings.append(
                f"Metric '{key}' value {val} is outside expected range [0, 100]"
            )

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


def render_mem_chart(
    metrics: dict[str, Any],
    arch: str = "gfx1250",
    normal_unit: str = "per_kernel",
    output_format: str = "text",
    compact: bool = False,
    show_debug: bool = False,
    width: int = 220,
) -> str:
    """
    Render the memory chart and return as a string.

    This is the main MCP-callable function for rendering the memory chart.

    Args:
        metrics: Dictionary of metric name -> value
        arch: Architecture name (default: 'gfx1250')
        normal_unit: Normalization unit (default: 'per_kernel')
        output_format: Output format - 'text', 'text_plain' (no colors), or 'svg'
        compact: If True, hide edges with zero/None values
        show_debug: If True, show debug info (block coordinates)
        width: Console width for rendering (default: 220)

    Returns:
        String representation of the diagram (with ANSI codes, plain text, or SVG)
    """
    from io import StringIO

    # FakeFile to force Rich to respect width parameter
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

    if output_format == "svg":
        # Use Rich's built-in SVG export
        svg_console = Console(
            file=StringIO(), force_terminal=True, width=width, height=100, record=True
        )
        create_mem_chart_diagram(
            arch,
            normal_unit,
            metrics,
            svg_console,
            show_debug=show_debug,
            compact=compact,
        )
        return svg_console.export_svg(title="gfx1250 Memory Chart")

    # Text output
    fake_file = FakeFile()
    no_color = output_format == "text_plain"
    console = Console(
        file=fake_file, force_terminal=True, width=width, height=100, no_color=no_color
    )
    create_mem_chart_diagram(
        arch, normal_unit, metrics, console, show_debug=show_debug, compact=compact
    )
    output = fake_file.getvalue()

    if output_format == "text_plain":
        # Strip ANSI escape codes
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        output = ansi_escape.sub("", output)

    return output


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="gfx1250 Memory Chart - CLI Visualization"
    )
    parser.add_argument("--data", "-d", help="JSON file with metrics data")
    parser.add_argument(
        "--debug", action="store_true", help="Show debug info (block coordinates)"
    )
    parser.add_argument("--arch", default="gfx1250", help="Architecture name")
    parser.add_argument("--norm", default="per_kernel", help="Normalization unit")
    parser.add_argument("--txt", "-t", help="Output to plain text file (no colors)")
    parser.add_argument("--svg", help="Output to SVG image file (with colors)")
    parser.add_argument(
        "--compact",
        "-c",
        action="store_true",
        help="Compact mode: hide edges with zero values",
    )
    args = parser.parse_args()

    # Load or use default data
    if args.data:
        metric_dict = load_metrics_from_json(args.data)
    else:
        # Default sample data (matches 0300_Memory_Chart.yaml metric names)
        metric_dict = {
            # Table 301: Instruction Cache
            "ICache Requests": 450,
            "ICache Utilization": 45.2,
            "ICache Hit Rate": 98.5,
            "ICache Request Stall Rate": 2.1,
            "ICache-GL1 Read Bandwidth": 57600000,  # 450 * 128 bytes
            # Table 302: Scalar Data Cache
            "Dcache Requests": 225,
            "Dcache Utilization": 38.7,
            "Dcache Hit Rate": 95.3,
            "Dcache Request Stall Rate": 1.8,
            "Dcache Request Bandwidth": 134217728,
            "Dcache-GL1 Read Bandwidth": 28800000,  # 225 * 128 bytes
            # Table 303: GL0 Cache
            "GL0 Utilization": 72.4,
            "GL0 Total Requests": 1250000,
            "GL0 Read Requests": 875000,
            "GL0 Write Requests": 375000,
            "GL0 Atomic Requests": 62500,
            "GL0 Hit Rate": 87.2,
            "GL0 Request Latency": 42.5,
            # Table 304: LDS
            "LDS Utilization": 65.8,
            "LDS Utilization - Load": 42.3,
            "LDS Utilization - Store": 23.5,
            "LDS Bank Conflict Stall Rate": 5.2,
            "LDS Address Conflict Stall Rate": 2.1,
            # LDS Request Counts
            "LDS Load Requests": 1250,
            "LDS Store Requests": 625,
            "LDS Atomic Requests": 1250,
            "Async Load Requests": 250,
            "Async Store Requests": 1000,
            # LDS Bandwidth
            "LDS Load Bandwidth": 268738355200,
            "LDS Store Bandwidth": 134247628800,
            "LDS Atomic Bandwidth": 16000000000,
            "Async Load Bandwidth": 50000000000,  # Async LDS to global memory
            "Async Store Bandwidth": 25000000000,
            "TDM Load Bandwidth": 80000000000,  # TDM to global memory
            "TDM Store Bandwidth": 40000000000,
            "TDM Load Requests": 5000,
            "TDM Store Requests": 2500,
            # Table 305: GL0-GL1 Interface
            "GL0-GL1 Read Requests": 125000,
            "GL0-GL1 Write Requests": 62500,
            "GL0-GL1 Read Bandwidth": 412850380800,
            "GL0-GL1 Write Bandwidth": 206425190400,
            "GL0-GL1 Atomic Bandwidth": 16000000000,  # Atomic requests * 128 bytes
            "GL0-GL1 Read Latency": 85.2,
            "GL0-GL1 Write Latency": 62.4,
            "GL0-GL1 Request Stall Rate": 8.5,
            "GL1A Utilization": 65.2,
            "GL1C Utilization": 72.8,
            "GL1C Buffer Full Stall": 12.5,
            # Table 306: GLARB
            "GLARBA Utilization": 52.3,
            "GLARBA Request Path Utilization": 48.7,
            "GLARBA Return Path Utilization": 45.2,
            "GLARBC Utilization": 58.6,
            "GLARBC Buffer Full Stall": 12.4,
            "GL1-GLARB Read Bandwidth": 412.8e9,  # 412.8 GB/s
            "GL1-GLARB Write Bandwidth": 156.3e9,  # 156.3 GB/s
            "GLARB-GL2 Read Bandwidth": 380.5e9,  # 380.5 GB/s
            "GLARB-GL2 Write Bandwidth": 142.8e9,  # 142.8 GB/s
            # 28.6 GB/s (atomic traffic is typically lower)
            "GLARB-GL2 Atomic Bandwidth": 28.6e9,
            # Table 307: GL2 Cache
            "GL2A Utilization": 74.2,
            "GL2C Hit Rate": 82.5,
            "GL2-EA Read Bandwidth": 485.6e9,  # 485.6 GB/s
            "GL2-EA Write Bandwidth": 362.4e9,  # 362.4 GB/s
            "GL2-EA Atomic Bandwidth": 25.2e9,  # 25.2 GB/s (atomic traffic to DRAM)
            # Table 308: EA to HBM
            "EA Utilization": 65.8,
            "DRAM Read Bandwidth": 512e9,  # 512 GB/s in Bytes
            "DRAM Write Bandwidth": 384e9,  # 384 GB/s in Bytes
            "IO Read Bandwidth": 32e9,  # 32 GB/s in Bytes
            "IO Write Bandwidth": 24e9,  # 24 GB/s in Bytes
            "HDM Read Bandwidth": 128e9,  # 128 GB/s in Bytes
            "HDM Write Bandwidth": 96e9,  # 96 GB/s in Bytes
            "GMI Read Bandwidth": 256e9,  # 256 GB/s in Bytes
            "GMI Write Bandwidth": 192e9,  # 192 GB/s in Bytes
            "Outstanding Requests": 48.5,
            "EA Stall Rate": 15.2,
        }

    # Create diagram - use FakeFile to force Rich to respect width parameter
    # (StringIO.isatty() returns False, causing Rich to ignore width)
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

    fake_file = FakeFile()

    # If outputting to file, use no_color to strip ANSI codes
    if args.txt:
        console = Console(
            file=fake_file, force_terminal=True, width=220, height=100, no_color=True
        )
    else:
        console = Console(file=fake_file, force_terminal=True, width=220, height=100)

    create_mem_chart_diagram(
        args.arch,
        args.norm,
        metric_dict,
        console,
        show_debug=args.debug,
        compact=args.compact,
    )
    output = fake_file.getvalue()

    # Strip ANSI escape codes for file outputs

    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    plain_text = ansi_escape.sub("", output)

    if args.svg:
        # Use Rich's built-in SVG export with colors
        from io import StringIO

        svg_console = Console(
            file=StringIO(), force_terminal=True, width=220, height=100, record=True
        )
        create_mem_chart_diagram(
            args.arch,
            args.norm,
            metric_dict,
            svg_console,
            show_debug=args.debug,
            compact=args.compact,
        )
        svg_output = svg_console.export_svg(title="gfx1250 Memory Chart")
        with open(args.svg, "w", encoding="utf-8") as f:
            f.write(svg_output)
        print(f"SVG saved to {args.svg}")
    elif args.txt:
        # Write plain text to file
        with open(args.txt, "w", encoding="utf-8") as f:
            f.write(plain_text)
        print(f"Output written to {args.txt}")
    else:
        print(output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
##############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
##############################################################################

"""
RDNA3.5 (gfx1150/Strix/Halo) Memory Architecture Diagram - CLI Visualization
=============================================================================

USAGE:
    python mem_chart_gfx11.py [--data metrics.json] [--debug] [--txt file.txt] [--svg file.svg]

API:
    plot_mem_chart(arch, normal_unit, metric_dict) -> str

RDNA3.5 MEMORY HIERARCHY:
   Kernel -> SQC (ICache/DCache) -> GL1C (L1) -> GL2C (L2) -> GCEA -> System Memory
         -> TCP (L0 Vector Cache)
         -> LDS (Local Data Share)
"""

import os
os.environ['COLUMNS'] = '200'

import argparse
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


COLORS = {
    'kernel': 'green',
    'block': 'blue',
    'read': 'bright_cyan',
    'write': 'bright_yellow',
    'atomic': 'bright_magenta',
    'util': 'bright_green',
    'hit': 'yellow',
    'stall': 'indian_red',
    'bw': 'bright_cyan',
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
    edges: List[Edge] = field(default_factory=list)
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
    attributes: List[str] = field(default_factory=list)
    y_offset: int = 0
    height: int = 5
    show_border: bool = True
    vertical_position: str = "middle"
    border_color: str = "blue"


@dataclass
class RegularBlock(RectBlock):
    sub_blocks: List[SubBlock] = field(default_factory=list)
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
                # Inner content width: box_width - 2 (for │ borders) - 2 (space after left │ and before right │)
                inner_width = box_width - 4
                bc = sub.border_color
                top_line = f"[{bc}]┌" + "─" * (box_width - 2) + f"┐[/{bc}]"
                bottom_line = f"[{bc}]└" + "─" * (box_width - 2) + f"┘[/{bc}]"
                temp_content.append(top_line)
                # Label line with right border
                label_clean = sub.label
                label_pad = " " * max(0, inner_width - len(label_clean))
                temp_content.append(f"[{bc}]│[/{bc}] [bold]{sub.label}[/bold]{label_pad} [{bc}]│[/{bc}]")
                for attr in sub.attributes:
                    if not attr:  # Skip empty attributes
                        continue
                    clean = re.sub(r'\[.*?\]', '', attr)
                    # Calculate padding to align right border
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
            height=self.height
        )


def format_value(value: Union[int, float, str, None], unit: str = '', precision: int = 1) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        try:
            value = float(value)
        except (ValueError, TypeError):
            return value
    if unit == '%':
        return f"{value:.{precision}f}%"
    elif unit == 'GB/s':
        return f"{value:.1f} GB/s"
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


def format_bw_gbps(value: Union[int, float, str, None], precision: int = 1) -> str:
    if value is None:
        return "N/A"
    try:
        value = float(value)
    except (ValueError, TypeError):
        return "N/A"
    gbps = value / 1e9
    if gbps >= 1000:
        return f"{gbps/1000:.{precision}f} TB/s"
    elif gbps >= 1:
        return f"{gbps:.{precision}f} GB/s"
    else:
        return f"{gbps*1000:.{precision}f} MB/s"


def get_metric(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    return d.get(key, default)


def metric_line(label: str, value: Any, unit: str = '%', color: str = 'bright_green') -> str:
    formatted = format_value(value, unit)
    return f"{label} [{color}]{formatted}[/{color}]"


def bar(pct: float, w: int = 10) -> str:
    if pct is None:
        return '░' * w
    try:
        pct = float(pct)
    except (ValueError, TypeError):
        return '░' * w
    filled = int(w * min(100, max(0, pct)) / 100)
    return '█' * filled + '░' * (w - filled)


def create_mem_chart_diagram(
    arch: str,
    normal_unit: str,
    metric_dict: Dict[str, Any],
    console: Console,
    show_debug: bool = False,
    compact: bool = False
) -> None:
    """Create the RDNA3.5 memory architecture diagram"""

    # Extract metrics
    icache_req = get_metric(metric_dict, 'ICache Requests')
    icache_hit = get_metric(metric_dict, 'ICache Hit Rate')
    icache_gl1_bw = get_metric(metric_dict, 'ICache-GL1 Read Bandwidth')

    dcache_req = get_metric(metric_dict, 'Dcache Requests')
    dcache_hit = get_metric(metric_dict, 'Dcache Hit Rate')
    dcache_gl1_bw = get_metric(metric_dict, 'Dcache-GL1 Read Bandwidth')

    tcp_read_req = get_metric(metric_dict, 'TCP Read Requests')
    tcp_write_req = get_metric(metric_dict, 'TCP Write Requests')
    tcp_hit = get_metric(metric_dict, 'TCP Hit Rate')
    tcp_bw = get_metric(metric_dict, 'TCP Request Bandwidth')

    lds_insts = get_metric(metric_dict, 'LDS Instructions')
    lds_bw = get_metric(metric_dict, 'LDS Estimated Bandwidth')
    lds_bank_conflict = get_metric(metric_dict, 'LDS Bank Conflict Rate')

    tcp_gl1_read_bw = get_metric(metric_dict, 'TCP-GL1 Read Bandwidth')
    tcp_gl1_write_bw = get_metric(metric_dict, 'TCP-GL1 Write Bandwidth')

    gl1c_util = get_metric(metric_dict, 'GL1C Utilization')
    gl1c_hit = get_metric(metric_dict, 'GL1C Hit Rate')
    gl1c_stall_gl2 = get_metric(metric_dict, 'GL1C Stall GL2 Backpressure')

    gl1_gl2_read_bw = get_metric(metric_dict, 'GL1-GL2 Read Bandwidth')
    gl1_gl2_write_bw = get_metric(metric_dict, 'GL1-GL2 Write Bandwidth')

    gl2c_util = get_metric(metric_dict, 'GL2C Utilization')
    gl2c_hit = get_metric(metric_dict, 'GL2C Hit Rate')
    gl2c_read_bw = get_metric(metric_dict, 'GL2C Read Bandwidth')
    gl2c_write_bw = get_metric(metric_dict, 'GL2C Write Bandwidth')

    sarb_util = get_metric(metric_dict, 'SARB Utilization')
    sarb_stall = get_metric(metric_dict, 'SARB Stall Rate')
    dram_read_bw = get_metric(metric_dict, 'DRAM Read Bandwidth', 0)
    dram_write_bw = get_metric(metric_dict, 'DRAM Write Bandwidth', 0)

    total_bw = (dram_read_bw or 0) + (dram_write_bw or 0)

    # Print header
    console.print()
    console.print(f"[bold]RDNA3.5 Memory Chart[/bold] [dim](Normalization: {normal_unit})[/dim]")
    console.print("|" + "-" * 68 + " [dim]GPU[/dim] " + "-" * 68 + "|" + "-" * 4 + " [dim]System Memory[/dim] " + "-" * 4 + "|")
    console.print()

    # Arrow constants
    std_arrow_len = 10
    std_arrow_left = "<" + "-" * std_arrow_len
    std_arrow_right = "-" * std_arrow_len + ">"

    kernel_edge_width = 18
    kernel_arrow_left = "<" + "-" * (kernel_edge_width - 1)
    kernel_arrow_right = "-" * (kernel_edge_width - 1) + ">"
    kernel_arrow_both = "<" + "-" * (kernel_edge_width - 2) + ">"

    def fmt_edge(label, value):
        label_str = f"{label:<8}"
        if value is not None:
            value_str = f": {format_sci(value):>8}"
        else:
            value_str = ""
        return f"{label_str}{value_str}"

    # Build blocks
    kernel = RegularBlock(
        label="Kernel",
        x_min=0, x_max=14, y_min=0, y_max=30,
        color=COLORS['kernel']
    )

    edges_kernel = AlignedEdgesGroup(
        label="Edges_Kernel",
        x_min=kernel.x_max + 2,
        x_max=kernel.x_max + 22,
        y_min=0, y_max=30,
        compact=compact,
        edges=[
            Edge(label=fmt_edge("TCP Rd", tcp_read_req), arrow=kernel_arrow_left, color=COLORS['read']),
            Edge(label=fmt_edge("TCP Wr", tcp_write_req), arrow=kernel_arrow_right, color=COLORS['write']),
            Edge(label=fmt_edge("LDS", lds_insts), arrow=kernel_arrow_both, color=COLORS['atomic']),
            Edge(label=fmt_edge("ICache", icache_req), arrow=kernel_arrow_left, color=COLORS['read']),
            Edge(label=fmt_edge("DCache", dcache_req), arrow=kernel_arrow_left, color=COLORS['read']),
        ]
    )

    tcp_block = RegularBlock(
        label="TCP/LDS/SQC",
        x_min=edges_kernel.x_max + 1,
        x_max=edges_kernel.x_max + 30,
        y_min=0, y_max=30,
        color=COLORS['block'],
        vertical_position="top",
        sub_blocks=[
            SubBlock(label="TCP (L0)", attributes=[
                metric_line("Hit Rate", tcp_hit, '%', COLORS['hit']),
                metric_line("BW", tcp_bw, 'GB/s', COLORS['bw']) if tcp_bw else "",
            ], height=7),
            SubBlock(label="LDS", attributes=[
                metric_line("BW", lds_bw, 'GB/s', COLORS['bw']) if lds_bw else "",
                metric_line("BankConf", lds_bank_conflict, '%', COLORS['stall']) if lds_bank_conflict else "",
            ], height=7),
            SubBlock(label="SQC", attributes=[
                metric_line("ICache", icache_hit, '%', COLORS['hit']),
                metric_line("DCache", dcache_hit, '%', COLORS['hit']),
            ], height=7)
        ]
    )

    edges_to_gl1 = AlignedEdgesGroup(
        label="Edges_GL1",
        x_min=tcp_block.x_max + 1,
        x_max=tcp_block.x_max + 12,
        y_min=0, y_max=20,
        top_padding=2,
        compact=compact,
        edges=[
            Edge(label=f"Read BW\n{format_bw_gbps(tcp_gl1_read_bw)}" if tcp_gl1_read_bw else "Read BW", arrow=std_arrow_left, color=COLORS['read']),
            Edge(label=f"Write BW\n{format_bw_gbps(tcp_gl1_write_bw)}" if tcp_gl1_write_bw else "Write BW", arrow=std_arrow_right, color=COLORS['write']),
        ]
    )

    gl1_block = RegularBlock(
        label="GL1C",
        x_min=edges_to_gl1.x_max + 1,
        x_max=edges_to_gl1.x_max + 16,
        y_min=0, y_max=30,
        color=COLORS['block'],
        vertical_position="top",
        sub_blocks=[
            SubBlock(label="", show_border=False, attributes=[
                metric_line("Util", gl1c_util, '%', COLORS['util']),
                f"[dim]{bar(gl1c_util)}[/dim]",
                "",
                metric_line("Hit Rate", gl1c_hit, '%', COLORS['hit']),
                f"[dim]{bar(gl1c_hit)}[/dim]",
                "",
                metric_line("GL2 Stall", gl1c_stall_gl2, '%', COLORS['stall']),
            ], height=12)
        ]
    )

    edges_gl1_gl2 = AlignedEdgesGroup(
        label="Edges_GL1_GL2",
        x_min=gl1_block.x_max + 1,
        x_max=gl1_block.x_max + 12,
        y_min=0, y_max=20,
        top_padding=2,
        compact=compact,
        edges=[
            Edge(label=f"Read BW\n{format_bw_gbps(gl1_gl2_read_bw)}" if gl1_gl2_read_bw else "Read BW", arrow=std_arrow_left, color=COLORS['read']),
            Edge(label=f"Write BW\n{format_bw_gbps(gl1_gl2_write_bw)}" if gl1_gl2_write_bw else "Write BW", arrow=std_arrow_right, color=COLORS['write']),
        ]
    )

    gl2_block = RegularBlock(
        label="GL2C",
        x_min=edges_gl1_gl2.x_max + 1,
        x_max=edges_gl1_gl2.x_max + 16,
        y_min=0, y_max=30,
        color=COLORS['block'],
        vertical_position="top",
        sub_blocks=[
            SubBlock(label="", show_border=False, attributes=[
                metric_line("Util", gl2c_util, '%', COLORS['util']),
                f"[dim]{bar(gl2c_util)}[/dim]",
                "",
                metric_line("Hit Rate", gl2c_hit, '%', COLORS['hit']),
                f"[dim]{bar(gl2c_hit)}[/dim]",
            ], height=10)
        ]
    )

    edges_gl2_gcea = AlignedEdgesGroup(
        label="Edges_GL2_GCEA",
        x_min=gl2_block.x_max + 1,
        x_max=gl2_block.x_max + 12,
        y_min=0, y_max=20,
        top_padding=2,
        compact=compact,
        edges=[
            Edge(label=f"Read BW\n{format_bw_gbps(gl2c_read_bw)}" if gl2c_read_bw else "Read BW", arrow=std_arrow_left, color=COLORS['read']),
            Edge(label=f"Write BW\n{format_bw_gbps(gl2c_write_bw)}" if gl2c_write_bw else "Write BW", arrow=std_arrow_right, color=COLORS['write']),
        ]
    )

    gcea_block = RegularBlock(
        label="GCEA",
        x_min=edges_gl2_gcea.x_max + 1,
        x_max=edges_gl2_gcea.x_max + 14,
        y_min=0, y_max=30,
        color=COLORS['block'],
        vertical_position="top",
        sub_blocks=[
            SubBlock(label="", show_border=False, attributes=[
                metric_line("SARB Util", sarb_util, '%', COLORS['util']),
                f"[dim]{bar(sarb_util)}[/dim]",
                "",
                metric_line("Stall", sarb_stall, '%', COLORS['stall']),
            ], height=8)
        ]
    )

    edges_to_dram = AlignedEdgesGroup(
        label="Edges_DRAM",
        x_min=gcea_block.x_max + 1,
        x_max=gcea_block.x_max + 12,
        y_min=0, y_max=20,
        top_padding=2,
        compact=compact,
        edges=[
            Edge(label=f"DRAM Rd\n{format_bw_gbps(dram_read_bw)}" if dram_read_bw else "DRAM Rd", arrow=std_arrow_left, color=COLORS['read']),
            Edge(label=f"DRAM Wr\n{format_bw_gbps(dram_write_bw)}" if dram_write_bw else "DRAM Wr", arrow=std_arrow_right, color=COLORS['write']),
        ]
    )

    dram_block = RegularBlock(
        label="DRAM",
        x_min=edges_to_dram.x_max + 1,
        x_max=edges_to_dram.x_max + 16,
        y_min=0, y_max=30,
        color=COLORS['block'],
        vertical_position="top",
        sub_blocks=[
            SubBlock(label="System Memory", show_border=False, attributes=[
                f"[dim]DDR5/LPDDR5[/dim]",
                "",
                f"Total: [bold bright_green]{format_bw_gbps(total_bw)}[/bold bright_green]",
            ], height=6)
        ]
    )

    # Build layout
    layout = Table.grid(padding=0)
    for _ in range(10):
        layout.add_column()

    kernel_panel = Panel(
        "\n" * 8 + "[dim]Shader Core[/dim]\n[dim]Wave Execution[/dim]",
        title=f"[bold {COLORS['kernel']}]Kernel[/bold {COLORS['kernel']}]",
        border_style=COLORS['kernel'],
        width=kernel.width,
        height=kernel.height
    )

    layout.add_row(
        kernel_panel,
        edges_kernel.render_text(),
        tcp_block.render(),
        edges_to_gl1.render_text(),
        gl1_block.render(),
        edges_gl1_gl2.render_text(),
        gl2_block.render(),
        edges_gl2_gcea.render_text(),
        gcea_block.render(),
        edges_to_dram.render_text(),
        dram_block.render()
    )

    console.print(layout)
    console.print()
    console.print(f"[dim]Legend:[/dim] [{COLORS['read']}]<----[/{COLORS['read']}] Read  [{COLORS['write']}]---->[/{COLORS['write']}] Write  [{COLORS['atomic']}]<--->[/{COLORS['atomic']}] Atomic  [{COLORS['util']}]█[/{COLORS['util']}] Util  [{COLORS['hit']}]█[/{COLORS['hit']}] Hit%  [{COLORS['stall']}]█[/{COLORS['stall']}] Stall")
    console.print()

    if show_debug:
        console.print("[dim]Block Coordinates:[/dim]")
        console.print(f"  Kernel: x({kernel.x_min}-{kernel.x_max}), y({kernel.y_min}-{kernel.y_max})")
        console.print(f"  TCP/LDS/SQC: x({tcp_block.x_min}-{tcp_block.x_max})")
        console.print(f"  GL1C: x({gl1_block.x_min}-{gl1_block.x_max})")
        console.print(f"  GL2C: x({gl2_block.x_min}-{gl2_block.x_max})")
        console.print(f"  GCEA: x({gcea_block.x_min}-{gcea_block.x_max})")
        console.print(f"  DRAM: x({dram_block.x_min}-{dram_block.x_max})")
        console.print()


def plot_mem_chart(arch: str, normal_unit: str, metric_dict: Dict[str, Any]) -> str:
    """Plot the memory chart and return as string."""
    class FakeFile:
        def __init__(self):
            self.data = []
        def write(self, s):
            self.data.append(s)
        def flush(self):
            pass
        def isatty(self):
            return True
        def getvalue(self):
            return ''.join(self.data)

    fake_file = FakeFile()
    console = Console(file=fake_file, force_terminal=True, width=200, height=80)
    create_mem_chart_diagram(arch, normal_unit, metric_dict, console, show_debug=False, compact=False)
    return fake_file.getvalue()


# Default sample metrics
DEFAULT_SAMPLE_METRICS = {
    'ICache Requests': 450,
    'ICache Utilization': 45.2,
    'ICache Hit Rate': 98.5,
    'ICache Miss Rate': 1.5,
    'ICache Request Stall Rate': 2.1,
    'ICache-GL1 Read Bandwidth': 57600000,
    'Dcache Requests': 225,
    'Dcache Utilization': 38.7,
    'Dcache Hit Rate': 95.3,
    'Dcache Request Stall Rate': 1.8,
    'Dcache-GL1 Read Bandwidth': 28800000,
    'TCP Total Requests': 1250000,
    'TCP Read Requests': 875000,
    'TCP Write Requests': 375000,
    'TCP Miss Requests': 150000,
    'TCP Hit Rate': 88.0,
    'TCP Request Bandwidth': 80.0,
    'LDS Instructions': 125000,
    'LDS Instruction Cycles': 250000,
    'LDS Wait Cycles': 12500,
    'LDS Bank Conflict Rate': 4.0,
    'LDS Estimated Bandwidth': 256.0,
    'TCP-GL1 Read Requests': 150000,
    'TCP-GL1 Write Requests': 50000,
    'TCP-GL1 Read Bandwidth': 96e9,
    'TCP-GL1 Write Bandwidth': 32e9,
    'GL1C Utilization': 65.2,
    'GL1C Total Requests': 200000,
    'GL1C Read Requests': 150000,
    'GL1C Write Requests': 50000,
    'GL1C Miss Requests': 30000,
    'GL1C Hit Rate': 85.0,
    'GL1C Starve Rate': 5.2,
    'GL1C Stall GL2 Backpressure': 8.5,
    'GL1-GL2 Read Requests': 30000,
    'GL1-GL2 Write Requests': 10000,
    'GL1-GL2 Read Bandwidth': 48e9,
    'GL1-GL2 Write Bandwidth': 16e9,
    'GL1-GL2 Read Latency': 85.2,
    'GL1-GL2 Write Latency': 62.4,
    'GL2C Utilization': 74.2,
    'GL2C Total Requests': 40000,
    'GL2C Read Requests': 30000,
    'GL2C Write Requests': 10000,
    'GL2C Atomic Requests': 1000,
    'GL2C Hit Rate': 82.5,
    'GL2C Read Bandwidth': 64e9,
    'GL2C Write Bandwidth': 24e9,
    'SARB Utilization': 52.3,
    'SARB Stall Rate': 12.4,
    'DRAM Read Requests': 25000,
    'DRAM Write Requests': 8000,
    'DRAM Read Bandwidth': 100e9,
    'DRAM Write Bandwidth': 60e9,
    'Read Returns': 25000,
    'Write Returns': 8000,
}


def get_sample_metrics() -> Dict[str, Any]:
    """Return sample metrics data for testing"""
    return DEFAULT_SAMPLE_METRICS.copy()


def main():
    parser = argparse.ArgumentParser(description='RDNA3.5 Memory Chart - CLI Visualization')
    parser.add_argument('--data', '-d', help='JSON file with metrics data')
    parser.add_argument('--debug', action='store_true', help='Show debug info')
    parser.add_argument('--arch', default='gfx1150', help='Architecture name')
    parser.add_argument('--norm', default='per_kernel', help='Normalization unit')
    parser.add_argument('--txt', '-t', help='Output to plain text file')
    parser.add_argument('--svg', help='Output to SVG file')
    parser.add_argument('--compact', '-c', action='store_true', help='Compact mode')
    args = parser.parse_args()

    # Load or use default data
    if args.data:
        with open(args.data, 'r') as f:
            metric_dict = json.load(f)
    else:
        metric_dict = DEFAULT_SAMPLE_METRICS.copy()

    # Create console
    class FakeFile:
        def __init__(self):
            self.data = []
        def write(self, s):
            self.data.append(s)
        def flush(self):
            pass
        def isatty(self):
            return True
        def getvalue(self):
            return ''.join(self.data)

    if args.txt:
        fake_file = FakeFile()
        console = Console(file=fake_file, force_terminal=True, width=200, height=80, no_color=True)
        create_mem_chart_diagram(args.arch, args.norm, metric_dict, console, args.debug, args.compact)
        import re
        output = fake_file.getvalue()
        plain = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', output)
        with open(args.txt, 'w') as f:
            f.write(plain)
        print(f"Output written to {args.txt}")
    elif args.svg:
        from io import StringIO
        svg_console = Console(file=StringIO(), force_terminal=True, width=200, height=80, record=True)
        create_mem_chart_diagram(args.arch, args.norm, metric_dict, svg_console, args.debug, args.compact)
        svg_output = svg_console.export_svg(title="RDNA3.5 Memory Chart")
        with open(args.svg, 'w') as f:
            f.write(svg_output)
        print(f"SVG saved to {args.svg}")
    else:
        fake_file = FakeFile()
        console = Console(file=fake_file, force_terminal=True, width=200, height=80)
        create_mem_chart_diagram(args.arch, args.norm, metric_dict, console, args.debug, args.compact)
        print(fake_file.getvalue())


if __name__ == "__main__":
    main()

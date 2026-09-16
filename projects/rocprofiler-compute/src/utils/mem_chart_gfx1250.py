# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""gfx1250 (CDNA5) memory chart renderer.

Renders the Instinct gfx1250 memory hierarchy as a Rich terminal diagram.
Layout: Kernel -> TCP(LDS+GL0)/SQC -> GL1 -> GLARB -> GL2 -> EA/DF -> HBM/IO/HDM/GMI

Architecture regions::

    |<--------- XCD (Compute Die) -------->|<---- AID (I/O Die) --->|
    Kernel -> SQC/TCP -> GL1 -> GLARB -> GL2 -> EA/DF -> HBM (DRAM)
                                                       -> IO  (PCIe)
                                                       -> HDM (CXL)
                                                       -> GMI (Multi-GPU)
"""

from typing import Any, Optional, Union

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from utils.mem_chart_common import (
    COLORS,
    build_bw_edges,
    build_cache_panel,
    build_legend,
    colored,
    format_edge,
    format_value,
    make_arrows,
    mem_chart_cli_main,
    metric_line,
    render_chart_to_string,
    safe_float,
)

# ---------------------------------------------------------------------------
# Metric keys — match 0300_Memory_Chart.yaml for gfx1250 (tables 301-308)
# ---------------------------------------------------------------------------
_MEM_CHART_DEFAULT_ROWS: tuple[tuple[str, Union[int, float]], ...] = (
    # Table 301: Instruction Cache
    ("ICache Requests", 450),
    ("ICache Utilization", 60.0),
    ("ICache Hit Rate", 98.5),
    ("ICache Latency", 20.0),
    ("ICache Request Stall Rate", 2.0),
    ("ICache-GL1 Read Bandwidth", 57.6e9),
    ("ICache-GL1 Request Latency", 50.0),
    # Table 302: Scalar Data Cache
    ("Dcache Requests", 225),
    ("Dcache Utilization", 50.0),
    ("Dcache Hit Rate", 95.3),
    ("Dcache Latency", 30.0),
    ("Dcache Request Stall Rate", 3.0),
    ("Dcache Request Bandwidth", 10e9),
    ("Dcache-GL1 Read Bandwidth", 28.8e9),
    ("Dcache-GL1 Request Latency", 60.0),
    # Table 303: GL0 Cache
    ("GL0 Utilization", 72.4),
    ("GL0 Read Requests", 875_000),
    ("GL0 Write Requests", 375_000),
    ("GL0 Atomic Requests", 62_500),
    ("GL0 Hit Rate", 87.2),
    # Table 304: LDS
    ("LDS Load Requests", 1_250),
    ("LDS Store Requests", 625),
    ("LDS Atomic Requests", 1_250),
    ("Async Load Requests", 250),
    ("Async Store Requests", 1_000),
    ("LDS Utilization", 65.8),
    ("LDS Bank Conflict Stall Rate", 5.2),
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
    ("GL0-GL1 Read Bandwidth", 412.8e9),
    ("GL0-GL1 Write Bandwidth", 206.4e9),
    ("GL0-GL1 Atomic Bandwidth", 16e9),
    ("GL1A Utilization", 65.2),
    ("GL1C Utilization", 72.8),
    ("GL1C Buffer Full Stall", 12.5),
    # Table 306: GLARB
    ("GLARBA Utilization", 52.3),
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
    ("DRAM Read Bandwidth", 512e9),
    ("DRAM Write Bandwidth", 384e9),
    ("IO Read Bandwidth", 32e9),
    ("IO Write Bandwidth", 24e9),
    ("HDM Read Bandwidth", 128e9),
    ("HDM Write Bandwidth", 96e9),
    ("GMI Read Bandwidth", 256e9),
    ("GMI Write Bandwidth", 192e9),
    ("EA Stall Rate", 15.2),
)

MEM_CHART_PANEL_METRIC_KEYS: tuple[str, ...] = tuple(
    k for k, _ in _MEM_CHART_DEFAULT_ROWS
)

DEFAULT_SAMPLE_METRICS: dict[str, Union[int, float]] = dict(_MEM_CHART_DEFAULT_ROWS)

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
_CONSOLE_WIDTH = 220

_KERNEL_W = 14
_TCP_W = 27
_SQC_W = 27
_GL1_W = 15
_GLARB_W = 15
_GL2_W = 15
_EA_DF_W = 15
_HBM_W = 16
_EXT_BLOCK_W = 16

_KERNEL_H = 65
_TCP_H = 43
_SQC_H = 17
_GL1_H = 65
_GLARB_H = 65
_GL2_H = 65
_EA_DF_H = 65
_HBM_H = 37
_EXT_BLOCK_H = 8

_ARROW_LEN = 10
_KERNEL_ARROW_LEN = 18


def _build_kernel_edge_column(
    entries: list[tuple[str, Union[int, float, None], str, str]],
    arrows: dict[str, str],
) -> Text:
    """Kernel -> TCP/SQC edge column with request-count labels."""
    lines: list[str] = []
    for label, value, arrow_key, color in entries:
        lines.append(colored(format_edge(label, value, width=8), color))
        lines.append(colored(arrows[arrow_key], color))
        lines.append("")
    return Text.from_markup("\n".join(lines))


def _build_lds_panel(
    m: dict[str, Any],
) -> Panel:
    """LDS sub-block panel (inside TCP)."""
    content = "\n".join([
        "",
        metric_line("Util", m.get("LDS Utilization"), "%"),
        metric_line(
            "Bank Conflict",
            m.get("LDS Bank Conflict Stall Rate"),
            "%",
            COLORS["stall"],
        ),
        "",
        metric_line(
            "Load BW",
            m.get("LDS Load Bandwidth"),
            "Bytes/s",
            COLORS["read"],
        ),
        metric_line(
            "Store BW",
            m.get("LDS Store Bandwidth"),
            "Bytes/s",
            COLORS["write"],
        ),
        metric_line(
            "Atomic BW",
            m.get("LDS Atomic Bandwidth"),
            "Bytes/s",
            COLORS["atomic"],
        ),
    ])
    return Panel(
        content,
        title=f"[bold {COLORS['block']}]LDS[/bold {COLORS['block']}]",
        border_style=COLORS["block"],
        width=_TCP_W - 4,
        height=12,
    )


def _build_gl0_panel(
    m: dict[str, Any],
) -> Panel:
    """GL0 sub-block panel (inside TCP)."""
    content = "\n".join([
        metric_line(
            "Util",
            m.get("GL0 Utilization"),
            "%",
        ),
        metric_line(
            "Hit Rate",
            m.get("GL0 Hit Rate"),
            "%",
            COLORS["hit"],
        ),
    ])
    return Panel(
        content,
        title=f"[bold {COLORS['block']}]GL0[/bold {COLORS['block']}]",
        border_style=COLORS["block"],
        width=_TCP_W - 4,
        height=8,
    )


def _build_tcp_panel(
    m: dict[str, Any],
) -> Panel:
    """TCP parent panel containing LDS and GL0 sub-panels."""
    lds = _build_lds_panel(m)
    gl0 = _build_gl0_panel(m)
    return Panel(
        Group(
            "[dim]TXA/TXD/TDM[/dim]",
            lds,
            gl0,
        ),
        title=f"[bold {COLORS['block']}]TCP[/bold {COLORS['block']}]",
        border_style=COLORS["block"],
        width=_TCP_W,
        height=_TCP_H,
    )


def _build_sqc_panel(
    m: dict[str, Any],
) -> Panel:
    """SQC panel with ICache and DCache sub-panels."""
    icache = Panel(
        metric_line(
            "Hit Rate",
            m.get("ICache Hit Rate"),
            "%",
            COLORS["hit"],
        ),
        title=(f"[bold {COLORS['block']}]Instruction Cache[/bold {COLORS['block']}]"),
        border_style=COLORS["block"],
        width=_SQC_W - 4,
        height=5,
    )
    dcache = Panel(
        metric_line(
            "Hit Rate",
            m.get("Dcache Hit Rate"),
            "%",
            COLORS["hit"],
        ),
        title=(f"[bold {COLORS['block']}]Scalar Data Cache[/bold {COLORS['block']}]"),
        border_style=COLORS["block"],
        width=_SQC_W - 4,
        height=5,
    )
    return Panel(
        Group(icache, dcache),
        title=f"[bold {COLORS['block']}]SQC[/bold {COLORS['block']}]",
        border_style=COLORS["block"],
        width=_SQC_W,
        height=_SQC_H,
    )


def _build_downstream(
    m: dict[str, Any],
) -> Table:
    """EA downstream: HBM at top, then IO/HDM/GMI stacked below."""
    dram_arrows = make_arrows(_ARROW_LEN)

    hbm_read_bw = m.get("DRAM Read Bandwidth")
    hbm_write_bw = m.get("DRAM Write Bandwidth")
    numeric_hbm_read_bw = safe_float(hbm_read_bw)
    numeric_hbm_write_bw = safe_float(hbm_write_bw)
    total_bw = (
        numeric_hbm_read_bw + numeric_hbm_write_bw
        if numeric_hbm_read_bw is not None and numeric_hbm_write_bw is not None
        else None
    )

    dram_edges = build_bw_edges(
        [
            ("DRAM Read", hbm_read_bw, "left", COLORS["read"]),
            ("DRAM Write", hbm_write_bw, "right", COLORS["write"]),
        ],
        dram_arrows,
    )

    hbm_panel = Panel(
        f"Total\n[bold bright_green]"
        f"{format_value(total_bw, 'Bytes/s')}"
        f"[/bold bright_green]",
        title=(f"[bold {COLORS['block']}]HBM[/bold {COLORS['block']}]"),
        border_style=COLORS["block"],
        width=_HBM_W,
        height=_HBM_H,
    )

    # Long arrows spanning DRAM-edge + HBM width
    ext_arrow_len = _ARROW_LEN + _HBM_W + 1
    ext_arrows = make_arrows(ext_arrow_len)

    def build_ext_section(
        dest: str,
        read_key: str,
        write_key: str,
        description: str = "",
    ) -> tuple[Text, Panel]:
        read_bw = m.get(read_key)
        write_bw = m.get(write_key)

        read_label = f"{dest} Read".rjust(ext_arrow_len)
        read_val = format_value(
            read_bw,
            "Bytes/s",
        ).rjust(ext_arrow_len)
        write_label = f"{dest} Write".rjust(ext_arrow_len)
        write_val = format_value(
            write_bw,
            "Bytes/s",
        ).rjust(ext_arrow_len)

        edge_lines: list[str] = [
            colored(read_label, COLORS["read"]),
            colored(read_val, COLORS["read"]),
            colored(ext_arrows["left"], COLORS["read"]),
            "",
            colored(write_label, COLORS["write"]),
            colored(write_val, COLORS["write"]),
            colored(ext_arrows["right"], COLORS["write"]),
        ]
        edge_text = Text.from_markup("\n".join(edge_lines))

        panel = Panel(
            f"[dim]{description}[/dim]" if description else "",
            title=(f"[bold {COLORS['block']}]{dest}[/bold {COLORS['block']}]"),
            border_style=COLORS["block"],
            width=_EXT_BLOCK_W,
            height=_EXT_BLOCK_H,
        )
        return edge_text, panel

    io_edge, io_panel = build_ext_section(
        "IO",
        "IO Read Bandwidth",
        "IO Write Bandwidth",
        "PCIe and\nother I/O",
    )
    hdm_edge, hdm_panel = build_ext_section(
        "HDM",
        "HDM Read Bandwidth",
        "HDM Write Bandwidth",
        "CXL",
    )
    gmi_edge, gmi_panel = build_ext_section(
        "GMI",
        "GMI Read Bandwidth",
        "GMI Write Bandwidth",
    )

    grid = Table.grid(padding=0)
    grid.add_column(no_wrap=True)
    grid.add_column(no_wrap=True)

    # HBM row
    hbm_row = Table.grid(padding=0)
    hbm_row.add_column(no_wrap=True)
    hbm_row.add_column(no_wrap=True)
    hbm_row.add_row(dram_edges, hbm_panel)
    grid.add_row(hbm_row, "")

    grid.add_row("", "")

    # IO, HDM, GMI rows
    grid.add_row(io_edge, io_panel)
    grid.add_row(hdm_edge, hdm_panel)
    grid.add_row(gmi_edge, gmi_panel)

    return grid


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_mem_chart_metrics(
    metric_dict: dict[str, Any],
) -> dict[str, Any]:
    """Filter to known gfx1250 metric keys, preserving order."""
    return {k: metric_dict.get(k) for k in MEM_CHART_PANEL_METRIC_KEYS}


def get_sample_metrics() -> dict[str, Any]:
    """Sample metrics for testing or demos."""
    return normalize_mem_chart_metrics(DEFAULT_SAMPLE_METRICS.copy())


# ---------------------------------------------------------------------------
# Main diagram
# ---------------------------------------------------------------------------


def create_mem_chart_diagram(
    metric_dict: dict[str, Any],
    console: Console,
    show_debug: bool = False,
    chart_title: str = "",
    gpu_arch: Optional[str] = None,
) -> None:
    """Render the gfx1250 memory diagram to *console*.

    ``show_debug`` and ``gpu_arch`` match the shared CLI scaffold used by
    gfx9/gfx11; gfx1250 does not branch on either today.
    """
    del show_debug, gpu_arch
    m = metric_dict

    # --- Heading ---
    if chart_title:
        console.print(chart_title)
    console.print()
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

    # --- Arrows ---
    kernel_arrows = make_arrows(_KERNEL_ARROW_LEN)
    std_arrows = make_arrows(_ARROW_LEN)

    # --- Kernel panel ---
    kernel_panel = Panel(
        "\n" * (_KERNEL_H // 2 - 4)
        + "[dim]Shader Core[/dim]\n"
        + "[dim]Wave[/dim]\n"
        + "[dim]Execution[/dim]",
        title=(f"[bold {COLORS['kernel']}]Kernel[/bold {COLORS['kernel']}]"),
        border_style=COLORS["kernel"],
        width=_KERNEL_W,
        height=_KERNEL_H,
    )

    # --- Kernel -> TCP/SQC edges ---
    color_read = COLORS["read"]
    color_write = COLORS["write"]
    color_atomic = COLORS["atomic"]

    lds_edges = _build_kernel_edge_column(
        [
            ("Load", m.get("LDS Load Requests"), "left", color_read),
            ("Store", m.get("LDS Store Requests"), "right", color_write),
            ("Atomic", m.get("LDS Atomic Requests"), "both", color_atomic),
            ("Async Ld", m.get("Async Load Requests"), "left", color_read),
            ("Async St", m.get("Async Store Requests"), "right", color_write),
            ("TDM Ld", m.get("TDM Load Requests"), "left", color_read),
            ("TDM St", m.get("TDM Store Requests"), "right", color_write),
        ],
        kernel_arrows,
    )

    gl0_edges = _build_kernel_edge_column(
        [
            ("Read", m.get("GL0 Read Requests"), "left", color_read),
            ("Write", m.get("GL0 Write Requests"), "right", color_write),
            ("Atomic", m.get("GL0 Atomic Requests"), "both", color_atomic),
        ],
        kernel_arrows,
    )

    icache_edge = _build_kernel_edge_column(
        [("ICACHE", m.get("ICache Requests"), "left", color_read)],
        kernel_arrows,
    )

    smem_edge = _build_kernel_edge_column(
        [("SMEM", m.get("Dcache Requests"), "left", color_read)],
        kernel_arrows,
    )

    edge_col_width = _KERNEL_ARROW_LEN + 1
    edges_col = Table.grid()
    edges_col.add_column(width=edge_col_width)
    edges_col.add_row("")
    edges_col.add_row(
        Text("Request", style="bold white", justify="center"),
    )
    edges_col.add_row("")
    edges_col.add_row("")
    edges_col.add_row(lds_edges)
    for _ in range(6):
        edges_col.add_row("")
    edges_col.add_row(gl0_edges)
    for _ in range(5):
        edges_col.add_row("")
    edges_col.add_row(icache_edge)
    edges_col.add_row("")
    edges_col.add_row("")
    edges_col.add_row("")
    edges_col.add_row(smem_edge)

    # --- TCP / SQC column ---
    tcp_panel = _build_tcp_panel(m)
    sqc_panel = _build_sqc_panel(m)

    blocks_col = Table.grid()
    blocks_col.add_column()
    blocks_col.add_row(tcp_panel)
    blocks_col.add_row(sqc_panel)

    # --- TCP/SQC -> GL1 edges ---
    lds_gl1_edges = build_bw_edges(
        [
            ("Async Load", m.get("Async Load Bandwidth"), "left", color_read),
            ("Async Store", m.get("Async Store Bandwidth"), "right", color_write),
            ("TDM Load", m.get("TDM Load Bandwidth"), "left", color_read),
            ("TDM Store", m.get("TDM Store Bandwidth"), "right", color_write),
        ],
        std_arrows,
    )

    gl0_gl1_edges = build_bw_edges(
        [
            ("Read BW", m.get("GL0-GL1 Read Bandwidth"), "left", color_read),
            ("Write BW", m.get("GL0-GL1 Write Bandwidth"), "right", color_write),
            ("Atomic BW", m.get("GL0-GL1 Atomic Bandwidth"), "both", color_atomic),
        ],
        std_arrows,
    )

    icache_gl1_edge = build_bw_edges(
        [("Read BW", m.get("ICache-GL1 Read Bandwidth"), "left", color_read)],
        std_arrows,
    )

    dcache_gl1_edge = build_bw_edges(
        [("Read BW", m.get("Dcache-GL1 Read Bandwidth"), "left", color_read)],
        std_arrows,
    )

    edges_to_gl1 = Table.grid()
    edges_to_gl1.add_column()
    for _ in range(7):
        edges_to_gl1.add_row("")
    edges_to_gl1.add_row(lds_gl1_edges)
    edges_to_gl1.add_row("")
    edges_to_gl1.add_row(gl0_gl1_edges)
    for _ in range(4):
        edges_to_gl1.add_row("")
    edges_to_gl1.add_row(icache_gl1_edge)
    edges_to_gl1.add_row(dcache_gl1_edge)

    # --- GL1 ---
    gl1_panel = build_cache_panel(
        "GL1",
        [
            (
                "GL1A Util",
                m.get("GL1A Utilization"),
                "%",
                COLORS["util"],
            ),
            (
                "GL1C Util",
                m.get("GL1C Utilization"),
                "%",
                COLORS["util"],
            ),
            (
                "Buf Stall",
                m.get("GL1C Buffer Full Stall"),
                "%",
                COLORS["stall"],
            ),
        ],
        _GL1_W,
        _GL1_H,
    )

    # --- GL1 -> GLARB edges ---
    gl1_glarb_edges = build_bw_edges(
        [
            ("Read BW", m.get("GL1-GLARB Read Bandwidth"), "left", color_read),
            ("Write BW", m.get("GL1-GLARB Write Bandwidth"), "right", color_write),
        ],
        std_arrows,
    )

    # --- GLARB ---
    glarb_panel = build_cache_panel(
        "GLARB",
        [
            (
                "GLARBA Util",
                m.get("GLARBA Utilization"),
                "%",
                COLORS["util"],
            ),
            (
                "GLARBC Util",
                m.get("GLARBC Utilization"),
                "%",
                COLORS["util"],
            ),
            (
                "Buf Stall",
                m.get("GLARBC Buffer Full Stall"),
                "%",
                COLORS["stall"],
            ),
        ],
        _GLARB_W,
        _GLARB_H,
    )

    # --- GLARB -> GL2 edges ---
    glarb_gl2_edges = build_bw_edges(
        [
            ("Read BW", m.get("GLARB-GL2 Read Bandwidth"), "left", color_read),
            ("Write BW", m.get("GLARB-GL2 Write Bandwidth"), "right", color_write),
            ("Atomic BW", m.get("GLARB-GL2 Atomic Bandwidth"), "both", color_atomic),
        ],
        std_arrows,
    )

    # --- GL2 ---
    gl2_panel = build_cache_panel(
        "GL2",
        [
            (
                "GL2A Util",
                m.get("GL2A Utilization"),
                "%",
                COLORS["util"],
            ),
            (
                "GL2C Hit",
                m.get("GL2C Hit Rate"),
                "%",
                COLORS["hit"],
            ),
        ],
        _GL2_W,
        _GL2_H,
    )

    # --- GL2 -> EA/DF edges ---
    gl2_df_edges = build_bw_edges(
        [
            ("Read BW", m.get("GL2-EA Read Bandwidth"), "left", color_read),
            ("Write BW", m.get("GL2-EA Write Bandwidth"), "right", color_write),
            ("Atomic BW", m.get("GL2-EA Atomic Bandwidth"), "both", color_atomic),
        ],
        std_arrows,
    )

    # --- EA/DF ---
    ea_df_panel = build_cache_panel(
        "EA/DF",
        [
            (
                "EA Util",
                m.get("EA Utilization"),
                "%",
                COLORS["util"],
            ),
            (
                "Stall",
                m.get("EA Stall Rate"),
                "%",
                COLORS["stall"],
            ),
        ],
        _EA_DF_W,
        _EA_DF_H,
    )

    # --- Downstream (HBM + IO/HDM/GMI) ---
    downstream = _build_downstream(m)

    # --- Assemble layout ---
    layout = Table.grid(padding=0)
    for _ in range(12):
        layout.add_column()

    layout.add_row(
        kernel_panel,
        edges_col,
        blocks_col,
        edges_to_gl1,
        gl1_panel,
        gl1_glarb_edges,
        glarb_panel,
        glarb_gl2_edges,
        gl2_panel,
        gl2_df_edges,
        ea_df_panel,
        downstream,
    )

    console.print(layout)
    console.print()
    console.print(build_legend(include_stall=True))
    console.print()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_mem_chart(
    metric_dict: dict[str, Any],
    *,
    chart_title: str,
    gpu_arch: Optional[str] = None,
) -> str:
    """Render the gfx1250 memory chart and return as string."""
    return render_chart_to_string(
        create_mem_chart_diagram,
        metric_dict,
        normalize_mem_chart_metrics,
        console_width=_CONSOLE_WIDTH,
        chart_title=chart_title,
        gpu_arch=gpu_arch,
    )


def main() -> None:
    """CLI entry point."""
    mem_chart_cli_main(
        "gfx1250 Memory Chart - CLI",
        create_mem_chart_diagram,
        normalize_mem_chart_metrics,
        DEFAULT_SAMPLE_METRICS,
        console_width=_CONSOLE_WIDTH,
    )


if __name__ == "__main__":
    main()

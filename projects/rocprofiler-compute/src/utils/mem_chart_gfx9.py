# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""CDNA memory chart renderer (MI200/MI300/MI350)."""

from typing import Any, Optional, Union

from rich.align import VerticalAlignMethod
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from utils.mem_chart_common import (
    COLORS,
    build_arch_notes,
    build_bw_edges,
    build_cache_panel,
    build_ip_block,
    build_kernel_panel,
    build_legend,
    colored,
    format_edge,
    format_scientific,
    format_value,
    make_arrows,
    mem_chart_cli_main,
    metric_line,
    pad_to,
    progress_bar,
    render_chart_to_string,
    stack_metrics,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MEM_CHART_DEFAULT_ROWS: tuple[tuple[str, Union[int, float, None]], ...] = (
    ("Wavefront Occupancy", 8),
    ("Wave Life", 4200),
    ("SALU", 1200),
    ("SMEM", 45),
    ("VALU", 3500),
    ("Matrix Ops", 800),
    ("VMEM", 220),
    ("LDS", 150),
    ("GWS", 0),
    ("BR", 90),
    ("VGPR", 64),
    ("SGPR", 32),
    ("LDS Allocation", 32768),
    ("Scratch Allocation", 0),
    ("Wavefronts", 16384),
    ("Workgroups", 256),
    ("Flat Read", 80),
    ("Flat Write", 20),
    ("Flat Atomic", 4),
    ("Buffer Read", 3000),
    ("Buffer Write", 400),
    ("Buffer Atomic", 8),
    ("LDS Req", 150),
    ("LDS Util", 45),
    ("LDS Latency", 28),
    ("LDS Read", None),
    ("LDS Write", None),
    ("LDS Atomic", None),
    ("VL1 Rd", 3200),
    ("VL1 Wr", 480),
    ("VL1 Atomic", 12),
    ("VL1 Hit", 92),
    ("VL1 Lat", 180),
    ("VL1 Coalesce", 87),
    ("VL1 Stall", 5),
    ("VL1_L2 Rd", 256),
    ("VL1_L2 Wr", 48),
    ("VL1_L2 Atomic", 12),
    ("VL1_L2 Read BW", 32e9),
    ("VL1_L2 Write BW", 3e9),
    ("VL1_L2 Atomic BW", 768e6),
    ("sL1D Rd", 45),
    ("sL1D Hit", 98),
    ("sL1D Lat", 85),
    ("sL1D_L2 Rd", 1),
    ("sL1D_L2 Wr", 0),
    ("sL1D_L2 Atomic", 0),
    ("sL1D_L2 Read BW", 64e6),
    ("IL1 Fetch", 32),
    ("IL1 Hit", 99),
    ("IL1 Lat", 42),
    ("IL1_L2 Rd", 1),
    ("IL1_L2 Read BW", 64e6),
    ("L2 Rd", 300),
    ("L2 Wr", 52),
    ("L2 Atomic", 12),
    ("L2 Hit", 85),
    ("L2 Rd Lat", 220),
    ("L2 Wr Lat", 180),
    ("Fabric_L2 Rd", 45),
    ("Fabric_L2 Wr", 8),
    ("Fabric_L2 Atomic", 1),
    ("L2-Fabric Read BW", 45e9),
    ("L2-Fabric Write and Atomic BW", 8e9),
    ("Fabric Rd Lat", 350),
    ("Fabric Wr Lat", 280),
    ("Fabric Atomic Lat", 310),
    ("HBM Rd", 42),
    ("HBM Wr", 7),
    ("HBM Read Traffic", None),
    ("HBM Write and Atomic Traffic", None),
    ("Remote Read Traffic", None),
    ("Remote Write and Atomic Traffic", None),
    ("HBM Read BW", None),
    ("HBM Write BW", None),
    ("HBM Atomic BW", None),
    ("xGMI Read BW", None),
    ("xGMI Write BW", None),
    ("xGMI Atomic BW", None),
    ("PCIe Read BW", None),
    ("PCIe Write BW", None),
    ("PCIe Atomic BW", None),
)

MEM_CHART_PANEL_METRIC_KEYS: tuple[str, ...] = tuple(
    k for k, _ in _MEM_CHART_DEFAULT_ROWS
)

DEFAULT_SAMPLE_METRICS: dict[str, Union[int, float, None]] = dict(
    _MEM_CHART_DEFAULT_ROWS
)


# ---------------------------------------------------------------------------
# Public API helpers
# ---------------------------------------------------------------------------


def normalize_mem_chart_metrics(metric_dict: dict[str, Any]) -> dict[str, Any]:
    """Filter/reorder input to panel key order; missing keys become None."""
    return {k: metric_dict.get(k) for k in MEM_CHART_PANEL_METRIC_KEYS}


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def _extract_metrics(metric_dict: dict[str, Any]) -> dict[str, Any]:
    """Extract rendered metrics from the flat dict. Missing keys → None."""
    metrics: dict[str, Any] = {}

    # Kernel→L1 request edges
    metrics["flat_read"] = metric_dict.get("Flat Read")
    metrics["flat_write"] = metric_dict.get("Flat Write")
    metrics["flat_atomic"] = metric_dict.get("Flat Atomic")
    metrics["buffer_read"] = metric_dict.get("Buffer Read")
    metrics["buffer_write"] = metric_dict.get("Buffer Write")
    metrics["buffer_atomic"] = metric_dict.get("Buffer Atomic")
    metrics["lds_req"] = metric_dict.get("LDS Req")
    metrics["lds_util"] = metric_dict.get("LDS Util")
    metrics["lds_read"] = metric_dict.get("LDS Read")
    metrics["lds_write"] = metric_dict.get("LDS Write")
    metrics["lds_atomic"] = metric_dict.get("LDS Atomic")
    metrics["smem_rd"] = metric_dict.get("sL1D Rd")
    metrics["icache_rd"] = metric_dict.get("IL1 Fetch")

    # L1 cache panels
    metrics["vl1_hit"] = metric_dict.get("VL1 Hit")
    metrics["sl1d_hit"] = metric_dict.get("sL1D Hit")
    metrics["il1_hit"] = metric_dict.get("IL1 Hit")

    # L1→L2 BW (Bytes/s from YAML)
    metrics["vl1_l2_rd_bw"] = metric_dict.get("VL1_L2 Read BW")
    metrics["vl1_l2_wr_bw"] = metric_dict.get("VL1_L2 Write BW")
    metrics["vl1_l2_atomic_bw"] = metric_dict.get("VL1_L2 Atomic BW")
    metrics["sl1d_l2_rd_bw"] = metric_dict.get("sL1D_L2 Read BW")
    metrics["il1_l2_rd_bw"] = metric_dict.get("IL1_L2 Read BW")

    # L2 panel
    metrics["l2_hit"] = metric_dict.get("L2 Hit")

    # L2→Fabric BW (Bytes/s)
    metrics["l2_fabric_read_bw"] = metric_dict.get("L2-Fabric Read BW")
    metrics["l2_fabric_wr_at_bw"] = metric_dict.get("L2-Fabric Write and Atomic BW")

    # Fabric→HBM
    metrics["hbm_rd"] = metric_dict.get("HBM Rd")
    metrics["hbm_wr"] = metric_dict.get("HBM Wr")
    metrics["hbm_read_traffic"] = metric_dict.get("HBM Read Traffic")
    metrics["hbm_wr_at_traffic"] = metric_dict.get("HBM Write and Atomic Traffic")
    metrics["remote_read_traffic"] = metric_dict.get("Remote Read Traffic")
    metrics["remote_wr_at_traffic"] = metric_dict.get("Remote Write and Atomic Traffic")
    metrics["hbm_read_bw"] = metric_dict.get("HBM Read BW")
    metrics["hbm_write_bw"] = metric_dict.get("HBM Write BW")
    metrics["hbm_atomic_bw"] = metric_dict.get("HBM Atomic BW")

    # xGMI / PCIe BW (gfx950 only)
    metrics["xgmi_read_bw"] = metric_dict.get("xGMI Read BW")
    metrics["xgmi_write_bw"] = metric_dict.get("xGMI Write BW")
    metrics["xgmi_atomic_bw"] = metric_dict.get("xGMI Atomic BW")
    metrics["pcie_read_bw"] = metric_dict.get("PCIe Read BW")
    metrics["pcie_write_bw"] = metric_dict.get("PCIe Write BW")
    metrics["pcie_atomic_bw"] = metric_dict.get("PCIe Atomic BW")

    return metrics


# ---------------------------------------------------------------------------
# Diagram building
# ---------------------------------------------------------------------------


# Arrow lengths
_KERNEL_ARROW_LEN = 16  # wider arrows from Kernel to L1 (long edge labels)
_STD_ARROW_LEN = 12  # standard inter-cache edge arrows

# Panel heights (L1 sub-panels stack to _TOTAL_H)
_VL1D_H = 16  # 14 content lines + 2 padding
_LDS_H = 10
_SL1D_H = 4
_L1I_H = 4
_TOTAL_H = _VL1D_H + _LDS_H + _SL1D_H + _L1I_H

# Panel widths — all non-Kernel IP blocks share one width for a uniform grid
_IP_BLOCK_W = 22
# IO row panels (separate layout above/below the main grid)
_XGMI_PANEL_W = 24  # fits "xGMI (to Peer GPU)" label
_PCIE_PANEL_W = 46  # fits "PCIe (to CPU or Non-xGMI connected GPU)" label

# Layout offsets
_IO_PAD_OFFSET = 3  # gap between fabric_col edge and xGMI/PCIe arrow text
# Arch capability sets
_MALL_ARCHS = frozenset({"gfx940", "gfx941", "gfx942", "gfx950"})
# Console dimensions
_CONSOLE_WIDTH = 240  # CDNA layout is wider (more IP blocks than RDNA3.5)


def _rwa_triplet(
    read_val: Any,  # noqa: ANN401
    write_val: Any,  # noqa: ANN401
    atomic_val: Any,  # noqa: ANN401
    arrows: dict[str, str],
) -> list[str]:
    """Six lines: Read/Write/Atomic edge labels with colored arrows."""
    color_read = COLORS["read"]
    color_write = COLORS["write"]
    color_atomic = COLORS["atomic"]
    return [
        colored(format_edge("Read", read_val), color_read),
        colored(arrows["left"], color_read),
        colored(format_edge("Write", write_val), color_write),
        colored(arrows["right"], color_write),
        colored(format_edge("Atomic", atomic_val), color_atomic),
        colored(arrows["both"], color_atomic),
    ]


def _build_request_edges(
    metrics: dict[str, Any],
    arrows: dict[str, str],
) -> Text:
    """Edges from Kernel to L1 caches, aligned to panel heights."""
    # VL1D scope — Non-buffer + Buffer requests
    vl1d_lines = [
        "[white]Non-buffer Request[/white]",
        *_rwa_triplet(
            metrics["flat_read"], metrics["flat_write"], metrics["flat_atomic"], arrows
        ),
        "[white]Buffer Request[/white]",
        *_rwa_triplet(
            metrics["buffer_read"],
            metrics["buffer_write"],
            metrics["buffer_atomic"],
            arrows,
        ),
    ]

    # LDS scope
    if metrics["lds_read"] is not None:
        lds_lines = [
            *_rwa_triplet(
                metrics["lds_read"],
                metrics["lds_write"],
                metrics["lds_atomic"],
                arrows,
            ),
            colored(format_edge("Instr", metrics["lds_req"]), "black"),
            colored(arrows["both"], "black"),
        ]
    else:
        lds_lines = [
            colored(format_edge("Instr", metrics["lds_req"]), "black"),
            colored(arrows["both"], "black"),
        ]

    # sL1D scope — SMEM
    color_read = COLORS["read"]
    sl1d_lines = [
        "[white]SMEM[/white]",
        colored(format_edge("Read", metrics["smem_rd"]), color_read),
        colored(arrows["left"], color_read),
    ]

    # L1I scope — ICACHE
    l1i_lines = [
        "[white]ICACHE[/white]",
        colored(format_edge("Read", metrics["icache_rd"]), color_read),
        colored(arrows["left"], color_read),
    ]

    lines = (
        pad_to(vl1d_lines, _VL1D_H)
        + pad_to(lds_lines, _LDS_H)
        + pad_to(sl1d_lines, _SL1D_H)
        + pad_to(l1i_lines, _L1I_H)
    )
    return Text.from_markup("\n".join(lines))


def _build_l1_stack(metrics: dict[str, Any]) -> Group:
    """Build vertically stacked L1 cache panels: VL1D, LDS, sL1D, L1I."""

    vl1_panel = build_cache_panel(
        "VL1D",
        [("Hit", metrics["vl1_hit"], "%", COLORS["hit"])],
        width=_IP_BLOCK_W,
        height=_VL1D_H,
    )
    lds_util_line = (
        f"{metric_line('Util', metrics['lds_util'], '%', COLORS['util'])}\n"
        f"[dim]{progress_bar(metrics['lds_util'])}[/dim]"
        if metrics["lds_util"] is not None
        else ""
    )
    lds_panel = build_ip_block("LDS", _IP_BLOCK_W, _LDS_H, stack_metrics(lds_util_line))
    sl1d_panel = build_cache_panel(
        "sL1D",
        [("Hit", metrics["sl1d_hit"], "%", COLORS["hit"])],
        width=_IP_BLOCK_W,
        height=_SL1D_H,
    )
    l1i_panel = build_cache_panel(
        "L1I",
        [("Hit", metrics["il1_hit"], "%", COLORS["hit"])],
        width=_IP_BLOCK_W,
        height=_L1I_H,
    )

    return Group(vl1_panel, lds_panel, sl1d_panel, l1i_panel)


def _read_bw_edge(
    bw_value: Any,  # noqa: ANN401
    arrow_left: str,
    color: str,
) -> list[str]:
    """Three markup lines: Read BW label, formatted value, left arrow."""
    formatted = format_value(bw_value, "Bytes/s", 1)
    return [
        colored("Read BW", color),
        colored(formatted, color),
        colored(arrow_left, color),
    ]


def _bw_label_value(
    label: str,
    value: Any,  # noqa: ANN401
    color: str,
) -> str:
    """Label and formatted BW value — joined by newline."""
    return "\n".join([
        colored(label, color),
        colored(format_value(value, "Bytes/s", 1), color),
    ])


def _build_l1_l2_edges(
    metrics: dict[str, Any],
    arrows: dict[str, str],
) -> Text:
    """L1->L2 edge column: BW (VL1D Rd/Wr/Atomic, sL1D Rd, L1I Rd)."""
    color_read = COLORS["read"]
    arrow_left = arrows["left"]

    vl1d_content = stack_metrics(
        _bw_label_value("Read BW", metrics["vl1_l2_rd_bw"], color_read)
        + "\n"
        + colored(arrow_left, color_read),
        _bw_label_value("Write BW", metrics["vl1_l2_wr_bw"], COLORS["write"])
        + "\n"
        + colored(arrows["right"], COLORS["write"]),
        _bw_label_value("Atomic BW", metrics["vl1_l2_atomic_bw"], COLORS["atomic"])
        + "\n"
        + colored(arrows["both"], COLORS["atomic"]),
    ).split("\n")

    lines = (
        pad_to(vl1d_content, _VL1D_H)
        + pad_to([], _LDS_H)
        + pad_to(
            _read_bw_edge(metrics["sl1d_l2_rd_bw"], arrow_left, color_read),
            _SL1D_H,
        )
        + pad_to(
            _read_bw_edge(metrics["il1_l2_rd_bw"], arrow_left, color_read),
            _L1I_H,
        )
    )
    return Text.from_markup("\n".join(lines))


def _build_fabric_content(metrics: dict[str, Any]) -> str:
    """Build Rich markup for the Data Fabric panel (gfx908–gfx942)."""
    color_read = COLORS["read"]
    color_write = COLORS["write"]
    hbm_section = "\n".join([
        "[white]To/From HBM (Req)[/white]",
        f"  Read {colored(format_scientific(metrics['hbm_rd']), color_read)}",
        f"  Write {colored(format_scientific(metrics['hbm_wr']), color_write)}",
    ])
    traffic_section = "\n".join([
        "[white]HBM Traffic[/white]",
        f"  {metric_line('Read', metrics['hbm_read_traffic'], '%', color_read)}",
        f"  {metric_line('Write', metrics['hbm_wr_at_traffic'], '%', color_write)}",
        "",
        "[white]Remote Traffic[/white]",
        f"  {metric_line('Read', metrics['remote_read_traffic'], '%', color_read)}",
        f"  {metric_line('Write', metrics['remote_wr_at_traffic'], '%', color_write)}",
    ])
    return stack_metrics(hbm_section, traffic_section)


def _build_hbm_content(
    metrics: dict[str, Any],
) -> str:
    """Build Rich markup for the HBM panel (gfx950 BW metrics)."""
    color_read = COLORS["read"]
    color_write = COLORS["write"]
    color_atomic = COLORS["atomic"]
    return stack_metrics(
        _bw_label_value("Read BW", metrics["hbm_read_bw"], color_read),
        _bw_label_value("Write BW", metrics["hbm_write_bw"], color_write),
        _bw_label_value("Atomic BW", metrics["hbm_atomic_bw"], color_atomic),
    )


def _build_io_row(
    metrics: dict[str, Any],
    fabric_col: int,
    *,
    bw_keys: tuple[str, str, str],
    panel_label: str,
    panel_width: int,
    panel_above: bool,
) -> Group:
    """Build an xGMI or PCIe block with Read/Write/Atomic BW arrows."""
    color_read = COLORS["read"]
    color_write = COLORS["write"]
    color_atomic = COLORS["atomic"]
    read_bw = format_value(metrics.get(bw_keys[0]), "Bytes/s", 1)
    write_bw = format_value(metrics.get(bw_keys[1]), "Bytes/s", 1)
    atomic_bw = format_value(metrics.get(bw_keys[2]), "Bytes/s", 1)

    panel = Panel(
        f"[dim]{panel_label}[/dim]",
        border_style=COLORS["block"],
        width=panel_width,
        height=3,
    )
    panel_grid = Table.grid(padding=0)
    col_offset = (panel_width - _XGMI_PANEL_W) // 2
    panel_grid.add_column(width=fabric_col - col_offset)
    panel_grid.add_column()
    panel_grid.add_row("", panel)

    arrow_grid = Table.grid(padding=0)
    arrow_grid.add_column(width=fabric_col + _IO_PAD_OFFSET)
    arrow_grid.add_column()
    arrow_grid.add_row(
        "",
        Text.from_markup(
            f"[{color_read}]||  Read BW"
            f"    {read_bw}[/{color_read}]\n"
            f"[{color_write}]||  Write BW"
            f"   {write_bw}[/{color_write}]\n"
            f"[{color_atomic}]||  Atomic BW"
            f"  {atomic_bw}[/{color_atomic}]"
        ),
    )

    if panel_above:
        return Group(panel_grid, arrow_grid)
    return Group(arrow_grid, panel_grid)


_FABRIC_COL_INDEX = 6  # kernel, req_edges, l1_stack, l1_l2_edges, l2, l2_fab_edges


def _build_scope_bar(total_width: int, fabric_col: int) -> str:
    """Build scope bar markup spanning *total_width*, split at *fabric_col*."""
    gpu_label = " [dim]GPU (XCD)[/dim] "
    fabric_label = " [dim]Fabric / Memory[/dim] "
    gpu_label_len = Text.from_markup(gpu_label).cell_len
    fabric_label_len = Text.from_markup(fabric_label).cell_len

    gpu_section = fabric_col - 1  # -1 for leading '|'
    fab_section = total_width - fabric_col - 2  # -2 for middle '|' and trailing '|'

    gpu_pad_l = (gpu_section - gpu_label_len) // 2
    gpu_pad_r = gpu_section - gpu_label_len - gpu_pad_l
    fab_pad_l = (fab_section - fabric_label_len) // 2
    fab_pad_r = fab_section - fabric_label_len - fab_pad_l

    return (
        f"|{'-' * gpu_pad_l}{gpu_label}{'-' * gpu_pad_r}"
        f"|{'-' * fab_pad_l}{fabric_label}{'-' * fab_pad_r}|"
    )


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
    """Create the CDNA memory diagram matching the reference PNG layout."""
    metrics = _extract_metrics(metric_dict)
    kernel_arrows = make_arrows(_KERNEL_ARROW_LEN)
    std_arrows = make_arrows(_STD_ARROW_LEN)
    is_gfx950 = gpu_arch is not None and gpu_arch.startswith("gfx950")
    has_mall = gpu_arch in _MALL_ARCHS

    # Build main diagram grid first (needed to measure width for scope bar)
    kernel = build_kernel_panel(_TOTAL_H, padding_lines=13)
    req_edges = _build_request_edges(metrics, kernel_arrows)
    l1_stack = _build_l1_stack(metrics)
    l1_l2_edges = _build_l1_l2_edges(metrics, std_arrows)
    l2 = build_cache_panel(
        "L2",
        [("Hit", metrics["l2_hit"], "%", COLORS["hit"])],
        width=_IP_BLOCK_W,
        height=_TOTAL_H,
    )
    l2_fab_edges = build_bw_edges(
        [
            ("Read BW", metrics["l2_fabric_read_bw"], "left", COLORS["read"]),
            (
                "Write/Atomic BW",
                metrics["l2_fabric_wr_at_bw"],
                "right",
                COLORS["write"],
            ),
        ],
        std_arrows,
    )
    if is_gfx950:
        fabric = build_ip_block("Data Fabric", _IP_BLOCK_W, _TOTAL_H)
        hbm_content = _build_hbm_content(metrics)
        hbm = build_ip_block("HBM", _IP_BLOCK_W, _TOTAL_H, hbm_content)
    else:
        fabric_content = _build_fabric_content(metrics)
        fabric = build_ip_block("Data Fabric", _IP_BLOCK_W, _TOTAL_H, fabric_content)
        hbm = build_ip_block("HBM", _IP_BLOCK_W, _TOTAL_H)
    umc = build_ip_block("UMC", _IP_BLOCK_W, _TOTAL_H)

    grid_cols: list[tuple[RenderableType, VerticalAlignMethod]] = [
        (kernel, "top"),
        (req_edges, "top"),
        (l1_stack, "top"),
        (l1_l2_edges, "top"),
        (l2, "top"),
        (l2_fab_edges, "middle"),
        (fabric, "top"),
    ]
    if has_mall:
        grid_cols.append((build_ip_block("MALL", _IP_BLOCK_W, _TOTAL_H), "top"))
    grid_cols.extend([(umc, "top"), (hbm, "top")])

    main_layout = Table.grid(padding=0)
    for _, vert in grid_cols:
        main_layout.add_column(vertical=vert)
    main_layout.add_row(*(col for col, _ in grid_cols))

    chart_width = console.measure(main_layout).maximum
    fabric_col = sum(
        console.measure(col).maximum for col, _ in grid_cols[:_FABRIC_COL_INDEX]
    )

    sections: list[RenderableType] = []
    if chart_title:
        sections.append(f"[bold]{chart_title}[/bold]")

    if is_gfx950:
        sections.append(
            _build_io_row(
                metrics,
                fabric_col,
                bw_keys=("xgmi_read_bw", "xgmi_write_bw", "xgmi_atomic_bw"),
                panel_label="xGMI (to Peer GPU)",
                panel_width=_XGMI_PANEL_W,
                panel_above=True,
            )
        )

    sections.append(_build_scope_bar(chart_width, fabric_col))
    sections.append("")
    sections.append(main_layout)

    if is_gfx950:
        sections.append("")
        sections.append(
            _build_io_row(
                metrics,
                fabric_col,
                bw_keys=("pcie_read_bw", "pcie_write_bw", "pcie_atomic_bw"),
                panel_label="PCIe (to CPU or Non-xGMI connected GPU)",
                panel_width=_PCIE_PANEL_W,
                panel_above=False,
            )
        )

    sections.append("")
    sections.append(build_legend())

    if show_debug:
        notes: list[tuple[str, str]] = [
            ("VL1D", "Per-CU vector data cache (Buffer/Non-buffer requests)"),
            ("LDS", "Local Data Share, on-CU scratchpad"),
            ("sL1D", "Per-CU scalar data cache (SMEM requests)"),
            ("L1I", "Per-CU instruction cache (ICACHE requests)"),
            ("L2 (TCC)", "Shared last-level cache"),
            ("Data Fabric", "Infinity Fabric interconnect"),
        ]
        if has_mall:
            notes.append(("MALL", "Mid-level Address Lookup Layer (MI300+)"))
        notes.extend([
            ("UMC", "Unified Memory Controller"),
            ("HBM", "High Bandwidth Memory"),
            (
                "xGMI",
                "Inter-GPU link (MI350 has individual counters;"
                " earlier cards use 'traffic to remote')",
            ),
            (
                "PCIe",
                "Host/non-xGMI link (MI350 has individual counters;"
                " earlier cards use 'traffic to remote')",
            ),
        ])
        sections.append("")
        sections.append(build_arch_notes(notes, heading="Architecture Notes (CDNA)"))

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
    """Render the CDNA memory chart and return as a string."""
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
    """CLI entry point for the CDNA memory chart."""
    mem_chart_cli_main(
        "CDNA Memory Chart - CLI",
        create_mem_chart_diagram,
        normalize_mem_chart_metrics,
        DEFAULT_SAMPLE_METRICS,
        console_width=_CONSOLE_WIDTH,
    )


if __name__ == "__main__":
    main()

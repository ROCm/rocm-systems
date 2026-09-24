# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Data-driven memory chart renderer.

Reads JSON layout files and renders the memory hierarchy diagram.
All topology comes from the JSON -- no architecture-specific code paths.
"""

import math
import re
from collections import defaultdict
from io import StringIO
from typing import Any, Optional, Union

from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from memory_chart.loader import load_layout

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

CachePanelRow = Union[
    tuple[str, Any, str, str],
    tuple[str, Any, str, str, bool],
]

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

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

_LEGEND_ENTRIES: tuple[tuple[str, str, str], ...] = (
    ("<----", "Read", "read"),
    ("---->", "Write", "write"),
    ("<--->", "Atomic", "atomic"),
    ("█", "Util", "util"),
    ("█", "Hit%", "hit"),
)

_STALL_ENTRY: tuple[str, str, str] = ("█", "Stall", "stall")

_DIRECTION_TO_ARROW: dict[str, str] = {
    "backward": "left",
    "forward": "right",
    "both": "both",
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_mem_chart(
    metric_dict: dict[str, Any],
    *,
    chart_title: str,
    gpu_arch: Optional[str] = None,
    membw: Optional[Any] = None,  # noqa: ANN401
) -> str:
    """Render the memory chart for gpu_arch and return as a string."""
    if gpu_arch is None:
        raise ValueError("gpu_arch is required")

    layout = load_layout(gpu_arch)
    metrics = _normalize_metrics(metric_dict, layout)

    console_width = _estimate_console_width(layout)
    buf = StringIO()
    console = Console(
        file=buf,
        force_terminal=True,
        width=console_width,
        height=80,
    )

    _create_diagram(
        layout,
        metrics,
        console,
        chart_title=chart_title,
        gpu_arch=gpu_arch,
        membw=membw,
    )
    return buf.getvalue()


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences."""
    return re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", text)


def format_mem_chart_heading(
    normal_unit: str,
    *,
    panel_id: int = 300,
    section_label: str = "Memory Chart",
) -> str:
    """Build heading like '3. Memory Chart (Normalization: per_kernel)'."""
    section = max(0, int(panel_id)) // 100
    return f"{section}. {section_label} (Normalization: {normal_unit})"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_value(
    value: Union[int, float, str, None],
    unit: str = "",
    precision: int = 1,
) -> str:
    """Format a metric value with unit."""
    if value is None:
        return "N/A"
    try:
        numeric = float(value)
    except (ValueError, TypeError):
        return "N/A"
    if math.isnan(numeric):
        return "N/A"
    if unit == "GB/s":
        return f"{numeric:.3f} GB/s"
    if unit == "Bytes/s":
        return f"{numeric / 1e9:.3f} GB/s"
    if unit == "%":
        return f"{numeric:.{precision}f}%"
    return f"{numeric:.{precision}f}{unit}"


def format_scientific(
    value: Union[int, float, str, None],
    precision: int = 2,
) -> str:
    """Format as integer (<1000) or scientific notation."""
    if value is None:
        return "N/A"
    try:
        numeric = float(value)
    except (ValueError, TypeError):
        return "N/A"
    if math.isnan(numeric):
        return "N/A"
    if abs(numeric) < 1000:
        return str(round(numeric))
    return f"{numeric:.{precision}e}"


def colored(text: str, color: str) -> str:
    """Wrap text in Rich color markup."""
    return f"[{color}]{text}[/{color}]"


def metric_line(
    label: str,
    value: Any,  # noqa: ANN401
    unit: str = "%",
    color: str = "bright_green",
) -> str:
    """Rich markup line: 'label formatted_value' in color."""
    return f"{label} {colored(format_value(value, unit), color)}"


def progress_bar(percent: Optional[float], width: int = 10) -> str:
    """Unicode progress bar."""
    if percent is None:
        return "░" * width
    try:
        numeric = float(percent)
    except (ValueError, TypeError):
        return "░" * width
    if math.isnan(numeric):
        return "░" * width
    filled = int(width * min(100, max(0, numeric)) / 100)
    return "█" * filled + "░" * (width - filled)


def safe_float(
    value: Union[int, float, str, None],
) -> Optional[float]:
    """Parse a numeric metric. None for None/NaN/unparseable."""
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def safe_float_sum(
    *values: Union[int, float, str, None],
) -> Optional[float]:
    """Sum numeric values, skipping None/NaN. None if all invalid."""
    terms: list[float] = []
    for v in values:
        parsed = safe_float(v)
        if parsed is not None:
            terms.append(parsed)
    return sum(terms) if terms else None


def format_edge(
    label: str,
    value: Any,  # noqa: ANN401
    width: int = 7,
) -> str:
    """Format edge label with scientific-notation value."""
    label_str = f"{label:<{width}}"
    if value is not None:
        value_str = f": {format_scientific(value):>7}"
    else:
        value_str = ""
    return f"{label_str}{value_str}"


def make_arrows(length: int) -> dict[str, str]:
    """Build arrow dict with left/right/both/plain keys."""
    return {
        "left": "<" + "-" * (length - 1),
        "right": "-" * (length - 1) + ">",
        "both": "<" + "-" * (length - 2) + ">",
        "plain": "-" * length,
    }


def stack_metrics(*lines: str) -> str:
    """Join metric lines with blank line, skip empty."""
    return "\n\n".join(line for line in lines if line)


def pad_to(lines: list[str], target: int) -> list[str]:
    """Pad or truncate lines to exactly target rows."""
    padded = lines + [""] * max(0, target - len(lines))
    return padded[:target]


def build_legend(
    *,
    include_atomic: bool = True,
    include_util: bool = True,
    include_stall: bool = False,
) -> str:
    """Build the color legend string."""
    exclude: set[str] = set()
    if not include_atomic:
        exclude.add("atomic")
    if not include_util:
        exclude.add("util")
    entries = [(s, lbl, k) for s, lbl, k in _LEGEND_ENTRIES if k not in exclude]
    if include_stall:
        entries.append(_STALL_ENTRY)
    items = [f"{colored(s, COLORS[k])} {lbl}" for s, lbl, k in entries]
    return f"[dim]Legend:[/dim] {'  '.join(items)}"


# ---------------------------------------------------------------------------
# Metric normalization
# ---------------------------------------------------------------------------


def _normalize_metrics(
    metric_dict: dict[str, Any],
    layout: dict[str, Any],
) -> dict[str, Any]:
    """Keep only metrics referenced by the layout."""
    keys: set[str] = set()
    for block in layout.get("blocks", []):
        for item in block.get("content", []):
            keys.add(item["metric"])
    for arrow in layout.get("arrows", []):
        keys.add(arrow["metric"])
    return {k: metric_dict.get(k) for k in keys}


# ---------------------------------------------------------------------------
# Panel builders (generic)
# ---------------------------------------------------------------------------


def _build_cu_panel(
    block: dict[str, Any],
    metrics: dict[str, Any],
    width: int,
    height: int,
) -> Panel:
    """Build Compute Units panel with stats list."""
    lines: list[str] = []
    for item in block.get("content", []):
        name = item["metric"]
        title = item.get("title", name)
        raw = metrics.get(name)
        unit = item.get("unit", "")
        if name in ("Scratch Allocation", "LDS Allocation"):
            val = safe_float(raw)
            val = val / 1024 if val is not None else None
            lines.append(metric_line(title, val, " KB"))
        elif name == "Wavefront Occupancy":
            lines.append(metric_line(title, raw, ""))
            lines.append(colored("waves/CU", COLORS["util"]))
        else:
            lines.append(metric_line(title, raw, unit))
    color = COLORS["kernel"]
    return Panel(
        "\n".join(lines),
        title=f"[bold {color}]Compute Units[/bold {color}]",
        border_style=color,
        width=width,
        height=height,
    )


def _build_cache_panel(
    block: dict[str, Any],
    metrics: dict[str, Any],
    width: int,
    height: int,
    extra_rows: Optional[list[CachePanelRow]] = None,
    border_override: Optional[str] = None,
) -> Panel:
    """Build cache panel with hit/util metrics and progress bars."""
    lines: list[str] = []
    for i, item in enumerate(block.get("content", [])):
        name = item["metric"]
        value = metrics.get(name)
        unit = item.get("unit", "%")
        cat = item.get("category", "util")
        clr = COLORS.get(cat, COLORS["util"])
        show_bar = unit == "%" and cat in ("hit", "util", "stall")
        if i > 0:
            lines.append("")
        lines.append(metric_line(item.get("title", name), value, unit, clr))
        if show_bar:
            lines.append(f"[dim]{progress_bar(value)}[/dim]")
    if extra_rows:
        for label, value, unit, clr, *_ in extra_rows:
            lines.append("")
            lines.append(metric_line(label, value, unit, clr))
    border = border_override or COLORS["block"]
    title = block.get("title", "")
    return Panel(
        "\n".join(lines),
        title=f"[bold {border}]{title}[/bold {border}]",
        border_style=border,
        width=width,
        height=height,
    )


def _build_ip_block(
    block: dict[str, Any],
    metrics: dict[str, Any],
    width: int,
    height: int,
    extra_content: str = "",
    border_override: Optional[str] = None,
    extra_rows: Optional[list[CachePanelRow]] = None,
) -> Panel:
    """Build simple IP block panel."""
    lines: list[str] = []
    for item in block.get("content", []):
        name = item["metric"]
        value = metrics.get(name)
        unit = item.get("unit", "")
        cat = item.get("category", "util")
        clr = COLORS.get(cat, COLORS["util"])
        title = item.get("title", name)
        if unit in ("Bytes/s",):
            lines.append(
                f"{colored(title, clr)}\n{colored(format_value(value, unit), clr)}"
            )
        else:
            lines.append(metric_line(title, value, unit, clr))
    if extra_rows:
        for label, value, unit, clr, *_ in extra_rows:
            lines.append("")
            lines.append(metric_line(label, value, unit, clr))
    content = stack_metrics(*lines) if not extra_rows else "\n".join(lines)
    if extra_content:
        content = stack_metrics(content, extra_content) if content else extra_content
    border = border_override or COLORS["block"]
    title = block.get("title", "")
    return Panel(
        content,
        title=f"[bold {border}]{title}[/bold {border}]",
        border_style=border,
        width=width,
        height=height,
    )


def _build_panel(
    block: dict[str, Any],
    metrics: dict[str, Any],
    width: int,
    height: int,
    **kwargs: Any,  # noqa: ANN401
) -> Panel:
    """Dispatch to CU, cache, or IP block builder."""
    if block.get("column") == 0:
        return _build_cu_panel(block, metrics, width, height)
    cats = {c.get("category") for c in block.get("content", [])}
    if cats & {"hit", "util", "stall"}:
        return _build_cache_panel(block, metrics, width, height, **kwargs)
    return _build_ip_block(block, metrics, width, height, **kwargs)


def _build_nested_panel(
    parent: dict[str, Any],
    all_blocks: dict[str, dict[str, Any]],
    metrics: dict[str, Any],
    width: int,
    height: int,
) -> Panel:
    """Build parent panel containing child sub-panels."""
    children_ids = parent.get("children", [])
    inner_w = width - 4
    child_h = max(4, (height - 4) // max(len(children_ids), 1) - 1)
    child_panels: list[RenderableType] = []
    for cid in children_ids:
        child = all_blocks.get(cid, {})
        child_panels.append(_build_panel(child, metrics, inner_w, child_h))
    border = COLORS["block"]
    title = parent.get("title", "")
    return Panel(
        Group(*child_panels),
        title=f"[bold {border}]{title}[/bold {border}]",
        border_style=border,
        width=width,
        height=height,
    )


# ---------------------------------------------------------------------------
# Edge builders (generic)
# ---------------------------------------------------------------------------


def _build_request_edge_section(
    arrows: list[dict[str, Any]],
    metrics: dict[str, Any],
    arrow_strs: dict[str, str],
) -> list[str]:
    """Build lines for a set of arrows with group headers."""
    lines: list[str] = []
    seen_groups: set[str] = set()
    for arrow in arrows:
        group = arrow.get("group")
        if group and group not in seen_groups:
            seen_groups.add(group)
            lines.append(f"[white]{group}[/white]")
        title = arrow.get("title", "")
        value = metrics.get(arrow["metric"])
        arrow_key = _DIRECTION_TO_ARROW.get(arrow["direction"], "both")
        cat_color = COLORS.get(arrow["category"], "white")
        if _is_bw_arrow(arrow):
            bw_str = format_value(value, "Bytes/s")
            lines.append(colored(title, cat_color))
            lines.append(colored(bw_str, cat_color))
            lines.append(colored(arrow_strs[arrow_key], cat_color))
        else:
            lines.append(colored(format_edge(title, value), cat_color))
            lines.append(colored(arrow_strs[arrow_key], cat_color))
    return lines


def _build_bw_edge_column(
    arrows: list[dict[str, Any]],
    metrics: dict[str, Any],
    arrow_strs: dict[str, str],
) -> Text:
    """Build BW edge text column."""
    content: list[str] = []
    for i, arrow in enumerate(arrows):
        if i > 0:
            content.append("")
        title = arrow.get("title", "")
        value = metrics.get(arrow["metric"])
        arrow_key = _DIRECTION_TO_ARROW.get(arrow["direction"], "both")
        clr = COLORS.get(arrow["category"], COLORS["read"])
        value_str = format_value(value, "Bytes/s")
        content.append(colored(title, clr))
        content.append(colored(value_str, clr))
        content.append(colored(arrow_strs[arrow_key], clr))
    return Text.from_markup("\n".join(content))


def _build_padded_request_edges(
    arrows: list[dict[str, Any]],
    target_blocks: list[dict[str, Any]],
    metrics: dict[str, Any],
    arrow_strs: dict[str, str],
    total_h: int,
    all_blocks: Optional[dict[str, dict[str, Any]]] = None,
    height_map: Optional[dict[str, int]] = None,
) -> list[str]:
    """Build request edge column padded to target panel heights."""
    arrows_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for arrow in arrows:
        arrows_by_target[arrow["to"]].append(arrow)

    lines: list[str] = []
    for block in target_blocks:
        block_id = block["id"]
        block_arrows = arrows_by_target.get(block_id, [])
        block_h = (
            height_map[block_id] if height_map and block_id in height_map else total_h
        )
        section_lines = _build_request_edge_section(block_arrows, metrics, arrow_strs)
        lines.extend(pad_to(section_lines, block_h))

    return lines


def _build_padded_bw_edges(
    arrows: list[dict[str, Any]],
    source_blocks: list[dict[str, Any]],
    metrics: dict[str, Any],
    arrow_strs: dict[str, str],
    total_h: int,
    all_blocks: Optional[dict[str, dict[str, Any]]] = None,
    height_map: Optional[dict[str, int]] = None,
) -> list[str]:
    """Build BW edge column padded to source panel heights."""
    child_to_parent: dict[str, str] = {}
    for block in source_blocks:
        for cid in block.get("children", []):
            child_to_parent[cid] = block["id"]
    arrows_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for arrow in arrows:
        source = arrow["from"]
        parent = child_to_parent.get(source, source)
        arrows_by_parent[parent].append(arrow)

    lines: list[str] = []
    for block in source_blocks:
        block_id = block["id"]
        block_arrows = arrows_by_parent.get(block_id, [])
        block_h = (
            height_map[block_id] if height_map and block_id in height_map else total_h
        )

        section: list[str] = []
        for i, arrow in enumerate(block_arrows):
            if i > 0:
                section.append("")
            title = arrow.get("title", "")
            value = metrics.get(arrow["metric"])
            arrow_key = _DIRECTION_TO_ARROW.get(arrow["direction"], "both")
            clr = COLORS.get(arrow["category"], COLORS["read"])
            section.append(colored(title, clr))
            section.append(colored(format_value(value, "Bytes/s"), clr))
            section.append(colored(arrow_strs[arrow_key], clr))
        lines.extend(pad_to(section, block_h))

    return lines


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def _is_bw_arrow(arrow: dict[str, Any]) -> bool:
    name = arrow.get("metric", "").lower()
    return "bw" in name or "bandwidth" in name


def _estimate_console_width(layout: dict[str, Any]) -> int:
    columns = {
        b["column"] for b in layout["blocks"] if b.get("position", "grid") == "grid"
    }
    return max(200, len(columns) * 30 + 40)


def _compute_panel_width(
    blocks: list[dict[str, Any]],
    all_blocks: Optional[dict[str, dict[str, Any]]] = None,
) -> int:
    """Compute uniform panel width from all blocks."""
    effective = list(blocks)
    if all_blocks:
        for block in blocks:
            for cid in block.get("children", []):
                child = all_blocks.get(cid)
                if child:
                    effective.append(child)
    max_title = max((len(b.get("title", "")) for b in effective), default=0)
    max_content = 0
    for block in effective:
        for item in block.get("content", []):
            line_len = len(item.get("title", "")) + 12
            max_content = max(max_content, line_len)
    child_title_w = max((len(b.get("title", "")) + 6 for b in effective), default=0)
    return max(16, max_title + 8, max_content + 4, child_title_w + 8)


def _content_height(
    block: dict[str, Any],
    blocks_by_id: Optional[dict[str, dict[str, Any]]] = None,
) -> int:
    """Minimum height a block needs for its own content."""
    children_ids = block.get("children", [])
    if children_ids and blocks_by_id:
        child_h = 0
        for cid in children_ids:
            child = blocks_by_id.get(cid, {})
            cc = len(child.get("content", []))
            child_h += max(6, cc * 3 + 4)
        return max(4, child_h + 6)
    content_count = len(block.get("content", []))
    return max(4, content_count * 3 + 4)


def _bw_edge_lines_needed(arrow_count: int) -> int:
    """Lines for BW arrows: label + value + arrow per entry."""
    if arrow_count <= 0:
        return 0
    return arrow_count * 3 + max(0, arrow_count - 1) + 2


def _request_edge_lines_needed(
    arrows: list[dict[str, Any]],
) -> int:
    """Lines for request arrows grouped by group field."""
    if not arrows:
        return 0
    groups: dict[str, list[dict[str, Any]]] = {}
    for arrow in arrows:
        gname = arrow.get("group", "")
        groups.setdefault(gname, []).append(arrow)
    total = 0
    for gname, group_arrows in groups.items():
        if gname:
            total += 1
        total += len(group_arrows) * 2
    return total + 2


def _compute_height_map(
    col_groups: dict[int, list[dict[str, Any]]],
    arrows_between: dict[tuple[int, int], list[dict[str, Any]]],
    blocks_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, int], int]:
    """Compute per-block heights satisfying content and edge constraints.

    Returns (height_map, total_h).  Both panel builders and edge
    builders use the same height_map so they always align.
    """
    height_map: dict[str, int] = {}

    # Step 1: baseline from content
    for blocks in col_groups.values():
        for block in blocks:
            height_map[block["id"]] = _content_height(block, blocks_by_id)

    # Step 2: inflate from edge constraints
    for (_src_col, _dst_col), arrows in arrows_between.items():
        bw_arrows = [a for a in arrows if _is_bw_arrow(a)]
        req_arrows = [a for a in arrows if not _is_bw_arrow(a)]

        # BW edges: group by source block
        if bw_arrows:
            by_source: dict[str, int] = defaultdict(int)
            for a in bw_arrows:
                by_source[a["from"]] += 1
            for src_id, count in by_source.items():
                needed = _bw_edge_lines_needed(count)
                parent_id = src_id
                for blocks in col_groups.values():
                    for block in blocks:
                        if src_id in block.get("children", []):
                            parent_id = block["id"]
                if parent_id in height_map:
                    height_map[parent_id] = max(height_map[parent_id], needed)

        # Request edges: group by target block
        if req_arrows:
            by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for a in req_arrows:
                by_target[a["to"]].append(a)
            for tgt_id, tgt_arrows in by_target.items():
                needed = _request_edge_lines_needed(tgt_arrows)
                if tgt_id in height_map:
                    height_map[tgt_id] = max(height_map[tgt_id], needed)

    # Step 3: total height per column, then global max
    col_heights: dict[int, int] = {}
    for col, blocks in col_groups.items():
        col_heights[col] = sum(height_map.get(b["id"], 4) for b in blocks)
    total_h = max(col_heights.values()) if col_heights else 30
    total_h = max(total_h, 20)

    # Step 4: distribute surplus to largest block in shorter cols
    for col, blocks in col_groups.items():
        surplus = total_h - col_heights.get(col, 0)
        if surplus > 0 and blocks:
            largest = max(
                blocks,
                key=lambda b: height_map.get(b["id"], 0),
            )
            height_map[largest["id"]] += surplus

    return height_map, total_h


def _compute_arrow_length(
    arrows: list[dict[str, Any]],
    from_cu: bool = False,
) -> int:
    """Compute arrow length from longest label."""
    max_label = max((len(a.get("title", "")) for a in arrows), default=4)
    base = max(8, max_label + 8)
    if from_cu:
        base = max(base, 16)
    return base


# ---------------------------------------------------------------------------
# Scope bar
# ---------------------------------------------------------------------------


def _find_scope_split(blocks: list[dict[str, Any]]) -> int:
    """Find column index where the memory region starts."""
    memory_titles = {
        "data fabric",
        "gcea",
        "ea/df",
        "dram",
        "hbm",
        "umc",
        "mall",
    }
    for block in sorted(blocks, key=lambda b: b.get("column", 0)):
        if block.get("position", "grid") != "grid":
            continue
        if block.get("title", "").lower() in memory_titles:
            return block["column"]
    cols = sorted(
        set(b["column"] for b in blocks if b.get("position", "grid") == "grid")
    )
    return cols[len(cols) // 2] if cols else 0


def _build_scope_bar(
    total_width: int,
    split_col_width: int,
    gpu_label: str,
    mem_label: str,
) -> str:
    """Build scope bar markup."""
    gpu_markup = f" [dim]{gpu_label}[/dim] "
    mem_markup = f" [dim]{mem_label}[/dim] "
    gpu_len = Text.from_markup(gpu_markup).cell_len
    mem_len = Text.from_markup(mem_markup).cell_len

    gpu_section = split_col_width - 1
    mem_section = total_width - split_col_width - 2

    gpu_pad_l = max(0, (gpu_section - gpu_len) // 2)
    gpu_pad_r = max(0, gpu_section - gpu_len - gpu_pad_l)
    mem_pad_l = max(0, (mem_section - mem_len) // 2)
    mem_pad_r = max(0, mem_section - mem_len - mem_pad_l)

    return (
        f"|{'-' * gpu_pad_l}{gpu_markup}{'-' * gpu_pad_r}"
        f"|{'-' * mem_pad_l}{mem_markup}{'-' * mem_pad_r}|"
    )


# ---------------------------------------------------------------------------
# Above/below blocks (xGMI, PCIe)
# ---------------------------------------------------------------------------


def _build_io_section(
    block: dict[str, Any],
    metrics: dict[str, Any],
    fabric_col: int,
    above: bool = True,
) -> Group:
    """Build an above/below IO block (xGMI, PCIe) with optional BW arrows."""
    title = block.get("title", "")
    panel_width = max(24, len(title) + 4)
    panel = Panel(
        f"[dim]{title}[/dim]",
        border_style=COLORS["block"],
        width=panel_width,
        height=3,
    )
    panel_grid = Table.grid(padding=0)
    panel_grid.add_column(width=fabric_col)
    panel_grid.add_column()
    panel_grid.add_row("", panel)

    content = block.get("content", [])
    arrow_lines: list[str] = []
    if content:
        for item in content:
            value = metrics.get(item["metric"])
            cat = item.get("category", "read")
            clr = COLORS.get(cat, COLORS["read"])
            label = item.get("title", item["metric"])
            bw_str = format_value(value, "Bytes/s", 1)
            arrow_lines.append(f"[{clr}]||  {label}    {bw_str}[/{clr}]")
    else:
        arrow_lines.append("[dim]||[/dim]")

    connector = Table.grid(padding=0)
    connector.add_column(width=fabric_col + 3)
    connector.add_column()
    connector.add_row("", Text.from_markup("\n".join(arrow_lines)))

    if above:
        return Group(panel_grid, connector)
    return Group(connector, panel_grid)


# ---------------------------------------------------------------------------
# Membw stall annotations
# ---------------------------------------------------------------------------

_STALL_LEVEL_MAP: dict[str, str] = {
    "vl1d": "GL1",
    "l2": "GL2",
    "data_fabric": "EA",
}


def _collect_stall_rows(
    membw: Any,  # noqa: ANN401
    block_id: str,
) -> list[CachePanelRow]:
    """Extract active stall rows for a block."""
    if membw is None:
        return []
    level = _STALL_LEVEL_MAP.get(block_id)
    if level is None:
        return []
    rows: list[CachePanelRow] = []
    for node in getattr(membw, "nodes", []):
        _collect_active_leaves(node, level, rows)
    return rows


def _collect_active_leaves(
    node: Any,  # noqa: ANN401
    level: str,
    rows: list[CachePanelRow],
) -> None:
    """Recursively collect active leaf nodes at level."""
    if getattr(node, "state", None) != "active":
        return
    is_leaf = not any(
        getattr(c, "state", None) == "active" for c in getattr(node, "children", [])
    )
    if getattr(node, "level", None) == level and is_leaf:
        supporting = getattr(node, "supporting", ())
        value = getattr(supporting[0], "value", None) if supporting else None
        label = getattr(node, "label", "stall")
        rows.append((
            f"[!] {label}",
            value,
            "%",
            COLORS["stall"],
            False,
        ))
    for child in getattr(node, "children", []):
        _collect_active_leaves(child, level, rows)


# ---------------------------------------------------------------------------
# Diagram assembly
# ---------------------------------------------------------------------------


def _create_diagram(
    layout: dict[str, Any],
    metrics: dict[str, Any],
    console: Console,
    *,
    chart_title: str = "",
    gpu_arch: Optional[str] = None,
    membw: Optional[Any] = None,  # noqa: ANN401
) -> None:
    """Build and print the full memory chart diagram."""
    blocks_by_id = {b["id"]: b for b in layout["blocks"]}

    # Partition blocks
    grid_blocks, above_blocks, below_blocks = _partition_blocks(layout, blocks_by_id)

    # Group grid blocks by column
    col_groups = _group_by_column(grid_blocks)
    sorted_columns = sorted(col_groups.keys())

    # Map block IDs to columns (including children)
    block_to_col = _build_block_col_map(grid_blocks)
    arrows_between = _group_arrows_by_columns(layout, block_to_col)

    # Compute dimensions
    all_effective = _collect_effective_blocks(grid_blocks, blocks_by_id)
    panel_w = _compute_panel_width(all_effective, blocks_by_id)
    height_map, total_h = _compute_height_map(col_groups, arrows_between, blocks_by_id)

    # Build grid
    grid_items, col_grid_indices = _build_grid_items(
        sorted_columns,
        col_groups,
        arrows_between,
        blocks_by_id,
        metrics,
        panel_w,
        total_h,
        membw,
        height_map,
    )

    # Assemble main grid
    main_layout = Table.grid(padding=0)
    for _, vert in grid_items:
        main_layout.add_column(vertical=vert)
    main_layout.add_row(*(col for col, _ in grid_items))

    # Measure for scope bar and IO positioning
    chart_width = console.measure(main_layout).maximum
    split_col = _find_scope_split(grid_blocks)
    split_blocks = col_groups.get(split_col, [])
    split_title = split_blocks[0]["title"] if split_blocks else ""
    fabric_col = _measure_fabric_col(main_layout, split_title, console)

    # Scope bar labels
    arch_id = layout.get("arch", "")
    gpu_label, mem_label = _scope_labels(arch_id)

    # Assemble sections
    sections: list[RenderableType] = []
    if chart_title:
        sections.append(f"[bold]{chart_title}[/bold]")

    for block in above_blocks:
        sections.append(_build_io_section(block, metrics, fabric_col, above=True))

    sections.append(_build_scope_bar(chart_width, fabric_col, gpu_label, mem_label))
    sections.append("")
    sections.append(main_layout)

    for block in below_blocks:
        sections.append("")
        sections.append(_build_io_section(block, metrics, fabric_col, above=False))

    # Legend
    has_atomic = any(a["category"] == "atomic" for a in layout.get("arrows", []))
    has_stall = any(
        c.get("category") == "stall"
        for b in layout["blocks"]
        for c in b.get("content", [])
    )
    has_membw_stalls = membw is not None and any(
        getattr(n, "state", None) == "active" for n in getattr(membw, "nodes", [])
    )
    sections.append("")
    sections.append(
        build_legend(
            include_atomic=has_atomic,
            include_stall=has_stall or has_membw_stalls,
        )
    )

    console.print(Group(*sections))


# ---------------------------------------------------------------------------
# Diagram assembly helpers
# ---------------------------------------------------------------------------


def _partition_blocks(
    layout: dict[str, Any],
    blocks_by_id: dict[str, dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Separate grid, above, and below blocks."""
    child_ids: set[str] = set()
    for block in layout["blocks"]:
        for cid in block.get("children", []):
            child_ids.add(cid)

    grid_blocks: list[dict[str, Any]] = []
    above_blocks: list[dict[str, Any]] = []
    below_blocks: list[dict[str, Any]] = []

    for block in layout["blocks"]:
        if block["id"] in child_ids:
            continue
        pos = block.get("position", "grid")
        if pos == "above":
            above_blocks.append(block)
        elif pos == "below":
            below_blocks.append(block)
        else:
            grid_blocks.append(block)
    return grid_blocks, above_blocks, below_blocks


def _group_by_column(
    blocks: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for block in blocks:
        groups[block["column"]].append(block)
    for col in groups.values():
        col.sort(key=lambda b: b.get("order", 0))
    return groups


def _build_block_col_map(
    grid_blocks: list[dict[str, Any]],
) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for block in grid_blocks:
        mapping[block["id"]] = block["column"]
        for cid in block.get("children", []):
            mapping[cid] = block["column"]
    return mapping


def _group_arrows_by_columns(
    layout: dict[str, Any],
    block_to_col: dict[str, int],
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    result: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for arrow in layout.get("arrows", []):
        from_col = block_to_col.get(arrow["from"], -1)
        to_col = block_to_col.get(arrow["to"], -1)
        if from_col >= 0 and to_col >= 0:
            result[(from_col, to_col)].append(arrow)
    return result


def _collect_effective_blocks(
    grid_blocks: list[dict[str, Any]],
    blocks_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    effective = list(grid_blocks)
    for block in grid_blocks:
        for cid in block.get("children", []):
            child = blocks_by_id.get(cid)
            if child:
                effective.append(child)
    return effective


def _build_grid_items(
    sorted_columns: list[int],
    col_groups: dict[int, list[dict[str, Any]]],
    arrows_between: dict[tuple[int, int], list[dict[str, Any]]],
    blocks_by_id: dict[str, dict[str, Any]],
    metrics: dict[str, Any],
    panel_w: int,
    total_h: int,
    membw: Optional[Any],  # noqa: ANN401
    height_map: Optional[dict[str, int]] = None,
) -> tuple[list[tuple[RenderableType, str]], dict[int, int]]:
    """Build grid column items and col→grid_item_index map."""
    grid_items: list[tuple[RenderableType, str]] = []
    col_grid_indices: dict[int, int] = {}

    for i, col_idx in enumerate(sorted_columns):
        blocks_in_col = col_groups[col_idx]

        # Edge column
        if i > 0:
            prev_col = sorted_columns[i - 1]
            edge_arrows = arrows_between.get((prev_col, col_idx), [])
            if not edge_arrows:
                for c in sorted_columns[:i]:
                    ea = arrows_between.get((c, col_idx), [])
                    edge_arrows.extend(ea)

            if edge_arrows:
                from_cu = any(
                    blocks_by_id.get(a["from"], {}).get("column") == 0
                    for a in edge_arrows
                )
                all_bw = all(_is_bw_arrow(a) for a in edge_arrows)
                arrow_len = _compute_arrow_length(edge_arrows, from_cu)
                arrow_strs = make_arrows(arrow_len)

                if all_bw:
                    # Check if prev column has stacked blocks
                    prev_blocks = col_groups.get(prev_col, [])
                    if len(prev_blocks) > 1:
                        edge_lines = _build_padded_bw_edges(
                            edge_arrows,
                            prev_blocks,
                            metrics,
                            arrow_strs,
                            total_h,
                            blocks_by_id,
                            height_map,
                        )
                        grid_items.append((
                            Text.from_markup("\n".join(edge_lines)),
                            "top",
                        ))
                    else:
                        edge_col = _build_bw_edge_column(
                            edge_arrows, metrics, arrow_strs
                        )
                        grid_items.append((edge_col, "middle"))
                else:
                    edge_lines = _build_padded_request_edges(
                        edge_arrows,
                        blocks_in_col,
                        metrics,
                        arrow_strs,
                        total_h,
                        blocks_by_id,
                        height_map,
                    )
                    grid_items.append((
                        Text.from_markup("\n".join(edge_lines)),
                        "top",
                    ))

        # Block column — record index, use unified height_map
        col_grid_indices[col_idx] = len(grid_items)
        panels: list[RenderableType] = []
        for block in blocks_in_col:
            block_h = height_map.get(block["id"], 4) if height_map else total_h

            if block.get("children"):
                panel = _build_nested_panel(
                    block, blocks_by_id, metrics, panel_w, block_h
                )
            else:
                stall_rows = _collect_stall_rows(membw, block["id"])
                if stall_rows:
                    stall_h = block_h + len(stall_rows) * 2
                    stall_w = max(
                        panel_w,
                        max(len(lbl) + 12 for lbl, *_ in stall_rows),
                    )
                    panel = _build_panel(
                        block,
                        metrics,
                        stall_w,
                        stall_h,
                        extra_rows=stall_rows,
                        border_override=COLORS["stall"],
                    )
                else:
                    panel = _build_panel(block, metrics, panel_w, block_h)
            panels.append(panel)

        if len(panels) == 1:
            grid_items.append((panels[0], "top"))
        else:
            grid_items.append((Group(*panels), "top"))

    return grid_items, col_grid_indices


def _measure_fabric_col(
    main_layout: RenderableType,
    split_block_title: str,
    console: Console,
) -> int:
    """Measure width up to the split column's panel in the rendered grid."""
    buf = StringIO()
    measure_console = Console(
        file=buf,
        force_terminal=True,
        width=console.width,
        height=80,
    )
    measure_console.print(main_layout, end="")
    rendered = strip_ansi(buf.getvalue())
    for line in rendered.splitlines():
        pos = line.find(split_block_title)
        if pos > 0:
            border = line.rfind("╭", 0, pos)
            if border >= 0:
                return border
            return pos
    return console.measure(main_layout).maximum // 2


def _scope_labels(arch_id: str) -> tuple[str, str]:
    """Return (gpu_label, memory_label) for the scope bar."""
    if arch_id.startswith("gfx11"):
        return ("GPU", "System Memory")
    if arch_id == "gfx1250":
        return ("XCD", "AID")
    return ("GPU (XCD)", "Fabric / Memory")

# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Shared helpers for memory chart renderers (gfx9, gfx11)."""

import argparse
import json
import math
import pathlib
import re
from collections.abc import Callable
from io import StringIO
from typing import Any, Optional, Union

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from utils.utils_analysis import format_bw_human_readable

CachePanelRow = Union[
    tuple[str, Any, str, str],
    tuple[str, Any, str, str, bool],
]

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


def format_value(
    value: Union[int, float, str, None], unit: str = "", precision: int = 1
) -> str:
    """Format a metric value with unit. Returns 'N/A' for None/NaN/invalid."""
    if value is None:
        return "N/A"
    try:
        numeric = float(value)
    except (ValueError, TypeError):
        return "N/A"
    if math.isnan(numeric):
        return "N/A"
    if unit in ("GB/s", "Bytes/s"):
        return format_bw_human_readable(value, unit, precision)
    if unit == "%":
        return f"{numeric:.{precision}f}%"
    return f"{numeric:.{precision}f}{unit}"


def format_scientific(value: Union[int, float, str, None], precision: int = 2) -> str:
    """Format as rounded integer (<1000) or scientific notation (>=1000)."""
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
    """Wrap *text* in Rich color markup tags."""
    return f"[{color}]{text}[/{color}]"


def metric_line(
    label: str,
    value: Any,  # noqa: ANN401
    unit: str = "%",
    color: str = "bright_green",
) -> str:
    """Rich markup line: 'label value_with_unit' in *color*."""
    return f"{label} {colored(format_value(value, unit), color)}"


def progress_bar(percent: Optional[float], width: int = 10) -> str:
    """Unicode progress bar. None/NaN/invalid -> empty bar."""
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


def safe_float_sum(
    *values: Union[int, float, str, None],
) -> Optional[float]:
    """Sum numeric values, skipping None/NaN/unparseable. None if all invalid."""
    terms: list[float] = []
    for value in values:
        try:
            numeric = float(value)  # type: ignore[arg-type]
        except (ValueError, TypeError):
            continue
        if not math.isnan(numeric):
            terms.append(numeric)
    return sum(terms) if terms else None


def format_edge(
    label: str,
    value: Any,  # noqa: ANN401
    width: int = 7,
) -> str:
    """Format edge label with optional scientific-notation value."""
    label_str = f"{label:<{width}}"
    if value is not None:
        value_str = f": {format_scientific(value):>7}"
    else:
        value_str = ""
    return f"{label_str}{value_str}"


def make_arrows(length: int) -> dict[str, str]:
    """Build arrow dict with left/right/both/plain keys of given length."""
    return {
        "left": "<" + "-" * (length - 1),
        "right": "-" * (length - 1) + ">",
        "both": "<" + "-" * (length - 2) + ">",
        "plain": "-" * length,
    }


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from *text*."""
    return re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", text)


def stack_metrics(*lines: str) -> str:
    """Join panel metric lines with a blank line, skipping empty ones."""
    return "\n\n".join(line for line in lines if line)


def pad_to(lines: list[str], target: int) -> list[str]:
    """Pad or truncate *lines* to exactly *target* rows (returns new list)."""
    padded = lines + [""] * max(0, target - len(lines))
    return padded[:target]


def format_mem_chart_heading(
    normal_unit: str,
    *,
    panel_id: int = 300,
    section_label: str = "Memory Chart",
) -> str:
    """Build heading: '{panel_id//100}. {label} (Normalization: {unit})'."""
    section = max(0, int(panel_id)) // 100
    return f"{section}. {section_label} (Normalization: {normal_unit})"


def build_legend(
    *,
    include_atomic: bool = True,
    include_util: bool = True,
    include_stall: bool = False,
) -> str:
    """Build the color legend string from ``_LEGEND_ENTRIES``."""
    exclude = set()
    if not include_atomic:
        exclude.add("atomic")
    if not include_util:
        exclude.add("util")
    entries = [
        (symbol, label, key)
        for symbol, label, key in _LEGEND_ENTRIES
        if key not in exclude
    ]
    if include_stall:
        entries.append(_STALL_ENTRY)
    items = [
        f"{colored(symbol, COLORS[color_key])} {label}"
        for symbol, label, color_key in entries
    ]
    return f"[dim]Legend:[/dim] {'  '.join(items)}"


def build_arch_notes(
    notes: list[tuple[str, str]],
    *,
    heading: str = "Architecture Notes",
) -> str:
    """Build architecture notes markup from *(abbreviation, description)* pairs."""
    lines = [f"[dim]{heading}:[/dim]"]
    for abbrev, desc in notes:
        lines.append(f"  {abbrev}: {desc}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shared Rich panel builders
# ---------------------------------------------------------------------------


def build_kernel_panel(
    height: int,
    padding_lines: int = 13,
) -> Panel:
    """Build the Kernel (shader core) panel used by both gfx9 and gfx11."""
    return Panel(
        "\n" * padding_lines + "[dim]Shader Core[/dim]\n[dim]Wave Execution[/dim]",
        title=(f"[bold {COLORS['kernel']}]Kernel[/bold {COLORS['kernel']}]"),
        border_style=COLORS["kernel"],
        width=14,
        height=height,
    )


def build_cache_panel(
    title: str,
    rows: list[CachePanelRow],
    width: int,
    height: int,
    border_style: str = COLORS["block"],
) -> Panel:
    """Cache panel with metric lines and optional progress bars."""
    lines: list[str] = []
    for i, row in enumerate(rows):
        label, value, unit, color = row[:4]
        show_bar = row[4] if len(row) > 4 else True  # type: ignore[arg-type]
        if i > 0:
            lines.append("")
        lines.append(metric_line(label, value, unit, color))
        if unit == "%" and show_bar:
            lines.append(f"[dim]{progress_bar(value)}[/dim]")
    return Panel(
        "\n".join(lines),
        title=f"[bold {border_style}]{title}[/bold {border_style}]",
        border_style=border_style,
        width=width,
        height=height,
    )


def build_ip_block(
    title: str,
    width: int,
    height: int,
    content: str = "",
    border_style: str = COLORS["block"],
) -> Panel:
    """Panel with bold-colored title styling shared with ``build_cache_panel``."""
    return Panel(
        content,
        title=f"[bold {border_style}]{title}[/bold {border_style}]",
        border_style=border_style,
        width=width,
        height=height,
    )


def build_bw_edge_column(
    entries: list[tuple[str, str, str, str]],
    arrows: dict[str, str],
) -> Text:
    """Vertical edge column with labeled BW arrows.

    Vertical centering is handled by the parent grid's
    ``vertical="middle"`` column setting.
    """
    content: list[str] = []
    for i, (label, value_str, arrow_key, color) in enumerate(entries):
        if i > 0:
            content.append("")
        content.append(colored(label, color))
        content.append(colored(value_str, color))
        content.append(colored(arrows[arrow_key], color))
    return Text.from_markup("\n".join(content))


# ---------------------------------------------------------------------------
# Shared rendering scaffold
# ---------------------------------------------------------------------------


def render_chart_to_string(
    create_fn: Callable[..., None],
    metric_dict: dict[str, Any],
    normalize_fn: Callable[[dict[str, Any]], dict[str, Any]],
    console_width: int = 240,
    **diagram_kwargs: Any,  # noqa: ANN401
) -> str:
    """Normalize metrics, render via *create_fn*, return the output."""
    flat = normalize_fn(metric_dict)
    buf = StringIO()
    console = Console(
        file=buf,
        force_terminal=True,
        width=console_width,
        height=80,
    )
    create_fn(flat, console, show_debug=False, **diagram_kwargs)
    return buf.getvalue()


def mem_chart_cli_main(
    description: str,
    create_fn: Callable[..., None],
    normalize_fn: Callable[[dict[str, Any]], dict[str, Any]],
    default_metrics: dict[str, Any],
    console_width: int = 240,
) -> None:
    """Shared CLI entry point for memory chart renderers."""
    arg_parser = argparse.ArgumentParser(description=description)
    arg_parser.add_argument("--data", "-d", help="JSON file with metrics data")
    arg_parser.add_argument("--debug", action="store_true", help="Show debug info")
    arg_parser.add_argument("--norm", default="per_kernel", help="Normalization unit")
    arg_parser.add_argument("--arch", default=None, help="GPU architecture")
    arg_parser.add_argument("--txt", help="Write plain text to file")
    arg_parser.add_argument("--svg", help="Write SVG to file")
    args = arg_parser.parse_args()

    if args.data:
        with pathlib.Path(args.data).open(encoding="utf-8") as fp:
            metrics = normalize_fn(json.load(fp))
    else:
        metrics = normalize_fn(dict(default_metrics))

    heading = format_mem_chart_heading(args.norm)
    arch_kwargs: dict[str, Any] = {}
    if args.arch:
        arch_kwargs["gpu_arch"] = args.arch

    if args.txt:
        buf = StringIO()
        console = Console(
            file=buf,
            force_terminal=False,
            width=console_width,
            height=80,
        )
        create_fn(
            metrics,
            console,
            show_debug=args.debug,
            chart_title=heading,
            **arch_kwargs,
        )
        with pathlib.Path(args.txt).open("w", encoding="utf-8") as fp:
            fp.write(strip_ansi(buf.getvalue()))
        return

    if args.svg:
        console = Console(
            record=True,
            width=console_width,
            height=80,
        )
        create_fn(
            metrics,
            console,
            show_debug=args.debug,
            chart_title=heading,
            **arch_kwargs,
        )
        console.save_svg(args.svg, title="Memory Chart")
        return

    console = Console(width=console_width)
    create_fn(
        metrics,
        console,
        show_debug=args.debug,
        chart_title=heading,
        **arch_kwargs,
    )

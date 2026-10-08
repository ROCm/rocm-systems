# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Draw a resolved memory chart layout in the terminal.

No architecture-specific code: placement comes from the layout, formatting from
the metric units, and sizes from the content.

- The first column is the compute side, drawn as a compact list.
- Arrows with a count unit are request arrows ("Read : 119"); others show a
  label, a value, and an arrow line.
- Arrows line up with the block on the side of the edge that has several or
  nested blocks; arrows between two single blocks are centred.

Vertical rules, which keep every arrow beside the box it belongs to:
- A box of height h at row top has interior rows [top + 1, top + h - 1).
- Arrows anchored on a box start at top + 1, and h >= 2 + their line count.
- A nested box starts below its parent's border, note, and earlier children;
  the parent's height is 2 + note lines + its children's heights.
- Centred arrows sit in the middle of the grid interior [1, H - 1).
"""

import re
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from io import StringIO
from typing import Any, Optional, Union

from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from membw_analysis.summary import active_stall_leaves
from memory_chart.loader import Layout, LayoutArrow, LayoutBlock, Layouts
from memory_chart.units import PLAIN, DisplayUnit, display_unit, format_value, numeric

COLORS = {
    "cu": "green",
    "block": "blue",
    "read": "bright_cyan",
    "write": "bright_yellow",
    "atomic": "bright_magenta",
    "util": "bright_green",
    "hit": "yellow",
    "stall": "indian_red",
    "bw": "bright_cyan",
    # Terminal default foreground, for values outside the legend
    "neutral": "default",
}

_LEGEND_ENTRIES: tuple[tuple[str, str, str], ...] = (
    ("<----", "Read", "read"),
    ("---->", "Write", "write"),
    ("<--->", "Atomic", "atomic"),
    ("█", "Util", "util"),
    ("█", "Hit%", "hit"),
    ("█", "Stall", "stall"),
)

_ARROW_GLYPHS = {"backward": "<{}", "forward": "{}>", "both": "<{}>"}

# Categories that get a progress bar under a percent value
_BAR_CATEGORIES = frozenset({"util", "hit", "stall", "neutral"})

# Columns reserved for a value when sizing a panel, so typical values fit on the
# label's line; a longer value moves to its own line.
_VALUE_RESERVE = {"percent": 6, "bandwidth": 12, "count": 8, "value": 9}

_BAR_WIDTH = 10
_MIN_TEXT_WIDTH = 10
_MIN_EDGE_WIDTH = 8
# Panel borders plus Rich's default horizontal padding
_PANEL_FRAME = 4
_MAX_CONSOLE_WIDTH = 400


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_mem_chart(
    metric_dict: dict[str, Any],
    *,
    chart_title: str,
    gpu_arch: str,
    units: dict[str, str],
    membw: Optional[Any] = None,  # noqa: ANN401
) -> str:
    """The memory chart of *gpu_arch* as text.

    *units* maps metric names to their config units. Active stalls in *membw*
    are listed in blocks with a stall_level.
    """
    layout = Layouts.for_arch(gpu_arch)
    if layout is None:
        raise ValueError(f"No memory chart layout for {gpu_arch!r}")
    return MemChart(layout, metric_dict, units, membw).render(chart_title)


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences."""
    return re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", text)


def format_mem_chart_heading(normal_unit: str, *, panel_id: int) -> str:
    """Heading like '3. Memory Chart (Normalization: per_kernel)'."""
    return f"{panel_id // 100}. Memory Chart (Normalization: {normal_unit})"


def progress_bar(percent: Union[int, float, str, None]) -> str:
    """Bar for a percentage, clamped to 0-100; empty when missing."""
    number = numeric(percent)
    filled = 0 if number is None else int(_BAR_WIDTH * min(100, max(0, number)) / 100)
    return "█" * filled + "░" * (_BAR_WIDTH - filled)


# ---------------------------------------------------------------------------
# Chart assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Placement:
    """Arrow lines placed in an edge column, beside their anchor block."""

    edge: int  # edge column after block column `edge`
    anchor: Optional[str]  # None for arrows centred in the edge
    start: int  # first grid row
    lines: tuple[str, ...]


class MemChart:
    """One layout drawn with one set of metric values."""

    def __init__(
        self,
        layout: Layout,
        values: dict[str, Any],
        units: dict[str, str],
        membw: Optional[Any] = None,  # noqa: ANN401
    ) -> None:
        self.layout = layout
        self.values = values
        self.units = units
        self.stalls = {b.id: _stall_rows(membw, b.stall_level) for b in layout.blocks()}
        self.blocks = {b.id: b for b in layout.blocks()}
        self.text_width = self._text_widths()
        self.edges = [self._edge(i) for i in range(len(layout.columns) - 1)]
        self._fit_scope_labels()
        self.height, self.total_height = self._heights()
        self.top = self._block_tops()
        self.placements = self._placements()

    # -- metric lookups ----------------------------------------------------

    def _unit(self, metric: str) -> DisplayUnit:
        return display_unit(self.units.get(metric)) or PLAIN

    def _shown(self, metric: str) -> str:
        return format_value(self.values.get(metric), self._unit(metric))

    def _is_request(self, arrow: LayoutArrow) -> bool:
        return self._unit(arrow.metric).kind == "count"

    # -- panel content -----------------------------------------------------

    def _needed_width(self, block: LayoutBlock) -> int:
        """Text width a block wants, before columns are evened out."""
        widths = [len(block.title) + 2, _MIN_TEXT_WIDTH]
        widths += [
            len(item.title) + 1 + _VALUE_RESERVE[self._unit(item.metric).kind]
            for item in block.items
        ]
        widths += [
            len(row.label) + 1 + _VALUE_RESERVE["percent"]
            for row in self.stalls[block.id]
        ]
        widths += [len(word) for word in (block.note or "").split()]
        widths += [self._needed_width(child) + _PANEL_FRAME for child in block.children]
        return max(widths)

    def _text_widths(self) -> dict[str, int]:
        """Text width of every block: the widest need in its column."""
        widths: dict[str, int] = {}
        for column in self.layout.columns:
            width = max(self._needed_width(block) for block in column)
            for block in column:
                self._assign_width(block, width, widths)
        for block in self.layout.blocks():
            for sub in (*block.above, *block.below):
                widths[sub.id] = max(len(sub.title), _MIN_TEXT_WIDTH)
        return widths

    def _fit_scope_labels(self) -> None:
        """Widen the last column of a scope region whose label does not fit.

        A region shows " label " with at least one dash on each side.
        """
        split = self.blocks[self.layout.scope_split].column
        left, right = self.layout.scope_labels
        for label, column in ((left, split - 1), (right, len(self.layout.columns) - 1)):
            widths = self._column_widths()
            split_x = sum(widths[: 2 * split])
            span = split_x - 1 if column < split else sum(widths) - split_x - 2
            deficit = len(label) + 4 - span
            for block in self.layout.columns[column] if deficit > 0 else ():
                self._assign_width(
                    block, self.text_width[block.id] + deficit, self.text_width
                )

    def _assign_width(
        self, block: LayoutBlock, width: int, widths: dict[str, int]
    ) -> None:
        widths[block.id] = width
        for child in block.children:
            self._assign_width(child, width - _PANEL_FRAME, widths)

    def _note_lines(self, block: LayoutBlock) -> list[str]:
        """A block's note, wrapped to its text width."""
        lines = textwrap.wrap(block.note or "", self.text_width[block.id])
        return [f"[dim]{line}[/dim]" for line in lines]

    def _lines(self, block: LayoutBlock) -> list[str]:
        """Markup lines inside a block without nested blocks."""
        width = self.text_width[block.id]
        lines = self._note_lines(block)
        compact = block.column == 0
        for item in block.items:
            if lines and not compact:
                lines.append("")
            shown = self._shown(item.metric)
            color = COLORS["util"] if compact else COLORS[item.category]
            if len(item.title) + 1 + len(shown) > width:
                lines += [item.title, _colored(shown, color)]
            else:
                lines.append(f"{item.title} {_colored(shown, color)}")
            unit = self._unit(item.metric)
            if (
                unit.kind == "percent"
                and item.category in _BAR_CATEGORIES
                and not compact
            ):
                lines.append(f"[dim]{progress_bar(self.values.get(item.metric))}[/dim]")
        for row in self.stalls[block.id]:
            if lines:
                lines.append("")
            shown = format_value(row.value, display_unit("Percent"))
            lines.append(f"{row.label} {_colored(shown, COLORS['stall'])}")
        return lines

    # -- edges -------------------------------------------------------------

    def _anchor(self, arrow: LayoutArrow) -> Optional[str]:
        """Block an arrow lines up with; None to centre it in the edge."""
        for end in (arrow.target, arrow.source):
            block = self.blocks[end]
            if block.parent or len(self.layout.columns[block.column]) > 1:
                return end
        return None

    def _edge(self, column: int) -> dict[Optional[str], list[LayoutArrow]]:
        """Arrows from *column* to the next, grouped by anchor block."""
        segments: dict[Optional[str], list[LayoutArrow]] = defaultdict(list)
        for arrow in self.layout.arrows:
            source, target = self.blocks[arrow.source], self.blocks[arrow.target]
            if target.position == "grid" and source.column == column:
                segments[self._anchor(arrow)].append(arrow)
        return segments

    def _request_label_width(
        self, segments: dict[Optional[str], list[LayoutArrow]]
    ) -> int:
        """Label width that lines up the values of an edge's request arrows."""
        arrows = [
            a for group in segments.values() for a in group if self._is_request(a)
        ]
        return max((len(a.title) for a in arrows), default=0)

    def _edge_text_width(self, segments: dict[Optional[str], list[LayoutArrow]]) -> int:
        """Width of an edge column; 0 when no arrows cross it."""
        arrows = [a for group in segments.values() for a in group]
        if not arrows:
            return 0
        label_width = self._request_label_width(segments)
        widths = [_MIN_EDGE_WIDTH]
        for arrow in arrows:
            if self._is_request(arrow):
                widths.append(label_width + 3 + _VALUE_RESERVE["count"])
            else:
                widths += [len(arrow.title), len(self._shown(arrow.metric))]
            if arrow.group:
                widths.append(len(arrow.group))
        return max(widths) + 1

    def _segment(
        self, arrows: list[LayoutArrow], width: int, label_width: int
    ) -> list[str]:
        """Lines for the arrows that share an anchor."""
        lines: list[str] = []
        group = None
        for arrow in arrows:
            color = COLORS[arrow.category]
            if arrow.group and arrow.group != group:
                lines.append(f"[white]{arrow.group}[/white]")
            elif lines and not self._is_request(arrow):
                lines.append("")
            group = arrow.group
            if self._is_request(arrow):
                shown = self._shown(arrow.metric)
                value_width = _VALUE_RESERVE["count"]
                lines.append(
                    _colored(
                        f"{arrow.title:<{label_width}} : {shown:>{value_width}}", color
                    )
                )
            else:
                lines.append(_colored(arrow.title, color))
                lines.append(_colored(self._shown(arrow.metric), color))
            lines.append(_colored(_arrow_line(arrow.direction, width - 1), color))
        return lines

    # -- heights -----------------------------------------------------------

    def _segment_lengths(self) -> dict[Optional[str], int]:
        """Rows each anchor's arrows take, on either side of the block."""
        lengths: dict[Optional[str], int] = defaultdict(int)
        for segments in self.edges:
            for anchor, arrows in segments.items():
                rows = len(self._segment(arrows, _MIN_EDGE_WIDTH, 0))
                lengths[anchor] = max(lengths[anchor], rows)
        return lengths

    def _heights(self) -> tuple[dict[str, int], int]:
        """Panel height of every block, and the height of the whole grid."""
        needs = self._segment_lengths()
        heights: dict[str, int] = {}
        totals = [
            sum(self._size(block, needs, heights) for block in column)
            for column in self.layout.columns
        ]
        total = max([*totals, needs.get(None, 0) + 2])
        # Leftover rows go to the last block of each column as blank interior rows
        for column, used in zip(self.layout.columns, totals):
            heights[column[-1].id] += total - used
        return heights, total

    def _size(
        self,
        block: LayoutBlock,
        needs: dict[Optional[str], int],
        heights: dict[str, int],
    ) -> int:
        """Record the height of a block and its nested blocks; return its own."""
        if block.children:
            inner = len(self._note_lines(block)) + sum(
                self._size(child, needs, heights) for child in block.children
            )
        else:
            inner = len(self._lines(block))
        heights[block.id] = max(inner, needs.get(block.id, 0), 1) + 2
        return heights[block.id]

    # -- drawing -----------------------------------------------------------

    def _panel(self, block: LayoutBlock, color: str) -> Panel:
        stalled = bool(self.stalls[block.id])
        border = COLORS["stall"] if stalled else color
        if block.children:
            body: RenderableType = Group(
                *self._note_lines(block),
                *(self._panel(child, COLORS["block"]) for child in block.children),
            )
        else:
            body = "\n".join(self._lines(block))
        return Panel(
            body,
            title=f"[bold {border}]{block.title}[/bold {border}]",
            border_style=border,
            width=self.text_width[block.id] + _PANEL_FRAME,
            height=self.height[block.id],
        )

    def _block_tops(self) -> dict[str, int]:
        """First row of every column block and nested block."""
        tops: dict[str, int] = {}
        for column in self.layout.columns:
            row = 0
            for block in column:
                self._place(block, row, tops)
                row += self.height[block.id]
        return tops

    def _place(self, block: LayoutBlock, top: int, tops: dict[str, int]) -> None:
        """Record the first row of a block and of its nested blocks."""
        tops[block.id] = top
        row = top + 1 + len(self._note_lines(block))
        for child in block.children:
            self._place(child, row, tops)
            row += self.height[child.id]

    def _placements(self) -> list[Placement]:
        """Where every edge's arrow lines go (see the vertical rules above)."""
        placements = []
        interior = self.total_height - 2
        for edge, segments in enumerate(self.edges):
            width = self._edge_text_width(segments)
            label_width = self._request_label_width(segments)
            for anchor, group in segments.items():
                lines = tuple(self._segment(group, width, label_width))
                if anchor:
                    start = self.top[anchor] + 1
                else:
                    start = 1 + (interior - len(lines)) // 2
                placements.append(Placement(edge, anchor, start, lines))
        return placements

    def _edge_column(self, edge: int) -> Text:
        rows = [""] * self.total_height
        for placement in self.placements:
            if placement.edge == edge:
                end = placement.start + len(placement.lines)
                rows[placement.start : end] = placement.lines
        return Text.from_markup("\n".join(rows))

    def _column_widths(self) -> list[int]:
        """Width of every grid column, alternating block and edge columns."""
        widths = []
        for i, column in enumerate(self.layout.columns):
            if i:
                widths.append(self._edge_text_width(self.edges[i - 1]))
            widths.append(self.text_width[column[0].id] + _PANEL_FRAME)
        return widths

    def _x(self, block_id: str) -> int:
        """Left edge of a column block."""
        return sum(self._column_widths()[: 2 * self.blocks[block_id].column])

    def _attached(self, host: LayoutBlock, sub: LayoutBlock, above: bool) -> Group:
        """A block drawn above or below *host*, joined by a connector."""
        x = self._x(host.id)
        panel = Table.grid(padding=0)
        panel.add_column(width=x)
        panel.add_column()
        panel.add_row(
            "",
            Panel(
                f"[dim]{sub.title}[/dim]",
                border_style=COLORS["block"],
                width=self.text_width[sub.id] + _PANEL_FRAME,
                height=3,
            ),
        )
        arrows = [a for a in self.layout.arrows if a.target == sub.id]
        label_width = max((len(a.title) for a in arrows), default=0)
        lines = [
            _colored(
                f"||  {a.title:<{label_width}}  {self._shown(a.metric)}",
                COLORS[a.category],
            )
            for a in arrows
        ] or ["[dim]||[/dim]"]
        connector = Table.grid(padding=0)
        connector.add_column(width=x + 3)
        connector.add_column()
        connector.add_row("", Text.from_markup("\n".join(lines)))
        return Group(panel, connector) if above else Group(connector, panel)

    def _scope_bar(self, total_width: int) -> str:
        split = self._x(self.layout.scope_split)
        left, right = self.layout.scope_labels
        left_part = _scope_section(left, split - 1)
        right_part = _scope_section(right, total_width - split - 2)
        return f"|{left_part}|{right_part}|"

    def render(self, chart_title: str) -> str:
        """The chart as text with terminal colors."""
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=_MAX_CONSOLE_WIDTH)
        console.print(self._renderable(chart_title))
        return buf.getvalue()

    def _renderable(self, chart_title: str) -> RenderableType:
        widths = self._column_widths()
        grid = Table.grid(padding=0)
        cells: list[RenderableType] = []
        for i, column in enumerate(self.layout.columns):
            if i:
                cells.append(self._edge_column(i - 1))
            color = COLORS["cu"] if i == 0 else COLORS["block"]
            cells.append(Group(*(self._panel(block, color) for block in column)))
        # Rich gives even an empty column a cell, so leave out unused edges
        kept = [(cell, width) for cell, width in zip(cells, widths) if width]
        for _, width in kept:
            grid.add_column(width=width, no_wrap=True)
        grid.add_row(*(cell for cell, _ in kept))

        hosts = [b for column in self.layout.columns for b in column]
        sections: list[RenderableType] = [f"[bold]{chart_title}[/bold]"]
        sections += [self._attached(h, sub, True) for h in hosts for sub in h.above]
        sections += [self._scope_bar(sum(widths)), "", grid]
        for host in hosts:
            for sub in host.below:
                sections += ["", self._attached(host, sub, False)]

        categories = {a.category for a in self.layout.arrows}
        categories |= {item.category for b in self.layout.blocks() for item in b.items}
        sections += [
            "",
            _legend(
                include_atomic="atomic" in categories,
                include_stall="stall" in categories or any(self.stalls.values()),
            ),
        ]
        return Group(*sections)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _colored(text: str, color: str) -> str:
    return f"[{color}]{text}[/{color}]"


def _legend(*, include_atomic: bool, include_stall: bool) -> str:
    skip = {
        k
        for k, keep in (("atomic", include_atomic), ("stall", include_stall))
        if not keep
    }
    items = [
        f"{_colored(symbol, COLORS[key])} {label}"
        for symbol, label, key in _LEGEND_ENTRIES
        if key not in skip
    ]
    return f"[dim]Legend:[/dim] {'  '.join(items)}"


def _cell_len(markup: str) -> int:
    return Text.from_markup(markup).cell_len


def _scope_section(label: str, span: int) -> str:
    """A scope-bar label centred in a run of dashes *span* cells wide."""
    markup = f" [dim]{label}[/dim] "
    fill = max(0, span - _cell_len(markup))
    return "-" * (fill // 2) + markup + "-" * (fill - fill // 2)


def _arrow_line(direction: str, length: int) -> str:
    template = _ARROW_GLYPHS[direction]
    return template.format("-" * (length - len(template) + 2))


# ---------------------------------------------------------------------------
# Membw stall annotations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _StallRow:
    label: str
    value: Optional[float]


def _stall_rows(membw: Any, level: Optional[str]) -> list[_StallRow]:  # noqa: ANN401
    """Active leaf bottlenecks of membw at *level*, in tree order."""
    return [
        _StallRow(
            f"[!] {node.label}",
            node.supporting[0].value if node.supporting else None,
        )
        for node in active_stall_leaves(membw, level)
    ]

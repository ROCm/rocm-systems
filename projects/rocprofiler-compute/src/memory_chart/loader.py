# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Load memory chart layouts and check them against the format rules in README.md.

A layout states where every block goes and how every part is drawn, so loading only
checks the file and groups its blocks for the renderer.
"""

import json
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, NoReturn, Optional, Tuple

import config
from membw_analysis.models import BOTTLENECK_LEVELS
from utils.logger import console_warning
from utils.utils_common import canonical_config_arch, load_panel_configs

LAYOUTS_DIR = config.rocprof_compute_home / "memory_chart" / "layouts"


class LayoutError(Exception):
    """A layout file that cannot be read or breaks the format rules."""


@dataclass(frozen=True)
class Keys:
    """Keys of one part of a layout file, in the order a layout file lists them."""

    order: Tuple[str, ...]
    optional: Tuple[str, ...] = ()

    @property
    def required(self) -> Tuple[str, ...]:
        return tuple(key for key in self.order if key not in self.optional)


# Version of the layout format; README.md says when to bump it
LAYOUT_VERSION = "2.0.0"

# Each Keys lists every key in file order, then the keys a file may leave out
LAYOUT_KEYS = Keys(("archs", "description", "scope", "blocks", "arrows"))
SCOPE_KEYS = Keys(("label", "columns"))
BLOCK_KEYS = Keys(
    (
        "id",
        "title",
        "column",
        "row",
        "order",
        "style",
        "parent",
        "stall_level",
        "content",
    ),
    ("row", "order", "parent", "stall_level", "content"),
)
ITEM_KEYS = Keys(("metric_id", "title", "category", "bar"))
ARROW_KEYS = Keys(
    ("from", "to", "direction", "metric_id", "title", "category", "style", "group"),
    ("group",),
)

_ROWS = (-1, 0, 1)  # above the grid, the grid, below the grid
_BLOCK_STYLES = ("panel", "list")
_ITEM_CATEGORIES = ("util", "hit", "stall", "neutral", "bw", "read", "write", "atomic")
_ARROW_CATEGORIES = ("read", "write", "atomic", "neutral")
_DIRECTIONS = ("forward", "backward", "both")
_ARROW_STYLES = ("inline", "stacked")

# A stack of blocks: (parent block id, column). The parent is None for top-level blocks.
StackKey = Tuple[Optional[str], int]


@dataclass(frozen=True)
class LayoutItem:
    """A metric shown in a block."""

    metric_id: str
    title: str
    category: str
    bar: bool  # draw a progress bar under the value


@dataclass(frozen=True)
class LayoutBlock:
    """A block and where it sits."""

    id: str
    title: str
    column: int
    row: int  # -1 above the grid, 0 in it, 1 below it
    style: str  # "panel", or "list" for a compact list
    parent: Optional[str]  # enclosing block of a nested block
    stall_level: Optional[str]  # membw level whose active stalls are listed here
    items: tuple[LayoutItem, ...]
    children: tuple["LayoutBlock", ...]


@dataclass(frozen=True)
class LayoutArrow:
    """A metric drawn between two blocks."""

    source: str
    target: str
    direction: str
    metric_id: str
    title: str
    category: str
    style: str  # "inline": one "title : value" line; "stacked": title, value, arrow
    group: Optional[str]


@dataclass(frozen=True)
class LayoutScope:
    """A label of the bar above the grid and the columns it spans."""

    label: str
    first_column: int
    last_column: int  # included


@dataclass(frozen=True)
class Layout:
    """A checked layout file."""

    path: Path
    archs: tuple[str, ...]
    description: str
    scopes: Tuple[LayoutScope, ...]  # left to right, covering every column
    columns: Tuple[Tuple[LayoutBlock, ...], ...]  # top-level blocks of row 0
    above: Optional[LayoutBlock]  # the block in row -1
    below: Optional[LayoutBlock]  # the block in row 1
    arrows: tuple[LayoutArrow, ...]

    def top_level_blocks(self) -> Iterator[LayoutBlock]:
        """The blocks stacked in the columns of row 0, column by column."""
        for column in self.columns:
            yield from column

    def blocks(self) -> Iterator[LayoutBlock]:
        """All blocks, column by column: each parent before its nested blocks, then
        the column's blocks above and below the grid."""
        for column, stack in enumerate(self.columns):
            for block in stack:
                yield from _with_descendants(block)
            for block in (self.above, self.below):
                if block is not None and block.column == column:
                    yield block

    def metrics(self) -> set[str]:
        """Ids of all metrics the layout shows."""
        ids = {item.metric_id for b in self.blocks() for item in b.items}
        return ids | {arrow.metric_id for arrow in self.arrows}


class Layouts:
    """The shipped layouts, read once and indexed by architecture.

    A broken layout file is skipped with a warning, so only its archs lose the chart.
    """

    _by_arch: Optional[dict[str, Layout]] = None

    @classmethod
    def for_arch(cls, gpu_arch: Optional[str]) -> Optional[Layout]:
        """The layout for a GPU architecture, or None if it has no chart."""
        if cls._by_arch is None:
            cls._by_arch = _index_by_arch(layout_files())
        return cls._by_arch.get(canonical_config_arch(gpu_arch) or "")


class PanelConfigs:
    """The memory chart panel config of each analysis-config arch, read on first use."""

    _by_arch: dict[str, Optional[dict[str, Any]]] = {}

    @classmethod
    def for_arch(cls, arch: str) -> Optional[dict[str, Any]]:
        """The memory chart panel config of *arch*, or None if it has none."""
        if arch not in cls._by_arch:
            panels = load_panel_configs([str(config.analysis_configs_dir / arch)])
            cls._by_arch[arch] = next(
                (panel for panel in panels.values() if is_memory_chart_panel(panel)),
                None,
            )
        return cls._by_arch[arch]


def is_memory_chart_panel(panel: dict[str, Any]) -> bool:
    """True when every table of a panel config is drawn as the memory chart."""
    sources = panel.get("data source") or []
    return bool(sources) and all(
        table.get("cli_style") == "mem_chart"
        for source in sources
        for table in source.values()
    )


def layout_files() -> list[Path]:
    """The shipped layout files, sorted by name."""
    return sorted(LAYOUTS_DIR.glob("*.json"))


def load_layout(path: Path) -> Layout:
    """Read and check a layout file; raise LayoutError if missing or malformed."""
    try:
        layout_json = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:  # ValueError covers invalid JSON
        _fail(path, f"cannot read the layout: {error}")
    if not isinstance(layout_json, dict):
        _fail(path, "the layout must be a JSON object")
    return layout_from_json(path, layout_json)


def layout_from_json(path: Path, data: dict[str, Any]) -> Layout:
    """Check a parsed layout file and index its blocks; *path* names it in errors."""
    _check_keys(path, data, LAYOUT_KEYS, "the layout")
    raw_blocks = _raw_blocks(path, data["blocks"])
    groups, column_count = _stacks(path, raw_blocks)
    columns = tuple(
        _stacked(path, groups, (None, column)) for column in range(column_count)
    )
    layout = Layout(
        path=path,
        archs=tuple(data["archs"]),
        description=data["description"],
        scopes=_scopes(path, data["scope"], column_count),
        columns=columns,
        above=_outside_grid(path, raw_blocks, column_count, -1),
        below=_outside_grid(path, raw_blocks, column_count, 1),
        arrows=tuple(_arrow(path, raw) for raw in data["arrows"]),
    )
    _check_references(layout, raw_blocks)
    return layout


# ---------------------------------------------------------------------------
# Reading the file
# ---------------------------------------------------------------------------


def _fail(path: Path, problem: str) -> NoReturn:
    raise LayoutError(f"{path.name}: {problem}")


def _index_by_arch(paths: list[Path]) -> dict[str, Layout]:
    index: dict[str, Layout] = {}
    for path in paths:
        try:
            layout = load_layout(path)
        except LayoutError as error:
            console_warning("memory chart", f"skipping {error}")
            continue
        for arch in layout.archs:
            if arch in index:
                console_warning(
                    "memory chart",
                    f"skipping {arch} in {path.name}: "
                    f"also listed by {index[arch].path.name}",
                )
                continue
            index[arch] = layout
    return index


def _with_descendants(block: LayoutBlock) -> Iterator[LayoutBlock]:
    yield block
    for child in block.children:
        yield from _with_descendants(child)


def _check_keys(path: Path, raw: dict[str, Any], keys: Keys, where: str) -> None:
    missing = set(keys.required) - set(raw)
    unknown = set(raw) - set(keys.order)
    if missing:
        _fail(path, f"missing {sorted(missing)} in {where}")
    if unknown:
        _fail(path, f"unknown keys {sorted(unknown)} in {where}")


def _check_choice(
    path: Path, value: object, choices: tuple[object, ...], what: str
) -> None:
    if value not in choices:
        _fail(path, f"{what} is {value!r}, not one of {list(choices)}")


def _check_index(path: Path, value: object, what: str) -> None:
    # bool is an int in Python, but not a column or an order
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(path, f"{what} must be a whole number of at least 0, not {value!r}")


def _check_row(path: Path, value: object, what: str) -> None:
    # bool is an int in Python, but not a row
    if isinstance(value, bool) or value not in _ROWS:
        _fail(path, f"{what} is {value!r}, not one of {list(_ROWS)}")


def _raw_blocks(path: Path, entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The file's blocks by id, each checked on its own."""
    by_id: dict[str, dict[str, Any]] = {}
    for raw in entries:
        where = repr(raw.get("id"))
        _check_keys(path, raw, BLOCK_KEYS, where)
        if raw["id"] in by_id:
            _fail(path, f"duplicate block id {raw['id']!r}")
        _check_index(path, raw["column"], f"column of {where}")
        if "order" in raw:
            _check_index(path, raw["order"], f"order of {where}")
        _check_choice(path, raw["style"], _BLOCK_STYLES, f"style of {where}")
        _check_choice(
            path,
            raw.get("stall_level"),
            (None, *sorted(BOTTLENECK_LEVELS)),
            f"stall_level of {where}",
        )
        _check_placement(path, raw, where)
        by_id[raw["id"]] = raw
    return by_id


def _check_placement(path: Path, raw: Dict[str, Any], where: str) -> None:
    """Check the row and order of a block, and what a block above or below the grid
    may have."""
    if "parent" in raw:
        if "row" in raw or "order" not in raw:
            _fail(path, f"{where} is nested, so it has an order and no row")
        return
    if "row" not in raw:
        _fail(path, f"{where} is not nested, so it needs a row")
    _check_row(path, raw["row"], f"row of {where}")
    if raw["row"] == 0:
        if "order" not in raw:
            _fail(path, f"{where} is in row 0, so it needs an order")
        return
    if {"order", "content", "stall_level"} & set(raw) or raw["style"] != "panel":
        _fail(
            path,
            f"{where} is above or below the grid: it needs the panel style, "
            "and no order, content, or stall_level",
        )


def _row(raw: Dict[str, Any]) -> int:
    """The row of a checked block; a nested block is in row 0 with its parent."""
    return raw.get("row", 0)


def _stacks(
    path: Path, raw_blocks: dict[str, dict[str, Any]]
) -> tuple[dict[StackKey, list[dict[str, Any]]], int]:
    """Blocks of row 0 grouped into the stacks they are drawn in, each sorted by order.

    A stack is keyed by (parent, column), with no parent for the top-level blocks of
    a column. Also returns the column count.
    """
    stacks: dict[StackKey, list[dict[str, Any]]] = defaultdict(list)
    for raw in raw_blocks.values():
        if _row(raw) != 0:
            continue
        parent_id = raw.get("parent")
        parent = raw_blocks.get(parent_id) if parent_id else None
        if parent_id and parent is None:
            _fail(path, f"{raw['id']!r} refers to unknown block {parent_id!r}")
        if parent and _row(parent) != 0:
            _fail(path, f"{raw['id']!r} must be nested in a block of row 0")
        if parent and parent["column"] != raw["column"]:
            _fail(path, f"{raw['id']!r} must have the column of {parent_id!r}")
        stacks[(parent_id, raw["column"])].append(raw)
    columns = sorted(column for parent, column in stacks if parent is None)
    if not columns or columns != list(range(len(columns))):
        _fail(path, f"columns must be numbered 0, 1, ... with no gaps, not {columns}")
    for members in stacks.values():
        members.sort(key=lambda raw: raw["order"])
        if [raw["order"] for raw in members] != list(range(len(members))):
            ids = [raw["id"] for raw in members]
            _fail(path, f"blocks {ids} must have orders 0, 1, ... with no gaps")
    return stacks, len(columns)


def _outside_grid(
    path: Path, raw_blocks: Dict[str, Dict[str, Any]], column_count: int, row: int
) -> Optional[LayoutBlock]:
    """The one block in *row*, -1 above the grid or 1 below it, or None."""
    found = [raw for raw in raw_blocks.values() if _row(raw) == row]
    if not found:
        return None
    if len(found) > 1:
        ids = [raw["id"] for raw in found]
        _fail(path, f"blocks {ids} are all in row {row}, which holds one block")
    if found[0]["column"] >= column_count:
        _fail(path, f"{found[0]['id']!r} must be in a column that has blocks in row 0")
    return _block(path, found[0], {})


def _stacked(
    path: Path, stacks: dict[StackKey, list[dict[str, Any]]], key: StackKey
) -> tuple["LayoutBlock", ...]:
    return tuple(_block(path, raw, stacks) for raw in stacks.get(key, []))


def _block(
    path: Path, raw: dict[str, Any], stacks: dict[StackKey, list[dict[str, Any]]]
) -> LayoutBlock:
    block_id, column = raw["id"], raw["column"]
    children = _stacked(path, stacks, (block_id, column))
    if children and {"content", "stall_level"} & set(raw):
        _fail(path, f"{block_id!r} has nested blocks, so its metrics go in them")
    items = tuple(_item(path, entry) for entry in raw.get("content", []))
    if raw["style"] == "list" and any(item.bar for item in items):
        _fail(path, f"{block_id!r} is a compact list, so its metrics have no bar")
    return LayoutBlock(
        id=block_id,
        title=raw["title"],
        column=column,
        row=_row(raw),
        style=raw["style"],
        parent=raw.get("parent"),
        stall_level=raw.get("stall_level"),
        items=items,
        children=children,
    )


def _check_metric_id(path: Path, raw: Dict[str, Any], where: str) -> None:
    if not isinstance(raw["metric_id"], str) or not raw["metric_id"]:
        _fail(path, f"metric_id of {where} must be a non-empty string")


def _item(path: Path, raw: dict[str, Any]) -> LayoutItem:
    where = repr(raw.get("metric_id"))
    _check_keys(path, raw, ITEM_KEYS, where)
    _check_metric_id(path, raw, where)
    _check_choice(path, raw["category"], _ITEM_CATEGORIES, f"category of {where}")
    if not isinstance(raw["bar"], bool):
        _fail(path, f"bar of {where} must be true or false, not {raw['bar']!r}")
    return LayoutItem(raw["metric_id"], raw["title"], raw["category"], raw["bar"])


def _arrow(path: Path, raw: dict[str, Any]) -> LayoutArrow:
    where = f"arrow {raw.get('metric_id')!r}"
    _check_keys(path, raw, ARROW_KEYS, where)
    _check_metric_id(path, raw, where)
    _check_choice(path, raw["direction"], _DIRECTIONS, f"direction of {where}")
    _check_choice(path, raw["category"], _ARROW_CATEGORIES, f"category of {where}")
    _check_choice(path, raw["style"], _ARROW_STYLES, f"style of {where}")
    return LayoutArrow(
        source=raw["from"],
        target=raw["to"],
        direction=raw["direction"],
        metric_id=raw["metric_id"],
        title=raw["title"],
        category=raw["category"],
        style=raw["style"],
        group=raw.get("group"),
    )


# ---------------------------------------------------------------------------
# Cross-reference checks
# ---------------------------------------------------------------------------


def _scopes(path: Path, entries: object, column_count: int) -> Tuple[LayoutScope, ...]:
    """The scopes, checked to cover columns 0 to the last in order, without gaps
    or overlaps."""
    if not isinstance(entries, list) or not entries:
        _fail(path, "scope must be a list of at least one label and its columns")
    scopes = tuple(_scope(path, raw) for raw in entries)
    starts = [scope.first_column for scope in scopes]
    next_starts = [scope.last_column + 1 for scope in scopes]
    if starts != [0, *next_starts[:-1]] or next_starts[-1] != column_count:
        spans = [[scope.first_column, scope.last_column] for scope in scopes]
        _fail(
            path,
            f"scope columns {spans} must cover columns 0 to {column_count - 1} "
            "in order, without gaps or overlaps",
        )
    return scopes


def _scope(path: Path, raw: object) -> LayoutScope:
    if not isinstance(raw, dict):
        _fail(path, f"a scope must be an object, not {raw!r}")
    where = f"scope {raw.get('label')!r}"
    _check_keys(path, raw, SCOPE_KEYS, where)
    if not isinstance(raw["label"], str) or not raw["label"]:
        _fail(path, f"label of {where} must be a non-empty string")
    columns = raw["columns"]
    if not isinstance(columns, list) or len(columns) != 2:
        _fail(path, f"columns of {where} must be [first, last], not {columns!r}")
    first, last = columns
    _check_index(path, first, f"first column of {where}")
    _check_index(path, last, f"last column of {where}")
    if last < first:
        _fail(path, f"columns of {where} must not end before they start")
    return LayoutScope(raw["label"], first, last)


def _check_references(layout: Layout, raw_blocks: dict[str, dict[str, Any]]) -> None:
    by_id = {block.id: block for block in layout.blocks()}
    # A block left out was in a parent cycle, so no column reaches it
    unplaced = sorted(set(raw_blocks) - set(by_id))
    if unplaced:
        _fail(layout.path, f"blocks {unplaced} nest inside each other")
    for arrow in layout.arrows:
        _check_arrow(layout, arrow, by_id)


def _check_arrow(
    layout: Layout, arrow: LayoutArrow, by_id: Dict[str, LayoutBlock]
) -> None:
    path, where = layout.path, f"arrow {arrow.metric_id!r}"
    for end in (arrow.source, arrow.target):
        if end not in by_id:
            _fail(path, f"{where} refers to unknown block {end!r}")
        if by_id[end].children:
            _fail(path, f"{where} must connect to a block nested in {end!r}")
    source, target = by_id[arrow.source], by_id[arrow.target]
    if target.row == 0:
        if source.row != 0 or target.column != source.column + 1:
            _fail(path, f"{where} must go from row 0 to a block in the next column")
        return
    stack = layout.columns[target.column]
    next_to_target = stack[0] if target.row < 0 else stack[-1]
    if source is not next_to_target:
        _fail(path, f"{where} to {target.id!r} must start at {next_to_target.id!r}")
    if arrow.style != "inline":
        _fail(path, f"{where} to a block above or below the grid must be inline")

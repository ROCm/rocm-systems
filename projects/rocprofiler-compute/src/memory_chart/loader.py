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
from typing import Any, Dict, List, NoReturn, Optional, Tuple, Union

import config
from membw_analysis.models import BOTTLENECK_LEVELS
from utils.logger import console_warning
from utils.utils_common import (
    canonical_config_arch,
    load_panel_configs,
    metric_ids_by_name,
)

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


# Format version of the layout files this code reads; bumped on breaking changes
LAYOUT_VERSION = 1
LAYOUT_KEYS = Keys(("version", "archs", "description", "scope", "blocks", "arrows"))
SCOPE_KEYS = Keys(("labels", "split"))
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
ITEM_KEYS = Keys(("metric_name", "metric_id", "title", "category", "bar"))
ARROW_KEYS = Keys(
    (
        "from",
        "to",
        "direction",
        "metric_name",
        "metric_id",
        "title",
        "category",
        "style",
        "group",
    ),
    ("group",),
)

_BLOCK_STYLES = ("panel", "list")
_ITEM_CATEGORIES = ("util", "hit", "stall", "neutral", "bw", "read", "write", "atomic")
_ARROW_CATEGORIES = ("read", "write", "atomic", "neutral")
_DIRECTIONS = ("forward", "backward", "both")
_ARROW_STYLES = ("inline", "stacked")

# (parent block id, or None for the top-level blocks of row 0; column) of a stack
StackKey = Tuple[Optional[str], int]


@dataclass(frozen=True)
class LayoutItem:
    """A metric shown in a block."""

    metric_name: str
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
    row: int  # 0 for the grid, negative above it, positive below it
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
    metric_name: str
    metric_id: str
    title: str
    category: str
    style: str  # "inline": one "title : value" line; "stacked": title, value, arrow
    group: Optional[str]


@dataclass(frozen=True)
class Layout:
    """A checked layout file."""

    path: Path
    archs: tuple[str, ...]
    description: str
    scope_labels: tuple[str, str]
    scope_split: str
    columns: Tuple[Tuple[LayoutBlock, ...], ...]  # top-level blocks of row 0
    off_grid: Tuple[LayoutBlock, ...]  # blocks outside row 0, by row, then column
    arrows: tuple[LayoutArrow, ...]

    def top_level_blocks(self) -> Iterator[LayoutBlock]:
        """The blocks stacked in the columns of row 0, column by column."""
        for column in self.columns:
            yield from column

    def blocks(self) -> Iterator[LayoutBlock]:
        """All blocks, column by column: row 0, each parent before its nested
        blocks, then the column's blocks in other rows."""
        for column, stack in enumerate(self.columns):
            for block in stack:
                yield from _with_descendants(block)
            yield from (block for block in self.off_grid if block.column == column)

    def rows(self) -> List[Tuple[int, List[LayoutBlock]]]:
        """The rows outside the grid, top to bottom, each with its blocks."""
        rows: Dict[int, List[LayoutBlock]] = defaultdict(list)
        for block in self.off_grid:
            rows[block.row].append(block)
        return sorted(rows.items())

    def inner_neighbor(self, block: LayoutBlock) -> LayoutBlock:
        """The block next to an off-grid block on the side toward row 0."""
        if abs(block.row) == 1:
            stack = self.columns[block.column]
            return stack[0] if block.row < 0 else stack[-1]
        cell = (_inner_row(block.row), block.column)
        return next(b for b in self.off_grid if (b.row, b.column) == cell)

    def metric_entries(self) -> List[Union[LayoutItem, LayoutArrow]]:
        """Every metric shown in a block, then every arrow."""
        return [*(item for b in self.blocks() for item in b.items), *self.arrows]

    def metrics(self) -> set[str]:
        """Names of all metrics the layout shows."""
        return {entry.metric_name for entry in self.metric_entries()}


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


def panel_metric_ids(arch: str) -> Dict[str, str]:
    """Id of every metric in the memory chart panel of *arch*, by metric name."""
    panel = PanelConfigs.for_arch(arch) or {}
    return {
        name: metric_id
        for source in panel.get("data source") or []
        for table in source.values()
        for name, metric_id in metric_ids_by_name(table).items()
    }


def metric_entries(layout_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every metric entry of a parsed layout file: block contents, then arrows."""
    blocks = layout_json["blocks"]
    items = [item for block in blocks for item in block.get("content", [])]
    return [*items, *layout_json["arrows"]]


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
    return layout_from_json(path, read_layout_json(path))


def read_layout_json(path: Path) -> Dict[str, Any]:
    """The parsed layout file, unchecked; raise LayoutError if it is not JSON."""
    try:
        layout_json = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:  # ValueError covers invalid JSON
        _fail(path, f"cannot read the layout: {error}")
    if not isinstance(layout_json, dict):
        _fail(path, "the layout must be a JSON object")
    return layout_json


def layout_from_json(path: Path, data: dict[str, Any]) -> Layout:
    """Check a parsed layout file and index its blocks; *path* names it in errors."""
    _check_keys(path, data, LAYOUT_KEYS, "the layout")
    _check_index(path, data["version"], "version")
    if data["version"] != LAYOUT_VERSION:
        _fail(path, f"version {data['version']} is not {LAYOUT_VERSION}")
    _check_keys(path, data["scope"], SCOPE_KEYS, "scope")
    if len(data["scope"]["labels"]) != 2:
        _fail(path, "scope needs two labels")
    raw_blocks = _raw_blocks(path, data["blocks"])
    groups, column_count = _stacks(path, raw_blocks)
    columns = tuple(
        _stacked(path, groups, (None, column)) for column in range(column_count)
    )
    left, right = data["scope"]["labels"]
    layout = Layout(
        path=path,
        archs=tuple(data["archs"]),
        description=data["description"],
        scope_labels=(left, right),
        scope_split=data["scope"]["split"],
        columns=columns,
        off_grid=_off_grid(path, raw_blocks, column_count),
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


def _inner_row(row: int) -> int:
    """The row next to *row* on the side toward row 0."""
    return row - 1 if row > 0 else row + 1


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
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(path, f"{what} must be a whole number, not {value!r}")


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
    """Check the row and order of a block, and what a block outside row 0 may have."""
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
            f"{where} is outside row 0: it needs the panel style, "
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


def _off_grid(
    path: Path, raw_blocks: Dict[str, Dict[str, Any]], column_count: int
) -> Tuple[LayoutBlock, ...]:
    """The blocks outside row 0, one per row and column, sorted by row, then column."""
    cells: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for raw in raw_blocks.values():
        row, column = _row(raw), raw["column"]
        if row == 0:
            continue
        if (row, column) in cells:
            ids = [cells[(row, column)]["id"], raw["id"]]
            _fail(path, f"blocks {ids} are both in row {row} of column {column}")
        if column >= column_count:
            _fail(path, f"{raw['id']!r} must be in a column that has blocks in row 0")
        cells[(row, column)] = raw
    for (row, column), raw in cells.items():
        inner_row = _inner_row(row)
        if inner_row and (inner_row, column) not in cells:
            _fail(
                path,
                f"{raw['id']!r} needs a block in row {inner_row} of its column",
            )
    return tuple(_block(path, cells[cell], {}) for cell in sorted(cells))


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


def _check_metric_ref(path: Path, raw: Dict[str, Any], where: str) -> None:
    for key in ("metric_name", "metric_id"):
        value = raw[key]
        if not isinstance(value, str) or not value:
            _fail(path, f"{key} of {where} must be a non-empty string, not {value!r}")


def _item(path: Path, raw: dict[str, Any]) -> LayoutItem:
    where = repr(raw.get("metric_name"))
    _check_keys(path, raw, ITEM_KEYS, where)
    _check_metric_ref(path, raw, where)
    _check_choice(path, raw["category"], _ITEM_CATEGORIES, f"category of {where}")
    if not isinstance(raw["bar"], bool):
        _fail(path, f"bar of {where} must be true or false, not {raw['bar']!r}")
    return LayoutItem(
        raw["metric_name"], raw["metric_id"], raw["title"], raw["category"], raw["bar"]
    )


def _arrow(path: Path, raw: dict[str, Any]) -> LayoutArrow:
    where = f"arrow {raw.get('metric_name')!r}"
    _check_keys(path, raw, ARROW_KEYS, where)
    _check_metric_ref(path, raw, where)
    _check_choice(path, raw["direction"], _DIRECTIONS, f"direction of {where}")
    _check_choice(path, raw["category"], _ARROW_CATEGORIES, f"category of {where}")
    _check_choice(path, raw["style"], _ARROW_STYLES, f"style of {where}")
    return LayoutArrow(
        source=raw["from"],
        target=raw["to"],
        direction=raw["direction"],
        metric_name=raw["metric_name"],
        metric_id=raw["metric_id"],
        title=raw["title"],
        category=raw["category"],
        style=raw["style"],
        group=raw.get("group"),
    )


# ---------------------------------------------------------------------------
# Cross-reference checks
# ---------------------------------------------------------------------------


def _check_references(layout: Layout, raw_blocks: dict[str, dict[str, Any]]) -> None:
    by_id = {block.id: block for block in layout.blocks()}
    # A block left out was in a parent cycle, so no column reaches it
    unplaced = sorted(set(raw_blocks) - set(by_id))
    if unplaced:
        _fail(layout.path, f"blocks {unplaced} nest inside each other")
    for arrow in layout.arrows:
        _check_arrow(layout, arrow, by_id)
    after_first = {b.id for b in layout.top_level_blocks() if b.column > 0}
    if layout.scope_split not in after_first:
        _fail(
            layout.path,
            f"scope split {layout.scope_split!r} must be a top-level block "
            "after the first column",
        )


def _check_arrow(
    layout: Layout, arrow: LayoutArrow, by_id: Dict[str, LayoutBlock]
) -> None:
    path, where = layout.path, f"arrow {arrow.metric_name!r}"
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
    if layout.inner_neighbor(target) is not source:
        _fail(
            path,
            f"{where} to {target.id!r} must start at the block next to it "
            "on the side toward row 0",
        )
    if arrow.style != "inline":
        _fail(path, f"{where} to a block outside row 0 must be inline")

# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Load memory chart layouts and resolve them into blocks and arrows.

A layout file states blocks, nesting, and the metric on each block and arrow.
This module checks the file and derives the rest: each block's column, parent
and position, and each arrow's direction.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, Optional

import config
from utils.logger import console_error
from utils.utils_common import canonical_config_arch

LAYOUTS_DIR = config.rocprof_compute_home / "memory_chart" / "layouts"


@dataclass(frozen=True)
class Keys:
    """Keys of one part of a layout file."""

    required: tuple[str, ...]
    optional: tuple[str, ...] = ()

    @property
    def order(self) -> tuple[str, ...]:
        """The order a layout file lists them in."""
        return self.required + self.optional


LAYOUT_KEYS = Keys(("archs", "description", "scope", "columns", "arrows"))
SCOPE_KEYS = Keys(("labels", "split"))
BLOCK_KEYS = Keys(
    ("id", "title"),
    ("note", "stall_level", "metrics", "children", "above", "below"),
)
NESTED_BLOCK_KEYS = Keys(
    ("id", "title"), ("note", "stall_level", "metrics", "children")
)
ATTACHED_BLOCK_KEYS = Keys(("id", "title"))  # above/below blocks are labelled boxes
ITEM_KEYS = Keys(("metric", "title", "category"))
ARROW_KEYS = Keys(("from", "to", "metric", "title", "category"), ("group",))

_ITEM_CATEGORIES = frozenset({
    "util",
    "hit",
    "stall",
    "neutral",
    "bw",
    "read",
    "write",
    "atomic",
})
# Arrow category -> direction it implies
_ARROW_DIRECTIONS = {
    "read": "backward",
    "write": "forward",
    "atomic": "both",
    "neutral": "both",
}
_STALL_LEVELS = frozenset({"GL1", "GL2", "EA"})


@dataclass(frozen=True)
class LayoutItem:
    """A metric shown in a block."""

    metric: str
    title: str
    category: str


@dataclass(frozen=True)
class LayoutBlock:
    """A block and where it sits."""

    id: str
    title: str
    column: int
    position: str  # "grid", "above", or "below"
    parent: Optional[str]  # enclosing block of a nested block
    host: Optional[str]  # block that an above/below block hangs off
    note: Optional[str]
    stall_level: Optional[str]  # membw level whose active stalls are listed here
    items: tuple[LayoutItem, ...]
    children: tuple["LayoutBlock", ...]
    above: tuple["LayoutBlock", ...]
    below: tuple["LayoutBlock", ...]


@dataclass(frozen=True)
class LayoutArrow:
    """A metric drawn between two blocks."""

    source: str
    target: str
    metric: str
    title: str
    category: str
    direction: str
    group: Optional[str]


@dataclass(frozen=True)
class Layout:
    """A resolved layout file."""

    path: Path
    archs: tuple[str, ...]
    description: str
    scope_labels: tuple[str, str]
    scope_split: str
    columns: tuple[tuple[LayoutBlock, ...], ...]
    arrows: tuple[LayoutArrow, ...]

    def blocks(self) -> Iterator[LayoutBlock]:
        """All blocks, each parent before its nested and attached blocks."""
        for column in self.columns:
            for block in column:
                yield from _with_descendants(block)

    def metrics(self) -> set[str]:
        """Names of all metrics the layout shows."""
        names = {item.metric for b in self.blocks() for item in b.items}
        return names | {arrow.metric for arrow in self.arrows}


class Layouts:
    """The shipped layouts, read once and indexed by architecture."""

    _by_arch: Optional[dict[str, Layout]] = None

    @classmethod
    def for_arch(cls, gpu_arch: Optional[str]) -> Optional[Layout]:
        """The layout for a GPU architecture, or None if it has no chart."""
        if cls._by_arch is None:
            cls._by_arch = _index_by_arch(layout_files())
        return cls._by_arch.get(canonical_config_arch(gpu_arch) or "")


def layout_files() -> list[Path]:
    return sorted(LAYOUTS_DIR.glob("*.json"))


def load_layout(path: Path) -> Layout:
    """Read and resolve a layout file; a malformed file is an error."""
    layout = _resolve(path, json.loads(path.read_text(encoding="utf-8")))
    _check_references(layout)
    return layout


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _fail(path: Path, problem: str) -> NoReturn:  # console_error exits
    console_error("memory chart", f"{path.name}: {problem}")


def _index_by_arch(paths: list[Path]) -> dict[str, Layout]:
    index: dict[str, Layout] = {}
    for path in paths:
        layout = load_layout(path)
        for arch in layout.archs:
            if arch in index:
                _fail(path, f"{arch} is also listed by {index[arch].path.name}")
            index[arch] = layout
    return index


def _with_descendants(block: LayoutBlock) -> Iterator[LayoutBlock]:
    yield block
    for sub in (*block.children, *block.above, *block.below):
        yield from _with_descendants(sub)


def _check_keys(path: Path, raw: dict[str, Any], keys: Keys, where: str) -> None:
    missing = set(keys.required) - set(raw)
    unknown = set(raw) - set(keys.order)
    if missing:
        _fail(path, f"missing {sorted(missing)} in {where}")
    if unknown:
        _fail(path, f"unknown keys {sorted(unknown)} in {where}")


def _resolve(path: Path, data: dict[str, Any]) -> Layout:
    _check_keys(path, data, LAYOUT_KEYS, "the layout")
    _check_keys(path, data["scope"], SCOPE_KEYS, "scope")
    if len(data["scope"]["labels"]) != 2:
        _fail(path, "scope needs two labels")
    if not data["columns"] or not all(data["columns"]):
        _fail(path, "every column must have a block")
    left, right = data["scope"]["labels"]
    return Layout(
        path=path,
        archs=tuple(data["archs"]),
        description=data["description"],
        scope_labels=(left, right),
        scope_split=data["scope"]["split"],
        columns=tuple(
            tuple(_block(path, raw, col) for raw in column)
            for col, column in enumerate(data["columns"])
        ),
        arrows=tuple(_arrow(path, raw) for raw in data["arrows"]),
    )


def _block(
    path: Path,
    raw: dict[str, Any],
    column: int,
    position: str = "grid",
    parent: Optional[str] = None,
    host: Optional[str] = None,
) -> LayoutBlock:
    where = repr(raw.get("id"))
    if position != "grid":
        keys = ATTACHED_BLOCK_KEYS
    else:
        keys = NESTED_BLOCK_KEYS if parent else BLOCK_KEYS
    _check_keys(path, raw, keys, where)
    if raw.get("stall_level") not in (None, *_STALL_LEVELS):
        _fail(path, f"unknown stall_level {raw['stall_level']!r} on {where}")
    if "children" in raw and ({"metrics", "stall_level"} & set(raw)):
        _fail(path, f"{where} has nested blocks, so its metrics go in them")
    block_id = raw["id"]
    return LayoutBlock(
        id=block_id,
        title=raw["title"],
        column=column,
        position=position,
        parent=parent,
        host=host,
        note=raw.get("note"),
        stall_level=raw.get("stall_level"),
        items=tuple(_item(path, entry) for entry in raw.get("metrics", [])),
        children=tuple(
            _block(path, child, column, parent=block_id)
            for child in raw.get("children", [])
        ),
        above=tuple(
            _block(path, sub, column, "above", host=block_id)
            for sub in raw.get("above", [])
        ),
        below=tuple(
            _block(path, sub, column, "below", host=block_id)
            for sub in raw.get("below", [])
        ),
    )


def _item(path: Path, raw: dict[str, Any]) -> LayoutItem:
    _check_keys(path, raw, ITEM_KEYS, repr(raw.get("metric")))
    if raw["category"] not in _ITEM_CATEGORIES:
        _fail(path, f"unknown category {raw['category']!r} for {raw['metric']!r}")
    return LayoutItem(raw["metric"], raw["title"], raw["category"])


def _arrow(path: Path, raw: dict[str, Any]) -> LayoutArrow:
    _check_keys(path, raw, ARROW_KEYS, f"arrow {raw.get('metric')!r}")
    category = raw["category"]
    if category not in _ARROW_DIRECTIONS:
        _fail(path, f"arrow {raw['metric']!r} has category {category!r}")
    return LayoutArrow(
        source=raw["from"],
        target=raw["to"],
        metric=raw["metric"],
        title=raw["title"],
        category=category,
        direction=_ARROW_DIRECTIONS[category],
        group=raw.get("group"),
    )


# ---------------------------------------------------------------------------
# Cross-reference checks
# ---------------------------------------------------------------------------


def _check_references(layout: Layout) -> None:
    by_id = _unique_blocks(layout)
    for arrow in layout.arrows:
        _check_arrow(layout.path, arrow, by_id)
    after_first = {b.id for column in layout.columns[1:] for b in column}
    if layout.scope_split not in after_first:
        _fail(
            layout.path,
            f"scope split {layout.scope_split!r} must be after the first column",
        )


def _unique_blocks(layout: Layout) -> dict[str, LayoutBlock]:
    by_id: dict[str, LayoutBlock] = {}
    for block in layout.blocks():
        if block.id in by_id:
            _fail(layout.path, f"duplicate block id {block.id!r}")
        by_id[block.id] = block
    return by_id


def _check_arrow(path: Path, arrow: LayoutArrow, by_id: dict[str, LayoutBlock]) -> None:
    for end in (arrow.source, arrow.target):
        if end not in by_id:
            _fail(path, f"arrow {arrow.metric!r} refers to unknown block {end!r}")
        if by_id[end].children:
            _fail(
                path,
                f"arrow {arrow.metric!r} must connect to a block nested in {end!r}",
            )
    source, target = by_id[arrow.source], by_id[arrow.target]
    if source.position != "grid":
        _fail(path, f"arrow {arrow.metric!r} must start at a block in a column")
    attached = target.host == arrow.source
    next_column = target.position == "grid" and target.column == source.column + 1
    if not (attached or next_column):
        _fail(
            path,
            f"arrow {arrow.metric!r} must go to a block in the next column "
            "or to a block attached to its source",
        )

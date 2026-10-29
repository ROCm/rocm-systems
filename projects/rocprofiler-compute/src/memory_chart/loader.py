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
from typing import Any, NoReturn, Optional

import config
from membw_analysis.models import BOTTLENECK_LEVELS
from utils.logger import console_warning
from utils.utils_common import canonical_config_arch, load_panel_configs

LAYOUTS_DIR = config.rocprof_compute_home / "memory_chart" / "layouts"


class LayoutError(Exception):
    """A layout file that cannot be read or breaks the format rules."""


@dataclass(frozen=True)
class Keys:
    """Keys of one part of a layout file."""

    required: tuple[str, ...]
    optional: tuple[str, ...] = ()

    @property
    def order(self) -> tuple[str, ...]:
        """The order a layout file lists them in."""
        return self.required + self.optional


LAYOUT_KEYS = Keys(("archs", "description", "scope", "blocks", "arrows"))
SCOPE_KEYS = Keys(("labels", "split"))
BLOCK_KEYS = Keys(
    ("id", "title", "column", "order", "position", "style"),
    ("parent", "host", "stall_level", "content"),
)
ITEM_KEYS = Keys(("metric", "title", "category", "bar"))
ARROW_KEYS = Keys(
    ("from", "to", "direction", "metric", "title", "category", "style"), ("group",)
)

_POSITIONS = ("grid", "above", "below")
_BLOCK_STYLES = ("panel", "list")
_ITEM_CATEGORIES = ("util", "hit", "stall", "neutral", "bw", "read", "write", "atomic")
_ARROW_CATEGORIES = ("read", "write", "atomic", "neutral")
_DIRECTIONS = ("forward", "backward", "both")
_ARROW_STYLES = ("inline", "stacked")

# (owner block id or None, position, column) of a stack of blocks
StackKey = tuple[Optional[str], str, int]


@dataclass(frozen=True)
class LayoutItem:
    """A metric shown in a block."""

    metric: str
    title: str
    category: str
    bar: bool  # draw a progress bar under the value


@dataclass(frozen=True)
class LayoutBlock:
    """A block and where it sits."""

    id: str
    title: str
    column: int
    position: str  # "grid", "above", or "below"
    style: str  # "panel", or "list" for a compact list
    parent: Optional[str]  # enclosing block of a nested block
    host: Optional[str]  # block that an above/below block hangs off
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
    direction: str
    metric: str
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
    columns: tuple[tuple[LayoutBlock, ...], ...]  # top-level grid blocks
    arrows: tuple[LayoutArrow, ...]

    def top_level_blocks(self) -> Iterator[LayoutBlock]:
        """The blocks stacked in the columns, column by column."""
        for column in self.columns:
            yield from column

    def blocks(self) -> Iterator[LayoutBlock]:
        """All blocks, each parent before its nested and attached blocks."""
        for block in self.top_level_blocks():
            yield from _with_descendants(block)

    def metrics(self) -> set[str]:
        """Names of all metrics the layout shows."""
        names = {item.metric for b in self.blocks() for item in b.items}
        return names | {arrow.metric for arrow in self.arrows}


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


def has_layout(gpu_arch: Optional[str]) -> bool:
    """Return whether a usable memory-chart layout is available for gpu_arch."""
    return Layouts.for_arch(gpu_arch) is not None


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
    _check_keys(path, data["scope"], SCOPE_KEYS, "scope")
    if len(data["scope"]["labels"]) != 2:
        _fail(path, "scope needs two labels")
    raw_blocks = _raw_blocks(path, data["blocks"])
    groups, column_count = _stacks(path, raw_blocks)
    columns = tuple(
        _stacked(path, groups, (None, "grid", column)) for column in range(column_count)
    )
    left, right = data["scope"]["labels"]
    layout = Layout(
        path=path,
        archs=tuple(data["archs"]),
        description=data["description"],
        scope_labels=(left, right),
        scope_split=data["scope"]["split"],
        columns=columns,
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
    for sub in (*block.children, *block.above, *block.below):
        yield from _with_descendants(sub)


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


def _raw_blocks(path: Path, entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The file's blocks by id, each checked on its own."""
    by_id: dict[str, dict[str, Any]] = {}
    for raw in entries:
        where = repr(raw.get("id"))
        _check_keys(path, raw, BLOCK_KEYS, where)
        if raw["id"] in by_id:
            _fail(path, f"duplicate block id {raw['id']!r}")
        _check_index(path, raw["column"], f"column of {where}")
        _check_index(path, raw["order"], f"order of {where}")
        _check_choice(path, raw["position"], _POSITIONS, f"position of {where}")
        _check_choice(path, raw["style"], _BLOCK_STYLES, f"style of {where}")
        _check_choice(
            path,
            raw.get("stall_level"),
            (None, *sorted(BOTTLENECK_LEVELS)),
            f"stall_level of {where}",
        )
        if raw["position"] == "grid" and "host" in raw:
            _fail(path, f"{where} is in the grid, so it has no host")
        if raw["position"] != "grid":
            extra = {"parent", "content", "stall_level"} & set(raw)
            if "host" not in raw or extra or raw["style"] != "panel":
                _fail(
                    path,
                    f"{where} is drawn {raw['position']} a block: it needs a host, "
                    "the panel style, and no parent, content, or stall_level",
                )
        by_id[raw["id"]] = raw
    return by_id


def _stacks(
    path: Path, raw_blocks: dict[str, dict[str, Any]]
) -> tuple[dict[StackKey, list[dict[str, Any]]], int]:
    """Blocks grouped into the stacks they are drawn in, each sorted by order.

    A stack is keyed by (owner, position, column): the owner is the parent or host,
    or None for the top-level blocks of a column. Also returns the column count.
    """
    stacks: dict[StackKey, list[dict[str, Any]]] = defaultdict(list)
    for raw in raw_blocks.values():
        owner_id = raw.get("parent") or raw.get("host")
        owner = raw_blocks.get(owner_id) if owner_id else None
        if owner_id and owner is None:
            _fail(path, f"{raw['id']!r} refers to unknown block {owner_id!r}")
        if owner and (
            owner["position"] != "grid" or ("host" in raw and "parent" in owner)
        ):
            _fail(path, f"{raw['id']!r} must hang off a block in a column")
        if owner and owner["column"] != raw["column"]:
            _fail(path, f"{raw['id']!r} must have the column of {owner_id!r}")
        stacks[(owner_id, raw["position"], raw["column"])].append(raw)
    columns = sorted(column for owner, _, column in stacks if owner is None)
    if not columns or columns != list(range(len(columns))):
        _fail(path, f"columns must be numbered 0, 1, ... with no gaps, not {columns}")
    for members in stacks.values():
        members.sort(key=lambda raw: raw["order"])
        if [raw["order"] for raw in members] != list(range(len(members))):
            ids = [raw["id"] for raw in members]
            _fail(path, f"blocks {ids} must have orders 0, 1, ... with no gaps")
    return stacks, len(columns)


def _stacked(
    path: Path, stacks: dict[StackKey, list[dict[str, Any]]], key: StackKey
) -> tuple["LayoutBlock", ...]:
    return tuple(_block(path, raw, stacks) for raw in stacks.get(key, []))


def _block(
    path: Path, raw: dict[str, Any], stacks: dict[StackKey, list[dict[str, Any]]]
) -> LayoutBlock:
    block_id, column = raw["id"], raw["column"]
    children = _stacked(path, stacks, (block_id, "grid", column))
    if children and {"content", "stall_level"} & set(raw):
        _fail(path, f"{block_id!r} has nested blocks, so its metrics go in them")
    items = tuple(_item(path, entry) for entry in raw.get("content", []))
    if raw["style"] == "list" and any(item.bar for item in items):
        _fail(path, f"{block_id!r} is a compact list, so its metrics have no bar")
    return LayoutBlock(
        id=block_id,
        title=raw["title"],
        column=raw["column"],
        position=raw["position"],
        style=raw["style"],
        parent=raw.get("parent"),
        host=raw.get("host"),
        stall_level=raw.get("stall_level"),
        items=items,
        children=children,
        above=_stacked(path, stacks, (block_id, "above", column)),
        below=_stacked(path, stacks, (block_id, "below", column)),
    )


def _item(path: Path, raw: dict[str, Any]) -> LayoutItem:
    where = repr(raw.get("metric"))
    _check_keys(path, raw, ITEM_KEYS, where)
    _check_choice(path, raw["category"], _ITEM_CATEGORIES, f"category of {where}")
    if not isinstance(raw["bar"], bool):
        _fail(path, f"bar of {where} must be true or false, not {raw['bar']!r}")
    return LayoutItem(raw["metric"], raw["title"], raw["category"], raw["bar"])


def _arrow(path: Path, raw: dict[str, Any]) -> LayoutArrow:
    where = f"arrow {raw.get('metric')!r}"
    _check_keys(path, raw, ARROW_KEYS, where)
    _check_choice(path, raw["direction"], _DIRECTIONS, f"direction of {where}")
    _check_choice(path, raw["category"], _ARROW_CATEGORIES, f"category of {where}")
    _check_choice(path, raw["style"], _ARROW_STYLES, f"style of {where}")
    return LayoutArrow(
        source=raw["from"],
        target=raw["to"],
        direction=raw["direction"],
        metric=raw["metric"],
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
        _check_arrow(layout.path, arrow, by_id)
    after_first = {b.id for b in layout.top_level_blocks() if b.column > 0}
    if layout.scope_split not in after_first:
        _fail(
            layout.path,
            f"scope split {layout.scope_split!r} must be a top-level block "
            "after the first column",
        )


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
    if attached and arrow.style != "inline":
        _fail(path, f"arrow {arrow.metric!r} to an attached block must be inline")

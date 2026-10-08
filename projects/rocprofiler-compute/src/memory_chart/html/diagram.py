# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Adapt resolved memory chart layouts to the HTML diagram and metric slots."""

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Mapping, Optional, Tuple, TypedDict

from ..loader import Layout, LayoutArrow, LayoutBlock
from ..mem_chart import _BAR_CATEGORIES
from ..units import PLAIN, DisplayUnit, display_unit

ArrowGroup = TypedDict(
    "ArrowGroup",
    {"from": str, "to": str, "lanes": List[Dict[str, object]]},
)


@dataclass(frozen=True)
class SlotSpec:
    """Describe one metric value displayed in the diagram."""

    slot_id: str
    metric: str
    unit: DisplayUnit
    category: str
    on_arrow: bool
    cu_block: bool
    bar: bool


def diagram_payload(
    layout: Layout, units: Optional[Mapping[str, str]] = None
) -> Dict[str, object]:
    """Copy loader-resolved topology and config-owned bar metadata for the page."""
    specs = {spec.slot_id: spec for spec in slot_specs(layout, units or {})}
    grid_blocks = [
        _block_payload(block, order, specs)
        for column in layout.columns
        for order, block in enumerate(column)
    ]
    io_blocks = [
        _block_payload(block, order, specs)
        for order, block in enumerate(layout.blocks())
        if block.position != "grid"
    ]
    split = next(block for block in layout.blocks() if block.id == layout.scope_split)
    return {
        "arch": layout.path.stem,
        "gridBlocks": grid_blocks,
        "ioBlocks": io_blocks,
        "arrows": _arrow_groups(layout.arrows),
        "scope": {
            "splitColumn": split.column,
            "splitBlock": split.id,
            "gpuLabel": layout.scope_labels[0],
            "memoryLabel": layout.scope_labels[1],
        },
    }


def slot_specs(layout: Layout, units: Mapping[str, str]) -> Tuple[SlotSpec, ...]:
    """Describe block slots in declared column order, followed by arrow slots."""
    specs: List[SlotSpec] = []
    for block in layout.blocks():
        for index, item in enumerate(block.items):
            unit = display_unit(units.get(item.metric)) or PLAIN
            specs.append(
                SlotSpec(
                    slot_id=f"{block.id}.{index}",
                    metric=item.metric,
                    unit=unit,
                    category=item.category,
                    on_arrow=False,
                    cu_block=block.column == 0,
                    bar=(
                        unit.kind == "percent"
                        and item.category in _BAR_CATEGORIES
                        and block.column != 0
                    ),
                )
            )
    specs.extend(
        SlotSpec(
            slot_id=f"arrow.{index}",
            metric=arrow.metric,
            unit=display_unit(units.get(arrow.metric)) or PLAIN,
            category=arrow.category,
            on_arrow=True,
            cu_block=False,
            bar=False,
        )
        for index, arrow in enumerate(layout.arrows)
    )
    return tuple(specs)


def diagram_metrics(layout: Layout) -> FrozenSet[str]:
    """Return the metric set already resolved by the loader."""
    return frozenset(layout.metrics())


def _block_payload(
    block: LayoutBlock, order: int, specs: Mapping[str, SlotSpec]
) -> Dict[str, object]:
    """Preserve display fields, nesting, and each attachment's host and column."""
    return {
        "id": block.id,
        "title": block.title,
        "column": block.column,
        "order": order,
        "position": block.position,
        "parent": block.parent,
        "host": block.host,
        "note": block.note,
        "stallLevel": block.stall_level,
        "content": [
            {
                "metric": item.metric,
                "title": item.title,
                "category": item.category,
                "slotId": f"{block.id}.{index}",
                "bar": specs[f"{block.id}.{index}"].bar,
            }
            for index, item in enumerate(block.items)
        ],
        "children": [
            _block_payload(child, index, specs)
            for index, child in enumerate(block.children)
        ],
    }


def _arrow_groups(arrows: Tuple[LayoutArrow, ...]) -> List[ArrowGroup]:
    """Collect lanes by endpoint pair without changing either declared order."""
    groups: Dict[Tuple[str, str], ArrowGroup] = {}
    for index, arrow in enumerate(arrows):
        pair = (arrow.source, arrow.target)
        if pair not in groups:
            groups[pair] = {"from": pair[0], "to": pair[1], "lanes": []}
        lanes = groups[pair]["lanes"]
        previous_group = lanes[-1].get("group") if lanes else None
        lanes.append({
            "from": arrow.source,
            "to": arrow.target,
            "metric": arrow.metric,
            "title": arrow.title,
            "category": arrow.category,
            "direction": arrow.direction,
            "group": arrow.group,
            "slotId": f"arrow.{index}",
            "groupHeader": arrow.group if arrow.group != previous_group else None,
        })
    return list(groups.values())

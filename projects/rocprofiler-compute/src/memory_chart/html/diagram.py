# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Turn a memory chart layout into a diagram model and metric slots."""

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Mapping, Set, Tuple, cast

_BAR_CATEGORIES = frozenset({"hit", "util", "stall"})


@dataclass(frozen=True)
class SlotSpec:
    """Describe one metric value displayed in the diagram."""

    slot_id: str
    metric: str
    unit: str
    category: str
    on_arrow: bool
    cu_block: bool
    bar: bool


def diagram_payload(layout: Mapping[str, object]) -> Dict[str, object]:
    """Build the ordered diagram model after checking block relationships."""
    blocks = cast(List[Dict[str, object]], layout["blocks"])
    arrows = cast(List[Dict[str, object]], layout["arrows"])
    blocks_by_id = {cast(str, block["id"]): block for block in blocks}
    child_ids = _validate_relationships(blocks, arrows, blocks_by_id)

    grid_blocks: List[Dict[str, object]] = []
    io_blocks: List[Dict[str, object]] = []
    for block in blocks:
        if block["id"] in child_ids:
            continue
        payload = _block_payload(block, blocks_by_id)
        if block.get("position", "grid") == "grid":
            grid_blocks.append(payload)
        else:
            io_blocks.append(payload)

    grid_blocks.sort(key=lambda block: (block["column"], block["order"]))
    return {
        "arch": layout["arch"],
        "gridBlocks": grid_blocks,
        "ioBlocks": io_blocks,
        "arrows": _arrow_groups(arrows),
    }


def slot_specs(layout: Mapping[str, object]) -> Tuple[SlotSpec, ...]:
    """Describe content slots in block order, followed by arrow slots."""
    specs: List[SlotSpec] = []
    for block in cast(List[Dict[str, object]], layout["blocks"]):
        for index, item in enumerate(cast(List[Dict[str, object]], block["content"])):
            category = cast(str, item["category"])
            default_unit = "%" if category in _BAR_CATEGORIES else ""
            unit = cast(str, item.get("unit", default_unit))
            specs.append(
                SlotSpec(
                    slot_id=f"{block['id']}.{index}",
                    metric=cast(str, item["metric"]),
                    unit=unit,
                    category=category,
                    on_arrow=False,
                    cu_block=block["column"] == 0,
                    bar=unit == "%" and category in _BAR_CATEGORIES,
                )
            )

    for index, arrow in enumerate(cast(List[Dict[str, object]], layout["arrows"])):
        metric = cast(str, arrow["metric"])
        metric_lower = metric.lower()
        unit = "Bytes/s" if "bw" in metric_lower or "bandwidth" in metric_lower else ""
        specs.append(
            SlotSpec(
                slot_id=f"arrow.{index}",
                metric=metric,
                unit=unit,
                category=cast(str, arrow["category"]),
                on_arrow=True,
                cu_block=False,
                bar=False,
            )
        )
    return tuple(specs)


def diagram_metrics(layout: Mapping[str, object]) -> FrozenSet[str]:
    """Return every metric referenced by content and arrows."""
    metrics = {
        cast(str, item["metric"])
        for block in cast(List[Dict[str, object]], layout["blocks"])
        for item in cast(List[Dict[str, object]], block["content"])
    }
    metrics.update(
        cast(str, arrow["metric"])
        for arrow in cast(List[Dict[str, object]], layout["arrows"])
    )
    return frozenset(metrics)


def _validate_relationships(
    blocks: List[Dict[str, object]],
    arrows: List[Dict[str, object]],
    blocks_by_id: Mapping[str, Dict[str, object]],
) -> Set[str]:
    """Check references and parenting before a recursive block is built."""
    child_ids: Set[str] = set()
    parent_by_child: Dict[str, str] = {}
    for block in blocks:
        for child_id in cast(List[str], block.get("children", [])):
            if child_id not in blocks_by_id:
                raise ValueError(f"Unknown child block {child_id!r}")
            if child_id in child_ids:
                raise ValueError(f"Child block {child_id!r} has several parents")
            if blocks_by_id[child_id]["column"] != block["column"]:
                raise ValueError(f"Child block {child_id!r} is in another column")
            child_ids.add(child_id)
            parent_by_child[child_id] = cast(str, block["id"])

    for arrow in arrows:
        for endpoint in ("from", "to"):
            if arrow[endpoint] not in blocks_by_id:
                raise ValueError(f"Unknown arrow endpoint {arrow[endpoint]!r}")

    _validate_acyclic_children(parent_by_child)
    return child_ids


def _validate_acyclic_children(parent_by_child: Mapping[str, str]) -> None:
    """Reject recursive children before building their nested payloads."""
    for child_id in parent_by_child:
        ancestors: Set[str] = set()
        current = child_id
        while current in parent_by_child:
            if current in ancestors:
                raise ValueError(f"Child block {current!r} forms a cycle")
            ancestors.add(current)
            current = parent_by_child[current]


def _block_payload(
    block: Dict[str, object], blocks_by_id: Mapping[str, Dict[str, object]]
) -> Dict[str, object]:
    """Copy display fields and nest children in their declared order."""
    payload = dict(block)
    payload["order"] = block.get("order", 0)
    payload["position"] = block.get("position", "grid")
    payload["content"] = [
        {**item, "slotId": f"{block['id']}.{index}"}
        for index, item in enumerate(cast(List[Dict[str, object]], block["content"]))
    ]
    payload["children"] = [
        _block_payload(blocks_by_id[child_id], blocks_by_id)
        for child_id in cast(List[str], block.get("children", []))
    ]
    return payload


def _arrow_groups(arrows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Collect lanes by endpoint pair without changing either file order."""
    groups: Dict[Tuple[str, str], Dict[str, object]] = {}
    for index, arrow in enumerate(arrows):
        pair = (cast(str, arrow["from"]), cast(str, arrow["to"]))
        if pair not in groups:
            groups[pair] = {"from": pair[0], "to": pair[1], "lanes": []}
        lanes = cast(List[Dict[str, object]], groups[pair]["lanes"])
        previous_group = lanes[-1].get("group") if lanes else None
        lane = dict(arrow)
        lane["slotId"] = f"arrow.{index}"
        lane["groupHeader"] = (
            arrow.get("group") if arrow.get("group") != previous_group else None
        )
        lanes.append(lane)
    return list(groups.values())

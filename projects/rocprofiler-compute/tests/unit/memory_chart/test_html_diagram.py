# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Resolved layouts retain their topology, metrics, and config units in HTML."""

from dataclasses import replace

import pytest

from memory_chart.html.diagram import diagram_metrics, diagram_payload, slot_specs
from memory_chart.loader import Layouts, load_layout
from memory_chart.units import PLAIN, display_unit, panel_units
from tests.unit.memory_chart.layout_cases import (
    LAYOUT_ARCH_IDS,
    LAYOUT_ARCHS,
    panel_config,
    panel_metric_names,
)


def flatten(blocks):
    for block in blocks:
        yield block
        yield from flatten(block["children"])


@pytest.mark.parametrize(("path", "arch"), LAYOUT_ARCHS, ids=LAYOUT_ARCH_IDS)
def test_resolved_payload_preserves_every_block_lane_and_slot(path, arch):
    layout = load_layout(path)
    units = panel_units(panel_config(arch))
    payload = diagram_payload(layout, units)
    assert set(payload) == {"arch", "gridBlocks", "ioBlocks", "arrows", "scope"}
    assert payload["arch"] == path.stem
    assert [block["id"] for block in payload["gridBlocks"]] == [
        block.id for column in layout.columns for block in column
    ]
    blocks = list(flatten(payload["gridBlocks"])) + payload["ioBlocks"]
    by_id = {block["id"]: block for block in blocks}
    assert len(by_id) == len(blocks) == len(list(layout.blocks()))
    for block in layout.blocks():
        shown = by_id[block.id]
        for key in ("column", "position", "host", "parent", "note"):
            assert shown[key] == getattr(block, key)
        assert shown["stallLevel"] == block.stall_level
        assert [child["id"] for child in shown["children"]] == [
            child.id for child in block.children
        ]
    assert payload["scope"] == {
        "splitColumn": by_id[layout.scope_split]["column"],
        "splitBlock": layout.scope_split,
        "gpuLabel": layout.scope_labels[0],
        "memoryLabel": layout.scope_labels[1],
    }
    pairs = list(dict.fromkeys((a.source, a.target) for a in layout.arrows))
    assert [(g["from"], g["to"]) for g in payload["arrows"]] == pairs
    for group in payload["arrows"]:
        expected = [
            (index, arrow)
            for index, arrow in enumerate(layout.arrows)
            if (arrow.source, arrow.target) == (group["from"], group["to"])
        ]
        assert [lane["slotId"] for lane in group["lanes"]] == [
            f"arrow.{index}" for index, _ in expected
        ]
        for key in ("direction", "metric", "title", "category"):
            assert [lane[key] for lane in group["lanes"]] == [
                getattr(arrow, key) for _, arrow in expected
            ]
    specs = slot_specs(layout, units)
    expected_ids = [
        f"{block.id}.{index}"
        for block in layout.blocks()
        for index, _ in enumerate(block.items)
    ] + [f"arrow.{index}" for index, _ in enumerate(layout.arrows)]
    assert [spec.slot_id for spec in specs] == expected_ids
    assert len(expected_ids) == len(set(expected_ids))
    bars = {
        item["slotId"]: item["bar"] for block in blocks for item in block["content"]
    }
    for spec in specs:
        assert spec.unit == (display_unit(units.get(spec.metric)) or PLAIN)
        assert spec.bar == (
            spec.unit.kind == "percent"
            and spec.category in {"hit", "util", "stall", "neutral"}
            and not spec.cu_block
            and not spec.on_arrow
        )
        if not spec.on_arrow:
            assert bars[spec.slot_id] == spec.bar


@pytest.mark.parametrize(("path", "arch"), LAYOUT_ARCHS, ids=LAYOUT_ARCH_IDS)
def test_diagram_metrics_match_panel_300_for_every_loader_arch(path, arch):
    assert diagram_metrics(load_layout(path)) == set(panel_metric_names(arch))


def test_units_override_metric_names_and_unknown_units_fall_back_to_plain():
    layout = Layouts.for_arch("gfx908")
    metric = layout.columns[1][0].items[0].metric
    spec = next(s for s in slot_specs(layout, {metric: "Cycles"}) if s.metric == metric)
    assert spec.unit == display_unit("Cycles")
    assert all(spec.unit == PLAIN and not spec.bar for spec in slot_specs(layout, {}))


def test_gfx1250_children_and_notes_are_preserved():
    payload = diagram_payload(Layouts.for_arch("gfx1250"))
    blocks = {block["id"]: block for block in payload["gridBlocks"]}
    assert [c["id"] for c in blocks["tcp"]["children"]] == ["lds", "gl0"]
    assert [c["id"] for c in blocks["sqc"]["children"]] == ["icache", "dcache"]
    assert any(block["note"] for block in flatten(payload["gridBlocks"]))


def test_scope_and_attachments_are_independent():
    layout = Layouts.for_arch("gfx950")
    changed = replace(layout, scope_split="l2", scope_labels=("Left", "Right"))
    payload = diagram_payload(changed)
    assert payload["scope"]["gpuLabel"] == "Left"
    assert payload["scope"]["splitColumn"] == 2
    assert [
        (b["id"], b["position"], b["host"], b["column"]) for b in payload["ioBlocks"]
    ] == [
        ("xgmi", "above", "data_fabric", 3),
        ("pcie", "below", "data_fabric", 3),
    ]


def test_gfx9_lane_headers_and_interleaved_pairs_keep_declared_order():
    layout = Layouts.for_arch("gfx908")
    group = diagram_payload(layout)["arrows"][0]
    assert [lane["groupHeader"] for lane in group["lanes"]] == [
        "Non-buffer Request",
        None,
        None,
        "Buffer Request",
        None,
        None,
    ]
    first = replace(layout.arrows[0], group="Alpha")
    other = replace(first, target="l2", metric="Other")
    second = replace(first, metric="Second", group="Beta")
    groups = diagram_payload(replace(layout, arrows=(first, other, second)))["arrows"]
    assert [lane["slotId"] for lane in groups[0]["lanes"]] == ["arrow.0", "arrow.2"]
    assert [lane["groupHeader"] for lane in groups[0]["lanes"]] == ["Alpha", "Beta"]

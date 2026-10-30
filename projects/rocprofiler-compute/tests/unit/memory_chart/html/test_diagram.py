# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Resolved layouts are adapted to diagram payloads and metric slots."""

from dataclasses import replace
from pathlib import Path

import pytest

from memory_chart.html.diagram import diagram_metrics, diagram_payload, slot_specs
from memory_chart.loader import Layout, LayoutArrow, LayoutBlock, LayoutItem
from memory_chart.units import PERCENT, PLAIN


def make_block(block_id="source", **changes):
    """Create a synthetic block with one metric and explicit overrides."""
    return replace(
        LayoutBlock(
            id=block_id,
            title=block_id.title(),
            column=0,
            position="grid",
            style="panel",
            parent=None,
            host=None,
            stall_level=None,
            items=(LayoutItem(block_id, block_id.title(), "hit", False),),
            children=(),
            above=(),
            below=(),
        ),
        **changes,
    )


def make_arrow(source="source", target="target", **changes):
    """Create a synthetic arrow with explicit overrides."""
    return replace(
        LayoutArrow(
            source=source,
            target=target,
            direction="forward",
            metric="Transfer",
            title="Transferred data",
            category="read",
            style="inline",
            group=None,
        ),
        **changes,
    )


def make_layout(columns=None, arrows=(), **changes):
    """Create an in-memory layout without loading architecture data."""
    if columns is None:
        columns = ((make_block(),), (make_block("target", column=1),))

    split = next(block for column in columns for block in column)

    return replace(
        Layout(
            path=Path("synthetic.json"),
            archs=(),
            description="Synthetic diagram layout",
            scope_labels=("Left", "Right"),
            scope_split=split.id,
            columns=columns,
            arrows=arrows,
        ),
        **changes,
    )


@pytest.fixture
def nested_layout():
    """Include siblings, grandchildren, and both attachment stacks."""
    leaf = make_block("leaf", parent="second")
    first = make_block("first", parent="source")
    second = make_block("second", parent="source", children=(leaf,))

    attached_leaf = make_block("attached_leaf", parent="link")
    link = make_block("link", parent="upper", children=(attached_leaf,))

    upper = make_block("upper", position="above", host="source", children=(link,))
    upper_second = make_block("upper_second", position="above", host="source")

    lower = make_block("lower", position="below", host="source")
    lower_second = make_block("lower_second", position="below", host="source")

    source = make_block(
        children=(first, second),
        above=(upper, upper_second),
        below=(lower, lower_second),
    )
    target = make_block("target", column=1)

    return make_layout(columns=((source,), (target,)), arrows=(make_arrow(),))


@pytest.fixture
def interleaved_layout():
    """Interleave two endpoint pairs in a nonalphabetical declaration order."""
    return make_layout(
        arrows=(
            make_arrow("target", "source", metric="First"),
            make_arrow(metric="Second"),
            make_arrow("target", "source", metric="Third"),
        )
    )


def test_payload_has_expected_top_level_fields():
    """Expose the complete set of top-level diagram payload fields."""
    assert set(diagram_payload(make_layout())) == {
        "arch",
        "gridBlocks",
        "ioBlocks",
        "arrows",
        "scope",
    }


def test_payload_identifier_comes_from_path_stem():
    """Derive the payload architecture identifier from the layout path stem."""
    layout = make_layout(path=Path("somewhere/example.layout.json"))

    assert diagram_payload(layout)["arch"] == "example.layout"


def test_payload_copies_block_metadata():
    """Copy each supported block metadata field into the payload."""
    block = make_block(
        "node", title="Display title", column=1, style="list", stall_level="cache"
    )
    layout = make_layout(columns=((), (block,)))

    shown_block = diagram_payload(layout)["gridBlocks"][0]
    expected = {
        "id": "node",
        "title": "Display title",
        "column": 1,
        "position": "grid",
        "style": "list",
        "stallLevel": "cache",
    }

    assert {field: shown_block[field] for field in expected} == expected


def test_payload_copies_item_metadata():
    """Copy each supported item metadata field into the payload."""
    item = LayoutItem("Activity", "Activity label", "util", True)
    layout = make_layout(columns=((make_block(items=(item,)),),))

    shown_item = diagram_payload(layout)["gridBlocks"][0]["content"][0]
    expected = {
        "metric": "Activity",
        "title": "Activity label",
        "category": "util",
    }

    assert {field: shown_item[field] for field in expected} == expected


@pytest.mark.parametrize("bar", [False, True])
def test_payload_preserves_item_bar_flag(bar):
    """Preserve the configured bar flag for each item."""
    item = LayoutItem("Activity", "Activity", "util", bar)
    layout = make_layout(columns=((make_block(items=(item,)),),))

    assert diagram_payload(layout)["gridBlocks"][0]["content"][0]["bar"] is bar


def test_payload_item_slot_ids_use_block_id_and_item_index():
    """Build item slot identifiers from their block ID and item index."""
    items = (
        LayoutItem("Shared", "First", "hit", False),
        LayoutItem("Shared", "Second", "hit", False),
    )
    layout = make_layout(columns=((make_block("node", items=items),),))
    content = diagram_payload(layout)["gridBlocks"][0]["content"]

    assert [item["slotId"] for item in content] == ["node.0", "node.1"]


def test_payload_grid_order_restarts_for_each_column():
    """Number blocks from zero independently within each grid column."""
    layout = make_layout(
        columns=(
            (make_block("z"), make_block("a")),
            (make_block("y", column=1), make_block("b", column=1)),
        )
    )
    blocks = diagram_payload(layout)["gridBlocks"]

    assert [(block["id"], block["order"]) for block in blocks] == [
        ("z", 0),
        ("a", 1),
        ("y", 0),
        ("b", 1),
    ]


@pytest.mark.parametrize(
    ("collection", "expected"),
    [("gridBlocks", "leaf"), ("ioBlocks", "attached_leaf")],
)
def test_payload_serializes_nested_children_recursively(
    nested_layout, collection, expected
):
    """Serialize nested children recursively for grid and attached blocks."""
    root = diagram_payload(nested_layout)[collection][0]
    child = root["children"][-1]

    assert [block["id"] for block in child["children"]] == [expected]


def test_payload_preserves_child_order(nested_layout):
    """Preserve each parent's declared child order in the payload."""
    children = diagram_payload(nested_layout)["gridBlocks"][0]["children"]

    assert [(child["id"], child["order"]) for child in children] == [
        ("first", 0),
        ("second", 1),
    ]


def test_payload_preserves_child_parent(nested_layout):
    """Record the parent block ID on each serialized child."""
    child = diagram_payload(nested_layout)["gridBlocks"][0]["children"][0]

    assert child["parent"] == "source"


def test_payload_collects_attachments_above_then_below(nested_layout):
    """List all above attachments before the below attachments."""
    attachments = diagram_payload(nested_layout)["ioBlocks"]

    assert [block["id"] for block in attachments] == [
        "upper",
        "upper_second",
        "lower",
        "lower_second",
    ]


def test_payload_attachment_order_restarts_for_each_stack(nested_layout):
    """Number above and below attachment stacks independently from zero."""
    attachments = diagram_payload(nested_layout)["ioBlocks"]

    assert [block["order"] for block in attachments] == [0, 1, 0, 1]


def test_payload_preserves_attachment_metadata(nested_layout):
    """Preserve position and host metadata for attached blocks."""
    attachments = diagram_payload(nested_layout)["ioBlocks"]

    assert [(block["position"], block["host"]) for block in attachments] == [
        ("above", "source"),
        ("above", "source"),
        ("below", "source"),
        ("below", "source"),
    ]


def test_payload_resolves_scope_metadata():
    """Resolve scope labels and split block metadata into the payload."""
    layout = make_layout(scope_split="target", scope_labels=("Compute", "Storage"))

    assert diagram_payload(layout)["scope"] == {
        "splitColumn": 1,
        "splitBlock": "target",
        "gpuLabel": "Compute",
        "memoryLabel": "Storage",
    }


@pytest.mark.parametrize("collection", ["gridBlocks", "ioBlocks", "arrows"])
def test_scope_changes_leave_topology_unchanged(nested_layout, collection):
    """Changing scope metadata leaves block and arrow topology unchanged."""
    changed = replace(nested_layout, scope_split="target", scope_labels=("A", "B"))
    original = diagram_payload(nested_layout)

    assert diagram_payload(changed)[collection] == original[collection]


def test_payload_groups_arrows_by_endpoint_pair(interleaved_layout):
    """Group arrows that share an endpoint pair while keeping directions distinct."""
    groups = diagram_payload(interleaved_layout)["arrows"]

    assert {
        (group["from"], group["to"]): {lane["metric"] for lane in group["lanes"]}
        for group in groups
    } == {
        ("target", "source"): {"First", "Third"},
        ("source", "target"): {"Second"},
    }


def test_payload_arrow_groups_follow_first_seen_order(interleaved_layout):
    """Order arrow groups by the first occurrence of each endpoint pair."""
    groups = diagram_payload(interleaved_layout)["arrows"]

    assert [(group["from"], group["to"]) for group in groups] == [
        ("target", "source"),
        ("source", "target"),
    ]


def test_payload_lanes_keep_declared_order_for_interleaved_pairs(interleaved_layout):
    """Keep each endpoint pair's lanes in their original declaration order."""
    lanes = diagram_payload(interleaved_layout)["arrows"][0]["lanes"]

    assert [lane["metric"] for lane in lanes] == ["First", "Third"]


def test_payload_arrow_slot_ids_keep_original_indices(interleaved_layout):
    """Preserve original arrow indices when assigning lane slot identifiers."""
    groups = diagram_payload(interleaved_layout)["arrows"]

    assert [[lane["slotId"] for lane in group["lanes"]] for group in groups] == [
        ["arrow.0", "arrow.2"],
        ["arrow.1"],
    ]


def test_payload_copies_lane_metadata():
    """Copy supported arrow lane metadata into the diagram payload."""
    arrow = make_arrow(
        direction="backward",
        metric="Traffic",
        title="Traffic label",
        category="write",
        style="stacked",
        group="Requests",
    )
    lane = diagram_payload(make_layout(arrows=(arrow,)))["arrows"][0]["lanes"][0]

    expected = {
        "from": "source",
        "to": "target",
        "direction": "backward",
        "metric": "Traffic",
        "title": "Traffic label",
        "category": "write",
        "style": "stacked",
        "group": "Requests",
    }

    assert {field: lane[field] for field in expected} == expected


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (("Alpha", "Beta", "Alpha"), ["Alpha", "Beta", "Alpha"]),
        ((None, None), [None, None]),
        (("Alpha", None, "Alpha"), ["Alpha", None, "Alpha"]),
    ],
)
def test_payload_group_header_marks_group_transitions(names, expected):
    """Add lane group headers only when the declared group changes."""
    layout = make_layout(arrows=tuple(make_arrow(group=name) for name in names))
    lanes = diagram_payload(layout)["arrows"][0]["lanes"]

    assert [lane["groupHeader"] for lane in lanes] == expected


def test_payload_group_headers_are_independent_between_endpoint_pairs():
    """Start group header tracking independently for each endpoint pair."""
    layout = make_layout(
        arrows=(
            make_arrow(group="Alpha"),
            make_arrow("target", "source", group="Alpha"),
            make_arrow(group="Alpha"),
        )
    )
    groups = diagram_payload(layout)["arrows"]

    assert [[lane["groupHeader"] for lane in group["lanes"]] for group in groups] == [
        ["Alpha", None],
        ["Alpha"],
    ]


def test_slot_specs_identifiers_distinguish_repeated_metrics():
    """Assign distinct slot identifiers to repeated block and arrow metrics."""
    item = LayoutItem("Shared", "Shared", "hit", False)
    layout = make_layout(
        columns=((make_block("node", items=(item, item)),),),
        arrows=(make_arrow(metric="Shared", target="node", source="node"),),
    )

    assert [spec.slot_id for spec in slot_specs(layout, {})] == [
        "node.0",
        "node.1",
        "arrow.0",
    ]


def test_slot_specs_follow_block_traversal_before_arrows(nested_layout):
    """Traverse nested grid and attached blocks before adding arrow slots."""
    assert [spec.metric for spec in slot_specs(nested_layout, {})] == [
        "source",
        "first",
        "second",
        "leaf",
        "upper",
        "link",
        "attached_leaf",
        "upper_second",
        "lower",
        "lower_second",
        "target",
        "Transfer",
    ]


def test_slot_specs_copy_metric_metadata():
    """Copy metric and category metadata into block and arrow slot specs."""
    block = make_block(items=(LayoutItem("Block metric", "Label", "hit", False),))
    layout = make_layout(
        columns=((block,),), arrows=(make_arrow(metric="Arrow metric"),)
    )

    assert [(spec.metric, spec.category) for spec in slot_specs(layout, {})] == [
        ("Block metric", "hit"),
        ("Arrow metric", "read"),
    ]


def test_slot_specs_disable_bars_on_arrows():
    """Disable bar rendering for arrow metric slot specs."""
    assert slot_specs(make_layout(arrows=(make_arrow(),)), {})[-1].bar is False


def test_slot_specs_mark_only_arrow_slots_as_on_arrow():
    """Mark arrow metric slots while leaving block slots unmarked."""
    layout = make_layout(arrows=(make_arrow(),))

    assert [spec.on_arrow for spec in slot_specs(layout, {})] == [False, False, True]


@pytest.mark.parametrize(("column", "expected"), [(0, True), (1, False)])
def test_slot_specs_classify_blocks_by_column(column, expected):
    """Classify block slots according to whether they occupy column zero."""
    block = make_block(column=column)
    layout = make_layout(columns=tuple(() for _ in range(column)) + ((block,),))

    assert slot_specs(layout, {})[0].cu_block is expected


def test_slot_specs_do_not_classify_arrows_as_column_zero_blocks():
    """Keep arrow slots outside the column-zero block classification."""
    assert slot_specs(make_layout(arrows=(make_arrow(),)), {})[-1].cu_block is False


@pytest.mark.parametrize(
    ("index", "metric"), [(0, "source"), (1, "Transfer")], ids=["block", "arrow"]
)
def test_slot_specs_use_supplied_units(index, metric):
    """Resolve recognized unit names for block and arrow metric slots."""
    layout = make_layout(columns=((make_block(),),), arrows=(make_arrow(),))

    assert slot_specs(layout, {metric: "Percent"})[index].unit == PERCENT


@pytest.mark.parametrize("on_arrow", [False, True], ids=["block", "arrow"])
@pytest.mark.parametrize(
    "units",
    [{}, {"source": "Unknown", "Transfer": "Unknown"}],
    ids=["missing", "unknown"],
)
def test_slot_specs_fall_back_to_plain_units(on_arrow, units):
    """Use plain units when a metric has no recognized unit mapping."""
    layout = make_layout(columns=((make_block(),),), arrows=(make_arrow(),))

    assert slot_specs(layout, units)[int(on_arrow)].unit == PLAIN


def test_diagram_metrics_collects_distinct_block_and_arrow_metrics():
    """Collect distinct metrics from repeated items, nested blocks, and arrows."""
    item = LayoutItem("Shared", "Shared", "hit", False)
    child = make_block("child", parent="source")
    attached = make_block("attached", position="above", host="source")
    block = make_block(items=(item, item), children=(child,), above=(attached,))
    layout = make_layout(
        columns=((block,),),
        arrows=(
            make_arrow("source", "child", metric="Shared"),
            make_arrow("source", "attached", metric="Transfer"),
        ),
    )

    assert diagram_metrics(layout) == frozenset({
        "Shared",
        "child",
        "attached",
        "Transfer",
    })

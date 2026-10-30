# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Checks every memory chart layout against the format rules and the panel configs."""

import copy
import json
import random
import re
from collections import Counter

import common
import pytest

from memory_chart import loader
from memory_chart.loader import (
    LayoutError,
    Layouts,
    PanelConfigs,
    has_layout,
    layout_files,
    layout_from_stored,
    load_layout,
)
from memory_chart.units import display_unit, panel_units

MEMORY_CHART_ARCHS = common.memory_chart_archs()

# Every (layout file, arch it serves) pair
LAYOUT_ARCHS = [
    (path, arch) for path in layout_files() for arch in load_layout(path).archs
]
LAYOUT_ARCH_IDS = [f"{path.stem}-{arch}" for path, arch in LAYOUT_ARCHS]

READ_ARROW = {
    "from": "cu",
    "to": "vl1d",
    "direction": "backward",
    "metric": "Flat Read",
    "title": "Read",
    "category": "read",
    "style": "inline",
}
INNER = {
    "id": "inner",
    "title": "Inner",
    "column": 4,
    "order": 0,
    "position": "grid",
    "style": "panel",
}
NESTED = {**INNER, "parent": "mall"}
ABOVE = {**INNER, "column": 3, "position": "above", "host": "data_fabric"}


def panel_metric_names(arch):
    """Memory chart metric names of an arch, in config order, duplicates kept."""
    return [
        name
        for source in PanelConfigs.for_arch(arch)["data source"]
        for table in source.values()
        for name in table.get("metric") or {}
    ]


def gfx950_data():
    return common.memory_chart_layout_data("gfx950")


def write_layout(tmp_path, data, name="layout.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def changed(**fields):
    """gfx950's layout with top-level fields replaced."""
    return {**gfx950_data(), **fields}


def with_block_change(block_id, **fields):
    """gfx950's layout with fields of one block replaced; None removes a field."""
    data = gfx950_data()
    for block in data["blocks"]:
        if block["id"] == block_id:
            block.update(copy.deepcopy(fields))
            for key in [k for k, v in fields.items() if v is None]:
                del block[key]
    return data


def with_blocks(*blocks):
    """gfx950's layout with extra blocks."""
    data = gfx950_data()
    data["blocks"] += [copy.deepcopy(block) for block in blocks]
    return data


def with_metric_change(metric, **fields):
    """gfx950's layout with fields set on the items and arrows of *metric*."""
    return common.with_metric_change(gfx950_data(), metric, **fields)


# ---------------------------------------------------------------------------
# Layouts against the panel configs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", MEMORY_CHART_ARCHS)
def test_every_config_arch_has_one_layout(arch):
    owners = [path.name for path, served in LAYOUT_ARCHS if served == arch]
    assert len(owners) == 1, f"{arch} is served by {owners}"


@pytest.mark.parametrize(("path", "arch"), LAYOUT_ARCHS, ids=LAYOUT_ARCH_IDS)
def test_layout_shows_every_panel_metric_and_no_other(path, arch):
    shown = load_layout(path).metrics()
    panel = set(panel_metric_names(arch))
    assert not shown - panel, (
        f"not in the {arch} memory chart panel: {sorted(shown - panel)}"
    )
    assert not panel - shown, f"missing from {path.name}: {sorted(panel - shown)}"


@pytest.mark.parametrize("arch", MEMORY_CHART_ARCHS)
def test_panel_metric_names_are_unique(arch):
    # Charts and the analysis database find metrics by name
    duplicates = [n for n, c in Counter(panel_metric_names(arch)).items() if c > 1]
    assert not duplicates


@pytest.mark.parametrize(("path", "arch"), LAYOUT_ARCHS, ids=LAYOUT_ARCH_IDS)
def test_every_shown_unit_has_a_display_rule(path, arch):
    units = panel_units(PanelConfigs.for_arch(arch))
    shown = load_layout(path).metrics()
    unknown = {m: units.get(m) for m in shown if display_unit(units.get(m)) is None}
    assert not unknown


@pytest.mark.parametrize(("path", "arch"), LAYOUT_ARCHS, ids=LAYOUT_ARCH_IDS)
def test_arrow_style_fits_the_metric_unit(path, arch):
    # A count reads best as one "title : value" line; a rate as title, value, arrow
    units = panel_units(PanelConfigs.for_arch(arch))
    layout = load_layout(path)
    grid = {b.id for b in layout.blocks() if b.position == "grid"}
    wrong = [
        arrow.metric
        for arrow in layout.arrows
        if arrow.target in grid
        and (arrow.style == "inline")
        != (display_unit(units[arrow.metric]).kind == "count")
    ]
    assert not wrong


@pytest.mark.parametrize(("path", "arch"), LAYOUT_ARCHS, ids=LAYOUT_ARCH_IDS)
def test_bars_are_only_on_percent_metrics(path, arch):
    units = panel_units(PanelConfigs.for_arch(arch))
    wrong = [
        item.metric
        for block in load_layout(path).blocks()
        for item in block.items
        if item.bar and display_unit(units[item.metric]).kind != "percent"
    ]
    assert not wrong


@pytest.mark.parametrize(
    ("gpu_arch", "layout"),
    [("gfx1150", "gfx115x"), ("gfx1153", "gfx115x"), ("gfx941", "gfx94x")],
)
def test_runtime_arch_resolves_to_its_family_layout(gpu_arch, layout):
    assert Layouts.for_arch(gpu_arch).path.stem == layout


@pytest.mark.parametrize("gpu_arch", ["gfx1030", None])
def test_arch_without_layout_has_no_chart(gpu_arch):
    assert Layouts.for_arch(gpu_arch) is None


@pytest.mark.parametrize("gpu_arch", MEMORY_CHART_ARCHS)
def test_has_layout_for_supported_arch(gpu_arch):
    """Confirm supported architectures have a memory chart layout."""
    assert has_layout(gpu_arch) is True


@pytest.mark.parametrize("gpu_arch", ["gfx1030", "unknown", None])
def test_has_layout_is_false_without_a_layout(gpu_arch):
    """Confirm unsupported, unknown, or missing architectures have no layout."""
    assert has_layout(gpu_arch) is False


def test_an_arch_in_two_layout_files_keeps_the_first(tmp_path, monkeypatch):
    warning = common.patch_console(monkeypatch, "memory_chart.loader", "warning")
    paths = [write_layout(tmp_path, gfx950_data(), f"{n}.json") for n in "ab"]
    assert loader._index_by_arch(paths)["gfx950"].path.name == "a.json"
    assert "also listed by a.json" in warning["warning"].call_args.args[1]


def test_a_broken_layout_is_skipped_and_others_still_load(tmp_path, monkeypatch):
    warning = common.patch_console(monkeypatch, "memory_chart.loader", "warning")
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    index = loader._index_by_arch([broken, write_layout(tmp_path, gfx950_data())])
    assert list(index) == ["gfx950"]
    assert "broken.json" in warning["warning"].call_args.args[1]


# ---------------------------------------------------------------------------
# Format rules
# ---------------------------------------------------------------------------


def test_block_list_order_does_not_matter(tmp_path):
    data = gfx950_data()
    random.Random(0).shuffle(data["blocks"])
    assert load_layout(write_layout(tmp_path, data)).columns == (
        Layouts.for_arch("gfx950").columns
    )


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (changed(schema_version=3), "unknown keys"),
        ({k: v for k, v in gfx950_data().items() if k != "scope"}, "missing ['scope']"),
        (changed(scope={"labels": ["a"], "split": "l2"}), "two labels"),
        (changed(scope={"labels": ["a", "b"], "split": "cu"}), "scope split"),
        (with_block_change("vl1d", note="x"), "unknown keys"),
        (with_block_change("vl1d", order=True), "order of 'vl1d'"),
        (with_block_change("vl1d", stall_level="L3"), "stall_level"),
        (with_block_change("mall", column=7), "with no gaps"),
        (with_block_change("lds", order=0), "orders 0, 1"),
        (with_blocks(gfx950_data()["blocks"][1]), "duplicate block id"),
        (with_block_change("vl1d", host="cu"), "has no host"),
        (with_block_change("xgmi", host=None), "needs a host"),
        (with_blocks({**ABOVE, "content": []}), "needs a host"),
        (with_blocks({**NESTED, "parent": "nowhere"}), "unknown block 'nowhere'"),
        (with_blocks({**NESTED, "parent": "xgmi", "column": 3}), "hang off a block"),
        (with_blocks({**ABOVE, "host": "xgmi"}), "hang off a block"),
        (with_blocks({**NESTED, "column": 2}), "the column of 'mall'"),
        (with_blocks({**NESTED, "parent": "inner"}), "nest inside each other"),
        (with_blocks({**NESTED, "parent": "vl1d", "column": 1}), "nested blocks"),
        (with_metric_change("VGPR", bar=True), "compact list"),
        (with_metric_change("L2 Hit", category="x"), "category"),
        (with_metric_change("L2 Hit", bar=1), "bar of 'L2 Hit'"),
        (with_metric_change("Flat Read", category="hit"), "category"),
        (with_metric_change("Flat Read", direction="up"), "direction"),
        (changed(arrows=[{**READ_ARROW, "to": "nowhere"}]), "unknown block"),
        (changed(arrows=[{**READ_ARROW, "to": "l2"}]), "next column"),
        (
            changed(arrows=[{**READ_ARROW, "from": "xgmi", "to": "mall"}]),
            "must start at a block in a column",
        ),
        (with_blocks(NESTED), "nested in 'mall'"),
        (with_metric_change("xGMI Read BW", style="stacked"), "must be inline"),
    ],
    ids=[
        "unknown-top-level-key",
        "missing-key",
        "scope-labels",
        "scope-split-in-first-column",
        "unknown-block-key",
        "order-not-a-number",
        "stall-level",
        "column-gap",
        "order-repeated",
        "duplicate-block-id",
        "grid-block-with-host",
        "attached-block-without-host",
        "attached-block-with-content",
        "unknown-parent",
        "parent-is-attached",
        "host-is-attached",
        "nested-block-in-another-column",
        "parent-cycle",
        "parent-has-metrics",
        "bar-in-a-list",
        "metric-category",
        "bar-not-a-bool",
        "arrow-category",
        "arrow-direction",
        "arrow-unknown-block",
        "arrow-skips-a-column",
        "arrow-from-an-attached-block",
        "arrow-to-a-parent-block",
        "stacked-arrow-to-an-attached-block",
    ],
)
def test_malformed_layout_is_rejected(tmp_path, data, message):
    with pytest.raises(LayoutError, match=re.escape(message)):
        load_layout(write_layout(tmp_path, data))


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (None, "cannot read the layout"),
        ("{", "cannot read the layout"),
        ("[]", "must be a JSON object"),
    ],
    ids=["missing-file", "invalid-json", "not-an-object"],
)
def test_unreadable_layout_is_rejected(tmp_path, text, message):
    path = tmp_path / "layout.json"
    if text is not None:
        path.write_text(text, encoding="utf-8")
    with pytest.raises(LayoutError, match=message):
        load_layout(path)


def test_stored_layout_takes_the_name_of_the_shipped_file_it_matches():
    """The diagram names the chart by the file stem, so a stored copy keeps it."""
    layout = layout_from_stored(gfx950_data())
    assert layout.path.stem == "gfx950"
    assert layout.archs == tuple(gfx950_data()["archs"])


def test_stored_layout_matching_no_shipped_file_is_named_by_its_arch():
    """A copy from another layout version still loads, named after its first arch."""
    data = changed(description="A layout no shipped file has")
    assert layout_from_stored(data).path.name == "gfx950.json"

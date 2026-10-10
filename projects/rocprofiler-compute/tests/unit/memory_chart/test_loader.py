# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Checks every memory chart layout against the format rules and the panel configs."""

import copy
import json
from collections import Counter

import pytest

from memory_chart import loader
from memory_chart.loader import Layouts, load_layout
from memory_chart.units import display_unit, panel_units
from tests.unit.memory_chart.layout_cases import (
    CONFIG_ARCHS,
    LAYOUT_ARCH_IDS,
    LAYOUT_ARCHS,
    panel_config,
    panel_metric_names,
)

READ_ARROW = {
    "from": "cu",
    "to": "vl1d",
    "metric": "Flat Read",
    "title": "Read",
    "category": "read",
}
INNER = {"id": "inner", "title": "Inner"}


def gfx950_data():
    return json.loads(Layouts.for_arch("gfx950").path.read_text(encoding="utf-8"))


def write_layout(tmp_path, data, name="layout.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def changed(**fields):
    """gfx950's layout with top-level fields replaced."""
    return {**gfx950_data(), **fields}


def with_block_change(block_id, **fields):
    """gfx950's layout with fields added to one column block."""
    data = gfx950_data()
    for column in data["columns"]:
        for block in column:
            if block["id"] == block_id:
                block.update(copy.deepcopy(fields))
    return data


def with_duplicate_block():
    data = gfx950_data()
    data["columns"][1].append(dict(data["columns"][1][0]))
    return data


# ---------------------------------------------------------------------------
# Layouts against the panel configs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", CONFIG_ARCHS)
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


@pytest.mark.parametrize("arch", CONFIG_ARCHS)
def test_panel_metric_names_are_unique(arch):
    # Charts and the analysis database find metrics by name
    duplicates = [n for n, c in Counter(panel_metric_names(arch)).items() if c > 1]
    assert not duplicates


@pytest.mark.parametrize(("path", "arch"), LAYOUT_ARCHS, ids=LAYOUT_ARCH_IDS)
def test_every_shown_unit_has_a_display_rule(path, arch):
    units = panel_units(panel_config(arch))
    shown = load_layout(path).metrics()
    unknown = {m: units.get(m) for m in shown if display_unit(units.get(m)) is None}
    assert not unknown


@pytest.mark.parametrize(
    ("gpu_arch", "layout"),
    [("gfx1150", "gfx115x"), ("gfx1153", "gfx115x"), ("gfx941", "gfx94x")],
)
def test_runtime_arch_resolves_to_its_family_layout(gpu_arch, layout):
    assert Layouts.for_arch(gpu_arch).path.stem == layout


@pytest.mark.parametrize("gpu_arch", ["gfx1030", None])
def test_arch_without_layout_has_no_chart(gpu_arch):
    assert Layouts.for_arch(gpu_arch) is None


def test_an_arch_in_two_layout_files_is_an_error(tmp_path, caplog):
    paths = [write_layout(tmp_path, gfx950_data(), f"{n}.json") for n in "ab"]
    with pytest.raises(SystemExit):
        loader._index_by_arch(paths)
    assert "gfx950 is also listed by a.json" in caplog.text


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_attached_blocks_take_their_host_column_and_side():
    blocks = {b.id: b for b in Layouts.for_arch("gfx950").blocks()}
    xgmi, pcie = blocks["xgmi"], blocks["pcie"]
    assert (xgmi.position, xgmi.host, xgmi.column) == ("above", "data_fabric", 3)
    assert (pcie.position, pcie.host) == ("below", "data_fabric")


def test_nested_blocks_take_their_parent_and_column():
    blocks = {b.id: b for b in Layouts.for_arch("gfx1250").blocks()}
    assert [c.id for c in blocks["tcp"].children] == ["lds", "gl0"]
    assert (blocks["gl0"].parent, blocks["gl0"].column) == ("tcp", 1)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (changed(schema_version=3), "unknown keys"),
        ({k: v for k, v in gfx950_data().items() if k != "scope"}, "missing ['scope']"),
        (changed(scope={"labels": ["a"], "split": "l2"}), "two labels"),
        (changed(columns=[*gfx950_data()["columns"], []]), "every column"),
        (changed(arrows=[{**READ_ARROW, "category": "hit"}]), "category"),
        (changed(arrows=[{**READ_ARROW, "to": "nowhere"}]), "unknown block"),
        (changed(arrows=[{**READ_ARROW, "to": "l2"}]), "next column"),
        (changed(scope={"labels": ["a", "b"], "split": "cu"}), "scope split"),
        (with_block_change("vl1d", childs=[]), "unknown keys"),
        (with_block_change("vl1d", stall_level="L3"), "stall_level"),
        (
            with_block_change(
                "l2", metrics=[{"metric": "L2 Hit", "title": "Hit", "category": "x"}]
            ),
            "category",
        ),
        (
            with_block_change("data_fabric", above=[{**INNER, "metrics": []}]),
            "unknown keys",
        ),
        (with_block_change("vl1d", children=[INNER]), "nested blocks"),
        (
            with_block_change("mall", children=[{**INNER, "above": [INNER]}]),
            "unknown keys",
        ),
        (with_block_change("mall", children=[INNER]), "nested in 'mall'"),
        (with_duplicate_block(), "duplicate block id"),
        (
            changed(arrows=[{**READ_ARROW, "from": "xgmi", "to": "mall"}]),
            "must start at a block in a column",
        ),
    ],
    ids=[
        "unknown-top-level-key",
        "missing-key",
        "scope-labels",
        "empty-column",
        "arrow-category",
        "arrow-unknown-block",
        "arrow-skips-a-column",
        "scope-split-in-first-column",
        "unknown-block-key",
        "stall-level",
        "metric-category",
        "content-on-attached-block",
        "metrics-on-a-parent-block",
        "attached-block-on-a-nested-block",
        "arrow-to-a-parent-block",
        "duplicate-block-id",
        "arrow-from-an-attached-block",
    ],
)
def test_malformed_layout_is_rejected(tmp_path, caplog, data, message):
    with pytest.raises(SystemExit):
        load_layout(write_layout(tmp_path, data))
    assert message in caplog.text

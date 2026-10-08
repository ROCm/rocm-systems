# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""The memory chart layout as stored in the analysis database."""

import json
import re

import pytest

from memory_chart.extdata import FORMAT_VERSION, layout_extdata
from memory_chart.loader import Layouts, load_layout
from tests.unit.memory_chart.layout_cases import (
    LAYOUT_ARCH_IDS,
    LAYOUT_ARCHS,
    panel_config,
    stored_blocks,
    unresolved_metrics,
)
from utils.utils_common import panel_metric_ids

# Block ids and metric references in the format Optiq's memory chart reads
OPTIQ_BLOCK_ID = re.compile(r"^[a-z][a-z0-9_]*$")
OPTIQ_METRIC_REF = re.compile(r"^\d+\.\d+\.\d+$")


def stored_layout(arch):
    return layout_extdata(Layouts.for_arch(arch), panel_metric_ids(panel_config(arch)))


def with_two_blocks_above_fabric(tmp_path):
    """gfx950's layout with a second block above the Data Fabric."""
    data = json.loads(Layouts.for_arch("gfx950").path.read_text(encoding="utf-8"))
    fabric = next(
        b for column in data["columns"] for b in column if b["id"] == "data_fabric"
    )
    fabric["above"].insert(0, {"id": "extra", "title": "Extra"})
    path = tmp_path / "layout.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return load_layout(path)


@pytest.mark.parametrize(("path", "arch"), LAYOUT_ARCHS, ids=LAYOUT_ARCH_IDS)
def test_ids_and_references_use_optiqs_formats(path, arch):
    data = layout_extdata(load_layout(path), panel_metric_ids(panel_config(arch)))
    blocks = list(stored_blocks(data["blocks"]))
    refs = [item["metric"] for b in blocks for item in b.get("content", [])]
    refs += [arrow["metric"] for arrow in data["arrows"]]
    assert data["version"] == FORMAT_VERSION
    assert all(OPTIQ_BLOCK_ID.match(block["id"]) for block in blocks)
    assert all(OPTIQ_METRIC_REF.match(ref) for ref in refs)


@pytest.mark.parametrize(("path", "arch"), LAYOUT_ARCHS, ids=LAYOUT_ARCH_IDS)
def test_each_reference_is_the_id_of_its_own_metric(path, arch):
    layout = load_layout(path)
    ids = panel_metric_ids(panel_config(arch))
    data = layout_extdata(layout, ids)
    stored = {b["id"]: b for b in stored_blocks(data["blocks"])}
    for block in layout.blocks():
        refs = [item["metric"] for item in stored[block.id].get("content", [])]
        assert refs == [ids[item.metric] for item in block.items], block.id
    assert [a["metric"] for a in data["arrows"]] == [
        ids[a.metric] for a in layout.arrows
    ]


def test_blocks_keep_their_column_order_and_nesting():
    blocks = {b["id"]: b for b in stored_blocks(stored_layout("gfx1250")["blocks"])}
    sqc = blocks["sqc"]
    assert (sqc["column"], sqc["row"], sqc["order"]) == (1, 0, 1)
    assert [c["id"] for c in blocks["tcp"]["children"]] == ["lds", "gl0"]
    assert (blocks["gl0"]["order"], blocks["tcp"]["note"]) == (1, "TXA/TXD/TDM")


def test_above_and_below_blocks_are_rows_of_their_host_column():
    blocks = {b["id"]: b for b in stored_layout("gfx950")["blocks"]}
    fabric = blocks["data_fabric"]
    assert (blocks["xgmi"]["row"], blocks["xgmi"]["host"]) == (-1, "data_fabric")
    assert (blocks["pcie"]["row"], blocks["pcie"]["host"]) == (1, "data_fabric")
    assert blocks["xgmi"]["column"] == blocks["pcie"]["column"] == fabric["column"]


def test_the_block_above_next_to_the_host_is_row_minus_one(tmp_path):
    layout = with_two_blocks_above_fabric(tmp_path)
    data = layout_extdata(layout, panel_metric_ids(panel_config("gfx950")))
    rows = {b["id"]: b["row"] for b in data["blocks"]}
    assert (rows["extra"], rows["xgmi"]) == (-2, -1)


def test_a_metric_without_an_id_is_stored_as_null():
    layout = Layouts.for_arch("gfx950")
    ids = panel_metric_ids(panel_config("gfx950"))
    del ids["Flat Read"]
    assert unresolved_metrics(layout, layout_extdata(layout, ids)) == {"Flat Read"}


def test_arrows_carry_direction_category_and_group():
    arrow = next(a for a in stored_layout("gfx950")["arrows"] if a["title"] == "Read")
    assert arrow == {
        "from": "cu",
        "to": "vl1d",
        "direction": "backward",
        "metric": "3.1.6",
        "title": "Read",
        "category": "read",
        "group": "Non-buffer Request",
    }

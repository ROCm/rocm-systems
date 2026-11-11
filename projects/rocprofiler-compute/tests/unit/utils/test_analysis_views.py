# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils/analysis_views.py.

tools/validate_views.py checks that the shipped views keep the panels' structure; these
tests cover what that check does not compare: the values a view row gives, the errors
for broken views, and the profile mode imports.
"""

import copy
import importlib
import sys

from conftest import ProfileModeImportGuard

import utils
from rocprof_compute_soc.metric_library import loader
from utils.analysis_views import (
    iter_view_rows,
    load_views,
    view_metric_errors,
    view_schema_errors,
    views_to_panel_configs,
)

ARCH = "gfx942"
# The instructions gfx942's compute.ipc_issued counts, per kernel.
ISSUED = (
    "SQ_INSTS_VALU + SQ_INSTS_VMEM + SQ_INSTS_SALU + SQ_INSTS_SMEM + SQ_INSTS_BRANCH"
    " + SQ_INSTS_SENDMSG + SQ_INSTS_VSKIPPED + SQ_INSTS_LDS"
)
# A view with one mistake of each kind the shape check reports.
BROKEN_VIEW = {
    "view": "Compute Units - Compute Pipeline",
    "legacy_panel_id": "1100",
    "archs": ["gfx942"],
    "tables": [
        {
            "id": 1101,
            "title": "Compute Speed-of-Light",
            "render": "chart",
            "cli_style": "pie",
        },
        {
            "id": 1102,
            "title": "Pipeline Statistics",
            "header": {"metric": "Metric", "avg": "Avg", "unit": "Unit"},
            "metrics": [
                {"metric": "compute.ipc", "metrics": {"avg": "compute.ipc"}},
                {"metrics": {"avg": "compute.ipc"}},
                {"metric": "compute.salu_util", "values": {"xfer": "HBM"}},
                {"metric": "compute.valu_util", "legacy_id": {"gfx942": "11.2.3"}},
                {"metric": "compute.vmem_util", "peak": "theoretical"},
                {"metric": "compute.mfma_util", "unit": "Percent"},
            ],
        },
    ],
}


def metric_rows(arch):
    """Return the arch's views, metrics, and panel config rows by numeric id."""
    views = load_views(arch)
    metrics = loader.MetricLibrary.load(
        loader.default_metric_library_dir(), arch
    ).metrics
    rows = {}
    for panel in views_to_panel_configs(views, metrics).values():
        for source in panel["data source"]:
            table = source.get("metric_table")
            if not table:
                continue
            number = f"{table['id'] // 100}.{table['id'] % 100}"
            names = [name for name in table["metric"] if name != "placeholder_range"]
            for index, name in enumerate(names):
                rows[f"{number}.{index}"] = (name, table["metric"][name])
    return views, metrics, rows


def test_rows_show_their_metric_as_the_view_asks():
    """Each display field of a view row changes what its panel config row shows."""
    _, metrics, rows = metric_rows(ARCH)
    ipc = metrics["compute.ipc_issued"]
    assert rows["11.2.1"] == (
        ipc["name"],
        {
            "avg": ipc["formula"],
            "min": f"MIN(({ISSUED}) / SQ_ACTIVE_INST_ANY)",
            "max": f"MAX(({ISSUED}) / SQ_ACTIVE_INST_ANY)",
            "unit": ipc["unit"],
        },
    )
    occupancy = metrics["wave.occupancy_pct"]
    assert rows["3.1.0"][1]["value"] == f"ROUND({occupancy['formula']}, 1)"
    valu_flops = metrics["compute.valu_flops_f16"]
    assert rows["4.1.0"][1]["peak"] == valu_flops["peak_empirical"]
    assert rows["2.1.7"][1]["pct_of_peak"] is False
    assert rows["16.4.0"][0] == "NC - Read"
    assert rows["16.4.0"][1]["xfer"] == "Read"
    assert rows["18.4.0"][1]["read req"] == metrics["l2.chan_rd"]["formula"]


def test_panel_descriptions_come_from_the_library():
    views, metrics, _ = metric_rows(ARCH)
    panel = views_to_panel_configs(views, metrics)[1100]
    ipc = metrics["compute.ipc"]
    assert panel["metrics_description"][ipc["name"]] == ipc["description"]


def test_view_schema_errors():
    errors = view_schema_errors({"gfx942/1100.yaml": BROKEN_VIEW})
    assert errors == [
        "[gfx942/1100.yaml] 'archs' is not a known field",
        "[gfx942/1100.yaml] 'legacy_panel_id' must be an integer",
        "[gfx942/1100.yaml] table 1101: render 'chart' is not one of "
        "('metric_table', 'raw_csv_table', 'pc_sampling_table')",
        "[gfx942/1100.yaml] table 1101: cli_style 'pie' is not one of "
        "('simple_box', 'simple_bar', 'simple_multiple_bar', 'mem_chart', 'Roofline')",
        "[gfx942/1100.yaml] table 1102: row 0: needs one of 'metric' and 'metrics'",
        "[gfx942/1100.yaml] table 1102: row 1: needs a 'label' when it has 'metrics'",
        "[gfx942/1100.yaml] table 1102: row 2: 'values' names 'xfer', which is not a "
        "column",
        "[gfx942/1100.yaml] table 1102: row 3: 'legacy_id' must be a string",
        "[gfx942/1100.yaml] table 1102: row 4: 'peak' may only be 'empirical'",
        "[gfx942/1100.yaml] table 1102: row 5: 'unit' is not a known field",
    ]


def test_view_metric_errors():
    """The shipped gfx942 views, with a metric and a measured peak the library lacks,
    and a panel shown twice."""
    views = load_views(ARCH)
    metrics = loader.MetricLibrary.load(
        loader.default_metric_library_dir(), ARCH
    ).metrics
    assert view_metric_errors(views, metrics) == []
    broken = copy.deepcopy(views)
    rows = dict(iter_view_rows(broken))
    rows["11.2.0"]["metric"] = "compute.made_up"
    rows["2.1.11"]["peak"] = "empirical"
    broken.append(copy.deepcopy(broken[0]))
    assert view_metric_errors(broken, metrics) == [
        "panel 0 has more than one view",
        "2.1.11 compute.branch_util has no 'peak_empirical'",
        "11.2.0 compute.made_up: not a metric of the arch's library",
    ]


def test_imports_only_profile_mode_dependencies(monkeypatch):
    """The views and the equation normalizer use only stdlib, project and vendored
    code. The re-import rebinds the utils package attributes, so they are restored too:
    later tests patch these modules by name."""
    for module in ("utils.analysis_views", "utils.equation_format"):
        name = module.split(".")[1]
        monkeypatch.setattr(utils, name, getattr(utils, name))
        monkeypatch.delitem(sys.modules, module, raising=False)
    with ProfileModeImportGuard():
        importlib.import_module("utils.analysis_views")

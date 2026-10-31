# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils/analysis_views.py.

tools/validate_views.py checks that the shipped views give the panel configs exactly;
these tests cover what that check does not reach: the errors for broken views, and
the profile mode imports.
"""

import copy
import importlib
import sys

from conftest import ProfileModeImportGuard

from utils.analysis_views import (
    iter_view_rows,
    load_views,
    view_metric_errors,
    view_schema_errors,
)
from utils.metric_library import default_metric_library_dir, load_metric_library

# A view with one mistake of each kind the shape check reports.
BROKEN_VIEW = {
    "view": "Compute Units - Compute Pipeline",
    "legacy_panel_id": "1100",
    "archs": ["gfx942", "gfx9999"],
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
                {"unit": "Percent", "display": "made_up_form"},
                {"metric": "compute.salu_util", "values": {"xfer": "HBM"}},
                {"metric": "compute.valu_util", "legacy_id": {"gfx9999": "11.2.3"}},
                {"metric": "compute.mfma_util", "colour": "red"},
            ],
        },
    ],
}


def test_view_schema_errors():
    errors = view_schema_errors({"cdna/1100.yaml": BROKEN_VIEW})
    assert errors == [
        "[cdna/1100.yaml] 'archs' names 'gfx9999', which is not a metric library arch",
        "[cdna/1100.yaml] 'legacy_panel_id' must be an integer",
        "[cdna/1100.yaml] table 1101: render 'chart' is not one of "
        "('metric_table', 'raw_csv_table', 'pc_sampling_table')",
        "[cdna/1100.yaml] table 1101: cli_style 'pie' is not one of "
        "('simple_box', 'simple_bar', 'simple_multiple_bar', 'mem_chart', 'Roofline')",
        "[cdna/1100.yaml] table 1102: row 0: has both 'metric' and 'metrics'",
        "[cdna/1100.yaml] table 1102: row 1: needs a 'label' when it has no 'metric'",
        "[cdna/1100.yaml] table 1102: row 1: display 'made_up_form' is not in "
        "DISPLAY_FORMS",
        "[cdna/1100.yaml] table 1102: row 2: 'values' names 'xfer', which is not a "
        "column",
        "[cdna/1100.yaml] table 1102: row 3: 'legacy_id' names 'gfx9999', which is "
        "not a metric library arch",
        "[cdna/1100.yaml] table 1102: row 4: 'colour' is not a known field",
    ]


def test_view_metric_errors():
    """The shipped gfx942 views, with a metric, a unit and a peak the library lacks,
    and a panel shown twice."""
    views = load_views("gfx942")
    metrics = load_metric_library(default_metric_library_dir(), "gfx942").metrics
    assert view_metric_errors(views, metrics) == []
    broken = copy.deepcopy(views)
    rows = dict(iter_view_rows(broken))
    rows["11.2.0"]["metric"] = "compute.made_up"
    rows["11.2.1"]["unit"] = "Hz"
    rows["2.1.11"]["peak"] = "empirical"
    broken.append(copy.deepcopy(broken[0]))
    assert view_metric_errors(broken, metrics) == [
        "panel 0 has more than one view",
        "2.1.11 compute.branch_util has no 'peak_empirical'",
        "11.2.0 compute.made_up: not a metric of the arch's library",
        "11.2.1 unit 'Hz' is not a display unit of 'Instructions per Cycle'",
    ]


def test_imports_only_profile_mode_dependencies(monkeypatch):
    """The views and the equation normalizer use only stdlib, project and vendored
    code."""
    for module in ("utils.analysis_views", "utils.equation_format"):
        monkeypatch.delitem(sys.modules, module, raising=False)
    with ProfileModeImportGuard():
        importlib.import_module("utils.analysis_views")

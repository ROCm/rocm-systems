##############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

##############################################################################

from unittest import mock

import numpy as np
import pandas as pd

from utils import schema
from utils.roofline_calc import calc_ai_analyze, sanitize_ai_value


def run_calc_ai_analyze_with_values(monkeypatch, metric_values):
    """
    Build mocks and invoke calc_ai_analyze with controlled metric values.

    ``metric_values`` is a dict with keys ``ai_hbm``, ``ai_l2``, ``ai_l1``,
    ``performance`` whose values are injected into the table-402 DataFrame that
    ``eval_metric`` would normally populate.

    Returns the plot-points dict produced by ``calc_ai_analyze``.
    """
    kernel_name = "test_kernel"
    kernel_id = 0

    workload = schema.Workload()
    workload.dfs = {1: pd.DataFrame({"Kernel_Name": [kernel_name]}, index=[kernel_id])}
    workload.sys_info = pd.DataFrame([{"gpu_arch": "gfx90a"}])
    workload.roofline_peaks = pd.DataFrame()
    workload.filter_kernel_ids = []
    workload.path = "/mock/path"

    arch_config = schema.ArchConfig()
    arch_config.dfs = {
        401: pd.DataFrame(),
        402: pd.DataFrame({
            "Metric": pd.Series(dtype="str"),
            "Value": pd.Series(dtype="object"),
        }),
    }
    arch_config.dfs_type = {401: "metric_table", 402: "metric_table"}

    pmc_data = pd.DataFrame({"Kernel_Name": [kernel_name]})
    filtered_pmc = pd.concat({"pmc_perf": pmc_data}, axis=1)
    monkeypatch.setattr(
        "utils.roofline_calc.apply_filters", lambda *a, **kw: filtered_pmc
    )

    def mock_eval_metric(
        dfs, dfs_type, sys_info_row, roofline_peaks, pmc_data, debug, config
    ):
        dfs[402] = pd.DataFrame({
            "Metric": [
                "AI HBM",
                "AI L2",
                "AI L1",
                "Performance (GFLOPs)",
            ],
            "Value": pd.array(
                [
                    metric_values["ai_hbm"],
                    metric_values["ai_l2"],
                    metric_values["ai_l1"],
                    metric_values["performance"],
                ],
                dtype=object,
            ),
        })

    monkeypatch.setattr("utils.roofline_calc.eval_metric", mock_eval_metric)

    monkeypatch.setattr("utils.roofline_calc.console_debug", lambda *a, **kw: None)
    monkeypatch.setattr("utils.roofline_calc.console_warning", lambda *a, **kw: None)

    return calc_ai_analyze(
        workload=workload,
        mspec=mock.MagicMock(),
        sort_type="kernels",
        config={},
        arch_config=arch_config,
    )


def test_calc_ai_analyze_replaces_inf_with_zero(monkeypatch):
    """np.inf / -np.inf metric values are replaced with 0."""
    result = run_calc_ai_analyze_with_values(
        monkeypatch,
        {
            "ai_hbm": np.inf,
            "ai_l2": -np.inf,
            "ai_l1": 1.5,
            "performance": 100.0,
        },
    )

    assert result["kernelNames"] == ["K0"]
    assert result["ai_hbm"][0] == [0], "np.inf should be replaced with 0"
    assert result["ai_hbm"][1] == [100.0]
    assert result["ai_l2"][0] == [0], "-np.inf should be replaced with 0"
    assert result["ai_l2"][1] == [100.0]
    assert result["ai_l1"][0] == [1.5], "valid float should pass through"
    assert result["ai_l1"][1] == [100.0]


def test_calc_ai_analyze_replaces_none_with_zero(monkeypatch):
    """None metric values are replaced with 0 and still included in plot points."""
    result = run_calc_ai_analyze_with_values(
        monkeypatch,
        {
            "ai_hbm": None,
            "ai_l2": None,
            "ai_l1": None,
            "performance": 50.0,
        },
    )

    assert result["kernelNames"] == ["K0"]
    assert result["ai_hbm"][0] == [0], "None should be replaced with 0"
    assert result["ai_l2"][0] == [0], "None should be replaced with 0"
    assert result["ai_l1"][0] == [0], "None should be replaced with 0"


def test_calc_ai_analyze_valid_values_pass_through(monkeypatch):
    """Normal positive floats pass through unchanged."""
    result = run_calc_ai_analyze_with_values(
        monkeypatch,
        {
            "ai_hbm": 2.5,
            "ai_l2": 3.0,
            "ai_l1": 1.5,
            "performance": 100.0,
        },
    )

    assert result["kernelNames"] == ["K0"]
    assert result["ai_hbm"][0] == [2.5]
    assert result["ai_hbm"][1] == [100.0]
    assert result["ai_l2"][0] == [3.0]
    assert result["ai_l2"][1] == [100.0]
    assert result["ai_l1"][0] == [1.5]
    assert result["ai_l1"][1] == [100.0]


def test_calc_ai_analyze_na_and_empty_replaced(monkeypatch):
    """Sentinel values 'N/A' and '' are replaced with 0."""
    result = run_calc_ai_analyze_with_values(
        monkeypatch,
        {
            "ai_hbm": "N/A",
            "ai_l2": "",
            "ai_l1": "N/A",
            "performance": 75.0,
        },
    )

    assert result["kernelNames"] == ["K0"]
    assert result["ai_hbm"][0] == [0], "'N/A' should be replaced with 0"
    assert result["ai_l2"][0] == [0], "'' should be replaced with 0"
    assert result["ai_l1"][0] == [0], "'N/A' should be replaced with 0"


def test_sanitize_ai_value_replaces_invalid_values_with_zero():
    """Invalid values are replaced with 0."""
    assert sanitize_ai_value(np.inf) == 0
    assert sanitize_ai_value(-np.inf) == 0
    assert sanitize_ai_value("N/A") == 0
    assert sanitize_ai_value("") == 0
    assert sanitize_ai_value(None) == 0
    assert sanitize_ai_value(1.5) == 1.5

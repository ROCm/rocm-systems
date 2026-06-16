# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT


import numpy as np
import pandas as pd

from utils import schema
from utils.mi_gpu_spec import mi_gpu_specs
from utils.roofline_calc import (
    calc_ai_analyze,
    sanitize_ai_value,
    sanitize_mem_level,
)


def run_calc_ai_analyze_with_values(monkeypatch, metric_values):
    """
    Build mocks and invoke calc_ai_analyze with controlled metric values.

    ``metric_values`` is a dict with keys ``ai_hbm``, ``ai_l2``, ``ai_l1``,
    ``ai_lds``, ``performance`` whose values are injected into the table-402
    DataFrame that ``eval_metric`` would normally populate.

    Returns the plot-points dict produced by ``calc_ai_analyze``.

    Note: this mock simulates MI350 AI metric values based on the architecture's cache
    levels available on the hardware. Cache levels will vary for other architectures.
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

    pmc_df = pd.DataFrame({"Kernel_Name": [kernel_name]})

    def mock_eval_metric(
        dfs, dfs_type, dfs_expressions, sys_info_row, roofline_peaks, pmc_data, debug
    ):
        dfs[402] = pd.DataFrame({
            "Metric": [
                "AI HBM",
                "AI L2",
                "AI L1",
                "AI LDS",
                "Performance (GFLOPs)",
            ],
            "Value": pd.array(
                [
                    metric_values["ai_hbm"],
                    metric_values["ai_l2"],
                    metric_values["ai_l1"],
                    metric_values["ai_lds"],
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
        pmc_df=pmc_df,
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
            "ai_lds": np.inf,
            "performance": 100.0,
        },
    )

    assert result["kernelNames"] == ["test_kernel"]
    assert result["ai_hbm"][0] == [0], "np.inf should be replaced with 0"
    assert result["ai_hbm"][1] == [100.0]
    assert result["ai_l2"][0] == [0], "-np.inf should be replaced with 0"
    assert result["ai_l2"][1] == [100.0]
    assert result["ai_l1"][0] == [1.5], "valid float should pass through"
    assert result["ai_l1"][1] == [100.0]
    assert result["ai_lds"][0] == [0], "np.inf should be replaced with 0"
    assert result["ai_lds"][1] == [100.0]


def test_calc_ai_analyze_replaces_none_with_zero(monkeypatch):
    """None metric values are replaced with 0 and still included in plot points."""
    result = run_calc_ai_analyze_with_values(
        monkeypatch,
        {
            "ai_hbm": None,
            "ai_l2": None,
            "ai_l1": None,
            "ai_lds": None,
            "performance": 50.0,
        },
    )

    assert result["kernelNames"] == ["test_kernel"]
    assert result["ai_hbm"][0] == [0], "None should be replaced with 0"
    assert result["ai_l2"][0] == [0], "None should be replaced with 0"
    assert result["ai_l1"][0] == [0], "None should be replaced with 0"
    assert result["ai_lds"][0] == [0], "None should be replaced with 0"


def test_calc_ai_analyze_valid_values_pass_through(monkeypatch):
    """Normal positive floats pass through unchanged."""
    result = run_calc_ai_analyze_with_values(
        monkeypatch,
        {
            "ai_hbm": 2.5,
            "ai_l2": 3.0,
            "ai_l1": 1.5,
            "ai_lds": 4.0,
            "performance": 100.0,
        },
    )

    assert result["kernelNames"] == ["test_kernel"]
    assert result["ai_hbm"][0] == [2.5]
    assert result["ai_hbm"][1] == [100.0]
    assert result["ai_l2"][0] == [3.0]
    assert result["ai_l2"][1] == [100.0]
    assert result["ai_l1"][0] == [1.5]
    assert result["ai_l1"][1] == [100.0]
    assert result["ai_lds"][0] == [4.0]
    assert result["ai_lds"][1] == [100.0]


def test_calc_ai_analyze_na_and_empty_replaced(monkeypatch):
    """Sentinel values 'N/A' and '' are replaced with 0."""
    result = run_calc_ai_analyze_with_values(
        monkeypatch,
        {
            "ai_hbm": "N/A",
            "ai_l2": "",
            "ai_l1": "N/A",
            "ai_lds": "",
            "performance": 75.0,
        },
    )

    assert result["kernelNames"] == ["test_kernel"]
    assert result["ai_hbm"][0] == [0], "'N/A' should be replaced with 0"
    assert result["ai_l2"][0] == [0], "'' should be replaced with 0"
    assert result["ai_l1"][0] == [0], "'N/A' should be replaced with 0"
    assert result["ai_lds"][0] == [0], "'' should be replaced with 0"


def test_sanitize_ai_value_replaces_invalid_values_with_zero():
    """Invalid values are replaced with 0."""
    assert sanitize_ai_value(np.inf) == 0
    assert sanitize_ai_value(-np.inf) == 0
    assert sanitize_ai_value("N/A") == 0
    assert sanitize_ai_value("") == 0
    assert sanitize_ai_value(None) == 0
    assert sanitize_ai_value(1.5) == 1.5


##############################################################################
# sanitize_mem_level Tests
##############################################################################


def test_sanitize_mem_level_all_falls_back_to_hierarchy():
    """'ALL' results in full memory levels for the model, minus MALL."""
    result = sanitize_mem_level("ALL", "mi210")
    # mi210 has no MALL, so result equals memory_levels directly
    assert result == mi_gpu_specs.get_memory_levels("mi210")


def test_sanitize_mem_level_all_list_falls_back_to_hierarchy():
    """['ALL'] behaves the same as 'ALL'."""
    result = sanitize_mem_level(["ALL"], "mi210")
    assert result == mi_gpu_specs.get_memory_levels("mi210")


def test_sanitize_mem_level_supported_string():
    """A supported single string level is returned as a single-item list."""
    result = sanitize_mem_level("HBM", "mi210")
    assert result == ["HBM"]


def test_sanitize_mem_level_supported_list():
    """A list of supported levels is returned unchanged."""
    result = sanitize_mem_level(["HBM", "L2"], "mi210")
    assert result == ["HBM", "L2"]


def test_sanitize_mem_level_unsupported_falls_back_to_hierarchy():
    """Fully unsupported input falls back to full memory levels, minus MALL."""
    result = sanitize_mem_level("HBM", "rdna35_halo")
    # rdna35_halo memory_levels includes MALL, which is stripped by sanitize_mem_level
    expected = [m for m in mi_gpu_specs.get_memory_levels("rdna35_halo") if m != "MALL"]
    assert result == expected


def test_sanitize_mem_level_mixed_filters_unsupported():
    """Unsupported levels in a mixed list are filtered out."""
    # rdna35_halo supports L0, L1, L2, MALL, LDS — not HBM; MALL is also stripped
    result = sanitize_mem_level(["HBM", "L2"], "rdna35_halo")
    assert result == ["L2"]


def test_sanitize_mem_level_mall_is_stripped():
    """MALL is always removed from results even when explicitly requested."""
    result = sanitize_mem_level("MALL", "rdna35_halo")
    assert "MALL" not in result


def test_sanitize_mem_level_vl1d_normalised():
    """'vL1D' is normalised to 'L1' before filtering."""
    result = sanitize_mem_level("vL1D", "mi210")
    assert result == ["L1"]

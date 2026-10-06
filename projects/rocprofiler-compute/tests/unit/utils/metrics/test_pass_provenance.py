# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for same-pass analyze provenance and binding."""

from __future__ import annotations

import pandas as pd

from utils.file_io import load_df_pmc
from utils.metrics.evaluation_pipeline import eval_metric
from utils.metrics.pass_provenance import (
    METRIC_ROW_PASS_ATTR,
    PassLayout,
    bind_expression,
    bind_expression_dataframe,
    build_pass_layout,
    extract_row_refs,
    natural_pass_sort_key,
    select_pass,
    select_pass_with_normalization_fallback,
)
from utils.utils_analysis import process_rocpd_csv

ROCPD_COUNTER_HEADER = (
    "GPU_ID,Dispatch_ID,Grid_Size,Workgroup_Size,LDS_Per_Workgroup,"
    "Scratch_Per_Workitem,Arch_VGPR,Accum_VGPR,SGPR,Kernel_Name,"
    "Start_Timestamp,End_Timestamp,Kernel_ID,Counter_Name,Counter_Value\n"
)
ROCPD_COUNTER_ROW_PREFIX = "0,0,256,64,0,0,8,0,16,kernel_a,10,20,0,"


def _write_gzip_csv(path, text: str) -> None:
    import gzip

    path.write_bytes(gzip.compress(text.encode("utf-8")))


def test_natural_pass_sort_key_orders_numeric_suffix() -> None:
    keys = ["pmc_perf_10", "pmc_perf_2", "pmc_perf_1"]
    assert sorted(keys, key=natural_pass_sort_key) == [
        "pmc_perf_1",
        "pmc_perf_2",
        "pmc_perf_10",
    ]


def test_load_df_pmc_shadow_columns_for_duplicates(tmp_path) -> None:
    _write_gzip_csv(
        tmp_path / "results_pmc_perf_0.csv.gz",
        ROCPD_COUNTER_HEADER
        + ROCPD_COUNTER_ROW_PREFIX
        + "GRBM_GUI_ACTIVE,100\n"
        + ROCPD_COUNTER_ROW_PREFIX
        + "TA_BUSY,50\n",
    )
    _write_gzip_csv(
        tmp_path / "results_pmc_perf_1.csv.gz",
        ROCPD_COUNTER_HEADER
        + ROCPD_COUNTER_ROW_PREFIX
        + "GRBM_GUI_ACTIVE,200\n"
        + ROCPD_COUNTER_ROW_PREFIX
        + "SQ_WAVES,4\n",
    )

    df, layout = load_df_pmc(str(tmp_path), verbose=0)

    assert layout.duplicated == frozenset({"GRBM_GUI_ACTIVE"})
    assert df["GRBM_GUI_ACTIVE"].iloc[0] == 100  # first pass base
    assert df["GRBM_GUI_ACTIVE@pass:pmc_perf_0"].iloc[0] == 100
    assert df["GRBM_GUI_ACTIVE@pass:pmc_perf_1"].iloc[0] == 200
    assert df["TA_BUSY"].iloc[0] == 50
    assert df["SQ_WAVES"].iloc[0] == 4


def test_load_df_pmc_no_duplicates_matches_unique_columns(tmp_path) -> None:
    _write_gzip_csv(
        tmp_path / "results_pmc_perf_0.csv.gz",
        ROCPD_COUNTER_HEADER + ROCPD_COUNTER_ROW_PREFIX + "SQ_WAVES,4\n",
    )
    _write_gzip_csv(
        tmp_path / "results_pmc_perf_1.csv.gz",
        ROCPD_COUNTER_HEADER + ROCPD_COUNTER_ROW_PREFIX + "SQ_BUSY_CYCLES,100\n",
    )

    df, layout = load_df_pmc(str(tmp_path), verbose=0)
    assert not layout.has_duplicates
    assert df["SQ_WAVES"].iloc[0] == 4
    assert df["SQ_BUSY_CYCLES"].iloc[0] == 100
    assert not any("@pass:" in column for column in df.columns)


def test_load_df_pmc_legacy_shape_for_iteration_multiplexing(tmp_path) -> None:
    _write_gzip_csv(
        tmp_path / "results_pmc_perf_0.csv.gz",
        ROCPD_COUNTER_HEADER + ROCPD_COUNTER_ROW_PREFIX + "GRBM_GUI_ACTIVE,100\n",
    )
    _write_gzip_csv(
        tmp_path / "results_pmc_perf_1.csv.gz",
        ROCPD_COUNTER_HEADER + ROCPD_COUNTER_ROW_PREFIX + "GRBM_GUI_ACTIVE,200\n",
    )

    df, layout = load_df_pmc(
        str(tmp_path),
        verbose=0,
        preserve_pass_provenance=False,
    )

    assert layout == PassLayout.empty()
    assert df["GRBM_GUI_ACTIVE"].iloc[0] == 200
    assert not any("@pass:" in column for column in df.columns)


def test_select_pass_and_bind_expression() -> None:
    layout = PassLayout(
        pass_keys=("pmc_perf_0", "pmc_perf_1"),
        counters_by_pass={
            "pmc_perf_0": frozenset({"GRBM_GUI_ACTIVE", "TA_BUSY"}),
            "pmc_perf_1": frozenset({"GRBM_GUI_ACTIVE", "SQ_WAVES"}),
        },
        duplicated=frozenset({"GRBM_GUI_ACTIVE"}),
    )
    required = frozenset({"TA_BUSY", "GRBM_GUI_ACTIVE"})
    assert select_pass(required, layout) == "pmc_perf_0"
    assert select_pass(frozenset({"SQ_WAVES", "TA_BUSY"}), layout) is None

    expr = "to_avg(100 * raw_pmc_df['TA_BUSY'] / raw_pmc_df['GRBM_GUI_ACTIVE'])"
    bound = bind_expression(expr, "pmc_perf_0", layout, frozenset())
    assert "GRBM_GUI_ACTIVE@pass:pmc_perf_0" in bound
    assert "raw_pmc_df['TA_BUSY']" in bound  # unique counter unchanged


def test_bind_expression_builtin_is_idempotent_and_prefix_safe() -> None:
    layout = PassLayout(
        pass_keys=("pmc_perf_0",),
        counters_by_pass={"pmc_perf_0": frozenset({"SQ_WAVES"})},
        duplicated=frozenset(),
    )
    expr = "ammolite__SQ_WAVES + ammolite__SQ_WAVES_SUM"
    bound = bind_expression(expr, "pmc_perf_0", layout, frozenset({"SQ_WAVES"}))
    assert bound == "ammolite__SQ_WAVES__pass0 + ammolite__SQ_WAVES_SUM"
    rebound = bind_expression(bound, "pmc_perf_0", layout, frozenset({"SQ_WAVES"}))
    assert rebound == bound


def test_extract_row_refs_ignores_system_vars() -> None:
    refs = extract_row_refs([
        "to_avg(raw_pmc_df['SQ_WAVES'] / ammolite__cu_per_gpu)",
        "to_avg(raw_pmc_df['SQ_BUSY_CU_CYCLES'] / ammolite__GRBM_GUI_ACTIVE_PER_XCD)",
    ])
    assert "SQ_WAVES" in refs.direct_counters
    assert "cu_per_gpu" not in refs.builtin_vars
    assert "GRBM_GUI_ACTIVE_PER_XCD" in refs.builtin_vars


def test_select_pass_returns_none_when_counters_span_passes() -> None:
    layout = PassLayout(
        pass_keys=("pmc_perf_0", "pmc_perf_1"),
        counters_by_pass={
            "pmc_perf_0": frozenset({"A"}),
            "pmc_perf_1": frozenset({"B"}),
        },
        duplicated=frozenset(),
    )
    assert select_pass(frozenset({"A", "B"}), layout) is None


def test_select_pass_relaxes_only_runtime_normalization_counter() -> None:
    layout = PassLayout(
        pass_keys=("pmc_perf_0", "pmc_perf_1"),
        counters_by_pass={
            "pmc_perf_0": frozenset({"TA_BUSY"}),
            "pmc_perf_1": frozenset({"SQ_WAVES"}),
        },
        duplicated=frozenset(),
    )

    selected, relaxed = select_pass_with_normalization_fallback(
        frozenset({"TA_BUSY", "SQ_WAVES"}),
        layout,
        "MI300",
    )

    assert selected == "pmc_perf_0"
    assert relaxed


def test_normalization_fallback_reports_failure_when_formula_still_spans_passes():
    layout = PassLayout(
        pass_keys=("pmc_perf_0", "pmc_perf_1"),
        counters_by_pass={
            "pmc_perf_0": frozenset({"TA_BUSY", "SQ_WAVES"}),
            "pmc_perf_1": frozenset({"TD_BUSY"}),
        },
        duplicated=frozenset(),
    )

    selected, relaxed = select_pass_with_normalization_fallback(
        frozenset({"TA_BUSY", "TD_BUSY", "SQ_WAVES"}),
        layout,
        "MI300",
    )

    assert selected is None
    assert not relaxed


def test_process_rocpd_csv_skips_provenance_merge_without_duplicates(
    monkeypatch,
) -> None:
    long_df = pd.DataFrame({
        "GPU_ID": [0, 0],
        "Dispatch_ID": [0, 0],
        "Grid_Size": [256, 256],
        "Workgroup_Size": [64, 64],
        "LDS_Per_Workgroup": [0, 0],
        "Scratch_Per_Workitem": [0, 0],
        "Arch_VGPR": [8, 8],
        "Accum_VGPR": [0, 0],
        "SGPR": [16, 16],
        "Kernel_Name": ["k", "k"],
        "Start_Timestamp": [10, 10],
        "End_Timestamp": [20, 20],
        "Kernel_ID": [0, 0],
        "Counter_Name": ["SQ_WAVES", "TA_BUSY"],
        "Counter_Value": [4, 50],
        "Pass_Key": ["pmc_perf_0", "pmc_perf_1"],
    })
    layout = build_pass_layout(long_df)

    def fail_provenance_merge(*_args, **_kwargs):
        raise AssertionError

    monkeypatch.setattr(
        "utils.utils_analysis._merge_counters_with_pass_provenance",
        fail_provenance_merge,
    )

    result = process_rocpd_csv(long_df, pass_layout=layout)

    assert result["SQ_WAVES"].iloc[0] == 4
    assert result["TA_BUSY"].iloc[0] == 50


def test_bind_expression_dataframe_groups_by_metric_id() -> None:
    layout = PassLayout(
        pass_keys=("pmc_perf_0", "pmc_perf_1"),
        counters_by_pass={
            "pmc_perf_0": frozenset({"GRBM_GUI_ACTIVE", "SQ_WAVES"}),
            "pmc_perf_1": frozenset({"GRBM_GUI_ACTIVE", "TA_BUSY"}),
        },
        duplicated=frozenset({"GRBM_GUI_ACTIVE"}),
    )
    expression_df = pd.DataFrame({
        "metric_id": ["9.9.9", "9.9.9", "1.0.0"],
        "value_name": ["Avg", "Max", "Avg"],
        "value": [
            "to_avg(100 * raw_pmc_df['TA_BUSY'] / raw_pmc_df['GRBM_GUI_ACTIVE'])",
            "to_max(100 * raw_pmc_df['TA_BUSY'] / raw_pmc_df['GRBM_GUI_ACTIVE'])",
            "to_avg(raw_pmc_df['SQ_WAVES'] / raw_pmc_df['GRBM_GUI_ACTIVE'])",
        ],
    })

    used = bind_expression_dataframe(expression_df, layout, "MI300")

    assert used == {"pmc_perf_0", "pmc_perf_1"}
    assert expression_df.attrs[METRIC_ROW_PASS_ATTR]["9.9.9"] == "pmc_perf_1"
    assert expression_df.attrs[METRIC_ROW_PASS_ATTR]["1.0.0"] == "pmc_perf_0"
    assert "GRBM_GUI_ACTIVE@pass:pmc_perf_1" in expression_df.at[0, "value"]
    assert "GRBM_GUI_ACTIVE@pass:pmc_perf_1" in expression_df.at[1, "value"]
    assert "GRBM_GUI_ACTIVE@pass:pmc_perf_0" in expression_df.at[2, "value"]


def test_eval_metric_uses_same_pass_for_duplicated_denominator() -> None:
    """TA_BUSY only in pass 1; GRBM in both — must use pass-1 GRBM (200)."""
    # pass 0: GRBM=100, SQ_WAVES=4
    # pass 1: GRBM=200, TA_BUSY=50
    long_df = pd.DataFrame({
        "GPU_ID": [0, 0, 0, 0],
        "Dispatch_ID": [0, 0, 0, 0],
        "Grid_Size": [256, 256, 256, 256],
        "Workgroup_Size": [64, 64, 64, 64],
        "LDS_Per_Workgroup": [0, 0, 0, 0],
        "Scratch_Per_Workitem": [0, 0, 0, 0],
        "Arch_VGPR": [8, 8, 8, 8],
        "Accum_VGPR": [0, 0, 0, 0],
        "SGPR": [16, 16, 16, 16],
        "Kernel_Name": ["k"] * 4,
        "Start_Timestamp": [10, 10, 10, 10],
        "End_Timestamp": [20, 20, 20, 20],
        "Kernel_ID": [0, 0, 0, 0],
        "Counter_Name": [
            "GRBM_GUI_ACTIVE",
            "SQ_WAVES",
            "GRBM_GUI_ACTIVE",
            "TA_BUSY",
        ],
        "Counter_Value": [100, 4, 200, 50],
        "Pass_Key": [
            "pmc_perf_0",
            "pmc_perf_0",
            "pmc_perf_1",
            "pmc_perf_1",
        ],
    })
    layout = build_pass_layout(long_df)
    raw_pmc_df = process_rocpd_csv(long_df, pass_layout=layout)

    metric_df = pd.DataFrame({
        "Metric_ID": ["9.9.9"],
        "Metric": ["Util"],
        "Avg": ["to_avg(100 * raw_pmc_df['TA_BUSY'] / raw_pmc_df['GRBM_GUI_ACTIVE'])"],
    }).set_index("Metric_ID")
    dfs = {1: metric_df}
    dfs_type = {1: "metric_table"}
    dfs_expressions = {1: [metric_df.at["9.9.9", "Avg"]]}
    sys_info = pd.Series({
        "ip_blocks": "standard",
        "gpu_arch": "gfx942",
        "se_per_gpu": 4,
        "sa_per_se": 1,
        "pipes_per_gpu": 4,
        "cu_per_gpu": 38,
        "simd_per_cu": 4,
        "sqc_per_gpu": 16,
        "lds_banks_per_cu": 32,
        "cur_sclk": 1800.0,
        "cur_mclk": 1200.0,
        "max_sclk": 2100.0,
        "max_mclk": 1600.0,
        "max_waves_per_cu": 32,
        "num_memory_channels": 4,
        "total_l2_chan": 16,
        "num_xcd": 1,
        "wave_size": 64,
    })

    eval_metric(
        dfs,
        dfs_type,
        dfs_expressions,
        sys_info,
        pd.DataFrame(),
        raw_pmc_df,
        debug=False,
        pass_layout=layout,
    )

    # Same-pass: 100 * 50 / 200 = 25. Cross-pass base would be 100 * 50 / 100 = 50.
    assert float(dfs[1].at["9.9.9", "Avg"]) == 25.0
    assert dfs[1].attrs.get("metric_row_pass", {}).get("9.9.9") == "pmc_perf_1"

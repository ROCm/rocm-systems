# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for same-pass analyze provenance and binding."""

import importlib
import logging

import pandas as pd
import pytest

from utils import utils_analysis
from utils.file_io import create_df_pmc_with_slot_ids, load_df_pmc
from utils.metrics.evaluation_pipeline import eval_metric
from utils.metrics.expression import build_eval_string, update_denominator_string
from utils.metrics.pass_layout import PassLayout
from utils.metrics.pass_provenance import (
    METRIC_ROW_PASS_ATTR,
    bind_expression,
    bind_expression_dataframe,
    bind_metric_tables_to_passes,
    build_pass_layout,
    choose_metric_pass,
    evaluate_pass_scoped_builtins,
    expand_required_counters,
    extract_row_refs,
    natural_pass_sort_key,
    resolve_weight_counter_column,
    select_pass,
    select_pass_with_normalization_fallback,
)
from utils.utils_analysis import process_rocpd_csv
from utils.utils_counter_defs import extract_counters_and_variables

ROCPD_COUNTER_HEADER = (
    "GPU_ID,Dispatch_ID,Grid_Size,Workgroup_Size,LDS_Per_Workgroup,"
    "Scratch_Per_Workitem,Arch_VGPR,Accum_VGPR,SGPR,Kernel_Name,"
    "Start_Timestamp,End_Timestamp,Kernel_ID,Counter_Name,Counter_Value\n"
)
ROCPD_COUNTER_ROW_PREFIX = "0,0,256,64,0,0,8,0,16,kernel_a,10,20,0,"

DISPATCH_FIELDS = {
    "GPU_ID": 0,
    "Dispatch_ID": 0,
    "Grid_Size": 256,
    "Workgroup_Size": 64,
    "LDS_Per_Workgroup": 0,
    "Scratch_Per_Workitem": 0,
    "Arch_VGPR": 8,
    "Accum_VGPR": 0,
    "SGPR": 16,
    "Kernel_Name": "k",
    "Start_Timestamp": 10,
    "End_Timestamp": 20,
    "Kernel_ID": 0,
}

# TA_BUSY and TD_BUSY are collected once each, in different passes.
SPLIT_LAYOUT = PassLayout(
    pass_keys=("pmc_perf_0", "pmc_perf_1"),
    counters_by_pass={
        "pmc_perf_0": frozenset({"TA_BUSY", "GRBM_GUI_ACTIVE"}),
        "pmc_perf_1": frozenset({"TD_BUSY", "GRBM_GUI_ACTIVE"}),
    },
    duplicated=frozenset({"GRBM_GUI_ACTIVE"}),
)
UNIQUE_COUNTERS_EXPR = "to_avg(raw_pmc_df['TA_BUSY'] / raw_pmc_df['TD_BUSY'])"
DUPLICATED_COUNTER_EXPR = (
    "to_avg((raw_pmc_df['TA_BUSY'] + raw_pmc_df['TD_BUSY'])"
    " / raw_pmc_df['GRBM_GUI_ACTIVE'])"
)
PER_WAVE_EXPR = "to_avg(raw_pmc_df['TA_BUSY'] / raw_pmc_df['SQ_WAVES'])"
PER_CYCLE_SUM_EXPR = "to_avg(raw_pmc_df['TA_BUSY'] / raw_pmc_df['GRBM_GUI_ACTIVE_sum'])"


def _counter_row(
    dispatch_id: int,
    counter_name: str,
    counter_value: float,
    pass_key: str,
) -> dict:
    return {
        "GPU_ID": 0,
        "Dispatch_ID": dispatch_id,
        "Grid_Size": 256,
        "Workgroup_Size": 64,
        "LDS_Per_Workgroup": 0,
        "Scratch_Per_Workitem": 0,
        "Arch_VGPR": 8,
        "Accum_VGPR": 0,
        "SGPR": 16,
        "Kernel_Name": "k",
        "Start_Timestamp": 10,
        "End_Timestamp": 20,
        "Kernel_ID": 0,
        "Counter_Name": counter_name,
        "Counter_Value": counter_value,
        "Pass_Key": pass_key,
    }


def _layout_with_divisor_on_another_pass(divisor: str) -> PassLayout:
    """TA_BUSY is copied across two passes. ``divisor`` is only on a third."""
    return PassLayout(
        pass_keys=("pmc_perf_0", "pmc_perf_1", "pmc_perf_2"),
        counters_by_pass={
            "pmc_perf_0": frozenset({"TA_BUSY"}),
            "pmc_perf_1": frozenset({"TA_BUSY"}),
            "pmc_perf_2": frozenset({divisor}),
        },
        duplicated=frozenset({"TA_BUSY"}),
    )


def _assert_normalization_metric_stays_unset(
    expression: str,
    divisor: str,
    gpu_series: str,
) -> None:
    layout = _layout_with_divisor_on_another_pass(divisor)
    selected, relaxed = select_pass_with_normalization_fallback(
        frozenset({"TA_BUSY", divisor}),
        layout,
        gpu_series,
    )
    assert selected is None
    assert not relaxed

    metric_df = pd.DataFrame({"Avg": [expression], "Min": ["None"]})
    used = bind_metric_tables_to_passes(
        {1: metric_df},
        {1: "metric_table"},
        layout,
        gpu_series,
        frozenset({"Avg", "Min"}),
    )
    assert used == set()
    assert metric_df.loc[0].tolist() == ["None", "None"]


def _write_gzip_csv(path, text: str) -> None:
    import gzip

    path.write_bytes(gzip.compress(text.encode("utf-8")))


def _sub_metric_frame(pass_key: str) -> pd.DataFrame:
    frame = pd.DataFrame({"Metric": ["read_sub"]})
    frame.attrs[METRIC_ROW_PASS_ATTR] = {frame.index[0]: pass_key}
    return frame


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


def test_operator_loader_keeps_duplicated_counter_pass_columns(tmp_path) -> None:
    """Operator filtering must keep the same shadow columns as the kernel path."""
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

    df, slot_to_id, layout = create_df_pmc_with_slot_ids(str(tmp_path), verbose=0)

    assert layout.duplicated == frozenset({"GRBM_GUI_ACTIVE"})
    assert df["GRBM_GUI_ACTIVE"].iloc[0] == 100
    assert df["GRBM_GUI_ACTIVE@pass:pmc_perf_1"].iloc[0] == 200
    assert len(slot_to_id) == 1


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
    shadow = {column for column in df.columns if "@pass:" in column}
    assert shadow == {
        "Start_Timestamp@pass:pmc_perf_0",
        "End_Timestamp@pass:pmc_perf_0",
        "Start_Timestamp@pass:pmc_perf_1",
        "End_Timestamp@pass:pmc_perf_1",
    }


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


def test_dispatch_unit_does_not_block_same_pass_bind() -> None:
    layout = PassLayout(
        pass_keys=("pmc_perf_0", "pmc_perf_1"),
        counters_by_pass={
            "pmc_perf_0": frozenset({"GRBM_GUI_ACTIVE"}),
            "pmc_perf_1": frozenset({"GRBM_GUI_ACTIVE", "TA_BUSY"}),
        },
        duplicated=frozenset({"GRBM_GUI_ACTIVE"}),
    )
    expression_df = pd.DataFrame({
        "metric_id": ["1.0.0"],
        "value": [
            "to_avg(raw_pmc_df['TA_BUSY'] / raw_pmc_df['GRBM_GUI_ACTIVE']"
            " / raw_pmc_df['Dispatch_Unit'])"
        ],
    })

    used = bind_expression_dataframe(expression_df, layout, "MI300")

    assert used == {"pmc_perf_1"}
    assert "GRBM_GUI_ACTIVE@pass:pmc_perf_1" in expression_df.at[0, "value"]
    assert "Dispatch_Unit" in expression_df.at[0, "value"]


def test_extract_row_refs_ignores_non_pmc_frame_columns() -> None:
    refs = extract_row_refs([
        "to_avg(raw_pmc_df['TA_BUSY'] / raw_pmc_df['Dispatch_Unit'])",
        "to_avg(raw_pmc_df['TA_BUSY'] / "
        "(raw_pmc_df['End_Timestamp'] - raw_pmc_df['Start_Timestamp']))",
        "to_avg(raw_pmc_df['Arch_VGPR'])",
        "to_avg(raw_pmc_df['Grid_Size'])",
    ])
    assert refs.direct_counters == frozenset({"TA_BUSY"})


def test_kernel_launch_fields_are_not_cleared_by_pass_bind() -> None:
    """VGPR, grid size, and the other launch fields are not perfmon counters."""
    layout = PassLayout(
        pass_keys=("pmc_perf_0",),
        counters_by_pass={"pmc_perf_0": frozenset({"SQ_WAVES"})},
        duplicated=frozenset(),
    )
    launch_exprs = [
        "to_avg(raw_pmc_df['Arch_VGPR'])",
        "to_avg(raw_pmc_df['Accum_VGPR'])",
        "to_avg(raw_pmc_df['SGPR'])",
        "to_avg(raw_pmc_df['Grid_Size'])",
        "to_avg(raw_pmc_df['Workgroup_Size'])",
        "to_avg(raw_pmc_df['LDS_Per_Workgroup'])",
        "to_avg(raw_pmc_df['Scratch_Per_Workitem'])",
    ]
    metric_df = pd.DataFrame({"Avg": launch_exprs, "Min": launch_exprs})
    used = bind_metric_tables_to_passes(
        {1: metric_df},
        {1: "metric_table"},
        layout,
        "MI300",
        frozenset({"Avg", "Min"}),
    )
    assert used == set()
    assert metric_df["Avg"].tolist() == launch_exprs
    assert metric_df["Min"].tolist() == launch_exprs


def test_extract_row_refs_normalizes_qualified_columns() -> None:
    refs = extract_row_refs([
        "to_avg(raw_pmc_df['GRBM_GUI_ACTIVE@pass:pmc_perf_1'] / raw_pmc_df['TA_BUSY'])",
    ])
    assert refs.direct_counters == frozenset({"GRBM_GUI_ACTIVE", "TA_BUSY"})


def test_build_pass_layout_rejects_unsafe_pass_key() -> None:
    long_df = pd.DataFrame({
        "Pass_Key": ["pmc_perf_0'); __import__('os')"],
        "Counter_Name": ["SQ_WAVES"],
    })
    with pytest.raises(ValueError, match="pass key"):
        build_pass_layout(long_df)


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


def test_denominator_only_on_another_pass_does_not_bind(caplog) -> None:
    """SQ_WAVES or GRBM_GUI_ACTIVE on a different pass leaves the metric unset."""
    sq_layout = PassLayout(
        pass_keys=("pmc_perf_0", "pmc_perf_1"),
        counters_by_pass={
            "pmc_perf_0": frozenset({"TA_BUSY"}),
            "pmc_perf_1": frozenset({"SQ_WAVES"}),
        },
        duplicated=frozenset(),
    )
    selected, relaxed = select_pass_with_normalization_fallback(
        frozenset({"TA_BUSY", "SQ_WAVES"}),
        sq_layout,
        "MI300",
    )
    assert selected is None
    assert not relaxed

    grbm_layout = PassLayout(
        pass_keys=("pmc_perf_0", "pmc_perf_1"),
        counters_by_pass={
            "pmc_perf_0": frozenset({"TA_BUSY"}),
            "pmc_perf_1": frozenset({"GRBM_GUI_ACTIVE"}),
        },
        duplicated=frozenset(),
    )
    selected, relaxed = select_pass_with_normalization_fallback(
        frozenset({"TA_BUSY", "GRBM_GUI_ACTIVE"}),
        grbm_layout,
        "MI300",
    )
    assert selected is None
    assert not relaxed

    with caplog.at_level(logging.WARNING):
        warned = choose_metric_pass(
            frozenset({"TA_BUSY", "SQ_WAVES"}),
            sq_layout,
            "MI300",
            "row sq",
        )
    assert warned is None
    assert "leaving the metric unset" in caplog.text
    assert "SQ_WAVES" in caplog.text


def test_same_pass_denominator_still_selects_that_pass() -> None:
    layout = PassLayout(
        pass_keys=("pmc_perf_0", "pmc_perf_1"),
        counters_by_pass={
            "pmc_perf_0": frozenset({"TA_BUSY", "SQ_WAVES", "GRBM_GUI_ACTIVE"}),
            "pmc_perf_1": frozenset({"TD_BUSY"}),
        },
        duplicated=frozenset(),
    )

    selected, relaxed = select_pass_with_normalization_fallback(
        frozenset({"TA_BUSY", "SQ_WAVES", "GRBM_GUI_ACTIVE"}),
        layout,
        "MI300",
    )

    assert selected == "pmc_perf_0"
    assert not relaxed
    assert (
        choose_metric_pass(
            frozenset({"TA_BUSY", "SQ_WAVES", "GRBM_GUI_ACTIVE"}),
            layout,
            "MI300",
            "row same",
        )
        == "pmc_perf_0"
    )


def test_per_wave_metric_is_unset_when_sq_waves_is_on_another_pass() -> None:
    """A per-wave result stays N/A instead of borrowing SQ_WAVES from another pass."""
    _assert_normalization_metric_stays_unset(PER_WAVE_EXPR, "SQ_WAVES", "MI300")


def test_per_cycle_metric_is_unset_when_gui_active_sum_is_on_another_pass() -> None:
    """gfx1250 per-cycle stays N/A instead of borrowing GRBM_GUI_ACTIVE_sum."""
    _assert_normalization_metric_stays_unset(
        PER_CYCLE_SUM_EXPR,
        "GRBM_GUI_ACTIVE_sum",
        "GFX1250_SERIES",
    )


def test_denom_choice_uses_only_the_counter_on_the_selected_pass(caplog) -> None:
    """$denom is expanded before the pass is chosen.

    Packing does not add SQ_WAVES or GRBM_GUI_ACTIVE for $denom. per_kernel
    and per_second still use the pass that holds the metric counter. per_wave
    and per_cycle stay unset when that counter is only on another pass.
    """
    formula = "SUM(TA_BUSY) / SUM($denom)"
    packing_counters, _variables = extract_counters_and_variables(
        formula, "MI300", include_supported_denom=False
    )
    assert packing_counters == {"TA_BUSY"}

    layout = PassLayout(
        pass_keys=("pmc_perf_0", "pmc_perf_1"),
        counters_by_pass={
            "pmc_perf_0": frozenset({"TA_BUSY"}),
            "pmc_perf_1": frozenset({"SQ_WAVES", "GRBM_GUI_ACTIVE"}),
        },
        duplicated=frozenset(),
    )
    expected_pass = {
        "per_kernel": "pmc_perf_0",
        "per_second": "pmc_perf_0",
        "per_wave": None,
        "per_cycle": None,
    }
    with caplog.at_level(logging.WARNING):
        for unit, chosen_pass in expected_pass.items():
            built = build_eval_string(update_denominator_string(formula, unit))
            required = expand_required_counters(extract_row_refs([built]), "MI300")
            frame = pd.DataFrame({"Avg": [built]})
            bind_metric_tables_to_passes(
                {1: frame},
                {1: "metric_table"},
                layout,
                "MI300",
                frozenset({"Avg"}),
            )
            assert choose_metric_pass(required, layout, "MI300", unit) == chosen_pass
            if chosen_pass is None:
                assert frame.at[0, "Avg"] == ""
            else:
                assert "raw_pmc_df['TA_BUSY']" in frame.at[0, "Avg"]
    assert "SQ_WAVES" in caplog.text
    assert "GRBM_GUI_ACTIVE" in caplog.text


def test_formula_that_spans_passes_stays_unset_with_its_denominator() -> None:
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


def test_missing_pass_cell_is_nan_and_warned(monkeypatch) -> None:
    rows = []
    for dispatch_id, pass_key, value in (
        (0, "pmc_perf_0", 100),
        (0, "pmc_perf_1", 200),
        (1, "pmc_perf_0", 50),
    ):
        rows.append({
            "GPU_ID": 0,
            "Dispatch_ID": dispatch_id,
            "Grid_Size": 256,
            "Workgroup_Size": 64,
            "LDS_Per_Workgroup": 0,
            "Scratch_Per_Workitem": 0,
            "Arch_VGPR": 8,
            "Accum_VGPR": 0,
            "SGPR": 16,
            "Kernel_Name": "k",
            "Start_Timestamp": 10,
            "End_Timestamp": 20,
            "Kernel_ID": 0,
            "Counter_Name": "GRBM_GUI_ACTIVE",
            "Counter_Value": value,
            "Pass_Key": pass_key,
        })
    long_df = pd.DataFrame(rows)
    warnings = []
    monkeypatch.setattr(
        "utils.utils_analysis.console_warning",
        lambda *args: warnings.append(args),
    )
    result, _slot_to_id = process_rocpd_csv(
        long_df, pass_layout=build_pass_layout(long_df)
    )
    missing = result.loc[
        result["GRBM_GUI_ACTIVE"] == 50, "GRBM_GUI_ACTIVE@pass:pmc_perf_1"
    ]
    assert missing.isna().all()
    assert warnings


def _two_pass_long_df(
    counters: list, values: list, passes: list, starts: list, ends: list
) -> pd.DataFrame:
    rows = len(counters)
    return pd.DataFrame({
        "GPU_ID": [0] * rows,
        "Dispatch_ID": [0] * rows,
        "Grid_Size": [256] * rows,
        "Workgroup_Size": [64] * rows,
        "LDS_Per_Workgroup": [0] * rows,
        "Scratch_Per_Workitem": [0] * rows,
        "Arch_VGPR": [8] * rows,
        "Accum_VGPR": [0] * rows,
        "SGPR": [16] * rows,
        "Kernel_Name": ["k"] * rows,
        "Start_Timestamp": starts,
        "End_Timestamp": ends,
        "Kernel_ID": [0] * rows,
        "Counter_Name": counters,
        "Counter_Value": values,
        "Pass_Key": passes,
    })


def _eval_sys_info() -> pd.Series:
    return pd.Series({
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


def _eval_one_metric(long_df: pd.DataFrame, expr: str) -> pd.DataFrame:
    layout = build_pass_layout(long_df)
    raw_pmc_df, _slot_to_id = process_rocpd_csv(long_df, pass_layout=layout)
    metric_df = pd.DataFrame({
        "Metric_ID": ["9.9.9"],
        "Metric": ["m"],
        "Avg": [expr],
    }).set_index("Metric_ID")
    eval_metric(
        {1: metric_df},
        {1: "metric_table"},
        {1: [expr]},
        _eval_sys_info(),
        pd.DataFrame(),
        raw_pmc_df,
        debug=False,
        pass_layout=layout,
    )
    return metric_df


def test_process_rocpd_csv_keeps_counters_bare_without_duplicates() -> None:
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

    result, _slot_to_id = process_rocpd_csv(long_df, pass_layout=layout)

    assert result["SQ_WAVES"].iloc[0] == 4
    assert result["TA_BUSY"].iloc[0] == 50
    assert "SQ_WAVES@pass:pmc_perf_0" not in result.columns
    assert result["End_Timestamp@pass:pmc_perf_1"].iloc[0] == 20


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

    rebound = bind_expression_dataframe(expression_df, layout, "MI300")
    assert rebound == used
    assert "GRBM_GUI_ACTIVE@pass:pmc_perf_1" in expression_df.at[0, "value"]
    assert expression_df.at[0, "value"].count("@pass:") == 1


def test_bind_leaves_metric_unset_when_unique_counters_span_passes() -> None:
    expression_df = pd.DataFrame({
        "metric_id": ["1.0.0"],
        "value": [UNIQUE_COUNTERS_EXPR],
    })

    used = bind_expression_dataframe(expression_df, SPLIT_LAYOUT, "MI300")

    assert used == set()
    assert expression_df.at[0, "value"] == "None"
    assert expression_df.attrs[METRIC_ROW_PASS_ATTR] == {}

    metric_df = pd.DataFrame({"Avg": [UNIQUE_COUNTERS_EXPR]})
    table_used = bind_metric_tables_to_passes(
        {1: metric_df},
        {1: "metric_table"},
        SPLIT_LAYOUT,
        "MI300",
        frozenset({"Avg"}),
    )
    assert table_used == set()
    assert metric_df.at[0, "Avg"] == "None"


def test_bind_leaves_metric_unset_when_duplicated_counter_spans_passes() -> None:
    expression_df = pd.DataFrame({
        "metric_id": ["1.0.0", "1.0.0"],
        "value": [DUPLICATED_COUNTER_EXPR, "None"],
    })

    used = bind_expression_dataframe(expression_df, SPLIT_LAYOUT, "MI300")

    assert used == set()
    assert expression_df["value"].tolist() == ["None", "None"]
    assert expression_df.attrs[METRIC_ROW_PASS_ATTR] == {}

    metric_df = pd.DataFrame({"Avg": [DUPLICATED_COUNTER_EXPR], "Min": ["None"]})
    table_used = bind_metric_tables_to_passes(
        {1: metric_df},
        {1: "metric_table"},
        SPLIT_LAYOUT,
        "MI300",
        frozenset({"Avg", "Min"}),
    )
    assert table_used == set()
    assert metric_df.loc[0].tolist() == ["None", "None"]


def test_both_binders_choose_the_pass_in_one_helper(monkeypatch) -> None:
    seen = []

    def record_choice(label, expressions, layout, gpu_series):
        seen.append((label, tuple(expressions)))
        return "pmc_perf_0", False

    monkeypatch.setattr(
        "utils.metrics.pass_provenance._choose_metric_pass", record_choice
    )
    expression_df = pd.DataFrame({
        "metric_id": ["1.0.0"],
        "value": [DUPLICATED_COUNTER_EXPR],
    })
    metric_df = pd.DataFrame({"Avg": [DUPLICATED_COUNTER_EXPR]})

    bind_expression_dataframe(expression_df, SPLIT_LAYOUT, "MI300")
    bind_metric_tables_to_passes(
        {1: metric_df},
        {1: "metric_table"},
        SPLIT_LAYOUT,
        "MI300",
        frozenset({"Avg"}),
    )

    assert seen == [
        ("metric 1.0.0", (DUPLICATED_COUNTER_EXPR,)),
        ("row 0", (DUPLICATED_COUNTER_EXPR,)),
    ]


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
    raw_pmc_df, _slot_to_id = process_rocpd_csv(long_df, pass_layout=layout)

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


def test_missing_denominator_keeps_pooled_ratio() -> None:
    """One dispatch lacks the denominator; the pooled ratio stays 50.

    Dispatch 1 is 50/100. Dispatch 2 has numerator 50 and no denominator,
    because that pass did not run. Two complete 50/100 dispatches pool to
    100/200 = 50. The numerator-only sum still includes both dispatches.
    """
    long_df = pd.DataFrame([
        _counter_row(0, "NUMERATOR", 50, "pmc_perf_0"),
        _counter_row(0, "DENOMINATOR", 100, "pmc_perf_0"),
        _counter_row(1, "NUMERATOR", 50, "pmc_perf_0"),
    ])
    layout = build_pass_layout(long_df)
    raw_pmc_df, _slot_to_id = process_rocpd_csv(long_df, pass_layout=layout)
    assert raw_pmc_df["DENOMINATOR"].isna().sum() == 1
    assert raw_pmc_df["NUMERATOR"].sum() == 100

    metric_df = pd.DataFrame({
        "Metric_ID": ["9.9.9", "9.9.8"],
        "Metric": ["Ratio", "Numerator"],
        "Avg": [
            (
                "100 * to_sum(raw_pmc_df['NUMERATOR'])"
                " / to_sum(raw_pmc_df['DENOMINATOR'])"
            ),
            "to_sum(raw_pmc_df['NUMERATOR'])",
        ],
    }).set_index("Metric_ID")
    dfs = {1: metric_df}
    dfs_type = {1: "metric_table"}
    dfs_expressions = {1: metric_df["Avg"].tolist()}
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

    assert float(dfs[1].at["9.9.9", "Avg"]) == 50.0
    assert float(dfs[1].at["9.9.8", "Avg"]) == 100.0


def test_weight_counter_uses_the_sub_metric_pass() -> None:
    layout = PassLayout(
        pass_keys=("pmc_perf_0", "pmc_perf_1"),
        counters_by_pass={
            "pmc_perf_0": frozenset({"TCC_EA0_RDREQ_sum", "SQ_A"}),
            "pmc_perf_1": frozenset({"TCC_EA0_RDREQ_sum", "SQ_A"}),
        },
        duplicated=frozenset({"TCC_EA0_RDREQ_sum", "SQ_A"}),
    )
    bound = resolve_weight_counter_column(
        "TCC_EA0_RDREQ_sum",
        "read_sub",
        _sub_metric_frame("pmc_perf_1"),
        layout,
    )
    assert bound == "TCC_EA0_RDREQ_sum@pass:pmc_perf_1"


def test_weight_counter_refuses_a_different_pass() -> None:
    layout = PassLayout(
        pass_keys=("pmc_perf_0", "pmc_perf_1"),
        counters_by_pass={
            "pmc_perf_0": frozenset({"TCC_EA0_RDREQ_sum", "SQ_A"}),
            "pmc_perf_1": frozenset({"SQ_A", "SQ_B"}),
        },
        duplicated=frozenset({"SQ_A"}),
    )
    bound = resolve_weight_counter_column(
        "TCC_EA0_RDREQ_sum",
        "read_sub",
        _sub_metric_frame("pmc_perf_1"),
        layout,
    )
    assert bound == "TCC_EA0_RDREQ_sum@pass:pmc_perf_1"


def test_weight_counter_keeps_bare_name_when_its_only_copy_is_that_pass() -> None:
    layout = PassLayout(
        pass_keys=("pmc_perf_0", "pmc_perf_1"),
        counters_by_pass={
            "pmc_perf_0": frozenset({"SQ_A"}),
            "pmc_perf_1": frozenset({"SQ_A", "TCC_EA0_RDREQ_sum"}),
        },
        duplicated=frozenset({"SQ_A"}),
    )
    bound = resolve_weight_counter_column(
        "TCC_EA0_RDREQ_sum",
        "read_sub",
        _sub_metric_frame("pmc_perf_1"),
        layout,
    )
    assert bound == "TCC_EA0_RDREQ_sum"


def test_weight_only_in_another_pass_is_not_the_bare_column() -> None:
    layout = PassLayout(
        pass_keys=("pmc_perf_0", "pmc_perf_1"),
        counters_by_pass={
            "pmc_perf_0": frozenset({"TCC_EA0_RDREQ_sum"}),
            "pmc_perf_1": frozenset({"SQ_A"}),
        },
        duplicated=frozenset(),
    )
    bound = resolve_weight_counter_column(
        "TCC_EA0_RDREQ_sum",
        "read_sub",
        _sub_metric_frame("pmc_perf_1"),
        layout,
    )
    assert bound == "TCC_EA0_RDREQ_sum@pass:pmc_perf_1"


def test_weight_counter_on_a_single_pass_layout_is_the_bare_column() -> None:
    layout = PassLayout(
        pass_keys=("pmc_perf_0",),
        counters_by_pass={"pmc_perf_0": frozenset({"TCC_EA0_RDREQ_sum", "SQ_A"})},
        duplicated=frozenset(),
    )
    frame = pd.DataFrame({"Metric": ["read_sub"]}, index=["1"])
    bound = resolve_weight_counter_column(
        "TCC_EA0_RDREQ_sum", "read_sub", frame, layout
    )
    assert bound == "TCC_EA0_RDREQ_sum"


def test_bind_moves_timestamps_onto_the_pmc_chosen_pass() -> None:
    layout = PassLayout(
        pass_keys=("pmc_perf_0", "pmc_perf_1"),
        counters_by_pass={
            "pmc_perf_0": frozenset({"TCP_TOTAL_CACHE_ACCESSES_sum"}),
            "pmc_perf_1": frozenset({"TCP_TOTAL_CACHE_ACCESSES_sum", "SQ_WAVES"}),
        },
        duplicated=frozenset({"TCP_TOTAL_CACHE_ACCESSES_sum"}),
    )
    expr = (
        "to_sum(raw_pmc_df['TCP_TOTAL_CACHE_ACCESSES_sum']) / "
        "to_sum(raw_pmc_df['End_Timestamp'] - raw_pmc_df['Start_Timestamp'])"
    )
    df = pd.DataFrame({"Avg": [expr], "Metric": ["rate"]})
    bind_metric_tables_to_passes(
        {1: df},
        {1: "metric_table"},
        layout,
        "MI300",
        frozenset({"Avg"}),
    )
    bound = df.at[0, "Avg"]
    assert "TCP_TOTAL_CACHE_ACCESSES_sum@pass:pmc_perf_0" in bound
    assert "raw_pmc_df['End_Timestamp@pass:pmc_perf_0']" in bound
    assert "raw_pmc_df['Start_Timestamp@pass:pmc_perf_0']" in bound


def test_eval_metric_rate_divides_by_the_selected_pass_duration() -> None:
    """TCP is only in pass 1 (duration 50); pass 0 lasted 10."""
    long_df = _two_pass_long_df(
        counters=["SQ_WAVES", "TCP_TOTAL_CACHE_ACCESSES_sum"],
        values=[4, 100],
        passes=["pmc_perf_0", "pmc_perf_1"],
        starts=[10, 100],
        ends=[20, 150],
    )
    expr = (
        "to_sum(raw_pmc_df['TCP_TOTAL_CACHE_ACCESSES_sum']) / "
        "to_sum(raw_pmc_df['End_Timestamp'] - raw_pmc_df['Start_Timestamp'])"
    )

    metric_df = _eval_one_metric(long_df, expr)

    # Pass-1 duration: 100 / 50 = 2. Pass-0 timestamps would give 100 / 10.
    assert float(metric_df.at["9.9.9", "Avg"]) == 2.0


def test_eval_metric_leaves_a_metric_unset_when_no_pass_holds_its_counters() -> None:
    long_df = _two_pass_long_df(
        counters=["TA_BUSY", "TD_BUSY"],
        values=[50, 25],
        passes=["pmc_perf_0", "pmc_perf_1"],
        starts=[10, 10],
        ends=[20, 20],
    )
    expr = "to_avg(100 * raw_pmc_df['TA_BUSY'] / raw_pmc_df['TD_BUSY'])"

    metric_df = _eval_one_metric(long_df, expr)

    assert metric_df.at["9.9.9", "Avg"] == ""


def test_bind_expression_dataframe_unsets_a_metric_that_spans_passes() -> None:
    layout = PassLayout(
        pass_keys=("pmc_perf_0", "pmc_perf_1"),
        counters_by_pass={
            "pmc_perf_0": frozenset({"TA_BUSY"}),
            "pmc_perf_1": frozenset({"TD_BUSY"}),
        },
        duplicated=frozenset(),
    )
    expression_df = pd.DataFrame({
        "metric_id": ["9.9.9"],
        "value_name": ["Avg"],
        "value": ["to_avg(100 * raw_pmc_df['TA_BUSY'] / raw_pmc_df['TD_BUSY'])"],
    })

    bind_expression_dataframe(expression_df, layout, "MI300")

    assert expression_df.at[0, "value"] == "None"


def test_pass_scoped_builtins_evaluate_per_xcd_first_in_pass_order() -> None:
    layout = PassLayout(
        pass_keys=("pmc_perf_0", "pmc_perf_1", "pmc_perf_10"),
        counters_by_pass={
            "pmc_perf_0": frozenset({"GRBM_GUI_ACTIVE"}),
            "pmc_perf_1": frozenset({"GRBM_GUI_ACTIVE", "SQ_BUSY_CU_CYCLES"}),
            "pmc_perf_10": frozenset({"GRBM_GUI_ACTIVE"}),
        },
        duplicated=frozenset({"GRBM_GUI_ACTIVE"}),
    )
    calls = []

    def record(variable_key, eval_string, pass_locals):
        calls.append((variable_key, sorted(pass_locals)))
        return len(calls)

    results = evaluate_pass_scoped_builtins(
        layout,
        {"pmc_perf_10", "pmc_perf_1"},
        "MI300",
        ["to_avg(ammolite__numActiveCUs__pass1)"],
        record,
        key_prefix="ammolite__",
    )

    assert calls == [
        ("GRBM_GUI_ACTIVE_PER_XCD", []),
        ("numActiveCUs", ["ammolite__GRBM_GUI_ACTIVE_PER_XCD__pass1"]),
        ("GRBM_GUI_ACTIVE_PER_XCD", []),
        ("numActiveCUs", ["ammolite__GRBM_GUI_ACTIVE_PER_XCD__pass2"]),
    ]
    assert results == {
        "ammolite__GRBM_GUI_ACTIVE_PER_XCD__pass1": 1,
        "ammolite__numActiveCUs__pass1": 2,
        "ammolite__GRBM_GUI_ACTIVE_PER_XCD__pass2": 3,
        "ammolite__numActiveCUs__pass2": 4,
    }


@pytest.mark.parametrize(
    ("module_name", "call_evaluator"),
    [
        (
            "utils.metrics.evaluation_pipeline",
            lambda module, layout: module._calc_pass_scoped_builtins(
                pd.DataFrame(), {}, "gfx942", layout, {"pmc_perf_0"}, ["x"]
            ),
        ),
        (
            "rocprof_compute_analyze.analysis_db",
            lambda module, layout: module.db_analysis.calc_pass_scoped_builtin_sys_info(
                pd.DataFrame(), {"gpu_arch": "gfx942"}, layout, {"pmc_perf_0"}, ["x"]
            ),
        ),
    ],
)
def test_cli_and_db_builtins_share_one_evaluation(
    monkeypatch, module_name, call_evaluator
) -> None:
    module = importlib.import_module(module_name)
    seen = []

    def record(layout, used_passes, gpu_series, expressions, evaluate, key_prefix):
        seen.append((used_passes, gpu_series, expressions))
        return {"marker": 1}

    monkeypatch.setattr(module, "evaluate_pass_scoped_builtins", record)

    assert call_evaluator(module, SPLIT_LAYOUT) == {"marker": 1}
    assert seen == [({"pmc_perf_0"}, "MI300", ["x"])]


def test_provenance_merge_pivots_each_dispatch_once(monkeypatch) -> None:
    rows = [
        ("pmc_perf_10", "GRBM_GUI_ACTIVE", 300),
        ("pmc_perf_2", "GRBM_GUI_ACTIVE", 200),
        ("pmc_perf_2", "TA_BUSY", 50),
        ("pmc_perf_1", "SQ_WAVES", 4),
        ("pmc_perf_1", "GRBM_GUI_ACTIVE", 100),
    ]
    long_df = pd.DataFrame([
        {**DISPATCH_FIELDS, "Pass_Key": key, "Counter_Name": name, "Counter_Value": v}
        for key, name, v in rows
    ])
    layout = build_pass_layout(long_df)
    pivots = []
    real_pivot = utils_analysis._pivot_counters_by_pass

    def count_pivot(group_df):
        pivots.append(len(group_df))
        return real_pivot(group_df)

    monkeypatch.setattr(utils_analysis, "_pivot_counters_by_pass", count_pivot)
    partial_dispatch = long_df[long_df["Pass_Key"] != "pmc_perf_10"].assign(
        Dispatch_ID=1
    )

    result, _slot_to_id = process_rocpd_csv(
        pd.concat([long_df, partial_dispatch], ignore_index=True),
        pass_layout=layout,
    )

    assert pivots == [5, 4]
    assert result["GRBM_GUI_ACTIVE"].tolist() == [100, 100]
    assert result["GRBM_GUI_ACTIVE@pass:pmc_perf_1"].tolist() == [100, 100]
    assert result["GRBM_GUI_ACTIVE@pass:pmc_perf_2"].tolist() == [200, 200]
    assert result.at[0, "GRBM_GUI_ACTIVE@pass:pmc_perf_10"] == 300
    assert pd.isna(result.at[1, "GRBM_GUI_ACTIVE@pass:pmc_perf_10"])
    assert result["TA_BUSY"].tolist() == [50, 50]
    assert "TA_BUSY@pass:pmc_perf_2" not in result.columns

# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for analysis_db.py static methods."""

import copy
import csv
import json
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest import mock
from unittest.mock import MagicMock, patch

import common
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from pc_sampling import per_kernel_isa_export, source_snapshot_analysis
from rocprof_compute_analyze.analysis_db import SourceFrameCollector, db_analysis
from utils import analysis_orm as orm
from utils import schema
from utils.metrics.noise_clamper import (
    clear_noise_clamp_warnings,
    get_noise_clamp_warnings,
)

ISA_WORKLOAD_NAME = "vector_copy"
ISA_WORKLOAD_SUB_NAME = "run"

VIEW_CSV_FILENAMES = frozenset({
    "kernel.csv",
    "kernel_metric.csv",
    "pc_sampling_summary.csv",
    "source_lines.csv",
    "workload_metric.csv",
})


def make_dual_issue_arch_config(metric_name: str, peak_col: str = "Peak"):
    """Build an arch_config with a metric_table carrying one VALU row."""
    metric_df = pd.DataFrame(
        {
            "Metric": [metric_name],
            "Value": ["unused_expression"],
            peak_col: ["unused_peak_expression"],
        },
        index=pd.Index(["1.1"], name="Metric_ID"),
    )
    arch_config = schema.ArchConfig()
    arch_config.dfs = {201: metric_df}
    arch_config.dfs_type = {201: "metric_table"}
    return arch_config


def store_instruction_lines(code_object_store):
    """Every instruction line a code object owns, across all its symbols."""
    return [
        line
        for symbol in code_object_store.kernel_symbols
        for line in symbol.instruction_lines
    ]


def kernel_instruction_lines(kernel):
    """Every instruction line attributed to a kernel, across all its symbols."""
    return [
        line for symbol in kernel.kernel_symbols for line in symbol.instruction_lines
    ]


def make_pc_sampling_dispatch(dispatch_id, kernel_id):
    """Build one PC-sampling kernel dispatch record."""
    return {
        "start_timestamp": 0,
        "end_timestamp": 0,
        "dispatch_info": {
            "dispatch_id": dispatch_id,
            "kernel_id": kernel_id,
            "agent_id": {"handle": 1},
        },
    }


def make_colliding_pc_sampling_tool_data(process_id: int, sample_count: int):
    """Build process-local sampling data that reuses shared display identities."""
    tool_data = make_pc_sampling_tool_data()
    shared_sample = tool_data["buffer_records"]["pc_sample_stochastic"][0]
    tool_data["metadata"]["pid"] = process_id
    tool_data["buffer_records"]["pc_sample_stochastic"] = [
        copy.deepcopy(shared_sample) for _ in range(sample_count)
    ]
    tool_data["buffer_records"]["kernel_dispatch"] = [make_pc_sampling_dispatch(0, 100)]
    tool_data["strings"] = {
        "pc_sample_instructions": ["v_mov"],
        "pc_sample_comments": ["/s/shared.cpp:7"],
    }
    tool_data["kernel_symbols"] = [tool_data["kernel_symbols"][0]]
    return tool_data


def make_source_frame_collector(workload, workload_path="/fake/workload"):
    """Build a collector for tests that call the insert paths directly."""
    return SourceFrameCollector(Path(workload_path), workload)


def make_pc_sampling_database_analyzer(
    tool_data_per_workload,
    filter_gpu_ids=(),
    filter_kernel_ids=(),
    filter_dispatch_ids=(),
    sys_info_row=None,
):
    """Build a database analyzer configured for sampling-only workloads."""
    analyzer = db_analysis(
        SimpleNamespace(output_name=None, output_format="database"),
        {},
    )
    analyzer._runs = {
        workload_path: schema.Workload(
            sys_info=pd.DataFrame([
                (
                    dict(sys_info_row)
                    if sys_info_row is not None
                    else {"gpu_arch": "gfx942"}
                )
            ]),
            filter_gpu_ids=list(filter_gpu_ids),
            filter_kernel_ids=list(filter_kernel_ids),
            filter_dispatch_ids=list(filter_dispatch_ids),
        )
        for workload_path in tool_data_per_workload
    }
    analyzer._roofline_ceilings_per_workload = {}
    analyzer._profiling_config = {"filter_blocks": ["pc_sampling"]}
    analyzer._pc_sampling_tool_data_per_workload = tool_data_per_workload
    analyzer._dispatch_data_per_workload = {
        workload_path: analyzer._build_pc_sampling_dispatch_data(
            tool_data_records, analyzer._runs[workload_path]
        )
        for workload_path, tool_data_records in tool_data_per_workload.items()
    }
    analyzer._roofline_data_per_kernel = {}
    analyzer._roofline_data_per_workload = {}
    return analyzer


def make_pc_sampling_only_database_analyzer(
    workload_path,
    tool_data_records,
    filter_kernel_ids=(),
    filter_dispatch_ids=(),
):
    """Build a database analyzer configured for one sampling-only workload."""
    return make_pc_sampling_database_analyzer(
        {workload_path: tool_data_records},
        filter_kernel_ids=filter_kernel_ids,
        filter_dispatch_ids=filter_dispatch_ids,
    )


def make_counter_backed_database_analyzer(
    workload_path,
    filter_blocks,
    tool_data_records,
):
    """Build a counter-backed analyzer with optional sampling records."""
    analyzer = db_analysis(
        SimpleNamespace(output_name=None, output_format="database"),
        {},
    )
    analyzer._runs = {
        workload_path: schema.Workload(
            sys_info=pd.DataFrame([{"gpu_arch": "gfx942"}]),
        )
    }
    analyzer._roofline_ceilings_per_workload = {}
    analyzer._profiling_config = {"filter_blocks": filter_blocks}
    analyzer._pc_sampling_tool_data_per_workload = {workload_path: tool_data_records}
    analyzer._dispatch_data_per_workload = {
        workload_path: pd.DataFrame([
            {
                "dispatch_id": 7,
                "kernel_name": "vecCopy",
                "gpu_id": 0,
                "start_timestamp": 10,
                "end_timestamp": 20,
            }
        ])
    }
    analyzer._roofline_data_per_kernel = {workload_path: pd.DataFrame()}
    analyzer._roofline_data_per_workload = {}
    analyzer._metrics_info_data_per_workload = {}
    analyzer._kernel_values_data_per_workload = {}
    analyzer._workload_values_data_per_workload = {}
    return analyzer


def run_analysis_with_existing_database(analyzer):
    """Run analysis while preserving the test's existing database session."""
    with ExitStack() as patch_stack:
        patch_stack.enter_context(patch.object(orm.Database, "init"))
        patch_stack.enter_context(patch.object(orm.Database, "create_views"))
        patch_stack.enter_context(patch.object(orm.Database, "write"))
        patch_stack.enter_context(
            patch(
                "rocprof_compute_analyze.analysis_db.get_version",
                return_value={"version": "test", "sha": "test"},
            )
        )
        analyzer.run_analysis()


def run_analysis_with_materialized_views(analyzer):
    """Run analysis while materializing views in the existing test database."""
    with ExitStack() as patch_stack:
        patch_stack.enter_context(patch.object(orm.Database, "init"))
        patch_stack.enter_context(patch.object(orm.Database, "write"))
        patch_stack.enter_context(patch.object(analyzer, "run_analysis_metrics"))
        patch_stack.enter_context(
            patch(
                "rocprof_compute_analyze.analysis_db.get_version",
                return_value={"version": "test", "sha": "test"},
            )
        )
        analyzer.run_analysis()


def cleanup_database() -> None:
    """Close and clear the process-wide test database state."""
    current_session = orm.Database._session
    if current_session is not None:
        current_session.close()
    current_engine = orm.Database._engine
    if current_engine is not None:
        current_engine.dispose()
    orm.Database._session = None
    orm.Database._engine = None


# =============================================================================
# db_analysis.evaluate() tests
# =============================================================================


def test_evaluate_parse_false_basic_expressions():
    """Test parse=False mode with basic expressions and substitutions."""
    pmc_df = pd.DataFrame({
        "Counter1": [10, 20, 30],
        "Counter2": [1, 2, 3],
    })
    sys_info = {"numCUs": 64, "clock_speed": 1500}

    # Test raw_pmc_df -> pmc_df substitution on flat single-index columns
    result = db_analysis.evaluate(
        "test_metric",
        "raw_pmc_df['Counter1']",
        pmc_df,
        sys_info,
        parse=False,
    )
    assert isinstance(result, pd.Series)
    assert list(result) == [10, 20, 30]

    # Test ammolite__ substitution for sys_info access
    result = db_analysis.evaluate(
        "test_metric",
        "ammolite__numCUs * 2",
        pmc_df,
        sys_info,
        parse=False,
    )
    assert result == 128

    # Test expression with helper function
    result = db_analysis.evaluate(
        "test_metric",
        "to_sum(raw_pmc_df['Counter1'])",
        pmc_df,
        sys_info,
        parse=False,
    )
    assert result == 60


def test_evaluate_parse_true_basic_expressions():
    """Test parse=True mode with $ substitution and AST transformation."""
    pmc_df = pd.DataFrame({
        "Counter1": [10, 20, 30],
        "Counter2": [2, 4, 6],
    })
    sys_info = {"numCUs": 64, "multiplier": 2}

    # Test $variable substitution
    result = db_analysis.evaluate(
        "test_metric",
        "$numCUs * $multiplier",
        pmc_df,
        sys_info,
        parse=True,
    )
    assert result == 128

    # Test AST transformation with SUPPORTED_CALL functions (SUM -> to_sum)
    # and bare identifiers (Counter1 -> raw_pmc_df["Counter1"])
    result = db_analysis.evaluate(
        "test_metric",
        "SUM(Counter1)",
        pmc_df,
        sys_info,
        parse=True,
    )
    assert result == 60

    # Test combined $ substitution and column access with AVG
    result = db_analysis.evaluate(
        "test_metric",
        "AVG(Counter1) + $numCUs",
        pmc_df,
        sys_info,
        parse=True,
    )
    assert result == 84  # avg(10,20,30)=20 + 64


def test_evaluate_none_and_na_handling():
    """Test evaluate() handling of None and NA values."""
    pmc_df = pd.DataFrame({"Counter1": [10, 20, 30]})
    sys_info = {}

    # Explicit None in expression result returns None without warning
    result = db_analysis.evaluate(
        "test_metric",
        "None",
        pmc_df,
        sys_info,
        parse=False,
    )
    assert result is None

    # Scalar NA values (NaN) return None
    pmc_df_nan = pd.DataFrame({"Counter1": [np.nan, np.nan, np.nan]})
    result = db_analysis.evaluate(
        "test_metric",
        "to_sum(raw_pmc_df['Counter1'])",
        pmc_df_nan,
        sys_info,
        parse=False,
    )
    assert result is None

    # Series with NA values are preserved (not converted to None)
    pmc_df_mixed = pd.DataFrame({"Counter1": [10, np.nan, 30]})
    result = db_analysis.evaluate(
        "test_metric",
        "raw_pmc_df['Counter1']",
        pmc_df_mixed,
        sys_info,
        parse=False,
    )
    assert isinstance(result, pd.Series)
    assert result.iloc[0] == 10
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == 30

    # Exceptions return None gracefully
    result = db_analysis.evaluate(
        "test_metric",
        "raw_pmc_df['NonExistent']",
        pmc_df,
        sys_info,
        parse=False,
    )
    assert result is None


def test_evaluate_with_none_in_formula_does_not_nullify_valid_result():
    """
    Test that expressions containing 'None' in formula string
    still return valid results when evaluation produces a value.

    This is a regression test for the bugfix where expressions like
    .where(..., None) were incorrectly returning None even when
    the actual result was valid.
    """
    pmc_df = pd.DataFrame({
        "Counter1": [10, 20, 30],
        "Counter2": [1, 0, 3],  # Has a zero for conditional
    })
    sys_info = {}

    # Expression with None as fallback in .where() - should return valid result
    # when condition is met for at least some values
    result = db_analysis.evaluate(
        "test_metric",
        "(raw_pmc_df['Counter1'] / "
        "raw_pmc_df['Counter2'].where("
        "raw_pmc_df['Counter2'] != 0, None))",
        pmc_df,
        sys_info,
        parse=False,
    )
    # Result should be a Series, not None
    assert result is not None
    assert isinstance(result, pd.Series)

    # Expression that literally has "None" string but evaluates to a number
    result = db_analysis.evaluate(
        "test_metric",
        "10 if True else None",
        pmc_df,
        sys_info,
        parse=False,
    )
    assert result == 10


def test_evaluate_divide_by_zero_silenced_and_logged_at_debug():
    """
    Divide-by-zero (x/0 -> inf, 0/0 -> NaN) emits a numpy RuntimeWarning
    that is captured and logged via console_debug. The "evaluated to N/A"
    console_warning must not fire when a RuntimeWarning was caught.
    """
    pmc_df = pd.DataFrame({"Counter1": [10, 20, 30]})
    sys_info = {}

    cases = [
        # x/0 yields scalar inf; evaluate() collapses to None
        "to_sum(raw_pmc_df['Counter1']) / 0",
        # 0/0 yields scalar NaN; evaluate() collapses to None
        "(to_sum(raw_pmc_df['Counter1']) * 0) / 0",
    ]

    for expr in cases:
        with patch(
            "rocprof_compute_analyze.analysis_db.console_warning"
        ) as mock_warning:
            with patch(
                "rocprof_compute_analyze.analysis_db.console_debug"
            ) as mock_debug:
                result = db_analysis.evaluate(
                    "test_metric",
                    expr,
                    pmc_df,
                    sys_info,
                    parse=False,
                )

        assert result is None, f"Expected None for '{expr}', got {result}"

        mock_warning.assert_not_called()
        debug_msgs = [str(call) for call in mock_debug.call_args_list]
        assert any("RuntimeWarning" in m for m in debug_msgs), (
            f"Expected RuntimeWarning in console_debug output for '{expr}', "
            f"got {debug_msgs}"
        )


# =============================================================================
# db_analysis.calc_builtin_vars() tests
# =============================================================================


def test_calc_builtin_vars_processes_per_xcd_first():
    """
    Test that PER_XCD variables are processed before non-PER_XCD variables,
    allowing non-PER_XCD vars to reference PER_XCD vars via $placeholder.
    """
    pmc_df = pd.DataFrame({
        "Counter1": [100, 200],
    })
    sys_info = {"base_value": 10, "gpu_arch": "gfx942"}

    # Mock BUILD_IN_VARS with dependency chain:
    # - PER_XCD_VAR: computed from base_value
    # - DERIVED_VAR: depends on PER_XCD_VAR via $PER_XCD_VAR
    mock_builtin_vars = {
        "PER_XCD_VAR": "$base_value * 2",  # Should be processed first -> 20
        "DERIVED_VAR": "$PER_XCD_VAR + 5",  # Depends on PER_XCD_VAR -> 25
    }

    with patch(
        "rocprof_compute_analyze.analysis_db.mi_gpu_specs.get_gpu_series",
        return_value="MI300",
    ):
        with patch(
            "rocprof_compute_analyze.analysis_db.get_build_in_vars",
            return_value=mock_builtin_vars,
        ):
            with patch(
                "utils.utils_counter_defs.get_build_in_vars",
                return_value=mock_builtin_vars,
            ):
                db_analysis.calc_builtin_vars(
                    pmc_df, sys_info, ["$PER_XCD_VAR", "$DERIVED_VAR"]
                )

    # Verify PER_XCD var was computed
    assert sys_info["PER_XCD_VAR"] == 20

    # Verify DERIVED_VAR used the computed PER_XCD_VAR value
    assert sys_info["DERIVED_VAR"] == 25


def test_calc_builtin_vars_with_dataframe_expressions():
    """Test builtin vars that operate on DataFrame columns."""
    pmc_df = pd.DataFrame({
        "Counter1": [10, 20, 30],
    })
    sys_info = {"multiplier": 2, "gpu_arch": "gfx942"}

    # Use SUPPORTED_CALL function names (SUM -> to_sum via CodeTransformer)
    mock_builtin_vars = {
        "TOTAL_COUNT": "SUM(Counter1)",  # 60
        "SCALED_TOTAL": "$TOTAL_COUNT * $multiplier",  # 120
    }

    with patch(
        "rocprof_compute_analyze.analysis_db.mi_gpu_specs.get_gpu_series",
        return_value="MI300",
    ):
        with patch(
            "rocprof_compute_analyze.analysis_db.get_build_in_vars",
            return_value=mock_builtin_vars,
        ):
            with patch(
                "utils.utils_counter_defs.get_build_in_vars",
                return_value=mock_builtin_vars,
            ):
                db_analysis.calc_builtin_vars(
                    pmc_df, sys_info, ["$TOTAL_COUNT", "$SCALED_TOTAL"]
                )

    assert sys_info["TOTAL_COUNT"] == 60
    assert sys_info["SCALED_TOTAL"] == 120


# =============================================================================
# db_analysis.calc_dataframe_expressions() tests
# =============================================================================


def test_calc_dataframe_expressions_applies_evaluate_to_rows():
    """Test that expressions are evaluated for each row of expression_df."""
    pmc_df = pd.DataFrame({
        "Counter1": [10, 20, 30],
        "Counter2": [1, 2, 3],
    })
    sys_info = {"scale": 100, "gpu_arch": "gfx942"}

    expression_df = pd.DataFrame({
        "metric_id": ["1.1", "1.2"],
        "value_name": ["sum", "scaled"],
        "value": [
            "to_sum(raw_pmc_df['Counter1'])",
            "ammolite__scale * 2",
        ],
    })

    with patch(
        "rocprof_compute_analyze.analysis_db.get_build_in_vars", return_value={}
    ):
        result = db_analysis.calc_dataframe_expressions(pmc_df, sys_info, expression_df)

    assert isinstance(result, pd.Series)
    assert len(result) == 2
    assert result.iloc[0] == 60  # sum of Counter1
    assert result.iloc[1] == 200  # 100 * 2


def test_calc_dataframe_expressions_with_builtin_vars():
    """Test that calc_dataframe_expressions calls calc_builtin_vars first."""
    pmc_df = pd.DataFrame({"Counter1": [10, 20, 30]})
    sys_info = {"base": 5, "gpu_arch": "gfx942"}

    # Expression references a builtin var that gets computed
    mock_builtin_vars = {
        "COMPUTED_VAR": "$base * 10",  # 50
    }

    expression_df = pd.DataFrame({
        "metric_id": ["1.1", "1.2"],
        "value_name": ["test", "none_result"],
        "value": [
            "ammolite__COMPUTED_VAR + 1",  # Should be 51
            "None",
        ],
    })

    with patch(
        "rocprof_compute_analyze.analysis_db.mi_gpu_specs.get_gpu_series",
        return_value="MI300",
    ):
        with patch(
            "rocprof_compute_analyze.analysis_db.get_build_in_vars",
            return_value=mock_builtin_vars,
        ):
            with patch(
                "utils.utils_counter_defs.get_build_in_vars",
                return_value=mock_builtin_vars,
            ):
                result = db_analysis.calc_dataframe_expressions(
                    pmc_df, sys_info, expression_df
                )

    assert result.iloc[0] == 51
    # None from evaluate becomes NaN in pandas Series
    assert pd.isna(result.iloc[1])


def test_calc_dataframe_expressions_empty_returns_assignable_series():
    """An empty expression_df returns an empty Series, not a DataFrame."""
    expression_df = pd.DataFrame(columns=["metric_id", "value_name", "value"])

    result = db_analysis.calc_dataframe_expressions(
        pd.DataFrame({"Counter1": [1, 2, 3]}),
        {"gpu_arch": "gfx942"},
        expression_df,
    )

    assert isinstance(result, pd.Series)
    assert result.empty
    # Reproduces the call site: assigning the result as a single column must
    # not raise "Columns must be same length as key".
    expression_df["value"] = result


# =============================================================================
# calc_metrics_data tests
# =============================================================================


def test_calc_metrics_data_builds_rows_and_preserves_schema():
    """Metric tables expand into rows with table-level fields resolved once;
    non-metric tables are skipped and the output frames keep their columns."""
    workload_path = "/fake/workload"
    metric_df = pd.DataFrame(
        {
            "Metric": ["Grid Size"],
            "Avg": [" 10 "],
            "Min": [" 5 "],
            "Max": [" 20 "],
            "Unit": ["Work items"],
            "Description": ["Grid size desc"],
        },
        index=pd.Index(["7.1.0"], name="Metric_ID"),
    )
    arch_config = schema.ArchConfig()
    # Table 1 has no Metric column and is skipped; table 701 maps to
    # panel 700 (table_name) and sub-table 701 (sub_table_name).
    arch_config.dfs = {
        1: pd.DataFrame({"from_csv": ["pmc_kernel_top.csv"]}),
        701: metric_df,
    }
    arch_config.panel_configs = {
        700: {
            "id": 700,
            "title": "Wavefront",
            "data source": [
                {"metric_table": {"id": 701, "title": "Wavefront Launch Stats"}}
            ],
        }
    }

    analyzer = db_analysis(MagicMock(), {})
    analyzer._pmc_df_per_workload = {workload_path: pd.DataFrame({"Counter1": [1]})}
    analyzer._runs = {
        workload_path: MagicMock(sys_info=pd.DataFrame([{"gpu_arch": "gfx942"}]))
    }
    analyzer._arch_configs = {"gfx942": arch_config}

    metrics_info, expressions = analyzer.calc_metrics_data()

    info = metrics_info[workload_path]
    assert "pct_of_peak" in info.columns
    assert list(info["metric_id"]) == ["7.1.0"]
    assert list(info["name"]) == ["Grid Size"]
    assert list(info["table_name"]) == ["Wavefront"]
    assert list(info["sub_table_name"]) == ["Wavefront Launch Stats"]
    assert not bool(info["pct_of_peak"].iloc[0])

    exprs = expressions[workload_path]
    assert list(exprs.columns) == ["metric_id", "value_name", "value"]
    # Metric/Unit/Description are non-expression columns, so only Avg/Min/Max
    # expand into expression rows, stripped and in dataframe-column order.
    assert list(exprs["value_name"]) == ["Avg", "Min", "Max"]
    assert list(exprs["value"]) == ["10", "5", "20"]
    assert set(exprs["metric_id"]) == {"7.1.0"}


LONG_FORM_PMC_PERF = (
    "GPU_ID,Dispatch_ID,Grid_Size,Workgroup_Size,LDS_Per_Workgroup,"
    "Scratch_Per_Workitem,Arch_VGPR,Accum_VGPR,SGPR,Kernel_Name,"
    "Start_Timestamp,End_Timestamp,Kernel_ID,Counter_Name,Counter_Value\n"
    "0,0,256,64,0,0,8,0,16,kernel_a,10,20,0,SQ_WAVES,4\n"
)


def test_calc_pmc_df_data_reads_the_compressed_merge(tmp_path):
    """The merged intermediate is discovered through the compression interface."""
    common.write_pmc_perf(tmp_path, LONG_FORM_PMC_PERF)

    analyzer = db_analysis(MagicMock(), {})
    analyzer._runs = {str(tmp_path): MagicMock()}
    analyzer._profiling_config = {}

    pmc_df_per_workload = analyzer.calc_pmc_df_data()

    assert pmc_df_per_workload[str(tmp_path)]["SQ_WAVES"].tolist() == [4]


def test_calc_pmc_df_data_skips_workload_without_a_merge(tmp_path):
    """A workload with no merged intermediate is skipped, not read as plain CSV."""
    (tmp_path / f"{schema.PMC_PERF_FILE_PREFIX}.csv").write_text(LONG_FORM_PMC_PERF)

    analyzer = db_analysis(MagicMock(), {})
    analyzer._runs = {str(tmp_path): MagicMock()}
    analyzer._profiling_config = {}

    assert analyzer.calc_pmc_df_data() == {}


def test_calc_metrics_data_exports_qualified_per_channel_names():
    """
    Panel 18's per-channel rows reach the database as "Channel N", not as a
    bare index.
    """
    workload_path = "/fake/workload"
    metric_df = pd.DataFrame(
        {
            "Metric": ["Channel 0", "Channel 1"],
            "Min": [" 33.29 ", " 33.33 "],
            "Max": [" 33.51 ", " 33.33 "],
        },
        index=pd.Index(["18.2.0", "18.2.1"], name="Metric_ID"),
    )
    arch_config = schema.ArchConfig()
    arch_config.dfs = {1802: metric_df}
    arch_config.panel_configs = {
        1800: {
            "id": 1800,
            "title": "L2 Cache (per Channel)",
            "data source": [
                {"metric_table": {"id": 1802, "title": "L2 Cache Hit Rate (Percent)"}}
            ],
        }
    }

    analyzer = db_analysis(MagicMock(), {})
    analyzer._pmc_df_per_workload = {workload_path: pd.DataFrame({"Counter1": [1]})}
    analyzer._runs = {
        workload_path: MagicMock(sys_info=pd.DataFrame([{"gpu_arch": "gfx942"}]))
    }
    analyzer._arch_configs = {"gfx942": arch_config}

    metrics_info, expressions = analyzer.calc_metrics_data()

    info = metrics_info[workload_path]
    assert list(info["name"]) == ["Channel 0", "Channel 1"]
    assert list(info["table_name"]) == ["L2 Cache (per Channel)"] * 2
    assert list(info["sub_table_name"]) == ["L2 Cache Hit Rate (Percent)"] * 2

    # The label column is non-expression, so only Min/Max become expressions.
    exprs = expressions[workload_path]
    assert set(exprs["value_name"]) == {"Min", "Max"}


# =============================================================================
# Noise-clamp warning + summary tests
# =============================================================================


def test_calc_expressions_noise_clamp():
    """Variance warnings fire only at workload level, summary once per workload.

    - evaluate(emit_variance_warnings=True) emits the per-metric warning when
      to_noise_clamp advances the global counter; the False kwarg stays silent.
    - calc_expressions emits exactly one variance warning per workload
      (kernel-level pass is silent) and calls print_noise_clamp_summary once.
    """
    workload_path = "/fake/workload"
    noise_clamp_expression = (
        "to_noise_clamp(to_min(raw_pmc_df['DIFF']), to_max(raw_pmc_df['REF']))"
    )
    # Two distinct kernels so groupby yields two kernel-level evaluate calls
    # in addition to one workload-level call. Without the kwarg gate the
    # unguarded code would emit three warnings; with the gate, exactly one.
    pmc_df = pd.DataFrame({
        "Kernel_Name": ["kernel_a", "kernel_b"],
        "DIFF": [-100.0, -100.0],
        "REF": [1000.0, 1000.0],
    })
    expression_template = pd.DataFrame({
        "metric_id": ["1.1"],
        "value_name": ["clamped"],
        "value": [noise_clamp_expression],
    })
    sys_info_df = pd.DataFrame([{"placeholder": 1, "gpu_arch": "gfx942"}])

    analyzer = db_analysis(MagicMock(), {})
    analyzer._pmc_df_per_workload = {workload_path: pmc_df}
    analyzer._metric_expression_data_per_workload = {workload_path: expression_template}
    analyzer._metrics_info_data_per_workload = {}
    analyzer._roofline_ceilings_per_workload = {workload_path: {}}
    analyzer._runs = {workload_path: MagicMock(sys_info=sys_info_df)}
    analyzer._arch_configs = MagicMock()

    # Direct evaluate kwarg behavior.
    clear_noise_clamp_warnings()
    with patch(
        "rocprof_compute_analyze.analysis_db.console_warning"
    ) as console_warning_mock:
        db_analysis.evaluate(
            "direct_test",
            noise_clamp_expression,
            pmc_df,
            {},
            emit_variance_warnings=True,
        )
        variance_warning_calls = [
            warning_call
            for warning_call in console_warning_mock.call_args_list
            if "Variance corrected for metric: direct_test" in warning_call.args[0]
        ]
        assert len(variance_warning_calls) == 1
        assert get_noise_clamp_warnings()["count"] >= 1

    clear_noise_clamp_warnings()
    with patch(
        "rocprof_compute_analyze.analysis_db.console_warning"
    ) as console_warning_mock:
        db_analysis.evaluate(
            "direct_test_off",
            noise_clamp_expression,
            pmc_df,
            {},
            emit_variance_warnings=False,
        )
        assert get_noise_clamp_warnings()["count"] >= 1
        variance_warning_calls = [
            warning_call
            for warning_call in console_warning_mock.call_args_list
            if "Variance corrected for metric:" in warning_call.args[0]
        ]
        assert variance_warning_calls == []

    # calc_expressions per-workload bracket.
    clear_noise_clamp_warnings()
    with patch(
        "rocprof_compute_analyze.analysis_db.get_build_in_vars", return_value={}
    ):
        with patch(
            "rocprof_compute_analyze.analysis_db.console_warning"
        ) as console_warning_mock:
            with patch(
                "rocprof_compute_analyze.analysis_db.print_noise_clamp_summary"
            ) as print_noise_clamp_summary_mock:
                with patch.object(db_analysis, "validate_dual_issue_metrics"):
                    analyzer.calc_expressions()

    variance_warning_calls = [
        warning_call
        for warning_call in console_warning_mock.call_args_list
        if "Variance corrected for metric:" in warning_call.args[0]
    ]
    assert len(variance_warning_calls) == 1
    assert "1.1 - clamped" in variance_warning_calls[0].args[0]
    print_noise_clamp_summary_mock.assert_called_once()
    assert get_noise_clamp_warnings()["count"] >= 1


# =============================================================================
# _derive_pct_of_peak_values tests
# =============================================================================


class TestDerivePctOfPeakValues:
    """Tests for db_analysis._derive_pct_of_peak_values."""

    def _make_values_df(
        self,
        metric_ids: list[str],
        value_names: list[str],
        values: list[float],
        kernel_names: Optional[list[str]] = None,
    ):
        """Build a long-format values DataFrame as produced by calc_expressions."""
        data = {
            "metric_id": metric_ids,
            "value_name": value_names,
            "value": values,
        }
        if kernel_names is not None:
            data["kernel_name"] = kernel_names
        return pd.DataFrame(data)

    def test_pct_of_peak_true_metric_appends_percent_of_peak_row(self):
        """A pct_of_peak-enabled metric produces one new Percent of Peak row."""
        values_df = self._make_values_df(
            metric_ids=["1.1", "1.1"],
            value_names=["Avg", "Peak"],
            values=[50.0, 200.0],
        )
        new_rows = db_analysis._derive_pct_of_peak_values({"1.1"}, values_df)
        assert len(new_rows) == 1
        assert new_rows[0]["value_name"] == "Percent of Peak"
        assert new_rows[0]["value"] == pytest.approx(25.0)

    def test_multi_kernel_produces_one_row_per_kernel(self):
        """Calling once per kernel produces one Percent of Peak row per kernel."""
        kernel_a_df = self._make_values_df(
            metric_ids=["1.1", "1.1"],
            value_names=["Avg", "Peak"],
            values=[100.0, 200.0],
            kernel_names=["kernel_a", "kernel_a"],
        )
        kernel_b_df = self._make_values_df(
            metric_ids=["1.1", "1.1"],
            value_names=["Avg", "Peak"],
            values=[60.0, 300.0],
            kernel_names=["kernel_b", "kernel_b"],
        )
        rows_a = db_analysis._derive_pct_of_peak_values({"1.1"}, kernel_a_df)
        rows_b = db_analysis._derive_pct_of_peak_values({"1.1"}, kernel_b_df)
        assert len(rows_a) == 1
        assert rows_a[0]["value"] == pytest.approx(50.0)  # 100/200*100
        assert len(rows_b) == 1
        assert rows_b[0]["value"] == pytest.approx(20.0)  # 60/300*100

    def test_pct_of_peak_false_metric_produces_no_pct_row(self):
        """A metric not in pct_of_peak_metric_ids produces no Percent of Peak row."""
        values_df = self._make_values_df(
            metric_ids=["1.1", "1.1"],
            value_names=["Avg", "Peak"],
            values=[50.0, 100.0],
        )
        new_rows = db_analysis._derive_pct_of_peak_values(set(), values_df)
        assert new_rows == []

    def test_incomplete_data_skips_metric(self):
        """A metric missing Peak or Avg/Value must be skipped gracefully."""
        incomplete_cases = [
            # Only "Avg" present -- no "Peak" row
            self._make_values_df(
                metric_ids=["1.1"], value_names=["Avg"], values=[50.0]
            ),
            # Only "Peak" present -- no "Avg" or "Value" row
            self._make_values_df(
                metric_ids=["1.1"], value_names=["Peak"], values=[100.0]
            ),
        ]
        for incomplete_values in incomplete_cases:
            new_rows = db_analysis._derive_pct_of_peak_values(
                {"1.1"}, incomplete_values
            )
            assert new_rows == []


# =============================================================================
# Dual-issue VALU validation tests
# =============================================================================


def test_validate_dual_issue_metrics_emits_warning_above_peak():
    """Long-format VALU Utilization above peak triggers the dual-issue warning."""
    arch_config = make_dual_issue_arch_config("VALU Utilization")
    workload_values_df = pd.DataFrame({
        "metric_id": ["1.1", "1.1"],
        "value_name": ["Value", "Peak"],
        "value": [150.0, 100.0],
    })
    pmc_df = pd.DataFrame({"GRBM_GUI_ACTIVE": [1000]})

    with patch("utils.metrics.common.console_warning") as console_warning_mock:
        db_analysis.validate_dual_issue_metrics(
            pmc_df,
            {"gpu_arch": "gfx942"},
            workload_values_df,
            arch_config,
        )

    console_warning_mock.assert_called_once()
    msg = console_warning_mock.call_args.args[0]
    assert "VALU Utilization can go up to 200%" in msg


def test_validate_dual_issue_metrics_silent_below_peak():
    """Below-peak VALU Utilization stays silent."""
    arch_config = make_dual_issue_arch_config("VALU Utilization")
    workload_values_df = pd.DataFrame({
        "metric_id": ["1.1", "1.1"],
        "value_name": ["Value", "Peak"],
        "value": [80.0, 100.0],
    })
    pmc_df = pd.DataFrame({"GRBM_GUI_ACTIVE": [1000]})

    with patch("utils.metrics.common.console_warning") as console_warning_mock:
        db_analysis.validate_dual_issue_metrics(
            pmc_df,
            {"gpu_arch": "gfx942"},
            workload_values_df,
            arch_config,
        )

    console_warning_mock.assert_not_called()


def test_validate_dual_issue_metrics_uses_peak_empirical_fallback():
    """Peak (Empirical) wins when present; falls back to Peak otherwise."""
    arch_config = make_dual_issue_arch_config(
        "VALU FLOPs (F64)", peak_col="Peak (Empirical)"
    )
    workload_values_df = pd.DataFrame({
        "metric_id": ["1.1", "1.1"],
        "value_name": ["Value", "Peak (Empirical)"],
        "value": [600.0, 400.0],
    })
    pmc_df = pd.DataFrame({"GRBM_GUI_ACTIVE": [1000]})

    with patch("utils.metrics.common.console_warning") as console_warning_mock:
        db_analysis.validate_dual_issue_metrics(
            pmc_df,
            {"gpu_arch": "gfx942"},
            workload_values_df,
            arch_config,
        )

    console_warning_mock.assert_called_once()
    msg = console_warning_mock.call_args.args[0]
    assert "VALU FLOPs can exceed the peak value" in msg


def test_validate_dual_issue_metrics_appends_valu2_suffix_on_gfx950():
    """gfx950 with non-zero SQ_ACTIVE_INST_VALU2 appends the confirmation."""
    arch_config = make_dual_issue_arch_config("VALU Utilization")
    workload_values_df = pd.DataFrame({
        "metric_id": ["1.1", "1.1"],
        "value_name": ["Value", "Peak"],
        "value": [150.0, 100.0],
    })
    pmc_df = pd.DataFrame({"SQ_ACTIVE_INST_VALU2": [1, 2, 3]})

    with patch("utils.metrics.common.console_warning") as console_warning_mock:
        db_analysis.validate_dual_issue_metrics(
            pmc_df,
            {"gpu_arch": "gfx950"},
            workload_values_df,
            arch_config,
        )

    msg = console_warning_mock.call_args.args[0]
    assert "Dual-issue activity detected via SQ_ACTIVE_INST_VALU2 counter" in msg


def test_validate_dual_issue_metrics_skips_non_metric_table_dfs():
    """dfs entries whose dfs_type is not metric_table are ignored."""
    arch_config = make_dual_issue_arch_config("VALU Utilization")
    arch_config.dfs_type = {201: "raw_csv_table"}
    workload_values_df = pd.DataFrame({
        "metric_id": ["1.1", "1.1"],
        "value_name": ["Value", "Peak"],
        "value": [150.0, 100.0],
    })
    pmc_df = pd.DataFrame({"GRBM_GUI_ACTIVE": [1000]})

    with patch("utils.metrics.common.console_warning") as console_warning_mock:
        db_analysis.validate_dual_issue_metrics(
            pmc_df,
            {"gpu_arch": "gfx942"},
            workload_values_df,
            arch_config,
        )

    console_warning_mock.assert_not_called()


# =============================================================================
# PC-sampling population
# =============================================================================


def make_pc_sampling_tool_data():
    """Two offsets under two kernels sharing one code object, with counts."""
    stall = "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_WAITCNT"
    inst_type = "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_VALU"
    return {
        "metadata": {"pid": 42},
        "buffer_records": {
            "pc_sample_host_trap": [],
            "pc_sample_stochastic": [
                {
                    "inst_index": 0,
                    "record": {
                        "pc": {"code_object_id": 5, "code_object_offset": 0x10},
                        "dispatch_id": 0,
                        "wave_issued": False,
                        "snapshot": {"stall_reason": stall},
                        "inst_type": inst_type,
                    },
                },
                {
                    "inst_index": 1,
                    "record": {
                        "pc": {"code_object_id": 5, "code_object_offset": 0x20},
                        "dispatch_id": 1,
                        "wave_issued": True,
                        "snapshot": {},
                        "inst_type": inst_type,
                    },
                },
            ],
            "kernel_dispatch": [
                make_pc_sampling_dispatch(0, 100),
                make_pc_sampling_dispatch(1, 101),
            ],
        },
        "strings": {
            "pc_sample_instructions": ["v_mov", "v_add"],
            "pc_sample_comments": ["/s/a.cpp:1", "/s/a.cpp:2"],
        },
        "kernel_symbols": [
            {
                "kernel_id": 100,
                "code_object_id": 5,
                "kernel_name": "_Z7vecCopyv.kd",
                "formatted_kernel_name": "vecCopy",
            },
            {
                "kernel_id": 101,
                "code_object_id": 5,
                "kernel_name": "vecAdd.kd",
                "formatted_kernel_name": "vecAdd",
            },
        ],
        "code_objects": [{"code_object_id": 5, "load_base": 0x1000}],
        "agents": [],
    }


def test_add_pc_sampling_data_no_tool_data_is_noop(db_session):
    """A workload without tool data inserts no rows."""
    workload = orm.Workload(name="w", sub_name="s")
    db_session.add(workload)
    analyzer = db_analysis(MagicMock(), {})
    analyzer._pc_sampling_tool_data_per_workload = {"/fake/workload": []}

    code_object_stores = analyzer.add_pc_sampling_data(
        "/fake/workload",
        workload,
        {},
        {},
        make_source_frame_collector(workload),
        sys_info={},
    )
    db_session.commit()

    assert code_object_stores == {}
    assert db_session.query(orm.CodeObjectStore).count() == 0
    assert db_session.query(orm.InstructionLine).count() == 0


def test_add_pc_sampling_data_populates_and_attributes_kernels(db_session):
    """Instruction lines use the workload kernels and transient store registry."""
    workload_path = "/fake/workload"
    workload = orm.Workload(name="w", sub_name="s")
    db_session.add(workload)
    kernel_objs = {
        "vecCopy": orm.Kernel(kernel_name="vecCopy", workload=workload),
        "vecAdd": orm.Kernel(kernel_name="vecAdd", workload=workload),
    }
    for kernel in kernel_objs.values():
        db_session.add(kernel)

    analyzer = db_analysis(MagicMock(), {})
    analyzer._pc_sampling_tool_data_per_workload = {
        workload_path: [make_pc_sampling_tool_data()]
    }
    code_object_stores = analyzer.add_pc_sampling_data(
        workload_path,
        workload,
        kernel_objs,
        {},
        make_source_frame_collector(workload),
        sys_info={},
    )
    db_session.commit()

    code_object = db_session.query(orm.CodeObjectStore).one()
    assert set(code_object_stores) == {(42, 5)}
    assert code_object_stores[(42, 5)] is code_object
    assert code_object.code_object_uuid is not None
    assert code_object.pid == 42
    assert code_object.load_base == 0x1000

    lines = db_session.query(orm.InstructionLine).all()
    kernel_by_offset = {
        line.code_object_offset: line.kernel_symbol.kernel.kernel_name for line in lines
    }
    assert kernel_by_offset == {0x10: "vecCopy", 0x20: "vecAdd"}

    # The stalled sample carries a stall-reason count; both carry an inst type.
    stalled = next(line for line in lines if line.code_object_offset == 0x10)
    assert stalled.pc_sample_state.stall_count == 1
    assert {
        r.stall_reason_lookup.text for r in stalled.pc_sample_state.stall_reasons
    } == {"WAITCNT"}


def test_add_pc_sampling_data_inserts_wave_measurements_from_sys_info(
    db_session,
):
    """Configured denominators populate percentages; absent ones stay null."""
    tool_data = make_pc_sampling_tool_data()
    samples = tool_data["buffer_records"]["pc_sample_stochastic"]
    samples[0]["record"].update({"exec_mask": 0b1111, "wave_cnt": 8})
    samples[1]["record"].update({"exec_mask": 0b11, "wave_cnt": 16})
    cases = [
        (
            "/fake/workload/missing-wave-denominators",
            {"gpu_arch": "gfx942"},
            {0x10: (None, None), 0x20: (None, None)},
        ),
        (
            "/fake/workload/configured-wave-denominators",
            {
                "gpu_arch": "gfx942",
                "wave_size": "64",
                "max_waves_per_cu": "32",
            },
            {0x10: (6.25, 25.0), 0x20: (3.125, 50.0)},
        ),
    ]

    for workload_path, sys_info_row, expected_by_offset in cases:
        workload = orm.Workload(name=workload_path, sub_name="sampling")
        db_session.add(workload)
        kernel_objs = {
            kernel_name: orm.Kernel(kernel_name=kernel_name, workload=workload)
            for kernel_name in ("vecCopy", "vecAdd")
        }
        db_session.add_all(kernel_objs.values())
        analyzer = make_pc_sampling_database_analyzer(
            {workload_path: [copy.deepcopy(tool_data)]},
            sys_info_row=sys_info_row,
        )

        code_object_stores = analyzer.add_pc_sampling_data(
            workload_path,
            workload,
            kernel_objs,
            {},
            make_source_frame_collector(workload, workload_path),
            sys_info=sys_info_row,
        )
        db_session.commit()

        sample_states_by_offset = {
            line.code_object_offset: line.pc_sample_state
            for code_object_store in code_object_stores.values()
            for line in store_instruction_lines(code_object_store)
        }
        actual_by_offset = {
            offset: (
                sample_state.active_thread_percent,
                sample_state.wave_occupancy_percent,
            )
            for offset, sample_state in sample_states_by_offset.items()
        }
        assert actual_by_offset == expected_by_offset


def test_add_pc_sampling_data_separates_shared_code_object_ids_across_pids(
    db_session,
):
    """Keep colliding code-object IDs isolated by process ID."""
    workload_path = "/fake/workload"
    workload = orm.Workload(name="w", sub_name="s")
    db_session.add(workload)
    kernel_objs = {
        kernel_name: orm.Kernel(
            kernel_name=kernel_name,
            workload=workload,
        )
        for kernel_name in ("vecCopy", "vecAdd")
    }
    for kernel in kernel_objs.values():
        db_session.add(kernel)

    first_tool_data = make_pc_sampling_tool_data()
    second_tool_data = copy.deepcopy(first_tool_data)
    second_tool_data["metadata"]["pid"] = 99
    second_tool_data["code_objects"][0]["load_base"] = 0x3000

    analyzer = db_analysis(MagicMock(), {})
    analyzer._pc_sampling_tool_data_per_workload = {
        workload_path: [first_tool_data, second_tool_data]
    }
    code_object_stores = analyzer.add_pc_sampling_data(
        workload_path,
        workload,
        kernel_objs,
        {},
        make_source_frame_collector(workload),
        sys_info={},
    )
    db_session.commit()

    assert set(code_object_stores) == {(42, 5), (99, 5)}
    first_store = code_object_stores[(42, 5)]
    second_store = code_object_stores[(99, 5)]
    assert first_store is not second_store
    assert first_store.code_object_uuid != second_store.code_object_uuid
    assert (first_store.pid, second_store.pid) == (42, 99)
    # Both stores attribute to the same workload kernels; the stores stay
    # distinct because code_object_id is process-local.
    expected_kernels = {kernel_objs["vecCopy"], kernel_objs["vecAdd"]}
    assert {
        line.kernel_symbol.kernel for line in store_instruction_lines(first_store)
    } == expected_kernels
    assert {
        line.kernel_symbol.kernel for line in store_instruction_lines(second_store)
    } == expected_kernels
    total_vec_copy_samples = sum(
        line.pc_sample_state.total_count
        for line in db_session.query(orm.InstructionLine).all()
        if line.kernel_symbol.kernel.kernel_name == "vecCopy"
    )
    assert total_vec_copy_samples == 2


def test_run_analysis_scopes_pc_sampling_uuids_by_process(db_session):
    """Assign distinct database identities to each process's sampling data."""
    workload_path = "/fake/workload"
    tool_data_records = [
        make_colliding_pc_sampling_tool_data(42, 1),
        make_colliding_pc_sampling_tool_data(99, 2),
    ]
    tool_data_records[0]["buffer_records"]["kernel_dispatch"].append(
        make_pc_sampling_dispatch(1, 100)
    )
    workload = schema.Workload(
        sys_info=pd.DataFrame([{"gpu_arch": "gfx942"}]),
    )
    analyzer = db_analysis(
        SimpleNamespace(output_name=None, output_format="database"),
        {},
    )
    analyzer._runs = {workload_path: workload}
    analyzer._roofline_ceilings_per_workload = {}
    analyzer._profiling_config = {"filter_blocks": ["pc_sampling"]}
    analyzer._pc_sampling_tool_data_per_workload = {workload_path: tool_data_records}
    analyzer._dispatch_data_per_workload = {
        workload_path: analyzer._build_pc_sampling_dispatch_data(
            tool_data_records, workload
        )
    }
    analyzer._roofline_data_per_kernel = {}
    analyzer._roofline_data_per_workload = {}

    with ExitStack() as patch_stack:
        patch_stack.enter_context(patch.object(orm.Database, "init"))
        patch_stack.enter_context(patch.object(orm.Database, "create_views"))
        patch_stack.enter_context(patch.object(orm.Database, "write"))
        patch_stack.enter_context(patch.object(analyzer, "run_analysis_metrics"))
        patch_stack.enter_context(
            patch(
                "rocprof_compute_analyze.analysis_db.get_version",
                return_value={"version": "test", "sha": "test"},
            )
        )
        analyzer.run_analysis()

    dispatches = (
        db_session
        .query(orm.Dispatch)
        .order_by(orm.Dispatch.kernel_uuid, orm.Dispatch.dispatch_id)
        .all()
    )
    code_object_stores = (
        db_session.query(orm.CodeObjectStore).order_by(orm.CodeObjectStore.pid).all()
    )
    assert [(store.pid, store.code_object_id) for store in code_object_stores] == [
        (42, 5),
        (99, 5),
    ]
    # A kernel is identified by name across the workload, so both processes
    # share one kernel row. Renumbering keeps their dispatch ids distinct.
    assert [
        (dispatch.dispatch_id, dispatch.kernel.kernel_name) for dispatch in dispatches
    ] == [
        (0, "vecCopy"),
        (1, "vecCopy"),
        (2, "vecCopy"),
    ]
    assert len({dispatch.kernel_uuid for dispatch in dispatches}) == 1
    assert len({dispatch.dispatch_uuid for dispatch in dispatches}) == 3

    assert len({store.code_object_uuid for store in code_object_stores}) == 2

    instruction_lines = db_session.query(orm.InstructionLine).all()
    assert len(instruction_lines) == 2
    assert len({line.instruction_uuid for line in instruction_lines}) == 2
    assert len({line.kernel_symbol.code_object_uuid for line in instruction_lines}) == 2
    assert {
        (line.code_object_offset, line.instruction) for line in instruction_lines
    } == {(0x10, "v_mov")}
    assert {
        (frame.source_line.source_file.file_path, frame.source_line.line_number)
        for line in instruction_lines
        for frame in line.source_lines
    } == {("/s/shared.cpp", 7)}

    instruction_by_process_id = {
        line.kernel_symbol.code_object_store.pid: line for line in instruction_lines
    }
    assert set(instruction_by_process_id) == {42, 99}
    assert {
        process_id: line.pc_sample_state.total_count
        for process_id, line in instruction_by_process_id.items()
    } == {42: 1, 99: 2}
    assert sum(line.pc_sample_state.total_count for line in instruction_lines) == 3

    # Each process's line resolves to the one shared kernel, which is reachable
    # from the kernel view because it owns dispatches.
    shared_kernel_uuid = dispatches[0].kernel_uuid
    for instruction_line in instruction_by_process_id.values():
        assert instruction_line.kernel_symbol.kernel_uuid == shared_kernel_uuid
        assert instruction_line.kernel_symbol.kernel.dispatches
        assert instruction_line.kernel_symbol.code_object_store.code_object_id == 5


def test_run_analysis_materialized_views_keep_pc_sampling_origins(
    db_session,
):
    """Preserve process-scoped code objects through materialized views."""
    workload_path = "/fake/workload"
    analyzer = make_pc_sampling_only_database_analyzer(
        workload_path,
        [
            make_colliding_pc_sampling_tool_data(42, 1),
            make_colliding_pc_sampling_tool_data(99, 2),
        ],
    )

    run_analysis_with_materialized_views(analyzer)

    code_object_stores = (
        db_session.query(orm.CodeObjectStore).order_by(orm.CodeObjectStore.pid).all()
    )
    assert [(store.pid, store.code_object_id) for store in code_object_stores] == [
        (42, 5),
        (99, 5),
    ]
    dispatches = db_session.query(orm.Dispatch).order_by(orm.Dispatch.dispatch_id).all()
    assert [
        (dispatch.dispatch_id, dispatch.kernel.kernel_name) for dispatch in dispatches
    ] == [
        (0, "vecCopy"),
        (1, "vecCopy"),
    ]
    assert len({dispatch.dispatch_uuid for dispatch in dispatches}) == 2
    assert len({dispatch.kernel_uuid for dispatch in dispatches}) == 1

    assert len(code_object_stores) == 2
    assert {store.code_object_id for store in code_object_stores} == {5}
    assert len({store.code_object_uuid for store in code_object_stores}) == 2

    instruction_lines = db_session.query(orm.InstructionLine).all()
    assert len(instruction_lines) == 2
    assert len({line.instruction_uuid for line in instruction_lines}) == 2
    assert len({line.kernel_symbol.code_object_uuid for line in instruction_lines}) == 2
    assert {line.code_object_offset for line in instruction_lines} == {0x10}
    assert {line.instruction for line in instruction_lines} == {"v_mov"}
    assert {
        (frame.source_line.source_file.file_path, frame.source_line.line_number)
        for line in instruction_lines
        for frame in line.source_lines
    } == {("/s/shared.cpp", 7)}

    instruction_by_process_id = {
        line.kernel_symbol.code_object_store.pid: line for line in instruction_lines
    }
    assert set(instruction_by_process_id) == {42, 99}
    assert {
        process_id: line.pc_sample_state.total_count
        for process_id, line in instruction_by_process_id.items()
    } == {42: 1, 99: 2}
    assert sum(line.pc_sample_state.total_count for line in instruction_lines) == 3

    shared_kernel_uuid = dispatches[0].kernel_uuid
    for instruction_line in instruction_by_process_id.values():
        assert instruction_line.kernel_symbol.kernel_uuid == shared_kernel_uuid
        assert (
            instruction_line.kernel_symbol.code_object_uuid
            == instruction_line.kernel_symbol.code_object_store.code_object_uuid
        )
        assert instruction_line.kernel_symbol.code_object_store.code_object_id == 5

    sample_states = db_session.query(orm.PCSampleState).all()
    stall_reasons = db_session.query(orm.PCSampleStallReason).all()
    instruction_samples = db_session.query(orm.InstructionSample).all()
    assert len(sample_states) == 2
    assert {state.total_count for state in sample_states} == {1, 2}
    assert len(stall_reasons) == 2
    assert {reason.count for reason in stall_reasons} == {1, 2}
    assert len(instruction_samples) == 2
    assert {sample.count for sample in instruction_samples} == {1, 2}

    orphan_queries = {
        "instruction_line_to_kernel_symbol": """
            SELECT COUNT(*)
            FROM compute_instruction_line AS instruction_line
            LEFT JOIN compute_kernel_symbol AS kernel_symbol
                ON instruction_line.kernel_symbol_uuid
                    = kernel_symbol.kernel_symbol_uuid
            WHERE kernel_symbol.kernel_symbol_uuid IS NULL
        """,
        "kernel_symbol_to_kernel": """
            SELECT COUNT(*)
            FROM compute_kernel_symbol AS kernel_symbol
            LEFT JOIN compute_kernel AS kernel
                ON kernel_symbol.kernel_uuid = kernel.kernel_uuid
            WHERE kernel.kernel_uuid IS NULL
        """,
        "kernel_symbol_to_code_object": """
            SELECT COUNT(*)
            FROM compute_kernel_symbol AS kernel_symbol
            LEFT JOIN compute_code_object_store AS code_object
                ON kernel_symbol.code_object_uuid = code_object.code_object_uuid
            WHERE code_object.code_object_uuid IS NULL
        """,
        "sample_state_to_instruction_line": """
            SELECT COUNT(*)
            FROM compute_pc_sample_state AS sample_state
            LEFT JOIN compute_instruction_line AS instruction_line
                ON sample_state.instruction_uuid = instruction_line.instruction_uuid
            WHERE instruction_line.instruction_uuid IS NULL
        """,
        "stall_reason_to_state_and_lookup": """
            SELECT COUNT(*)
            FROM compute_pc_sample_stall_reason AS stall_reason
            LEFT JOIN compute_pc_sample_state AS sample_state
                ON stall_reason.pc_sample_state_uuid =
                    sample_state.pc_sample_state_uuid
            LEFT JOIN compute_pc_sample_stall_reason_lookup AS stall_lookup
                ON stall_reason.pc_sample_stall_reason_lookup_uuid =
                    stall_lookup.pc_sample_stall_reason_lookup_uuid
            WHERE sample_state.pc_sample_state_uuid IS NULL
                OR stall_lookup.pc_sample_stall_reason_lookup_uuid IS NULL
        """,
        "instruction_sample_to_state_and_lookup": """
            SELECT COUNT(*)
            FROM compute_instruction_sample AS instruction_sample
            LEFT JOIN compute_pc_sample_state AS sample_state
                ON instruction_sample.pc_sample_state_uuid =
                    sample_state.pc_sample_state_uuid
            LEFT JOIN compute_instruction_sample_lookup AS sample_lookup
                ON instruction_sample.instruction_sample_lookup_uuid =
                    sample_lookup.instruction_sample_lookup_uuid
            WHERE sample_state.pc_sample_state_uuid IS NULL
                OR sample_lookup.instruction_sample_lookup_uuid IS NULL
        """,
    }
    orphan_counts = {
        relationship: db_session.execute(text(query)).scalar_one()
        for relationship, query in orphan_queries.items()
    }
    assert orphan_counts == dict.fromkeys(orphan_queries, 0)

    pc_sampling_rows = (
        db_session
        .execute(
            text(
                "SELECT pid, kernel_uuid, kernel_name, count "
                "FROM compute_pc_sampling_summary_view ORDER BY pid"
            )
        )
        .mappings()
        .all()
    )
    assert [row["kernel_name"] for row in pc_sampling_rows] == [
        "vecCopy",
        "vecCopy",
    ]
    # Both rows name the same kernel; pid is what keeps them apart.
    assert len({row["kernel_uuid"] for row in pc_sampling_rows}) == 1
    pc_sampling_mapping = {row["pid"]: row["count"] for row in pc_sampling_rows}
    expected_pc_sampling_mapping = {
        line.kernel_symbol.code_object_store.pid: line.pc_sample_state.total_count
        for line in instruction_lines
    }
    assert pc_sampling_mapping == expected_pc_sampling_mapping
    assert sum(pc_sampling_mapping.values()) == 3

    kernel_rows = (
        db_session
        .execute(
            text(
                "SELECT kernel_uuid, kernel_name, dispatch_count "
                "FROM compute_kernel_view ORDER BY kernel_uuid"
            )
        )
        .mappings()
        .all()
    )
    # The kernel view inner-joins dispatches, so the shared kernel appears once
    # and carries both processes' dispatches.
    assert [row["kernel_name"] for row in kernel_rows] == ["vecCopy"]
    assert kernel_rows[0]["dispatch_count"] == 2
    # Every sampling row resolves to a kernel the kernel view exposes.
    assert {row["kernel_uuid"] for row in pc_sampling_rows} == {
        row["kernel_uuid"] for row in kernel_rows
    }


def test_run_analysis_exports_process_scoped_pc_sampling_csv(
    tmp_path,
):
    """Export one PC sampling CSV row per process, resolvable in kernel.csv."""
    try:
        orm.Database.init(":memory:")
        database_session = orm.Database.get_session()
        assert database_session is not None
        analyzer = make_pc_sampling_only_database_analyzer(
            str(tmp_path),
            [
                make_colliding_pc_sampling_tool_data(42, 1),
                make_colliding_pc_sampling_tool_data(99, 2),
            ],
        )

        run_analysis_with_materialized_views(analyzer)

        expected_pc_sampling_mapping = {
            pid: count
            for pid, count in database_session.execute(
                text(
                    "SELECT pid, count "
                    "FROM compute_pc_sampling_summary_view ORDER BY pid"
                )
            )
        }
        expected_kernel_mapping = {
            kernel_uuid: dispatch_count
            for kernel_uuid, dispatch_count in database_session.execute(
                text(
                    "SELECT kernel_uuid, dispatch_count "
                    "FROM compute_kernel_view ORDER BY kernel_uuid"
                )
            )
        }

        csv_directory = tmp_path / "csv"
        orm.Database.write_csv_dir(csv_directory)

        output_filenames = {path.name for path in csv_directory.iterdir()}
        assert output_filenames == VIEW_CSV_FILENAMES
        assert "dispatch.csv" not in output_filenames

        pc_sampling_frame = pd.read_csv(csv_directory / "pc_sampling_summary.csv")
        kernel_frame = pd.read_csv(csv_directory / "kernel.csv")
        assert "pid" in pc_sampling_frame.columns
        assert "code_object_id" in pc_sampling_frame.columns
        assert "pid" not in kernel_frame.columns
        assert list(pc_sampling_frame["kernel_name"]) == ["vecCopy", "vecCopy"]
        assert list(kernel_frame["kernel_name"]) == ["vecCopy"]

        pc_sampling_csv_mapping = {
            row.pid: row.count for row in pc_sampling_frame.itertuples(index=False)
        }
        assert pc_sampling_csv_mapping == {42: 1, 99: 2}
        assert pc_sampling_csv_mapping == expected_pc_sampling_mapping

        kernel_csv_mapping = {
            row.kernel_uuid: row.dispatch_count
            for row in kernel_frame.itertuples(index=False)
        }
        assert kernel_csv_mapping == expected_kernel_mapping
        assert set(kernel_csv_mapping.values()) == {2}

        # T1 regression guard: every sampling row must resolve in kernel.csv.
        assert set(pc_sampling_frame["kernel_uuid"]) <= set(kernel_frame["kernel_uuid"])
    finally:
        cleanup_database()


def test_run_analysis_keeps_mixed_counter_and_pc_sampling_ownership(
    db_session,
    tmp_path,
):
    """Attribute every process's sampling lines to the dispatched kernel."""
    workload_path = str(tmp_path)
    first_tool_data = make_colliding_pc_sampling_tool_data(42, 1)
    second_tool_data = make_colliding_pc_sampling_tool_data(99, 2)
    second_tool_data["code_objects"][0]["load_base"] = 0x3000
    tool_data_records = [first_tool_data, second_tool_data]

    process_id_by_load_base = {0x1000: 42, 0x3000: 99}
    sample_count_by_process_id = {42: 1, 99: 2}
    for load_base, process_id in process_id_by_load_base.items():
        (tmp_path / f"{process_id}_code_obj_info.json").write_text(
            json.dumps({
                "code_objects": [
                    make_disasm_code_object(
                        5,
                        [
                            {
                                "virtual_address": load_base + 0x30,
                                "name": "s_nop",
                                "comment": "retained ISA",
                            }
                        ],
                        symbol_name="_Z7vecCopyv",
                    )
                ]
            }),
            encoding="utf-8",
        )

    analyzer = make_counter_backed_database_analyzer(
        workload_path,
        ["1", "pc_sampling"],
        tool_data_records,
    )
    analyzer._roofline_data_per_kernel = {
        workload_path: pd.DataFrame([{"kernel_name": "vecCopy", "total_flops": 64.0}])
    }
    analyzer._metrics_info_data_per_workload = {
        workload_path: pd.DataFrame([
            {
                "name": "Counter metric",
                "metric_id": "1.1",
                "description": "Counter-derived value",
                "unit": "cycles",
                "table_name": "Counter",
                "sub_table_name": "Counter values",
            }
        ])
    }
    analyzer._kernel_values_data_per_workload = {
        workload_path: pd.DataFrame([
            {
                "metric_id": "1.1",
                "kernel_name": "vecCopy",
                "value_name": "avg",
                "value": 12.5,
            }
        ])
    }

    run_analysis_with_existing_database(analyzer)

    dispatch = db_session.query(orm.Dispatch).one()
    aggregate_kernel = dispatch.kernel
    assert dispatch.dispatch_id == 7
    assert aggregate_kernel.kernel_name == "vecCopy"

    metric_value = db_session.query(orm.KernelMetricValue).one()
    roofline_data = db_session.query(orm.KernelRooflineData).one()
    assert metric_value.kernel_uuid == aggregate_kernel.kernel_uuid
    assert metric_value.value == 12.5
    assert roofline_data.kernel_uuid == aggregate_kernel.kernel_uuid
    assert roofline_data.total_flops == 64.0

    # One kernel row per name, so the sampling lines attribute to the same
    # kernel the counter dispatch created and stay reachable from kernel.csv.
    kernels = db_session.query(orm.Kernel).all()
    assert len(kernels) == 1
    assert kernels == [aggregate_kernel]

    stores = (
        db_session
        .query(orm.CodeObjectStore)
        .order_by(orm.CodeObjectStore.load_base)
        .all()
    )
    assert [(store.pid, store.code_object_id, store.load_base) for store in stores] == [
        (42, 5, 0x1000),
        (99, 5, 0x3000),
    ]

    chain_by_process_id = {}
    for store in stores:
        instruction_by_offset = {
            line.code_object_offset: line for line in store_instruction_lines(store)
        }
        assert set(instruction_by_offset) == {0x10, 0x30}

        sampled_line = instruction_by_offset[0x10]
        isa_line = instruction_by_offset[0x30]
        assert (
            sampled_line.kernel_symbol.kernel_uuid == isa_line.kernel_symbol.kernel_uuid
        )
        assert isa_line.pc_sample_state is None

        process_id = store.pid
        assert process_id_by_load_base[store.load_base] == process_id
        assert (
            sampled_line.pc_sample_state.total_count
            == sample_count_by_process_id[process_id]
        )
        chain_by_process_id[process_id] = (
            sampled_line.kernel_symbol.kernel_uuid,
            store.code_object_uuid,
            sampled_line.instruction_uuid,
            isa_line.instruction_uuid,
        )

    assert set(chain_by_process_id) == {42, 99}
    (
        kernel_uuid_chain,
        code_object_uuid_chain,
        sampled_instruction_uuid_chain,
        isa_instruction_uuid_chain,
    ) = zip(*chain_by_process_id.values())
    # Process identity lives on the code object, not the kernel.
    assert set(kernel_uuid_chain) == {aggregate_kernel.kernel_uuid}
    assert len(set(code_object_uuid_chain)) == 2
    assert len(set(sampled_instruction_uuid_chain)) == 2
    assert len(set(isa_instruction_uuid_chain)) == 2

    assert len(kernel_instruction_lines(aggregate_kernel)) == 4
    assert aggregate_kernel.dispatches
    assert aggregate_kernel.metric_values
    assert aggregate_kernel.roofline_data_points

    sample_states = db_session.query(orm.PCSampleState).all()
    assert len(sample_states) == 2
    assert all(
        state.instruction_line.kernel_symbol.kernel_uuid == aggregate_kernel.kernel_uuid
        for state in sample_states
    )
    assert db_session.query(orm.PCSampleStallReason).count() == 2
    assert db_session.query(orm.InstructionSample).count() == 2
    assert db_session.query(orm.InstructionLine).count() == 4


def test_run_analysis_does_not_register_filtered_pc_sampling_symbols(
    db_session,
    tmp_path,
):
    """Exclude sampled symbols whose kernels were filtered from dispatches."""
    workload_path = str(tmp_path)
    analyzer = make_counter_backed_database_analyzer(
        workload_path,
        ["1", "pc_sampling"],
        [make_pc_sampling_tool_data()],
    )

    run_analysis_with_existing_database(analyzer)

    dispatch = db_session.query(orm.Dispatch).one()

    kernels = db_session.query(orm.Kernel).all()
    assert len(kernels) == 1
    assert {kernel.kernel_name for kernel in kernels} == {"vecCopy"}

    instruction_lines = db_session.query(orm.InstructionLine).all()
    assert [line.code_object_offset for line in instruction_lines] == [0x10]

    # vecAdd was filtered out of the dispatches, so only vecCopy's line survives
    # and it attributes to the dispatched kernel.
    sampled_line = instruction_lines[0]
    assert sampled_line.kernel_symbol.code_object_store.pid == 42
    assert sampled_line.kernel_symbol.kernel_uuid == dispatch.kernel_uuid
    assert sampled_line.pc_sample_state.total_count == 1
    assert db_session.query(orm.PCSampleState).count() == 1
    assert db_session.query(orm.PCSampleStallReason).count() == 1
    assert db_session.query(orm.InstructionSample).count() == 1


# =============================================================================
# Code-object ISA ingestion (add_code_object_isa)
# =============================================================================


def make_disasm_code_object(code_object_id, instructions, symbol_name="sym"):
    """Build one code_obj_info code object with a single symbol."""
    return {
        "id": code_object_id,
        "symbols": [
            {
                "name": symbol_name,
                "virtual_address": instructions[0]["virtual_address"],
                "instructions": instructions,
            }
        ],
    }


def make_source_workload_tool_data_records(
    workload_path,
    snapshot_sources,
    sampled_sources,
):
    """Create PC-sampling inputs whose comments point at snapshot sources.

    A path named only in sampled_sources is absent from the snapshot.
    """
    workload_path.mkdir(parents=True, exist_ok=True)
    for original_source_path, content in snapshot_sources.items():
        snapshot_path = (
            workload_path / "src" / Path(original_source_path).relative_to("/")
        )
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(content, encoding="utf-8")

    tool_data = make_pc_sampling_tool_data()
    tool_data["strings"]["pc_sample_comments"] = list(sampled_sources)
    return [tool_data]


def fetch_source_lines_by_workload(session):
    """Group (file path, digest, line, content) tuples under each workload."""
    source_lines_by_workload = {}
    for source_file in session.query(orm.SourceFile).all():
        source_lines_by_workload.setdefault(source_file.workload.sub_name, []).extend(
            (
                source_file.file_path,
                source_file.md5_checksum,
                source_line.line_number,
                source_line.content,
            )
            for source_line in source_file.source_lines
        )
    return {
        workload_name: sorted(
            rows, key=lambda row: (row[0], row[2] is not None, row[2])
        )
        for workload_name, rows in source_lines_by_workload.items()
    }


def make_source_export_analyzer(tmp_path, output_format):
    """Build an analyzer and the source export its result folder should hold.

    Each workload holds one file its samples name and one it does not, so the
    export is pinned to the files the database records.
    """
    unreferenced_source_path = "/workspace/include/unreferenced.hpp"
    tool_data_per_workload = {}
    expected_exported_files = {}
    for workload_name, referenced_source_path, contents in (
        ("first", "/workspace/src/first.cpp", "int first;\n"),
        ("second", "/workspace/src/second.cpp", "int second;\n"),
    ):
        workload_path = tmp_path / "workloads" / workload_name / "run"
        tool_data_per_workload[str(workload_path)] = (
            make_source_workload_tool_data_records(
                workload_path,
                {
                    referenced_source_path: contents,
                    unreferenced_source_path: "int unreferenced;\n",
                },
                [f"{referenced_source_path}:1"],
            )
        )
        expected_exported_files[
            Path(workload_name)
            / "run"
            / "source"
            / Path(referenced_source_path).relative_to("/")
        ] = contents.encode()

    analyzer = make_pc_sampling_database_analyzer(tool_data_per_workload)
    result_path = tmp_path / f"{output_format}_analysis"
    analyzer.get_args().output_name = str(result_path)
    analyzer.get_args().output_format = output_format
    return analyzer, result_path, expected_exported_files


def run_source_export_analysis(analyzer):
    """Run source-export analysis and always clear the test database state."""
    try:
        with ExitStack() as patch_stack:
            patch_stack.enter_context(patch.object(analyzer, "run_analysis_metrics"))
            patch_stack.enter_context(
                patch(
                    "rocprof_compute_analyze.analysis_db.get_version",
                    return_value={"version": "test", "sha": "test"},
                )
            )
            analyzer.run_analysis()
    finally:
        cleanup_database()


def read_exported_source_files(result_path):
    """Return the exported source files under a CSV result folder."""
    exported_files = common.read_binary_file_tree(
        result_path / per_kernel_isa_export.PER_KERNEL_DIRECTORY_NAME
    )
    return {
        exported_path: contents
        for exported_path, contents in exported_files.items()
        if source_snapshot_analysis.SOURCE_EXPORT_DIRECTORY_NAME in exported_path.parts
    }


def read_view_csv_files(result_path):
    """Return view CSV contents keyed by filename."""
    return {
        csv_path.name: csv_path.read_bytes() for csv_path in result_path.glob("*.csv")
    }


def write_view_csv_files_and_capture(
    original_write_csv_dir,
    captured_view_csv_files,
    csv_directory,
):
    """Write view CSVs and capture their contents before source export."""
    original_write_csv_dir(csv_directory)
    captured_view_csv_files.update(read_view_csv_files(csv_directory))


def test_run_analysis_per_kernel_export_is_not_created_for_db_output(tmp_path):
    """Do not export per-kernel ISA or source snapshots for database output."""
    analyzer, result_path, _expected_exported_files = make_source_export_analyzer(
        tmp_path,
        "db",
    )

    with patch(
        "rocprof_compute_analyze.analysis_db.export_source_snapshot_files"
    ) as export_source_snapshot_files:
        run_source_export_analysis(analyzer)

    export_source_snapshot_files.assert_not_called()
    assert Path(f"{result_path}.db").is_file()
    assert not (result_path / per_kernel_isa_export.PER_KERNEL_DIRECTORY_NAME).exists()


def test_run_analysis_source_export_scopes_workloads_and_preserves_view_csvs(
    tmp_path,
):
    """Isolate workload exports without changing the generated view CSVs."""
    analyzer, result_path, expected_exported_files = make_source_export_analyzer(
        tmp_path,
        "csv",
    )
    view_csvs_before_source_export = {}
    original_write_csv_dir = orm.Database.write_csv_dir

    with ExitStack() as patch_stack:
        patch_stack.enter_context(
            patch.object(
                orm.Database,
                "write_csv_dir",
                side_effect=partial(
                    write_view_csv_files_and_capture,
                    original_write_csv_dir,
                    view_csvs_before_source_export,
                ),
            )
        )
        export_source_snapshot_files = patch_stack.enter_context(
            patch(
                "rocprof_compute_analyze.analysis_db.export_source_snapshot_files",
                wraps=source_snapshot_analysis.export_source_snapshot_files,
            )
        )
        run_source_export_analysis(analyzer)

    # The analyzer names each export folder, rather than the export rederiving
    # it, so a folder cannot drift from the workload row it belongs to.
    export_arguments = export_source_snapshot_files.call_args.kwargs
    assert (
        export_arguments["export_directory"] == result_path / "per_kernel_pc_sampling"
    )
    assert [
        (snapshot.workload_name, snapshot.workload_sub_name)
        for snapshot in export_arguments["workload_source_snapshots"]
    ] == [("first", "run"), ("second", "run")]

    view_csvs_after_analysis = read_view_csv_files(result_path)
    assert set(view_csvs_before_source_export) == VIEW_CSV_FILENAMES
    assert set(view_csvs_after_analysis) == VIEW_CSV_FILENAMES
    assert view_csvs_after_analysis == view_csvs_before_source_export

    exported_source_files = read_exported_source_files(result_path)
    assert exported_source_files == expected_exported_files

    # Every path the CSV records locates its copy by dropping the leading "/".
    source_lines_frame = pd.read_csv(result_path / "source_lines.csv")
    assert {
        Path(file_path).relative_to("/")
        for file_path in source_lines_frame["file_path"]
    } == {
        # The first three components name the workload and the source folder.
        Path(*exported_path.parts[3:])
        for exported_path in exported_source_files
    }


def test_run_analysis_keeps_each_workloads_source_files_apart(db_session, tmp_path):
    """Two workloads sharing a basename each keep their own absolute path."""
    tool_data_per_workload = {}
    for workload_name, source_path in (
        ("first", "/home/u/first/src/a.cpp"),
        ("second", "/opt/projects/second/src/a.cpp"),
    ):
        workload_path = tmp_path / workload_name
        tool_data_per_workload[str(workload_path)] = (
            make_source_workload_tool_data_records(
                workload_path,
                {source_path: "int first;\nint second;\n"},
                [f"{source_path}:1", f"{source_path}:2"],
            )
        )
    analyzer = make_pc_sampling_database_analyzer(tool_data_per_workload)

    run_analysis_with_materialized_views(analyzer)

    file_paths_by_workload = {
        workload_name: sorted({row[0] for row in rows})
        for workload_name, rows in fetch_source_lines_by_workload(db_session).items()
    }
    assert file_paths_by_workload == {
        "first": ["/home/u/first/src/a.cpp"],
        "second": ["/opt/projects/second/src/a.cpp"],
    }
    assert db_session.query(orm.SourceFile).count() == 2


def test_run_analysis_stores_lines_no_instruction_references(db_session, tmp_path):
    """A referenced file is stored whole, not just the lines a comment names."""
    workload_path = tmp_path / "workload"
    tool_data_records = make_source_workload_tool_data_records(
        workload_path,
        {"/home/u/app/vcopy.cpp": "int first;\nint second;\nint third;\n"},
        ["/home/u/app/vcopy.cpp:2", "/home/u/app/vcopy.cpp:2"],
    )
    analyzer = make_pc_sampling_database_analyzer({
        str(workload_path): tool_data_records
    })

    run_analysis_with_materialized_views(analyzer)

    digest = "86812da3721fe4b16ee110ad3dbcbbfa"
    assert fetch_source_lines_by_workload(db_session)["workload"] == [
        ("/home/u/app/vcopy.cpp", digest, 1, "int first;"),
        ("/home/u/app/vcopy.cpp", digest, 2, "int second;"),
        ("/home/u/app/vcopy.cpp", digest, 3, "int third;"),
    ]


def test_run_analysis_records_source_file_missing_from_snapshot(db_session, tmp_path):
    """A referenced file absent from the snapshot still gets its rows."""
    workload_path = tmp_path / "workload"
    tool_data_records = make_source_workload_tool_data_records(
        workload_path,
        {"/home/u/app/present.cpp": "int present;\n"},
        ["/home/u/app/present.cpp:1", "/home/u/app/absent.cpp:7"],
    )
    analyzer = make_pc_sampling_database_analyzer({
        str(workload_path): tool_data_records
    })

    run_analysis_with_materialized_views(analyzer)

    assert fetch_source_lines_by_workload(db_session)["workload"] == [
        ("/home/u/app/absent.cpp", None, 7, None),
        (
            "/home/u/app/present.cpp",
            "d4aeed8f3a65c2f28866bd51dff2ebfd",
            1,
            "int present;",
        ),
    ]


def test_run_analysis_records_line_past_end_of_source_file(db_session, tmp_path):
    """A frame naming a line the snapshot copy lacks gets a contentless row."""
    workload_path = tmp_path / "workload"
    tool_data_records = make_source_workload_tool_data_records(
        workload_path,
        {"/home/u/app/vcopy.cpp": "int only;\n"},
        ["/home/u/app/vcopy.cpp:1", "/home/u/app/vcopy.cpp:99"],
    )
    analyzer = make_pc_sampling_database_analyzer({
        str(workload_path): tool_data_records
    })

    run_analysis_with_materialized_views(analyzer)

    digest = "ac72447959bb8fac84d23eea9b103598"
    assert fetch_source_lines_by_workload(db_session)["workload"] == [
        ("/home/u/app/vcopy.cpp", digest, 1, "int only;"),
        ("/home/u/app/vcopy.cpp", digest, 99, None),
    ]


def test_run_analysis_links_frames_innermost_first(db_session, tmp_path):
    """An inline stack is stored in order and rebuilt by the summary view."""
    workload_path = tmp_path / "workload"
    tool_data_records = make_source_workload_tool_data_records(
        workload_path,
        {
            "/opt/rocm/hip.h": "line one\nline two\n",
            "/home/u/app/vcopy.cpp": "int a;\nint b;\n",
        },
        [
            "/opt/rocm/hip.h:? -> /opt/rocm/hip.h:2 -> /home/u/app/vcopy.cpp:1",
            "/home/u/app/vcopy.cpp:2",
        ],
    )
    analyzer = make_pc_sampling_database_analyzer({
        str(workload_path): tool_data_records
    })

    run_analysis_with_materialized_views(analyzer)

    chained_line = (
        db_session.query(orm.InstructionLine).filter_by(code_object_offset=0x10).one()
    )
    assert [
        (
            frame.frame_index,
            frame.source_line.source_file.file_path,
            frame.source_line.line_number,
        )
        for frame in chained_line.source_lines
    ] == [
        (0, "/opt/rocm/hip.h", None),
        (1, "/opt/rocm/hip.h", 2),
        (2, "/home/u/app/vcopy.cpp", 1),
    ]

    rebuilt_sources = {
        offset: source
        for offset, source in db_session.execute(
            text(
                "SELECT offset, source FROM compute_pc_sampling_summary_view "
                "ORDER BY offset"
            )
        )
    }
    assert rebuilt_sources == {
        0x10: ("/opt/rocm/hip.h:? -> /opt/rocm/hip.h:2 -> /home/u/app/vcopy.cpp:1"),
        0x20: "/home/u/app/vcopy.cpp:2",
    }


def test_add_code_object_isa_adds_unsampled_lines(db_session):
    """Un-sampled instructions of a dispatched kernel are added and attributed
    via the mangled-name join; a disassembly offset that matches an already-
    sampled offset inserts no duplicate row; ISA of an un-dispatched symbol is
    not stored at all; a non-surviving pid's disassembly is skipped."""
    workload_path = common.get_output_dir()
    Path(workload_path).mkdir(parents=True, exist_ok=True)
    load_base = 0x1000
    try:
        # 0x1010 -> offset 0x10 is already sampled; 0x1030 -> offset 0x30 is new
        # and joins vecCopy through its mangled ELF name; 0x1040 -> offset 0x40
        # belongs to a symbol never dispatched, so it is dropped entirely.
        code_objects = [
            {
                "id": 5,
                "symbols": [
                    {
                        "name": "_Z7vecCopyv",
                        "virtual_address": load_base + 0x10,
                        "instructions": [
                            {
                                "virtual_address": load_base + 0x10,
                                "name": "v_mov",
                                "comment": "",
                            },
                            {
                                "virtual_address": load_base + 0x30,
                                "name": "s_nop",
                                "comment": "c",
                            },
                        ],
                    },
                    {
                        "name": "_Z7unknownv",
                        "virtual_address": load_base + 0x40,
                        "instructions": [
                            {
                                "virtual_address": load_base + 0x40,
                                "name": "s_endpgm",
                                "comment": "",
                            }
                        ],
                    },
                ],
            }
        ]
        (Path(workload_path) / "42_code_obj_info.json").write_text(
            json.dumps({"code_objects": code_objects}), encoding="utf-8"
        )
        # A non-surviving pid (metadata pid is 42) reuses code_object_id 5; its
        # 0x50 offset must never be stored since only pid 42's load_base is known.
        (Path(workload_path) / "99_code_obj_info.json").write_text(
            json.dumps({
                "code_objects": [
                    make_disasm_code_object(
                        5,
                        [
                            {
                                "virtual_address": load_base + 0x50,
                                "name": "s_nop",
                                "comment": "",
                            }
                        ],
                        symbol_name="_Z7vecCopyv",
                    )
                ]
            }),
            encoding="utf-8",
        )

        workload = orm.Workload(name="w", sub_name="s")
        db_session.add(workload)
        kernel_objs = {
            "vecCopy": orm.Kernel(kernel_name="vecCopy", workload=workload),
            "vecAdd": orm.Kernel(kernel_name="vecAdd", workload=workload),
        }
        for kernel in kernel_objs.values():
            db_session.add(kernel)

        analyzer = db_analysis(MagicMock(), {})
        analyzer._pc_sampling_tool_data_per_workload = {
            workload_path: [make_pc_sampling_tool_data()]
        }
        kernel_symbols = {}
        source_frames = make_source_frame_collector(workload)
        code_object_stores = analyzer.add_pc_sampling_data(
            workload_path,
            workload,
            kernel_objs,
            kernel_symbols,
            source_frames,
            sys_info={},
        )
        analyzer.add_code_object_isa(
            workload_path,
            workload,
            kernel_objs,
            code_object_stores,
            kernel_symbols,
            source_frames,
        )
        db_session.commit()

        lines = db_session.query(orm.InstructionLine).all()
        by_offset = {line.code_object_offset: line for line in lines}
        # Two sampled offsets + one new un-sampled offset; the un-dispatched
        # symbol's 0x40 line and the non-surviving pid's 0x50 line are dropped,
        # and 0x10 is not duplicated.
        assert set(by_offset) == {0x10, 0x20, 0x30}
        assert set(code_object_stores) == {(42, 5)}
        code_object_store = code_object_stores[(42, 5)]
        assert code_object_store.pid == 42
        assert db_session.query(orm.CodeObjectStore).one() is code_object_store
        assert code_object_store is by_offset[0x10].kernel_symbol.code_object_store
        # The disassembly-only line joins its kernel and carries no sample state.
        isa_line = by_offset[0x30]
        assert isa_line.kernel_symbol.kernel.kernel_name == "vecCopy"
        assert isa_line.pc_sample_state is None
        assert isa_line.instruction == "s_nop"
        # The sampled line at 0x10 kept its kernel attribution and sample state.
        assert by_offset[0x10].kernel_symbol.kernel.kernel_name == "vecCopy"
        assert by_offset[0x10].pc_sample_state is not None
        # Both belong to the same (reused) code object store.
        assert isa_line.kernel_symbol.code_object_store is code_object_store
        assert {line.kernel_symbol.code_object_uuid for line in by_offset.values()} == {
            code_object_store.code_object_uuid
        }
        # The sampling and ISA passes resolved to one shared symbol, whose own
        # offset the ISA pass backfilled from the symbol's virtual address.
        vec_copy_symbol = by_offset[0x10].kernel_symbol
        assert isa_line.kernel_symbol is vec_copy_symbol
        assert vec_copy_symbol.code_object_offset == 0x10
        # vecAdd's sampled line at 0x20 gets its own symbol in the same object.
        assert by_offset[0x20].kernel_symbol is not vec_copy_symbol
        assert db_session.query(orm.KernelSymbol).count() == 2
    finally:
        common.clean_output_dir(True, workload_path)


def test_add_code_object_isa_scopes_unsampled_code_objects_by_process(db_session):
    """Matching ISA-only code objects retain distinct process-local UUID chains."""
    workload_path = common.get_output_dir()
    Path(workload_path).mkdir(parents=True, exist_ok=True)
    try:
        first_tool_data = make_pc_sampling_tool_data()
        first_tool_data["buffer_records"]["pc_sample_stochastic"] = []
        # code object 9 has a load_base and a dispatched kernel, but no samples.
        first_tool_data["code_objects"].append({
            "code_object_id": 9,
            "load_base": 0x2000,
        })
        first_tool_data["kernel_symbols"].append({
            "kernel_id": 102,
            "code_object_id": 9,
            "kernel_name": "_Z6helperv.kd",
            "formatted_kernel_name": "helper",
        })
        first_tool_data["buffer_records"]["kernel_dispatch"].append(
            make_pc_sampling_dispatch(2, 102)
        )
        second_tool_data = copy.deepcopy(first_tool_data)
        second_tool_data["metadata"]["pid"] = 99
        second_tool_data["code_objects"][-1]["load_base"] = 0x4000

        first_code_objects = [
            make_disasm_code_object(
                9,
                [{"virtual_address": 0x2000 + 0x8, "name": "s_endpgm", "comment": ""}],
                symbol_name="_Z6helperv",
            )
        ]
        second_code_objects = [
            make_disasm_code_object(
                9,
                [{"virtual_address": 0x4000 + 0x8, "name": "s_endpgm", "comment": ""}],
                symbol_name="_Z6helperv",
            )
        ]
        (Path(workload_path) / "42_code_obj_info.json").write_text(
            json.dumps({"code_objects": first_code_objects}), encoding="utf-8"
        )
        (Path(workload_path) / "99_code_obj_info.json").write_text(
            json.dumps({"code_objects": second_code_objects}), encoding="utf-8"
        )

        workload = orm.Workload(name="w", sub_name="s")
        db_session.add(workload)
        kernel_objs = {
            "helper": orm.Kernel(kernel_name="helper", workload=workload),
        }
        db_session.add_all(kernel_objs.values())

        analyzer = db_analysis(MagicMock(), {})
        analyzer._pc_sampling_tool_data_per_workload = {
            workload_path: [first_tool_data, second_tool_data]
        }
        kernel_symbols = {}
        source_frames = make_source_frame_collector(workload, workload_path)
        code_object_stores = analyzer.add_pc_sampling_data(
            workload_path,
            workload,
            kernel_objs,
            kernel_symbols,
            source_frames,
            sys_info={},
        )
        assert code_object_stores == {}
        analyzer.add_code_object_isa(
            workload_path,
            workload,
            kernel_objs,
            code_object_stores,
            kernel_symbols,
            source_frames,
        )
        db_session.commit()

        assert set(code_object_stores) == {(42, 9), (99, 9)}
        first_store = code_object_stores[(42, 9)]
        second_store = code_object_stores[(99, 9)]
        assert (first_store.pid, second_store.pid) == (42, 99)
        assert first_store.load_base == 0x2000
        assert second_store.load_base == 0x4000
        assert first_store.code_object_uuid != second_store.code_object_uuid

        first_line = store_instruction_lines(first_store)[0]
        second_line = store_instruction_lines(second_store)[0]
        assert first_line.code_object_offset == second_line.code_object_offset == 0x8
        assert first_line.instruction == second_line.instruction == "s_endpgm"
        assert first_line.instruction_uuid != second_line.instruction_uuid
        assert first_line.kernel_symbol.kernel is kernel_objs["helper"]
        assert second_line.kernel_symbol.kernel is kernel_objs["helper"]
    finally:
        common.clean_output_dir(True, workload_path)


def test_add_code_object_isa_skips_code_object_without_load_base(db_session):
    """A code object with no known load_base cannot be offset-mapped, so it is
    skipped rather than stored with an inconsistent offset."""
    workload_path = common.get_output_dir()
    Path(workload_path).mkdir(parents=True, exist_ok=True)
    try:
        tool_data = make_pc_sampling_tool_data()
        tool_data["code_objects"].append({"code_object_id": 9, "load_base": None})
        tool_data["kernel_symbols"].append({
            "kernel_id": 102,
            "code_object_id": 9,
            "kernel_name": "_Z6helperv.kd",
            "formatted_kernel_name": "helper",
        })
        tool_data["buffer_records"]["kernel_dispatch"].append(
            make_pc_sampling_dispatch(2, 102)
        )
        code_objects = [
            make_disasm_code_object(
                9,
                [{"virtual_address": 0x500, "name": "s_endpgm", "comment": ""}],
                symbol_name="_Z6helperv",
            )
        ]
        (Path(workload_path) / "42_code_obj_info.json").write_text(
            json.dumps({"code_objects": code_objects}), encoding="utf-8"
        )

        workload = orm.Workload(name="w", sub_name="s")
        db_session.add(workload)
        helper = orm.Kernel(kernel_name="helper", workload=workload)
        db_session.add(helper)
        kernel_objs = {"helper": helper}

        analyzer = db_analysis(MagicMock(), {})
        analyzer._pc_sampling_tool_data_per_workload = {workload_path: [tool_data]}
        kernel_symbols = {}
        source_frames = make_source_frame_collector(workload)
        code_object_stores = analyzer.add_pc_sampling_data(
            workload_path,
            workload,
            kernel_objs,
            kernel_symbols,
            source_frames,
            sys_info={},
        )
        analyzer.add_code_object_isa(
            workload_path,
            workload,
            kernel_objs,
            code_object_stores,
            kernel_symbols,
            source_frames,
        )
        db_session.commit()

        # The store exists (its kernel was dispatched) but no ISA line was added.
        store = db_session.query(orm.CodeObjectStore).filter_by(code_object_id=9).one()
        assert code_object_stores[(42, 9)] is store
        assert store.pid == 42
        assert store_instruction_lines(store) == []
    finally:
        common.clean_output_dir(True, workload_path)


def test_add_code_object_isa_scopes_duplicate_offsets_by_process(db_session):
    """One process's ISA offset cannot suppress the matching offset in another."""
    workload_path = common.get_output_dir()
    Path(workload_path).mkdir(parents=True, exist_ok=True)
    try:
        first_tool_data = make_pc_sampling_tool_data()
        second_tool_data = copy.deepcopy(first_tool_data)
        second_tool_data["metadata"]["pid"] = 99
        second_tool_data["code_objects"][0]["load_base"] = 0x3000

        (Path(workload_path) / "42_code_obj_info.json").write_text(
            json.dumps({
                "code_objects": [
                    make_disasm_code_object(
                        5,
                        [
                            {
                                "virtual_address": 0x1000 + 0x30,
                                "name": "s_shared",
                                "comment": "",
                            }
                        ],
                        symbol_name="_Z7vecCopyv",
                    )
                ]
            }),
            encoding="utf-8",
        )
        (Path(workload_path) / "99_code_obj_info.json").write_text(
            json.dumps({
                "code_objects": [
                    make_disasm_code_object(
                        5,
                        [
                            {
                                "virtual_address": 0x3000 + 0x30,
                                "name": "s_shared",
                                "comment": "",
                            }
                        ],
                        symbol_name="_Z7vecCopyv",
                    )
                ]
            }),
            encoding="utf-8",
        )

        workload = orm.Workload(name="w", sub_name="s")
        db_session.add(workload)
        kernel_objs = {
            kernel_name: orm.Kernel(
                kernel_name=kernel_name,
                workload=workload,
            )
            for kernel_name in ("vecCopy", "vecAdd")
        }
        for kernel in kernel_objs.values():
            db_session.add(kernel)

        analyzer = db_analysis(MagicMock(), {})
        analyzer._pc_sampling_tool_data_per_workload = {
            workload_path: [first_tool_data, second_tool_data]
        }
        kernel_symbols = {}
        source_frames = make_source_frame_collector(workload)
        code_object_stores = analyzer.add_pc_sampling_data(
            workload_path,
            workload,
            kernel_objs,
            kernel_symbols,
            source_frames,
            sys_info={},
        )
        analyzer.add_code_object_isa(
            workload_path,
            workload,
            kernel_objs,
            code_object_stores,
            kernel_symbols,
            source_frames,
        )
        db_session.commit()

        assert set(code_object_stores) == {(42, 5), (99, 5)}
        first_store = code_object_stores[(42, 5)]
        second_store = code_object_stores[(99, 5)]
        assert (first_store.pid, second_store.pid) == (42, 99)
        assert first_store.code_object_uuid != second_store.code_object_uuid

        first_line = next(
            line
            for line in store_instruction_lines(first_store)
            if line.code_object_offset == 0x30
        )
        second_line = next(
            line
            for line in store_instruction_lines(second_store)
            if line.code_object_offset == 0x30
        )
        assert first_line.instruction == second_line.instruction == "s_shared"
        assert first_line.instruction_uuid != second_line.instruction_uuid
        assert (
            first_line.kernel_symbol.code_object_uuid
            != second_line.kernel_symbol.code_object_uuid
        )
        assert first_line.kernel_symbol.kernel is kernel_objs["vecCopy"]
        assert second_line.kernel_symbol.kernel is kernel_objs["vecCopy"]
    finally:
        common.clean_output_dir(True, workload_path)


def test_add_code_object_isa_requires_process_local_dispatch(db_session):
    """Store only symbols dispatched by the process that loaded the code object."""
    workload_path = common.get_output_dir()
    Path(workload_path).mkdir(parents=True, exist_ok=True)
    try:
        first_tool_data = make_pc_sampling_tool_data()
        first_tool_data["buffer_records"]["kernel_dispatch"] = [
            make_pc_sampling_dispatch(0, 100)
        ]

        second_tool_data = copy.deepcopy(first_tool_data)
        second_tool_data["metadata"]["pid"] = 99
        second_tool_data["code_objects"][0]["load_base"] = 0x3000
        second_tool_data["buffer_records"]["kernel_dispatch"] = [
            make_pc_sampling_dispatch(0, 101)
        ]

        (Path(workload_path) / "99_code_obj_info.json").write_text(
            json.dumps({
                "code_objects": [
                    {
                        "id": 5,
                        "symbols": [
                            {
                                "name": "_Z7vecCopyv",
                                "virtual_address": 0x3000 + 0x30,
                                "instructions": [
                                    {
                                        "virtual_address": 0x3000 + 0x30,
                                        "name": "not_dispatched",
                                        "comment": "",
                                    }
                                ],
                            },
                            {
                                "name": "vecAdd",
                                "virtual_address": 0x3000 + 0x40,
                                "instructions": [
                                    {
                                        "virtual_address": 0x3000 + 0x40,
                                        "name": "dispatched",
                                        "comment": "",
                                    }
                                ],
                            },
                        ],
                    }
                ]
            }),
            encoding="utf-8",
        )

        workload = orm.Workload(name="w", sub_name="s")
        db_session.add(workload)
        kernel_objs = {
            "vecCopy": orm.Kernel(kernel_name="vecCopy", workload=workload),
            "vecAdd": orm.Kernel(kernel_name="vecAdd", workload=workload),
        }
        db_session.add_all(kernel_objs.values())

        analyzer = db_analysis(MagicMock(), {})
        analyzer._pc_sampling_tool_data_per_workload = {
            workload_path: [first_tool_data, second_tool_data]
        }
        code_object_stores = {}
        analyzer.add_code_object_isa(
            workload_path,
            workload,
            kernel_objs,
            code_object_stores,
            {},
            make_source_frame_collector(workload),
        )
        db_session.commit()

        assert set(code_object_stores) == {(99, 5)}
        second_store = code_object_stores[(99, 5)]
        assert second_store.pid == 99
        assert second_store.code_object_uuid is not None
        stored_isa = {
            (line.instruction, line.kernel_symbol.kernel.kernel_name)
            for line in store_instruction_lines(second_store)
        }
        assert stored_isa == {("dispatched", "vecAdd")}
    finally:
        common.clean_output_dir(True, workload_path)


def test_add_pc_sampling_data_drops_lines_without_kernel(db_session):
    """Lines whose kernel is absent from kernel_objs (filtered out) are dropped
    along with their sample state and child counts, not attributed to no kernel."""
    workload_path = "/fake/workload"
    workload = orm.Workload(name="w", sub_name="s")
    db_session.add(workload)
    # Only vecCopy survives filtering; vecAdd's line must be dropped.
    kernel_objs = {"vecCopy": orm.Kernel(kernel_name="vecCopy", workload=workload)}
    db_session.add(kernel_objs["vecCopy"])

    analyzer = db_analysis(MagicMock(), {})
    analyzer._pc_sampling_tool_data_per_workload = {
        workload_path: [make_pc_sampling_tool_data()]
    }
    analyzer.add_pc_sampling_data(
        workload_path,
        workload,
        kernel_objs,
        {},
        make_source_frame_collector(workload),
        sys_info={},
    )
    db_session.commit()

    lines = db_session.query(orm.InstructionLine).all()
    assert [line.code_object_offset for line in lines] == [0x10]
    retained_line = lines[0]
    assert retained_line.kernel_symbol.kernel is kernel_objs["vecCopy"]

    retained_sample_state = db_session.query(orm.PCSampleState).one()
    assert retained_line.pc_sample_state is retained_sample_state

    stall_reason_count = db_session.query(orm.PCSampleStallReason).one()
    assert stall_reason_count.pc_sample_state is retained_sample_state
    assert retained_sample_state.stall_reasons == [stall_reason_count]
    assert stall_reason_count.stall_reason_lookup.text == "WAITCNT"
    assert stall_reason_count.count == 1

    instruction_type_count = db_session.query(orm.InstructionSample).one()
    assert instruction_type_count.pc_sample_state is retained_sample_state
    assert retained_sample_state.instruction_samples == [instruction_type_count]
    assert instruction_type_count.instruction_sample_lookup.text == "VALU"
    assert instruction_type_count.count == 1


def test_run_analysis_keeps_sampled_row_without_a_comment(db_session, tmp_path):
    """An instruction with no comment keeps its row with a null source."""
    workload_path = tmp_path / "workload"
    tool_data_records = make_source_workload_tool_data_records(
        workload_path,
        {"/home/u/app/vcopy.cpp": "int a;\n"},
        # The sampling path turns the empty comment into the "N/A" sentinel.
        ["/home/u/app/vcopy.cpp:1", ""],
    )
    analyzer = make_pc_sampling_database_analyzer({
        str(workload_path): tool_data_records
    })

    run_analysis_with_materialized_views(analyzer)

    assert [
        source_file.file_path for source_file in db_session.query(orm.SourceFile)
    ] == ["/home/u/app/vcopy.cpp"]
    rebuilt_sources = {
        offset: source
        for offset, source in db_session.execute(
            text(
                "SELECT offset, source FROM compute_pc_sampling_summary_view "
                "ORDER BY offset"
            )
        )
    }
    assert rebuilt_sources == {0x10: "/home/u/app/vcopy.cpp:1", 0x20: None}


def make_csv_run_analyzer(tmp_path, tool_data_per_workload, **filters):
    """Build a CSV-output analyzer over sampling-only workloads."""
    analyzer = make_pc_sampling_database_analyzer(tool_data_per_workload, **filters)
    result_path = tmp_path / "csv_analysis"
    analyzer.get_args().output_name = str(result_path)
    analyzer.get_args().output_format = "csv"
    return analyzer, result_path


def read_per_kernel_isa_file(result_path, kernel_uuid, code_object_id=5, pid=42):
    """Return one exported ISA file as its header and its rows."""
    export_path = (
        result_path
        / per_kernel_isa_export.PER_KERNEL_DIRECTORY_NAME
        / ISA_WORKLOAD_NAME
        / ISA_WORKLOAD_SUB_NAME
        / f"kernel_{kernel_uuid}"
        / f"isa_code_object_id_{code_object_id}_pid_{pid}.csv"
    )
    with export_path.open(newline="", encoding="utf-8") as export_file:
        header, *rows = list(csv.reader(export_file))
    return header, rows


def stall_reason_columns(header):
    """Return the pivoted stall-reason columns of one ISA header."""
    return header[header.index("Active thread percent") + 1 : header.index("Source")]


def per_kernel_isa_paths(result_path):
    """Return the exported ISA files, relative to the per-kernel folder."""
    per_kernel_directory = result_path / per_kernel_isa_export.PER_KERNEL_DIRECTORY_NAME
    return sorted(
        str(export_path.relative_to(per_kernel_directory))
        for export_path in per_kernel_directory.rglob("isa_*.csv")
    )


def test_run_analysis_writes_one_isa_file_per_kernel_code_object_and_process(
    tmp_path,
):
    """A kernel compiled twice, and a code object in two processes, split up."""
    workload_path = tmp_path / "workloads" / ISA_WORKLOAD_NAME / ISA_WORKLOAD_SUB_NAME
    first_process_tool_data = make_pc_sampling_tool_data()
    second_process_tool_data = make_pc_sampling_tool_data()
    second_process_tool_data["metadata"]["pid"] = 43
    # The same kernel, sampled a second time out of another code object.
    other_code_object_tool_data = make_pc_sampling_tool_data()
    other_code_object_tool_data["metadata"]["pid"] = 44
    other_code_object_tool_data["code_objects"] = [
        {"code_object_id": 6, "load_base": 0x2000}
    ]
    for sample in other_code_object_tool_data["buffer_records"]["pc_sample_stochastic"]:
        sample["record"]["pc"]["code_object_id"] = 6
    for kernel_symbol in other_code_object_tool_data["kernel_symbols"]:
        kernel_symbol["code_object_id"] = 6

    analyzer, result_path = make_csv_run_analyzer(
        tmp_path,
        {
            str(workload_path): [
                first_process_tool_data,
                second_process_tool_data,
                other_code_object_tool_data,
            ]
        },
    )
    run_source_export_analysis(analyzer)

    kernel_frame = pd.read_csv(result_path / "kernel.csv")
    kernel_uuids = dict(
        zip(kernel_frame["kernel_name"], kernel_frame["kernel_uuid"], strict=True)
    )
    assert per_kernel_isa_paths(result_path) == sorted(
        f"vector_copy/run/kernel_{kernel_uuid}"
        f"/isa_code_object_id_{code_object_id}_pid_{pid}.csv"
        for kernel_uuid in kernel_uuids.values()
        for code_object_id, pid in ((5, 42), (5, 43), (6, 44))
    )


@pytest.mark.parametrize("filter_gpu_ids", [(), ["0"]])
def test_run_analysis_isa_file_carries_the_kernels_sampled_lines(
    tmp_path, filter_gpu_ids
):
    """One kernel's file holds its own offsets, counts and source.

    Filtering on the workload's own GPU keeps every row it already had.
    """
    workload_path = tmp_path / "workloads" / ISA_WORKLOAD_NAME / ISA_WORKLOAD_SUB_NAME
    analyzer, result_path = make_csv_run_analyzer(
        tmp_path,
        {str(workload_path): [make_pc_sampling_tool_data()]},
        filter_gpu_ids=filter_gpu_ids,
    )
    run_source_export_analysis(analyzer)

    kernel_frame = pd.read_csv(result_path / "kernel.csv")
    kernel_uuids = dict(
        zip(kernel_frame["kernel_name"], kernel_frame["kernel_uuid"], strict=True)
    )
    header, rows = read_per_kernel_isa_file(result_path, kernel_uuids["vecCopy"])

    assert header == [
        "Instruction line number",
        "Code object offset",
        "Instruction line",
        "Total count",
        "Active count",
        "Stall count",
        "Wave occupancy percent",
        "Active thread percent",
        "Stall WAITCNT",
        "Source",
        "Code object id",
        "Pid",
    ]
    assert rows == [
        [
            "1",
            str(0x10),
            "v_mov",
            "1",
            "0",
            "1",
            "",
            "",
            "1",
            "/s/a.cpp:1",
            "5",
            "42",
        ]
    ]


def test_run_analysis_isa_stall_columns_follow_the_workloads_reasons(tmp_path):
    """Each observed reason gets a column, empty where it was not seen."""
    workload_path = tmp_path / "workloads" / ISA_WORKLOAD_NAME / ISA_WORKLOAD_SUB_NAME
    tool_data = make_pc_sampling_tool_data()
    tool_data["buffer_records"]["pc_sample_stochastic"][1]["record"]["snapshot"] = {
        "stall_reason": (
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_SLEEP_WAIT"
        )
    }
    tool_data["buffer_records"]["pc_sample_stochastic"][1]["record"]["wave_issued"] = (
        False
    )
    # Both offsets belong to one kernel, so one file holds both reasons.
    tool_data["kernel_symbols"][1]["kernel_id"] = 100
    tool_data["buffer_records"]["kernel_dispatch"][1]["dispatch_info"]["kernel_id"] = (
        100
    )

    analyzer, result_path = make_csv_run_analyzer(
        tmp_path,
        {str(workload_path): [tool_data]},
    )
    run_source_export_analysis(analyzer)

    kernel_uuid = pd.read_csv(result_path / "kernel.csv")["kernel_uuid"].iloc[0]
    header, rows = read_per_kernel_isa_file(result_path, kernel_uuid)

    assert stall_reason_columns(header) == ["Stall SLEEP_WAIT", "Stall WAITCNT"]
    sleep_index, waitcnt_index = (
        header.index("Stall SLEEP_WAIT"),
        header.index("Stall WAITCNT"),
    )
    assert [(row[sleep_index], row[waitcnt_index]) for row in rows] == [
        ("", "1"),
        ("1", ""),
    ]


def test_run_analysis_isa_carries_no_stall_columns_for_host_trap(tmp_path):
    """host_trap records no stall reason, so its files carry no stall column."""
    workload_path = tmp_path / "workloads" / ISA_WORKLOAD_NAME / ISA_WORKLOAD_SUB_NAME
    tool_data = make_pc_sampling_tool_data()
    tool_data["buffer_records"]["pc_sample_host_trap"] = [
        {
            "inst_index": 0,
            "record": {
                "pc": {"code_object_id": 5, "code_object_offset": 0x10},
                "dispatch_id": 0,
            },
        }
    ]
    tool_data["buffer_records"]["pc_sample_stochastic"] = []

    analyzer, result_path = make_csv_run_analyzer(
        tmp_path,
        {str(workload_path): [tool_data]},
    )
    run_source_export_analysis(analyzer)

    kernel_uuid = pd.read_csv(result_path / "kernel.csv")["kernel_uuid"].iloc[0]
    header, rows = read_per_kernel_isa_file(result_path, kernel_uuid)

    assert stall_reason_columns(header) == []
    # host_trap knows the sample landed, but not whether the wave issued.
    assert rows[0][3:6] == ["1", "", ""]


def test_run_analysis_kernel_filter_reaches_a_sampling_only_workload(tmp_path):
    """-k keeps one kernel's rows, and no code object left with nothing in it."""
    workload_path = tmp_path / "workloads" / ISA_WORKLOAD_NAME / ISA_WORKLOAD_SUB_NAME
    tool_data = make_pc_sampling_tool_data()
    # The filter indexes kernels by descending dispatch duration.
    tool_data["buffer_records"]["kernel_dispatch"][0]["end_timestamp"] = 10
    other_code_object_tool_data = make_pc_sampling_tool_data()
    other_code_object_tool_data["metadata"]["pid"] = 43
    other_code_object_tool_data["code_objects"] = [
        {"code_object_id": 6, "load_base": 0x2000}
    ]
    # Only the kernel the filter drops was sampled out of this code object.
    other_code_object_tool_data["buffer_records"]["pc_sample_stochastic"] = [
        other_code_object_tool_data["buffer_records"]["pc_sample_stochastic"][1]
    ]
    other_code_object_tool_data["buffer_records"]["pc_sample_stochastic"][0]["record"][
        "pc"
    ]["code_object_id"] = 6
    for kernel_symbol in other_code_object_tool_data["kernel_symbols"]:
        kernel_symbol["code_object_id"] = 6

    analyzer, result_path = make_csv_run_analyzer(
        tmp_path,
        {str(workload_path): [tool_data, other_code_object_tool_data]},
        filter_kernel_ids=[0],
    )
    run_source_export_analysis(analyzer)

    kernel_frame = pd.read_csv(result_path / "kernel.csv")
    assert list(kernel_frame["kernel_name"]) == ["vecCopy"]
    summary_frame = pd.read_csv(result_path / "pc_sampling_summary.csv")
    assert list(summary_frame["offset"]) == [0x10]
    # The second code object held only the kernel the filter dropped.
    assert set(summary_frame["code_object_id"]) == {5}
    assert per_kernel_isa_paths(result_path) == [
        f"vector_copy/run/kernel_{kernel_frame['kernel_uuid'].iloc[0]}"
        "/isa_code_object_id_5_pid_42.csv"
    ]


@pytest.mark.parametrize("filter_dispatch_ids", [["1"], [">0"], ["> 0"]])
def test_run_analysis_dispatch_filter_reaches_a_sampling_only_workload(
    tmp_path, filter_dispatch_ids
):
    """-d drops a dispatch, and the kernel that keeps none of its own."""
    workload_path = tmp_path / "workloads" / ISA_WORKLOAD_NAME / ISA_WORKLOAD_SUB_NAME
    analyzer, result_path = make_csv_run_analyzer(
        tmp_path,
        {str(workload_path): [make_pc_sampling_tool_data()]},
        filter_dispatch_ids=filter_dispatch_ids,
    )

    run_source_export_analysis(analyzer)

    kernel_frame = pd.read_csv(result_path / "kernel.csv")
    assert list(kernel_frame["kernel_name"]) == ["vecAdd"]
    assert per_kernel_isa_paths(result_path) == [
        f"vector_copy/run/kernel_{kernel_frame['kernel_uuid'].iloc[0]}"
        "/isa_code_object_id_5_pid_42.csv"
    ]


# =============================================================================
# TESTS FOR Analysis DB mode: Analysis DB mode code path
# =============================================================================


def test_calc_roofline_data_early_exit_on_empty_roofline_df(monkeypatch):
    """Test calc_roofline_data exits early when roofline data is empty.

    This test verifies that when the roofline dataframe (ID 402) is empty
    or filtered out, the function logs a warning and skips that workload
    without adding it to the result dictionary.
    """
    from rocprof_compute_analyze.analysis_db import db_analysis

    # Create mock db_analysis instance
    analyzer = mock.MagicMock(spec=db_analysis)

    # Mock workload data
    workload_path = "/mock/workload/path"
    mock_runs = {
        workload_path: mock.MagicMock(sys_info=pd.DataFrame([{"gpu_arch": "gfx90a"}]))
    }

    # Mock PMC dataframe with kernel data
    mock_pmc_df = pd.DataFrame({
        "Kernel_Name": ["kernel1", "kernel2"],
        "Start_Timestamp": [100, 200],
        "End_Timestamp": [150, 300],
    })

    # Mock architecture config with EMPTY roofline dataframe (ID 402)
    mock_arch_config = mock.MagicMock()
    mock_arch_config.dfs = {
        402: pd.DataFrame()  # Empty roofline dataframe triggers early exit
    }

    # Setup instance variables
    analyzer._runs = mock_runs
    analyzer._pmc_df_per_workload = {workload_path: mock_pmc_df}
    analyzer._arch_configs = {"gfx90a": mock_arch_config}
    analyzer.get_args = mock.MagicMock(return_value=mock.MagicMock(max_stat_num=10))

    # Mock console_warning to verify it's called
    warning_messages = []

    def mock_warning(msg):
        warning_messages.append(msg)

    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_db.console_warning", mock_warning
    )
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_db.console_debug", lambda msg: None
    )

    # Call the actual function
    result = db_analysis.calc_roofline_data(analyzer)

    # Verify early exit behavior
    assert len(result[0]) == 0, (
        "Should return empty kernel level dict when roofline data is empty"
    )
    assert len(result[1]) == 0, (
        "Should return empty workload level dict when roofline data is empty"
    )
    assert len(warning_messages) == 1, "Should log one warning message"
    assert "Roofline data is filtered out or not found" in warning_messages[0]
    assert workload_path in warning_messages[0]

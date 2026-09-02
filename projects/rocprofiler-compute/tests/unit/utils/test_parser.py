# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils.parser.build_dfs, apply_filters, apply_kernel_filter,
load_pc_sampling_data, utils_analysis filter resolution, and
utils_common.expand_placeholder_ranges."""

from collections import OrderedDict
from types import SimpleNamespace
from typing import Any

import common
import pandas as pd
import pytest

from pc_sampling.pc_sampling_analysis import load_pc_sample_records
from utils import schema
from utils.parser import (
    apply_filters,
    apply_kernel_filter,
    build_dfs,
    load_pc_sampling_data,
)
from utils.utils_common import (
    convert_filter_blocks_to_panel_ids,
    expand_placeholder_ranges,
)

# =============================================================================
# Helpers to build in-memory panel configs
# =============================================================================


def _metric_panel(
    panel_id: int,
    table_id: int,
    metrics: dict[str, dict[str, str]],
    title: str = "Test Panel",
) -> dict[str, Any]:
    """Build a minimal panel with a single metric_table data source."""
    return {
        "id": panel_id,
        "title": title,
        "data source": [
            {
                "metric_table": {
                    "id": table_id,
                    "title": f"Table {table_id}",
                    "header": {"metric": "Metric", "value": "Avg"},
                    "metric": metrics,
                }
            }
        ],
    }


def _raw_csv_panel(panel_id: int, table_id: int, source: str) -> dict[str, Any]:
    return {
        "id": panel_id,
        "title": "Raw CSV Panel",
        "data source": [{"raw_csv_table": {"id": table_id, "source": source}}],
    }


def _make_arch_config(panels: list[tuple[int, dict[str, Any]]]) -> schema.ArchConfig:
    ac = schema.ArchConfig()
    ac.panel_configs = OrderedDict(panels)
    return ac


def _sys_info() -> dict[str, Any]:
    return {"total_l2_chan": 4}


# =============================================================================
# convert_filter_blocks_to_panel_ids
# =============================================================================


class TestResolveFilterBlocksToPanelIds:
    def test_empty_list_returns_empty_set(self):
        assert convert_filter_blocks_to_panel_ids([]) == set()

    def test_numeric_ids_resolve_to_file_ids(self):
        result = convert_filter_blocks_to_panel_ids(["2", "11.1", "11.1.5"])
        assert result == {200, 1100}

    def test_alias_tokens_resolve_via_arch_map(self, monkeypatch):
        monkeypatch.setattr(
            "utils.utils_common.get_arch_alias_to_panel_id",
            lambda arch: {"lds": "10", "roofline": "4"},
        )
        result = convert_filter_blocks_to_panel_ids(["lds", "roofline"], arch="gfx942")
        assert result == {1000, 400}

    def test_mixed_alias_and_numeric_tokens(self, monkeypatch):
        monkeypatch.setattr(
            "utils.utils_common.get_arch_alias_to_panel_id",
            lambda arch: {"lds": "10"},
        )
        result = convert_filter_blocks_to_panel_ids(["lds", "2", "11.1"], arch="gfx942")
        assert result == {1000, 200, 1100}

    def test_unknown_alias_raises(self, monkeypatch):
        monkeypatch.setattr(
            "utils.utils_common.get_arch_alias_to_panel_id",
            lambda arch: {"lds": "10"},
        )
        with pytest.raises(SystemExit):
            convert_filter_blocks_to_panel_ids(["bogus"], arch="gfx942")


# =============================================================================
# build_dfs
# =============================================================================


def _two_block_config() -> schema.ArchConfig:
    """Config with data_source-0 (1), system panel (100), block 2, block 11."""
    return _make_arch_config([
        (0, _raw_csv_panel(0, 1, "kernel_top.csv")),
        (100, _raw_csv_panel(100, 101, "sysinfo.csv")),
        (
            200,
            _metric_panel(
                200,
                201,
                metrics={
                    "M1": {"value": "AVG(COUNTER_A)"},
                    "M2": {"value": "AVG(COUNTER_B)"},
                },
            ),
        ),
        (
            1100,
            _metric_panel(
                1100,
                1101,
                metrics={"X1": {"value": "AVG(COUNTER_C)"}},
            ),
        ),
    ])


class TestBuildDfs:
    def test_no_filter_builds_all_metrics(self):
        ac = _two_block_config()
        build_dfs(ac, filter_metrics=None, sys_info=_sys_info(), profiling_config={})

        assert set(ac.dfs.keys()) == {1, 101, 201, 1101}
        assert len(ac.dfs[201]) == 2
        assert len(ac.dfs[1101]) == 1

    def test_filter_metrics_overrides_profiling_filter_blocks(self):
        ac = _two_block_config()
        build_dfs(
            ac,
            filter_metrics=["2"],
            sys_info=_sys_info(),
            profiling_config={"filter_blocks": ["11"]},  # ignored
        )

        # filter_metrics wins -> block 200 present, block 1100 absent.
        assert set(ac.dfs.keys()) == {1, 101, 201}
        assert len(ac.dfs[201]) == 2

    def test_profiling_filter_blocks_keeps_only_matching_block(self):
        ac = _two_block_config()
        build_dfs(
            ac,
            filter_metrics=None,
            sys_info=_sys_info(),
            profiling_config={"filter_blocks": ["11"]},
        )

        # System panel always present; block 1100 in filter; block 200 dropped.
        assert set(ac.dfs.keys()) == {1, 101, 1101}

    def test_system_panels_and_data_source_zero_always_present(self):
        ac = _two_block_config()
        build_dfs(
            ac,
            filter_metrics=None,
            sys_info=_sys_info(),
            profiling_config={"filter_blocks": ["99"]},  # excludes blocks 2 and 11
        )

        # data_source 0 and system panel survive; M-blocks filtered out.
        assert set(ac.dfs.keys()) == {1, 101}

    def test_placeholder_range_entries_are_expanded(self):
        # Use an integer range value so we don't depend on sys_info plumbing here.
        metrics: dict[str, Any] = {
            "Channel_::_1": {"value": "AVG(TCC_HIT[::_1])"},
            "placeholder_range": {"::_1": 3},
        }
        ac = _make_arch_config([
            (1800, _metric_panel(1800, 1801, metrics=metrics)),
        ])
        build_dfs(ac, filter_metrics=None, sys_info=_sys_info(), profiling_config={})

        df = ac.dfs[1801]
        # Three expanded rows: Channel_0, Channel_1, Channel_2
        assert len(df) == 3
        assert set(df["Metric"]) == {"Channel_0", "Channel_1", "Channel_2"}

    def test_metric_level_filter_drops_siblings_keeps_headers(self):
        ac = _two_block_config()
        build_dfs(
            ac, filter_metrics=["2.1.0"], sys_info=_sys_info(), profiling_config={}
        )

        df = ac.dfs[201]
        # Headers preserved
        assert list(df.columns) == ["Metric", "Avg"]
        # Only the matching metric remains
        assert list(df["Metric"]) == ["M1"]

    def test_whole_block_filter_keeps_every_metric_in_block(self):
        ac = _two_block_config()
        build_dfs(ac, filter_metrics=["2"], sys_info=_sys_info(), profiling_config={})

        df = ac.dfs[201]
        assert list(df["Metric"]) == ["M1", "M2"]

    def test_profile_side_alias_resolves_to_panel_id(self, monkeypatch):
        monkeypatch.setattr(
            "utils.utils_common.get_arch_alias_to_panel_id",
            lambda arch: {"lds": "11"},
        )
        ac = _two_block_config()
        build_dfs(
            ac,
            filter_metrics=None,
            sys_info=_sys_info(),
            profiling_config={"filter_blocks": ["lds"]},
            arch="gfx942",
        )
        assert set(ac.dfs.keys()) == {1, 101, 1101}

    def test_analyze_side_alias_resolves_to_panel_id(self, monkeypatch):
        monkeypatch.setattr(
            "utils.utils_common.get_arch_alias_to_panel_id",
            lambda arch: {"lds": "11"},
        )
        ac = _two_block_config()
        build_dfs(
            ac,
            filter_metrics=["lds"],
            sys_info=_sys_info(),
            profiling_config={"filter_blocks": ["2"]},  # ignored
            arch="gfx942",
        )
        assert set(ac.dfs.keys()) == {1, 101, 1101}

    def test_analyze_mixed_numeric_and_alias_filter(self, monkeypatch):
        monkeypatch.setattr(
            "utils.utils_common.get_arch_alias_to_panel_id",
            lambda arch: {"lds": "11"},
        )
        ac = _two_block_config()
        build_dfs(
            ac,
            filter_metrics=["2.1.0", "lds"],
            sys_info=_sys_info(),
            profiling_config={},
            arch="gfx942",
        )
        # Numeric token keeps only M1 in block 2; alias keeps block 11.
        assert set(ac.dfs.keys()) == {1, 101, 201, 1101}
        assert list(ac.dfs[201]["Metric"]) == ["M1"]
        assert list(ac.dfs[1101]["Metric"]) == ["X1"]

    def test_metric_counters_only_for_built_metrics(self):
        ac = _make_arch_config([
            (
                200,
                _metric_panel(
                    200,
                    201,
                    metrics={
                        "Kept": {"value": "AVG(COUNTER_KEPT)"},
                        "Dropped": {"value": "AVG(COUNTER_DROPPED)"},
                    },
                ),
            ),
        ])
        build_dfs(
            ac, filter_metrics=["2.1.0"], sys_info=_sys_info(), profiling_config={}
        )

        assert "Kept" in ac.metric_counters
        assert "Dropped" not in ac.metric_counters
        assert ac.metric_counters["Kept"] == ["COUNTER_KEPT"]
        assert ac.dfs_expressions[201] == ["AVG(COUNTER_KEPT)"]


# =============================================================================
# expand_placeholder_ranges
# =============================================================================


def _placeholder_panel(range_value: Any) -> OrderedDict[int, dict[str, Any]]:
    metrics: dict[str, Any] = {
        "Channel_::_1": {"value": "AVG(TCC_HIT[::_1])"},
        "placeholder_range": {"::_1": range_value},
    }
    return OrderedDict([(1800, _metric_panel(1800, 1801, metrics=metrics))])


class TestExpandPlaceholderRanges:
    def test_integer_placeholder_value_expands_n_times(self):
        configs = _placeholder_panel(3)
        result = expand_placeholder_ranges(configs, _sys_info())

        expanded = result[1800]["data source"][0]["metric_table"]["metric"]
        assert list(expanded) == ["Channel_0", "Channel_1", "Channel_2"]

    def test_total_l2_chan_resolves_from_sys_info(self):
        configs = _placeholder_panel("$total_l2_chan")
        result = expand_placeholder_ranges(configs, {"total_l2_chan": 4})

        expanded = result[1800]["data source"][0]["metric_table"]["metric"]
        assert list(expanded) == [
            "Channel_0",
            "Channel_1",
            "Channel_2",
            "Channel_3",
        ]

    def test_unsupported_builtin_var_exits(self):
        configs = _placeholder_panel("$unsupported")
        with pytest.raises(SystemExit):
            expand_placeholder_ranges(configs, _sys_info())

    def test_key_prefix_with_space_survives_expansion(self):
        """
        Panel 18 labels its rows "Channel 0", "Channel 1", ... by carrying
        the literal prefix on the placeholder key. Only the placeholder token
        is substituted; the space-bearing prefix must come through intact."""
        metrics: dict[str, Any] = {
            "Channel ::_1": {"expr": "AVG(TCC_HIT[::_1])"},
            "placeholder_range": {"::_1": "$total_l2_chan"},
        }
        configs = OrderedDict([(1800, _metric_panel(1800, 1802, metrics=metrics))])

        result = expand_placeholder_ranges(configs, {"total_l2_chan": 3})

        expanded = result[1800]["data source"][0]["metric_table"]["metric"]
        assert list(expanded) == ["Channel 0", "Channel 1", "Channel 2"]
        assert expanded["Channel 2"] == {"expr": "AVG(TCC_HIT[2])"}

    def test_none_sys_info_clears_metric_dict(self):
        configs = _placeholder_panel(3)
        result = expand_placeholder_ranges(configs, None)

        expanded = result[1800]["data source"][0]["metric_table"]["metric"]
        assert expanded == {}


# =============================================================================
# Tests for utils.parser.apply_filters
# =============================================================================


def _filter_workload() -> SimpleNamespace:
    """Workload stub exposing raw_pmc and filter attributes for apply_filters."""
    return SimpleNamespace(
        raw_pmc=pd.DataFrame({
            "GPU_ID": [0, 0, 1, 1],
            "Kernel_Name": ["vecCopy", "vecAdd", "vecCopy", "vecMul"],
            "Dispatch_ID": [0, 1, 2, 3],
        }),
        filter_gpu_ids=None,
        filter_kernel_ids=None,
        filter_dispatch_ids=None,
    )


def _kernel_filter_workload() -> SimpleNamespace:
    """Workload stub with dfs populated for apply_kernel_filter tests."""
    return SimpleNamespace(
        dfs={
            1: pd.DataFrame({
                "Kernel_Name": ["kernel_a", "kernel_b", "kernel_c"],
                "Count": [2, 1, 1],
                "Sum(ns)": [900, 800, 200],
                "Selected": ["", "", ""],
            }),
            2: pd.DataFrame({
                "Dispatch_ID": [1, 2, 3, 4],
                "Kernel_Name": ["kernel_a", "kernel_b", "kernel_a", "kernel_c"],
                "GPU_ID": [0, 0, 1, 0],
            }),
        },
        filter_kernel_ids=[],
        filter_dispatch_ids=None,
    )


def _flat_raw_df() -> pd.DataFrame:
    """Flat single-index raw_pmc DataFrame for apply_kernel_filter tests."""
    return pd.DataFrame({
        "Kernel_Name": ["kernel_a", "kernel_b", "kernel_a", "kernel_c"],
        "GPU_ID": [0, 0, 1, 0],
        "Dispatch_ID": [1, 2, 3, 4],
    })


class TestApplyFilters:
    """Tests for utils.parser.apply_filters."""

    def test_gpu_string_filter(self) -> None:
        """A string GPU filter keeps only matching rows."""
        workload = _filter_workload()
        workload.filter_gpu_ids = "0"
        assert len(apply_filters(workload, "/tmp", False, False)) == 2

    def test_kernel_name_filter(self) -> None:
        """A kernel-name filter keeps only matching rows."""
        workload = _filter_workload()
        workload.filter_kernel_ids = ["vecCopy"]
        assert len(apply_filters(workload, "/tmp", False, False)) == 2

    def test_dispatch_id_filter(self) -> None:
        """A dispatch-ID filter keeps only matching rows."""
        workload = _filter_workload()
        workload.filter_dispatch_ids = ["0", "1"]
        assert len(apply_filters(workload, "/tmp", False, False)) == 2

    def test_gpu_integer_list_filter(self) -> None:
        """A GPU filter given as a list of integers keeps all matching rows."""
        workload = _filter_workload()
        workload.filter_gpu_ids = [0, 1]
        assert len(apply_filters(workload, "/tmp", False, False)) == 4


class TestApplyKernelFilter:
    """Tests for utils.parser.apply_kernel_filter."""

    def test_integer_ids_select_and_mark(self) -> None:
        """Integer kernel IDs filter rows and set the Selected marker."""
        workload = _kernel_filter_workload()
        workload.filter_kernel_ids = [0]
        result_df = apply_kernel_filter(_flat_raw_df(), workload)

        assert len(result_df) == 2
        assert all(result_df["Kernel_Name"] == "kernel_a")
        assert workload.dfs[1].loc[0, "Selected"] == "*"

    def test_multiple_integer_ids(self) -> None:
        """Multiple integer kernel IDs keep the union of matching rows."""
        workload = _kernel_filter_workload()
        workload.filter_kernel_ids = [0, 1]
        result_df = apply_kernel_filter(_flat_raw_df(), workload)
        assert len(result_df) == 3

    def test_invalid_id_errors(self, monkeypatch) -> None:
        """An out-of-bounds kernel ID triggers console_error and exits."""
        error_calls = []

        def record_and_exit(*args, **_kwargs):
            error_calls.append(args)
            raise SystemExit(1)

        common.patch_console(
            monkeypatch, "utils.parser", "error", error=record_and_exit
        )
        workload = _kernel_filter_workload()
        workload.filter_kernel_ids = [99]
        with pytest.raises(SystemExit):
            apply_kernel_filter(_flat_raw_df(), workload)
        assert error_calls
        assert "99" in str(error_calls[0])

    def test_exact_name_match(self) -> None:
        """A string kernel name filters to the exact match."""
        workload = _kernel_filter_workload()
        workload.filter_kernel_ids = ["kernel_b"]
        result_df = apply_kernel_filter(_flat_raw_df(), workload)
        assert len(result_df) == 1
        assert result_df["Kernel_Name"].iloc[0] == "kernel_b"

    def test_name_match_strips_whitespace(self) -> None:
        """Kernel names with surrounding whitespace are stripped before matching."""
        raw_df_with_whitespace = pd.DataFrame({
            "Kernel_Name": [" kernel_a ", "kernel_b", "kernel_a"],
            "GPU_ID": [0, 0, 1],
            "Dispatch_ID": [1, 2, 3],
        })
        workload = _kernel_filter_workload()
        workload.filter_kernel_ids = ["kernel_a"]
        result_df = apply_kernel_filter(raw_df_with_whitespace, workload)
        assert len(result_df) == 2


# =============================================================================
# Tests for load_pc_sampling_data
# =============================================================================


def _kernel_top_workload() -> SimpleNamespace:
    """Workload stub with dfs[1] populated for load_pc_sampling_data tests."""
    return SimpleNamespace(
        filter_kernel_ids=[],
        dfs={
            1: pd.DataFrame({
                "Kernel_Name": ["kernel_a", "kernel_b", "kernel_c"],
                "Count": [2, 1, 1],
                "Sum(ns)": [900, 800, 200],
            }),
        },
    )


def test_load_pc_sampling_data_missing_or_empty_sources_return_empty() -> None:
    """Absent tool data and empty buffer records both yield empty frames."""
    workload = SimpleNamespace(filter_kernel_ids=[])

    assert load_pc_sampling_data(workload, "count", None).empty

    workload.filter_kernel_ids = [0, 1, 2]
    assert load_pc_sampling_data(workload, "count", None).empty

    empty_records = load_pc_sample_records({
        "buffer_records": {
            "pc_sample_stochastic": [],
            "pc_sample_host_trap": [],
            "kernel_dispatch": [],
        },
    })
    assert empty_records.empty


def test_load_pc_sampling_data_out_of_bounds_kernel_warns(monkeypatch) -> None:
    """An out-of-bounds kernel index warns and returns empty."""
    mock_warning = common.patch_console(monkeypatch, "utils.parser", "warning")[
        "warning"
    ]
    workload = _kernel_top_workload()
    tool_data = {
        "buffer_records": {"pc_sample_stochastic": [{}], "pc_sample_host_trap": []}
    }

    workload.filter_kernel_ids = [99]
    result = load_pc_sampling_data(workload, "count", [tool_data])

    mock_warning.assert_called()
    call_args_str = str(mock_warning.call_args)
    assert "out of bounds" in call_args_str or "99" in call_args_str
    assert result.empty


# ---------------------------------------------------------------------------
# Torch operator pattern matching (fnmatch glob)
# ---------------------------------------------------------------------------

H3 = "nn.Module.Net.forward/torch.nn.functional.relu/torch.relu"
H2 = "nn.Module.Net.forward/torch.nn.functional.conv2d"
H1 = "torch.relu"


@pytest.mark.torch_ops
def test_all_keyword():
    """'all' maps to '**' and matches every hierarchy."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("all", H3)
    assert m("all", H2)
    assert m("all", H1)
    assert not m("all", "")


@pytest.mark.torch_ops
def test_bare_pattern_requires_exact_match():
    """A pattern without wildcards matches only when equal to the full target."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("torch.relu", H1)
    assert not m("torch.relu", H3)
    assert not m("torch.nn.functional.conv2d", H2)
    assert not m("relu", H3)
    assert not m("forward", H3)
    assert not m("sigmoid", H3)


@pytest.mark.torch_ops
def test_substring_wildcard_pattern():
    """``*`` matches any run of characters, including ``/``."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("*torch.relu", H3)
    assert m("*torch.*", H3)
    assert m("*relu", H3)
    assert m("*conv*", H2)
    assert m("*relu*", H3)
    assert not m("conv*", H2)
    assert not m("sigm*", H3)


@pytest.mark.torch_ops
def test_hierarchy_glob():
    """Patterns with '/' match across multiple hierarchy components."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("nn.Module.Net.forward/*/torch.relu", H3)
    assert m("*/torch.nn.functional.conv2d", H2)
    assert not m("nn.Module.Net.forward/torch.relu", H3)


@pytest.mark.torch_ops
def test_leading_slash_is_cosmetic():
    """A leading ``/`` in the pattern is stripped before matching."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("/nn.Module.Net.forward/*/torch.relu", H3)
    assert m("/torch.relu", H1)


@pytest.mark.torch_ops
def test_trailing_slash_is_cosmetic():
    """A trailing ``/`` in the pattern is stripped before matching."""
    from utils.parser import torch_operator_pattern_matches as m

    assert not m("nn.Module.Net.forward/", H3)
    assert m("torch.relu/", H1)


@pytest.mark.torch_ops
def test_regex_not_supported():
    """Regex syntax has no special meaning; treated as literal glob text."""
    from utils.parser import torch_operator_pattern_matches as m

    assert not m("relu|conv2d", H3)
    assert not m("^torch\\.relu$", H3)
    assert not m("not:relu", H3)
    assert not m("2:functional", H3)


@pytest.mark.torch_ops
def test_empty_inputs():
    """Empty pattern or operator_name returns False."""
    from utils.parser import torch_operator_pattern_matches as m

    assert not m("", H3)
    assert not m("relu", "")
    assert not m("", "")


@pytest.mark.torch_ops
def test_slash_only_markers():
    """Scope-marker-only tokens should not match any hierarchy."""
    from utils.parser import torch_operator_pattern_matches as m

    assert not m("/", H3)
    assert not m("//", H3)


# -- get_matched_torch_operators_for_display ---------------------------------


def get_matched_torch_operators_for_display(
    torch_operators: dict[str, pd.DataFrame],
    pattern_list: list[str],
) -> list[tuple[str, pd.DataFrame]]:
    """Return (operator_name, filtered_df) for each operator matching any pattern.

    Test-only helper: iterates every unique Operator_Name across all torch trace
    DataFrames and checks each against the supplied glob patterns.
    """
    from utils.parser import torch_operator_pattern_matches

    if not torch_operators or not pattern_list:
        return []
    result: list[tuple[str, pd.DataFrame]] = []
    seen: set[str] = set()
    for _, df in torch_operators.items():
        if df is None or df.empty or "Operator_Name" not in df.columns:
            continue
        for op_name in df["Operator_Name"].dropna().unique():
            op_str = str(op_name).strip()
            if op_str in seen:
                continue
            for pattern in pattern_list:
                if torch_operator_pattern_matches(pattern.strip(), op_str):
                    seen.add(op_str)
                    result.append((op_str, df.loc[df["Operator_Name"] == op_name]))
                    break
    return result


@pytest.mark.torch_ops
def test_display_match_hierarchy_glob():
    """Full hierarchy globs are honored by display helper."""
    df = pd.DataFrame({
        "Operator_Name": [H3, H3, H2],
        "Kernel_Name": ["k1", "k2", "k3"],
    })
    torch_operators = {"trace_0": df}

    matched = get_matched_torch_operators_for_display(torch_operators, ["*/torch.relu"])
    assert len(matched) == 1
    assert matched[0][0] == H3


@pytest.mark.torch_ops
def test_display_match_multi_patterns():
    """Multiple glob patterns match their respective operators."""
    df = pd.DataFrame({
        "Operator_Name": [H3, H2],
        "Kernel_Name": ["k1", "k2"],
    })
    torch_operators = {"trace_0": df}

    matched = get_matched_torch_operators_for_display(
        torch_operators, ["*relu", "*conv*"]
    )
    assert len(matched) == 2


@pytest.mark.torch_ops
def test_display_no_match():
    """No matches returns empty list."""
    df = pd.DataFrame({
        "Operator_Name": [H3],
        "Kernel_Name": ["k1"],
    })
    assert get_matched_torch_operators_for_display({"t": df}, ["sigmoid"]) == []


@pytest.mark.torch_ops
def test_display_empty_inputs():
    """Empty torch_operators or pattern_list returns []."""
    assert get_matched_torch_operators_for_display({}, ["relu"]) == []
    assert get_matched_torch_operators_for_display({"x": pd.DataFrame()}, []) == []


# -- Additional coverage (xuchen #26) ----------------------------------------


@pytest.mark.torch_ops
def test_double_star_explicit():
    """'**' matches any hierarchy depth."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("**", H3)
    assert m("**", H2)
    assert m("**", H1)
    assert m("**", "a/b/c/d/e")
    assert not m("**", "")


@pytest.mark.torch_ops
def test_single_char_wildcard():
    """``?`` matches exactly one character."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("*torch.rel?", H3)
    assert m("*torch.?elu", H3)
    assert not m("torch.?", H3)
    assert not m("?", H1)
    assert m("*torch.nn.functional.conv?d", H2)


@pytest.mark.torch_ops
def test_long_hierarchy():
    """Patterns apply to deeply nested hierarchies."""
    from utils.parser import torch_operator_pattern_matches as m

    deep = "/".join([f"level{i}" for i in range(20)])
    assert m("*level19", deep)
    assert m("*19", deep)
    assert m("*/level19", deep)
    assert m("all", deep)
    assert not m("level0", deep)
    assert m("*level0*", deep)


@pytest.mark.torch_ops
def test_long_component_names():
    """Patterns apply to components with long names."""
    from utils.parser import torch_operator_pattern_matches as m

    long_name = "a" * 500
    hierarchy = f"root/{long_name}"
    assert m(f"*{long_name}", hierarchy)
    assert m("*a*", hierarchy)
    assert not m("a*", hierarchy)
    assert not m("b*", hierarchy)


@pytest.mark.torch_ops
def test_special_characters_in_names():
    """Dots and underscores are treated literally."""
    from utils.parser import torch_operator_pattern_matches as m

    h = "nn.Module._internal/torch.nn.functional.conv2d"
    assert m("*torch.nn.functional.conv2d", h)
    assert m("*conv2d", h)
    assert m("nn.Module._internal/*", h)
    assert not m("nn_Module._internal/*", h)


@pytest.mark.torch_ops
def test_bracket_glob_pattern():
    """Character classes ``[abc]`` are supported."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("*torch.rel[uv]", H3)
    assert not m("*torch.rel[ab]", H3)


@pytest.mark.torch_ops
def test_single_component_hierarchy():
    """A single-component target matches an equal bare pattern."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("torch.relu", "torch.relu")
    assert m("*relu", "torch.relu")
    assert m("torch.*", "torch.relu")
    assert not m("*/torch.relu", "torch.relu")


@pytest.mark.torch_ops
def test_whitespace_only_pattern():
    """Whitespace-only patterns normalize to empty and return False."""
    from utils.parser import torch_operator_pattern_matches as m

    assert not m("   ", H3)
    assert not m("\t", H3)


@pytest.mark.torch_ops
def test_star_pattern_matches_all():
    """``*`` matches any non-empty target."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("*", H3)
    assert m("*", H2)
    assert m("*", H1)
    assert m("*", "a/b/c/d/e")
    assert not m("*", "")


@pytest.mark.torch_ops
def test_star_normalize_equivalence():
    """``"all"`` and ``"*"`` both match any non-empty target."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("all", H3)
    assert m("*", H3)


@pytest.mark.torch_ops
def test_case_sensitivity():
    """Pattern matching is case-sensitive."""
    from utils.parser import torch_operator_pattern_matches as m

    assert not m("Torch.Relu", H3)
    assert not m("TORCH.RELU", H3)
    assert not m("ALL", H3)
    assert m("all", H3)


@pytest.mark.torch_ops
def test_all_keyword_case_sensitive():
    """The ``"all"`` alias is case-sensitive; other casings are literal."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("all", H3)
    assert not m("ALL", H3)
    assert not m("All", H3)


@pytest.mark.torch_ops
def test_consecutive_slashes_in_target():
    """Consecutive slashes in the target are treated literally."""
    from utils.parser import torch_operator_pattern_matches as m

    h = "a//b///torch.relu"
    assert m("*torch.relu", h)
    assert m("*relu", h)


@pytest.mark.torch_ops
def test_dots_in_patterns():
    """Dots are treated literally, not as regex wildcards."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("*torch.relu", H3)
    assert not m("*torchXrelu", H3)
    h = "root/torchXrelu"
    assert not m("*torch.relu", h)
    assert m("*torchXrelu", h)


@pytest.mark.torch_ops
def test_pattern_with_spaces():
    """Spaces are treated literally."""
    from utils.parser import torch_operator_pattern_matches as m

    h = "module/ spaced op /torch.relu"
    assert m("*torch.relu", h)
    assert not m(" spaced op ", h)
    assert m("* spaced op */*", h)


@pytest.mark.torch_ops
def test_colons_in_operator_names():
    """Colons are treated literally."""
    from utils.parser import torch_operator_pattern_matches as m

    h = "nn.Module/aten::relu_"
    assert m("*aten::relu_", h)
    assert m("*relu_", h)
    assert m("*aten::*", h)
    assert not m("*relu", h)
    assert not m("*torch.relu", h)


@pytest.mark.torch_ops
def test_display_star_matches_all_operators():
    """'*' pattern matches all operators in display helper."""
    df = pd.DataFrame({
        "Operator_Name": [H3, H2],
        "Kernel_Name": ["k1", "k2"],
    })
    torch_operators = {"trace_0": df}

    matched = get_matched_torch_operators_for_display(torch_operators, ["*"])
    assert len(matched) == 2


@pytest.mark.torch_ops
def test_display_dedup_across_dataframes():
    """Same operator in multiple DataFrames is matched only once."""
    df1 = pd.DataFrame({"Operator_Name": [H3], "Kernel_Name": ["k1"]})
    df2 = pd.DataFrame({"Operator_Name": [H3], "Kernel_Name": ["k2"]})
    torch_operators = {"trace_0": df1, "trace_1": df2}

    matched = get_matched_torch_operators_for_display(torch_operators, ["all"])
    op_names = [name for name, _ in matched]
    assert op_names.count(H3) == 1

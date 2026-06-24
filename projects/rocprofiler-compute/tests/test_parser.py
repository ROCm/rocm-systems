# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils.parser.build_dfs, utils_analysis filter resolution,
and utils_common.expand_placeholder_ranges."""

from collections import OrderedDict
from typing import Any

import pytest

from utils import schema
from utils.parser import build_dfs
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


def _pc_sampling_panel(panel_id: int, table_id: int, source: str) -> dict[str, Any]:
    return {
        "id": panel_id,
        "title": "PC Sampling Panel",
        "data source": [{"pc_sampling_table": {"id": table_id, "source": source}}],
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

    def test_none_sys_info_clears_metric_dict(self):
        configs = _placeholder_panel(3)
        result = expand_placeholder_ranges(configs, None)

        expanded = result[1800]["data source"][0]["metric_table"]["metric"]
        assert expanded == {}

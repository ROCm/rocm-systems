# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils/mem_chart_common.py."""

import pytest

from utils import mem_chart_common

# =============================================================================
# mem_chart_common helpers
# =============================================================================


class TestFormatValue:
    @pytest.mark.parametrize(
        "value, unit, prec, expected",
        [
            (85.5, "%", 1, "85.5%"),
            (None, "%", 1, "N/A"),
            ("50.5", "%", 1, "50.5%"),
            ("invalid", "%", 1, "N/A"),
        ],
    )
    def test_format(self, value, unit, prec, expected):
        assert mem_chart_common.format_value(value, unit, prec) == expected

    @pytest.mark.parametrize(
        "value, unit, expected",
        [(123.4567e9, "Bytes/s", "123.457 GB/s"), (1.5, "GB/s", "1.500 GB/s")],
    )
    def test_bandwidth_is_fixed_gbps(self, value, unit, expected):
        assert mem_chart_common.format_value(value, unit, 0) == expected


class TestFormatScientific:
    @pytest.mark.parametrize(
        "value, expected_contains",
        [(100, "100"), (999, "999"), (1_000_000, "e"), (None, "N/A"), (-500, "-500")],
    )
    def test_format(self, value, expected_contains):
        assert expected_contains in mem_chart_common.format_scientific(value)


class TestProgressBar:
    @pytest.mark.parametrize(
        "pct, filled, empty",
        [(100, 10, 0), (0, 0, 10), (50, 5, 5), (None, 0, 10), (150, 10, 0)],
    )
    def test_bar(self, pct, filled, empty):
        assert mem_chart_common.progress_bar(pct, 10) == "█" * filled + "░" * empty


class TestSafeFloat:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (1.5, 1.5),
            (0, 0.0),
            ("10", 10.0),
            (None, None),
            ("N/A", None),
            (float("nan"), None),
        ],
    )
    def test_parse(self, value, expected):
        assert mem_chart_common.safe_float(value) == expected


class TestSafeFloatSum:
    @pytest.mark.parametrize(
        "args, expected",
        [((1.5, None, 2.5), 4.0), ((None, None), None), (("10", 5), 15.0)],
    )
    def test_sum(self, args, expected):
        assert mem_chart_common.safe_float_sum(*args) == expected


class TestFormatEdge:
    @pytest.mark.parametrize(
        "label, value, check_in, check_not_in",
        [("Read", 1_500_000, "1.50e+06", None), ("Write", None, "Write", ":")],
    )
    def test_edge(self, label, value, check_in, check_not_in):
        result = mem_chart_common.format_edge(label, value)
        assert check_in in result
        if check_not_in is not None:
            assert check_not_in not in result


class TestMetricLine:
    def test_basic_metric(self):
        result = mem_chart_common.metric_line("Util", 75.5, "%", "green")
        assert "Util" in result
        assert "75.5%" in result
        assert "green" in result

    def test_with_none_value(self):
        result = mem_chart_common.metric_line("BW", None, "GB/s", "cyan")
        assert "BW" in result
        assert "N/A" in result

    def test_precision(self):
        result = mem_chart_common.metric_line("Scratch", 32 / 1024, " KB", precision=3)
        assert "0.031 KB" in result


class TestBuildCuPanel:
    def test_stat_precision_defaults_to_one(self):
        panel = mem_chart_common.build_cu_panel(20, stats=[("LDS Alloc", 2.0, " KB")])
        assert "2.0 KB" in panel.renderable

    def test_stat_precision_override_keeps_sub_kb_values(self):
        panel = mem_chart_common.build_cu_panel(
            20, stats=[("Scratch", 32 / 1024, " KB", 3)]
        )
        assert "0.031 KB" in panel.renderable


class TestBuildCuStats:
    def test_rows_and_units(self):
        stats = mem_chart_common.build_cu_stats(
            {
                "Wavefront Occupancy": 45.5,
                "VGPR": 64,
                "SGPR": 32,
                "Scratch Allocation": 0.125,
                "LDS Allocation": 2048,
                "Workgroups": 2.5,
            },
            "WGP",
        )
        rows = {row[0]: row for row in stats}
        assert rows["Wave Occ"][1:] == (45.5, "%")
        assert rows["Scratch/Wave"][1:] == (0.125, " KB", 3)
        assert rows["LDS Alloc"][1:] == (2.0, " KB")
        assert rows["Workgroups/WGP"][1:] == (2.5, "")

    def test_missing_metrics_render_as_none(self):
        stats = mem_chart_common.build_cu_stats({})
        assert [row[1] for row in stats] == [None] * 6
        assert stats[-1][0] == "Workgroups/CU"


class TestFormatMemChartHeading:
    @pytest.mark.parametrize(
        "unit, panel_id, expected",
        [
            ("per_kernel", 300, "3. Memory Chart (Normalization: per_kernel)"),
            ("per_wave", 500, "5. Memory Chart (Normalization: per_wave)"),
        ],
    )
    def test_heading(self, unit, panel_id, expected):
        result = mem_chart_common.format_mem_chart_heading(unit, panel_id=panel_id)
        assert result == expected


class TestBuildLegend:
    def test_contains_read_write_atomic(self):
        legend = mem_chart_common.build_legend()
        assert "Read" in legend and "Write" in legend and "Atomic" in legend

    def test_stall_optional(self):
        assert "Stall" not in mem_chart_common.build_legend()
        assert "Stall" in mem_chart_common.build_legend(include_stall=True)

    def test_exclude_atomic(self):
        legend = mem_chart_common.build_legend(include_atomic=False)
        assert "Atomic" not in legend


class TestMakeArrows:
    def test_all_keys_same_length(self):
        arrows = mem_chart_common.make_arrows(8)
        for key in ("left", "right", "both", "plain"):
            assert len(arrows[key]) == 8


class TestPadTo:
    def test_pads_short_list(self):
        assert mem_chart_common.pad_to(["a"], 3) == ["a", "", ""]

    def test_truncates_long_list(self):
        assert mem_chart_common.pad_to(["a", "b", "c"], 2) == ["a", "b"]

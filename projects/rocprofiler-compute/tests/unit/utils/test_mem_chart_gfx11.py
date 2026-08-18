# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the gfx11 memory-chart renderer and shared helpers."""

from pathlib import Path

import common
import pytest
import yaml

from utils import mem_chart_common, mem_chart_gfx11
from utils.utils_analysis import format_bw_human_readable

DEFAULT_TITLE = "3. Memory Chart (Normalization: per_kernel)"

MEM_CHART_PANEL_YAML = (
    Path(common.SRC)
    / "rocprof_compute_soc"
    / "analysis_configs"
    / "gfx115x"
    / "0300_memory_chart.yaml"
)


def mem_chart_yaml_metric_names() -> list[str]:
    panel = yaml.safe_load(MEM_CHART_PANEL_YAML.read_text(encoding="utf-8"))
    return [
        name
        for entry in panel["Panel Config"]["data source"]
        for name in entry["metric_table"]["metric"]
    ]


# =============================================================================
# format_bw_human_readable
# =============================================================================


class TestFormatBwHumanReadable:
    @pytest.mark.parametrize(
        "value, unit, prec, expected",
        [
            (1e12, "Bytes/s", 1, "1.0 TB/s"),
            (2.5e12, "Bytes/s", 1, "2.5 TB/s"),
            (1e9, "Bytes/s", 1, "1.0 GB/s"),
            (100e9, "Bytes/s", 1, "100.0 GB/s"),
            (1e6, "Bytes/s", 1, "1.0 MB/s"),
            (1e3, "Bytes/s", 1, "1.0 KB/s"),
            (500, "Bytes/s", 1, "500.0 B/s"),
            (0, "Bytes/s", 1, "0.0 B/s"),
            (100, "GB/s", 1, "100.0 GB/s"),
            (1500, "GB/s", 1, "1.5 TB/s"),
            (None, "Bytes/s", 1, "N/A"),
            ("invalid", "Bytes/s", 1, "N/A"),
        ],
    )
    def test_format(self, value, unit, prec, expected):
        assert format_bw_human_readable(value, unit, prec) == expected

    @pytest.mark.parametrize(
        "prec, expected",
        [(0, "123 GB/s"), (1, "123.5 GB/s"), (2, "123.46 GB/s")],
    )
    def test_precision(self, prec, expected):
        assert format_bw_human_readable(123.456789e9, "Bytes/s", prec) == expected


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

    def test_bytes_per_second_routes_to_human_readable(self):
        assert "GB/s" in mem_chart_common.format_value(100e9, "Bytes/s", 1)


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


# =============================================================================
# gfx11 get_sample_metrics
# =============================================================================


class TestGetSampleMetrics:
    def test_returns_copy(self):
        first = mem_chart_gfx11.get_sample_metrics()
        second = mem_chart_gfx11.get_sample_metrics()
        first["TCP-GL1 Read Bandwidth"] = 0
        assert second["TCP-GL1 Read Bandwidth"] != 0


# =============================================================================
# gfx11 plot_mem_chart
# =============================================================================


class TestPlotMemChartGfx11:
    def test_full_chart(self):
        result = mem_chart_gfx11.plot_mem_chart(
            mem_chart_gfx11.get_sample_metrics(), chart_title=DEFAULT_TITLE
        )
        clean = common.strip_ansi(result)
        assert len(result) > 100
        assert "3. Memory Chart" in clean
        assert "GPU" in clean and "System Memory" in clean

    @pytest.mark.parametrize("block", ["TCP", "GL1 Cache", "GL2 Cache", "GCEA", "DRAM"])
    def test_contains_arch_element(self, block):
        result = mem_chart_gfx11.plot_mem_chart(
            mem_chart_gfx11.get_sample_metrics(), chart_title=DEFAULT_TITLE
        )
        assert block in result

    def test_normalize_drops_unknown_fills_missing(self):
        raw = {"GL0 Cache Hit Rate (TCP Cache)": 1.0, "noise": 99}
        norm = mem_chart_gfx11.normalize_mem_chart_metrics(raw)
        assert norm["GL0 Cache Hit Rate (TCP Cache)"] == 1.0
        assert "noise" not in norm
        assert norm["ICache Requests"] is None


# =============================================================================
# gfx11 zero vs missing metrics
# =============================================================================


class TestZeroVersusMissingMetrics:
    """A measured 0 must render; an absent counter must never read as 0."""

    GUARDED_LINES = [
        ("LDS Utilization", "Util 0.0%"),
        ("LDS Estimated Bandwidth", "BW 0.0 B/s"),
        ("LDS Bank Conflict Rate", "Bank Conflict"),
        ("GL0 Cache BW (TCP Cache)", "BW 0.0 B/s"),
    ]

    @pytest.mark.parametrize(("metric", "expected"), GUARDED_LINES)
    def test_zero_valued_metric_is_displayed(self, metric, expected):
        output = common.strip_ansi(
            mem_chart_gfx11.plot_mem_chart({metric: 0}, chart_title=DEFAULT_TITLE)
        )
        assert expected in output

    def test_all_zero_metrics_render_no_placeholders(self):
        metrics = dict.fromkeys(mem_chart_gfx11.MEM_CHART_PANEL_METRIC_KEYS, 0)
        output = common.strip_ansi(
            mem_chart_gfx11.plot_mem_chart(metrics, chart_title=DEFAULT_TITLE)
        )
        assert "N/A" not in output

    def test_missing_counters_are_not_reported_as_zero(self):
        output = common.strip_ansi(
            mem_chart_gfx11.plot_mem_chart({}, chart_title=DEFAULT_TITLE)
        )
        assert "0.0" not in output


# =============================================================================
# gfx11 panel YAML sync
# =============================================================================


class TestMemChartPanelYamlSync:
    """gfx115x chart metric names must track the YAML on disk."""

    def test_panel_metric_keys_match_yaml_in_order(self):
        assert (
            list(mem_chart_gfx11.MEM_CHART_PANEL_METRIC_KEYS)
            == mem_chart_yaml_metric_names()
        )

    def test_every_chart_lookup_resolves_against_yaml(self):
        metrics = dict.fromkeys(mem_chart_yaml_metric_names(), 1.0)
        extracted = mem_chart_gfx11._extract_metrics(
            mem_chart_gfx11.normalize_mem_chart_metrics(metrics)
        )
        unresolved = sorted(key for key, value in extracted.items() if value is None)
        assert not unresolved, f"metric names drifted from the YAML: {unresolved}"


# =============================================================================
# gfx11 DEFAULT_SAMPLE_METRICS
# =============================================================================


class TestDefaultSampleMetrics:
    def test_keys_match_panel_keys(self):
        assert set(mem_chart_gfx11.DEFAULT_SAMPLE_METRICS) == set(
            mem_chart_gfx11.MEM_CHART_PANEL_METRIC_KEYS
        )

    def test_has_all_memory_hierarchy_levels(self):
        metrics = mem_chart_gfx11.DEFAULT_SAMPLE_METRICS
        for prefix in ("TCP", "LDS", "GL1", "GL2", "DRAM"):
            assert any(prefix in k for k in metrics), f"Missing {prefix}"


def test_chart_title_appears_as_first_line():
    chart_title = "7. Memory Chart (Normalization: per_kernel)"
    output = common.strip_ansi(
        mem_chart_gfx11.plot_mem_chart(
            mem_chart_gfx11.get_sample_metrics(),
            chart_title=chart_title,
        )
    )
    assert output.strip().splitlines()[0] == chart_title
    assert "3. Memory Chart" not in output

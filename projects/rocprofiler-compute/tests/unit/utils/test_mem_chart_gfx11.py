# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the gfx11 memory-chart renderer."""

from pathlib import Path

import common
import pytest
import yaml

from utils import mem_chart_gfx11
from utils.mem_chart_common import strip_ansi

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
        clean = strip_ansi(result)
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
        output = strip_ansi(
            mem_chart_gfx11.plot_mem_chart({metric: 0}, chart_title=DEFAULT_TITLE)
        )
        assert expected in output

    def test_all_zero_metrics_render_no_placeholders(self):
        metrics = dict.fromkeys(mem_chart_gfx11.MEM_CHART_PANEL_METRIC_KEYS, 0)
        output = strip_ansi(
            mem_chart_gfx11.plot_mem_chart(metrics, chart_title=DEFAULT_TITLE)
        )
        assert "N/A" not in output

    def test_missing_counters_are_not_reported_as_zero(self):
        output = strip_ansi(
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
    output = strip_ansi(
        mem_chart_gfx11.plot_mem_chart(
            mem_chart_gfx11.get_sample_metrics(),
            chart_title=chart_title,
        )
    )
    assert output.strip().splitlines()[0] == chart_title
    assert "3. Memory Chart" not in output

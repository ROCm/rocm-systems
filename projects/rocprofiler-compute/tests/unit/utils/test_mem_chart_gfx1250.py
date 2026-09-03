# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the gfx1250 memory-chart renderer."""

import re
from pathlib import Path

from utils import mem_chart_gfx1250
from utils.mem_chart_common import strip_ansi
from vendored import yaml

DEFAULT_TITLE = "3. Memory Chart (Normalization: per_kernel)"


# =============================================================================
# gfx1250 get_sample_metrics
# =============================================================================


class TestGetSampleMetrics:
    def test_returns_copy(self):
        first = mem_chart_gfx1250.get_sample_metrics()
        second = mem_chart_gfx1250.get_sample_metrics()
        first["DRAM Read Bandwidth"] = 0
        assert second["DRAM Read Bandwidth"] != 0


# =============================================================================
# gfx1250 plot_mem_chart
# =============================================================================


class TestPlotMemChartGfx1250:
    """Tests for gfx1250 CDNA5 memory chart generation."""

    def test_returns_string(self):
        metrics = mem_chart_gfx1250.get_sample_metrics()
        result = mem_chart_gfx1250.plot_mem_chart(metrics, chart_title=DEFAULT_TITLE)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_complete_gfx1250_architecture(self):
        """gfx1250 output contains every rendered memory component."""
        metrics = mem_chart_gfx1250.get_sample_metrics()
        output = strip_ansi(
            mem_chart_gfx1250.plot_mem_chart(metrics, chart_title=DEFAULT_TITLE)
        )
        expected_components = (
            "Kernel",
            "TCP",
            "LDS",
            "GL0",
            "SQC",
            "GL1",
            "GLARB",
            "GL2",
            "EA/DF",
            "HBM",
        )
        for component in expected_components:
            assert component in output, f"Missing gfx1250 component: {component}"

    def test_contains_directional_connectors(self):
        """gfx1250 output exposes directional bandwidth connectors."""
        metrics = mem_chart_gfx1250.get_sample_metrics()
        output = strip_ansi(
            mem_chart_gfx1250.plot_mem_chart(metrics, chart_title=DEFAULT_TITLE)
        )
        assert re.search(r"<(?!-+>)-{3,}", output)
        assert re.search(r"(?<![<-])-{3,}>", output)
        assert re.search(r"<-{3,}>", output)

    def test_contains_bandwidth_values(self):
        metrics = mem_chart_gfx1250.get_sample_metrics()
        result = mem_chart_gfx1250.plot_mem_chart(metrics, chart_title=DEFAULT_TITLE)
        assert "GB/s" in result

    def test_empty_metrics(self):
        result = mem_chart_gfx1250.plot_mem_chart({}, chart_title=DEFAULT_TITLE)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_partial_metrics(self):
        partial = {
            "GL0-GL1 Read Bandwidth": 50e9,
            "GL2-EA Read Bandwidth": 200e9,
        }
        result = mem_chart_gfx1250.plot_mem_chart(partial, chart_title=DEFAULT_TITLE)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_extreme_bandwidth_values(self):
        extreme = {
            "DRAM Read Bandwidth": 10e12,
            "DRAM Write Bandwidth": 5e12,
        }
        result = mem_chart_gfx1250.plot_mem_chart(extreme, chart_title=DEFAULT_TITLE)
        assert "TB/s" in result

    def test_zero_bandwidth_values(self):
        zero = {
            "DRAM Read Bandwidth": 0,
            "DRAM Write Bandwidth": 0,
        }
        result = mem_chart_gfx1250.plot_mem_chart(zero, chart_title=DEFAULT_TITLE)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_unavailable_hbm_bandwidth_does_not_abort_chart(self):
        metrics = mem_chart_gfx1250.get_sample_metrics()
        metrics["DRAM Read Bandwidth"] = "N/A"

        result = strip_ansi(
            mem_chart_gfx1250.plot_mem_chart(metrics, chart_title=DEFAULT_TITLE)
        )

        assert "HBM" in result
        assert "N/A" in result
        assert "896.0 GB/s" not in result

    def test_normalize_mem_chart_metrics_flat_ordered(self):
        """Metrics are flattened to panel YAML order; extras dropped; missing None."""
        raw = {"ICache Requests": 42.0, "noise_key": 99}
        norm = mem_chart_gfx1250.normalize_mem_chart_metrics(raw)
        assert list(norm.keys()) == list(mem_chart_gfx1250.MEM_CHART_PANEL_METRIC_KEYS)
        assert norm["ICache Requests"] == 42.0
        assert "noise_key" not in norm
        assert norm["DRAM Read Bandwidth"] is None


# =============================================================================
# gfx1250 DEFAULT_SAMPLE_METRICS
# =============================================================================


class TestDefaultSampleMetricsGfx1250:
    def test_keys_match_panel_metric_keys(self):
        assert set(mem_chart_gfx1250.DEFAULT_SAMPLE_METRICS) == set(
            mem_chart_gfx1250.MEM_CHART_PANEL_METRIC_KEYS
        )
        assert len(mem_chart_gfx1250.DEFAULT_SAMPLE_METRICS) == len(
            mem_chart_gfx1250.MEM_CHART_PANEL_METRIC_KEYS
        )

    def test_all_bandwidth_values_positive(self):
        for key, value in mem_chart_gfx1250.DEFAULT_SAMPLE_METRICS.items():
            if "Bandwidth" in key:
                assert value >= 0, f"{key} should be non-negative"

    def test_all_rate_values_in_range(self):
        for key, value in mem_chart_gfx1250.DEFAULT_SAMPLE_METRICS.items():
            if "Rate" in key or "Utilization" in key:
                assert 0 <= value <= 100, f"{key} should be between 0 and 100"

    def test_has_all_memory_hierarchy_levels(self):
        metrics = mem_chart_gfx1250.DEFAULT_SAMPLE_METRICS
        assert any("ICache" in k for k in metrics)
        assert any("LDS" in k for k in metrics)
        assert any("GL0" in k for k in metrics)
        assert any("GL1" in k for k in metrics)
        assert any("GL2" in k for k in metrics)
        assert any("DRAM" in k for k in metrics)

    def test_keys_match_memory_chart_panel(self):
        project_root = Path(__file__).parents[3]
        config_path = (
            project_root
            / "src/rocprof_compute_soc/analysis_configs/gfx1250"
            / "0300_memory_chart.yaml"
        )
        with config_path.open(encoding="utf-8") as config_file:
            panel_config = yaml.safe_load(config_file)["Panel Config"]

        panel_metric_keys = tuple(
            metric_name
            for table in panel_config["data source"]
            for metric_name in table["metric_table"]["metric"]
        )

        assert mem_chart_gfx1250.MEM_CHART_PANEL_METRIC_KEYS == panel_metric_keys


# =============================================================================
# Integration Tests (gfx1250)
# =============================================================================


class TestIntegrationGfx1250:
    def test_full_workflow_with_sample_data(self):
        metrics = mem_chart_gfx1250.get_sample_metrics()
        chart = mem_chart_gfx1250.plot_mem_chart(
            metrics,
            chart_title="3. Memory Chart (Normalization: per_dispatch)",
        )
        assert isinstance(chart, str)
        assert len(chart) > 100
        assert "Kernel" in chart
        assert "Legend" in chart

    def test_bandwidth_unit_consistency(self):
        metrics = {
            "GL0-GL1 Read Bandwidth": 100e9,
            "GL0-GL1 Write Bandwidth": 50e9,
            "GL2-EA Read Bandwidth": 200e9,
            "DRAM Read Bandwidth": 512e9,
            "DRAM Write Bandwidth": 384e9,
        }
        chart = mem_chart_gfx1250.plot_mem_chart(metrics, chart_title=DEFAULT_TITLE)
        assert chart.count("GB/s") >= 5


def test_chart_title_appears_as_first_line():
    chart_title = "7. Memory Chart (Normalization: per_kernel)"
    output = strip_ansi(
        mem_chart_gfx1250.plot_mem_chart(
            mem_chart_gfx1250.get_sample_metrics(),
            chart_title=chart_title,
        )
    )
    assert output.strip().splitlines()[0] == chart_title
    assert "3. Memory Chart" not in output

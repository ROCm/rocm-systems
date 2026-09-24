# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the gfx1250 memory-chart renderer."""

import re

from memory_chart.mem_chart import plot_mem_chart, strip_ansi
from tests.unit.memory_chart.conftest import (
    DEFAULT_TITLE,
    panel_yaml_metric_keys,
    sample_metrics_for_arch,
)

GFX1250_ARCH = "gfx1250"


def sample_metrics_realistic() -> dict[str, float]:
    """Realistic sample with BW in Bytes/s and rates in %."""
    metrics = dict.fromkeys(panel_yaml_metric_keys(GFX1250_ARCH), 50.0)
    for key in metrics:
        if "Bandwidth" in key:
            metrics[key] = 100e9
        elif "Rate" in key or "Utilization" in key:
            metrics[key] = 65.0
    return metrics


class TestPlotMemChartGfx1250:
    def test_returns_string(self) -> None:
        result = plot_mem_chart(
            sample_metrics_for_arch(GFX1250_ARCH),
            chart_title=DEFAULT_TITLE,
            gpu_arch=GFX1250_ARCH,
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_complete_gfx1250_architecture(self) -> None:
        output = strip_ansi(
            plot_mem_chart(
                sample_metrics_realistic(),
                chart_title=DEFAULT_TITLE,
                gpu_arch=GFX1250_ARCH,
            )
        )
        expected_components = (
            "Compute Units",
            "TCP",
            "LDS",
            "GL0",
            "SQC",
            "GL1",
            "GLARB",
            "GL2",
            "EA/DF",
            "HBM",
            "IO",
            "HDM",
            "GMI",
        )
        for component in expected_components:
            assert component in output, f"Missing gfx1250 component: {component}"

    def test_contains_directional_connectors(self) -> None:
        output = strip_ansi(
            plot_mem_chart(
                sample_metrics_for_arch(GFX1250_ARCH),
                chart_title=DEFAULT_TITLE,
                gpu_arch=GFX1250_ARCH,
            )
        )
        assert re.search(r"<(?!-+>)-{3,}", output)
        assert re.search(r"(?<![<-])-{3,}>", output)
        assert re.search(r"<-{3,}>", output)

    def test_contains_bandwidth_values(self) -> None:
        result = plot_mem_chart(
            sample_metrics_for_arch(GFX1250_ARCH),
            chart_title=DEFAULT_TITLE,
            gpu_arch=GFX1250_ARCH,
        )
        assert "GB/s" in result

    def test_empty_metrics(self) -> None:
        result = plot_mem_chart(
            {},
            chart_title=DEFAULT_TITLE,
            gpu_arch=GFX1250_ARCH,
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_partial_metrics(self) -> None:
        partial = {
            "GL0-GL1 Read Bandwidth": 50e9,
            "GL2-EA Read Bandwidth": 200e9,
        }
        result = plot_mem_chart(
            partial,
            chart_title=DEFAULT_TITLE,
            gpu_arch=GFX1250_ARCH,
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_extreme_bandwidth_values(self) -> None:
        extreme = {
            "DRAM Read Bandwidth": 10e12,
            "DRAM Write Bandwidth": 5e12,
        }
        result = plot_mem_chart(
            extreme,
            chart_title=DEFAULT_TITLE,
            gpu_arch=GFX1250_ARCH,
        )
        assert "GB/s" in result

    def test_zero_bandwidth_values(self) -> None:
        zero = {
            "DRAM Read Bandwidth": 0,
            "DRAM Write Bandwidth": 0,
        }
        result = plot_mem_chart(
            zero,
            chart_title=DEFAULT_TITLE,
            gpu_arch=GFX1250_ARCH,
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_unavailable_hbm_bandwidth_does_not_abort_chart(
        self,
    ) -> None:
        metrics = sample_metrics_for_arch(GFX1250_ARCH)
        metrics["DRAM Read Bandwidth"] = "N/A"
        result = strip_ansi(
            plot_mem_chart(
                metrics,
                chart_title=DEFAULT_TITLE,
                gpu_arch=GFX1250_ARCH,
            )
        )
        assert "HBM" in result
        assert "N/A" in result


class TestIntegrationGfx1250:
    def test_full_workflow_with_sample_data(self) -> None:
        chart = plot_mem_chart(
            sample_metrics_realistic(),
            chart_title="3. Memory Chart (Normalization: per_dispatch)",
            gpu_arch=GFX1250_ARCH,
        )
        assert isinstance(chart, str)
        assert len(chart) > 100
        assert "Compute Units" in chart
        assert "Legend" in chart

    def test_bandwidth_unit_consistency(self) -> None:
        metrics = {
            "GL0-GL1 Read Bandwidth": 100e9,
            "GL0-GL1 Write Bandwidth": 50e9,
            "GL2-EA Read Bandwidth": 200e9,
            "DRAM Read Bandwidth": 512e9,
            "DRAM Write Bandwidth": 384e9,
        }
        chart = plot_mem_chart(
            metrics,
            chart_title=DEFAULT_TITLE,
            gpu_arch=GFX1250_ARCH,
        )
        assert chart.count("GB/s") >= 5


def test_chart_title_appears_as_first_line() -> None:
    chart_title = "7. Memory Chart (Normalization: per_kernel)"
    output = strip_ansi(
        plot_mem_chart(
            sample_metrics_for_arch(GFX1250_ARCH),
            chart_title=chart_title,
            gpu_arch=GFX1250_ARCH,
        )
    )
    assert output.strip().splitlines()[0] == chart_title
    assert "3. Memory Chart" not in output

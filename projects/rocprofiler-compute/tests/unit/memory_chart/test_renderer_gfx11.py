# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the gfx11 memory-chart renderer."""

import pytest

from memory_chart.mem_chart import plot_mem_chart, strip_ansi
from tests.unit.memory_chart.conftest import (
    DEFAULT_TITLE,
    panel_yaml_metric_keys,
    sample_metrics_for_arch,
)

GFX11_ARCH = "gfx115x"


class TestPlotMemChartGfx11:
    def test_full_chart(self) -> None:
        result = plot_mem_chart(
            sample_metrics_for_arch(GFX11_ARCH),
            chart_title=DEFAULT_TITLE,
            gpu_arch=GFX11_ARCH,
        )
        clean = strip_ansi(result)
        assert len(result) > 100
        assert "3. Memory Chart" in clean

    @pytest.mark.parametrize(
        "block",
        ["GL0", "GL1", "GL2", "GCEA", "DRAM"],
    )
    def test_contains_arch_element(self, block: str) -> None:
        result = plot_mem_chart(
            sample_metrics_for_arch(GFX11_ARCH),
            chart_title=DEFAULT_TITLE,
            gpu_arch=GFX11_ARCH,
        )
        clean = strip_ansi(result)
        assert block in clean


class TestZeroVersusMissingMetrics:
    """A measured 0 must render; an absent counter must never read as 0."""

    GUARDED_LINES = [
        ("LDS Utilization", "Util 0.0%"),
        ("LDS Estimated Bandwidth", "BW 0.000 GB/s"),
    ]

    @pytest.mark.parametrize(("metric", "expected"), GUARDED_LINES)
    def test_zero_valued_metric_is_displayed(self, metric: str, expected: str) -> None:
        output = strip_ansi(
            plot_mem_chart(
                {metric: 0},
                chart_title=DEFAULT_TITLE,
                gpu_arch=GFX11_ARCH,
            )
        )
        assert expected in output

    def test_all_zero_metrics_render_no_placeholders(self) -> None:
        metrics = dict.fromkeys(panel_yaml_metric_keys(GFX11_ARCH), 0)
        output = strip_ansi(
            plot_mem_chart(
                metrics,
                chart_title=DEFAULT_TITLE,
                gpu_arch=GFX11_ARCH,
            )
        )
        assert "N/A" not in output

    def test_missing_counters_are_not_reported_as_zero(self) -> None:
        output = strip_ansi(
            plot_mem_chart(
                {},
                chart_title=DEFAULT_TITLE,
                gpu_arch=GFX11_ARCH,
            )
        )
        assert "0.0" not in output


def test_chart_title_appears_as_first_line() -> None:
    chart_title = "7. Memory Chart (Normalization: per_kernel)"
    output = strip_ansi(
        plot_mem_chart(
            sample_metrics_for_arch(GFX11_ARCH),
            chart_title=chart_title,
            gpu_arch=GFX11_ARCH,
        )
    )
    assert output.strip().splitlines()[0] == chart_title
    assert "3. Memory Chart" not in output

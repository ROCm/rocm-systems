# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Integration tests for gfx9 memory-chart panel YAML and rendering."""

import functools
from pathlib import Path

import common
import pytest
import yaml

from utils import mem_chart_gfx9

DEFAULT_TITLE = "3. Memory Chart (Normalization: per_kernel)"
ANALYSIS_CONFIGS = Path(common.SRC) / "rocprof_compute_soc" / "analysis_configs"
GFX9_ARCHITECTURES = (
    "gfx908",
    "gfx90a",
    "gfx940",
    "gfx941",
    "gfx942",
    "gfx950",
)
DISCOVERED_GFX9_ARCHITECTURES = tuple(
    path.parent.name
    for path in sorted(ANALYSIS_CONFIGS.glob("gfx9*/0300_memory_chart.yaml"))
)
GFX94X_ARCHITECTURES = frozenset({"gfx940", "gfx941", "gfx942"})
GFX9_ALWAYS_MISSING_METRIC_KEYS = frozenset({"Active CUs"})
GFX94X_MISSING_METRIC_KEYS = frozenset({"L2 Rd Lat", "L2 Wr Lat", "VL1 Lat"})
GFX9_YAML_ONLY_METRIC_KEYS = frozenset({"Active CUs (deprecated)"})


@functools.lru_cache(maxsize=None)
def panel_yaml_metric_keys(architecture: str) -> frozenset[str]:
    config_path = ANALYSIS_CONFIGS / architecture / "0300_memory_chart.yaml"
    panel_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    return frozenset(
        metric_name
        for data_source in panel_config["Panel Config"]["data source"]
        for metric_name in data_source["metric_table"]["metric"]
    )


def expected_missing_metric_keys(architecture: str) -> frozenset[str]:
    if architecture in GFX94X_ARCHITECTURES:
        return GFX9_ALWAYS_MISSING_METRIC_KEYS | GFX94X_MISSING_METRIC_KEYS
    return GFX9_ALWAYS_MISSING_METRIC_KEYS


def normalized_panel_metrics(architecture: str) -> dict[str, int | None]:
    panel_metric_keys = panel_yaml_metric_keys(architecture)
    return {
        metric_name: 42 if metric_name in panel_metric_keys else None
        for metric_name in common.GFX9_SAMPLE_METRICS
    }


class TestIntegrationGfx9:
    def test_discovers_expected_architectures(self):
        """Verify all supported gfx9 memory-chart architectures are discovered."""
        assert DISCOVERED_GFX9_ARCHITECTURES == GFX9_ARCHITECTURES

    @pytest.mark.parametrize("architecture", GFX9_ARCHITECTURES)
    def test_panel_yaml_matches_sample_metric_contract(self, architecture):
        """Verify each panel YAML matches the shared sample metric contract."""
        panel_metric_keys = panel_yaml_metric_keys(architecture)
        sample_metric_keys = frozenset(common.GFX9_SAMPLE_METRICS)
        missing_metric_keys = expected_missing_metric_keys(architecture)
        normalized_metrics = normalized_panel_metrics(architecture)

        assert sample_metric_keys - panel_metric_keys == missing_metric_keys
        assert panel_metric_keys - sample_metric_keys == GFX9_YAML_ONLY_METRIC_KEYS
        assert tuple(normalized_metrics) == tuple(common.GFX9_SAMPLE_METRICS)
        assert {
            metric_name
            for metric_name, value in normalized_metrics.items()
            if value is None
        } == missing_metric_keys

    @pytest.mark.parametrize("architecture", GFX9_ARCHITECTURES)
    def test_normalized_panel_metrics_render_expected_shape_and_placeholders(
        self, architecture
    ):
        """Verify normalized metrics render with expected shape and placeholders."""
        output = common.strip_ansi(
            mem_chart_gfx9.plot_mem_chart(
                normalized_panel_metrics(architecture),
                chart_title=DEFAULT_TITLE,
            )
        )
        expected_na_count = 3 if architecture in GFX94X_ARCHITECTURES else 0

        assert len(output.splitlines()) == 43
        assert output.count("N/A") == expected_na_count
        assert "n/a:" not in output

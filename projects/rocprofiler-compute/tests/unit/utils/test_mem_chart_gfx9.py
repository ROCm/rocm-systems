# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the gfx9 memory-chart renderer and panel contract."""

import functools
import re
from pathlib import Path

import common
import pytest
import yaml

from utils import mem_chart_gfx9
from utils.mem_chart_common import strip_ansi

DEFAULT_TITLE = "3. Memory Chart (Normalization: per_kernel)"

MEMORY_CHART_CONFIG_FILENAME = "0300_memory_chart.yaml"

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
    for path in sorted(ANALYSIS_CONFIGS.glob(f"gfx9*/{MEMORY_CHART_CONFIG_FILENAME}"))
)

GFX94X_ARCHITECTURES = frozenset({"gfx940", "gfx941", "gfx942"})

GFX94X_MISSING_METRIC_KEYS = frozenset({"L2 Rd Lat", "L2 Wr Lat", "VL1 Lat"})

GFX950_EXTRA_METRIC_KEYS = frozenset({
    "LDS Read",
    "LDS Write",
    "LDS Atomic",
    "HBM Read BW",
    "HBM Write BW",
    "HBM Atomic BW",
    "xGMI Read BW",
    "xGMI Write BW",
    "xGMI Atomic BW",
    "PCIe Read BW",
    "PCIe Write BW",
    "PCIe Atomic BW",
})

GFX950_MISSING_METRIC_KEYS = frozenset({
    "HBM Read Traffic",
    "HBM Write and Atomic Traffic",
    "Remote Read Traffic",
    "Remote Write and Atomic Traffic",
})

CHART_PANEL_TITLES = (
    "Kernel",
    "VL1D",
    "LDS",
    "sL1D",
    "L1I",
    "L2",
    "Data Fabric",
    "HBM",
)

METRICS_THAT_RENDER_NA = frozenset({
    "VL1 Hit",
    "VL1_L2 Read BW",
    "VL1_L2 Write BW",
    "VL1_L2 Atomic BW",
    "sL1D Hit",
    "sL1D_L2 Read BW",
    "IL1 Hit",
    "IL1_L2 Read BW",
    "L2 Hit",
    "L2-Fabric Read BW",
    "L2-Fabric Write and Atomic BW",
    "HBM Read Traffic",
    "HBM Write and Atomic Traffic",
    "Remote Read Traffic",
    "Remote Write and Atomic Traffic",
})

GFX9_SAMPLE_METRICS = {
    k: v if v is not None else 1
    for k, v in mem_chart_gfx9.DEFAULT_SAMPLE_METRICS.items()
}


def render_gfx9_chart(metrics, chart_title=DEFAULT_TITLE, gpu_arch=None):
    return strip_ansi(
        mem_chart_gfx9.plot_mem_chart(
            dict(metrics), chart_title=chart_title, gpu_arch=gpu_arch
        )
    )


@functools.lru_cache(maxsize=None)
def panel_yaml_metric_keys(architecture: str) -> frozenset[str]:
    config_path = ANALYSIS_CONFIGS / architecture / MEMORY_CHART_CONFIG_FILENAME
    panel_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    return frozenset(
        metric_name
        for data_source in panel_config["Panel Config"]["data source"]
        for metric_name in data_source["metric_table"]["metric"]
    )


def expected_architecture_extra_metric_keys(
    architecture: str,
) -> frozenset[str]:
    if architecture == "gfx950":
        return GFX950_EXTRA_METRIC_KEYS
    return frozenset()


def expected_architecture_missing_metric_keys(
    architecture: str,
) -> frozenset[str]:
    missing = frozenset()
    if architecture in GFX94X_ARCHITECTURES:
        missing = missing | GFX94X_MISSING_METRIC_KEYS
    if architecture == "gfx950":
        missing = missing | GFX950_MISSING_METRIC_KEYS
    return missing


def panel_yaml_metrics(architecture: str) -> dict[str, int]:
    return dict.fromkeys(panel_yaml_metric_keys(architecture), 1)


class TestPlotMemChartGfx9:
    def test_render_helper_does_not_mutate_input_metrics(self):
        metrics = {"HBM Rd": "invalid"}
        output = render_gfx9_chart(metrics)
        assert metrics == {"HBM Rd": "invalid"}
        assert "N/A" in output

    def test_full_sample_metrics_render_without_na_placeholders(self):
        output = render_gfx9_chart(GFX9_SAMPLE_METRICS)
        assert "n/a" not in output.casefold()

    @pytest.mark.parametrize("missing_metric", sorted(METRICS_THAT_RENDER_NA))
    def test_each_missing_metric_renders_one_placeholder(self, missing_metric):
        metrics = dict(GFX9_SAMPLE_METRICS)
        del metrics[missing_metric]
        output = render_gfx9_chart(metrics)
        assert output.casefold().count("n/a") == 1

    def test_partial_metrics_render_values_and_placeholders(self):
        output = render_gfx9_chart({"HBM Rd": 100, "VL1 Hit": 90})
        assert "100" in output
        assert "N/A" in output

    def test_contains_complete_cdna_architecture(self):
        output = render_gfx9_chart(GFX9_SAMPLE_METRICS)
        for panel_title in CHART_PANEL_TITLES:
            assert panel_title in output, f"Missing panel: {panel_title}"

    def test_caller_title_appears_once(self):
        chart_title = "7. Memory Chart (Normalization: per_kernel)"
        output = render_gfx9_chart(GFX9_SAMPLE_METRICS, chart_title=chart_title)
        assert output.count(chart_title) == 1
        assert DEFAULT_TITLE not in output

    def test_directional_connectors(self):
        output = render_gfx9_chart(GFX9_SAMPLE_METRICS)
        for label in ("Read", "Write", "Atomic"):
            assert label in output
        assert re.search(r"<(?!-+>)-{3,}", output)
        assert re.search(r"(?<![<-])-{3,}>", output)
        assert re.search(r"<-{3,}>", output)

    @pytest.mark.parametrize(
        ("metric_name", "metric_value", "expected_text"),
        [
            pytest.param("VL1 Hit", 92, "Hit 92.0%", id="vector-l1"),
            pytest.param("sL1D Hit", 98, "Hit 98.0%", id="scalar-l1d"),
            pytest.param("IL1 Hit", 99, "Hit 99.0%", id="instruction-l1"),
            pytest.param("L2 Hit", 85, "Hit 85.0%", id="l2"),
        ],
    )
    def test_hit_rate_renders_in_output(self, metric_name, metric_value, expected_text):
        output = render_gfx9_chart({metric_name: metric_value})
        assert expected_text in output

    def test_bandwidth_renders_human_readable(self):
        output = render_gfx9_chart({"VL1_L2 Read BW": 32e9})
        assert "32.0 GB/s" in output

    def test_empty_placeholders_do_not_render_metric_suffixes(self):
        output = render_gfx9_chart({})
        assert re.search(r"N/A[ \t]*(?:%|cycles)", output) is None

    def test_lds_util_renders(self):
        output = render_gfx9_chart({"LDS Util": 45})
        assert "Util 45.0%" in output

    def test_gfx908_no_mall_no_io(self):
        output = render_gfx9_chart(GFX9_SAMPLE_METRICS, gpu_arch="gfx908")
        assert "MALL" not in output
        assert "xGMI" not in output
        assert "PCIe" not in output

    def test_gfx940_has_mall(self):
        output = render_gfx9_chart(GFX9_SAMPLE_METRICS, gpu_arch="gfx940")
        assert "MALL" in output
        assert "xGMI" not in output
        assert "PCIe" not in output

    def test_gfx950_has_mall_and_io(self):
        metrics = dict(GFX9_SAMPLE_METRICS)
        metrics.update({
            "HBM Read BW": 100e9,
            "HBM Write BW": 50e9,
            "HBM Atomic BW": 1e9,
            "xGMI Read BW": 20e9,
            "xGMI Write BW": 10e9,
            "xGMI Atomic BW": 500e6,
            "PCIe Read BW": 15e9,
            "PCIe Write BW": 8e9,
            "PCIe Atomic BW": 200e6,
        })
        output = render_gfx9_chart(metrics, gpu_arch="gfx950")
        assert "MALL" in output
        assert "xGMI" in output
        assert "PCIe" in output


class TestPanelYamlGfx9:
    def test_discovers_expected_architectures(self):
        assert DISCOVERED_GFX9_ARCHITECTURES == GFX9_ARCHITECTURES

    @pytest.mark.parametrize("architecture", GFX9_ARCHITECTURES)
    def test_panel_yaml_keys_match_architecture_contract(self, architecture):
        reference_keys = panel_yaml_metric_keys(GFX9_ARCHITECTURES[0])
        panel_keys = panel_yaml_metric_keys(architecture)
        assert reference_keys - panel_keys == (
            expected_architecture_missing_metric_keys(architecture)
        )
        assert panel_keys - reference_keys == (
            expected_architecture_extra_metric_keys(architecture)
        )

    @pytest.mark.parametrize("architecture", GFX9_ARCHITECTURES)
    def test_panel_yaml_metrics_render_with_gpu_arch(self, architecture):
        output = render_gfx9_chart(
            panel_yaml_metrics(architecture), gpu_arch=architecture
        )
        assert len(output) > 100

    def test_panel_yaml_metrics_render_same_line_count_across_base_architectures(self):
        base_archs = [a for a in GFX9_ARCHITECTURES if a != "gfx950"]
        line_counts = {
            arch: len(
                render_gfx9_chart(panel_yaml_metrics(arch), gpu_arch=arch).splitlines()
            )
            for arch in base_archs
        }
        assert len(set(line_counts.values())) == 1, line_counts

    @pytest.mark.parametrize("architecture", GFX9_ARCHITECTURES)
    def test_panel_yaml_metrics_render_expected_placeholders(self, architecture):
        output = render_gfx9_chart(panel_yaml_metrics(architecture))
        expected_na_count = sum(
            1
            for key in METRICS_THAT_RENDER_NA
            if key not in panel_yaml_metric_keys(architecture)
        )
        assert output.count("N/A") == expected_na_count, (
            f"{architecture}: expected {expected_na_count} N/A, "
            f"got {output.count('N/A')}"
        )

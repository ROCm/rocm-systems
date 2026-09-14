# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the gfx9 memory-chart renderer and panel contract."""

import functools
import math
import re
from pathlib import Path

import common
import pytest
import yaml

from utils import mem_chart_gfx9

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

CHART_BLOCK_LABELS = (
    "Instr Buff",
    "Instr Dispatch",
    "Exec",
    "LDS",
    "Vector L1 Cache",
    "Scalar L1D Cache",
    "Instr L1 Cache",
    "L2 Cache",
    "Fabric",
    "HBM",
)

GFX9_SAMPLE_METRICS = {
    "Wavefront Occupancy": 1,
    "Wave Life": 2,
    "SALU": 3,
    "SMEM": 4,
    "VALU": 5,
    "Matrix Ops": 6,
    "VMEM": 7,
    "LDS": 8,
    "GWS": 9,
    "BR": 10,
    "Active CUs": 11,
    "Num CUs": 12,
    "VGPR": 13,
    "SGPR": 14,
    "LDS Allocation": 15,
    "Scratch Allocation": 16,
    "Wavefronts": 17,
    "Workgroups": 18,
    "LDS Req": 19,
    "LDS Util": 20,
    "LDS Latency": 21,
    "VL1 Rd": 22,
    "VL1 Wr": 23,
    "VL1 Atomic": 24,
    "VL1 Hit": 25,
    "VL1 Lat": 26,
    "VL1 Coalesce": 27,
    "VL1 Stall": 28,
    "sL1D Rd": 29,
    "sL1D Hit": 30,
    "sL1D Lat": 31,
    "IL1 Fetch": 32,
    "IL1 Hit": 33,
    "IL1 Lat": 34,
    "VL1_L2 Rd": 36,
    "VL1_L2 Wr": 37,
    "VL1_L2 Atomic": 38,
    "sL1D_L2 Rd": 39,
    "sL1D_L2 Wr": 40,
    "sL1D_L2 Atomic": 41,
    "IL1_L2 Rd": 42,
    "L2 Hit": 43,
    "L2 Rd": 44,
    "L2 Wr": 45,
    "L2 Atomic": 46,
    "L2 Rd Lat": 47,
    "L2 Wr Lat": 48,
    "Fabric_L2 Rd": 49,
    "Fabric_L2 Wr": 50,
    "Fabric_L2 Atomic": 51,
    "Fabric Rd Lat": 52,
    "Fabric Wr Lat": 53,
    "Fabric Atomic Lat": 54,
    "HBM Rd": 55,
    "HBM Wr": 56,
}


def render_gfx9_chart(metrics, chart_title=DEFAULT_TITLE):
    """Render a GFX9 memory chart without ANSI styling or input mutation."""
    return common.strip_ansi(
        mem_chart_gfx9.plot_mem_chart(dict(metrics), chart_title=chart_title)
    )


@functools.lru_cache(maxsize=None)
def panel_yaml_metric_keys(architecture: str) -> frozenset[str]:
    """Return memory-chart metric keys configured for an architecture."""
    config_path = ANALYSIS_CONFIGS / architecture / MEMORY_CHART_CONFIG_FILENAME
    panel_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    return frozenset(
        metric_name
        for data_source in panel_config["Panel Config"]["data source"]
        for metric_name in data_source["metric_table"]["metric"]
    )


def expected_architecture_missing_metric_keys(
    architecture: str,
) -> frozenset[str]:
    """Return metric keys intentionally omitted for an architecture."""
    if architecture in GFX94X_ARCHITECTURES:
        return GFX94X_MISSING_METRIC_KEYS
    return frozenset()


def panel_yaml_metrics(architecture: str) -> dict[str, int]:
    """Return sample values for an architecture's configured panel metrics."""
    return dict.fromkeys(panel_yaml_metric_keys(architecture), 1)


def chart_block_header_positions(output: str) -> dict[str, tuple[int, int]]:
    """Locate the first occurrence of each memory-chart block header."""
    lines = output.splitlines()
    positions = {}

    for block_label in CHART_BLOCK_LABELS:
        matches = [
            (row_index, line.index(block_label))
            for row_index, line in enumerate(lines)
            if block_label in line
        ]
        assert matches, f"Missing chart block label: {block_label}"
        positions[block_label] = min(matches)

    return positions


def assert_text_in_chart_block(
    output: str,
    block_label: str,
    expected_text: str,
) -> None:
    """Assert that text appears once within the requested chart block."""
    assert output.count(expected_text) == 1, (
        f"Expected one {expected_text!r} occurrence, "
        f"found {output.count(expected_text)}"
    )
    lines = output.splitlines()
    text_positions = [
        (row_index, line.index(expected_text))
        for row_index, line in enumerate(lines)
        if expected_text in line
    ]

    header_positions = chart_block_header_positions(output)
    header_row, header_column = header_positions[block_label]
    text_row, text_column = text_positions[0]
    header_columns = sorted({column for _, column in header_positions.values()})
    column_index = header_columns.index(header_column)
    left_column = (
        -math.inf
        if column_index == 0
        else (header_columns[column_index - 1] + header_column) / 2
    )
    right_column = (
        math.inf
        if column_index == len(header_columns) - 1
        else (header_column + header_columns[column_index + 1]) / 2
    )

    assert left_column <= text_column < right_column, (
        f"{expected_text!r} rendered outside the {block_label!r} column region"
    )

    stacked_header_rows = sorted(
        row for row, column in header_positions.values() if column == header_column
    )
    if len(stacked_header_rows) > 1:
        row_index = stacked_header_rows.index(header_row)
        next_header_row = (
            math.inf
            if row_index == len(stacked_header_rows) - 1
            else stacked_header_rows[row_index + 1]
        )
        assert header_row <= text_row < next_header_row, (
            f"{expected_text!r} rendered outside the {block_label!r} row region"
        )


def assert_text_below_chart_label(
    output: str,
    field_label: str,
    expected_text: str,
) -> None:
    """Assert that text appears directly below the requested chart label."""
    assert output.count(field_label) == 1
    assert output.count(expected_text) == 1
    lines = output.splitlines()
    label_row = next(row for row, line in enumerate(lines) if field_label in line)
    label_column = lines[label_row].index(field_label)
    text_row = next(row for row, line in enumerate(lines) if expected_text in line)
    text_column = lines[text_row].index(expected_text)

    assert text_row == label_row + 1
    assert label_column <= text_column <= label_column + len(field_label)


class TestMakeFormatSpecGfx9:
    """Tests for GFX9 format-specification generation."""

    @pytest.mark.parametrize(
        ("value", "expected_spec"),
        [
            pytest.param(6, ">6", id="integer"),
            pytest.param(123456789, ">123456789", id="large-integer"),
            pytest.param(0, ">0", id="zero"),
            pytest.param(6.0, ">6.0f", id="whole-number-float"),
            pytest.param(3.14, ">3.14f", id="decimal-float"),
            pytest.param(-42, ">-42", id="negative-integer"),
            pytest.param(-3.14, ">3.14f", id="negative-float"),
            pytest.param(1e20, ">1e+20f", id="scientific-float"),
            pytest.param(math.nan, ">nanf", id="nan"),
            pytest.param(math.inf, ">inff", id="infinity"),
        ],
    )
    def test_default_alignment_specs(self, value, expected_spec):
        """Verify default right-aligned format specifications for numeric samples."""
        assert mem_chart_gfx9.make_format_spec(value) == expected_spec

    @pytest.mark.parametrize(
        ("alignment", "expected_spec"),
        [
            pytest.param("<", "<12.34f", id="left"),
            pytest.param(">", ">12.34f", id="right"),
            pytest.param("^", "^12.34f", id="center"),
        ],
    )
    def test_explicit_alignment_specs(self, alignment, expected_spec):
        """Verify explicit alignments are preserved in format specifications."""
        assert mem_chart_gfx9.make_format_spec(12.34, alignment) == expected_spec

    def test_invalid_alignment_raises(self):
        """Verify unsupported alignment markers raise ValueError."""
        with pytest.raises(ValueError) as error:
            mem_chart_gfx9.make_format_spec(12.34, "@")

        assert str(error.value) == "align must be one of '<', '>', or '^'"


class TestIsValueValidGfx9:
    """Tests for GFX9 metric-value validation."""

    @pytest.mark.parametrize(
        ("value", "expected_validity"),
        [
            pytest.param(5, True, id="integer"),
            pytest.param(5.0, True, id="float"),
            pytest.param(0, True, id="zero"),
            pytest.param(True, True, id="boolean"),
            pytest.param(None, False, id="none"),
            pytest.param("abc", False, id="string"),
            pytest.param(math.nan, True, id="nan"),
            pytest.param(math.inf, True, id="infinity"),
        ],
    )
    def test_value_validity(self, value, expected_validity):
        """Verify which metric value types are considered valid."""
        assert mem_chart_gfx9.is_value_valid(value) is expected_validity


class TestFormatTextGfx9:
    """Tests for GFX9 metric-text formatting."""

    def test_basic_key_and_value(self):
        """Verify basic key-and-value formatting with the default separator."""
        result = mem_chart_gfx9.format_text(
            85.5,
            key="Util",
            value_step_prec_rightalign=4.0,
        )

        assert result == "Util:   86"

    @pytest.mark.parametrize(
        ("value", "format_sample", "expected_text"),
        [
            pytest.param(85.5, 4.1, "85.5 %", id="valid-value-appends-suffix"),
            pytest.param(None, 5.1, "  N/A", id="na-suppresses-suffix"),
        ],
    )
    def test_unit_suffix_rules(self, value, format_sample, expected_text):
        """Verify unit suffixes appear only for valid values."""
        result = mem_chart_gfx9.format_text(
            value,
            post_description_with_space=" %",
            value_step_prec_rightalign=format_sample,
        )

        assert result == expected_text

    @pytest.mark.parametrize(
        "invalid_value",
        [
            pytest.param(None, id="none"),
            pytest.param("abc", id="string"),
        ],
    )
    def test_invalid_values_format_as_na(self, invalid_value):
        """Verify invalid metric values render as N/A."""
        result = mem_chart_gfx9.format_text(
            invalid_value,
            key="Util",
            value_step_prec_rightalign=5.1,
        )

        assert result == "Util:   N/A"

    def test_precision_comes_from_sample(self):
        """Verify the format sample controls decimal precision."""
        result = mem_chart_gfx9.format_text(
            1.234,
            value_step_prec_rightalign=6.2,
        )

        assert result == "  1.23"

    def test_custom_separator_keeps_string_key_unformatted(self):
        """Verify custom separators bypass string-key width formatting."""
        result = mem_chart_gfx9.format_text(
            7,
            key="LDS",
            mark_between="=",
            value_step_prec_rightalign=0,
            key_step_prec_leftalign=10,
            key_align="^",
        )

        assert result == "LDS=7"

    def test_numeric_key_alignment(self):
        """Verify numeric keys honor the configured field width."""
        result = mem_chart_gfx9.format_text(
            7,
            key=3,
            value_step_prec_rightalign=2,
            key_step_prec_leftalign=4,
        )

        assert result == "3   :  7"

    @pytest.mark.parametrize(
        ("key_alignment", "value_alignment", "expected_text"),
        [
            pytest.param(">", "<", "  3: 7  ", id="right-key-left-value"),
            pytest.param("^", "^", " 3 :  7 ", id="centered"),
        ],
    )
    def test_alternate_key_and_value_alignments(
        self,
        key_alignment,
        value_alignment,
        expected_text,
    ):
        """Verify numeric keys and values honor alternate alignments."""
        result = mem_chart_gfx9.format_text(
            7,
            key=3,
            value_step_prec_rightalign=3,
            key_step_prec_leftalign=3,
            key_align=key_alignment,
            value_align=value_alignment,
        )

        assert result == expected_text

    @pytest.mark.parametrize(
        ("value", "expected_text"),
        [
            pytest.param(1.234, "1.234", id="numeric-value"),
            pytest.param(None, "N/A", id="na-value"),
        ],
    )
    def test_zero_width(self, value, expected_text):
        """Verify zero-width formatting emits unpadded values and N/A."""
        assert (
            mem_chart_gfx9.format_text(value, value_step_prec_rightalign=0)
            == expected_text
        )

    @pytest.mark.parametrize(
        "format_sample",
        [
            pytest.param(1e20, id="scientific-float"),
            pytest.param(math.nan, id="nan"),
            pytest.param(math.inf, id="infinity"),
        ],
    )
    def test_unusable_sample_specs_raise(self, format_sample):
        """Verify scientific and non-finite format samples raise ValueError."""
        with pytest.raises(ValueError, match="Invalid format specifier"):
            mem_chart_gfx9.format_text(
                1,
                value_step_prec_rightalign=format_sample,
            )

    @pytest.mark.parametrize(
        ("value", "expected_text"),
        [
            pytest.param(math.nan, "nan", id="nan"),
            pytest.param(math.inf, "inf", id="infinity"),
        ],
    )
    def test_non_finite_values_are_formatted(self, value, expected_text):
        """Verify non-finite metric values render as standard text."""
        assert mem_chart_gfx9.format_text(value) == expected_text


# =============================================================================
# Tests for plot_mem_chart function (gfx9)
# =============================================================================


class TestPlotMemChartGfx9:
    """Tests for gfx9 plot_mem_chart - CDNA memory chart generation."""

    def test_render_helper_does_not_mutate_input_metrics(self):
        """Protect callers from the renderer's top-level value normalization."""
        metrics = {"HBM Rd": "invalid"}

        output = render_gfx9_chart(metrics)

        assert metrics == {"HBM Rd": "invalid"}
        assert "N/A" in output

    def test_full_sample_metrics_render_without_na_placeholders(self):
        """Render complete gfx9 metrics without N/A placeholders."""
        output = render_gfx9_chart(GFX9_SAMPLE_METRICS)
        assert isinstance(output, str)
        assert len(output) > 0
        assert "n/a" not in output.casefold()

    @pytest.mark.parametrize("missing_metric", GFX9_SAMPLE_METRICS)
    def test_each_missing_metric_renders_one_placeholder(self, missing_metric):
        """Render one placeholder for each omitted gfx9 metric."""
        metrics = dict(GFX9_SAMPLE_METRICS)
        del metrics[missing_metric]

        output = render_gfx9_chart(metrics)
        assert isinstance(output, str)
        assert len(output) > 0
        assert output.casefold().count("n/a") == 1

    def test_partial_metrics_render_values_and_placeholders(self):
        """Render supplied gfx9 values and placeholders for missing metrics."""
        partial = {"HBM Rd": 100}
        output = render_gfx9_chart(partial)
        assert isinstance(output, str)
        assert len(output) > 0
        assert "Rd:  100" in output
        assert "N/A" in output

    def test_contains_complete_cdna_architecture(self):
        """CDNA output contains every component enabled by the renderer."""
        output = render_gfx9_chart(GFX9_SAMPLE_METRICS)
        expected_components = (
            "Instr Buff",
            "Instr Dispatch",
            "Exec",
            "LDS",
            "Vector L1 Cache",
            "Scalar L1D Cache",
            "Instr L1 Cache",
            "L2 Cache",
            "xGMI/PCIe",
            "Fabric",
            "HBM",
        )

        for component in expected_components:
            assert component in output, f"Missing CDNA component: {component}"

        assert re.search(r"(?<!x)GMI(?!/PCIe)", output)

    def test_caller_title_appears_once_outside_chart(self):
        """Print a caller-supplied title once outside the chart."""
        chart_title = "7. Memory Chart (Normalization: per_kernel)"
        output = render_gfx9_chart(
            GFX9_SAMPLE_METRICS,
            chart_title=chart_title,
        )
        output_lines = output.splitlines()

        assert output_lines[0] == chart_title
        assert DEFAULT_TITLE not in output
        assert chart_title not in "\n".join(output_lines[1:])

    def test_gfx9_chart_contains_directional_connectors(self):
        """Render the expected CDNA directional connectors."""
        output = render_gfx9_chart(GFX9_SAMPLE_METRICS)
        output_lines = output.splitlines()

        assert "Instr Buff" in output_lines[4]

        for connector_label in ("Req", "Fetch", "Rd", "Wr", "Atomic"):
            assert connector_label in output

        assert re.search(r"<(?!-+>)-{3,}", output)
        assert re.search(r"(?<![<-])-{3,}>", output)
        assert re.search(r"<-{3,}>", output)

    @pytest.mark.parametrize(
        (
            "metric_name",
            "metric_value",
            "block_label",
            "expected_text",
        ),
        [
            pytest.param(
                "SALU",
                7101,
                "Instr Dispatch",
                "SALU: 7101",
                id="salu-instruction",
            ),
            pytest.param(
                "SMEM",
                7102,
                "Instr Dispatch",
                "SMEM: 7102",
                id="smem-instruction",
            ),
            pytest.param(
                "VGPR",
                7201,
                "Exec",
                "VGPRs:  7201",
                id="vgpr-allocation",
            ),
            pytest.param(
                "SGPR",
                7202,
                "Exec",
                "SGPRs:  7202",
                id="sgpr-allocation",
            ),
            pytest.param(
                "LDS Req",
                1234,
                "LDS",
                "Req: 1234",
                id="lds-request",
            ),
            pytest.param(
                "LDS Latency",
                321,
                "LDS",
                "Lat:    321 cycles",
                id="lds-latency",
            ),
            pytest.param(
                "L2 Hit",
                87,
                "L2 Cache",
                "Hit:     87 %",
                id="l2-hit-rate",
            ),
            pytest.param(
                "L2 Rd Lat",
                2468,
                "L2 Cache",
                "Rd:   2468",
                id="l2-read-latency",
            ),
            pytest.param(
                "L2 Wr Lat",
                1357,
                "L2 Cache",
                "Wr:   1357",
                id="l2-write-latency",
            ),
            pytest.param(
                "VL1 Rd",
                7401,
                "Vector L1 Cache",
                "Rd: 7401",
                id="vector-l1-read",
            ),
            pytest.param(
                "VL1 Wr",
                7402,
                "Vector L1 Cache",
                "Wr: 7402",
                id="vector-l1-write",
            ),
            pytest.param(
                "VL1 Hit",
                2581,
                "Vector L1 Cache",
                "Hit:   2581 %",
                id="vector-l1-hit-rate",
            ),
            pytest.param(
                "sL1D Lat",
                3692,
                "Scalar L1D Cache",
                "Lat:   3692 cycles",
                id="scalar-l1d-latency",
            ),
            pytest.param(
                "IL1 Hit",
                4703,
                "Instr L1 Cache",
                "Hit:   4703 %",
                id="instruction-l1-hit-rate",
            ),
            pytest.param(
                "Fabric Rd Lat",
                5814,
                "Fabric",
                "Rd:   5814",
                id="fabric-read-latency",
            ),
            pytest.param(
                "HBM Rd",
                6925,
                "HBM",
                "Rd: 6925",
                id="hbm-read-bandwidth",
            ),
        ],
    )
    def test_metric_routes_to_expected_chart_region(
        self,
        metric_name,
        metric_value,
        block_label,
        expected_text,
    ):
        """Place each gfx9 metric in the expected chart region."""
        output = render_gfx9_chart({metric_name: metric_value})

        assert_text_in_chart_block(output, block_label, expected_text)

    @pytest.mark.parametrize(
        ("metric_name", "metric_value", "field_label"),
        [
            pytest.param("Wavefronts", 7301, "Wavefronts:", id="wavefront-count"),
            pytest.param("Workgroups", 7302, "Workgroups:", id="workgroup-count"),
        ],
    )
    def test_exec_count_routes_below_expected_label(
        self,
        metric_name,
        metric_value,
        field_label,
    ):
        """Place unlabeled execution counts below their semantic field labels."""
        expected_text = str(metric_value)
        output = render_gfx9_chart({metric_name: metric_value})

        assert_text_in_chart_block(output, "Exec", expected_text)
        assert_text_below_chart_label(output, field_label, expected_text)

    def test_empty_placeholders_do_not_render_metric_suffixes(self):
        """Omit percentage and cycle suffixes from N/A placeholders."""
        output = render_gfx9_chart({})

        assert re.search(r"N/A[ \t]*(?:%|cycles)", output) is None


class TestPanelYamlGfx9:
    """Tests for gfx9 panel YAML and renderer contracts."""

    def test_discovers_expected_architectures(self):
        """Verify all supported gfx9 memory-chart architectures are discovered."""
        assert DISCOVERED_GFX9_ARCHITECTURES == GFX9_ARCHITECTURES

    @pytest.mark.parametrize("architecture", GFX9_ARCHITECTURES)
    def test_panel_yaml_keys_match_architecture_contract(self, architecture):
        """Verify each panel YAML differs only by architecture-specific metrics."""
        reference_keys = panel_yaml_metric_keys(GFX9_ARCHITECTURES[0])
        panel_keys = panel_yaml_metric_keys(architecture)

        assert reference_keys - panel_keys == (
            expected_architecture_missing_metric_keys(architecture)
        )
        assert panel_keys - reference_keys == frozenset()

    def test_panel_yaml_metrics_render_same_line_count_across_architectures(self):
        """Verify all gfx9 panel inputs render the same architecture shape."""
        line_counts = {
            architecture: len(
                render_gfx9_chart(panel_yaml_metrics(architecture)).splitlines()
            )
            for architecture in GFX9_ARCHITECTURES
        }

        assert len(set(line_counts.values())) == 1, line_counts

    @pytest.mark.parametrize("architecture", GFX9_ARCHITECTURES)
    def test_panel_yaml_metrics_render_expected_placeholders(self, architecture):
        """Verify omitted architecture metrics render uppercase placeholders."""
        output = render_gfx9_chart(panel_yaml_metrics(architecture))

        assert output.count("N/A") == len(
            expected_architecture_missing_metric_keys(architecture)
        )

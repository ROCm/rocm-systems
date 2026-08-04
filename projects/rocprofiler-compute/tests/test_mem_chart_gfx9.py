# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the gfx9 memory-chart renderer and formatting helpers."""

import math
import re

import common
import pytest

from utils import mem_chart_gfx9

DEFAULT_TITLE = "3. Memory Chart (Normalization: per_kernel)"


def render_gfx9_chart(metrics):
    return common.strip_ansi(
        mem_chart_gfx9.plot_mem_chart(metrics, chart_title=DEFAULT_TITLE)
    )


class TestMakeFormatSpecGfx9:
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

    def test_full_sample_metrics_render_without_na_placeholders(self):
        """Render complete gfx9 metrics without N/A placeholders."""
        output = render_gfx9_chart(common.GFX9_SAMPLE_METRICS)
        assert isinstance(output, str)
        assert len(output) > 0
        assert "N/A" not in output

    def test_empty_metrics_render_expected_placeholders(self):
        """Render 54 N/A placeholders when no gfx9 metrics are provided."""
        output = render_gfx9_chart({})
        assert isinstance(output, str)
        assert len(output) > 0

        # The renderer consumes 55 keys but emits 54 uppercase placeholders because
        # missing Active CUs/Num CUs render together as "n/a: N/A".
        assert output.count("N/A") == 54

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
        output = render_gfx9_chart(common.GFX9_SAMPLE_METRICS)
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

    def test_gfx9_heading_outside_chart_and_directional_connectors(self):
        """CDNA output prints the heading once outside the chart, with connectors."""
        output = render_gfx9_chart(common.GFX9_SAMPLE_METRICS)
        output_lines = output.splitlines()

        assert output_lines[0] == DEFAULT_TITLE
        assert "3. Memory Chart" not in "\n".join(output_lines[1:])
        assert "Instr Buff" in output_lines[4]
        assert output.count("(Normalization: per_kernel)") == 1

        for connector_label in ("Req", "Fetch", "Rd", "Wr", "Atomic"):
            assert connector_label in output

        assert re.search(r"<(?!-+>)-{3,}", output)
        assert re.search(r"(?<![<-])-{3,}>", output)
        assert re.search(r"<-{3,}>", output)

    @pytest.mark.parametrize(
        (
            "metric_name",
            "metric_value",
            "expected_text",
            "expected_row_index",
            "expected_column_slice",
        ),
        [
            pytest.param(
                "SALU",
                7101,
                "SALU: 7101",
                6,
                slice(38, 58),
                id="salu-instruction",
            ),
            pytest.param(
                "SMEM",
                7102,
                "SMEM: 7102",
                9,
                slice(38, 58),
                id="smem-instruction",
            ),
            pytest.param(
                "VGPR",
                7201,
                "RVGPRseq:  7201",
                11,
                slice(58, 78),
                id="vgpr-allocation",
            ),
            pytest.param(
                "SGPR",
                7202,
                "SGPRs:  7202",
                14,
                slice(58, 78),
                id="sgpr-allocation",
            ),
            pytest.param(
                "Wavefronts",
                7301,
                "7301",
                28,
                slice(58, 78),
                id="wavefront-count",
            ),
            pytest.param(
                "Workgroups",
                7302,
                "7302",
                32,
                slice(58, 78),
                id="workgroup-count",
            ),
            pytest.param(
                "LDS Req",
                1234,
                "Req: 1234",
                7,
                slice(78, 95),
                id="lds-request",
            ),
            pytest.param(
                "LDS Latency",
                321,
                "Lat:    321 cycles",
                9,
                slice(95, 122),
                id="lds-latency",
            ),
            pytest.param(
                "L2 Hit",
                87,
                "Hit:     87 %",
                9,
                slice(140, 165),
                id="l2-hit-rate",
            ),
            pytest.param(
                "L2 Rd Lat",
                2468,
                "Rd:   2468",
                27,
                slice(140, 165),
                id="l2-read-latency",
            ),
            pytest.param(
                "L2 Wr Lat",
                1357,
                "Wr:   1357",
                29,
                slice(140, 165),
                id="l2-write-latency",
            ),
            pytest.param(
                "VL1 Rd",
                7401,
                "Rd: 7401",
                15,
                slice(78, 95),
                id="vector-l1-read",
            ),
            pytest.param(
                "VL1 Wr",
                7402,
                "Wr: 7402",
                17,
                slice(78, 95),
                id="vector-l1-write",
            ),
            pytest.param(
                "VL1 Hit",
                2581,
                "Hit:   2581 %",
                15,
                slice(95, 122),
                id="vector-l1-hit-rate",
            ),
            pytest.param(
                "sL1D Lat",
                3692,
                "Lat:   3692 cycles",
                29,
                slice(95, 122),
                id="scalar-l1d-latency",
            ),
            pytest.param(
                "IL1 Hit",
                4703,
                "Hit:   4703 %",
                35,
                slice(95, 122),
                id="instruction-l1-hit-rate",
            ),
            pytest.param(
                "Fabric Rd Lat",
                5814,
                "Rd:   5814",
                20,
                slice(183, 207),
                id="fabric-read-latency",
            ),
            pytest.param(
                "HBM Rd",
                6925,
                "Rd: 6925",
                18,
                slice(207, 221),
                id="hbm-read-bandwidth",
            ),
        ],
    )
    def test_metric_routes_to_expected_chart_region(
        self,
        metric_name,
        metric_value,
        expected_text,
        expected_row_index,
        expected_column_slice,
    ):
        """Place each gfx9 metric in the expected chart region."""
        output_lines = render_gfx9_chart({metric_name: metric_value}).splitlines()

        assert expected_text in output_lines[expected_row_index][expected_column_slice]

    def test_empty_placeholders_do_not_render_metric_suffixes(self):
        """Omit percentage and cycle suffixes from N/A placeholders."""
        output = render_gfx9_chart({})

        assert re.search(r"N/A[ \t]*(?:%|cycles)", output) is None

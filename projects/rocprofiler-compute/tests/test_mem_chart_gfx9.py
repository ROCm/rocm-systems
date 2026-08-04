# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for gfx9 memory-chart formatting helpers."""

import math

import pytest

from utils import mem_chart_gfx9


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

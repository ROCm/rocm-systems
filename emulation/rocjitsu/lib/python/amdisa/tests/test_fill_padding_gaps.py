# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for _fill_padding_gaps."""

import pytest

from amdisa.gpuisa import MicrocodeField
from amdisa.parser import _fill_padding_gaps


def _field(name: str, bit_cnt: int, bit_offset: int) -> MicrocodeField:
    return MicrocodeField(name, bit_cnt, bit_offset)


def _names(fields: list[MicrocodeField]) -> list[str]:
    return [f.name for f in fields]


def _total_bits(fields: list[MicrocodeField]) -> int:
    return sum(f.bit_cnt for f in fields)


class TestFillPaddingGaps:
    def test_no_gaps_returns_empty(self):
        """When all bits are accounted for, no padding is synthesized."""
        fields = [_field('op', 8, 0), _field('imm', 8, 8)]
        pads = _fill_padding_gaps(fields, 16)
        assert pads == []

    def test_gap_before_first_field(self):
        """A gap at bit 0 synthesizes a leading pad; no trailing pad when field fills the rest."""
        fields = [_field('op', 4, 4)]
        pads = _fill_padding_gaps(fields, 8)
        assert len(pads) == 1
        assert pads[0].bit_offset == 0
        assert pads[0].bit_cnt == 4

    def test_gap_before_first_field_only(self):
        """Single leading gap: field starts at bit 2, total width 6."""
        fields = [_field('op', 4, 2)]
        pads = _fill_padding_gaps(fields, 6)
        assert len(pads) == 1
        assert pads[0].bit_offset == 0
        assert pads[0].bit_cnt == 2
        assert pads[0].name == 'pad_0_1'

    def test_trailing_gap_after_last_field(self):
        """A gap after the last field synthesizes a trailing pad."""
        fields = [_field('op', 4, 0)]
        pads = _fill_padding_gaps(fields, 8)
        assert len(pads) == 1
        assert pads[0].bit_offset == 4
        assert pads[0].bit_cnt == 4
        assert pads[0].name == 'pad_4_7'

    def test_single_bit_pad_name_has_no_underscore_suffix(self):
        """Single-bit pads use 'pad_N' (no range suffix)."""
        fields = [_field('op', 1, 1)]
        pads = _fill_padding_gaps(fields, 2)
        assert len(pads) == 1
        assert pads[0].name == 'pad_0'
        assert pads[0].bit_cnt == 1

    def test_gap_between_fields(self):
        """A gap between two fields synthesizes an interior pad."""
        fields = [
            _field('a', 4, 0),
            _field('b', 4, 8),
        ]
        pads = _fill_padding_gaps(fields, 12)
        assert len(pads) == 1
        mid = pads[0]
        assert mid.bit_offset == 4
        assert mid.bit_cnt == 4
        assert mid.name == 'pad_4_7'

    def test_multiple_gaps(self):
        """Multiple gaps each synthesize a separate pad entry."""
        fields = [
            _field('a', 2, 2),
            _field('b', 2, 6),
        ]
        pads = _fill_padding_gaps(fields, 10)
        assert len(pads) == 3
        offsets = {p.bit_offset for p in pads}
        assert 0 in offsets  # leading gap
        assert 4 in offsets  # middle gap
        assert 8 in offsets  # trailing gap

    def test_returned_pads_plus_original_cover_full_width(self):
        """Original fields + synthesized pads together cover 100% of bits."""
        fields = [
            _field('enc', 6, 0),
            _field('op', 8, 8),
        ]
        pads = _fill_padding_gaps(fields, 32)
        total = _total_bits(fields) + _total_bits(pads)
        assert total == 32

    def test_32bit_encoding_typical(self):
        """Simulated 32-bit encoding with op and imm fields, gap in middle."""
        fields = [
            _field('enc', 9, 23),
            _field('op', 8, 8),
            _field('imm', 8, 0),
        ]
        sorted_fields = sorted(fields, key=lambda f: f.bit_offset)
        pads = _fill_padding_gaps(sorted_fields, 32)
        total = _total_bits(sorted_fields) + _total_bits(pads)
        assert total == 32

# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for _parse_enc_id_masks."""

from amdisa.parser import _parse_enc_id_masks


class TestParseEncIdMasks:
    def test_single_run_no_truncation(self):
        """Single run of 1s, enc field fits within max_enc_bits."""
        # 9-bit enc at bits [0,9), 8-bit op immediately after
        mask = '111111111' + '0' * 23
        flat_enc, op_mask, dont_care = _parse_enc_id_masks(mask, 9, 9, 8)
        assert flat_enc == (0, 9)
        assert op_mask == (9, 17)
        assert dont_care == 0

    def test_single_run_truncated(self):
        """When the single run exceeds max_enc_bits, flat_enc is truncated."""
        mask = '1' * 12 + '0' * 20
        flat_enc, op_mask, dont_care = _parse_enc_id_masks(mask, 9, 9, 8)
        assert flat_enc == (0, 9)
        assert op_mask == (9, 17)
        assert dont_care == 0

    def test_two_runs(self):
        """Two separate runs of 1s: enc from run[0], op from run[1]."""
        # enc at [0,6), gap at [6,8), op at [8,16)
        mask = '111111' + '00' + '11111111' + '0' * 16
        flat_enc, op_mask, dont_care = _parse_enc_id_masks(mask, 6, 6, 8)
        assert flat_enc == (0, 6)
        assert op_mask == (8, 16)
        assert dont_care == 0

    def test_dont_care_bits(self):
        """dont_care_bits = max_enc_bits - width(flat_enc)."""
        mask = '111100' + '0' * 26
        flat_enc, op_mask, dont_care = _parse_enc_id_masks(mask, 6, 4, 8)
        assert flat_enc == (0, 4)
        assert dont_care == 2  # 6 - 4

    def test_dont_care_zero_when_enc_fills_max(self):
        """dont_care_bits is 0 when flat_enc uses all of max_enc_bits."""
        mask = '111111' + '0' * 26
        flat_enc, op_mask, dont_care = _parse_enc_id_masks(mask, 6, 6, 8)
        assert dont_care == 0

    def test_op_mask_length_matches_op_field_bit_cnt(self):
        """op_mask width always equals op_field_bit_cnt for single-run masks."""
        mask = '111111111' + '0' * 23
        _, op_mask, _ = _parse_enc_id_masks(mask, 9, 9, 8)
        assert op_mask[1] - op_mask[0] == 8

    def test_two_run_op_mask_length_matches(self):
        """op_mask width always equals op_field_bit_cnt for two-run masks."""
        mask = '111111' + '00' + '11111111' + '0' * 16
        _, op_mask, _ = _parse_enc_id_masks(mask, 6, 6, 8)
        assert op_mask[1] - op_mask[0] == 8

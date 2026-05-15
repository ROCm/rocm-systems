# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from amdisa.layout_catalog import (
    LAYOUT_CATALOG,
    MFMA_F32_16X16X16_F16,
    MFMA_F32_16X16X32_F16,
    MFMA_F32_32X32X8_F16,
    MFMA_I32_16X16X32_I8,
    WMMA_F32_16X16X16_F16,
    WMMA_I32_16X16X32_I8,
    compute_xor_mask,
    derive_layout_descriptor,
    find_descriptor,
)


class TestMatrixLayoutDescriptor:
    def test_mfma_16x16_row_group_lanes(self):
        assert MFMA_F32_16X16X16_F16.row_group_lanes == (0, 16, 32, 48)

    def test_wmma_16x16_row_group_lanes(self):
        assert WMMA_F32_16X16X16_F16.row_group_lanes == (0, 32, 16, 48)

    def test_mfma_32x32_row_group_lanes(self):
        assert MFMA_F32_32X32X8_F16.row_group_lanes == (0, 32)

    def test_lane_for_row_mfma(self):
        d = MFMA_F32_16X16X16_F16
        assert d.lane_for_row(0) == 0
        assert d.lane_for_row(4) == 16
        assert d.lane_for_row(8) == 32
        assert d.lane_for_row(12) == 48

    def test_lane_for_row_wmma(self):
        d = WMMA_F32_16X16X16_F16
        assert d.lane_for_row(0) == 0
        assert d.lane_for_row(4) == 32
        assert d.lane_for_row(8) == 16
        assert d.lane_for_row(12) == 48

    def test_num_row_groups(self):
        assert MFMA_F32_16X16X16_F16.num_row_groups == 4
        assert MFMA_F32_32X32X8_F16.num_row_groups == 2


class TestComputeXorMask:
    def test_mfma_to_wmma_16x16(self):
        result = compute_xor_mask(MFMA_F32_16X16X16_F16, WMMA_F32_16X16X16_F16)
        assert result is not None
        xor_byte, start, end = result
        assert xor_byte == 192
        assert start == 16
        assert end == 48

    def test_identity_returns_none(self):
        assert compute_xor_mask(
            MFMA_F32_16X16X16_F16, MFMA_F32_16X16X16_F16) is None

    def test_symmetric(self):
        fwd = compute_xor_mask(MFMA_F32_16X16X16_F16, WMMA_F32_16X16X16_F16)
        rev = compute_xor_mask(WMMA_F32_16X16X16_F16, MFMA_F32_16X16X16_F16)
        assert fwd == rev


class TestFindDescriptor:
    def test_exact_match(self):
        d = find_descriptor('V_MFMA_F32_16X16X16_BF16')
        assert d is not None
        assert d.kind == 'mfma'

    def test_wildcard_match(self):
        d = find_descriptor('V_MFMA_F32_16X16X16_F16')
        assert d is not None
        assert d.kind == 'mfma'
        assert d.m == 16

    def test_wmma_match(self):
        d = find_descriptor('V_WMMA_F32_16X16X16_F16')
        assert d is not None
        assert d.kind == 'wmma'

    def test_unknown_returns_none(self):
        assert find_descriptor('V_SOME_UNKNOWN_INST') is None


class TestDeriveLayoutDescriptor:
    def test_mfma_f32_16x16x16_f16(self):
        d = derive_layout_descriptor('V_MFMA_F32_16X16X16_F16')
        assert d is not None
        assert d.row_group_lanes == MFMA_F32_16X16X16_F16.row_group_lanes
        assert d.dst_vgprs == MFMA_F32_16X16X16_F16.dst_vgprs
        assert d.kind == 'mfma'

    def test_mfma_f32_32x32x8_f16(self):
        d = derive_layout_descriptor('V_MFMA_F32_32X32X8_F16')
        assert d is not None
        assert d.row_group_lanes == MFMA_F32_32X32X8_F16.row_group_lanes
        assert d.dst_vgprs == MFMA_F32_32X32X8_F16.dst_vgprs

    def test_mfma_f32_16x16x32_f16(self):
        d = derive_layout_descriptor('V_MFMA_F32_16X16X32_F16')
        assert d is not None
        assert d.row_group_lanes == MFMA_F32_16X16X32_F16.row_group_lanes

    def test_mfma_i32_16x16x32_i8(self):
        d = derive_layout_descriptor('V_MFMA_I32_16X16X32_I8')
        assert d is not None
        assert d.row_group_lanes == MFMA_I32_16X16X32_I8.row_group_lanes

    def test_wmma_f32_16x16x16_f16(self):
        d = derive_layout_descriptor('V_WMMA_F32_16X16X16_F16')
        assert d is not None
        assert d.row_group_lanes == WMMA_F32_16X16X16_F16.row_group_lanes
        assert d.kind == 'wmma'

    def test_wmma_i32_16x16x32_i8(self):
        d = derive_layout_descriptor('V_WMMA_I32_16X16X32_I8')
        assert d is not None
        assert d.row_group_lanes == WMMA_I32_16X16X32_I8.row_group_lanes

    def test_auto_derived_xor_matches_catalog(self):
        mfma = derive_layout_descriptor('V_MFMA_F32_16X16X16_F16')
        wmma = derive_layout_descriptor('V_WMMA_F32_16X16X16_F16')
        assert mfma is not None and wmma is not None
        result = compute_xor_mask(mfma, wmma)
        assert result == (192, 16, 48)

    def test_unknown_returns_none(self):
        assert derive_layout_descriptor('V_ADD_F32') is None

    def test_bf16_mfma(self):
        d = derive_layout_descriptor('V_MFMA_F32_16X16X16_BF16')
        assert d is not None
        assert d.src_elem_bits == 16
        assert d.dst_elem_bits == 32


class TestCatalogCompleteness:
    def test_catalog_has_entries(self):
        assert len(LAYOUT_CATALOG) >= 10

    def test_all_entries_have_valid_lanes(self):
        for name, desc in LAYOUT_CATALOG.items():
            for lane in desc.row_group_lanes:
                assert 0 <= lane < desc.wave_size, f'{name}: invalid lane {lane}'

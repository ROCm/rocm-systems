# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for packed execute code generation."""

from amdisa.codegen.execute.packed import (
    gen_dot2,
    gen_dot2_true16,
    gen_dot4,
    gen_dot8,
    gen_mad_mix_bf16,
    gen_mad_mix_f32,
    gen_mad_mix_lo_hi,
    gen_pk_binop,
    gen_pk_binop_f32,
    gen_pk_ternary,
)
from amdisa.codegen.execute.simd_codegen import vop3p_local_simd_probe_line


def test_dot4_iu8_uses_operand_signedness_modifiers():
    cpp = gen_dot4(['vdst'], ['src0', 'src1', 'src2'], 'dot4_i32_iu8')

    assert 'src0_signed = (inst_.neg & 0x1u) != 0' in cpp
    assert 'src1_signed = (inst_.neg & 0x2u) != 0' in cpp
    assert 'static_cast<int8_t>(raw_a)' in cpp
    assert 'static_cast<int8_t>(raw_b)' in cpp


def test_dot8_iu4_uses_operand_signedness_modifiers():
    cpp = gen_dot8(['vdst'], ['src0', 'src1', 'src2'], 'dot8_i32_iu4')

    assert 'src0_signed = (inst_.neg & 0x1u) != 0' in cpp
    assert 'src1_signed = (inst_.neg & 0x2u) != 0' in cpp
    assert 'raw_a | ~0xF' in cpp
    assert 'raw_b | ~0xF' in cpp


def test_pk_f16_binop_narrows_inline_float_constants():
    cpp = gen_pk_binop(
        ['vdst'],
        ['src0', 'src1'],
        'add',
        'f16',
        opsel_exprs=('inst_.op_sel', 'inst_.op_sel_hi'),
    )

    assert 'if (amdgpu::pk16_src_needs_narrowing(inst_.src0, src0.size_bits()))' in cpp
    assert 'raw0 = util::f32_to_f16(std::bit_cast<float>(raw0));' in cpp
    assert 'if (amdgpu::pk16_src_needs_narrowing(inst_.src1, src1.size_bits()))' in cpp
    assert 'raw1 = util::f32_to_f16(std::bit_cast<float>(raw1));' in cpp
    # Keyed on the selector field, never on the operand: an IsaOperand keeps a
    # 255-literal's value in encoding_value_.
    assert 'encoding_value_' not in cpp


def test_pk_bf16_ternary_narrows_inline_float_constants_as_bf16():
    cpp = gen_pk_ternary(
        ['vdst'],
        ['src0', 'src1', 'src2'],
        'fma',
        'bf16',
        op_sel_hi_2_expr='inst_.op_sel_hi_2',
        opsel_exprs=('inst_.op_sel', 'inst_.op_sel_hi'),
    )

    assert 'raw0 = util::f32_to_bf16(std::bit_cast<float>(raw0));' in cpp
    assert 'raw2 = util::f32_to_bf16(std::bit_cast<float>(raw2));' in cpp
    assert 'f32_to_f16' not in cpp


def test_dot2_half_forms_narrow_inline_float_constants():
    for cls, narrow in (
        ('dot2_f32_f16', 'f32_to_f16'),
        ('dot2_f32_bf16', 'f32_to_bf16'),
    ):
        cpp = gen_dot2(
            ['vdst'],
            ['src0', 'src1', 'src2'],
            cls,
            opsel_exprs=('inst_.op_sel', 'inst_.op_sel_hi'),
        )

        assert (
            'if (amdgpu::pk16_src_needs_narrowing(inst_.src0, src0.size_bits()))' in cpp
        )
        assert f'raw0 = util::{narrow}(std::bit_cast<float>(raw0));' in cpp
        assert (
            'if (amdgpu::pk16_src_needs_narrowing(inst_.src1, src1.size_bits()))' in cpp
        )
        assert f'raw1 = util::{narrow}(std::bit_cast<float>(raw1));' in cpp
        # src2 is an f32 accumulator on this family, so it stays 32-bit.
        assert 'raw2' not in cpp


def test_dot2_true16_narrows_inline_float_constants():
    """v_dot2_f16_f16 / v_dot2_bf16_bf16 read src0/src1 as packed v2 halves off a
    32-bit read_lane, so they need the same narrowing the VOP3P dots do. Only
    src2 goes through read_vop3_true16_src, which is already 16-bit."""
    for cls, narrow in (
        ('dot2_f16_f16', 'f32_to_f16'),
        ('dot2_bf16_bf16', 'f32_to_bf16'),
    ):
        cpp = gen_dot2_true16(['vdst'], ['src0', 'src1', 'src2'], cls)

        assert (
            'if (amdgpu::pk16_src_needs_narrowing(inst_.src0, src0.size_bits()))' in cpp
        )
        assert f'raw0 = util::{narrow}(std::bit_cast<float>(raw0));' in cpp
        assert (
            'if (amdgpu::pk16_src_needs_narrowing(inst_.src1, src1.size_bits()))' in cpp
        )
        assert f'raw1 = util::{narrow}(std::bit_cast<float>(raw1));' in cpp
        # src2 is read at 16 bits by read_vop3_true16_src, so it is left alone.
        assert 'inst_.src2, src2.size_bits()' not in cpp


def test_dot2_integer_forms_leave_inline_constants_alone():
    for cls in ('dot2_i32_i16', 'dot2_u32_u16'):
        cpp = gen_dot2(
            ['vdst'],
            ['src0', 'src1', 'src2'],
            cls,
            opsel_exprs=('inst_.op_sel', 'inst_.op_sel_hi'),
        )

        assert 'pk16_src_needs_narrowing' not in cpp
        assert 'bit_cast' not in cpp


def test_pk_integer_binop_leaves_inline_constants_alone():
    for dtype in ('i16', 'u16'):
        cpp = gen_pk_binop(
            ['vdst'],
            ['src0', 'src1'],
            'add',
            dtype,
            opsel_exprs=('inst_.op_sel', 'inst_.op_sel_hi'),
        )

        assert '240u' not in cpp
        assert 'bit_cast' not in cpp


def test_gfx1250_pk_f32_uses_literal_aware_pair_helper():
    cpp = gen_pk_binop_f32(
        ['vdst'],
        ['src0', 'src1'],
        'add',
        opsel_exprs=('inst_.opsel', 'inst_.opsel_hi'),
        use_cdna5_helpers=True,
    )

    assert 'const auto s0 = read_pk_f32_words(src0, wf, lane)' in cpp
    assert 'read_pk_f32_words(src0, wf, lane)' in cpp
    assert 's0_hi_w' not in cpp
    assert 's0.hi' in cpp
    assert 'read_lane64' not in cpp


def test_renamed_vop3p_packed_f32_probe_passes_profile_selectors():
    probe = vop3p_local_simd_probe_line('v_pk_add_f32_vop3p', ('opsel', 'opsel_hi'))

    assert probe is not None
    assert 'ROCJITSU_TRY_SIMD_VOP3P_PK_BINARY_F32_SELECTORS' in probe
    assert 'inst_.opsel, inst_.opsel_hi' in probe
    assert (
        vop3p_local_simd_probe_line('v_pk_add_f32_vop3p', ('op_sel', 'op_sel_hi'))
        is None
    )
    assert (
        vop3p_local_simd_probe_line('v_pk_add_f16_vop3p', ('opsel', 'opsel_hi')) is None
    )


def test_renamed_vop3p_packed_fma_f32_probe_passes_all_profile_selectors():
    probe = vop3p_local_simd_probe_line(
        'v_pk_fma_f32_vop3p',
        ('opsel', 'opsel_hi'),
        'inst_.pad_14',
    )

    assert probe is not None
    assert 'ROCJITSU_TRY_SIMD_VOP3P_PK_TERNARY_F32_SELECTORS' in probe
    assert 'inst_.opsel, inst_.opsel_hi, inst_.pad_14' in probe
    assert (
        vop3p_local_simd_probe_line(
            'v_pk_fma_f32_vop3p',
            ('op_sel', 'op_sel_hi'),
        )
        is None
    )


def test_gfx1250_mad_mix_f32_uses_helper_and_fma():
    cpp = gen_mad_mix_f32(
        ['vdst'],
        ['src0', 'src1', 'src2'],
        op_sel_hi_2_expr='inst_.pad_14',
        opsel_exprs=('inst_.opsel', 'inst_.opsel_hi'),
        use_cdna5_helpers=True,
    )

    assert 'read_fma_mix_source_f32(src0, wf, lane' in cpp
    assert 'std::fma(a, b, c)' in cpp
    assert 'a * b + c' not in cpp


def test_mad_mix_applies_abs_before_neg():
    cpp = gen_mad_mix_f32(
        ['vdst'],
        ['src0', 'src1', 'src2'],
        op_sel_hi_2_expr='inst_.op_sel_hi_2',
        opsel_exprs=('inst_.op_sel', 'inst_.op_sel_hi'),
    )

    abs_line = 'if (inst_.neg_hi & 1) a = std::fabs(a);'
    neg_line = 'if (inst_.neg & 1) a = -a;'
    assert 'read_mix_src(raw1, inst_.src1' in cpp
    assert 'uint16_t bits = amdgpu::is_inline_float_src(src_selector)' in cpp
    assert abs_line in cpp
    assert cpp.index(abs_line) < cpp.index(neg_line)


def test_gfx1250_mad_mixlo_f16_uses_helper_and_fma():
    cpp = gen_mad_mix_lo_hi(
        ['vdst'],
        ['src0', 'src1', 'src2'],
        is_lo=True,
        op_sel_hi_2_expr='inst_.pad_14',
        opsel_exprs=('inst_.opsel', 'inst_.opsel_hi'),
        use_cdna5_helpers=True,
    )

    assert 'read_fma_mix_source_f32(src0, wf, lane' in cpp
    assert 'std::fma(a, b, c)' in cpp
    assert 'util::f32_to_f16_mode(result, wf.fp16_ovfl())' in cpp


def test_mad_mixhi_f16_uses_true16_high_write():
    cpp = gen_mad_mix_lo_hi(
        ['vdst'],
        ['src0', 'src1', 'src2'],
        is_lo=False,
        op_sel_hi_2_expr='inst_.op_sel_hi_2',
        opsel_exprs=('inst_.op_sel', 'inst_.op_sel_hi'),
    )

    assert 'write_vop3_true16_dst(vdst, wf, lane, 0x8u, h)' in cpp
    assert 'vdst.write_lane(wf, lane, (prev & 0x0000FFFFu)' not in cpp


def test_mad_mixlo_bf16_uses_true16_low_write():
    cpp = gen_mad_mix_bf16(
        ['vdst'],
        ['src0', 'src1', 'src2'],
        result='lo',
        op_sel_hi_2_expr='inst_.op_sel_hi_2',
        opsel_exprs=('inst_.op_sel', 'inst_.op_sel_hi'),
    )

    assert 'write_vop3_true16_dst(vdst, wf, lane, 0u, h)' in cpp
    assert 'vdst.write_lane(wf, lane, (prev & 0xFFFF0000u)' not in cpp


def test_gfx1250_bf16_mad_mix_variants_use_bf16_helper():
    cpp_f32 = gen_mad_mix_bf16(
        ['vdst'],
        ['src0', 'src1', 'src2'],
        result='f32',
        op_sel_hi_2_expr='inst_.pad_14',
        opsel_exprs=('inst_.opsel', 'inst_.opsel_hi'),
        use_cdna5_helpers=True,
    )
    cpp_lo = gen_mad_mix_bf16(
        ['vdst'],
        ['src0', 'src1', 'src2'],
        result='lo',
        op_sel_hi_2_expr='inst_.pad_14',
        opsel_exprs=('inst_.opsel', 'inst_.opsel_hi'),
        use_cdna5_helpers=True,
    )

    assert 'read_fma_mix_bf16_source_f32(src0, wf, lane' in cpp_f32
    assert 'std::bit_cast<uint32_t>(result)' in cpp_f32
    assert 'util::f32_to_bf16(result)' in cpp_lo

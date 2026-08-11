# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for packed execute code generation."""

from amdisa.codegen.execute.packed import (
    gen_dot4,
    gen_dot8,
    gen_mad_mix_bf16,
    gen_mad_mix_f32,
    gen_mad_mix_lo_hi,
    gen_pk_binop_f32,
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


def test_gfx1250_pk_f32_uses_literal_aware_pair_helper():
    cpp = gen_pk_binop_f32(
        ['vdst'],
        ['src0', 'src1'],
        'add',
        opsel_exprs=('inst_.opsel', 'inst_.opsel_hi'),
        use_gfx1250_helpers=True,
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
        use_gfx1250_helpers=True,
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
    assert 'src_selector >= 240u && src_selector <= 248u' in cpp
    assert abs_line in cpp
    assert cpp.index(abs_line) < cpp.index(neg_line)


def test_gfx1250_mad_mixlo_f16_uses_helper_and_fma():
    cpp = gen_mad_mix_lo_hi(
        ['vdst'],
        ['src0', 'src1', 'src2'],
        is_lo=True,
        op_sel_hi_2_expr='inst_.pad_14',
        opsel_exprs=('inst_.opsel', 'inst_.opsel_hi'),
        use_gfx1250_helpers=True,
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
        use_gfx1250_helpers=True,
    )
    cpp_lo = gen_mad_mix_bf16(
        ['vdst'],
        ['src0', 'src1', 'src2'],
        result='lo',
        op_sel_hi_2_expr='inst_.pad_14',
        opsel_exprs=('inst_.opsel', 'inst_.opsel_hi'),
        use_gfx1250_helpers=True,
    )

    assert 'read_fma_mix_bf16_source_f32(src0, wf, lane' in cpp_f32
    assert 'std::bit_cast<uint32_t>(result)' in cpp_f32
    assert 'util::f32_to_bf16(result)' in cpp_lo

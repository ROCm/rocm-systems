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
    gen_pk_fmac_vop2,
    gen_pk_fmac_vop3,
    gen_pk_binop_u64,
    gen_pk_lshl_add_u64,
    gen_pk_ternary,
)
from amdisa.codegen.execute.simd_codegen import vop3p_local_simd_probe_line


def test_dot4_iu8_uses_operand_signedness_modifiers():
    cpp = gen_dot4(['vdst'], ['src0', 'src1', 'src2'], 'dot4_i32_iu8')

    assert 'src0_signed = (inst_.neg & 0x1u) != 0' in cpp
    assert 'src1_signed = (inst_.neg & 0x2u) != 0' in cpp
    assert 'static_cast<int8_t>(raw_a)' in cpp
    assert 'static_cast<int8_t>(raw_b)' in cpp
    assert 'int64_t sum' in cpp
    assert 'std::numeric_limits<int32_t>::min()' in cpp
    assert 'std::numeric_limits<int32_t>::max()' in cpp


def test_dot2_integer_clamp_uses_widened_signed_and_unsigned_ranges():
    signed = gen_dot2(
        ['vdst'],
        ['src0', 'src1', 'src2'],
        'dot2_i32_i16',
        ('inst_.op_sel', 'inst_.op_sel_hi'),
    )
    unsigned = gen_dot2(
        ['vdst'],
        ['src0', 'src1', 'src2'],
        'dot2_u32_u16',
        ('inst_.op_sel', 'inst_.op_sel_hi'),
    )

    assert 'int64_t result' in signed
    assert 'std::numeric_limits<int32_t>::min()' in signed
    assert 'std::numeric_limits<int32_t>::max()' in signed
    assert 'uint64_t result' in unsigned
    assert 'std::numeric_limits<uint32_t>::max()' in unsigned


def test_pk_fmac_vop2_reads_old_destination_and_fuses_both_halves():
    cpp = gen_pk_fmac_vop2(['vdst'], ['src0', 'vsrc1'])

    assert 'read_lane(vdst, lane)' in cpp
    assert cpp.count('amdgpu::fp_mode::fma_f16') == 2
    assert 'wf.fp_round_mode_f16_f64()' in cpp
    assert 'wf.fp_denorm_mode_f16_f64()' in cpp
    assert ', 0, false, wf.fp16_ovfl(), amdgpu::floating_clamp_nan_to_zero(wf))' in cpp


def test_promoted_pk_fmac_applies_vop3_modifiers_to_multiplicands_only():
    cpp = gen_pk_fmac_vop3(['vdst'], ['src0', 'src1'])

    assert 'read_lane(vdst, lane)' in cpp
    assert cpp.count('amdgpu::fp_mode::fma_f16') == 2
    assert 'inst_.abs & 1u, inst_.abs & 2u, false' in cpp
    assert 'inst_.neg & 1u, inst_.neg & 2u, false' in cpp
    assert 'effective_f16_omod' in cpp
    assert 'wf.ieee_mode(), true, inst_.omod' in cpp
    assert 'omod, inst_.clamp' in cpp
    assert 'op_sel' not in cpp


def test_pk_fma_f16_uses_mode_helper_and_clamp_for_both_halves():
    cpp = gen_pk_ternary(
        ['vdst'],
        ['src0', 'src1', 'src2'],
        'fma',
        'f16',
        op_sel_hi_2_expr='inst_.op_sel_hi_2',
        opsel_exprs=('inst_.op_sel', 'inst_.op_sel_hi'),
    )

    assert cpp.count('amdgpu::fp_mode::fma_f16') == 2
    assert 'wf.fp_round_mode_f16_f64()' in cpp
    assert 'wf.fp_denorm_mode_f16_f64()' in cpp
    assert (
        ', 0, inst_.clamp, wf.fp16_ovfl(), ' 'amdgpu::floating_clamp_nan_to_zero(wf))'
    ) in cpp


def test_pk_add_minmax_saturates_add_before_selecting_third_operand():
    signed = gen_pk_ternary(
        ['vdst'],
        ['src0', 'src1', 'src2'],
        'add_max_sat',
        'i16',
        op_sel_hi_2_expr='inst_.pad_14',
        opsel_exprs=('inst_.opsel', 'inst_.opsel_hi'),
    )
    unsigned = gen_pk_ternary(
        ['vdst'],
        ['src0', 'src1', 'src2'],
        'add_min_sat',
        'u16',
        op_sel_hi_2_expr='inst_.pad_14',
        opsel_exprs=('inst_.opsel', 'inst_.opsel_hi'),
    )

    assert 'std::clamp(static_cast<int32_t>(a_lo) + b_lo, -32768, 32767)' in signed
    assert 'std::max(sum_lo, c_lo)' in signed
    assert 'if (inst_.clamp)' in signed
    assert 'std::max<int16_t>(static_cast<int16_t>(rlo), 0)' in signed
    assert 'std::min(static_cast<uint32_t>(a_lo) + b_lo, 65535u)' in unsigned
    assert 'std::min(sum_lo, c_lo)' in unsigned


def test_pk_mad_integer_clamp_saturates_both_selected_halves():
    signed = gen_pk_ternary(
        ['vdst'],
        ['src0', 'src1', 'src2'],
        'mad',
        'i16',
        op_sel_hi_2_expr='inst_.op_sel_hi_2',
        opsel_exprs=('inst_.op_sel', 'inst_.op_sel_hi'),
        integer_clamp=True,
    )
    unsigned = gen_pk_ternary(
        ['vdst'],
        ['src0', 'src1', 'src2'],
        'mad',
        'u16',
        op_sel_hi_2_expr='inst_.op_sel_hi_2',
        opsel_exprs=('inst_.op_sel', 'inst_.op_sel_hi'),
        integer_clamp=True,
    )

    assert signed.count('vop3_integer_mad<int16_t, 16>') == 2
    assert signed.count('inst_.clamp') == 2
    assert unsigned.count('vop3_integer_mad<uint16_t, 16>') == 2
    assert unsigned.count('inst_.clamp') == 2


def test_dot8_iu4_uses_operand_signedness_modifiers():
    cpp = gen_dot8(['vdst'], ['src0', 'src1', 'src2'], 'dot8_i32_iu4')

    assert 'src0_signed = (inst_.neg & 0x1u) != 0' in cpp
    assert 'src1_signed = (inst_.neg & 0x2u) != 0' in cpp
    assert 'raw_a | ~0xF' in cpp
    assert 'raw_b | ~0xF' in cpp
    assert 'int64_t sum' in cpp
    assert 'std::numeric_limits<int32_t>::min()' in cpp
    assert 'std::numeric_limits<int32_t>::max()' in cpp


def test_unsigned_dot_clamps_widened_accumulator_before_narrowing():
    dot4 = gen_dot4(['vdst'], ['src0', 'src1', 'src2'], 'dot4_u32_u8')
    dot8 = gen_dot8(['vdst'], ['src0', 'src1', 'src2'], 'dot8_u32_u4')

    assert 'uint64_t sum' in dot4
    assert 'if (inst_.clamp && amdgpu::dot4_clamp_supported(wf))' in dot4
    assert 'std::numeric_limits<uint32_t>::max()' in dot4
    assert 'uint64_t sum' in dot8
    assert 'if (inst_.clamp)' in dot8
    assert 'std::numeric_limits<uint32_t>::max()' in dot8


def test_signed_dot4_clamp_uses_profile_policy():
    cpp = gen_dot4(['vdst'], ['src0', 'src1', 'src2'], 'dot4_i32_iu8')

    assert 'int64_t sum' in cpp
    assert 'if (inst_.clamp && amdgpu::dot4_clamp_supported(wf))' in cpp
    assert 'std::numeric_limits<int32_t>::min()' in cpp
    assert 'std::numeric_limits<int32_t>::max()' in cpp


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


def test_cdna_pk_f32_reads_all_register_pairs():
    cpp = gen_pk_binop_f32(
        ['vdst'],
        ['src0', 'src1'],
        'add',
        opsel_exprs=('inst_.op_sel', 'inst_.op_sel_hi'),
    )

    assert 'read_lane_pair32(src0, lane)' in cpp
    assert 'const uint32_t s0_lo_w = s0_pair_w.lo' in cpp
    assert 'const uint32_t s0_hi_w = s0_pair_w.hi' in cpp
    assert 'encoding_value_ >= 256' not in cpp


def test_pk_lshl_add_u64_operates_on_two_independent_64_bit_elements():
    cpp = gen_pk_lshl_add_u64(['vdst'], ['src0', 'src1', 'src2'])

    assert 'const auto values = read_pk_u64_pair(src0, wf, lane);' in cpp
    assert 'const auto shifts = read_pk_u32_pair(src1, wf, lane);' in cpp
    assert 'const auto addends = read_pk_u64_pair(src2, wf, lane);' in cpp
    assert (
        'amdgpu::lshl_masked(values.lo, static_cast<uint64_t>(shifts.lo)) + addends.lo'
        in cpp
    )
    assert (
        'amdgpu::lshl_masked(values.hi, static_cast<uint64_t>(shifts.hi)) + addends.hi'
        in cpp
    )
    assert 'lshl_masked(values.lo, shifts.lo)' not in cpp
    assert 'lshl_masked(values.hi, shifts.hi)' not in cpp
    assert 'results[lane] = {result_lo, result_hi};' in cpp
    assert 'write_pk_u64_pair(vdst, wf, lane, results[lane]);' in cpp
    assert 'inst_.neg' not in cpp
    assert 'inst_.clamp' not in cpp


def test_pk_lshl_add_u64_rejects_unproven_counts_before_any_write():
    cpp = gen_pk_lshl_add_u64(['vdst'], ['src0', 'src1', 'src2'])

    reject = 'if (shifts.lo > 4u || shifts.hi > 4u)'
    report = 'wf.report_instruction_execution_error('
    write = 'write_pk_u64_pair(vdst, wf, lane, results[lane]);'
    assert reject in cpp
    assert report in cpp
    assert write in cpp
    assert cpp.index(reject) < cpp.index(report) < cpp.index(write)


def test_pk_u64_add_sub_generate_per_element_negation_and_integer_clamp():
    add = gen_pk_binop_u64(['vdst'], ['src0', 'src1'], 'add')
    sub = gen_pk_binop_u64(['vdst'], ['src0', 'src1'], 'sub')

    for cpp in (add, sub):
        assert 'const auto lhs = read_pk_u64_pair(src0, wf, lane);' in cpp
        assert 'const auto rhs = read_pk_u64_pair(src1, wf, lane);' in cpp
        assert 'using Wide = util::int128_t;' in cpp
        assert 'if (negate_lhs) lhs_wide = -lhs_wide;' in cpp
        assert 'if (negate_rhs) rhs_wide = -rhs_wide;' in cpp
        assert 'if (result < Wide{}) return 0;' in cpp
        assert 'if (result > kMax) return std::numeric_limits<uint64_t>::max();' in cpp
        assert '(inst_.neg & 1u) != 0' in cpp
        assert '(inst_.neg_hi & 2u) != 0' in cpp
        assert 'write_pk_u64_pair(vdst, wf, lane, {result_lo, result_hi});' in cpp
        assert cpp.index('if (negate_lhs)') < cpp.index('const Wide result')
        assert cpp.index('const Wide result') < cpp.index('if (inst_.clamp)')
    assert 'const Wide result = lhs_wide + rhs_wide;' in add
    assert 'const Wide result = lhs_wide - rhs_wide;' in sub


def test_pk_u64_binop_rejects_unknown_operation():
    import pytest

    with pytest.raises(ValueError, match='unsupported packed U64 binary operation'):
        gen_pk_binop_u64(['vdst'], ['src0', 'src1'], 'mul')


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


def test_gfx1250_bf16_mad_mix_variants_use_mode_rounding_helper():
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
    cpp_hi = gen_mad_mix_bf16(
        ['vdst'],
        ['src0', 'src1', 'src2'],
        result='hi',
        op_sel_hi_2_expr='inst_.pad_14',
        opsel_exprs=('inst_.opsel', 'inst_.opsel_hi'),
        use_cdna5_helpers=True,
    )

    assert 'read_fma_mix_bf16_source_f32(src0, wf, lane' in cpp_f32
    assert 'std::bit_cast<uint32_t>(result)' in cpp_f32
    mode_round = 'amdgpu::fp_mode::detail::fma_f32_to_bf16_nearest_environment('
    assert mode_round in cpp_lo
    assert mode_round in cpp_hi
    assert 'amdgpu::fp_mode::detail::ScopedFenv nearest_environment(0);' in cpp_lo
    assert 'amdgpu::fp_mode::detail::ScopedFenv nearest_environment(0);' in cpp_hi
    assert 'amdgpu::fp_mode::detail::ScopedFenv nearest_environment(0);' not in cpp_f32
    assert 'std::fma(a, b, c)' not in cpp_lo
    assert 'std::fma(a, b, c)' not in cpp_hi
    assert 'wf.fp_round_mode_f16_f64(), inst_.clamp' in cpp_lo
    assert 'util::f32_to_bf16(result)' not in cpp_lo
    assert 'util::f32_to_bf16(result)' not in cpp_hi

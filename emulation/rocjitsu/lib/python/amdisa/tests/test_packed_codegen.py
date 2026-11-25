import pytest

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for packed execute code generation."""

from amdisa.codegen.execute.packed import (
    gen_pk_binop,
    gen_pk_binop_f32,
    gen_pk_binop_f64,
    gen_pk_ternary_f64,
    gen_pk_binop_u64,
    gen_pk_lshl_add_u64,
    gen_pk_ternary,
)
from amdisa.codegen.execute.simd_codegen import vop3p_local_simd_probe_line


def test_pk_fma_f16_has_no_legacy_ternary_fallback():
    import pytest

    with pytest.raises(ValueError, match='does not support fma'):
        gen_pk_ternary(
            ['d'],
            ['a', 'b', 'c'],
            'fma',
            'f16',
            op_sel_hi_2_expr='hi_c',
            opsel_exprs=('lo', 'hi'),
        )


def test_pk_bf16_ternary_preserves_fp32_inline_constants():
    cpp = gen_pk_ternary(
        ['vdst'],
        ['src0', 'src1', 'src2'],
        'fma',
        'bf16',
        op_sel_hi_2_expr='inst_.op_sel_hi_2',
        opsel_exprs=('inst_.op_sel', 'inst_.op_sel_hi'),
    )

    assert 'pk16_src_needs_narrowing' not in cpp
    assert 'raw0 = util::f32_to_bf16' not in cpp
    assert 'raw2 = util::f32_to_bf16' not in cpp
    assert 'f32_to_f16' not in cpp


def test_pk_bf16_binary_preserves_fp32_inline_constants():
    for op in ('add', 'mul', 'min', 'max'):
        cpp = gen_pk_binop(
            ['vdst'],
            ['src0', 'src1'],
            op,
            'bf16',
            opsel_exprs=('inst_.opsel', 'inst_.opsel_hi'),
        )
        assert 'pk16_src_needs_narrowing' not in cpp
        assert 'raw0 = util::f32_to_bf16' not in cpp
        assert 'raw1 = util::f32_to_bf16' not in cpp


def test_gfx1250_pk_f32_uses_literal_aware_pair_helper():
    cpp = gen_pk_binop_f32(
        ['vdst'],
        ['src0', 'src1'],
        'add',
        opsel_exprs=('inst_.opsel', 'inst_.opsel_hi'),
        use_cdna5_helpers=True,
    )

    assert 'const auto s0 = amdgpu::RegisterAccess(wf).read_lane_pair32(' in cpp
    assert 'src0, lane, amdgpu::ScalarPairMode::Replicate32)' in cpp
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


def test_pk_lshl_add_u64_routes_through_count_validating_adapter():
    cpp = gen_pk_lshl_add_u64(['vdst'], ['src0', 'src1', 'src2'])
    assert 'execute_packed_u64<true>' in cpp
    assert 'rj_arith_v_pk_lshl_add_u64_w32, 0, vdst, src0, src1, &src2' in cpp
    assert 'inst_.neg' not in cpp
    assert 'inst_.clamp' not in cpp


def test_pk_u64_add_sub_forward_per_element_modifiers():
    for op in ('add', 'sub'):
        cpp = gen_pk_binop_u64(['vdst'], ['src0', 'src1'], op)
        assert 'execute_packed_u64<false>' in cpp
        assert f'rj_arith_v_pk_{op}_nc_u64_w32, modifiers, vdst, src0, src1' in cpp
        assert 'inst_.neg | (uint64_t(inst_.neg_hi) << 3)' in cpp
        assert 'inst_.clamp ? RJ_ARITH_PK_CLAMP : 0' in cpp
        assert 'int128' not in cpp


def test_pk_u64_binop_rejects_unknown_operation():
    import pytest

    with pytest.raises(ValueError, match='unsupported packed U64 binary operation'):
        gen_pk_binop_u64(['vdst'], ['src0', 'src1'], 'mul')


def test_pk_f64_binops_use_mode_aware_two_element_execution():
    expected_operations = {
        'add': 'Add',
        'mul': 'Multiply',
        'max_num': 'MaximumNumber',
        'min_num': 'MinimumNumber',
    }
    for operation, enum_name in expected_operations.items():
        cpp = gen_pk_binop_f64(['vdst'], ['src0', 'src1'], operation)

        assert 'const auto lhs = read_pk_u64_pair(src0, wf, lane);' in cpp
        assert 'const auto rhs = read_pk_u64_pair(src1, wf, lane);' in cpp
        assert f'BinaryF64Op::{enum_name}' in cpp
        assert cpp.count('amdgpu::fp_mode::binary_f64(') == 1
        assert 'wf.fp_round_mode_f16_f64()' in cpp
        assert 'wf.fp_denorm_mode_f16_f64()' in cpp
        assert 'lhs_bits ^= kSignBit' in cpp
        assert 'rhs_bits ^= kSignBit' in cpp
        assert 'amdgpu::fp_mode::finish_f64(' in cpp
        assert 'inst_.clamp' in cpp
        assert 'write_pk_u64_pair(vdst, wf, lane, {result_lo, result_hi});' in cpp


def test_pk_f64_binop_rejects_unknown_operation():
    import pytest

    with pytest.raises(ValueError, match='unsupported packed F64 binary operation'):
        gen_pk_binop_f64(['vdst'], ['src0', 'src1'], 'divide')


def test_pk_f64_fma_uses_mode_aware_fused_two_element_execution():
    cpp = gen_pk_ternary_f64(['vdst'], ['src0', 'src1', 'src2'], 'fma')

    assert 'const auto multiplicand = read_pk_u64_pair(src0, wf, lane);' in cpp
    assert 'const auto multiplier = read_pk_u64_pair(src1, wf, lane);' in cpp
    assert 'const auto addend = read_pk_u64_pair(src2, wf, lane);' in cpp
    assert cpp.count('amdgpu::fp_mode::fma_f64(') == 1
    assert 'wf.fp_round_mode_f16_f64()' in cpp
    assert 'wf.fp_denorm_mode_f16_f64()' in cpp
    assert 'multiplicand_bits ^= kSignBit' in cpp
    assert 'multiplier_bits ^= kSignBit' in cpp
    assert 'addend_bits ^= kSignBit' in cpp
    assert 'amdgpu::fp_mode::finish_f64(' in cpp
    assert 'inst_.clamp' in cpp
    assert 'write_pk_u64_pair(vdst, wf, lane, {result_lo, result_hi});' in cpp


def test_pk_f64_ternary_rejects_unknown_operation():
    import pytest

    with pytest.raises(ValueError, match='unsupported packed F64 ternary operation'):
        gen_pk_ternary_f64(['vdst'], ['src0', 'src1', 'src2'], 'mad')


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


def test_renamed_packed_float_routes_use_exact_helpers():
    for spelling, operation, bf16 in (
        ('v_pk_add_bf16_vop3p', 'ADD', True),
        ('v_pk_fma_bf16_vop3p', 'FMA', True),
    ):
        probe = vop3p_local_simd_probe_line(spelling, ('opsel', 'opsel_hi'))
        assert (
            probe
            == f'  ROCJITSU_TRY_SIMD_PACKED_FLOAT({operation}, {str(bf16).lower()});'
        )


def test_dot_dispatch_covers_every_semantic_class():
    """The DOT semantic classes and their DISPATCH handlers must agree exactly.

    The generator keeps no dot2_/dot4_/dot8_ prefix fallback, so a class with
    no entry would generate an unimplemented-instruction reporting stub. Checking
    the other direction too rejects handlers for classes no ISA defines,
    including bare ``dot2``/``dot4``/``dot8`` routes.
    """
    import os
    from pathlib import Path

    from amdisa.__main__ import _PROFILES
    from amdisa.codegen.execute import DISPATCH
    from amdisa.parser import Parser
    from amdisa.semantics import derive_all_semantics

    mrisa = Path(
        os.environ.get(
            'MRISA_PATH',
            Path(__file__).resolve().parents[6]
            / 'shared'
            / 'machine-readable-isa'
            / 'isa',
        )
    )
    prefixes = ('dot2_', 'dot4_', 'dot8_')
    found = set()
    # Taken from the generator's own profile table, minus the bare 'cdna' alias
    # that has no ISA XML of its own, so a newly supported ISA is covered here
    # the moment it is registered.
    for name, profile in _PROFILES.items():
        if name == 'cdna':
            continue
        spec = Parser(
            str(mrisa / f'amdgpu_isa_{name.replace(".", "_")}.xml'), profile()
        ).parse()
        for sem in derive_all_semantics(spec).instructions.values():
            if sem.semantic_class.startswith(prefixes):
                found.add(sem.semantic_class)

    assert found, 'no DOT semantic classes found; the ISA specs did not load'
    routes = ('dot2', 'dot4', 'dot8')
    assert found == {cls for cls in DISPATCH if cls.startswith(routes)}


def test_pk_f16_binary_has_no_legacy_fallback():
    with pytest.raises(ValueError, match='does not support f16'):
        gen_pk_binop(['vdst'], ['src0', 'src1'], 'add', 'f16')

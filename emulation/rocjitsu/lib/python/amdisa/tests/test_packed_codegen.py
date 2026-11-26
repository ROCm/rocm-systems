# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for packed execute code generation."""

import pytest

from amdisa.codegen.execute.packed import (
    gen_pk_binop,
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


def test_pk_f64_binops_use_carrier_aware_arithmetic_adapter():
    for operation in ('add', 'mul', 'min_num', 'max_num'):
        cpp = gen_pk_binop_f64(['vdst'], ['src0', 'src1'], operation)
        assert 'execute_packed_f64<false>' in cpp
        assert f'rj_arith_v_pk_{operation}_f64_w32, modifiers, vdst, src0, src1' in cpp
        assert 'inst_.neg | (uint64_t(inst_.neg_hi) << 3)' in cpp
        assert 'inst_.clamp ? RJ_ARITH_PK_CLAMP : 0' in cpp
        assert 'fp_mode::' not in cpp


def test_pk_f64_binop_rejects_unknown_operation():
    import pytest

    with pytest.raises(ValueError, match='unsupported packed F64 binary operation'):
        gen_pk_binop_f64(['vdst'], ['src0', 'src1'], 'divide')


def test_pk_f64_fma_uses_guest_fp_adapter():
    cpp = gen_pk_ternary_f64(['vdst'], ['src0', 'src1', 'src2'], 'fma')
    assert 'execute_packed_f64<true>' in cpp
    assert 'rj_arith_v_pk_fma_f64_w32, modifiers, vdst, src0, src1, &src2' in cpp
    assert 'inst_.neg | (uint64_t(inst_.neg_hi) << 3)' in cpp
    assert 'inst_.clamp ? RJ_ARITH_PK_CLAMP : 0' in cpp


def test_pk_f64_ternary_rejects_unknown_operation():
    import pytest

    with pytest.raises(ValueError, match='unsupported packed F64 ternary operation'):
        gen_pk_ternary_f64(['vdst'], ['src0', 'src1', 'src2'], 'mad')


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

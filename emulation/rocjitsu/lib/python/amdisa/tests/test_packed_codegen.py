# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for packed execute code generation."""

import pytest


from amdisa.codegen.execute.packed import (
    gen_pk_binop_f64,
    gen_pk_ternary_f64,
    gen_pk_binop_u64,
    gen_pk_lshl_add_u64,
)


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

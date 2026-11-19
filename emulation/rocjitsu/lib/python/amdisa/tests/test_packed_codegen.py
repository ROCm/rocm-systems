# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from amdisa.isa_profile import FloatDotAccumulation
from amdisa.codegen.execute.packed import (
    gen_dot2,
    gen_dot2_true16,
    gen_dot4,
    gen_dot8,
    gen_pk_binop,
    gen_pk_binop_f32,
    gen_pk_binop_f64,
    gen_pk_binop_u64,
    gen_pk_lshl_add_u64,
    gen_pk_ternary,
    gen_pk_ternary_f64,
)
from amdisa.codegen.execute.simd_codegen import vop3p_local_simd_probe_line
import pytest

"""Tests for packed execute code generation."""


def test_pk_fma_f16_has_no_legacy_ternary_fallback():
    import pytest

    with pytest.raises(ValueError, match='does not support f16'):
        gen_pk_ternary(
            ['d'],
            ['a', 'b', 'c'],
            'fma',
            'f16',
            op_sel_hi_2_expr='hi_c',
            opsel_exprs=('lo', 'hi'),
        )


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


def test_pk_f64_binop_rejects_unknown_operation():
    import pytest

    with pytest.raises(ValueError, match='unsupported packed F64 binary operation'):
        gen_pk_binop_f64(['vdst'], ['src0', 'src1'], 'divide')


def test_pk_f64_ternary_rejects_unknown_operation():
    import pytest

    with pytest.raises(ValueError, match='unsupported packed F64 ternary operation'):
        gen_pk_ternary_f64(['vdst'], ['src0', 'src1', 'src2'], 'mad')


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


def test_pk_f64_binops_use_carrier_aware_arithmetic_adapter():
    for operation in ('add', 'mul', 'min_num', 'max_num'):
        cpp = gen_pk_binop_f64(['vdst'], ['src0', 'src1'], operation)
        assert 'execute_packed_f64<false>' in cpp
        assert f'rj_arith_v_pk_{operation}_f64_w32, modifiers, vdst, src0, src1' in cpp
        assert 'inst_.neg | (uint64_t(inst_.neg_hi) << 3)' in cpp
        assert 'inst_.clamp ? RJ_ARITH_PK_CLAMP : 0' in cpp
        assert 'fp_mode::' not in cpp


def test_pk_f64_fma_uses_guest_fp_adapter():
    cpp = gen_pk_ternary_f64(['vdst'], ['src0', 'src1', 'src2'], 'fma')
    assert 'execute_packed_f64<true>' in cpp
    assert 'rj_arith_v_pk_fma_f64_w32, modifiers, vdst, src0, src1, &src2' in cpp
    assert 'inst_.neg | (uint64_t(inst_.neg_hi) << 3)' in cpp
    assert 'inst_.clamp ? RJ_ARITH_PK_CLAMP : 0' in cpp

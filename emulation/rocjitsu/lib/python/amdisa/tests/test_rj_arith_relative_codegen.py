# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from types import SimpleNamespace
from amdisa.codegen.execute import ExecuteContext


def context(arch, name, dst, src, *, is_vop3=False, result_writer=None):
    return ExecuteContext(
        inst=SimpleNamespace(name=name),
        sem=None,
        profile=None,
        arch_name=arch,
        dst_ops=dst,
        src_ops=src,
        enc_name='',
        is_vop3=is_vop3,
        has_abs=is_vop3,
        result_writer=result_writer,
    )


from pathlib import Path

import pytest

from amdisa.codegen.execute.rj_arith_scalar import RELATIVE_MOVES, gen_scalar
from amdisa.codegen.execute.vector_special import gen_vector_movrel, gen_vector_swaprel


@pytest.mark.parametrize('name', sorted(RELATIVE_MOVES))
def test_scalar_relative_moves_keep_selector_resolution_and_use_rj_arith_copy(name):
    for arch in ('rdna3', 'rdna4'):
        body = gen_scalar(context(arch, name, ['d'], ['a']))
        bits = 64 if name.endswith('B64') else 32
        assert f'rj_arith_s_mov_b{bits}(' in body
        assert 'relative_scalar_selector(wf,' in body
        if name.startswith('S_MOVRELS'):
            assert 'src_reg.value_or(0)' in body
        if name.startswith('S_MOVRELD') or name == 'S_MOVRELSD_2_B32':
            assert body.index('if (!dst_reg) return;') < body.index(
                'registers.read_scalar'
            )
        assert 'wf.m0()' in body
        assert 'wf.exec()' not in body
        assert 'status != RJ_ARITH_SUCCESS' in body
        assert body.index('status != RJ_ARITH_SUCCESS') < body.index(
            'registers.write_scalar'
        )
    assert gen_scalar(context('cdna5', name, ['d'], ['a'])) is None


@pytest.mark.parametrize('op', ['src', 'dst', 'srcdst', 'srcdst2'])
def test_relative_vector_moves_stage_address_and_modifiers_before_rj_arith(op):
    body = gen_vector_movrel(
        ['d'],
        ['a'],
        op,
        supports_dpp=True,
        supports_dpp8=True,
        use_rj_arith=True,
    )
    assert 'relative_vgpr_index' in body
    assert 'amdgpu::arith::execute_unary32' in body
    assert 'rj_arith_v_mov_b32_w64' in body
    assert 'for (uint32_t lane' not in body
    if op != 'dst':
        assert body.index('relative_vgpr_index') < body.index('amdgpu::dpp::apply_dpp(')
        assert body.index('ScopedOperandDelegate') < body.index(
            'amdgpu::arith::execute_unary32'
        )
    legacy = gen_vector_movrel(['d'], ['a'], op)
    assert 'for (uint32_t lane' in legacy
    assert 'arith::' not in legacy


def test_relative_swap_validates_both_operands_before_rj_arith():
    body = gen_vector_swaprel(['d'], ['a'], use_rj_arith=True)
    assert body.count('relative_vgpr_index') == 2
    assert 'amdgpu::arith::execute_swap<false>' in body
    assert body.rindex('relative_vgpr_index') < body.index(
        'amdgpu::arith::execute_swap'
    )
    assert 'for (uint32_t lane' not in body
    assert 'arith::' not in gen_vector_swaprel(['d'], ['a'])


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_all_relative_methods_use_rj_arith_after_emulator_address_resolution(arch):
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    cases = {
        'sop1': [
            'SMovrelsB32',
            'SMovrelsB64',
            'SMovreldB32',
            'SMovreldB64',
            'SMovrelsd2B32',
        ],
        'vop1': [
            'VMovrelsB32',
            'VMovreldB32',
            'VMovrelsdB32',
            'VMovrelsd2B32',
            'VSwaprelB32',
        ],
        'vop3': ['VMovrelsB32', 'VMovreldB32', 'VMovrelsdB32', 'VMovrelsd2B32'],
    }
    for encoding, names in cases.items():
        source = (root / f'{encoding}_exec.cpp').read_text()
        for name in names:
            methods = ['execute_impl']
            if encoding != 'sop1' and name != 'VSwaprelB32':
                methods.append('execute_modifier_impl')
            for method in methods:
                begin = source.index(f'void {name}{encoding.capitalize()}::{method}(')
                end = source.index('\n}\n', begin)
                body = source[begin:end]
                assert 'relative_' in body
                assert 'rj_arith_' in body or 'arith::' in body
                assert 'RegisterAccess(wf).write_lane' not in body

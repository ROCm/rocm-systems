# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest

from amdisa.codegen.execute.rj_arith_scalar import RELATIVE_MOVES, gen_scalar

from amdisa.codegen.execute.vector_special import gen_vector_movrel, gen_vector_swaprel


@pytest.mark.parametrize('op', ['src', 'dst', 'srcdst', 'srcdst2'])
def test_relative_vector_moves_stage_address_and_modifiers_before_rj_arith(op):
    body = gen_vector_movrel(['d'], ['a'], op, supports_dpp=True, supports_dpp8=True)
    assert 'relative_vgpr_index' in body
    assert 'amdgpu::arith::execute_move32' in body
    assert 'for (uint32_t lane' not in body
    if op != 'dst':
        assert body.index('relative_vgpr_index') < body.index('amdgpu::dpp::apply_dpp(')
        assert body.index('ScopedOperandDelegate') < body.index(
            'amdgpu::arith::execute_move32'
        )


def test_relative_swap_validates_both_operands_before_rj_arith():
    body = gen_vector_swaprel(['d'], ['a'])
    assert body.count('relative_vgpr_index') == 2
    assert 'amdgpu::arith::execute_swap<false>' in body
    assert body.rindex('relative_vgpr_index') < body.index(
        'amdgpu::arith::execute_swap'
    )
    assert 'for (uint32_t lane' not in body


@pytest.mark.parametrize('arch', ['rdna1', 'rdna2', 'rdna3', 'rdna3_5', 'rdna4'])
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
    if arch in ('rdna1', 'rdna2', 'rdna3_5'):
        del cases['sop1']
    if arch in ('rdna1', 'rdna2'):
        cases['vop3'] = ['VMovreldB32']
    for encoding, names in cases.items():
        source = (root / f'{encoding}_exec.cpp').read_text()
        for name in names:
            methods = ['execute_impl']
            if (
                encoding != 'sop1'
                and name != 'VSwaprelB32'
                and not (encoding == 'vop3' and arch in ('rdna1', 'rdna2'))
            ):
                methods.append('execute_modifier_impl')
            for method in methods:
                begin = source.index(f'void {name}{encoding.capitalize()}::{method}(')
                end = source.index('\n}\n', begin)
                body = source[begin:end]
                assert 'relative_' in body
                assert 'rj_arith_' in body or 'arith::' in body
                assert 'RegisterAccess(wf).write_lane' not in body

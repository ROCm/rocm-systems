# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pair moves retain their raw words and defer source routing to existing wrappers."""

from types import SimpleNamespace

import pytest

from amdisa.codegen.execute.rj_arith_pair_move import gen_pair_move


@pytest.mark.parametrize('arch', ['cdna3', 'cdna4', 'cdna5'])
@pytest.mark.parametrize('packed', [False, True])
def test_pair_move_binding(arch, packed):
    ctx = SimpleNamespace(
        arch_name=arch,
        inst=SimpleNamespace(name='V_PK_MOV_B32' if packed else 'V_MOV_B64'),
        dst_ops=['dst'],
        src_ops=['a', 'b'] if packed else ['a'],
        opsel_exprs=('inst_.op_sel', 'inst_.op_sel_hi'),
    )
    body = gen_pair_move(ctx)
    assert f'execute_pair_move<{str(packed).lower()}>' in body
    if packed:
        assert 'RJ_ARITH_PK_LO_A_HIGH' in body
        assert 'RJ_ARITH_PK_LO_B_HIGH' in body
        assert 'dst, a, &b' in body
    else:
        assert 'wf, 0, dst, a, nullptr' in body
    ctx.arch_name = 'cdna2'
    assert gen_pair_move(ctx) is None

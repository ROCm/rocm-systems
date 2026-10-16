# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT


import pytest

from amdisa.codegen.execute.rj_arith_scalar import MOVES, gen_scalar
from amdisa.codegen.execute.rj_arith_vector import gen_vector32, gen_integer16


@pytest.mark.parametrize('name', sorted(MOVES))
def test_scalar_moves_preserve_conditional_observation(name):
    for arch in ('rdna3', 'rdna4'):
        body = gen_scalar(arch, name, ['d'], ['a'])
        assert f'rj_arith_{name.lower()}(' in body
        assert 'wf.write_scc' not in body
        assert 'read_scalar(d)' not in body
        if name.startswith('S_CMOV'):
            assert body.index('if (!input_scc) return;') < body.index(
                'RegisterAccess registers'
            )
        else:
            assert 'read_scc' not in body
        assert ('uint16_t(a.encoding_value())' in body) == ('MOVK' in name)
        assert body.index('status != RJ_ARITH_SUCCESS') < body.index(
            'registers.write_scalar'
        )
    assert gen_scalar('cdna5', name, ['d'], ['a']) is None


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_vector_move_and_conditional_controls(arch):
    word = gen_vector32(arch, 'V_MOV_B32', ['d'], ['a'], True)
    assert 'rj_arith_v_mov_b32_w64' in word
    assert 'inst_.clamp' not in word and 'inst_.neg' not in word
    half = gen_integer16(arch, 'V_MOV_B16', ['d'], ['a'], True)
    assert 'rj_arith_v_mov_b16_w64' in half
    assert 'RJ_ARITH_ALU_HIGH_A' in half and 'RJ_ARITH_ALU_HIGH_D' in half
    for bits in (16, 32):
        body = gen_vector32(
            arch, f'V_CNDMASK_B{bits}', ['d'], ['a', 'b', 'condition'], True
        )
        assert f'rj_arith_v_cndmask_b{bits}_w64' in body
        assert 'read_wave_mask_scalar(condition, wf)' in body
        assert ('RJ_ARITH_ALU_NEG_A' in body) == (bits == 32)
        assert ('RJ_ARITH_ALU_HIGH_D' in body) == (bits == 16)
    assert 'wf.vcc_mask()' in gen_vector32(
        arch, 'V_CNDMASK_B32', ['d'], ['a', 'b'], False
    )

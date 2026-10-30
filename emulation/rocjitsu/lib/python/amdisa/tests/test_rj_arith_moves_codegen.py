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
import re

import pytest

from amdisa.codegen.execute.rj_arith_scalar import MOVES, gen_scalar
from amdisa.codegen.execute.rj_arith_vector import gen_vector32, gen_integer16


@pytest.mark.parametrize('name', sorted(MOVES))
def test_scalar_moves_preserve_conditional_observation(name):
    for arch in ('rdna3', 'rdna4'):
        body = gen_scalar(context(arch, name, ['d'], ['a']))
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
    assert gen_scalar(context('cdna5', name, ['d'], ['a'])) is None


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_vector_move_and_conditional_controls(arch):
    word = gen_vector32(context(arch, 'V_MOV_B32', ['d'], ['a'], is_vop3=True))
    assert 'rj_arith_v_mov_b32_w64' in word
    assert 'inst_.clamp' not in word and 'inst_.neg' not in word
    half = gen_integer16(context(arch, 'V_MOV_B16', ['d'], ['a'], is_vop3=True))
    assert 'rj_arith_v_mov_b16_w64' in half
    assert 'RJ_ARITH_ALU_HIGH_A' in half and 'RJ_ARITH_ALU_HIGH_D' in half
    for bits in (16, 32):
        body = gen_vector32(
            context(
                arch, f'V_CNDMASK_B{bits}', ['d'], ['a', 'b', 'condition'], is_vop3=True
            )
        )
        assert f'rj_arith_v_cndmask_b{bits}_w64' in body
        assert 'read_wave_mask_scalar(condition, wf)' in body
        for modifier in ('ABS_A', 'ABS_B', 'NEG_A', 'NEG_B'):
            assert 'RJ_ARITH_ALU_' + modifier in body
        assert ('RJ_ARITH_ALU_HIGH_D' in body) == (bits == 16)
    assert 'wf.vcc_mask()' in gen_vector32(
        context(arch, 'V_CNDMASK_B32', ['d'], ['a', 'b'], is_vop3=False)
    )


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_all_move_and_selection_encodings_route_to_rj_arith(arch):
    root = (
        Path(__file__).resolve().parents[3]
        / 'rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated'
        / arch
    )
    count = 0
    for file in ('sop1', 'sopk', 'vop1', 'vop2', 'vop3'):
        source = (root / (file + '_exec.cpp')).read_text()
        pattern = r'void ((?:S(?:Mov|Cmov)(?:B32|B64|kI32)|VMovB(?:16|32)|VCndmaskB(?:16|32))\w+)::execute(?:_modifier)?_impl\([^)]*\)\s*\{'
        for match in re.finditer(pattern, source):
            end, depth = match.end(), 1
            while depth:
                depth += (source[end] == '{') - (source[end] == '}')
                end += 1
            body = source[match.end() : end - 1]
            assert re.search(r'\brj_arith(?:_|::)', body), match[1]
            assert 'try_execute_' not in body
            assert 'amdgpu::execute_s_' not in body
            assert 'amdgpu::execute_v_' not in body
            count += 1
    assert count == 20  # Six scalar methods, eight MOV and six CNDMASK methods.

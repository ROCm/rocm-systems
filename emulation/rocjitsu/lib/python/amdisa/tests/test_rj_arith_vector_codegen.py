# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
from types import SimpleNamespace
from amdisa.codegen.execute import ExecuteContext
import re

import pytest


from amdisa.codegen.execute.rj_arith_vector import gen_vector32


def context(
    arch,
    name,
    dst,
    src,
    is_vop3,
    result_writer=None,
    *,
    enc_name='',
    opsel_exprs=('', ''),
    op_sel_hi_2_expr='',
):
    return ExecuteContext(
        inst=SimpleNamespace(name=name),
        sem=None,
        profile=None,
        arch_name=arch,
        dst_ops=dst,
        src_ops=src,
        enc_name=enc_name,
        opsel_exprs=opsel_exprs,
        op_sel_hi_2_expr=op_sel_hi_2_expr,
        is_vop3=is_vop3,
        has_abs=is_vop3,
        result_writer=result_writer,
    )


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
@pytest.mark.parametrize('name', ['V_CVT_I32_I16', 'V_CVT_U32_U16'])
def test_integer_widening_half_selection_and_ignored_modifiers(arch, name):
    for vop3 in (False, True):
        body = gen_vector32(context(arch, name, ['dst'], ['src'], vop3))
        assert 'rj_arith_' + name.lower() + '_w64' in body
        assert 'execute_unary32' in body
        assert ('vop3_opsel' in body) == vop3
        assert 'RJ_ARITH_ALU_HIGH_A ? 0xC : 0x3' in body
        assert 'arithmetic_flags()' in body
        for ignored in ('inst_.abs', 'inst_.neg', 'inst_.omod', 'inst_.clamp'):
            assert ignored not in body


@pytest.mark.parametrize('arch', ['rdna3', 'rdna4'])
def test_packed_saturation_reads_word_and_writes_selected_half(arch):
    for vop3 in (False, True):
        body = gen_vector32(context(arch, 'V_SAT_PK_U8_I16', ['dst'], ['src'], vop3))
        assert 'rj_arith_v_sat_pk_u8_i16_w64' in body
        assert 'execute_unary32' in body
        assert ('vop3_opsel(inst_) & 8' in body) == vop3
        assert '0xF, modifiers & RJ_ARITH_ALU_HIGH_D ? 0xC : 0x3' in body
        assert 'inst_.abs' not in body and 'inst_.neg' not in body

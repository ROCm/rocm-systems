# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Wave-mask operands come from the MR ISA ``FMT_NUM_M64`` data format.

A wave mask is one SGPR in Wave32 and an SGPR pair in Wave64, and RDNA decoders
size it for Wave64. Generated constructors mark it with ``set_wave_mask()`` so a
consumer that knows the wave size can narrow it. A genuine 64-bit SGPR value on
the same instruction is ``FMT_NUM_U64`` and stays unmarked.
"""

from pathlib import Path

import pytest

from amdisa.codegen._generator import CodeGenerator
from amdisa.gpuisa import Operand
from amdisa.isa_profile import Rdna4Profile
from amdisa.parser import Parser

_RDNA4_XML = (
    Path(__file__).resolve().parents[6]
    / 'shared'
    / 'machine-readable-isa'
    / 'isa'
    / 'amdgpu_isa_rdna4.xml'
)


@pytest.fixture(scope='module')
def rdna4_operands():
    spec = Parser(str(_RDNA4_XML), Rdna4Profile()).parse()
    return {
        (enc.enc_name, inst.name): {opnd.name: opnd for opnd in inst.operands}
        for enc in spec.inst_encodings
        for inst in enc.insts
    }


@pytest.mark.parametrize(
    'encoding, instruction, operand',
    [
        ('ENC_VOP3', 'V_CMP_EQ_U32', 'vdst'),
        ('ENC_VOP3', 'V_CNDMASK_B32', 'src2'),
        ('VOP3_SDST_ENC', 'V_ADD_CO_CI_U32', 'sdst'),
        ('VOP3_SDST_ENC', 'V_ADD_CO_CI_U32', 'src2'),
        ('VOP3_SDST_ENC', 'V_MAD_CO_U64_U32', 'sdst'),
    ],
)
def test_m64_operands_are_wave_masks(rdna4_operands, encoding, instruction, operand):
    assert rdna4_operands[(encoding, instruction)][operand].wave_mask


@pytest.mark.parametrize(
    'encoding, instruction, operand',
    [
        ('VOP3_SDST_ENC', 'V_MAD_CO_U64_U32', 'src2'),
        ('ENC_VOP3', 'V_CMP_EQ_U64', 'src0'),
    ],
)
def test_u64_sgpr_operands_are_not_wave_masks(
    rdna4_operands, encoding, instruction, operand
):
    assert not rdna4_operands[(encoding, instruction)][operand].wave_mask


def test_fieldless_vcc_destination_keeps_its_m64_flag(rdna4_operands):
    # Blank data_format_name must not hide the flag: SDWA sd=1 turns this
    # destination into an SGPR on RDNA1/2.
    vcc = rdna4_operands[('ENC_VOPC', 'V_CMP_EQ_U32')]['vcc']
    assert vcc.fieldless
    assert vcc.wave_mask


def _sgpr_operand(name, wave_mask, fieldless=False):
    return Operand(
        name, 64, 'OPR_SREG', False, True, False, False, 0, '', fieldless, wave_mask
    )


def test_wave32_capable_profile_marks_field_bearing_wave_masks():
    operands = [
        _sgpr_operand('sdst', wave_mask=True),
        _sgpr_operand('src2', wave_mask=False),
        _sgpr_operand('vcc', wave_mask=True, fieldless=True),
    ]
    assert CodeGenerator._wave_mask_stmts(operands, wave32_capable=True) == [
        'sdst.set_wave_mask();'
    ]


def test_wave64_only_profile_marks_nothing():
    operands = [_sgpr_operand('sdst', wave_mask=True)]
    assert CodeGenerator._wave_mask_stmts(operands, wave32_capable=False) == []

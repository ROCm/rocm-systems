# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the execution-block and instruction-class flag emission.

Instruction::amdgpu_execution_block(), is_transcendental(), suspends_wave()
and writes_pc() read these flags instead of matching mnemonics, so an
instruction the generator does not flag is silently misclassified.
"""

import pytest

from amdisa.codegen._generator import (
    _execution_block_flag_stmt,
    _instruction_class_flag_stmts,
)
from amdisa.semantics import InstructionSemantics


def _flags(name, sem):
    """Return the set of flag names the generator emits for one instruction."""
    return {s[len('flags_ |= ') : -1] for s in _instruction_class_flag_stmts(name, sem)}


@pytest.mark.parametrize(
    'enc_name, block',
    [
        ('ENC_SOP1', 'SALU'),
        ('ENC_SOPP', 'SALU'),
        # SMEM, not SALU: s_memtime and s_memrealtime included.
        ('ENC_SMEM', 'SMEM'),
        ('ENC_VOP3', 'VALU'),
        ('VOP3P_MFMA', 'VALU'),
        ('ENC_VINTRP', 'VALU'),
        ('ENC_DS', 'LDS'),
        ('ENC_LDSDIR', 'LDS'),
        ('ENC_VDSDIR', 'LDS'),
        ('ENC_FLAT', 'VMEM'),
        # Carries the CDNA5 cluster loads.
        ('ENC_VGLOBAL', 'VMEM'),
        # Carries the CDNA5 tensor transfers.
        ('ENC_VIMAGE', 'VMEM'),
        ('ENC_VSAMPLE', 'VMEM'),
    ],
)
def test_encoding_sets_its_block(enc_name, block):
    assert _execution_block_flag_stmt(enc_name) == f' flags_ |= {block};'


@pytest.mark.parametrize('enc_name', ['ENC_EXP', 'ENC_VEXPORT'])
def test_exports_issue_to_no_block(enc_name):
    assert _execution_block_flag_stmt(enc_name) == ''


def test_unclassified_encoding_fails_generation():
    with pytest.raises(ValueError, match='ENC_NEW'):
        _execution_block_flag_stmt('ENC_NEW')


@pytest.mark.parametrize(
    'sem',
    [
        InstructionSemantics('V_RCP_F32', 'vector_unary', operation='rcp'),
        InstructionSemantics('V_RCP_IFLAG_F32', 'vector_unary', operation='rcp_iflag'),
        InstructionSemantics('V_RSQ_F64', 'vector_unary', operation='rsq'),
        InstructionSemantics('V_SQRT_BF16', 'vector_unary', operation='sqrt'),
        InstructionSemantics('V_EXP_LEGACY_F32', 'vector_unary', operation='exp2'),
        InstructionSemantics('V_LOG_F16', 'vector_unary', operation='log2'),
        InstructionSemantics('V_SIN_F32', 'vector_unary', operation='sin'),
        InstructionSemantics('V_COS_F16', 'vector_unary', operation='cos'),
        InstructionSemantics('V_S_EXP_F32', 'pseudo_scalar_unary', operation='exp2'),
    ],
    ids=lambda sem: sem.name,
)
def test_transcendentals_are_flagged(sem):
    assert _flags(sem.name, sem) == {'TRANSCENDENTAL'}


@pytest.mark.parametrize(
    'sem',
    [
        InstructionSemantics('V_MOV_B32', 'vector_mov', data_type='b32'),
        InstructionSemantics('V_FRACT_F32', 'vector_unary', operation='fract'),
        InstructionSemantics('V_ADD_F32', 'vector_binop', operation='add'),
    ],
    ids=lambda sem: sem.name,
)
def test_other_vector_ops_are_not_transcendental(sem):
    assert _flags(sem.name, sem) == set()


@pytest.mark.parametrize(
    'name', ['S_SLEEP', 'S_SLEEP_VAR', 'S_SETHALT', 'S_SENDMSGHALT']
)
def test_wave_suspenders_are_flagged(name):
    assert _flags(name, InstructionSemantics(name, 'true_nop')) == {'SUSPENDS_WAVE'}


@pytest.mark.parametrize(
    'name', ['S_TRAP', 'S_RFE_B64', 'S_RFE_I64', 'S_RFE_RESTORE_B64']
)
def test_trap_entry_and_return_are_flagged(name):
    assert _flags(name, InstructionSemantics(name, 'true_nop')) == {'TRAP_CONTROL'}


def test_flagless_instruction_without_semantics():
    assert _flags('S_NOP', None) == set()

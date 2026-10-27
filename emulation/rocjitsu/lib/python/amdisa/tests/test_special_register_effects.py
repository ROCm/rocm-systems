# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage is a positive contract, never inferred from absent effects/stubs."""

from types import SimpleNamespace

import pytest

from amdisa.codegen._generator import CodeGenerator
from amdisa.codegen.special_register_effects import special_register_write_flags
from amdisa.sema_ast import ExecModel, SemaBlock, SemaNode, SemaNodeKind
from amdisa.semantics import InstructionSemantics, derive_semantics


def flags(name, encoding='ENC_SOP1', *, cmpx_vcc=False, outputs=1):
    inst = SimpleNamespace(
        enc_name=encoding,
        operands=[SimpleNamespace(is_input=True, is_output=False, fieldless=False)]
        + [SimpleNamespace(is_input=False, is_output=True, fieldless=False)] * outputs,
    )
    sem = derive_semantics(name, encoding)
    return set(
        special_register_write_flags(
            inst, sem, SimpleNamespace(cmpx_writes_vcc=cmpx_vcc)
        )
    )


KNOWN = {'flags_ |= EXEC_WRITE_EFFECTS_KNOWN;', 'flags_ |= VCC_WRITE_EFFECTS_KNOWN;'}


@pytest.mark.parametrize(
    'name,encoding',
    [
        ('S_NOP', 'ENC_SOPP'),
        ('V_NOP', 'ENC_VOP1'),
        ('S_MOV_B32', 'ENC_SOP1'),
        ('V_MOV_B32', 'ENC_VOP1'),
        ('S_ADD_U32', 'ENC_SOP2'),
        ('V_ADD_F32', 'ENC_VOP2'),
        ('S_WAITCNT', 'ENC_SOPP'),
        ('S_WAIT_ALU', 'ENC_SOPP'),
        ('S_BCNT1_I32_B64', 'ENC_SOP1'),
        ('S_FF1_I32_B64', 'ENC_SOP1'),
        ('S_CTZ_I32_B64', 'ENC_SOP1'),
        ('GLOBAL_LOAD_DWORD', 'ENC_FLAT'),
        ('GLOBAL_STORE_DWORD', 'ENC_FLAT'),
    ],
)
def test_supported_families_have_explicit_coverage(name, encoding):
    assert KNOWN <= flags(name, encoding)


@pytest.mark.parametrize(
    'name,encoding',
    [
        ('S_RFE_B64', 'ENC_SOP1'),
        ('S_SETPRIO', 'ENC_SOPP'),
        ('S_SENDMSG', 'ENC_SOPP'),
        ('S_TRAP', 'ENC_SOPP'),
        ('S_SETREG_B32', 'ENC_SOPK'),
        ('V_NOT_A_REAL_INSTRUCTION', 'ENC_VOP1'),
    ],
)
def test_stub_and_system_effects_are_unknown(name, encoding):
    assert not flags(name, encoding)


def test_no_semantics_is_unknown():
    assert special_register_write_flags(SimpleNamespace(), None, None) == []


def test_input_only_operand_cannot_substitute_for_a_destination():
    assert flags('S_MOV_B32', outputs=0) == set()


def test_new_opaque_helper_in_known_family_is_unknown(monkeypatch):
    call = SemaNode(SemaNodeKind.CALL, call_name='future_opaque_helper')
    monkeypatch.setattr(
        'amdisa.codegen.special_register_effects.derive_sema_block',
        lambda sem: SemaBlock(sem.name, ExecModel.SCALAR, call),
    )
    assert flags('S_MOV_B32') == set()


def test_empty_derivation_is_unknown_even_for_a_known_family(monkeypatch):
    monkeypatch.setattr(
        'amdisa.codegen.special_register_effects.derive_sema_block',
        lambda sem: SemaBlock(
            sem.name, ExecModel.SCALAR, SemaNode(SemaNodeKind.SEQ, children=())
        ),
    )
    assert flags('S_MOV_B32') == set()


def test_cmpx_target_difference_and_vop3_destination_redirection():
    cdna = flags('V_CMPX_EQ_U32', 'ENC_VOPC', cmpx_vcc=True)
    rdna = flags('V_CMPX_EQ_U32', 'ENC_VOPC', cmpx_vcc=False)
    assert KNOWN <= cdna and KNOWN <= rdna
    assert 'flags_ |= WRITES_EXEC;' in rdna
    assert 'flags_ |= WRITES_VCC;' in cdna
    assert 'flags_ |= WRITES_VCC;' not in rdna
    assert 'flags_ |= WRITES_VCC;' in flags('V_CMP_EQ_U32', 'ENC_VOPC')
    assert 'flags_ |= WRITES_VCC;' not in flags('V_CMP_EQ_U32', 'ENC_VOP3')


@pytest.mark.parametrize(
    'name,cls,index',
    [
        ('exec_lo', 'EXEC', 0),
        ('exec_hi', 'EXEC', 1),
        ('vcc_lo', 'VCC', 0),
        ('VCC_HI', 'VCC', 1),
        ('m0', 'M0', 0),
        ('flat_scratch_hi', 'FLAT_SCRATCH', 1),
    ],
)
def test_special_selectors_are_structural(name, cls, index):
    assert CodeGenerator._named_reg_ref(name) == (f'RegClass::{cls}', index)


@pytest.mark.parametrize(
    'register,flag', [('EXEC', 'WRITES_EXEC'), ('VCC', 'WRITES_VCC')]
)
def test_direct_context_writes_are_positive_effects(monkeypatch, register, flag):
    assignment = SemaNode(
        SemaNodeKind.ASSIGN,
        children=(
            SemaNode(SemaNodeKind.ID, id_name=register),
            SemaNode(SemaNodeKind.LIT, lit_value='0'),
        ),
    )
    monkeypatch.setattr(
        'amdisa.codegen.special_register_effects.derive_sema_block',
        lambda sem: SemaBlock(sem.name, ExecModel.SCALAR, assignment),
    )
    assert f'flags_ |= {flag};' in flags('S_MOV_B32')

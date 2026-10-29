# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage is a positive contract, never inferred from absent effects/stubs."""

from types import SimpleNamespace
from pathlib import Path

import pytest

from amdisa.codegen._generator import CodeGenerator, _exec_mask_flag_stmts
from amdisa.codegen.special_register_effects import special_register_write_flags
from amdisa.sema_ast import ExecModel, SemaBlock, SemaNode, SemaNodeKind
from amdisa.semantics import derive_all_semantics, derive_semantics
from amdisa.isa_profile import CdnaProfile, Cdna5Profile, Rdna4Profile
from amdisa.parser import Parser


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
        ('S_CBRANCH_G_FORK', 'ENC_SOP2'),
        ('S_CBRANCH_I_FORK', 'ENC_SOPK'),
        ('S_CBRANCH_JOIN', 'ENC_SOP1'),
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


@pytest.mark.parametrize(
    'cls,name,output,expected',
    [
        ('vector_cmp', 'vdst', True, True),
        ('vector_cndmask', 'src2', False, True),
        ('vector_add_co', 'sdst', True, True),
        ('vector_add_co', 'src2', False, True),
        ('vector_div_scale', 'sdst', True, True),
        ('vector_mad_64_32', 'sdst', True, True),
        ('vector_binop', 'src0', False, False),
        ('vector_cmp', 'src0', False, False),
    ],
)
def test_wave_mask_extent_is_not_a_fixed_scalar_pair(cls, name, output, expected):
    operand = SimpleNamespace(name=name, is_output=output)
    semantics = SimpleNamespace(semantic_class=cls)
    assert (
        CodeGenerator._is_wave_mask_operand('ENC_VOP3', operand, semantics) == expected
    )
    assert not CodeGenerator._is_wave_mask_operand('ENC_SOP2', operand, semantics)


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


@pytest.mark.parametrize('kind', [SemaNodeKind.ADD_ASSIGN, SemaNodeKind.SUB_ASSIGN])
def test_compound_exec_write_cannot_strengthen_legacy_metadata(monkeypatch, kind):
    bit = SemaNode(
        SemaNodeKind.ARRAYDEREF,
        children=(
            SemaNode(SemaNodeKind.ID, id_name='EXEC_RAW'),
            SemaNode(SemaNodeKind.LIT, lit_value='0'),
        ),
    )
    block = SemaBlock(
        'S_MOV_B32',
        ExecModel.SCALAR,
        SemaNode(kind, children=(bit, SemaNode(SemaNodeKind.LIT, lit_value='1'))),
    )
    monkeypatch.setattr(
        'amdisa.codegen.special_register_effects.derive_sema_block', lambda sem: block
    )
    actual = flags('S_MOV_B32')
    assert 'flags_ |= EXEC_WRITE_EFFECTS_KNOWN;' not in actual
    assert 'flags_ |= WRITES_EXEC;' not in actual
    assert 'flags_ |= VCC_WRITE_EFFECTS_KNOWN;' in actual


def test_cmpx_target_difference_and_vop3_destination_redirection():
    cdna = flags('V_CMPX_EQ_U32', 'ENC_VOPC', cmpx_vcc=True)
    rdna = flags('V_CMPX_EQ_U32', 'ENC_VOPC', cmpx_vcc=False)
    assert KNOWN <= cdna and KNOWN <= rdna
    assert 'flags_ |= WRITES_EXEC;' in _exec_mask_flag_stmts(
        derive_semantics('V_CMPX_EQ_U32', 'ENC_VOPC')
    )
    assert 'flags_ |= WRITES_EXEC;' not in rdna
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
def test_context_writes_feed_separate_metadata_consumers(monkeypatch, register, flag):
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
    monkeypatch.setattr(
        'amdisa.sema_derive.derive_sema_block',
        lambda sem: SemaBlock(sem.name, ExecModel.SCALAR, assignment),
    )
    coverage = flags('S_MOV_B32')
    properties = set(_exec_mask_flag_stmts(derive_semantics('S_MOV_B32', 'ENC_SOP1')))
    assert f'flags_ |= {flag};' in coverage | properties
    assert 'flags_ |= WRITES_EXEC;' not in coverage
    assert KNOWN <= coverage


def test_nested_exec_write_without_established_property_loses_coverage(monkeypatch):
    destination = SemaNode(SemaNodeKind.ID, id_name='EXEC_RAW')
    for _ in range(2):
        destination = SemaNode(
            SemaNodeKind.CAST,
            children=(
                SemaNode(
                    SemaNodeKind.ARRAYDEREF,
                    children=(destination, SemaNode(SemaNodeKind.LIT, lit_value='0')),
                ),
            ),
        )
    block = SemaBlock(
        'S_MOV_B32',
        ExecModel.SCALAR,
        SemaNode(
            SemaNodeKind.ASSIGN,
            children=(destination, SemaNode(SemaNodeKind.LIT, lit_value='1')),
        ),
    )
    monkeypatch.setattr(
        'amdisa.codegen.special_register_effects.derive_sema_block', lambda sem: block
    )
    actual = flags('S_MOV_B32')
    assert 'flags_ |= EXEC_WRITE_EFFECTS_KNOWN;' not in actual
    assert 'flags_ |= WRITES_EXEC;' not in actual
    assert 'flags_ |= VCC_WRITE_EFFECTS_KNOWN;' in actual


@pytest.mark.parametrize(
    'arch,profile_type',
    [
        ('cdna3', CdnaProfile),
        ('rdna4', Rdna4Profile),
        ('cdna5', Cdna5Profile),
    ],
)
def test_absolute_return_classification_survives_missing_mask_coverage(
    tmp_path,
    monkeypatch,
    arch,
    profile_type,
):
    xml = (
        Path(__file__).resolve().parents[6]
        / 'shared/machine-readable-isa/isa'
        / f'amdgpu_isa_{arch}.xml'
    )
    spec = Parser(str(xml), profile_type()).parse()
    transfers = []
    for encoding in spec.inst_encodings:
        encoding.insts = [
            inst
            for inst in encoding.insts
            if (sem := derive_semantics(inst.name, inst.enc_name)) is not None
            and sem.semantic_class
            in ('scalar_setpc', 'scalar_addpc', 'scalar_getpc', 'scalar_swappc')
        ]
        transfers.extend(encoding.insts)
    assert any(
        derive_semantics(inst.name, inst.enc_name).semantic_class == 'scalar_setpc'
        for inst in transfers
    )
    semantics = derive_all_semantics(spec)
    monkeypatch.setattr(
        'amdisa.codegen._generator.special_register_write_flags', lambda *args: []
    )
    CodeGenerator(spec, str(tmp_path), semantics).gen_insts()
    cpp = (tmp_path / arch / 'sop1.cpp').read_text()
    for inst in transfers:
        constructor = cpp.split(f'{inst.fmt_name}::{inst.fmt_name}(', 1)[1].split(
            'namespace detail', 1
        )[0]
        assert 'EXEC_WRITE_EFFECTS_KNOWN' not in constructor
        assert 'VCC_WRITE_EFFECTS_KNOWN' not in constructor
        absolute = semantics.instructions[inst.name].semantic_class == 'scalar_setpc'
        assert ('flags_ |= ABSOLUTE_INDIRECT_BRANCH;' in constructor) == absolute

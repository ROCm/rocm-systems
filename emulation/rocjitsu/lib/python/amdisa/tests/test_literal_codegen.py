# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for implied-literal constructor fixups."""

from types import SimpleNamespace

from amdisa.codegen._generator import CodeGenerator
from amdisa.gpuisa import InstEncoding, Instruction, Operand


def _enc(name: str) -> InstEncoding:
    return InstEncoding(
        name,
        order=0,
        bit_cnt=32,
        enc_field_bit_cnt=6,
        op_field_bit_cnt=8,
        ucode_fields=[],
        enc_conds=[],
    )


def _literal_operand(size: int, operand_type: str) -> Operand:
    return Operand(
        'literal',
        size,
        operand_type,
        is_input=True,
        is_output=False,
        is_implicit=False,
        is_binary_ucode_required=True,
        order=2,
    )


def test_implied_literal_uses_parent_encoding_literal_struct():
    inst = Instruction(
        'S_FMAAK_F32',
        'SOP2_INST_LITERAL',
        opcode=69,
        operands=[],
        is_implied_literal_enc=True,
    )

    info = CodeGenerator._literal_encoding_info(
        _enc('ENC_SOP2'), _enc('SOP2_INST_LITERAL'), inst
    )

    assert info == ('Sop2InstLiteralMachineInst', ('ssrc0', 'ssrc1'))


def test_literal_fixups_require_generated_machine_inst_struct():
    inst = Instruction('V_PK_ADD_I16', 'ENC_VOP3P', opcode=0, operands=[])

    info = CodeGenerator._literal_encoding_info(_enc('ENC_VOP3P'), None, inst)

    assert info == ('Vop3pInstLiteralMachineInst', ('src0', 'src1', 'src2'))

    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(inst_encodings=[_enc('VOP3P_INST_LITERAL')])
    assert codegen._has_machine_inst_struct(info[0])

    codegen.isa_spec = SimpleNamespace(inst_encodings=[_enc('ENC_VOP3P')])
    assert not codegen._has_machine_inst_struct(info[0])


def test_simm32_literal_operand_is_initialized_from_extension_word():
    stmt = CodeGenerator._literal_operand_fixup_stmt(
        _literal_operand(32, 'OPR_SIMM32'), 'Sop2InstLiteralMachineInst'
    )

    assert (
        'literal = Operand(32, OperandType::OPR_SIMM32, '
        'static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));'
    ) == stmt


def test_simm16_literal_operand_uses_low_half_of_extension_word():
    stmt = CodeGenerator._literal_operand_fixup_stmt(
        _literal_operand(16, 'OPR_SIMM16'), 'Vop2InstLiteralMachineInst'
    )

    assert (
        'literal = Operand(16, OperandType::OPR_SIMM16, '
        'static_cast<int>((reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32 '
        '& 0xFFFFu)));'
    ) == stmt


def test_existing_literal_operand_does_not_need_simm32_fallback_member():
    inst = Instruction(
        'V_FMAAK_F32',
        'VOP2_INST_LITERAL',
        opcode=45,
        operands=[_literal_operand(32, 'OPR_SIMM32')],
        is_implied_literal_enc=True,
    )

    assert CodeGenerator._has_inline_literal_operand(inst)


def test_scalar_literal_fma_synthesizes_third_source_operand():
    inst = Instruction(
        'S_FMAAK_F32',
        'SOP2_INST_LITERAL',
        opcode=69,
        operands=[],
        is_implied_literal_enc=True,
    )

    fixed = CodeGenerator._with_scalar_literal_fma_operand(inst)

    assert fixed is not inst
    assert fixed.operands[-1].name == 'src2'
    assert fixed.operands[-1].operand_type == 'OPR_SIMM32'
    assert fixed.operands[-1].is_input


def test_literal_fma_mnemonics_are_not_shared_across_isa_layouts():
    assert 's_fmaak_f32' in CodeGenerator._NON_SHAREABLE_MNEMONICS
    assert 's_fmamk_f32' in CodeGenerator._NON_SHAREABLE_MNEMONICS

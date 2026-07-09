# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for implied-literal constructor fixups."""

from types import SimpleNamespace

from amdisa.codegen._generator import CodeGenerator
from amdisa.gpuisa import InstEncoding, Instruction, Operand
from amdisa.isa_profile import Rdna4Profile
from amdisa.semantics import InstructionSemantics


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


def _operand(
    name: str,
    operand_type: str,
    *,
    size: int = 32,
    is_input: bool = True,
    is_output: bool = False,
    order: int = 0,
) -> Operand:
    return Operand(
        name,
        size,
        operand_type,
        is_input=is_input,
        is_output=is_output,
        is_implicit=False,
        is_binary_ucode_required=False,
        order=order,
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


def test_scalar_fmamk_semantic_sources_use_synthesized_literal_as_multiplier():
    operands = [
        _operand('sdst', 'OPR_SDST', is_input=False, is_output=True, order=0),
        _operand('ssrc0', 'OPR_SSRC', order=1),
        _operand('ssrc1', 'OPR_SSRC', order=2),
        _operand('src2', 'OPR_SIMM32', order=3),
    ]
    inst = Instruction('S_FMAMK_F32', 'ENC_SOP2', opcode=70, operands=operands)

    ordered = CodeGenerator._semantic_source_operands(inst, operands[1:])

    assert [op.name for op in ordered] == ['ssrc0', 'src2', 'ssrc1']


def test_scalar_fmamk_semantic_sources_keep_explicit_literal_as_multiplier():
    operands = [
        _operand('ssrc0', 'OPR_SSRC', order=0),
        _operand('literal', 'OPR_SIMM32', order=1),
        _operand('ssrc1', 'OPR_SSRC', order=2),
        _operand('src2', 'OPR_SIMM32', order=3),
    ]
    inst = Instruction('S_FMAMK_F32', 'ENC_SOP2', opcode=70, operands=operands)

    ordered = CodeGenerator._semantic_source_operands(inst, operands)

    assert [op.name for op in ordered] == ['ssrc0', 'literal', 'ssrc1', 'src2']


def test_scalar_fmamk_generated_execute_uses_literal_multiplier():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name='rdna4',
        profile=Rdna4Profile(),
        inst_encodings=[],
        encoding_map={},
    )
    operands = [
        _operand('sdst', 'OPR_SDST', is_input=False, is_output=True, order=0),
        _operand('ssrc0', 'OPR_SSRC', order=1),
        _operand('ssrc1', 'OPR_SSRC', order=2),
        _operand('src2', 'OPR_SIMM32', order=3),
    ]
    inst = Instruction('S_FMAMK_F32', 'ENC_SOP2', opcode=70, operands=operands)
    sem = InstructionSemantics(
        'S_FMAMK_F32',
        'scalar_binop',
        operation='fma',
        data_type='f32',
        sets_scc='none',
    )

    body = codegen._gen_execute_body(inst, sem, 'ENC_SOP2')

    assert 'std::bit_cast<float>(ssrc0.read_scalar(wf))' in body
    assert 'std::bit_cast<float>(src2.read_scalar(wf))' in body
    assert 'std::bit_cast<float>(ssrc1.read_scalar(wf))' in body
    assert body.index('src2.read_scalar(wf)') < body.index('ssrc1.read_scalar(wf)')


def test_scalar_mul_u64_generated_execute_reads_full_source_pairs():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name='rdna4',
        profile=Rdna4Profile(),
        inst_encodings=[],
        encoding_map={},
    )
    operands = [
        _operand(
            'sdst',
            'OPR_SDST',
            size=64,
            is_input=False,
            is_output=True,
            order=0,
        ),
        _operand('ssrc0', 'OPR_SSRC', size=64, order=1),
        _operand('ssrc1', 'OPR_SSRC', size=64, order=2),
    ]
    inst = Instruction('S_MUL_U64', 'ENC_SOP2', opcode=68, operands=operands)
    sem = InstructionSemantics(
        'S_MUL_U64',
        'scalar_binop',
        operation='mul',
        data_type='u64',
        sets_scc='none',
    )

    body = codegen._gen_execute_body(inst, sem, 'ENC_SOP2')

    assert 'ssrc0.read_scalar64(wf)' in body
    assert 'ssrc1.read_scalar64(wf)' in body
    assert 'sdst.write_scalar64(wf, result);' in body
    assert 'ssrc0.read_scalar(wf)' not in body
    assert 'ssrc1.read_scalar(wf)' not in body


def test_literal_fma_mnemonics_are_not_shared_across_isa_layouts():
    assert 's_fmaak_f32' in CodeGenerator._NON_SHAREABLE_MNEMONICS
    assert 's_fmamk_f32' in CodeGenerator._NON_SHAREABLE_MNEMONICS

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for implied-literal constructor fixups."""

from types import SimpleNamespace

import pytest

from amdisa.codegen._generator import CodeGenerator
from amdisa.codegen.execute.simd_codegen import (
    SIMD_VOP2_TERNARY,
    simd_ternary_literal_operand_name,
)
from amdisa.cross_isa import SharedInstInfo, SharedInstructionPlan
from amdisa.gpuisa import InstEncoding, Instruction, Operand
from amdisa.isa_profile import Cdna5Profile, Rdna4Profile
from amdisa.parser import Parser, _uniquify_fieldless_names
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


def _literal_operand(
    size: int, operand_type: str, data_format_name: str = ''
) -> Operand:
    return Operand(
        'literal',
        size,
        operand_type,
        is_input=True,
        is_output=False,
        is_implicit=False,
        is_binary_ucode_required=True,
        order=2,
        data_format_name=data_format_name,
    )


def _operand(
    name: str,
    operand_type: str,
    *,
    size: int = 32,
    is_input: bool = True,
    is_output: bool = False,
    order: int = 0,
    data_format_name: str = '',
    fieldless: bool = False,
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
        data_format_name=data_format_name,
        fieldless=fieldless,
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


def test_implied_literal64_uses_its_three_dword_machine_inst():
    inst = Instruction(
        'V_FMAMK_F64',
        'VOP2_INST_LITERAL64',
        opcode=35,
        operands=[],
        is_implied_literal_enc=True,
    )

    info = CodeGenerator._literal_encoding_info(
        _enc('ENC_VOP2'), _enc('VOP2_INST_LITERAL64'), inst
    )

    assert info == ('Vop2InstLiteral64MachineInst', ('src0',))


def test_implied_literal_extension_preserves_literal64_width():
    parent = _enc('ENC_VOP2')
    literal64 = _enc('VOP2_INST_LITERAL64')
    literal64.bit_cnt = 96

    words = Parser.implied_literal_extension_words(literal64, parent)
    parent.implied_literal_ops['35'] = words

    assert words == 2
    assert parent.has_implied_literal_ops
    assert parent.has_variable_implied_literal_size


def test_implied_literal_extension_keeps_literal32_as_one_word():
    parent = _enc('ENC_VOP2')
    literal32 = _enc('VOP2_INST_LITERAL')
    literal32.bit_cnt = 64

    words = Parser.implied_literal_extension_words(literal32, parent)
    parent.implied_literal_ops['44'] = words

    assert words == 1
    assert not parent.has_variable_implied_literal_size


def test_vop3p_literal64_rejection_uses_complete_encoding_capability():
    unsupported = _enc('ENC_VOP3P')
    assert CodeGenerator._rejects_unencoded_vop3p_literal64(unsupported)

    explicit = _enc('ENC_VOP3P')
    explicit.enc_conds.append(('has_lit64_0', 'inst_.src0 == 254'))
    assert not CodeGenerator._rejects_unencoded_vop3p_literal64(explicit)

    implied = _enc('ENC_VOP3P')
    implied.insts.append(
        Instruction(
            'V_FUTURE_LITERAL64',
            'VOP3P_INST_LITERAL64',
            opcode=1,
            operands=[],
            is_implied_literal_enc=True,
        )
    )
    assert not CodeGenerator._rejects_unencoded_vop3p_literal64(implied)


def test_gfx1250_packed_f32_reader_has_no_unreachable_literal64_branch():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        inst_encodings=[
            SimpleNamespace(
                enc_name='ENC_VOP3P',
                ucode_fields=[SimpleNamespace(bit_offset=14, name='opsel_hi_2')],
            )
        ]
    )
    source = codegen._emit_cdna5_matrix_fmt_helpers().execution[0]

    reader_start = source.index('PkF32Words read_pk_f32_words')
    reader_end = source.index('\n}', reader_start)
    reader = source[reader_start:reader_end]
    assert 'literal64_value' not in reader
    assert 'read_lane_pair32(operand, lane)' in reader
    assert 'return {pair.lo, pair.hi};' in reader


def test_literal_fixups_require_generated_machine_inst_struct():
    inst = Instruction('V_PK_ADD_I16', 'ENC_VOP3P', opcode=0, operands=[])

    info = CodeGenerator._literal_encoding_info(_enc('ENC_VOP3P'), None, inst)

    assert info == ('Vop3pInstLiteralMachineInst', ('src0', 'src1', 'src2'))

    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(inst_encodings=[_enc('VOP3P_INST_LITERAL')])
    assert codegen._has_machine_inst_struct(info[0])

    codegen.isa_spec = SimpleNamespace(inst_encodings=[_enc('ENC_VOP3P')])
    assert not codegen._has_machine_inst_struct(info[0])


def test_literal_extension_fields_follow_each_opcodes_operands():
    enc = _enc('ENC_VOP3P')
    enc.insts = [
        Instruction(
            'V_PK_ADD_I16',
            'ENC_VOP3P',
            opcode=2,
            operands=[
                _operand('src0', 'OPR_SRC'),
                _operand('src1', 'OPR_SRC'),
            ],
        ),
        Instruction('V_NOP', 'ENC_VOP3P', opcode=384, operands=[]),
    ]

    masks = CodeGenerator._encoded_literal_field_masks(enc, ('src0', 'src1', 'src2'))

    assert masks == {2: ('src0', 'src1'), 384: ()}


def test_only_literal_capable_operand_fields_select_extension_words():
    enc = _enc('ENC_SOPC')
    enc.insts = [
        Instruction(
            'S_SET_GPR_IDX_ON',
            'ENC_SOPC',
            opcode=17,
            operands=[
                _operand('ssrc0', 'OPR_SSRC'),
                _operand('ssrc1', 'OPR_SIMM4'),
            ],
        ),
        Instruction(
            'S_CBRANCH_G_FORK',
            'ENC_SOPC',
            opcode=18,
            operands=[
                _operand('ssrc0', 'OPR_SSRC_NOLIT'),
                _operand('ssrc1', 'OPR_SRC_NOLIT'),
            ],
        ),
        Instruction(
            'V_READFIRSTLANE_B32',
            'ENC_SOPC',
            opcode=19,
            operands=[_operand('ssrc0', 'OPR_VGPR')],
        ),
    ]

    masks = CodeGenerator._encoded_literal_field_masks(enc, ('ssrc0', 'ssrc1'))
    helper = CodeGenerator._encoded_literal_helper_impl(enc, ('ssrc0', 'ssrc1'), 255)

    assert masks == {17: ('ssrc0',), 18: (), 19: ()}
    assert 'inst_.ssrc0 == 255' in helper
    assert 'inst_.ssrc1 == 255' not in helper


def test_simm32_literal_operand_is_initialized_from_extension_word():
    stmt = CodeGenerator._literal_operand_fixup_stmt(
        _literal_operand(32, 'OPR_SIMM32'), 'Sop2InstLiteralMachineInst'
    )

    assert (
        'literal = Operand(32, OperandType::OPR_SIMM32, '
        'static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));'
    ) == stmt


def test_f64_simm32_literal_operand_uses_extension_word_as_double_high_bits():
    stmt = CodeGenerator._literal_operand_fixup_stmt(
        _operand(
            'src0',
            'OPR_SRC_VGPR',
            size=64,
            data_format_name='FMT_NUM_F64',
        ),
        'Vop2InstLiteralMachineInst',
        literal_operand_type='OPR_SIMM32',
    )

    assert (
        'src0 = Operand::make_literal32(64, static_cast<uint32_t>('
        'reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32), '
        'Operand::Literal32Widening::F64HighBits);'
    ) == stmt


def test_64bit_simm32_literal_operand_requires_data_format():
    with pytest.raises(
        ValueError,
        match=(
            "architecture 'rdna4', instruction 'V_FMAC_F64', encoding 'ENC_VOP2': "
            "64-bit SIMM32 input operand 'src0' has unsupported data format '<missing>'"
        ),
    ):
        CodeGenerator._literal_operand_fixup_stmt(
            _operand('src0', 'OPR_SRC_VGPR', size=64),
            'Vop2InstLiteralMachineInst',
            literal_operand_type='OPR_SIMM32',
            arch_name='rdna4',
            inst_name='V_FMAC_F64',
            enc_name='ENC_VOP2',
        )


def test_mixed_width_literal_operands_classify_per_operand_signature():
    src0_stmt = CodeGenerator._literal_operand_fixup_stmt(
        _operand(
            'src0',
            'OPR_SRC_NOLIT',
            size=64,
            data_format_name='FMT_NUM_F64',
        ),
        'Vop3InstLiteralMachineInst',
        literal_operand_type='OPR_SIMM32',
    )
    src1_stmt = CodeGenerator._literal_operand_fixup_stmt(
        _operand(
            'src1',
            'OPR_SRC_SIMPLE',
            size=32,
            data_format_name='FMT_NUM_B32',
        ),
        'Vop3InstLiteralMachineInst',
        literal_operand_type='OPR_SIMM32',
    )

    assert (
        'src0 = Operand::make_literal32(64, static_cast<uint32_t>('
        'reinterpret_cast<const Vop3InstLiteralMachineInst *>(inst)->simm32), '
        'Operand::Literal32Widening::F64HighBits);'
    ) == src0_stmt
    assert (
        'src1 = Operand(32, OperandType::OPR_SIMM32, '
        'static_cast<int>(reinterpret_cast<const Vop3InstLiteralMachineInst *>(inst)->simm32));'
    ) == src1_stmt


def test_u64_simm32_literal_operand_keeps_low_32_bit_value():
    stmt = CodeGenerator._literal_operand_fixup_stmt(
        _operand('ssrc0', 'OPR_SSRC', size=64, data_format_name='FMT_NUM_U64'),
        'Sop2InstLiteralMachineInst',
        literal_operand_type='OPR_SIMM32',
    )

    assert (
        'ssrc0 = Operand::make_literal32(64, static_cast<uint32_t>('
        'reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32), '
        'Operand::Literal32Widening::ZeroExtend);'
    ) == stmt


def test_pk_f32_simm32_literal_operand_replicates_extension_word():
    stmt = CodeGenerator._literal_operand_fixup_stmt(
        _operand(
            'src0',
            'OPR_SRC',
            size=64,
            data_format_name='FMT_NUM_PK2_F32',
        ),
        'Vop3pInstLiteralMachineInst',
        literal_operand_type='OPR_SIMM32',
    )

    assert (
        'src0 = Operand::make_literal32(64, static_cast<uint32_t>('
        'reinterpret_cast<const Vop3pInstLiteralMachineInst *>(inst)->simm32), '
        'Operand::Literal32Widening::Replicate32);'
    ) == stmt


def test_i64_simm32_literal_operand_sign_extends_from_data_format():
    stmt = CodeGenerator._literal_operand_fixup_stmt(
        _operand(
            'src0',
            'OPR_SRC',
            size=64,
            data_format_name='FMT_NUM_I64',
        ),
        'Vop3InstLiteralMachineInst',
        literal_operand_type='OPR_SIMM32',
    )

    assert (
        'src0 = Operand::make_literal32(64, '
        'static_cast<uint32_t>(reinterpret_cast<const '
        'Vop3InstLiteralMachineInst *>(inst)->simm32), '
        'Operand::Literal32Widening::SignExtend);'
    ) == stmt


def test_64bit_simm32_literal_operand_rejects_unrecognized_data_format():
    with pytest.raises(
        ValueError,
        match=(
            "architecture 'cdna4', instruction 'S_FAKE_I64', encoding 'ENC_SOP2': "
            "64-bit SIMM32 input operand 'ssrc0' has unsupported data format 'FMT_NUM_X64'"
        ),
    ):
        CodeGenerator._literal_operand_fixup_stmt(
            _operand(
                'ssrc0',
                'OPR_SSRC',
                size=64,
                data_format_name='FMT_NUM_X64',
            ),
            'Sop2InstLiteralMachineInst',
            literal_operand_type='OPR_SIMM32',
            arch_name='cdna4',
            inst_name='S_FAKE_I64',
            enc_name='ENC_SOP2',
        )


def test_mixed_width_i64_literal_fixup_only_sign_extends_i64_operand():
    operands = (
        _operand('src0', 'OPR_SRC', size=32, data_format_name='FMT_NUM_I32'),
        _operand('src1', 'OPR_SRC', size=32, data_format_name='FMT_NUM_I32'),
        _operand('src2', 'OPR_SRC', size=64, data_format_name='FMT_NUM_I64'),
    )

    stmts = [
        CodeGenerator._literal_operand_fixup_stmt(
            operand,
            'Vop3InstLiteralMachineInst',
            literal_operand_type='OPR_SIMM32',
        )
        for operand in operands
    ]

    assert all(stmt is not None for stmt in stmts)
    assert 'make_literal32' not in stmts[0]
    assert 'make_literal32' not in stmts[1]
    assert 'Operand::Literal32Widening::SignExtend' in stmts[2]


def test_generated_operand_tracks_literal32_widening_without_literal64_provenance(
    tmp_path,
):
    generator = CodeGenerator(
        SimpleNamespace(
            arch_name='rdna4',
            generated_dir_name='rdna4',
            cpp_namespace='rdna4',
            opnd_selectors=[],
            operand_types=['OPR_SIMM16', 'OPR_SIMM32', 'OPR_VGPR'],
            profile=Rdna4Profile(),
        ),
        str(tmp_path),
    )

    generator.gen_operand()
    operand_cpp = (tmp_path / 'rdna4' / 'operand.cpp').read_text()
    operand_exec_cpp = (tmp_path / 'rdna4' / 'operand_exec.cpp').read_text()
    operand_h = (tmp_path / 'rdna4' / 'operand.h').read_text()

    assert 'enum class Literal32Widening' in operand_h
    assert 'static Operand make_literal32(' in operand_h
    assert 'uint64_t signed_literal32_value_' not in operand_h
    assert 'uint64_t widened_literal32_value() const;' in operand_h
    assert 'std::optional<Literal32Widening> literal32_widening_' in operand_h
    assert 'literal32_display' not in operand_h
    assert 'Operand operand(size_bits, OperandType::OPR_SIMM32' in operand_cpp
    assert 'operand.literal32_widening_ = widening;' in operand_cpp
    assert operand_exec_cpp.count('if (literal32_widening_)') == 2
    assert operand_exec_cpp.count('return widened_literal32_value();') == 2
    assert 'case Literal32Widening::Replicate32:' in operand_cpp
    assert (
        'return (static_cast<uint64_t>(literal_value) << 32) | literal_value;'
        in operand_cpp
    )
    assert 'case Literal32Widening::F64HighBits:' in operand_cpp
    assert 'static_cast<uint32_t>(encoding_value_)' in operand_cpp
    assert 'if (!has_literal64_)\n    return std::nullopt;' in operand_cpp


def test_b64_simm32_literal_operand_keeps_low_32_bit_value():
    stmt = CodeGenerator._literal_operand_fixup_stmt(
        _operand('ssrc0', 'OPR_SSRC', size=64, data_format_name='FMT_NUM_B64'),
        'Sop2InstLiteralMachineInst',
        literal_operand_type='OPR_SIMM32',
    )

    assert (
        'ssrc0 = Operand::make_literal32(64, static_cast<uint32_t>('
        'reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32), '
        'Operand::Literal32Widening::ZeroExtend);'
    ) == stmt


def test_m64_simm32_literal_operand_keeps_low_32_bit_mask():
    stmt = CodeGenerator._literal_operand_fixup_stmt(
        _operand('src2', 'OPR_SRC', size=64, data_format_name='FMT_NUM_M64'),
        'Vop3InstLiteralMachineInst',
        literal_operand_type='OPR_SIMM32',
    )

    assert (
        'src2 = Operand::make_literal32(64, static_cast<uint32_t>('
        'reinterpret_cast<const Vop3InstLiteralMachineInst *>(inst)->simm32), '
        'Operand::Literal32Widening::ZeroExtend);'
    ) == stmt


def test_64bit_simm32_literal_preserves_effective_operand_size():
    stmt = CodeGenerator._literal_operand_fixup_stmt(
        _operand('src2', 'OPR_SREG', size=64, data_format_name='FMT_NUM_M64'),
        'Vop3InstLiteralMachineInst',
        size_expr='32',
        literal_operand_type='OPR_SIMM32',
    )

    assert (
        'src2 = Operand::make_literal32(32, static_cast<uint32_t>('
        'reinterpret_cast<const Vop3InstLiteralMachineInst *>(inst)->simm32), '
        'Operand::Literal32Widening::ZeroExtend);'
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


def test_simm64_literal_operand_reads_both_unaligned_extension_words():
    stmt = CodeGenerator._literal_operand_fixup_stmt(
        _literal_operand(64, 'OPR_SIMM64'), 'Vop2InstLiteral64MachineInst'
    )

    assert (
        'literal = Operand(64, OperandType::OPR_SIMM64, '
        '(static_cast<uint64_t>(reinterpret_cast<const uint32_t *>(inst)[2]) << 32) | '
        'reinterpret_cast<const uint32_t *>(inst)[1], true);'
    ) == stmt


def test_declared_16bit_simm32_literal_uses_low_half_of_extension_word():
    stmt = CodeGenerator._literal_operand_fixup_stmt(
        _literal_operand(16, 'OPR_SIMM32'), 'Vop2InstLiteralMachineInst'
    )

    assert (
        'literal = Operand(16, OperandType::OPR_SIMM32, '
        'static_cast<int>((reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32 '
        '& 0xFFFFu)));'
    ) == stmt


def test_non_opsel_16bit_retyped_simm32_literal_uses_low_half():
    stmt = CodeGenerator._literal_operand_fixup_stmt(
        _operand('src0', 'OPR_SRC', size=16),
        'Vop2InstLiteralMachineInst',
        literal_operand_type='OPR_SIMM32',
    )

    assert (
        'src0 = Operand(16, OperandType::OPR_SIMM32, '
        'static_cast<int>((reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32 '
        '& 0xFFFFu)));'
    ) == stmt


def test_dynamic_true16_simm32_literal_keeps_raw_and_selected_display_values():
    stmt = CodeGenerator._literal_operand_fixup_stmt(
        _operand('src0', 'OPR_SRC', size=16),
        'Vop3InstLiteralMachineInst',
        literal_operand_type='OPR_SIMM32',
        dynamic_true16_opsel_bit=0,
    )

    assert (
        'src0 = Operand(16, OperandType::OPR_SIMM32, '
        'static_cast<int>(reinterpret_cast<const Vop3InstLiteralMachineInst *>(inst)->simm32), '
        'static_cast<uint16_t>((reinterpret_cast<const Vop3InstLiteralMachineInst *>(inst)->simm32 '
        '>> (((amdgpu::vop3_opsel(inst_) >> 0) & 1u) * 16u)) & 0xFFFFu), true);'
    ) == stmt


def test_dynamic_true16_display_uses_each_sources_opsel_bit():
    stmt = CodeGenerator._literal_operand_fixup_stmt(
        _operand('src2', 'OPR_SRC', size=16),
        'Vop3InstLiteralMachineInst',
        literal_operand_type='OPR_SIMM32',
        dynamic_true16_opsel_bit=2,
    )

    assert '((amdgpu::vop3_opsel(inst_) >> 2) & 1u) * 16u' in stmt


def test_fieldless_sopk_simm32_can_be_initialized_from_literal_member():
    op = _operand('simm32', 'OPR_SIMM32', order=1, fieldless=True)

    stmt = CodeGenerator._literal_operand_from_expr_stmt(op, 'literal_')

    assert (
        'simm32 = Operand(32, OperandType::OPR_SIMM32, static_cast<int>(literal_));'
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


def test_scalar_fmamk_semantic_sources_use_fieldless_simm32_as_multiplier():
    operands = [
        _operand('sdst', 'OPR_SDST', is_input=False, is_output=True, order=0),
        _operand('ssrc0', 'OPR_SSRC', order=1),
        _operand('ssrc1', 'OPR_SSRC', order=2),
        _operand('simm32', 'OPR_SIMM32', order=3, fieldless=True),
    ]
    inst = Instruction('S_FMAMK_F32', 'ENC_SOP2', opcode=70, operands=operands)

    ordered = CodeGenerator._semantic_source_operands(inst, operands[1:])

    assert [op.name for op in ordered] == ['ssrc0', 'simm32', 'ssrc1']


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
        _operand('simm32', 'OPR_SIMM32', order=3, fieldless=True),
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

    assert 'std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_scalar(ssrc0))' in body
    assert (
        'std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_scalar(simm32))' in body
    )
    assert 'std::bit_cast<float>(amdgpu::RegisterAccess(wf).read_scalar(ssrc1))' in body
    assert body.index('amdgpu::RegisterAccess(wf).read_scalar(simm32)') < body.index(
        'amdgpu::RegisterAccess(wf).read_scalar(ssrc1)'
    )


def test_vector_fmaak_execute_uses_fieldless_simm32_operand():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name='rdna4',
        profile=Rdna4Profile(),
        inst_encodings=[],
        encoding_map={},
    )
    operands = [
        _operand('vdst', 'OPR_VGPR', is_input=False, is_output=True, order=0),
        _operand('src0', 'OPR_SRC', order=1),
        _operand('vsrc1', 'OPR_VGPR', order=2),
        _operand('simm32', 'OPR_SIMM32', order=3, fieldless=True),
    ]
    inst = Instruction('V_FMAAK_F32', 'ENC_VOP2', opcode=70, operands=operands)
    sem = InstructionSemantics(
        'V_FMAAK_F32',
        'vector_fmaak',
        data_type='f32',
    )

    body = codegen._gen_execute_body(inst, sem, 'ENC_VOP2')

    assert 'std::bit_cast<float>(simm32.encoding_value_)' in body
    assert 'simm32_' not in body


def test_scalar_setreg_imm_execute_reads_fieldless_simm32_source():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name='rdna4',
        profile=Rdna4Profile(),
        inst_encodings=[],
        encoding_map={},
    )
    operands = [
        _operand('simm16', 'OPR_HWREG', is_input=False, is_output=False, order=0),
        _operand('simm32', 'OPR_SIMM32', order=1, fieldless=True),
    ]
    inst = Instruction('S_SETREG_IMM32_B32', 'ENC_SOPK', opcode=20, operands=operands)
    sem = InstructionSemantics('S_SETREG_IMM32_B32', 'scalar_setreg_imm')

    body = codegen._gen_execute_body(inst, sem, 'ENC_SOPK')

    assert 'uint32_t src = amdgpu::RegisterAccess(wf).read_scalar(simm32);' in body
    assert 'uint32_t src = literal_;' not in body


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

    assert 'amdgpu::RegisterAccess(wf).read_scalar64(ssrc0)' in body
    assert 'amdgpu::RegisterAccess(wf).read_scalar64(ssrc1)' in body
    assert 'amdgpu::RegisterAccess(wf).write_scalar64(sdst, result);' in body
    assert 'amdgpu::RegisterAccess(wf).read_scalar(ssrc0)' not in body
    assert 'amdgpu::RegisterAccess(wf).read_scalar(ssrc1)' not in body


def test_scalar_addpc_generated_execute_uses_unsigned_pc_addition():
    codegen = object.__new__(CodeGenerator)
    codegen.isa_spec = SimpleNamespace(
        arch_name='cdna5',
        profile=Cdna5Profile(),
        inst_encodings=[],
        encoding_map={},
    )
    operands = [_operand('ssrc0', 'OPR_SSRC', size=64, order=0)]
    inst = Instruction('S_ADD_PC_I64', 'ENC_SOP1', opcode=75, operands=operands)
    sem = InstructionSemantics('S_ADD_PC_I64', 'scalar_addpc')

    body = codegen._gen_execute_body(inst, sem, 'ENC_SOP1')

    assert body == '  wf.pc += amdgpu::RegisterAccess(wf).read_scalar64(ssrc0);'
    assert 'int64_t' not in body


def test_literal_fma_can_share_with_matching_operand_layouts_only():
    plan = SharedInstructionPlan()
    plan.family_shared['rdna'] = {}
    plan.family_shared['rdna'][('s_fmaak_f32', 'ENC_SOP2')] = SharedInstInfo(
        mnemonic='s_fmaak_f32',
        encoding_name='ENC_SOP2',
        field_layout=(),
        semantic_class='scalar_binop',
        operation='fma',
        data_type='f32',
        isa_names=['rdna3_5', 'rdna4'],
    )

    rdna_codegen = object.__new__(CodeGenerator)
    rdna_codegen.isa_spec = SimpleNamespace(arch_name='rdna4', profile=Rdna4Profile())
    rdna_codegen.shared_plan = plan
    rdna_codegen.config = SimpleNamespace(unshared_execute_keys=frozenset())

    gfx_codegen = object.__new__(CodeGenerator)
    gfx_codegen.isa_spec = SimpleNamespace(arch_name='cdna5', profile=Cdna5Profile())
    gfx_codegen.shared_plan = plan
    gfx_codegen.config = SimpleNamespace(unshared_execute_keys=frozenset())

    assert rdna_codegen._can_share_execute('s_fmaak_f32', enc_name='ENC_SOP2')
    assert not gfx_codegen._can_share_execute('s_fmaak_f32', enc_name='ENC_SOP2')
    # Only ENC_SOP2 is shared, so querying ENC_SOP1 must NOT match
    assert not rdna_codegen._can_share_execute('s_fmaak_f32', enc_name='ENC_SOP1')


# ---------------------------------------------------------------------------
# The shared SIMD ternary table hard-codes the inline-literal operand name
# (`inst.simm32.encoding_value_`). simd_ternary_literal_operand_name() exposes
# that assumption so the generator can assert it per ISA reaching the shared
# path (see the tripwire in _generator.py). These pin the helper's contract.
# ---------------------------------------------------------------------------
def test_simd_ternary_literal_operand_name_for_inline_literal_forms():
    # Every inline-literal FMA entry reads its literal from `simm32`.
    assert simd_ternary_literal_operand_name('v_fmaak_f32_vop2') == 'simm32'
    assert simd_ternary_literal_operand_name('v_fmamk_f32_vop2') == 'simm32'
    assert simd_ternary_literal_operand_name('v_fmaak_f16_vop2') is None


def test_simd_ternary_literal_operand_name_none_for_accumulate_and_unknown():
    # Dst-accumulate forms carry no inline literal (k == "0u").
    assert simd_ternary_literal_operand_name('v_fmac_f32_vop2') is None
    assert simd_ternary_literal_operand_name('v_mac_f32_vop2') is None
    # Non-ternary / unknown templates return None.
    assert simd_ternary_literal_operand_name('v_add_f32_vop2') is None
    assert simd_ternary_literal_operand_name('not_a_template') is None


def test_simd_ternary_literal_name_covers_every_inline_literal_entry():
    # Any inline-literal entry (k_expr other than "0u") must resolve to a real
    # operand name, so the generator tripwire fires for all of them, not a
    # hand-picked subset. Guards against a new entry using an unparseable k_expr.
    for template_name, (_cpp_t, k_expr, _op) in SIMD_VOP2_TERNARY.items():
        resolved = simd_ternary_literal_operand_name(template_name)
        if k_expr == '0u':
            assert resolved is None, template_name
        else:
            assert resolved == 'simm32', (template_name, k_expr, resolved)


# ---------------------------------------------------------------------------
# The inline-literal VOP2 FMA execute bodies (vector_fmamk / vector_fmaak) read
# K from the literal source operand. If that operand is missing, codegen raises
# rather than emit an out-of-range src_ops[] index. These classes are not in
# _SEMA_CLASSES and not in the DISPATCH registry, so _gen_execute_body falls
# through to the legacy branch where the len(src_ops) < 3 guard lives.
# ---------------------------------------------------------------------------
def _fma_codegen() -> CodeGenerator:
    cg = object.__new__(CodeGenerator)
    cg.isa_spec = SimpleNamespace(
        arch_name='rdna4',
        profile=Rdna4Profile(),
        inst_encodings=[],
        encoding_map={},
    )
    return cg


def _fma_inst(*, with_simm32: bool) -> Instruction:
    ops = [
        Operand('vdst', 32, 'OPR_VGPR', False, True, False, False, 1),
        Operand('src0', 32, 'OPR_SRC', True, False, False, False, 2),
    ]
    if with_simm32:
        ops.append(
            Operand(
                'simm32', 32, 'OPR_SIMM32', True, False, False, False, 3, fieldless=True
            )
        )
    ops.append(Operand('vsrc1', 32, 'OPR_VGPR', True, False, False, False, 4))
    _uniquify_fieldless_names(ops)
    return Instruction('V_FMAMK_F32', 'ENC_VOP2', 0, ops)


@pytest.mark.parametrize('cls', ['vector_fmamk', 'vector_fmaak'])
def test_inline_literal_fma_raises_without_literal_source(cls):
    # Only two field-bearing sources (src0, vsrc1) -> src_ops has length 2, so
    # the K operand is missing and codegen must refuse rather than index
    # src_ops[2] / src_ops[1] out of / into the wrong slot.
    cg = _fma_codegen()
    sem = InstructionSemantics('V_FMAMK_F32', cls, data_type='f32')
    with pytest.raises(ValueError, match='expected fieldless simm32 operand'):
        cg._gen_execute_body(_fma_inst(with_simm32=False), sem, 'ENC_VOP2')


def test_inline_literal_fma_emits_body_with_literal_source():
    # With the fieldless simm32 present (src_ops length 3), the guard passes and
    # the body reads K from the literal operand -- proving the raise above is
    # meaningful, not vacuous.
    cg = _fma_codegen()
    sem = InstructionSemantics('V_FMAMK_F32', 'vector_fmamk', data_type='f32')
    body = cg._gen_execute_body(_fma_inst(with_simm32=True), sem, 'ENC_VOP2')
    assert 'simm32.encoding_value_' in body

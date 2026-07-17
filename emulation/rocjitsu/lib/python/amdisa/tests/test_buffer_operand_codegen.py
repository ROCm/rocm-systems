# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Codegen regressions for semantic legacy buffer operands."""

from amdisa.codegen._generator import CodeGenerator


def test_legacy_buffer_vaddr_width_follows_address_mode():
    for enc_name in ('ENC_MUBUF', 'ENC_MTBUF'):
        assert CodeGenerator._buffer_vaddr_operand_size_expr(enc_name, 'vaddr') == (
            'buffer_vaddr_bits(reinterpret_cast<const OpEncoding *>(inst))'
        )
    assert CodeGenerator._buffer_vaddr_operand_size_expr('ENC_VBUFFER', 'vaddr') == (
        'vbuffer_vaddr_bits(reinterpret_cast<const OpEncoding *>(inst))'
    )


def test_non_buffer_vaddr_keeps_xml_width():
    assert CodeGenerator._buffer_vaddr_operand_size_expr('ENC_FLAT', 'vaddr') is None
    assert CodeGenerator._buffer_vaddr_operand_size_expr('ENC_MUBUF', 'vdata') is None


def test_legacy_buffer_srsrc_is_scaled_by_four():
    for enc_name in ('ENC_MUBUF', 'ENC_MTBUF'):
        expr = CodeGenerator._operand_encoding_value_expr(
            'srsrc', enc_name=enc_name, packed_16bit=False
        )
        assert expr == '(reinterpret_cast<const OpEncoding*>(inst)->srsrc * 4)'


def test_other_srsrc_fields_are_not_scaled():
    expr = CodeGenerator._operand_encoding_value_expr(
        'srsrc', enc_name='ENC_VBUFFER', packed_16bit=False
    )
    assert expr == 'reinterpret_cast<const OpEncoding*>(inst)->srsrc'


def test_cdna_memory_acc_bit_selects_accvgpr_bank():
    for enc_name in (
        'ENC_DS',
        'ENC_MUBUF',
        'ENC_MTBUF',
        'ENC_FLAT',
        'ENC_FLAT_GLBL',
        'ENC_FLAT_SCRATCH',
        'ENC_MIMG',
    ):
        expr = CodeGenerator._operand_encoding_value_expr(
            'vdst',
            enc_name=enc_name,
            packed_16bit=False,
            operand_type='OPR_VGPR_OR_ACCVGPR',
            has_acc_field=True,
        )
        assert expr == (
            '(reinterpret_cast<const OpEncoding*>(inst)->vdst + '
            '(reinterpret_cast<const OpEncoding*>(inst)->acc ? '
            'OpSelVgprOrAccvgpr::OPR_VGPR_OR_ACCVGPR_ACC_MIN : 0))'
        )


def test_non_accvgpr_operand_ignores_acc_bit():
    expr = CodeGenerator._operand_encoding_value_expr(
        'addr',
        enc_name='ENC_DS',
        packed_16bit=False,
        operand_type='OPR_VGPR',
        has_acc_field=True,
    )
    assert expr == 'reinterpret_cast<const OpEncoding*>(inst)->addr'


def test_non_memory_acc_field_does_not_select_accvgpr_bank():
    expr = CodeGenerator._operand_encoding_value_expr(
        'vdst',
        enc_name='ENC_VOP3P',
        packed_16bit=False,
        operand_type='OPR_VGPR_OR_ACCVGPR',
        has_acc_field=True,
    )
    assert expr == 'reinterpret_cast<const OpEncoding*>(inst)->vdst'


def test_mfma_acc_cd_selects_accvgpr_c_and_d_banks():
    vdst_expr = CodeGenerator._operand_encoding_value_expr(
        'vdst',
        enc_name='ENC_VOP3P',
        packed_16bit=False,
        operand_type='OPR_VGPR_OR_ACCVGPR',
        has_acc_cd_field=True,
    )
    assert vdst_expr == (
        '(reinterpret_cast<const OpEncoding*>(inst)->vdst + '
        '(reinterpret_cast<const OpEncoding*>(inst)->acc_cd ? '
        'OpSelVgprOrAccvgpr::OPR_VGPR_OR_ACCVGPR_ACC_MIN : 0))'
    )

    src2_expr = CodeGenerator._operand_encoding_value_expr(
        'src2',
        enc_name='ENC_VOP3P_MFMA',
        packed_16bit=False,
        operand_type='OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST',
        has_acc_cd_field=True,
    )
    assert src2_expr == (
        '(reinterpret_cast<const OpEncoding*>(inst)->src2 + '
        '(reinterpret_cast<const OpEncoding*>(inst)->acc_cd ? '
        '(OpSelSrcVgprOrAccvgprOrConst::'
        'OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST_ACC_MIN - '
        'OpSelSrcVgprOrAccvgprOrConst::'
        'OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST_VGPR_MIN) : 0))'
    )


def test_mfma_acc_bits_select_accvgpr_multiplicand_banks():
    for operand_name, acc_mask in (('src0', '0x1u'), ('src1', '0x2u')):
        expr = CodeGenerator._operand_encoding_value_expr(
            operand_name,
            enc_name='ENC_VOP3P',
            packed_16bit=False,
            operand_type='OPR_SRC_VGPR_OR_ACCVGPR',
            has_acc_field=True,
            has_acc_cd_field=True,
        )
        assert expr == (
            f'(reinterpret_cast<const OpEncoding*>(inst)->{operand_name} + '
            f'((reinterpret_cast<const OpEncoding*>(inst)->acc & {acc_mask}) ? '
            '(OpSelSrcVgprOrAccvgpr::OPR_SRC_VGPR_OR_ACCVGPR_ACC_MIN - '
            'OpSelSrcVgprOrAccvgpr::OPR_SRC_VGPR_OR_ACCVGPR_VGPR_MIN) : 0))'
        )

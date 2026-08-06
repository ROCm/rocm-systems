# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Codegen regressions for semantic register operands."""

from amdisa.codegen._generator import CodeGenerator, _OperandCtx


def test_legacy_buffer_vaddr_width_follows_address_mode():
    for enc_name in ('ENC_MUBUF', 'ENC_MTBUF'):
        assert CodeGenerator._buffer_vaddr_operand_size_expr(enc_name, 'vaddr') == (
            'buffer_vaddr_bits(reinterpret_cast<const OpEncoding *>(inst))'
        )
    assert CodeGenerator._buffer_vaddr_operand_size_expr('ENC_VBUFFER', 'vaddr') == (
        'vbuffer_vaddr_bits(reinterpret_cast<const OpEncoding *>(inst))'
    )


def test_gfx12_flat_vaddr_width_follows_saddr_mode():
    for enc_name in ('ENC_VFLAT', 'ENC_VGLOBAL'):
        assert CodeGenerator._vflat_vaddr_operand_size_expr(enc_name, 'vaddr') == (
            'vflat_vaddr_bits(reinterpret_cast<const OpEncoding *>(inst))'
        )

    assert CodeGenerator._vflat_vaddr_operand_size_expr('ENC_VSCRATCH', 'vaddr') is None
    assert CodeGenerator._vflat_vaddr_operand_size_expr('ENC_VFLAT', 'vdst') is None
    assert 'inst->saddr == OPR_SREG_NULL' in CodeGenerator._emit_vflat_helpers()


def test_buffer_vaddr_helper_maps_address_modes_to_zero_one_or_two_vgprs():
    helper = CodeGenerator._emit_buffer_vaddr_helpers(
        'buffer_vaddr_bits', 'BufferMachineInst', templated=True
    )
    assert helper == '''\
namespace {
template <typename BufferMachineInst>
uint32_t buffer_vaddr_bits(const BufferMachineInst *inst) {
  if (inst->idxen && inst->offen)
    return 64;
  if (inst->idxen || inst->offen)
    return 32;
  return 0;
}
} // namespace'''


def test_non_buffer_vaddr_keeps_xml_width():
    assert CodeGenerator._buffer_vaddr_operand_size_expr('ENC_FLAT', 'vaddr') is None
    assert CodeGenerator._buffer_vaddr_operand_size_expr('ENC_MUBUF', 'vdata') is None


def test_legacy_buffer_srsrc_is_scaled_by_four():
    for enc_name in ('ENC_MUBUF', 'ENC_MTBUF'):
        expr = CodeGenerator._operand_encoding_value_expr(
            _OperandCtx('srsrc', enc_name, packed_16bit=False)
        )
        assert expr == '(reinterpret_cast<const OpEncoding*>(inst)->srsrc * 4)'


def test_other_srsrc_fields_are_not_scaled():
    expr = CodeGenerator._operand_encoding_value_expr(
        _OperandCtx('srsrc', 'ENC_VBUFFER', packed_16bit=False)
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
            _OperandCtx(
                'vdst',
                enc_name,
                packed_16bit=False,
                operand_type='OPR_VGPR_OR_ACCVGPR',
                has_acc_field=True,
            )
        )
        assert expr == (
            '(reinterpret_cast<const OpEncoding*>(inst)->vdst + '
            '(reinterpret_cast<const OpEncoding*>(inst)->acc ? '
            'OpSelVgprOrAccvgpr::OPR_VGPR_OR_ACCVGPR_ACC_MIN : 0))'
        )


def test_non_accvgpr_operand_ignores_acc_bit():
    expr = CodeGenerator._operand_encoding_value_expr(
        _OperandCtx(
            'addr',
            'ENC_DS',
            packed_16bit=False,
            operand_type='OPR_VGPR',
            has_acc_field=True,
        )
    )
    assert expr == 'reinterpret_cast<const OpEncoding*>(inst)->addr'


def test_non_memory_acc_field_does_not_select_accvgpr_bank():
    expr = CodeGenerator._operand_encoding_value_expr(
        _OperandCtx(
            'vdst',
            'ENC_VOP3P',
            packed_16bit=False,
            operand_type='OPR_VGPR_OR_ACCVGPR',
            has_acc_field=True,
        )
    )
    assert expr == 'reinterpret_cast<const OpEncoding*>(inst)->vdst'


def test_mfma_acc_cd_selects_accvgpr_c_and_d_banks():
    vdst_expr = CodeGenerator._operand_encoding_value_expr(
        _OperandCtx(
            'vdst',
            'ENC_VOP3P',
            packed_16bit=False,
            operand_type='OPR_VGPR_OR_ACCVGPR',
            has_acc_cd_field=True,
        )
    )
    assert vdst_expr == (
        '(reinterpret_cast<const OpEncoding*>(inst)->vdst + '
        '(reinterpret_cast<const OpEncoding*>(inst)->acc_cd ? '
        'OpSelVgprOrAccvgpr::OPR_VGPR_OR_ACCVGPR_ACC_MIN : 0))'
    )

    src2_expr = CodeGenerator._operand_encoding_value_expr(
        _OperandCtx(
            'src2',
            'ENC_VOP3P_MFMA',
            packed_16bit=False,
            operand_type='OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST',
            has_acc_cd_field=True,
        )
    )
    assert src2_expr == (
        'mfma_src2_encoding(reinterpret_cast<const OpEncoding*>(inst)->src2, '
        'reinterpret_cast<const OpEncoding*>(inst)->acc_cd)'
    )


def test_mfma_acc_bits_select_accvgpr_multiplicand_banks():
    for operand_name, acc_mask in (('src0', '0x1u'), ('src1', '0x2u')):
        expr = CodeGenerator._operand_encoding_value_expr(
            _OperandCtx(
                operand_name,
                'ENC_VOP3P',
                packed_16bit=False,
                operand_type='OPR_SRC_VGPR_OR_ACCVGPR',
                has_acc_field=True,
                has_acc_cd_field=True,
            )
        )
        assert expr == (
            f'(reinterpret_cast<const OpEncoding*>(inst)->{operand_name} + '
            f'((reinterpret_cast<const OpEncoding*>(inst)->acc & {acc_mask}) ? '
            '(OpSelSrcVgprOrAccvgpr::OPR_SRC_VGPR_OR_ACCVGPR_ACC_MIN - '
            'OpSelSrcVgprOrAccvgpr::OPR_SRC_VGPR_OR_ACCVGPR_VGPR_MIN) : 0))'
        )


def test_mfma_acc_without_acc_cd_does_not_fold_multiplicand_banks():
    for operand_name in ('src0', 'src1'):
        expr = CodeGenerator._operand_encoding_value_expr(
            _OperandCtx(
                operand_name,
                'ENC_VOP3P',
                packed_16bit=False,
                operand_type='OPR_SRC_VGPR_OR_ACCVGPR',
                has_acc_field=True,
                has_acc_cd_field=False,
            )
        )
        assert expr == f'reinterpret_cast<const OpEncoding*>(inst)->{operand_name}'

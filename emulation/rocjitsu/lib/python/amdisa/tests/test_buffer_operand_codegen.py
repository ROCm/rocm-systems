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

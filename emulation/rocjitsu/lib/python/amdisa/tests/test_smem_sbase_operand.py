# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Codegen regression test for the SMEM SBASE ×2 operand scale.

SMEM encodes SBASE in units of 2 SGPRs (raw N names s[2N:2N+1]). The operand
model must scale by 2 so its register-ref matches execution / hardware. This
guards ``_operand_encoding_value_expr`` against silently dropping the scale.
"""

from amdisa.codegen._generator import CodeGenerator


def test_smem_sbase_is_scaled_by_two():
    expr = CodeGenerator._operand_encoding_value_expr(
        'sbase', is_smem=True, packed_16bit=False
    )
    assert expr == '(reinterpret_cast<const OpEncoding*>(inst)->sbase * 2)'


def test_non_smem_sbase_is_not_scaled():
    # The scale is gated on SMEM; an 'sbase'-named field elsewhere is untouched.
    expr = CodeGenerator._operand_encoding_value_expr(
        'sbase', is_smem=False, packed_16bit=False
    )
    assert '* 2' not in expr
    assert expr == 'reinterpret_cast<const OpEncoding*>(inst)->sbase'


def test_smem_non_sbase_operand_is_not_scaled():
    # Other SMEM operands (e.g. sdata) keep the raw field.
    expr = CodeGenerator._operand_encoding_value_expr(
        'sdata', is_smem=True, packed_16bit=False
    )
    assert '* 2' not in expr


def test_packed_16bit_wrap_composes_with_scale():
    # The packed-16bit cast still wraps the (already scaled) value; ordering
    # must match the generator (scale first, then cast).
    expr = CodeGenerator._operand_encoding_value_expr(
        'sbase', is_smem=True, packed_16bit=True
    )
    assert expr == (
        'static_cast<unsigned short>'
        '((reinterpret_cast<const OpEncoding*>(inst)->sbase * 2))'
    )

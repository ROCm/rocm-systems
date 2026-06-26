# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the ``Instruction.implicit_operands`` data model.

The MR ISA XML declares some operands implicit: they have no ``<FieldName>`` and
represent side-effect state (EXEC, SCC, VCC, M0, PC, memory). The parser adds
them to ``Instruction.implicit_operands`` while explicit field-bearing operands
go to ``Instruction.operands``.
"""

from __future__ import annotations

from amdisa.gpuisa import Instruction, Operand


def _operand(name: str, operand_type: str, *, implicit: bool) -> Operand:
    # Operand(name, size, operand_type, is_input, is_output, is_implicit,
    #         is_bin_ucode_required, order)
    return Operand(name, 32, operand_type, False, True, implicit, implicit, 0)


def test_stores_explicit_and_implicit_operands_independently():
    # The MR ISA gives s_getpc an explicit destination (sdst) and an implicit
    # PC source; the two land on separate lists and do not bleed into each other.
    sdst = _operand('sdst', 'OPR_SDST', implicit=False)
    pc = _operand('', 'OPR_PC', implicit=True)
    inst = Instruction('S_GETPC_B64', 'ENC_SOP1', 0, [sdst], implicit_operands=[pc])

    assert inst.operands == [sdst]
    assert inst.implicit_operands == [pc]


def test_has_implicit_operand_matches_by_type():
    # The generic accessor matches on operand type and ignores the explicit
    # operand list. Used by codegen as inst.has_implicit_operand('OPR_PC').
    inst = Instruction(
        'S_AND_SAVEEXEC_B32',
        'ENC_SOP1',
        0,
        [_operand('sdst', 'OPR_SDST', implicit=False)],
        implicit_operands=[
            _operand('', 'OPR_SDST_EXEC', implicit=True),
            _operand('', 'OPR_SSRC_SPECIAL_SCC', implicit=True),
        ],
    )

    assert inst.has_implicit_operand('OPR_SDST_EXEC')
    assert inst.has_implicit_operand('OPR_SSRC_SPECIAL_SCC')
    assert not inst.has_implicit_operand('OPR_PC')
    # An explicit operand's type does not count as an implicit operand.
    assert not inst.has_implicit_operand('OPR_SDST')


def test_has_implicit_operand_false_when_none():
    assert not Instruction('S_NOP', 'ENC_SOPP', 0, []).has_implicit_operand('OPR_PC')


def test_implicit_operands_default_to_empty_list():
    inst = Instruction('S_MOV_B32', 'ENC_SOP1', 0, [])
    assert inst.implicit_operands == []


def test_implicit_operands_default_is_not_shared_between_instances():
    # Guards against the classic mutable-default-argument bug: each instruction
    # must get its own list.
    a = Instruction('A', 'ENC_SOP1', 0, [])
    b = Instruction('B', 'ENC_SOP1', 0, [])
    a.implicit_operands.append(_operand('', 'OPR_PC', implicit=True))
    assert b.implicit_operands == []


def test_implicit_operand_preserves_type_and_direction():
    # The list keeps the operand's type and is_input/is_output direction, so a
    # future read-vs-write analysis (e.g. reads PC vs writes PC) can use them.
    pc_read = Operand('', 64, 'OPR_PC', True, False, True, True, 1)
    pc_write = Operand('', 64, 'OPR_PC', False, True, True, True, 1)

    reader = Instruction('S_GETPC_B64', 'ENC_SOP1', 0, [], implicit_operands=[pc_read])
    writer = Instruction('S_RFE_B64', 'ENC_SOP1', 0, [], implicit_operands=[pc_write])

    assert (
        reader.implicit_operands[0].is_input
        and not reader.implicit_operands[0].is_output
    )
    assert (
        writer.implicit_operands[0].is_output
        and not writer.implicit_operands[0].is_input
    )

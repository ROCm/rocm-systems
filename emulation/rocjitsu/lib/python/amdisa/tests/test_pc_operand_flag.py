# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the PC_OPERAND flag.

The code generator emits ``flags_ |= PC_OPERAND;`` into an instruction's
constructor when the instruction has an implicit ``OPR_PC`` operand.
The DBI anchor validator currently uses that flag to reject non-relocatable
trampoline anchors.

These tests:

* exercise the production predicate directly on hand-built ``Instruction`` /
  ``Operand`` fixtures
* read the committed generated ``.cpp`` to confirm the real parser -> codegen
  pipeline actually emits the flag for the MR ISA's PC families
* check instructions that do not have the ``OPR_PC`` operand
"""

from __future__ import annotations

import pathlib

from amdisa.gpuisa import Instruction, Operand

_GEN_ROOT = (
    pathlib.Path(__file__).resolve().parents[4]
    / 'lib'
    / 'rocjitsu'
    / 'src'
    / 'rocjitsu'
    / 'isa'
    / 'arch'
    / 'amdgpu'
)


def _implicit(
    operand_type: str, *, reads: bool, writes: bool, size: int = 64
) -> Operand:
    """A field-less implicit operand, as the parser produces for OPR_* operands."""
    return Operand('', size, operand_type, reads, writes, True, True, 1)


# ---------------------------------------------------------------------------
# Predicate: Instruction.has_implicit_operand('OPR_PC')
# ---------------------------------------------------------------------------


def test_pc_read_operand_is_detected():
    # MR ISA: s_getpc_b64 has an implicit OPR_PC operand with Input=true.
    getpc = Instruction(
        'S_GETPC_B64',
        'ENC_SOP1',
        0,
        [],
        implicit_operands=[_implicit('OPR_PC', reads=True, writes=False)],
    )
    assert getpc.has_implicit_operand('OPR_PC')


def test_pc_write_operand_is_detected():
    # MR ISA: s_rfe_b64 has an implicit OPR_PC operand with Output=true. It is
    # classified true_nop and carries no control-flow flag, so this operand is
    # the only structural signal that it touches PC.
    rfe = Instruction(
        'S_RFE_B64',
        'ENC_SOP1',
        0,
        [],
        implicit_operands=[_implicit('OPR_PC', reads=False, writes=True)],
    )
    assert rfe.has_implicit_operand('OPR_PC')


def test_non_pc_implicit_operand_is_not_detected():
    # MR ISA: s_and_b32 has an implicit SCC operand (OPR_SSRC_SPECIAL_SCC) and no
    # OPR_PC. The predicate must discriminate by operand type, not merely flag
    # any instruction that has implicit operands.
    s_and = Instruction(
        'S_AND_B32',
        'ENC_SOP2',
        0,
        [],
        implicit_operands=[
            _implicit('OPR_SSRC_SPECIAL_SCC', reads=False, writes=True, size=1)
        ],
    )
    assert s_and.implicit_operands  # it does carry an implicit operand ...
    assert not s_and.has_implicit_operand('OPR_PC')  # ... just not a PC one


def test_instruction_without_pc_operand_is_not_detected():
    # MR ISA: s_branch carries a label operand (not OPR_PC); s_mov_b32 has no
    # implicit operands at all. Neither reads nor writes PC.
    branch = Instruction('S_BRANCH', 'ENC_SOPP', 0, [])
    mov = Instruction('S_MOV_B32', 'ENC_SOP1', 0, [])
    assert not branch.has_implicit_operand('OPR_PC')
    assert not mov.has_implicit_operand('OPR_PC')


# ---------------------------------------------------------------------------
# Generated-output: the real pipeline emits the flag for the MR ISA families
# ---------------------------------------------------------------------------


def _ctor_body(src: str, cls: str) -> str:
    """Slice out the constructor body for class `cls` (ctor up to its execute_impl)."""
    start = src.index(f'{cls}::{cls}(')
    end = src.index(f'void {cls}::execute_impl', start)
    return src[start:end]


def test_generated_cdna_constructors_emit_pc_operand_flag():
    src = (_GEN_ROOT / 'cdna2' / 'sop1.cpp').read_text()

    assert 'flags_ |= PC_OPERAND;' in _ctor_body(src, 'SGetpcB64Sop1')
    assert 'flags_ |= PC_OPERAND;' in _ctor_body(src, 'SRfeB64Sop1')
    # Ordinary scalar move must NOT be flagged.
    assert 'flags_ |= PC_OPERAND;' not in _ctor_body(src, 'SMovB32Sop1')


def test_generated_gfx1250_renamed_constructors_emit_pc_operand_flag():
    # gfx1250 renames the family (s_getpc_b64 -> s_get_pc_i64) but keeps the
    # OPR_PC operand, so the flag must still be emitted
    src = (_GEN_ROOT / 'gfx1250' / 'sop1.cpp').read_text()

    assert 'flags_ |= PC_OPERAND;' in _ctor_body(src, 'SGetPcI64Sop1')
    assert 'flags_ |= PC_OPERAND;' in _ctor_body(src, 'SRfeI64Sop1')
    assert 'flags_ |= PC_OPERAND;' not in _ctor_body(src, 'SMovB32Sop1')

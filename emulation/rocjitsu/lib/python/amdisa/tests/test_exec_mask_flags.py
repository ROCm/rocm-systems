# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for EXEC-mask instruction flag emission."""

import pytest

from amdisa.codegen._generator import (
    _exec_mask_flag_stmts,
    _result_combinator_flag_stmts,
)
from amdisa.semantics import InstructionSemantics

# Flag names emitted for EXEC tracking.
ALL = {'IGNORES_EXEC', 'WRITES_EXEC'}


def _flags(sem):
    """Run the generator helper and return the set of emitted flag names."""
    return {s[len('flags_ |= ') : -1] for s in _exec_mask_flag_stmts(sem)}


# (InstructionSemantics, expected-present flags, id). Anything in ALL but not in
# the expected set must be ABSENT — see the test below.
_CASES = [
    # Scalar EXEC writers: write EXEC, but are not themselves EXEC-masked.
    pytest.param(
        InstructionSemantics(
            'S_AND_SAVEEXEC_B64', 'scalar_saveexec', operation='and', data_type='b64'
        ),
        {'WRITES_EXEC'},
        id='scalar_saveexec',
    ),
    pytest.param(
        InstructionSemantics(
            'S_WREXEC_B64', 'scalar_wrexec', operation='and', data_type='b64'
        ),
        {'WRITES_EXEC'},
        id='scalar_wrexec',
    ),
    # Vector ALU: no EXEC interaction flags.
    pytest.param(
        InstructionSemantics(
            'V_ADD_F32', 'vector_binop', operation='add', data_type='f32'
        ),
        set(),
        id='vector_binop',
    ),
    pytest.param(
        InstructionSemantics('V_MOV_B32', 'vector_mov', data_type='b32'),
        set(),
        id='vector_mov',
    ),
    pytest.param(
        InstructionSemantics('V_CNDMASK_B32', 'vector_cndmask', data_type='b32'),
        set(),
        id='vector_cndmask',
    ),
    # Vector compare-and-set-exec: writes EXEC.
    pytest.param(
        InstructionSemantics(
            'V_CMPX_LT_F32', 'vector_cmpx', operation='lt', data_type='f32'
        ),
        {'WRITES_EXEC'},
        id='vector_cmpx',
    ),
    # Branches: ignore EXEC, never EXEC-masked.
    pytest.param(
        InstructionSemantics('S_BRANCH', 'branch'),
        {'IGNORES_EXEC'},
        id='branch',
    ),
    pytest.param(
        InstructionSemantics('S_CBRANCH_SCC1', 'cbranch', branch_condition='scc1'),
        {'IGNORES_EXEC'},
        id='cbranch',
    ),
    # Plain scalar op: no EXEC interaction at all.
    pytest.param(
        InstructionSemantics('S_MOV_B32', 'scalar_mov', data_type='b32'),
        set(),
        id='scalar_mov',
    ),
]


class TestExecMaskFlagStmts:
    def test_none_semantics_yields_no_flags(self):
        assert _exec_mask_flag_stmts(None) == []

    def test_unsupported_class_yields_no_flags(self):
        # A semantic class with no registered deriver -> derive_sema_block
        # returns None -> conservatively no flags.
        sem = InstructionSemantics('NOT_A_REAL_INST', 'no_such_class')
        assert _exec_mask_flag_stmts(sem) == []

    @pytest.mark.parametrize('sem, expected', _CASES)
    def test_flags_per_instruction_kind(self, sem, expected):
        flags = _flags(sem)
        assert expected <= flags, f'{sem.name}: missing {expected - flags}'
        unexpected = (ALL - expected) & flags
        assert not unexpected, f'{sem.name}: unexpected {unexpected}'

    def test_flag_statement_format(self):
        sem = InstructionSemantics(
            'S_OR_SAVEEXEC_B64', 'scalar_saveexec', operation='or', data_type='b64'
        )
        stmts = _exec_mask_flag_stmts(sem)
        assert stmts
        for s in stmts:
            assert s.startswith('flags_ |= ')
            assert s.endswith(';')


class TestResultCombinatorFlagStmts:
    """RESULT_COPY / RESULT_OR drive EXEC-state all-ones reasoning."""

    def _combinator(self, sem):
        return {s[len('flags_ |= ') : -1] for s in _result_combinator_flag_stmts(sem)}

    def test_scalar_mov_is_copy(self):
        sem = InstructionSemantics('S_MOV_B64', 'scalar_mov', data_type='b64')
        assert self._combinator(sem) == {'RESULT_COPY'}

    def test_binop_or_is_or(self):
        sem = InstructionSemantics(
            'S_OR_B64', 'scalar_binop', operation='or', data_type='b64'
        )
        assert self._combinator(sem) == {'RESULT_OR'}

    def test_saveexec_or_is_or(self):
        sem = InstructionSemantics(
            'S_OR_SAVEEXEC_B64', 'scalar_saveexec', operation='or', data_type='b64'
        )
        assert self._combinator(sem) == {'RESULT_OR'}

    def test_saveexec_and_is_other(self):
        # exec = exec & src -> not provably all-ones; no combinator flag.
        sem = InstructionSemantics(
            'S_AND_SAVEEXEC_B64', 'scalar_saveexec', operation='and', data_type='b64'
        )
        assert _result_combinator_flag_stmts(sem) == []

    def test_binop_and_is_other(self):
        sem = InstructionSemantics(
            'S_AND_B64', 'scalar_binop', operation='and', data_type='b64'
        )
        assert _result_combinator_flag_stmts(sem) == []

    def test_vector_mov_is_not_copy(self):
        # RESULT_COPY is scalar-only; a vector mov must not emit it.
        sem = InstructionSemantics('V_MOV_B32', 'vector_mov', data_type='b32')
        assert _result_combinator_flag_stmts(sem) == []

    def test_vector_binop_or_is_not_or(self):
        # RESULT_OR is scalar-only; a vector bitwise-or must not emit it.
        sem = InstructionSemantics(
            'V_OR_B32', 'vector_binop', operation='or', data_type='b32'
        )
        assert _result_combinator_flag_stmts(sem) == []

    def test_vector_cndmask_is_not_copy(self):
        # A non-scalar select is not a plain copy.
        sem = InstructionSemantics('V_CNDMASK_B32', 'vector_cndmask', data_type='b32')
        assert _result_combinator_flag_stmts(sem) == []

    def test_none_is_empty(self):
        assert _result_combinator_flag_stmts(None) == []

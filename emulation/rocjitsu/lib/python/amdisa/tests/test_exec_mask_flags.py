# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for EXEC-mask instruction flag emission in the C++ generator.

``_exec_mask_flag_stmts`` turns an instruction's derived semantic properties
into ``flags_ |= ...;`` constructor statements (EXEC_MASKED / IGNORES_EXEC /
WRITES_EXEC / READS_EXEC). Generic liveness/dataflow analyses consume these
flags, so the mapping from instruction semantics to flags must stay stable
across the range of instruction kinds below.
"""

import pytest

from amdisa.codegen._generator import _exec_mask_flag_stmts
from amdisa.semantics import InstructionSemantics

# Flag names emitted for EXEC tracking.
ALL = {'EXEC_MASKED', 'IGNORES_EXEC', 'WRITES_EXEC', 'READS_EXEC'}


def _flags(sem):
    """Run the generator helper and return the set of emitted flag names."""
    return {s[len('flags_ |= ') : -1] for s in _exec_mask_flag_stmts(sem)}


# (id, InstructionSemantics, expected-present flags). Anything in ALL but not in
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
    # Vector ALU: EXEC-masked (per-lane), no EXEC read/write.
    pytest.param(
        InstructionSemantics(
            'V_ADD_F32', 'vector_binop', operation='add', data_type='f32'
        ),
        {'EXEC_MASKED'},
        id='vector_binop',
    ),
    pytest.param(
        InstructionSemantics('V_MOV_B32', 'vector_mov', data_type='b32'),
        {'EXEC_MASKED'},
        id='vector_mov',
    ),
    pytest.param(
        InstructionSemantics('V_CNDMASK_B32', 'vector_cndmask', data_type='b32'),
        {'EXEC_MASKED'},
        id='vector_cndmask',
    ),
    # Vector compare-and-set-exec: masked AND reads+writes EXEC.
    pytest.param(
        InstructionSemantics(
            'V_CMPX_LT_F32', 'vector_cmpx', operation='lt', data_type='f32'
        ),
        {'EXEC_MASKED', 'WRITES_EXEC', 'READS_EXEC'},
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

    def test_branch_is_never_exec_masked(self):
        # IGNORES_EXEC and EXEC_MASKED are mutually exclusive by construction.
        for sem, _ in [(c.values[0], c.values[1]) for c in _CASES]:
            flags = _flags(sem)
            assert not ('IGNORES_EXEC' in flags and 'EXEC_MASKED' in flags)

    def test_flag_statement_format(self):
        sem = InstructionSemantics(
            'S_OR_SAVEEXEC_B64', 'scalar_saveexec', operation='or', data_type='b64'
        )
        stmts = _exec_mask_flag_stmts(sem)
        assert stmts
        for s in stmts:
            assert s.startswith('flags_ |= ')
            assert s.endswith(';')

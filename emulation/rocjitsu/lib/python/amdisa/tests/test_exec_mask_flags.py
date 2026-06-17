# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for EXEC-mask instruction flag emission in the C++ generator.

``_exec_mask_flag_stmts`` turns an instruction's derived semantic properties
into ``flags_ |= ...;`` constructor statements (EXEC_MASKED / IGNORES_EXEC /
WRITES_EXEC / READS_EXEC). Generic liveness/dataflow analyses consume these
flags, so the mapping from semantics to flags must stay stable.
"""

from amdisa.codegen._generator import _exec_mask_flag_stmts
from amdisa.semantics import InstructionSemantics


def _flags(stmts):
    """Extract the bare flag names from `flags_ |= NAME;` statements."""
    return {s[len('flags_ |= ') : -1] for s in stmts}


class TestExecMaskFlagStmts:
    def test_none_semantics_yields_no_flags(self):
        assert _exec_mask_flag_stmts(None) == []

    def test_unsupported_class_yields_no_flags(self):
        # A semantic class with no registered deriver -> derive_sema_block
        # returns None -> conservatively no flags.
        sem = InstructionSemantics('NOT_A_REAL_INST', 'no_such_class')
        assert _exec_mask_flag_stmts(sem) == []

    def test_saveexec_writes_exec_but_is_not_exec_masked(self):
        # s_and_saveexec_b64 is a SCALAR instruction that writes EXEC. It must
        # be tagged WRITES_EXEC and must NOT be tagged EXEC_MASKED (it is not a
        # per-lane vector op whose inactive lanes are preserved).
        sem = InstructionSemantics(
            'S_AND_SAVEEXEC_B64',
            'scalar_saveexec',
            operation='and',
            data_type='b64',
        )
        flags = _flags(_exec_mask_flag_stmts(sem))
        assert 'WRITES_EXEC' in flags
        assert 'EXEC_MASKED' not in flags

    def test_flag_statement_format(self):
        sem = InstructionSemantics(
            'S_OR_SAVEEXEC_B64',
            'scalar_saveexec',
            operation='or',
            data_type='b64',
        )
        stmts = _exec_mask_flag_stmts(sem)
        # Every emitted statement is a well-formed C++ flag-OR statement.
        assert stmts
        for s in stmts:
            assert s.startswith('flags_ |= ')
            assert s.endswith(';')

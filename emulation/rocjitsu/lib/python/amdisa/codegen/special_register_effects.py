# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fail-closed coverage for EXEC/VCC writes, independent of GPU target names.

Coverage means the generated destinations plus positive implicit-write flags
cover every possible write. It does not imply a write is unconditional or that
an old value is dead. Runtime consumers must still resolve actual selectors.
"""

from amdisa.sema_ast import SemaNodeKind
from amdisa.sema_derive import derive_sema_block

# These execution families write only their explicit destinations, SCC, and
# the mask results identified below. Address helpers change no mask state.
# Exclude indirect register indexing, HWREG/system operations, image/matrix
# helpers and unimplemented semantic stubs until their effects are audited.
_COVERED_CLASSES = frozenset(
    {
        'scalar_mov',
        'scalar_cmov',
        'scalar_unary',
        'scalar_binop',
        'scalar_bfe',
        'scalar_cmp',
        'scalar_cmpk',
        'scalar_bitcmp',
        'scalar_saveexec',
        'scalar_wrexec',
        'scalar_movk',
        'scalar_cmovk',
        'scalar_addk',
        'scalar_mulk',
        'scalar_cselect',
        'vector_mov',
        'vector_unary',
        'vector_binop',
        'vector_ternary',
        'vector_cmp',
        'vector_cmp_class',
        'vector_cmpx',
        'vector_cmpx_class',
        'vector_add_co',
        'vector_cndmask',
        'vector_readfirstlane',
        'vector_readlane',
        'vector_writelane',
        'smem_load',
        'smem_store',
        'flat_load',
        'flat_store',
        'buffer_load',
        'buffer_store',
        'ds_read',
        'ds_write',
        'ds_read2',
        'ds_write2',
        'waitcnt',
        'wait_counter',
        'branch',
        'cbranch',
    }
)

# Helpers with audited EXEC/VCC effects. An unfamiliar helper must not inherit
# coverage merely because it occurs in a previously covered semantic family.
_MASK_PRESERVING_CALLS = frozenset(
    {
        'addc',
        'subb',
        'mulhi',
        'mul_hi',
        'std::min',
        'std::max',
        'util::arithmetic_shr',
        'ashr_i64',
        'brev',
        'brev64',
        'bcnt1',
        'bcnt164',
        'bcnt0',
        'bcnt064',
        'ctz',
        'ctz64',
        'clz',
        'clz64',
        'cls',
        'cls64',
        'ff0',
        'ff064',
        'ff1',
        'ff164',
        'flbit',
        'flbit64',
        'wqm',
        'wqm64',
        'quadmask',
        'quadmask64',
        'v_readfirstlane',
        'v_readlane',
        'v_writelane',
        'CalcScalarGlobalAddr',
        'CalcScalarBufferAddr',
        'CalcFlatAddr',
        'CalcGlobalAddr',
        'CalcBufferAddr',
        'CalcDsAddr',
        'waitcnt',
        'branch',
        'eval_scc0',
        'eval_scc1',
        'eval_vccz',
        'eval_vccnz',
        'eval_execz',
        'eval_execnz',
    }
)

# The semantic classifier uses nop/true_nop for unsupported instructions too
# (including RFE). Empty ASTs are never evidence, except these audited forms.
_AUDITED_EMPTY_FORMS = frozenset(
    {'S_NOP', 'V_NOP', 'S_WAIT_ALU', 'S_WAITCNT_DEPCTR', 'S_DELAY_ALU'}
)


def special_register_write_flags(inst, sem, profile) -> list[str]:
    """Emit coverage only for an audited family and complete operand schema."""
    if sem is None:
        return []
    cls = sem.semantic_class
    flags = ['EXEC_WRITE_EFFECTS_KNOWN', 'VCC_WRITE_EFFECTS_KNOWN']
    if cls in ('nop', 'true_nop'):
        return (
            [f'flags_ |= {flag};' for flag in flags]
            if sem.name in _AUDITED_EMPTY_FORMS
            else []
        )
    if cls in ('scalar_setpc', 'scalar_getpc', 'wait_idle'):
        # These emitters do not use SemaAST. SETPC reads its sole source and
        # changes PC; GETPC writes its decoded destination; WAIT_IDLE only waits.
        if cls == 'scalar_setpc':
            if not any(op.is_input and not op.fieldless for op in inst.operands):
                return []
            flags.append('ABSOLUTE_INDIRECT_BRANCH')
        elif cls == 'scalar_getpc' and not any(op.is_output for op in inst.operands):
            return []
    else:
        if cls not in _COVERED_CLASSES:
            return []
        block = derive_sema_block(sem)
        if block is None or block.is_empty:
            return []
        nodes = tuple(block.body.walk())
        if any(
            node.kind == SemaNodeKind.CALL
            and node.call_name not in _MASK_PRESERVING_CALLS
            for node in nodes
        ):
            return []
        # Track direct context writes as well as decoded destinations. Do not
        # rely on a missing positive flag from a different metadata derivation.
        for node in nodes:
            if (
                node.kind
                not in (
                    SemaNodeKind.ASSIGN,
                    SemaNodeKind.ADD_ASSIGN,
                    SemaNodeKind.SUB_ASSIGN,
                )
                or not node.children
            ):
                continue
            lhs = node.children[0]
            while (
                lhs.kind in (SemaNodeKind.CAST, SemaNodeKind.ARRAYDEREF)
                and lhs.children
            ):
                lhs = lhs.children[0]
            if lhs.kind != SemaNodeKind.ID:
                continue
            if lhs.id_name in ('EXEC', 'EXEC_RAW'):
                flags.append('WRITES_EXEC')
            elif lhs.id_name == 'VCC':
                redirected = (
                    cls in ('vector_cmp', 'vector_cmp_class')
                    and inst.enc_name.upper() != 'ENC_VOPC'
                )
                if not redirected:
                    flags.append('WRITES_VCC')
            elif (
                lhs.id_name
                and lhs.id_name.isupper()
                and lhs.id_name not in ('SCC', 'MEM', 'LDS')
            ):
                return []
        # A semantic D[n] needs an actual output, not an input-only XML field.
        outputs = [op for op in inst.operands if op.is_output and not op.fieldless]
        for node in nodes:
            if (
                node.kind == SemaNodeKind.INSTOPERAND
                and len(node.children) == 2
                and node.children[0].id_name == 'D'
            ):
                index = int(node.children[1].lit_value)
                if index >= len(outputs):
                    return []
    # The semantic CMP AST uses VCC symbolically even when VOP3 redirects the
    # result to a decoded SGPR destination. Only VOPC has the implicit VCC result.
    if (
        cls in ('vector_cmp', 'vector_cmp_class')
        and inst.enc_name.upper() == 'ENC_VOPC'
    ):
        flags.append('WRITES_VCC')
    if cls in ('vector_cmpx', 'vector_cmpx_class'):
        flags.append('WRITES_EXEC')
        if profile.cmpx_writes_vcc:
            flags.append('WRITES_VCC')
    if cls == 'vector_add_co' and inst.enc_name.upper() == 'ENC_VOP2':
        flags.append('WRITES_VCC')
    return [f'flags_ |= {flag};' for flag in dict.fromkeys(flags)]

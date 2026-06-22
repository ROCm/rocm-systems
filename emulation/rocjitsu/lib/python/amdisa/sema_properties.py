# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Derive instruction properties from SemaAST tree walking.

Replaces hard-coded property lists with algorithmic derivation from
the semantic expression tree. Each property is detected by pattern
matching on the AST nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Flag, auto

from amdisa.sema_ast import (
    ExecModel,
    SemaBlock,
    SemaNode,
    SemaNodeKind,
)


class InstructionProperty(Flag):
    NONE = 0
    EXEC_MASKED = auto()
    IS_MATRIX = auto()
    IS_BARRIER = auto()
    USES_ACCVGPR = auto()
    IS_WAITCNT = auto()
    IGNORES_EXEC = auto()
    CROSS_LANE = auto()
    DS_PERMUTE = auto()
    WRITES_SCC = auto()
    WRITES_VCC = auto()
    WRITES_EXEC = auto()
    READS_EXEC = auto()
    READS_VCC = auto()
    IS_BRANCH = auto()
    IS_MEMORY = auto()
    IS_ENDPGM = auto()


_CROSS_LANE_CALLS: frozenset[str] = frozenset(
    {
        'v_permlane16',
        'v_permlanex16',
        'v_readfirstlane',
        'v_readlane',
        'v_writelane',
        'ds_bpermute',
        'ds_swizzle',
    }
)

_MATRIX_CALLS: frozenset[str] = frozenset(
    {
        'mfma_compute',
    }
)

_BARRIER_CALLS: frozenset[str] = frozenset(
    {
        'barrier',
        'CheckBarrierComplete',
    }
)

_WAITCNT_CALLS: frozenset[str] = frozenset(
    {
        'waitcnt',
    }
)

_ADDR_CALC_CALLS: frozenset[str] = frozenset(
    {
        'CalcBufferAddr',
        'CalcFlatAddr',
        'CalcGlobalAddr',
        'CalcDsAddr',
        'CalcScalarGlobalAddr',
        'CalcScalarBufferAddr',
        'CalcScratchAddr',
    }
)


def derive_properties(block: SemaBlock) -> InstructionProperty:
    """Derive instruction properties by walking the SemaAST.

    Args:
        block: The instruction's SemaBlock (pre- or post-enrichment).

    Returns:
        Bitwise OR of all detected :class:`InstructionProperty` flags.
    """
    if block.is_empty:
        return InstructionProperty.NONE

    props = InstructionProperty.NONE

    if block.pragma == ExecModel.VECTOR:
        props |= InstructionProperty.EXEC_MASKED
    elif block.pragma == ExecModel.BRANCH:
        props |= InstructionProperty.IS_BRANCH
        props |= InstructionProperty.IGNORES_EXEC

    for node in block.body.walk():
        if node.kind == SemaNodeKind.ASSIGN and node.children:
            lhs = _unwrap_cast(node.children[0])
            if lhs.kind == SemaNodeKind.ID and lhs.id_name:
                if lhs.id_name == 'SCC':
                    props |= InstructionProperty.WRITES_SCC
                elif lhs.id_name == 'EXEC':
                    props |= InstructionProperty.WRITES_EXEC
            if lhs.kind == SemaNodeKind.ARRAYDEREF and lhs.children:
                arr = _unwrap_cast(lhs.children[0])
                if arr.kind == SemaNodeKind.ID:
                    if arr.id_name == 'VCC':
                        props |= InstructionProperty.WRITES_VCC
                    elif arr.id_name == 'EXEC':
                        props |= InstructionProperty.WRITES_EXEC

        if node.kind == SemaNodeKind.CALL and node.call_name:
            cn = node.call_name
            if cn in _CROSS_LANE_CALLS:
                props |= InstructionProperty.CROSS_LANE
            if cn in _MATRIX_CALLS:
                props |= InstructionProperty.IS_MATRIX
            if cn in _BARRIER_CALLS:
                props |= InstructionProperty.IS_BARRIER
            if cn in _WAITCNT_CALLS:
                props |= InstructionProperty.IS_WAITCNT
            if cn in _ADDR_CALC_CALLS:
                props |= InstructionProperty.IS_MEMORY
            if cn == 'endpgm':
                props |= InstructionProperty.IS_ENDPGM

        if node.kind == SemaNodeKind.ARRAYDEREF and node.children:
            arr = _unwrap_cast(node.children[0])
            if arr.kind == SemaNodeKind.ID:
                if arr.id_name == 'MEM':
                    props |= InstructionProperty.IS_MEMORY
                elif arr.id_name == 'LDS':
                    props |= InstructionProperty.IS_MEMORY
                elif arr.id_name == 'VCC':
                    props |= InstructionProperty.READS_VCC
                elif arr.id_name == 'EXEC':
                    props |= InstructionProperty.READS_EXEC

    return props


def _unwrap_cast(node: SemaNode) -> SemaNode:
    while node.kind == SemaNodeKind.CAST and node.children:
        node = node.children[0]
    return node


@dataclass
class PropertySummary:
    """Summary of derived properties for a set of instructions."""

    total: int = 0
    exec_masked: int = 0
    writes_scc: int = 0
    writes_vcc: int = 0
    writes_exec: int = 0
    cross_lane: int = 0
    is_matrix: int = 0
    is_memory: int = 0
    is_branch: int = 0
    is_barrier: int = 0
    is_waitcnt: int = 0


def summarize_properties(
    blocks: dict[str, SemaBlock],
) -> PropertySummary:
    """Compute property statistics across all instructions."""
    s = PropertySummary()
    for block in blocks.values():
        if block.is_empty:
            continue
        s.total += 1
        props = derive_properties(block)
        if InstructionProperty.EXEC_MASKED in props:
            s.exec_masked += 1
        if InstructionProperty.WRITES_SCC in props:
            s.writes_scc += 1
        if InstructionProperty.WRITES_VCC in props:
            s.writes_vcc += 1
        if InstructionProperty.WRITES_EXEC in props:
            s.writes_exec += 1
        if InstructionProperty.CROSS_LANE in props:
            s.cross_lane += 1
        if InstructionProperty.IS_MATRIX in props:
            s.is_matrix += 1
        if InstructionProperty.IS_MEMORY in props:
            s.is_memory += 1
        if InstructionProperty.IS_BRANCH in props:
            s.is_branch += 1
        if InstructionProperty.IS_BARRIER in props:
            s.is_barrier += 1
        if InstructionProperty.IS_WAITCNT in props:
            s.is_waitcnt += 1
    return s

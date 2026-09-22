# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Semantic fingerprinting for cross-ISA equivalence detection.

Computes a stable SHA-256 hash of a :class:`~amdisa.sema_ast.SemaBlock`'s
semantic content. Two instructions with identical fingerprints are
semantically equivalent — this enables algorithmic cross-ISA comparison
for DBT legalization, replacing hand-maintained mnemonic rename maps.

**What is hashed:** tree structure, node kinds, type annotations, cast
targets, call targets, wavefront context IDs (SCC, VCC, EXEC, etc.),
operand class tags (S, D), operand index literals, bit-range bounds.

**What is normalized:** local temporary names (alpha-renameable), general
literal values, comment text.

**IMPORTANT:** Fingerprinting must operate on pre-enrichment ASTs.
Enrichment adds encoding-specific modifier nodes (NEG/ABS/CLAMP/OMOD)
that vary across ISAs even when the core semantics are identical.
"""

from __future__ import annotations

import hashlib
import logging

from amdisa.sema_ast import SemaBlock, SemaNode, SemaNodeKind

_log = logging.getLogger(__name__)

_CONTEXT_IDS: frozenset[str] = frozenset(
    {
        'SCC',
        'VCC',
        'EXEC',
        'EXEC_LO',
        'MEM',
        'LDS',
        'VGPR',
        'SGPR',
        'laneId',
        'M0',
        'PC',
        'TRAPSTS',
        'VCCZ',
        'EXECZ',
        'MODE',
        'WAVE64',
        'TBA',
        'TTMP',
        'SHADER_CYCLES_HI',
        'SHADER_CYCLES_LO',
        'WAVE_STATUS',
        'ROUND_MODE',
        'DENORM',
        'OPSEL',
        'OPSEL_HI',
        'BARRIER_STATE',
        'HW_REGISTERS',
    }
)

_OPERAND_TAGS: frozenset[str] = frozenset({'S', 'D'})

_LIT_HASH_PARENTS: frozenset[SemaNodeKind] = frozenset(
    {
        SemaNodeKind.INSTOPERAND,
        SemaNodeKind.ARRAYSLICE,
        SemaNodeKind.ARRAYSLICESIZE,
    }
)


def fingerprint(block: SemaBlock) -> bytes:
    """Compute a stable hash of a SemaBlock's semantic content.

    Returns a 32-byte SHA-256 digest.

    Raises:
        AssertionError: If ``block.enriched`` is True.
    """
    assert not block.enriched, (
        "fingerprinting must operate on pre-enrichment ASTs; "
        "call fingerprint() before sema_enrich.enrich_block()"
    )
    h = hashlib.sha256()
    h.update(b'v1')
    h.update(block.pragma.value.encode())
    _hash_node(h, block.body, parent_kind=None)
    return h.digest()


def _hash_node(
    h: hashlib._Hash,
    node: SemaNode,
    parent_kind: SemaNodeKind | None,
) -> None:
    h.update(node.kind.value.encode())

    if node.ty:
        h.update(f'{node.ty.base}{node.ty.size}'.encode())

    if node.cast_target:
        h.update(f'cast:{node.cast_target.base}{node.cast_target.size}'.encode())

    if node.call_name:
        h.update(f'call:{node.call_name}'.encode())

    if node.kind == SemaNodeKind.ID and node.id_name:
        if node.id_name in _CONTEXT_IDS or node.id_name in _OPERAND_TAGS:
            h.update(f'id:{node.id_name}'.encode())

    if node.kind == SemaNodeKind.LIT and parent_kind in _LIT_HASH_PARENTS:
        h.update(f'lit:{node.lit_value}'.encode())

    for child in node.children:
        _hash_node(h, child, parent_kind=node.kind)


def are_equivalent(a: SemaBlock, b: SemaBlock) -> bool:
    """True if two SemaBlocks have identical semantic fingerprints."""
    return fingerprint(a) == fingerprint(b)


def build_equivalence_map(
    src_blocks: dict[str, SemaBlock],
    dst_blocks: dict[str, SemaBlock],
) -> dict[str, str | None]:
    """Map each source mnemonic to its semantically equivalent target mnemonic.

    Uses O(N+M) hash-table lookup: builds a ``dict[bytes, str]`` of target
    fingerprints, then looks up each source fingerprint.

    Args:
        src_blocks: Source ISA SemaBlocks keyed by mnemonic.
        dst_blocks: Target ISA SemaBlocks keyed by mnemonic.

    Returns:
        Dict mapping source mnemonic to equivalent target mnemonic, or
        ``None`` if no equivalent exists.
    """
    dst_by_fp: dict[bytes, str] = {}
    for mnemonic, block in dst_blocks.items():
        if block.is_empty:
            continue
        fp = fingerprint(block)
        if fp in dst_by_fp:
            _log.debug(
                'fingerprint collision: %s and %s have identical semantics; '
                'keeping %s',
                dst_by_fp[fp],
                mnemonic,
                dst_by_fp[fp],
            )
        else:
            dst_by_fp[fp] = mnemonic

    result: dict[str, str | None] = {}
    for mnemonic, block in src_blocks.items():
        if block.is_empty:
            result[mnemonic] = None
            continue
        fp = fingerprint(block)
        result[mnemonic] = dst_by_fp.get(fp)

    return result

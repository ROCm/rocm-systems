# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AST-based semantic equivalence for DBT legalization.

Provides :func:`build_sema_legalization` which uses SemaAST fingerprints
to detect semantically equivalent instructions across ISA pairs. This
supplements (and will eventually replace) the hand-maintained mnemonic
rename map in ``legalization.py``.

Two instructions are semantically equivalent if their pre-enrichment
SemaAST fingerprints match — regardless of mnemonic name differences
(e.g., ``DS_READ_B32`` → ``DS_LOAD_B32``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from amdisa.sema_ast import SemaBlock
from amdisa.sema_fingerprint import build_equivalence_map, fingerprint

_log = logging.getLogger(__name__)


@dataclass
class SemaEquivalence:
    """Result of AST-based equivalence analysis for one ISA pair.

    Attributes:
        src_isa: Source ISA name (e.g., ``cdna4``).
        dst_isa: Target ISA name (e.g., ``rdna4``).
        equivalences: Maps source mnemonic to equivalent target mnemonic,
            or None if no equivalent found.
        identity_count: Instructions with same mnemonic and same fingerprint.
        rename_count: Instructions with different mnemonic but same fingerprint.
        no_match_count: Instructions with no fingerprint match on the target.
        stub_count: Source instructions that are stubs (empty semantics).
    """

    src_isa: str
    dst_isa: str
    equivalences: dict[str, str | None]
    identity_count: int = 0
    rename_count: int = 0
    no_match_count: int = 0
    stub_count: int = 0


def build_sema_equivalences(
    src_isa: str,
    dst_isa: str,
    src_blocks: dict[str, SemaBlock],
    dst_blocks: dict[str, SemaBlock],
) -> SemaEquivalence:
    """Build AST-based equivalence map for an ISA pair.

    Uses O(N+M) fingerprint comparison via
    :func:`~amdisa.sema_fingerprint.build_equivalence_map`.

    Args:
        src_isa: Source ISA name.
        dst_isa: Target ISA name.
        src_blocks: Source ISA SemaBlocks keyed by mnemonic.
        dst_blocks: Target ISA SemaBlocks keyed by mnemonic.

    Returns:
        :class:`SemaEquivalence` with per-instruction equivalence mapping
        and summary statistics.
    """
    equiv_map = build_equivalence_map(src_blocks, dst_blocks)

    identity = 0
    rename = 0
    no_match = 0
    stub = 0

    for src_name, dst_name in equiv_map.items():
        src_block = src_blocks[src_name]
        if src_block.is_empty:
            stub += 1
            continue
        if dst_name is None:
            no_match += 1
        elif dst_name == src_name:
            identity += 1
        else:
            rename += 1
            _log.debug(
                '%s→%s: %s is equivalent to %s (renamed)',
                src_isa,
                dst_isa,
                src_name,
                dst_name,
            )

    return SemaEquivalence(
        src_isa=src_isa,
        dst_isa=dst_isa,
        equivalences=equiv_map,
        identity_count=identity,
        rename_count=rename,
        no_match_count=no_match,
        stub_count=stub,
    )


def merge_equivalences_into_union_find(
    sema_equiv: SemaEquivalence,
    canon_to_ids: dict[str, list[int]],
    uf: object,
) -> int:
    """Merge AST-based equivalences into an existing union-find structure.

    For each renamed equivalence (src_name → dst_name where src != dst),
    merges the e-class IDs of src and dst in the union-find. This
    supplements the mnemonic rename map with AST-derived equivalences.

    Args:
        sema_equiv: The AST equivalence result.
        canon_to_ids: Maps canonical mnemonic to list of e-class IDs
            (from ``LegalizationGenerator``).
        uf: Union-find object with ``union(a, b)`` method.

    Returns:
        Number of new merges performed.
    """
    merges = 0
    for src_name, dst_name in sema_equiv.equivalences.items():
        if dst_name is None or dst_name == src_name:
            continue
        src_ids = canon_to_ids.get(src_name, [])
        dst_ids = canon_to_ids.get(dst_name, [])
        if src_ids and dst_ids:
            uf.union(src_ids[0], dst_ids[0])
            merges += 1
    return merges

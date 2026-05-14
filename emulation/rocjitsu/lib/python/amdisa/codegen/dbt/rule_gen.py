# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Auto-generate DBT TranslationRule entries from SemaAST.

Combines semantic fingerprinting, instruction properties,
and encoding metadata to classify every source instruction for a given
cross-ISA pair. Produces a list of :class:`TranslationRule` entries that
the C++ ``BinaryTranslator`` consumes at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from amdisa.sema_ast import SemaBlock
from amdisa.sema_fingerprint import fingerprint
from amdisa.sema_properties import InstructionProperty, derive_properties
from amdisa.codegen.dbt.sema_equivalence import build_sema_equivalences


class RuleAction(Enum):
    IDENTITY = auto()
    SUBSTITUTE = auto()
    LOWER = auto()
    EXPAND = auto()


class ExpansionStrategy(Enum):
    GENERIC = auto()
    MFMA_TO_WMMA = auto()
    WMMA_TO_MFMA = auto()
    ACCVGPR = auto()
    CMP_REMOVED = auto()
    CROSS_LANE_ADJUST = auto()
    WAITCNT_REMAP = auto()


@dataclass
class TranslationRule:
    """One instruction's translation rule for a cross-ISA pair."""

    src_mnemonic: str
    action: RuleAction
    dst_mnemonic: str | None = None
    expansion: ExpansionStrategy | None = None
    src_properties: InstructionProperty = InstructionProperty.NONE
    dst_properties: InstructionProperty = InstructionProperty.NONE


def generate_rules(
    src_isa: str,
    dst_isa: str,
    src_blocks: dict[str, SemaBlock],
    dst_blocks: dict[str, SemaBlock],
) -> list[TranslationRule]:
    """Generate translation rules for a cross-ISA pair.

    Algorithm:
    1. Build fingerprint-based equivalence map (O(N+M))
    2. For each source instruction:
       a. If stub → EXPAND with GENERIC
       b. If fingerprint matches a target instruction:
          - Same mnemonic → IDENTITY
          - Different mnemonic → SUBSTITUTE
       c. If no match → classify by properties:
          - IS_MATRIX → MFMA_TO_WMMA or WMMA_TO_MFMA
          - IS_WAITCNT → WAITCNT_REMAP
          - CROSS_LANE → CROSS_LANE_ADJUST
          - Otherwise → LOWER (generic)

    Args:
        src_isa: Source ISA name.
        dst_isa: Target ISA name.
        src_blocks: Source ISA SemaBlocks.
        dst_blocks: Target ISA SemaBlocks.

    Returns:
        List of TranslationRule, one per source instruction.
    """
    equiv = build_sema_equivalences(src_isa, dst_isa, src_blocks, dst_blocks)
    rules: list[TranslationRule] = []

    for src_name, src_block in src_blocks.items():
        src_props = derive_properties(src_block)
        dst_name = equiv.equivalences.get(src_name)

        if src_block.is_empty:
            rules.append(TranslationRule(
                src_mnemonic=src_name,
                action=RuleAction.EXPAND,
                expansion=ExpansionStrategy.GENERIC,
                src_properties=src_props,
            ))
            continue

        if dst_name is not None:
            dst_block = dst_blocks.get(dst_name)
            dst_props = derive_properties(dst_block) if dst_block else InstructionProperty.NONE

            if dst_name == src_name:
                rules.append(TranslationRule(
                    src_mnemonic=src_name,
                    action=RuleAction.IDENTITY,
                    dst_mnemonic=dst_name,
                    src_properties=src_props,
                    dst_properties=dst_props,
                ))
            else:
                rules.append(TranslationRule(
                    src_mnemonic=src_name,
                    action=RuleAction.SUBSTITUTE,
                    dst_mnemonic=dst_name,
                    src_properties=src_props,
                    dst_properties=dst_props,
                ))
            continue

        expansion = _classify_no_match(src_name, src_props)
        rules.append(TranslationRule(
            src_mnemonic=src_name,
            action=RuleAction.EXPAND if expansion else RuleAction.LOWER,
            expansion=expansion,
            src_properties=src_props,
        ))

    return rules


def _classify_no_match(
    name: str, props: InstructionProperty,
) -> ExpansionStrategy | None:
    """Classify an unmatched instruction by its properties."""
    if InstructionProperty.IS_MATRIX in props:
        if 'WMMA' in name:
            return ExpansionStrategy.WMMA_TO_MFMA
        return ExpansionStrategy.MFMA_TO_WMMA

    if InstructionProperty.IS_WAITCNT in props:
        return ExpansionStrategy.WAITCNT_REMAP

    if InstructionProperty.CROSS_LANE in props:
        return ExpansionStrategy.CROSS_LANE_ADJUST

    if 'ACCVGPR' in name:
        return ExpansionStrategy.ACCVGPR

    return None


@dataclass
class RuleSummary:
    """Summary statistics for a generated rule set."""

    total: int = 0
    identity: int = 0
    substitute: int = 0
    lower: int = 0
    expand: int = 0


def summarize_rules(rules: list[TranslationRule]) -> RuleSummary:
    """Compute summary statistics for a set of translation rules."""
    s = RuleSummary(total=len(rules))
    for r in rules:
        if r.action == RuleAction.IDENTITY:
            s.identity += 1
        elif r.action == RuleAction.SUBSTITUTE:
            s.substitute += 1
        elif r.action == RuleAction.LOWER:
            s.lower += 1
        elif r.action == RuleAction.EXPAND:
            s.expand += 1
    return s

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
from amdisa.layout_catalog import (
    MatrixLayoutDescriptor,
    compute_xor_mask,
    derive_layout_descriptor,
)


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


# =========================================================================
# Matrix expand rule generation from layout catalog
# =========================================================================

def _extract_src_type(mnemonic: str) -> str:
    """Extract the source element type from a matrix mnemonic."""
    parts = mnemonic.split('_')
    return parts[-1] if parts else ''


def _find_best_target(
    src_mn: str,
    src_desc: MatrixLayoutDescriptor,
    dst_by_mn: dict[str, MatrixLayoutDescriptor],
) -> tuple[str, MatrixLayoutDescriptor] | None:
    """Find the best matching target for a source matrix instruction."""
    src_type = _extract_src_type(src_mn)
    for dst_mn, dst_desc in dst_by_mn.items():
        if dst_desc.m != src_desc.m or dst_desc.n != src_desc.n:
            continue
        if _extract_src_type(dst_mn) == src_type:
            return (dst_mn, dst_desc)
    return None


@dataclass(frozen=True)
class MatrixExpandRule:
    """Auto-derived matrix translation rule with XOR permutation data."""

    src_mnemonic: str
    dst_mnemonic: str
    xor_byte_mask: int
    range_start: int
    range_end: int
    src_m: int
    src_n: int
    src_k: int
    dst_vgprs: int


def generate_matrix_expand_rules(
    src_mnemonics: list[str],
    dst_mnemonics: list[str],
) -> list[MatrixExpandRule]:
    """Generate matrix expand rules for a cross-ISA pair.

    For each source matrix instruction, finds the best matching target
    instruction (same M×N dimensions, compatible element types), derives
    both layout descriptors, and computes the XOR mask.

    Args:
        src_mnemonics: Source ISA matrix instruction mnemonics.
        dst_mnemonics: Target ISA matrix instruction mnemonics.

    Returns:
        List of MatrixExpandRule with auto-derived XOR masks.
    """
    dst_by_mn: dict[str, MatrixLayoutDescriptor] = {}
    for mn in dst_mnemonics:
        desc = derive_layout_descriptor(mn)
        if desc:
            dst_by_mn[mn] = desc

    rules: list[MatrixExpandRule] = []
    for src_mn in src_mnemonics:
        src_desc = derive_layout_descriptor(src_mn)
        if not src_desc:
            continue

        match = _find_best_target(src_mn, src_desc, dst_by_mn)
        if not match:
            continue
        dst_mn, dst_desc = match

        xor_result = compute_xor_mask(src_desc, dst_desc)
        if xor_result is None:
            continue

        xor_byte, start, end = xor_result
        rules.append(MatrixExpandRule(
            src_mnemonic=src_mn,
            dst_mnemonic=dst_mn,
            xor_byte_mask=xor_byte,
            range_start=start,
            range_end=end,
            src_m=src_desc.m,
            src_n=src_desc.n,
            src_k=src_desc.k,
            dst_vgprs=src_desc.dst_vgprs,
        ))

    return rules


def emit_matrix_conversions_header(
    rules: list[MatrixExpandRule],
    guard: str = 'ROCJITSU_MATRIX_CONVERSIONS_H',
) -> str:
    """Generate a C++ header with a constexpr lookup table of matrix conversions.

    The table replaces hand-written LaneLayout constants and ExpandFn entries.
    The runtime BinaryTranslator uses this for ds_bpermute address computation.
    """
    lines = [
        f'#pragma once',
        f'// Auto-generated by amdisa layout_catalog — do not edit.',
        f'',
        f'#include <cstdint>',
        f'',
        f'namespace rocjitsu {{',
        f'',
        f'struct MatrixConversion {{',
        f'  const char* src_mnemonic;',
        f'  const char* dst_mnemonic;',
        f'  uint32_t xor_byte_mask;',
        f'  uint8_t range_start;',
        f'  uint8_t range_end;',
        f'  uint8_t dst_vgprs;',
        f'}};',
        f'',
        f'inline constexpr MatrixConversion kMatrixConversions[] = {{',
    ]

    for r in sorted(rules, key=lambda r: r.src_mnemonic):
        src = r.src_mnemonic.lower()
        dst = r.dst_mnemonic.lower()
        lines.append(
            f'    {{"{src}", "{dst}", '
            f'{r.xor_byte_mask}u, {r.range_start}, {r.range_end}, '
            f'{r.dst_vgprs}}},'
        )

    lines.extend([
        f'}};',
        f'',
        f'inline constexpr size_t kMatrixConversionCount = '
        f'sizeof(kMatrixConversions) / sizeof(kMatrixConversions[0]);',
        f'',
        f'}}  // namespace rocjitsu',
        f'',
    ])

    return '\n'.join(lines)

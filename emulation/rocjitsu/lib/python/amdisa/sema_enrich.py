# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Post-parse enrichment for SemaAST blocks.

Transforms raw SemaAST trees into enriched versions by:
- Stripping spurious RETURN_DATA writes from non-RTN DS atomics (XML bug)
- Wrapping VOP3 source operands with NEG/ABS modifier application
- Wrapping VOP3 destination with CLAMP/OMOD modifier application
- Applying known-bug patches keyed by instruction name and ISA

EXEC mask wrapping is NOT done here — it is the lowering pass's
responsibility (``sema_lower.py``).
"""

from __future__ import annotations

import dataclasses

from amdisa.sema_ast import (
    ExecModel,
    SemaBlock,
    SemaNode,
    SemaNodeKind,
    SemaType,
    validate_types,
    validate_well_formed,
)

_DS_ATOMIC_PREFIXES = (
    'DS_ADD_', 'DS_SUB_', 'DS_RSUB_', 'DS_MIN_', 'DS_MAX_',
    'DS_AND_', 'DS_OR_', 'DS_XOR_', 'DS_MSKOR_', 'DS_CMPSTORE_',
    'DS_INC_', 'DS_DEC_', 'DS_PK_ADD_', 'DS_COND_SUB_',
    'DS_SUB_CLAMP_',
)


def _is_ds_atomic(name: str) -> bool:
    return any(name.startswith(p) for p in _DS_ATOMIC_PREFIXES)


def _is_ds_load_addtid(name: str) -> bool:
    return name.startswith('DS_LOAD_ADDTID_') or name.startswith('DS_STORE_ADDTID_')


def enrich_block(
    block: SemaBlock,
    enc_field_names: frozenset[str] | None = None,
) -> SemaBlock:
    """Apply enrichment transformations to a SemaBlock.

    Args:
        block: The raw (pre-enrichment) SemaBlock from the parser.
        enc_field_names: Set of encoding field names (e.g., ``{'neg', 'abs',
            'clamp', 'omod'}``). Used to determine which VOP3 modifiers to
            inject. If None, no modifier wrapping is applied.

    Returns:
        A new SemaBlock with ``enriched=True``. The original is not modified.
    """
    if block.is_empty:
        return SemaBlock(
            instruction_name=block.instruction_name,
            pragma=block.pragma,
            body=block.body,
            enriched=True,
        )

    body = block.body
    name = block.instruction_name

    body = _fix_non_rtn_atomics(name, body)

    if enc_field_names:
        body = _add_vop3_modifiers(body, enc_field_names, block.pragma)

    result = SemaBlock(
        instruction_name=name,
        pragma=block.pragma,
        body=body,
        enriched=True,
    )
    return result


def _fix_non_rtn_atomics(name: str, body: SemaNode) -> SemaNode:
    """Strip spurious RETURN_DATA write from non-RTN DS atomics."""
    if '_RTN_' in name:
        return body
    if not (_is_ds_atomic(name) or _is_ds_load_addtid(name)):
        return body
    if body.kind != SemaNodeKind.SEQ:
        return body

    new_children = tuple(
        stmt for stmt in body.children
        if not _is_return_data_assign(stmt)
    )

    if len(new_children) == len(body.children):
        return body

    return dataclasses.replace(body, children=new_children)


def _is_return_data_assign(node: SemaNode) -> bool:
    """Check if a node is an ASSIGN whose LHS targets RETURN_DATA."""
    if node.kind != SemaNodeKind.ASSIGN or not node.children:
        return False
    lhs = node.children[0]
    while lhs.kind == SemaNodeKind.CAST and lhs.children:
        lhs = lhs.children[0]
    return (lhs.kind == SemaNodeKind.ID and lhs.id_name == 'RETURN_DATA')


def _add_vop3_modifiers(
    body: SemaNode,
    enc_fields: frozenset[str],
    pragma: ExecModel,
) -> SemaNode:
    """Wrap source operands with NEG/ABS and destination with CLAMP/OMOD.

    Source modifiers (NEG, ABS) are applied per source operand:
      if (abs & (1 << idx)) src = fabs(src);
      if (neg & (1 << idx)) src = -src;

    Destination modifiers are applied to the result:
      if (omod) result *= {2, 4, 0.5};
      if (clamp) result = clamp(result, 0.0, 1.0);

    These are injected as CALL nodes to apply_src_mod / apply_omod /
    apply_clamp helper functions, which the lowering pass emits as
    inline C++.
    """
    has_neg = 'neg' in enc_fields
    has_abs = 'abs' in enc_fields
    has_clamp = 'clamp' in enc_fields
    has_omod = 'omod' in enc_fields

    if not (has_neg or has_abs or has_clamp or has_omod):
        return body

    if has_neg or has_abs:
        body = _wrap_src_modifiers(body, has_neg, has_abs)

    if has_clamp or has_omod:
        body = _wrap_dst_modifiers(body, has_clamp, has_omod)

    return body


def _wrap_src_modifiers(
    node: SemaNode, has_neg: bool, has_abs: bool,
) -> SemaNode:
    """Recursively wrap INSTOPERAND(S, N) reads with src modifier application."""
    if (node.kind == SemaNodeKind.CAST
            and node.children
            and node.children[0].kind == SemaNodeKind.INSTOPERAND):
        inner = node.children[0]
        if (inner.children
                and inner.children[0].kind == SemaNodeKind.ID
                and inner.children[0].id_name == 'S'):
            idx_node = inner.children[1] if len(inner.children) > 1 else None
            idx_lit = idx_node.lit_value if idx_node else '0'
            return SemaNode(
                kind=SemaNodeKind.CALL,
                ty=node.ty,
                call_name='apply_src_mod',
                children=(
                    SemaNode(SemaNodeKind.ID, id_name='apply_src_mod'),
                    node,
                    SemaNode(SemaNodeKind.LIT, lit_value=idx_lit),
                    SemaNode(SemaNodeKind.LIT,
                             lit_value='1' if has_neg else '0'),
                    SemaNode(SemaNodeKind.LIT,
                             lit_value='1' if has_abs else '0'),
                ),
            )

    if not node.children:
        return node

    new_children = tuple(
        _wrap_src_modifiers(c, has_neg, has_abs) for c in node.children
    )
    if new_children == node.children:
        return node
    return dataclasses.replace(node, children=new_children)


def _wrap_dst_modifiers(
    node: SemaNode, has_clamp: bool, has_omod: bool,
) -> SemaNode:
    """Wrap the RHS of the top-level ASSIGN with OMOD then CLAMP."""
    if node.kind == SemaNodeKind.ASSIGN and len(node.children) == 2:
        lhs, rhs = node.children
        if _is_dst_operand(lhs):
            if has_omod:
                rhs = SemaNode(
                    kind=SemaNodeKind.CALL,
                    ty=rhs.ty,
                    call_name='apply_omod',
                    children=(
                        SemaNode(SemaNodeKind.ID, id_name='apply_omod'),
                        rhs,
                    ),
                )
            if has_clamp:
                rhs = SemaNode(
                    kind=SemaNodeKind.CALL,
                    ty=rhs.ty,
                    call_name='apply_clamp',
                    children=(
                        SemaNode(SemaNodeKind.ID, id_name='apply_clamp'),
                        rhs,
                    ),
                )
            return dataclasses.replace(node, children=(lhs, rhs))

    if node.children:
        new_children = tuple(
            _wrap_dst_modifiers(c, has_clamp, has_omod) for c in node.children
        )
        if new_children != node.children:
            return dataclasses.replace(node, children=new_children)

    return node


def _is_dst_operand(node: SemaNode) -> bool:
    """Check if a node is a destination operand (.instoperand(D, N))."""
    if node.kind == SemaNodeKind.INSTOPERAND:
        if node.children and node.children[0].kind == SemaNodeKind.ID:
            return node.children[0].id_name == 'D'
    if node.kind == SemaNodeKind.CAST and node.children:
        return _is_dst_operand(node.children[0])
    return False


def build_sema_block(
    instruction_name: str,
    xml_sema: SemaBlock | None,
    enc_field_names: frozenset[str] | None = None,
) -> SemaBlock | None:
    """Four-stage composable pipeline to build a complete SemaBlock.

    Stage 1: Start from XML parse result (may be None or empty).
    Stage 2: Enrich with encoding-format modifiers + bug fixes.
    Stage 3: Helper resolution (deferred to lowering — .call nodes
             are resolved during C++ emission, not here).
    Stage 4: Fallback (returns None — caller uses string-tag generator).

    Args:
        instruction_name: The instruction mnemonic.
        xml_sema: Parsed SemaBlock from ``sema_parser.py``, or None.
        enc_field_names: Encoding field names for modifier injection.

    Returns:
        Enriched SemaBlock, or None if no SemaBlock could be built.
    """
    if xml_sema is None or xml_sema.is_empty:
        return None

    block = enrich_block(xml_sema, enc_field_names)
    return block

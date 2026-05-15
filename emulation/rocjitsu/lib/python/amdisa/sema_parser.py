# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Parse instruction semantics into SemaAST nodes.

Parses ``amdgpu_isa_*.semantics.xml`` files — AMD's machine-readable
execution semantics — into :class:`~amdisa.sema_ast.SemaBlock`
trees. Each instruction's ``<InstructionSemantics>`` element becomes one
``SemaBlock`` containing a tree of ``SemaNode`` objects.

The parser handles:
- All 66+ XML op types (mapped via :data:`_OP_TYPE_MAP`)
- ``<lit>`` and ``<id>`` leaf elements
- ``<ty>`` type annotations including ``<lambda>``, ``<arr>``, ``<rec>``
- ``:pragma`` extraction for execution model (scalar/vector/branch)
- ``.cast`` target type extraction from ``<type>`` child
- ``.call`` callee name extraction from first ``<id>`` child
- Stub detection (empty ``<op type="" />``)
- ``laneID`` → ``laneId`` normalization (XML inconsistency)
- Unknown op type fallback to SEQ with warning
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET

from amdisa.sema_ast import (
    ExecModel,
    SemaBlock,
    SemaNode,
    SemaNodeKind,
    SemaType,
    _STATEMENT_KINDS,
    validate_types,
    validate_well_formed,
)

_log = logging.getLogger(__name__)

_SUPPORTED_SCHEMA_MAJOR = 1

_OP_TYPE_MAP: dict[str, SemaNodeKind] = {
    k.value: k for k in SemaNodeKind
    if not k.value.startswith('_')
}


def parse_semantics_xml(path: str) -> dict[str, SemaBlock]:
    """Parse a semantics file into a dict of SemaBlocks keyed by mnemonic.

    Args:
        path: Filesystem path to the ``*.semantics.xml`` file.

    Returns:
        Dict mapping instruction mnemonic to its parsed SemaBlock.
        Stub instructions (empty ``<op type="" />``) are included with
        ``block.is_empty == True``.

    Raises:
        ValueError: If the schema major version exceeds the supported version.
    """
    tree = ET.parse(path)
    root = tree.getroot()

    version_elem = root.find('.//SchemaVersion')
    if version_elem is not None and version_elem.text:
        major = int(version_elem.text.split('.')[0])
        if major > _SUPPORTED_SCHEMA_MAJOR:
            raise ValueError(
                f"Unsupported schema version {version_elem.text} "
                f"(max supported major: {_SUPPORTED_SCHEMA_MAJOR})"
            )

    result: dict[str, SemaBlock] = {}
    for inst_elem in root.iter('Instruction'):
        name_elem = inst_elem.find('InstructionName')
        if name_elem is None or not name_elem.text:
            continue
        name = name_elem.text.strip()
        sema_elem = inst_elem.find('InstructionSemantics')
        if sema_elem is None:
            continue
        block = _parse_instruction(name, sema_elem)
        if block is not None:
            result[name] = block
    return result


def _parse_instruction(name: str, sema_elem: ET.Element) -> SemaBlock | None:
    """Parse one instruction's ``<InstructionSemantics>`` into a SemaBlock."""
    ops = [c for c in sema_elem if c.tag == 'op']
    if not ops:
        return None

    root_op = ops[0]
    op_type = root_op.get('type', '')
    if not op_type:
        return SemaBlock(
            name, ExecModel.UNKNOWN,
            SemaNode(SemaNodeKind.SEQ, children=()),
        )

    root_node = _parse_node(root_op)
    if root_node is None:
        return SemaBlock(
            name, ExecModel.UNKNOWN,
            SemaNode(SemaNodeKind.SEQ, children=()),
        )

    pragma = ExecModel.UNKNOWN
    body: SemaNode
    if root_node.kind == SemaNodeKind.PRAGMA and root_node.children:
        first = root_node.children[0]
        if first.kind == SemaNodeKind.LIT and first.lit_value:
            try:
                pragma = ExecModel(first.lit_value)
            except ValueError:
                pragma = ExecModel.UNKNOWN
        body = root_node.children[1] if len(root_node.children) >= 2 else \
            SemaNode(SemaNodeKind.SEQ, children=())
    else:
        body = root_node

    block = SemaBlock(name, pragma, body)

    try:
        validate_types(body)
        validate_well_formed(body)
    except AssertionError as e:
        _log.warning("Validation failed for %s: %s", name, e)

    return block


def _parse_type(elem: ET.Element) -> SemaType | None:
    """Parse a ``<ty><t base="..." size="..."/></ty>`` annotation.

    Handles compound type forms: ``<t>`` (scalar), ``<lambda>`` (function),
    ``<arr>`` (array), ``<rec>`` (record/struct — returns None with warning).
    """
    ty_elem = elem.find('ty')
    if ty_elem is None:
        return None

    t_elem = ty_elem.find('t')
    if t_elem is not None:
        return SemaType(
            t_elem.get('base', 'B'),
            int(t_elem.get('size', '0')),
        )

    lambda_elem = ty_elem.find('lambda')
    if lambda_elem is not None:
        ret = lambda_elem.find('ret/t')
        if ret is not None:
            return SemaType(
                ret.get('base', 'B'),
                int(ret.get('size', '0')),
            )
        ret_direct = lambda_elem.find('ret')
        if ret_direct is not None:
            t_in_ret = ret_direct.find('t')
            if t_in_ret is not None:
                return SemaType(
                    t_in_ret.get('base', 'B'),
                    int(t_in_ret.get('size', '0')),
                )

    arr_elem = ty_elem.find('arr')
    if arr_elem is not None:
        arr_t = arr_elem.find('t')
        if arr_t is not None:
            return SemaType(
                arr_t.get('base', 'B'),
                int(arr_t.get('size', '0')),
            )

    rec_elem = ty_elem.find('rec')
    if rec_elem is not None:
        _log.debug(
            "Unhandled <rec> type in element '%s'; treating as untyped.",
            elem.get('type', '<unknown>'),
        )
        return None

    return None


def _parse_node(elem: ET.Element) -> SemaNode | None:
    """Recursively parse an XML element into a SemaNode."""
    tag = elem.tag

    if tag == 'op':
        return _parse_op(elem)
    elif tag == 'lit':
        return _parse_lit(elem)
    elif tag == 'id':
        return _parse_id(elem)
    else:
        children: list[SemaNode] = []
        for child in elem:
            if child.tag in ('ty', 'type'):
                continue
            child_node = _parse_node(child)
            if child_node is not None:
                children.append(child_node)
        if children:
            if len(children) == 1:
                return children[0]
            return SemaNode(SemaNodeKind.SEQ, children=tuple(children))
        return None


def _parse_op(elem: ET.Element) -> SemaNode | None:
    """Parse an ``<op type="...">`` element."""
    op_type = elem.get('type', '')
    if not op_type:
        return None

    kind = _OP_TYPE_MAP.get(op_type)
    unknown_op: str | None = None
    if kind is None:
        _log.warning("Unknown op type: %r", op_type)
        kind = SemaNodeKind.SEQ
        unknown_op = op_type

    ty = _parse_type(elem)

    cast_target: SemaType | None = None
    if kind == SemaNodeKind.CAST:
        type_child = elem.find('type')
        if type_child is not None:
            t_elem = type_child.find('t')
            if t_elem is not None:
                cast_target = SemaType(
                    t_elem.get('base', 'B'),
                    int(t_elem.get('size', '0')),
                )

    parsed_children: list[SemaNode] = []
    for child in elem:
        if child.tag in ('ty', 'type'):
            continue
        child_node = _parse_node(child)
        if child_node is not None:
            parsed_children.append(child_node)

    call_name: str | None = None
    call_lambda: SemaType | None = None
    if kind == SemaNodeKind.CALL and parsed_children:
        first = parsed_children[0]
        if first.kind == SemaNodeKind.ID:
            call_name = first.id_name
            if first.ty and first.ty.base != 'B':
                call_lambda = first.ty

    if kind in _STATEMENT_KINDS:
        ty = None

    return SemaNode(
        kind=kind,
        ty=ty,
        children=tuple(parsed_children),
        cast_target=cast_target,
        call_name=call_name,
        call_lambda=call_lambda,
        unknown_op=unknown_op,
    )


def _parse_lit(elem: ET.Element) -> SemaNode:
    """Parse a ``<lit val="...">`` leaf element."""
    val = elem.get('val', '')
    ty = _parse_type(elem)
    return SemaNode(kind=SemaNodeKind.LIT, ty=ty, lit_value=val)


def _parse_id(elem: ET.Element) -> SemaNode:
    """Parse an ``<id val="...">`` leaf element."""
    name = elem.get('val', '')
    if name == 'laneID':
        name = 'laneId'
    ty = _parse_type(elem)
    return SemaNode(kind=SemaNodeKind.ID, ty=ty, id_name=name)

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validated instruction and encoding-identifier additions for MR ISA XMLs.

The vendor MR ISA XML remains immutable. An additions document may add complete
``<Instruction>`` nodes and complete identifiers to an existing encoding's
``<EncodingIdentifiers>`` list. It cannot replace or delete existing data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import re
import xml.etree.ElementTree as elem_tree

from amdisa import xml_schema as xs
from amdisa.gpuisa import (
    IsaAdditionProvenance,
    format_instruction_name,
    opcode_constant_name,
)
from amdisa.isa_profile import IsaProfile

ADDITIONS_ROOT = 'IsaAdditions'
INSTRUCTION_ADDITIONS = 'InstructionAdditions'
IDENTIFIER_ADDITIONS = 'EncodingIdentifierAdditions'
IDENTIFIER_ADDITION = 'EncodingIdentifierAddition'
ADDITIONS_ID_ATTR = 'Id'
ADDITIONS_BASE_ARCH_ATTR = 'BaseArchitecture'
ADDITIONS_BASE_SCHEMA_ATTR = 'BaseSchemaVersion'
ADDITION_SOURCE_ATTR = '_amdisa_addition_id'

_ADDITIONS_ATTRIBUTES = {
    ADDITIONS_ID_ATTR,
    ADDITIONS_BASE_ARCH_ATTR,
    ADDITIONS_BASE_SCHEMA_ATTR,
}
_ID_PATTERN = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]*\Z')
_INSTRUCTION_NAME_PATTERN = re.compile(r'[A-Za-z][A-Za-z0-9_]*\Z')


class IsaAdditionError(ValueError):
    """An ISA additions document is malformed or conflicts with its base XML."""


@dataclass(frozen=True)
class _ImpliedLiteralOperandContract:
    """Identity and role of an implied-literal operand established by the base."""

    field_name: str
    operand_type: str
    is_input: bool
    is_output: bool
    is_implicit: bool
    is_binary_microcode_required: bool


@dataclass(frozen=True)
class _EncodingDefinition:
    """Base-owned encoding information needed to validate ISA additions."""

    name: str
    identifiers_node: elem_tree.Element
    bit_width: int
    radix: int
    flat_encoding_slice: tuple[int, int]
    opcode_slice: tuple[int, int]
    dont_care_bits: int
    opcode_bit_count: int
    opcode_modifier_bit_count: int
    base_layout_signatures: frozenset[str]
    base_encoding_values: frozenset[int]
    identifier_texts: frozenset[str]
    decoded_opcodes: frozenset[int]
    condition_ids: dict[str, frozenset[str]]
    operand_field_names: frozenset[str]
    implied_literal_operand_contracts: set[_ImpliedLiteralOperandContract]
    parent_name: str | None
    is_implied_literal: bool

    @property
    def opcode_limit(self) -> int:
        return 1 << (self.opcode_bit_count + self.opcode_modifier_bit_count)


@dataclass(frozen=True)
class _InstructionAddition:
    node: elem_tree.Element
    name: str
    path: str
    addition_id: str
    forms: tuple[tuple[str, int, str], ...]


@dataclass(frozen=True)
class _IdentifierAddition:
    node: elem_tree.Element
    encoding: _EncodingDefinition
    opcode: int


def parse_encoding_identifier_mask(
    enc_id_mask: str,
    max_enc_bits: int,
    enc_field_bit_cnt: int,
    op_field_bit_cnt: int,
) -> tuple[tuple[int, int], tuple[int, int], int]:
    """Derive primary-encoding and opcode slices from an identifier mask."""
    bit_masks = [
        (match.start(), match.end()) for match in re.finditer(r'1+', enc_id_mask)
    ]
    if not bit_masks:
        raise IsaAdditionError('encoding identifier mask contains no set bits')
    flat_enc_mask = bit_masks[0]
    if len(bit_masks) == 1:
        op_mask = (
            bit_masks[0][0] + enc_field_bit_cnt,
            bit_masks[0][0] + enc_field_bit_cnt + op_field_bit_cnt,
        )
        if (flat_enc_mask[1] - flat_enc_mask[0]) > max_enc_bits:
            flat_enc_mask = (
                flat_enc_mask[0],
                flat_enc_mask[0] + max_enc_bits,
            )
    else:
        op_mask = (
            bit_masks[1][0],
            bit_masks[1][0] + op_field_bit_cnt,
        )
    dont_care_bits = max_enc_bits - (flat_enc_mask[1] - flat_enc_mask[0])
    if dont_care_bits < 0:
        raise IsaAdditionError(
            'encoding identifier mask exceeds the primary decode width'
        )
    return flat_enc_mask, op_mask, dont_care_bits


def _required_text(parent: elem_tree.Element, tag: str, context: str) -> str:
    node = parent.find(tag)
    if node is None or node.text is None or not node.text.strip():
        raise IsaAdditionError(f'{context}: missing non-empty <{tag}>')
    return node.text.strip()


def _required_decimal(text: str, context: str) -> int:
    try:
        value = int(text)
    except ValueError as error:
        raise IsaAdditionError(f'{context}: invalid decimal value {text!r}') from error
    return value


def _implied_literal_operand_contract(
    operand: elem_tree.Element,
    enc_name: str,
    profile: IsaProfile,
    context: str,
) -> _ImpliedLiteralOperandContract:
    raw_field_name = _required_text(operand, xs.FIELD_NAME, context).lower()
    operand_type = _required_text(operand, xs.OPERAND_TYPE, context)
    return _ImpliedLiteralOperandContract(
        field_name=profile.normalize_operand_field_name(enc_name, raw_field_name),
        operand_type=profile.normalize_operand_type(
            enc_name, raw_field_name, operand_type
        ),
        is_input=operand.attrib[xs.OPERAND_ATTR_INPUT].lower() == 'true',
        is_output=operand.attrib[xs.OPERAND_ATTR_OUTPUT].lower() == 'true',
        is_implicit=operand.attrib[xs.OPERAND_ATTR_IS_IMPLICIT].lower() == 'true',
        is_binary_microcode_required=(
            operand.attrib[xs.OPERAND_ATTR_IS_BINARY_MICROCODE_REQUIRED].lower()
            == 'true'
        ),
    )


def _validate_node_shape(
    node: elem_tree.Element,
    context: str,
    *,
    allowed_attributes: frozenset[str] = frozenset(),
    required_attributes: frozenset[str] = frozenset(),
    child_counts: Mapping[str, tuple[int, int | None]] | None = None,
) -> None:
    unknown_attributes = set(node.attrib) - allowed_attributes
    if unknown_attributes:
        names = ', '.join(sorted(unknown_attributes))
        raise IsaAdditionError(f'{context}: unknown attributes: {names}')
    missing_attributes = required_attributes - set(node.attrib)
    if missing_attributes:
        names = ', '.join(sorted(missing_attributes))
        raise IsaAdditionError(f'{context}: missing required attributes: {names}')

    allowed_children = child_counts or {}
    unknown_children = [
        child.tag for child in node if child.tag not in allowed_children
    ]
    if unknown_children:
        names = ', '.join(f'<{name}>' for name in unknown_children)
        raise IsaAdditionError(f'{context}: unknown elements: {names}')
    for tag, (minimum, maximum) in allowed_children.items():
        count = sum(child.tag == tag for child in node)
        if count < minimum or (maximum is not None and count > maximum):
            if minimum == maximum:
                expected = f'exactly {minimum}'
            elif maximum is None:
                expected = f'at least {minimum}'
            else:
                expected = f'between {minimum} and {maximum}'
            raise IsaAdditionError(
                f'{context}: expected {expected} <{tag}> element(s), found {count}'
            )


def _validate_text_leaf(
    node: elem_tree.Element,
    context: str,
    *,
    allowed_attributes: frozenset[str] = frozenset(),
    required_attributes: frozenset[str] = frozenset(),
) -> str:
    _validate_node_shape(
        node,
        context,
        allowed_attributes=allowed_attributes,
        required_attributes=required_attributes,
    )
    if node.text is None or not node.text.strip():
        raise IsaAdditionError(f'{context}: value must not be empty')
    return node.text.strip()


def _validate_boolean_text(node: elem_tree.Element, context: str) -> None:
    value = _validate_text_leaf(node, context).lower()
    if value not in {'true', 'false'}:
        raise IsaAdditionError(
            f'{context}: expected boolean true or false, found {value!r}'
        )


def _validate_instruction_structure(
    instruction: elem_tree.Element, context: str
) -> None:
    """Validate one complete Instruction subtree accepted by Parser.parse_insts."""
    _validate_node_shape(
        instruction,
        context,
        child_counts={
            xs.INST_FLAGS: (1, 1),
            xs.INST_NAME: (1, 1),
            'AliasedInstructionNames': (0, 1),
            'Description': (1, 1),
            xs.INST_ENCODINGS: (1, 1),
            'FunctionalGroup': (1, 1),
        },
    )

    flags = instruction.find(xs.INST_FLAGS)
    assert flags is not None
    flag_names = (
        'IsBranch',
        'IsConditionalBranch',
        'IsIndirectBranch',
        'IsProgramTerminator',
        'IsImmediatelyExecuted',
    )
    _validate_node_shape(
        flags,
        f'{context}/<{xs.INST_FLAGS}>',
        child_counts={name: (1, 1) for name in flag_names},
    )
    for name in flag_names:
        node = flags.find(name)
        assert node is not None
        _validate_boolean_text(node, f'{context}/<{xs.INST_FLAGS}>/<{name}>')

    name_node = instruction.find(xs.INST_NAME)
    description = instruction.find('Description')
    assert name_node is not None and description is not None
    _validate_text_leaf(name_node, f'{context}/<{xs.INST_NAME}>')
    _validate_text_leaf(description, f'{context}/<Description>')

    aliases = instruction.find('AliasedInstructionNames')
    if aliases is not None:
        _validate_node_shape(
            aliases,
            f'{context}/<AliasedInstructionNames>',
            child_counts={xs.INST_NAME: (1, None)},
        )
        for alias in aliases:
            _validate_text_leaf(
                alias, f'{context}/<AliasedInstructionNames>/<{xs.INST_NAME}>'
            )

    encodings = instruction.find(xs.INST_ENCODINGS)
    assert encodings is not None
    _validate_node_shape(
        encodings,
        f'{context}/<{xs.INST_ENCODINGS}>',
        child_counts={xs.INST_ENCODING: (1, None)},
    )
    for encoding in encodings:
        encoding_context = f'{context}/<{xs.INST_ENCODING}>'
        _validate_node_shape(
            encoding,
            encoding_context,
            child_counts={
                xs.ENCODING_NAME: (1, 1),
                xs.ENCODING_COND: (1, 1),
                xs.OPCODE: (1, 1),
                xs.OPERANDS: (1, 1),
            },
        )
        encoding_name = encoding.find(xs.ENCODING_NAME)
        condition = encoding.find(xs.ENCODING_COND)
        opcode = encoding.find(xs.OPCODE)
        operands = encoding.find(xs.OPERANDS)
        assert (
            encoding_name is not None
            and condition is not None
            and opcode is not None
            and operands is not None
        )
        _validate_text_leaf(encoding_name, f'{encoding_context}/<{xs.ENCODING_NAME}>')
        _validate_text_leaf(
            condition,
            f'{encoding_context}/<{xs.ENCODING_COND}>',
            allowed_attributes=frozenset({'Id'}),
        )
        _validate_text_leaf(
            opcode,
            f'{encoding_context}/<{xs.OPCODE}>',
            allowed_attributes=frozenset({xs.ENC_IDENTIFER_ATTR_RADIX}),
            required_attributes=frozenset({xs.ENC_IDENTIFER_ATTR_RADIX}),
        )
        if opcode.attrib[xs.ENC_IDENTIFER_ATTR_RADIX] != '10':
            raise IsaAdditionError(
                f'{encoding_context}/<{xs.OPCODE}>: Radix must be 10'
            )
        _validate_node_shape(
            operands,
            f'{encoding_context}/<{xs.OPERANDS}>',
            child_counts={xs.OPERAND: (0, None)},
        )
        for operand in operands:
            operand_context = f'{encoding_context}/<{xs.OPERANDS}>/<{xs.OPERAND}>'
            operand_attributes = frozenset(
                {
                    xs.OPERAND_ATTR_INPUT,
                    xs.OPERAND_ATTR_OUTPUT,
                    xs.OPERAND_ATTR_IS_IMPLICIT,
                    xs.OPERAND_ATTR_IS_BINARY_MICROCODE_REQUIRED,
                    xs.OPERAND_ATTR_ORDER,
                }
            )
            _validate_node_shape(
                operand,
                operand_context,
                allowed_attributes=operand_attributes,
                required_attributes=operand_attributes,
                child_counts={
                    xs.FIELD_NAME: (0, 1),
                    xs.DATA_FORMAT_NAME: (1, 1),
                    xs.OPERAND_TYPE: (1, 1),
                    xs.OPERAND_SIZE: (1, 1),
                },
            )
            for attribute in (
                xs.OPERAND_ATTR_INPUT,
                xs.OPERAND_ATTR_OUTPUT,
                xs.OPERAND_ATTR_IS_IMPLICIT,
                xs.OPERAND_ATTR_IS_BINARY_MICROCODE_REQUIRED,
            ):
                value = operand.attrib[attribute].lower()
                if value not in {'true', 'false'}:
                    raise IsaAdditionError(
                        f'{operand_context}: attribute {attribute} must be '
                        f'true or false, found {value!r}'
                    )
            order = _required_decimal(
                operand.attrib[xs.OPERAND_ATTR_ORDER],
                f'{operand_context} Order',
            )
            if order < 1:
                raise IsaAdditionError(f'{operand_context}: Order must be positive')
            for tag in (xs.FIELD_NAME, xs.DATA_FORMAT_NAME, xs.OPERAND_TYPE):
                node = operand.find(tag)
                if node is not None:
                    _validate_text_leaf(node, f'{operand_context}/<{tag}>')
            size_node = operand.find(xs.OPERAND_SIZE)
            assert size_node is not None
            size = _required_decimal(
                _validate_text_leaf(
                    size_node, f'{operand_context}/<{xs.OPERAND_SIZE}>'
                ),
                f'{operand_context}/<{xs.OPERAND_SIZE}>',
            )
            if size < 1:
                raise IsaAdditionError(
                    f'{operand_context}/<{xs.OPERAND_SIZE}>: value must be positive'
                )

    functional_group = instruction.find('FunctionalGroup')
    assert functional_group is not None
    functional_context = f'{context}/<FunctionalGroup>'
    _validate_node_shape(
        functional_group,
        functional_context,
        child_counts={'Name': (1, 1), 'FunctionalSubgroups': (1, 1)},
    )
    functional_name = functional_group.find('Name')
    subgroups = functional_group.find('FunctionalSubgroups')
    assert functional_name is not None and subgroups is not None
    _validate_text_leaf(functional_name, f'{functional_context}/<Name>')
    _validate_node_shape(
        subgroups,
        f'{functional_context}/<FunctionalSubgroups>',
        child_counts={'Subgroup': (1, None)},
    )
    for subgroup in subgroups:
        _validate_text_leaf(
            subgroup, f'{functional_context}/<FunctionalSubgroups>/<Subgroup>'
        )


def _binary_text(node: elem_tree.Element, bit_width: int, context: str) -> str:
    unknown_attributes = set(node.attrib) - {xs.ENC_IDENTIFER_ATTR_RADIX}
    if unknown_attributes:
        names = ', '.join(sorted(unknown_attributes))
        raise IsaAdditionError(f'{context}: unknown attributes: {names}')
    radix_text = node.attrib.get(xs.ENC_IDENTIFER_ATTR_RADIX)
    if radix_text is None:
        raise IsaAdditionError(f'{context}: missing Radix attribute')
    radix = _required_decimal(radix_text, f'{context} Radix')
    if radix != 2:
        raise IsaAdditionError(
            f'{context}: radix {radix} does not match binary MR ISA layout'
        )
    if node.text is None or not node.text.strip():
        raise IsaAdditionError(f'{context}: missing identifier value')
    value = node.text.strip()
    if not re.fullmatch(r'[01]+', value):
        raise IsaAdditionError(f'{context}: malformed radix-2 identifier {value!r}')
    if len(value) != bit_width:
        raise IsaAdditionError(
            f'{context}: identifier width {len(value)} does not match encoding '
            f'width {bit_width}'
        )
    return value


def _field_info(
    encoding: elem_tree.Element, profile: IsaProfile, context: str
) -> tuple[int, int, int, frozenset[str]]:
    enc_name = _required_text(encoding, xs.ENCODING_NAME, context)
    renames = profile.field_renames(enc_name.upper())
    enc_field_bits: int | None = None
    op_field_bits = 0
    opm_field_bits = 0
    field_names: set[str] = set()
    for field in encoding.findall(f'./{xs.UCODE_FMT}/{xs.BITMAP}/{xs.FIELD}'):
        field_name = _required_text(field, xs.FIELD_NAME, context).lower()
        field_name = renames.get(field_name, field_name)
        field_names.add(field_name)
        ranges = sorted(
            field.findall(f'{xs.BIT_LAYOUT}/{xs.RANGE}'),
            key=lambda node: int(node.attrib.get('Order', 0)),
        )
        for range_index, range_node in enumerate(ranges):
            width = _required_decimal(
                _required_text(range_node, xs.BIT_CNT, context), context
            )
            range_name = field_name
            if range_index > 0:
                range_name = 'opm' if field_name == 'op' and range_index == 1 else ''
            if range_name == 'encoding':
                enc_field_bits = width
            elif range_name == 'op':
                op_field_bits = width
            elif range_name == 'opm':
                opm_field_bits = width
    if enc_field_bits is None:
        raise IsaAdditionError(f'{context}: microcode format has no encoding field')
    return enc_field_bits, op_field_bits, opm_field_bits, frozenset(field_names)


def _condition_ids(
    encoding: elem_tree.Element, context: str
) -> dict[str, frozenset[str]]:
    conditions: dict[str, set[str]] = {}
    for condition in encoding.findall(f'./{xs.ENCODING_CONDS}/{xs.ENCODING_COND}'):
        name = _required_text(condition, xs.COND_NAME, context)
        condition_id = condition.findtext('ConditionId')
        ids = conditions.setdefault(name, set())
        if condition_id is not None and condition_id.strip():
            ids.add(condition_id.strip())
    return {name: frozenset(ids) for name, ids in conditions.items()}


def _applicable_condition_ids(
    encoding: _EncodingDefinition,
    encodings: Mapping[str, _EncodingDefinition],
) -> dict[str, frozenset[str]]:
    """Return conditions accepted for an encoding by the parser's inheritance."""

    applicable: dict[str, set[str]] = {
        name: set(ids) for name, ids in encoding.condition_ids.items()
    }

    def include(conditions: Mapping[str, frozenset[str]]) -> None:
        for name, ids in conditions.items():
            applicable.setdefault(name, set()).update(ids)

    if encoding.parent_name is not None:
        parent = encodings.get(encoding.parent_name)
        if parent is not None:
            include(parent.condition_ids)
    else:
        for child in encodings.values():
            if child.parent_name == encoding.name:
                include(child.condition_ids)
    return {name: frozenset(ids) for name, ids in applicable.items()}


def _layout_signature(text: str, opcode_slice: tuple[int, int]) -> str:
    start, end = opcode_slice
    return f'{text[:start]}{"0" * (end - start)}{text[end:]}'


def _decode_identifier(
    text: str,
    encoding_slice: tuple[int, int],
    opcode_slice: tuple[int, int],
    dont_care_bits: int,
) -> tuple[int, int]:
    encoding_value = int(text[encoding_slice[0] : encoding_slice[1]], 2)
    encoding_value <<= dont_care_bits
    opcode_text = text[opcode_slice[0] : opcode_slice[1]]
    opcode = int(opcode_text, 2) if opcode_text else 0
    return encoding_value, opcode


def _base_metadata(
    root: elem_tree.Element,
) -> tuple[str, str, elem_tree.Element, elem_tree.Element]:
    try:
        isa_node = xs.get_node(root, xs.ISA)
        arch_node = xs.get_node(isa_node, xs.ARCH)
        arch_name = xs.get_node_text(xs.get_node(arch_node, xs.ARCH_NAME)).strip()
        document = xs.get_node(root, xs.DOCUMENT)
        schema_version = xs.get_node_text(
            xs.get_node(document, xs.SCHEMA_VERSION)
        ).strip()
        instructions = xs.get_node(isa_node, xs.INSTS)
        encodings = xs.get_node(isa_node, xs.ENCODINGS)
    except (xs.SchemaValueError, AttributeError) as error:
        raise IsaAdditionError(
            f'base MR ISA XML is missing required metadata: {error}'
        ) from error
    return arch_name, schema_version, encodings, instructions


def _base_definitions(root: elem_tree.Element, profile: IsaProfile) -> tuple[
    dict[str, _EncodingDefinition],
    set[str],
    dict[tuple[str, int], str],
    set[str],
    set[str],
    dict[str, str],
    dict[str, str],
]:
    isa_node = xs.get_node(root, xs.ISA)
    encodings_node = xs.get_node(isa_node, xs.ENCODINGS)
    instructions_node = xs.get_node(isa_node, xs.INSTS)
    operand_types_node = xs.get_node(isa_node, xs.OPERAND_TYPES)

    encoding_names = {
        _required_text(encoding, xs.ENCODING_NAME, 'base encoding')
        for encoding in encodings_node
    }
    encoding_definitions: dict[str, _EncodingDefinition] = {}
    for encoding in encodings_node:
        context = 'base encoding'
        name = _required_text(encoding, xs.ENCODING_NAME, context)
        if name in profile.skip_encodings:
            continue
        if name in encoding_definitions:
            raise IsaAdditionError(f'base encoding {name!r} is duplicated')
        bit_width = _required_decimal(
            _required_text(encoding, xs.BIT_CNT, f'base encoding {name!r}'),
            f'base encoding {name!r} bit width',
        )
        mask_node = encoding.find(xs.ENCODING_IDENTIFIER_MASK)
        if mask_node is None:
            raise IsaAdditionError(
                f'base encoding {name!r}: missing <{xs.ENCODING_IDENTIFIER_MASK}>'
            )
        mask = _binary_text(mask_node, bit_width, f'base encoding {name!r} mask')
        identifiers_node = encoding.find(xs.ENCODING_IDENTIFERS)
        if identifiers_node is None:
            raise IsaAdditionError(
                f'base encoding {name!r}: missing <{xs.ENCODING_IDENTIFERS}>'
            )
        enc_bits, op_bits, opm_bits, own_field_names = _field_info(
            encoding, profile, f'base encoding {name!r}'
        )
        condition_ids = _condition_ids(encoding, f'base encoding {name!r}')
        try:
            encoding_slice, opcode_slice, dont_care_bits = (
                parse_encoding_identifier_mask(
                    mask, profile.max_enc_bits, enc_bits, op_bits
                )
            )
        except IsaAdditionError as error:
            raise IsaAdditionError(f'base encoding {name!r}: {error}') from error
        if encoding_slice[1] > bit_width or opcode_slice[1] > bit_width:
            raise IsaAdditionError(
                f'base encoding {name!r}: identifier mask slices exceed bit width '
                f'{bit_width}'
            )

        texts: set[str] = set()
        layout_signatures: set[str] = set()
        encoding_values: set[int] = set()
        decoded_opcodes: set[int] = set()
        for identifier in identifiers_node:
            text = _binary_text(
                identifier, bit_width, f'base encoding {name!r} identifier'
            )
            encoding_value, opcode = _decode_identifier(
                text, encoding_slice, opcode_slice, dont_care_bits
            )
            texts.add(text)
            layout_signatures.add(_layout_signature(text, opcode_slice))
            encoding_values.add(encoding_value)
            decoded_opcodes.add(opcode)

        parent_name: str | None = None
        is_implied_literal = False
        operand_field_names = own_field_names
        if profile.is_alt_encoding(name):
            parent_name = profile.derive_parent_enc_name(name)
            parent = encoding_definitions.get(parent_name)
            if parent is None:
                raise IsaAdditionError(
                    f'base encoding {name!r}: parent {parent_name!r} must appear first'
                )
            condition_names = [(condition_name, '') for condition_name in condition_ids]
            is_implied_literal = profile.is_implied_literal_encoding(
                name, condition_names, bit_width, parent.bit_width
            )
            operand_field_names = parent.operand_field_names
            if not is_implied_literal:
                operand_field_names |= own_field_names

        encoding_definitions[name] = _EncodingDefinition(
            name=name,
            identifiers_node=identifiers_node,
            bit_width=bit_width,
            radix=2,
            flat_encoding_slice=encoding_slice,
            opcode_slice=opcode_slice,
            dont_care_bits=dont_care_bits,
            opcode_bit_count=op_bits,
            opcode_modifier_bit_count=opm_bits,
            base_layout_signatures=frozenset(layout_signatures),
            base_encoding_values=frozenset(encoding_values),
            identifier_texts=frozenset(texts),
            decoded_opcodes=frozenset(decoded_opcodes),
            condition_ids=condition_ids,
            operand_field_names=operand_field_names,
            implied_literal_operand_contracts=set(),
            parent_name=parent_name,
            is_implied_literal=is_implied_literal,
        )
    operand_types = {
        _required_text(node, xs.OPERAND_TYPE_NAME, 'base operand type')
        for node in operand_types_node
    }
    instruction_names: set[str] = set()
    opcode_owners: dict[tuple[str, int], str] = {}
    instruction_symbol_owners: dict[str, str] = {}
    opcode_symbol_owners: dict[str, str] = {}
    for instruction in instructions_node:
        name = _required_text(instruction, xs.INST_NAME, 'base instruction')
        instruction_names.add(name)
        encodings = instruction.find(xs.INST_ENCODINGS)
        if encodings is None:
            continue
        for encoding in encodings:
            enc_name = _required_text(
                encoding, xs.ENCODING_NAME, f'base instruction {name}'
            )
            opcode_text = _required_text(
                encoding, xs.OPCODE, f'base instruction {name}/{enc_name}'
            )
            try:
                opcode = int(opcode_text)
            except ValueError as error:
                raise IsaAdditionError(
                    f'base instruction {name}/{enc_name}: invalid decimal opcode '
                    f'{opcode_text!r}'
                ) from error
            opcode_owners.setdefault((enc_name, opcode), name)
            encoding_definition = encoding_definitions.get(enc_name)
            condition = _required_text(
                encoding,
                xs.ENCODING_COND,
                f'base instruction {name}/{enc_name}',
            )
            if encoding_definition is not None:
                # Some older vendor XMLs reference compatibility conditions from
                # instructions without declaring them under EncodingConditions.
                # Treat those base-established spellings as authoritative while
                # still rejecting novel addition-only typos.
                ids = set(encoding_definition.condition_ids.get(condition, ()))
                condition_node = encoding.find(xs.ENCODING_COND)
                assert condition_node is not None
                condition_id = condition_node.attrib.get('Id')
                if condition_id is not None:
                    ids.add(condition_id)
                encoding_definition.condition_ids[condition] = frozenset(ids)
                if encoding_definition.is_implied_literal:
                    operands = encoding.find(xs.OPERANDS)
                    if operands is not None:
                        for operand in operands:
                            if operand.find(xs.FIELD_NAME) is None:
                                continue
                            contract = _implied_literal_operand_contract(
                                operand,
                                enc_name,
                                profile,
                                f'base instruction {name}/{enc_name} operand',
                            )
                            if (
                                contract.field_name
                                not in encoding_definition.operand_field_names
                            ):
                                encoding_definition.implied_literal_operand_contracts.add(
                                    contract
                                )
            if encoding_definition is None or profile.skip_inst_encoding(
                enc_name, condition
            ):
                continue
            is_implied_literal = encoding_definition.is_implied_literal
            instruction_symbol_owners.setdefault(
                format_instruction_name(
                    name,
                    enc_name,
                    is_implied_literal_enc=is_implied_literal,
                ),
                name,
            )
            opcode_symbol_owners.setdefault(
                opcode_constant_name(
                    name,
                    enc_name,
                    is_implied_literal_enc=is_implied_literal,
                ),
                name,
            )

    for slot, owner in profile.compatibility_instruction_slots.items():
        enc_name, _ = slot
        encoding_definition = encoding_definitions.get(enc_name)
        if encoding_definition is None:
            continue
        opcode_owners.setdefault(slot, owner)
        instruction_names.add(owner)
        is_implied_literal = encoding_definition.is_implied_literal
        instruction_symbol_owners.setdefault(
            format_instruction_name(
                owner,
                enc_name,
                is_implied_literal_enc=is_implied_literal,
            ),
            owner,
        )
        opcode_symbol_owners.setdefault(
            opcode_constant_name(
                owner,
                enc_name,
                is_implied_literal_enc=is_implied_literal,
            ),
            owner,
        )
    return (
        encoding_definitions,
        encoding_names,
        opcode_owners,
        operand_types,
        instruction_names,
        instruction_symbol_owners,
        opcode_symbol_owners,
    )


def _parse_additions(path: str) -> elem_tree.Element:
    try:
        return elem_tree.parse(path).getroot()
    except (OSError, elem_tree.ParseError) as error:
        raise IsaAdditionError(
            f'{path}: cannot parse ISA additions document: {error}'
        ) from error


def _validate_additions_root(
    root: elem_tree.Element, path: str, base_arch: str, base_schema: str
) -> tuple[str, elem_tree.Element | None, elem_tree.Element]:
    if root.tag != ADDITIONS_ROOT:
        raise IsaAdditionError(
            f'{path}: expected <{ADDITIONS_ROOT}> root, found <{root.tag}>'
        )
    unknown_attributes = set(root.attrib) - _ADDITIONS_ATTRIBUTES
    missing_attributes = _ADDITIONS_ATTRIBUTES - set(root.attrib)
    if unknown_attributes:
        names = ', '.join(sorted(unknown_attributes))
        raise IsaAdditionError(f'{path}: unknown additions root attributes: {names}')
    if missing_attributes:
        names = ', '.join(sorted(missing_attributes))
        raise IsaAdditionError(f'{path}: missing additions root attributes: {names}')

    identifier = root.attrib[ADDITIONS_ID_ATTR].strip()
    if not _ID_PATTERN.fullmatch(identifier):
        raise IsaAdditionError(
            f'{path}: invalid additions document Id {identifier!r}; use letters, '
            'digits, dot, underscore, or hyphen'
        )

    declared_arch = root.attrib[ADDITIONS_BASE_ARCH_ATTR].strip()
    if declared_arch != base_arch:
        raise IsaAdditionError(
            f'{path}: additions base architecture {declared_arch!r} does not match '
            f'{base_arch!r}'
        )
    declared_schema = root.attrib[ADDITIONS_BASE_SCHEMA_ATTR].strip()
    if declared_schema != base_schema:
        raise IsaAdditionError(
            f'{path}: additions base schema {declared_schema!r} does not match '
            f'{base_schema!r}'
        )

    children = list(root)
    allowed_children = {IDENTIFIER_ADDITIONS, INSTRUCTION_ADDITIONS}
    unknown_children = [
        child.tag for child in children if child.tag not in allowed_children
    ]
    if unknown_children:
        names = ', '.join(f'<{name}>' for name in unknown_children)
        raise IsaAdditionError(f'{path}: unknown additions root elements: {names}')
    instructions = [child for child in children if child.tag == INSTRUCTION_ADDITIONS]
    identifier_additions = [
        child for child in children if child.tag == IDENTIFIER_ADDITIONS
    ]
    if len(instructions) != 1:
        raise IsaAdditionError(
            f'{path}: additions document must contain exactly one '
            f'<{INSTRUCTION_ADDITIONS}> element'
        )
    if len(identifier_additions) > 1:
        raise IsaAdditionError(
            f'{path}: additions document may contain at most one '
            f'<{IDENTIFIER_ADDITIONS}> element'
        )
    instructions_node = instructions[0]
    if instructions_node.attrib:
        names = ', '.join(sorted(instructions_node.attrib))
        raise IsaAdditionError(
            f'{path}: <{INSTRUCTION_ADDITIONS}> must not have attributes: {names}'
        )
    identifier_group = identifier_additions[0] if identifier_additions else None
    if identifier_group is not None and identifier_group.attrib:
        names = ', '.join(sorted(identifier_group.attrib))
        raise IsaAdditionError(
            f'{path}: <{IDENTIFIER_ADDITIONS}> must not have attributes: {names}'
        )
    if identifier_group is not None and not list(identifier_group):
        raise IsaAdditionError(f'{path}: <{IDENTIFIER_ADDITIONS}> must not be empty')
    return identifier, identifier_group, instructions_node


def _validate_instruction_addition(
    instruction: elem_tree.Element,
    *,
    path: str,
    addition_id: str,
    encodings: Mapping[str, _EncodingDefinition],
    encoding_names: set[str],
    opcode_owners: dict[tuple[str, int], str],
    opcode_limits: Mapping[str, int],
    operand_types: set[str],
    instruction_names: set[str],
    instruction_symbol_owners: dict[str, str],
    opcode_symbol_owners: dict[str, str],
    profile: IsaProfile,
) -> _InstructionAddition:
    if instruction.tag != xs.INST:
        raise IsaAdditionError(
            f'{path}: <{INSTRUCTION_ADDITIONS}> may contain only <{xs.INST}> '
            f'elements, found <{instruction.tag}>'
        )

    context = f'{path}: additions document {addition_id!r}'
    _validate_instruction_structure(instruction, f'{context}: <{xs.INST}>')
    name = _required_text(instruction, xs.INST_NAME, context)
    if not _INSTRUCTION_NAME_PATTERN.fullmatch(name):
        raise IsaAdditionError(
            f'{context}: invalid instruction name {name!r}; use ASCII letters, '
            'digits, and underscores, starting with a letter'
        )
    if name in instruction_names:
        raise IsaAdditionError(f'{context}: instruction {name!r} already exists')

    instruction_encodings = instruction.find(xs.INST_ENCODINGS)
    if instruction_encodings is None or not list(instruction_encodings):
        raise IsaAdditionError(
            f'{context}: instruction {name!r} has no <{xs.INST_ENCODING}> entries'
        )

    forms_in_order: list[tuple[str, int, str]] = []
    forms: set[tuple[str, int, str]] = set()
    active_instruction_forms: dict[str, tuple[str, int, str]] = {}
    for encoding in instruction_encodings:
        if encoding.tag != xs.INST_ENCODING:
            raise IsaAdditionError(
                f'{context}: <{xs.INST_ENCODINGS}> may contain only '
                f'<{xs.INST_ENCODING}> elements, found <{encoding.tag}>'
            )
        enc_name = _required_text(
            encoding, xs.ENCODING_NAME, f'{context}: instruction {name!r}'
        )
        if enc_name not in encoding_names:
            raise IsaAdditionError(
                f'{context}: instruction {name!r} references unknown encoding '
                f'{enc_name!r}'
            )
        opcode_text = _required_text(
            encoding, xs.OPCODE, f'{context}: instruction {name!r}/{enc_name}'
        )
        try:
            opcode = int(opcode_text)
        except ValueError as error:
            raise IsaAdditionError(
                f'{context}: instruction {name!r}/{enc_name} has invalid decimal '
                f'opcode {opcode_text!r}'
            ) from error
        if opcode < 0:
            raise IsaAdditionError(
                f'{context}: instruction {name!r}/{enc_name} has negative opcode '
                f'{opcode}'
            )
        opcode_limit = opcode_limits.get(enc_name)
        if opcode_limit is not None and opcode >= opcode_limit:
            raise IsaAdditionError(
                f'{context}: instruction {name!r}/{enc_name} opcode {opcode} is '
                f'outside the encoding range [0, {opcode_limit})'
            )

        slot = (enc_name, opcode)
        owner = opcode_owners.get(slot)
        if owner is not None:
            raise IsaAdditionError(
                f'{context}: opcode {opcode} in encoding {enc_name!r} is already '
                f'owned by instruction {owner!r}'
            )
        condition = _required_text(
            encoding,
            xs.ENCODING_COND,
            f'{context}: instruction {name!r}/{enc_name}',
        )
        encoding_definition = encodings.get(enc_name)
        if encoding_definition is not None:
            applicable_conditions = _applicable_condition_ids(
                encoding_definition, encodings
            )
            if condition not in applicable_conditions:
                raise IsaAdditionError(
                    f'{context}: instruction {name!r}/{enc_name} references '
                    f'unknown encoding condition {condition!r}'
                )
            condition_node = encoding.find(xs.ENCODING_COND)
            assert condition_node is not None
            condition_id = condition_node.attrib.get('Id')
            declared_ids = applicable_conditions[condition]
            if (
                condition_id is not None
                and declared_ids
                and condition_id not in declared_ids
            ):
                expected = ', '.join(sorted(declared_ids))
                raise IsaAdditionError(
                    f'{context}: instruction {name!r}/{enc_name} condition '
                    f'{condition!r} has Id {condition_id!r}, expected {expected}'
                )
        form = (enc_name, opcode, condition)
        if form in forms:
            raise IsaAdditionError(
                f'{context}: instruction {name!r} repeats opcode {opcode} in '
                f'encoding {enc_name!r} with condition {condition!r}'
            )
        forms.add(form)
        forms_in_order.append(form)

        if encoding_definition is not None and not profile.skip_inst_encoding(
            enc_name, condition
        ):
            is_implied_literal = encoding_definition.is_implied_literal
            instruction_symbol = format_instruction_name(
                name,
                enc_name,
                is_implied_literal_enc=is_implied_literal,
            )
            previous_form = active_instruction_forms.get(instruction_symbol)
            if previous_form is not None:
                previous_encoding, previous_opcode, previous_condition = previous_form
                raise IsaAdditionError(
                    f'{context}: instruction {name!r} has multiple active forms '
                    f'for generated instruction symbol {instruction_symbol!r}: '
                    f'{previous_encoding!r} opcode {previous_opcode} condition '
                    f'{previous_condition!r}, and {enc_name!r} opcode {opcode} '
                    f'condition {condition!r}'
                )
            active_instruction_forms[instruction_symbol] = form
            generated_symbols = (
                (
                    'instruction',
                    instruction_symbol,
                    instruction_symbol_owners,
                ),
                (
                    'opcode constant',
                    opcode_constant_name(
                        name,
                        enc_name,
                        is_implied_literal_enc=is_implied_literal,
                    ),
                    opcode_symbol_owners,
                ),
            )
            for symbol_kind, symbol, owners in generated_symbols:
                symbol_owner = owners.get(symbol)
                if symbol_owner is not None and symbol_owner != name:
                    raise IsaAdditionError(
                        f'{context}: instruction {name!r}/{enc_name} normalizes '
                        f'to {symbol_kind} symbol {symbol!r}, already owned by '
                        f'instruction {symbol_owner!r}'
                    )
                owners[symbol] = name

        operands = encoding.find(xs.OPERANDS)
        if operands is None:
            raise IsaAdditionError(
                f'{context}: instruction {name!r}/{enc_name} is missing '
                f'<{xs.OPERANDS}>'
            )
        for operand in operands:
            operand_context = f'{context}: instruction {name!r}/{enc_name} operand'
            op_type = _required_text(
                operand,
                xs.OPERAND_TYPE,
                operand_context,
            )
            if op_type not in operand_types:
                raise IsaAdditionError(
                    f'{context}: instruction {name!r}/{enc_name} references '
                    f'unknown operand type {op_type!r}'
                )
            field_node = operand.find(xs.FIELD_NAME)
            if field_node is None or encoding_definition is None:
                continue
            field_name = _required_text(operand, xs.FIELD_NAME, operand_context)
            contract = _implied_literal_operand_contract(
                operand, enc_name, profile, operand_context
            )
            if contract.field_name in encoding_definition.operand_field_names:
                continue
            if contract in encoding_definition.implied_literal_operand_contracts:
                continue
            established_implied_literal_fields = {
                item.field_name
                for item in encoding_definition.implied_literal_operand_contracts
            }
            if contract.field_name in established_implied_literal_fields:
                raise IsaAdditionError(
                    f'{context}: instruction {name!r}/{enc_name} references '
                    f'implied-literal operand field {field_name!r} with a type or '
                    'role not established by the base XML'
                )
            raise IsaAdditionError(
                f'{context}: instruction {name!r}/{enc_name} references '
                f'unknown operand field {field_name!r}'
            )

    addition = deepcopy(instruction)
    addition.attrib[ADDITION_SOURCE_ATTR] = addition_id
    return _InstructionAddition(
        node=addition,
        name=name,
        path=path,
        addition_id=addition_id,
        forms=tuple(forms_in_order),
    )


def _single_child(
    parent: elem_tree.Element, tag: str, context: str
) -> elem_tree.Element:
    nodes = [child for child in parent if child.tag == tag]
    if len(nodes) != 1:
        raise IsaAdditionError(f'{context}: expected exactly one <{tag}> element')
    return nodes[0]


def _validate_identifier_addition(
    addition: elem_tree.Element,
    *,
    path: str,
    addition_id: str,
    encodings: Mapping[str, _EncodingDefinition],
    all_encoding_names: set[str],
    occupied_texts: dict[str, set[str]],
    occupied_opcodes: dict[str, set[int]],
) -> _IdentifierAddition:
    context = f'{path}: additions document {addition_id!r} identifier addition'
    if addition.tag != IDENTIFIER_ADDITION:
        raise IsaAdditionError(
            f'{path}: <{IDENTIFIER_ADDITIONS}> may contain only '
            f'<{IDENTIFIER_ADDITION}> elements, found <{addition.tag}>'
        )
    if addition.attrib:
        names = ', '.join(sorted(addition.attrib))
        raise IsaAdditionError(f'{context}: unknown attributes: {names}')
    allowed_children = {xs.ENCODING_NAME, xs.OPCODE, xs.ENCODING_IDENTIFER_ALT}
    unknown_children = [
        child.tag for child in addition if child.tag not in allowed_children
    ]
    if unknown_children:
        names = ', '.join(f'<{name}>' for name in unknown_children)
        raise IsaAdditionError(f'{context}: unknown elements: {names}')

    encoding_name_node = _single_child(addition, xs.ENCODING_NAME, context)
    opcode_node = _single_child(addition, xs.OPCODE, context)
    identifier_node = _single_child(addition, xs.ENCODING_IDENTIFER_ALT, context)
    if encoding_name_node.attrib or opcode_node.attrib:
        raise IsaAdditionError(
            f'{context}: <{xs.ENCODING_NAME}> and <{xs.OPCODE}> must not '
            'have attributes'
        )

    encoding_name = _required_text(addition, xs.ENCODING_NAME, context)
    if encoding_name not in all_encoding_names:
        raise IsaAdditionError(f'{context}: unknown encoding {encoding_name!r}')
    encoding = encodings.get(encoding_name)
    if encoding is None:
        raise IsaAdditionError(
            f'{context}: encoding {encoding_name!r} is not active in this ISA profile'
        )
    opcode_text = _required_text(addition, xs.OPCODE, context)
    opcode = _required_decimal(opcode_text, f'{context} opcode')
    if opcode < 0 or opcode >= encoding.opcode_limit:
        raise IsaAdditionError(
            f'{context}: opcode {opcode} is outside encoding {encoding_name!r} '
            f'range [0, {encoding.opcode_limit})'
        )

    radix_text = identifier_node.attrib.get(xs.ENC_IDENTIFER_ATTR_RADIX)
    if radix_text is None:
        raise IsaAdditionError(f'{context}: <EncodingIdentifier> is missing Radix')
    radix = _required_decimal(radix_text, f'{context} identifier Radix')
    if radix != encoding.radix:
        raise IsaAdditionError(
            f'{context}: identifier radix {radix} does not match encoding '
            f'radix {encoding.radix}'
        )
    text = _binary_text(identifier_node, encoding.bit_width, context)
    encoding_value, decoded_opcode = _decode_identifier(
        text,
        encoding.flat_encoding_slice,
        encoding.opcode_slice,
        encoding.dont_care_bits,
    )
    if decoded_opcode != opcode:
        raise IsaAdditionError(
            f'{context}: identifier decodes opcode {decoded_opcode}, not declared '
            f'opcode {opcode} for encoding {encoding_name!r}'
        )
    if (
        encoding_value not in encoding.base_encoding_values
        or _layout_signature(text, encoding.opcode_slice)
        not in encoding.base_layout_signatures
    ):
        raise IsaAdditionError(
            f'{context}: identifier for encoding {encoding_name!r} is '
            'incompatible with the base identifier mask/layout'
        )
    if text in occupied_texts[encoding_name]:
        raise IsaAdditionError(
            f'{context}: duplicate identifier for encoding {encoding_name!r}'
        )
    if opcode in occupied_opcodes[encoding_name]:
        raise IsaAdditionError(
            f'{context}: identifier collides with an existing decode slot for '
            f'encoding {encoding_name!r} opcode {opcode}'
        )

    copied_identifier = deepcopy(identifier_node)
    copied_identifier.attrib[ADDITION_SOURCE_ATTR] = addition_id
    occupied_texts[encoding_name].add(text)
    occupied_opcodes[encoding_name].add(opcode)
    return _IdentifierAddition(
        node=copied_identifier,
        encoding=encoding,
        opcode=opcode,
    )


def _expanded_opcodes(encoding: _EncodingDefinition, base_opcode: int) -> set[int]:
    """Return all complete opcodes represented by a decoded identifier opcode."""
    base_count = 1 << encoding.opcode_bit_count
    return {
        base_opcode + modifier * base_count
        for modifier in range(1 << encoding.opcode_modifier_bit_count)
    }


def _identifier_owners(
    addition: _IdentifierAddition,
    opcode_owners: Mapping[tuple[str, int], str],
) -> set[str]:
    return {
        owner
        for opcode in _expanded_opcodes(addition.encoding, addition.opcode)
        if (owner := opcode_owners.get((addition.encoding.name, opcode))) is not None
    }


def _reachable_opcodes(
    encodings: Mapping[str, _EncodingDefinition],
    identifiers: Sequence[_IdentifierAddition],
) -> dict[str, set[int]]:
    base_opcodes = {
        name: set(encoding.decoded_opcodes) for name, encoding in encodings.items()
    }
    for addition in identifiers:
        base_opcodes[addition.encoding.name].add(addition.opcode)
    return {
        name: {
            expanded
            for opcode in base_opcodes[name]
            for expanded in _expanded_opcodes(encoding, opcode)
        }
        for name, encoding in encodings.items()
    }


def apply_isa_additions(
    base_root: elem_tree.Element,
    addition_paths: Sequence[str],
    profile: IsaProfile,
) -> tuple[IsaAdditionProvenance, ...]:
    """Validate and atomically merge ISA additions into a base MR ISA tree.

    Validation of every input completes before the base tree is modified. The
    returned provenance and merge order follow ``addition_paths`` and document
    order. Identifier additions are merged before the caller constructs normal
    decode tables.
    """
    if not addition_paths:
        return ()

    base_arch, base_schema, _, base_instructions = _base_metadata(base_root)
    (
        encodings,
        encoding_names,
        opcode_owners,
        operand_types,
        instruction_names,
        instruction_symbol_owners,
        opcode_symbol_owners,
    ) = _base_definitions(base_root, profile)
    opcode_limits = {
        name: encoding.opcode_limit for name, encoding in encodings.items()
    }
    pending_instructions: list[_InstructionAddition] = []
    pending_identifier_groups: list[tuple[str, str, elem_tree.Element]] = []
    pending_identifiers: list[_IdentifierAddition] = []
    addition_opcode_owners: dict[tuple[str, int], str] = {}
    provenance: list[IsaAdditionProvenance] = []
    addition_ids: set[str] = set()

    for raw_path in addition_paths:
        path = str(raw_path)
        addition_root = _parse_additions(path)
        addition_id, identifier_additions, instruction_additions = (
            _validate_additions_root(addition_root, path, base_arch, base_schema)
        )
        if addition_id in addition_ids:
            raise IsaAdditionError(
                f'{path}: duplicate additions document Id {addition_id!r}'
            )

        addition_ids.add(addition_id)
        provenance.append(IsaAdditionProvenance(addition_id, path))
        if identifier_additions is not None:
            pending_identifier_groups.append((path, addition_id, identifier_additions))
        for instruction in instruction_additions:
            addition = _validate_instruction_addition(
                instruction,
                path=path,
                addition_id=addition_id,
                encodings=encodings,
                encoding_names=encoding_names,
                opcode_owners=opcode_owners,
                opcode_limits=opcode_limits,
                operand_types=operand_types,
                instruction_names=instruction_names,
                instruction_symbol_owners=instruction_symbol_owners,
                opcode_symbol_owners=opcode_symbol_owners,
                profile=profile,
            )
            instruction_names.add(addition.name)
            for enc_name, opcode, _ in addition.forms:
                slot = (enc_name, opcode)
                opcode_owners[slot] = addition.name
                addition_opcode_owners[slot] = addition.name
            pending_instructions.append(addition)

    occupied_texts = {
        name: set(encoding.identifier_texts) for name, encoding in encodings.items()
    }
    occupied_opcodes = {
        name: set(encoding.decoded_opcodes) for name, encoding in encodings.items()
    }
    for path, addition_id, identifier_group in pending_identifier_groups:
        for identifier in identifier_group:
            addition = _validate_identifier_addition(
                identifier,
                path=path,
                addition_id=addition_id,
                encodings=encodings,
                all_encoding_names=encoding_names,
                occupied_texts=occupied_texts,
                occupied_opcodes=occupied_opcodes,
            )
            owners = _identifier_owners(addition, addition_opcode_owners)
            if not owners:
                raise IsaAdditionError(
                    f'{path}: additions document {addition_id!r} identifier for encoding '
                    f'{addition.encoding.name!r} opcode {addition.opcode} is '
                    'unowned; it must correspond to an added instruction'
                )
            if addition.encoding.is_implied_literal:
                parent_name = addition.encoding.parent_name
                assert parent_name is not None
                parent_encoding = encodings[parent_name]
                parent_owners = {
                    owner
                    for opcode in _expanded_opcodes(parent_encoding, addition.opcode)
                    if (owner := addition_opcode_owners.get((parent_name, opcode)))
                    is not None
                }
                if parent_owners != owners:
                    owner_text = ', '.join(sorted(owners))
                    parent_owner_text = (
                        ', '.join(sorted(parent_owners))
                        if parent_owners
                        else 'no owner'
                    )
                    raise IsaAdditionError(
                        f'{path}: additions document {addition_id!r} '
                        'implied-literal identifier '
                        f'for {addition.encoding.name!r} opcode {addition.opcode} '
                        f'is owned by {owner_text!r}, but parent encoding '
                        f'{parent_name!r} has {parent_owner_text}'
                    )
            pending_identifiers.append(addition)

    reachable = _reachable_opcodes(encodings, pending_identifiers)
    for addition in pending_instructions:
        active_forms = [
            form
            for form in addition.forms
            if form[0] not in profile.skip_encodings
            and not profile.skip_inst_encoding(form[0], form[2])
        ]
        if not active_forms:
            raise IsaAdditionError(
                f'{addition.path}: additions document {addition.addition_id!r} '
                f'instruction {addition.name!r} has no encoding form active in '
                'the selected ISA profile'
            )
        for enc_name, opcode, _condition in active_forms:
            if opcode not in reachable[enc_name]:
                raise IsaAdditionError(
                    f'{addition.path}: additions document '
                    f'{addition.addition_id!r} instruction '
                    f'{addition.name!r} encoding {enc_name!r} opcode {opcode} is '
                    'unreachable; add a validated encoding identifier'
                )

    for addition in pending_identifiers:
        addition.encoding.identifiers_node.append(addition.node)
    base_instructions.extend(addition.node for addition in pending_instructions)
    return tuple(provenance)

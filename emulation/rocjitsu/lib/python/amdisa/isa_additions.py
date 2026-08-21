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
from amdisa.gpuisa import IsaAdditionProvenance
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


class IsaAdditionError(ValueError):
    """An ISA additions document is malformed or conflicts with its base XML."""


@dataclass(frozen=True)
class _EncodingDefinition:
    """Base-owned encoding information needed to validate identifier additions."""

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


def _field_widths(
    encoding: elem_tree.Element, profile: IsaProfile, context: str
) -> tuple[int, int, int]:
    enc_name = _required_text(encoding, xs.ENCODING_NAME, context)
    renames = profile.field_renames(enc_name.upper())
    enc_field_bits: int | None = None
    op_field_bits = 0
    opm_field_bits = 0
    for field in encoding.findall(f'./{xs.UCODE_FMT}/{xs.BITMAP}/{xs.FIELD}'):
        field_name = _required_text(field, xs.FIELD_NAME, context).lower()
        field_name = renames.get(field_name, field_name)
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
    return enc_field_bits, op_field_bits, opm_field_bits


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
        enc_bits, op_bits, opm_bits = _field_widths(
            encoding, profile, f'base encoding {name!r}'
        )
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
        if profile.is_alt_encoding(name):
            parent_name = profile.derive_parent_enc_name(name)
            parent = encoding_definitions.get(parent_name)
            if parent is None:
                raise IsaAdditionError(
                    f'base encoding {name!r}: parent {parent_name!r} must appear first'
                )
            condition_names = [
                (_required_text(condition, xs.COND_NAME, f'base encoding {name!r}'), '')
                for condition in encoding.findall(
                    f'./{xs.ENCODING_CONDS}/{xs.ENCODING_COND}'
                )
            ]
            is_implied_literal = profile.is_implied_literal_encoding(
                name, condition_names, bit_width, parent.bit_width
            )

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
            parent_name=parent_name,
            is_implied_literal=is_implied_literal,
        )
    operand_types = {
        _required_text(node, xs.OPERAND_TYPE_NAME, 'base operand type')
        for node in operand_types_node
    }
    instruction_names: set[str] = set()
    opcode_owners: dict[tuple[str, int], str] = {}
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
    return (
        encoding_definitions,
        encoding_names,
        opcode_owners,
        operand_types,
        instruction_names,
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
    encoding_names: set[str],
    opcode_owners: dict[tuple[str, int], str],
    opcode_limits: Mapping[str, int],
    operand_types: set[str],
    instruction_names: set[str],
) -> _InstructionAddition:
    if instruction.tag != xs.INST:
        raise IsaAdditionError(
            f'{path}: <{INSTRUCTION_ADDITIONS}> may contain only <{xs.INST}> '
            f'elements, found <{instruction.tag}>'
        )

    context = f'{path}: additions document {addition_id!r}'
    name = _required_text(instruction, xs.INST_NAME, context)
    if name in instruction_names:
        raise IsaAdditionError(f'{context}: instruction {name!r} already exists')

    instruction_encodings = instruction.find(xs.INST_ENCODINGS)
    if instruction_encodings is None or not list(instruction_encodings):
        raise IsaAdditionError(
            f'{context}: instruction {name!r} has no <{xs.INST_ENCODING}> entries'
        )

    forms_in_order: list[tuple[str, int, str]] = []
    forms: set[tuple[str, int, str]] = set()
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
        form = (enc_name, opcode, condition)
        if form in forms:
            raise IsaAdditionError(
                f'{context}: instruction {name!r} repeats opcode {opcode} in '
                f'encoding {enc_name!r} with condition {condition!r}'
            )
        forms.add(form)
        forms_in_order.append(form)

        operands = encoding.find(xs.OPERANDS)
        if operands is None:
            raise IsaAdditionError(
                f'{context}: instruction {name!r}/{enc_name} is missing '
                f'<{xs.OPERANDS}>'
            )
        for operand in operands:
            op_type = _required_text(
                operand,
                xs.OPERAND_TYPE,
                f'{context}: instruction {name!r}/{enc_name} operand',
            )
            if op_type not in operand_types:
                raise IsaAdditionError(
                    f'{context}: instruction {name!r}/{enc_name} references '
                    f'unknown operand type {op_type!r}'
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


def _reachable_opcodes(
    encodings: Mapping[str, _EncodingDefinition],
    identifiers: Sequence[_IdentifierAddition],
) -> dict[str, set[int]]:
    reachable = {
        name: set(encoding.decoded_opcodes) for name, encoding in encodings.items()
    }
    for addition in identifiers:
        reachable[addition.encoding.name].add(addition.opcode)
    for name, encoding in encodings.items():
        if encoding.opcode_modifier_bit_count == 0:
            continue
        base_count = 1 << encoding.opcode_bit_count
        base_opcodes = tuple(reachable[name])
        for modifier in range(1, 1 << encoding.opcode_modifier_bit_count):
            reachable[name].update(
                opcode + modifier * base_count for opcode in base_opcodes
            )
    return reachable


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
                encoding_names=encoding_names,
                opcode_owners=opcode_owners,
                opcode_limits=opcode_limits,
                operand_types=operand_types,
                instruction_names=instruction_names,
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
            owner = addition_opcode_owners.get(
                (addition.encoding.name, addition.opcode)
            )
            if owner is None:
                raise IsaAdditionError(
                    f'{path}: additions document {addition_id!r} identifier for encoding '
                    f'{addition.encoding.name!r} opcode {addition.opcode} is '
                    'unowned; it must correspond to an added instruction'
                )
            if addition.encoding.is_implied_literal:
                parent_name = addition.encoding.parent_name
                parent_owner = addition_opcode_owners.get(
                    (parent_name, addition.opcode)
                )
                if parent_owner != owner:
                    parent_owner_text = (
                        repr(parent_owner) if parent_owner is not None else 'no owner'
                    )
                    raise IsaAdditionError(
                        f'{path}: additions document {addition_id!r} '
                        'implied-literal identifier '
                        f'for {addition.encoding.name!r} opcode {addition.opcode} '
                        f'is owned by {owner!r}, but parent encoding '
                        f'{parent_name!r} has {parent_owner_text}'
                    )
            pending_identifiers.append(addition)

    reachable = _reachable_opcodes(encodings, pending_identifiers)
    for addition in pending_instructions:
        for enc_name, opcode, condition in addition.forms:
            if enc_name in profile.skip_encodings or profile.skip_inst_encoding(
                enc_name, condition
            ):
                continue
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

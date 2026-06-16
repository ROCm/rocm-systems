# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""XML element and attribute names for the AMD machine-readable ISA spec.

Constants, error types, and accessor helpers for the XML schema used across
supported spec versions. They are collected here so that the parser
and data model do not encode schema knowledge in multiple places.
"""

import xml.etree.ElementTree as elem_tree
from typing import NamedTuple


class SchemaVersion(NamedTuple):
    """Parsed AMD ISA XML schema version triple (major.minor.patch).

    Use ``SchemaVersion.parse('1.1.1')`` to construct from a version string.
    All three known versions (1.0.0, 1.1.0, 1.1.1) are represented this way.
    """

    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, s: str) -> 'SchemaVersion':
        """Parse a dot-separated version string (e.g. '1.2.0') into a SchemaVersion."""
        parts = s.strip().split('.')
        if len(parts) != 3:
            raise ValueError(f"Expected 'major.minor.patch' version string, got {s!r}")
        try:
            return cls(int(parts[0]), int(parts[1]), int(parts[2]))
        except ValueError as e:
            raise ValueError(f"Non-integer component in version string {s!r}") from e

    def __str__(self) -> str:
        return f'{self.major}.{self.minor}.{self.patch}'


# Maps logical field names to the actual XML element name per schema version.
# When an element name has a known typo in some versions, both the canonical
# name and the typo are listed. ``get_node_with_fallback`` uses the primary
# constant (COND_EXPR) as the first try and COND_EXPR_ALT as the fallback.
#
# All three known versions (1.0.0, 1.1.0, 1.1.1) use the typo'd names below;
# the corrected names are included as comments for a future schema version.
_VERSIONED_FIELDS: dict[str, dict[SchemaVersion, str]] = {
    'COND_EXPR': {
        SchemaVersion(1, 0, 0): 'CondtionExpression',  # typo: missing 'd'
        SchemaVersion(1, 1, 0): 'CondtionExpression',
        SchemaVersion(1, 1, 1): 'CondtionExpression',
        # SchemaVersion(2, 0, 0): 'ConditionExpression',  # if/when fixed
    },
    'ENCODING_IDENTIFER': {
        SchemaVersion(1, 0, 0): 'EncodingIdentifer',  # typo: missing 'i'
        SchemaVersion(1, 1, 0): 'EncodingIdentifer',
        SchemaVersion(1, 1, 1): 'EncodingIdentifer',
        # SchemaVersion(2, 0, 0): 'EncodingIdentifier',  # if/when fixed
    },
}

# Top-level structure.
ARCH = 'Architecture'
ARCH_ID = 'ArchitectureId'
ARCH_NAME = 'ArchitectureName'
DOCUMENT = 'Document'
ISA = 'ISA'
SCHEMA_VERSION = 'SchemaVersion'

# Encoding elements.
BIT_CNT = 'BitCount'
BIT_LAYOUT = 'BitLayout'
BITMAP = 'BitMap'
BIT_OFF = 'BitOffset'
# NOTE: Versions 1.0.0, 1.1.0, and 1.1.1 have a typo here: Condtion. If a future
# version fixes the spelling, add the corrected name as a fallback.
COND_EXPR = 'CondtionExpression'
COND_EXPR_ALT = 'ConditionExpression'
COND_NAME = 'ConditionName'
ENCODING = 'Encoding'
ENCODING_COND = 'EncodingCondition'
ENCODING_CONDS = 'EncodingConditions'
# NOTE: Same typo as COND_EXPR above: Identifer (missing 'i'). If a future version
# fixes the spelling, add the corrected name as a fallback.
ENCODING_IDENTIFER = 'EncodingIdentifer'
ENCODING_IDENTIFER_ALT = 'EncodingIdentifier'
ENCODING_IDENTIFIER_MASK = 'EncodingIdentifierMask'
# NOTE: The plural form uses the correct spelling (EncodingIdentifiers with
# 'i') while the singular ENCODING_IDENTIFER above has the typo. Both must
# match the XML exactly.
ENCODING_IDENTIFERS = 'EncodingIdentifiers'
ENCODING_NAME = 'EncodingName'
ENCODINGS = 'Encodings'

# Expression AST elements.
EXPR = 'Expression'
LABEL = 'Label'
OPERATOR = 'Operator'
SUB_EXPR = 'Subexpressions'
VALUE = 'Value'
VALUE_TYPE = 'ValueType'
SIZE = 'Size'

# Field elements.
FIELD = 'Field'
FIELD_NAME = 'FieldName'
UCODE_FMT = 'MicrocodeFormat'
RANGE = 'Range'

# Instruction elements.
INST = 'Instruction'
INST_ENCODING = 'InstructionEncoding'
INST_ENCODINGS = 'InstructionEncodings'
INST_FLAGS = 'InstructionFlags'
INST_NAME = 'InstructionName'
INSTS = 'Instructions'
OPCODE = 'Opcode'

# Operand elements.
NAME = 'Name'
OPERAND = 'Operand'
OPERAND_PREDEFINED_VALS = 'OperandPredefinedValues'
OPERAND_SIZE = 'OperandSize'
OPERAND_TYPE = 'OperandType'
OPERAND_TYPE_NAME = 'OperandTypeName'
OPERAND_TYPES = 'OperandTypes'
OPERANDS = 'Operands'
PREDEFINED_VAL = 'PredefinedValue'

# Operand attributes.
OPERAND_ATTR_INPUT = 'Input'
OPERAND_ATTR_OUTPUT = 'Output'
OPERAND_ATTR_IS_IMPLICIT = 'IsImplicit'
OPERAND_ATTR_IS_BINARY_MICROCODE_REQUIRED = 'IsBinaryMicrocodeRequired'
OPERAND_ATTR_ORDER = 'Order'

# Encoding attributes.
ENC_ATTR_ORDER = 'Order'
ENC_IDENTIFER_ATTR_RADIX = 'Radix'

# Expression attributes and their possible values.
EXPR_ATTR_TYPE = 'Type'
EXPR_TYPE_VAL_ID = 'Id'
EXPR_TYPE_VAL_LITERAL = 'Literal'
EXPR_TYPE_VAL_OPERATOR = 'Operator'
EXPR_TYPE_VAL_RETURN = 'ReturnType'


class SchemaValueError(ValueError):
    """Exception indicating a value is missing from the XML schema."""

    def __init__(self, message: str) -> None:
        super().__init__(message)

    def __str__(self) -> str:
        return f'Value not defined in the XML schema: {super().__str__()}'


def get_node(parent: elem_tree.Element, tag: str) -> elem_tree.Element:
    """Find a child element by tag.

    Raises:
        SchemaValueError: If the tag is not found under parent.
    """
    node = parent.find(tag)
    if node is None:
        raise SchemaValueError(tag)
    return node


def get_node_with_fallback(
    parent: elem_tree.Element, tag: str, alt_tag: str
) -> elem_tree.Element:
    """Find a child element by tag, falling back to an alternate spelling.

    This handles known typos in the XML schema that may be corrected in future
    versions. Tries ``tag`` first, then ``alt_tag``.

    Raises:
        SchemaValueError: If neither tag is found under parent.
    """
    node = parent.find(tag)
    if node is None:
        node = parent.find(alt_tag)
    if node is None:
        raise SchemaValueError(f'{tag} (also tried {alt_tag})')
    return node


def get_node_text(node: elem_tree.Element | None, default: str | None = None) -> str:
    """Get the text content of an XML element.

    When ``default`` is provided it is returned instead of raising for a None
    node or a node with no text content. This handles CDNA4 schema version
    1.1.1 XML bug where some ``<Value />`` elements are self-closing (empty)
    to indicate an always-true literal (see ``_VERSIONED_FIELDS`` note).

    Raises:
        SchemaValueError: If node is None or has no text content and no
            ``default`` was provided.
    """
    if node is None:
        if default is not None:
            return default
        raise SchemaValueError('expected XML element, got None')
    if node.text is None:
        if default is not None:
            return default
        raise SchemaValueError(f'expected text in <{node.tag}>, got None')
    return node.text

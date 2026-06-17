# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Semantic AST types for AMD GPU instruction semantics.

Represents instruction semantics as an expression tree.
Every XML ``<op>`` element becomes a :class:`SemaNode`. The AST is a tree
(not SSA/graph), and nodes are immutable for safe functional construction
during enrichment.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Generator


@dataclass(frozen=True)
class SemaType:
    """Type annotation from the XML ``<t base="..." size="..."/>`` elements."""

    base: str
    size: int = 0

    U1: ClassVar[SemaType]
    U32: ClassVar[SemaType]
    U64: ClassVar[SemaType]
    I32: ClassVar[SemaType]
    I64: ClassVar[SemaType]
    F16: ClassVar[SemaType]
    F32: ClassVar[SemaType]
    F64: ClassVar[SemaType]
    B32: ClassVar[SemaType]
    B64: ClassVar[SemaType]
    BF16: ClassVar[SemaType]
    FP8: ClassVar[SemaType]
    BF8: ClassVar[SemaType]
    FP6: ClassVar[SemaType]
    FP4: ClassVar[SemaType]

    @property
    def cpp_type(self) -> str:
        """Map to C++ simulation type."""
        if self.base == 'F':
            if self.size == 16:
                return 'uint16_t'
            if self.size == 32:
                return 'float'
            if self.size == 64:
                return 'double'
        if self.base in ('U', 'B'):
            if self.size <= 8:
                return 'uint8_t'
            if self.size <= 16:
                return 'uint16_t'
            if self.size <= 32:
                return 'uint32_t'
            return 'uint64_t'
        if self.base == 'I':
            if self.size <= 8:
                return 'int8_t'
            if self.size <= 16:
                return 'int16_t'
            if self.size <= 32:
                return 'int32_t'
            return 'int64_t'
        if self.base == 'BF':
            return 'uint8_t' if self.size <= 8 else 'uint16_t'
        if self.base == 'FP':
            if self.size <= 4:
                return 'uint8_t'
            if self.size <= 8:
                return 'uint8_t'
            return 'uint16_t'
        return 'uint32_t'


SemaType.U1 = SemaType('U', 1)
SemaType.U32 = SemaType('U', 32)
SemaType.U64 = SemaType('U', 64)
SemaType.I32 = SemaType('I', 32)
SemaType.I64 = SemaType('I', 64)
SemaType.F16 = SemaType('F', 16)
SemaType.F32 = SemaType('F', 32)
SemaType.F64 = SemaType('F', 64)
SemaType.B32 = SemaType('B', 32)
SemaType.BF16 = SemaType('BF', 16)
SemaType.FP8 = SemaType('FP', 8)
SemaType.BF8 = SemaType('BF', 8)
SemaType.FP6 = SemaType('FP', 6)
SemaType.FP4 = SemaType('FP', 4)
SemaType.B64 = SemaType('B', 64)


class SemaNodeKind(Enum):
    """Discriminant for SemaNode — one per XML op type."""

    # Arithmetic
    ADD = '+'
    SUB = '-'
    MUL = '*'
    DIV = '/'
    MOD = '%'
    POW = '**'
    ADD_ASSIGN = '+='
    SUB_ASSIGN = '-='

    # Comparison
    EQ = '=='
    NE = '!='
    LT = '<'
    GT = '>'
    LE = '<='
    GE = '>='
    UNORD_NE = '<>'

    # Bitwise
    AND = '&'
    OR = '|'
    XOR = '^'
    SHL = '<<'
    SHR = '>>'
    BITNEG = '.bitneg'
    BITCAT = '.bitcat'

    # Logical
    LAND = '&&'
    LOR = '||'
    BOOLNEG = '.boolneg'

    # FP intrinsics
    FMA = '.fma'
    SQRT = '.sqrt'
    SIN = '.sin'
    COS = '.cos'
    LOG2 = '.log2'
    LDEXP = '.ldexp'
    FLOOR = '.floor'
    TRUNC = '.trunc'
    FRACT = '.fract'
    FPOW = '.pow'

    # Sign/abs
    ABS = '.abs'
    UMINUS = '.uminus'
    UPLUS = '.uplus'
    SIGN = '.sign'
    SIGNEXT = '.signext'
    SIGNEXT_FROM_BIT = '.signext_from_bit'

    # Type conversion
    CAST = '.cast'

    # Memory / register
    INSTOPERAND = '.instoperand'
    ARRAYDEREF = '.arrayderef'
    ARRAYSLICE = '.arrayslice'
    ARRAYSLICESIZE = '.arrayslicesize'
    FIELDDEREF = '.fieldderef'
    CONS_ARRAY = '.cons_array'

    # Function call
    CALL = '.call'
    LAMBDA = '.lambda'

    # Control flow
    SEQ = ':seq'
    IF = ':if'
    FOR = ':for'
    WHILE = ':while'
    BREAK = ':break'
    CONTINUE = ':continue'
    RETURN = ':return'

    # Declaration / metadata
    DECLARE = ':declare'
    COMMENT = ':comment'
    EVAL = ':eval'
    PRAGMA = ':pragma'

    # Assignment
    ASSIGN = '='

    # Ternary
    TERNARY = '?:'

    # Forward-compatibility
    SWITCH = ':switch'
    CASE = ':case'
    DEFAULT = ':default'

    # Aggregation
    SUM = '.sum'
    WITHIN = '.within'

    # FP decomposition
    EXPONENT = '.exponent'
    MANTISSA = '.mantissa'

    # Synthetic leaf nodes (not XML op types — parser-synthesized)
    LIT = '_lit'
    ID = '_id'


_STATEMENT_KINDS: frozenset[SemaNodeKind] = frozenset(
    {
        SemaNodeKind.SEQ,
        SemaNodeKind.ASSIGN,
        SemaNodeKind.IF,
        SemaNodeKind.FOR,
        SemaNodeKind.WHILE,
        SemaNodeKind.BREAK,
        SemaNodeKind.CONTINUE,
        SemaNodeKind.COMMENT,
        SemaNodeKind.DECLARE,
        SemaNodeKind.PRAGMA,
        SemaNodeKind.RETURN,
        SemaNodeKind.ADD_ASSIGN,
        SemaNodeKind.SUB_ASSIGN,
    }
)

_BINARY_KINDS: frozenset[SemaNodeKind] = frozenset(
    {
        SemaNodeKind.ADD,
        SemaNodeKind.SUB,
        SemaNodeKind.MUL,
        SemaNodeKind.DIV,
        SemaNodeKind.MOD,
        SemaNodeKind.AND,
        SemaNodeKind.OR,
        SemaNodeKind.XOR,
        SemaNodeKind.SHL,
        SemaNodeKind.SHR,
        SemaNodeKind.LAND,
        SemaNodeKind.LOR,
        SemaNodeKind.EQ,
        SemaNodeKind.NE,
        SemaNodeKind.LT,
        SemaNodeKind.GT,
        SemaNodeKind.LE,
        SemaNodeKind.GE,
        SemaNodeKind.UNORD_NE,
        SemaNodeKind.POW,
        SemaNodeKind.FPOW,
        SemaNodeKind.LDEXP,
    }
)

_UNARY_KINDS: frozenset[SemaNodeKind] = frozenset(
    {
        SemaNodeKind.BITNEG,
        SemaNodeKind.BOOLNEG,
        SemaNodeKind.ABS,
        SemaNodeKind.UMINUS,
        SemaNodeKind.UPLUS,
        SemaNodeKind.SIGN,
        SemaNodeKind.SIGNEXT,
        SemaNodeKind.CAST,
        SemaNodeKind.EVAL,
        SemaNodeKind.FLOOR,
        SemaNodeKind.TRUNC,
        SemaNodeKind.COS,
        SemaNodeKind.SIN,
        SemaNodeKind.SQRT,
        SemaNodeKind.LOG2,
        SemaNodeKind.FRACT,
        SemaNodeKind.EXPONENT,
        SemaNodeKind.MANTISSA,
    }
)

_TERNARY_KINDS: frozenset[SemaNodeKind] = frozenset(
    {
        SemaNodeKind.TERNARY,
        SemaNodeKind.ARRAYSLICE,
        SemaNodeKind.ARRAYSLICESIZE,
        SemaNodeKind.FMA,
    }
)


@dataclass(frozen=True, slots=True)
class SemaNode:
    """A single node in the semantic AST tree.

    Each XML ``<op type="...">`` element becomes one SemaNode. Children are
    the operands/subexpressions. Leaf nodes (literals, identifiers) use
    LIT and ID kinds respectively.

    Immutable (frozen) with tuple children. Enrichment creates new trees
    via ``dataclasses.replace()``, never mutates existing nodes.
    """

    kind: SemaNodeKind
    ty: SemaType | None = None
    children: tuple[SemaNode, ...] = ()
    lit_value: str | None = None
    id_name: str | None = None
    cast_target: SemaType | None = None
    call_name: str | None = None
    call_lambda: SemaType | None = None
    unknown_op: str | None = None

    def is_leaf(self) -> bool:
        return self.kind in (SemaNodeKind.LIT, SemaNodeKind.ID)

    def walk(self) -> Generator[SemaNode, None, None]:
        """Yield all nodes in pre-order traversal."""
        yield self
        for child in self.children:
            yield from child.walk()


class ExecModel(Enum):
    """Execution model for an instruction, from the ``:pragma`` annotation."""

    SCALAR = 'scalar'
    VECTOR = 'vector'
    BRANCH = 'branch'
    UNKNOWN = 'unknown'


@dataclass
class SemaBlock:
    """Semantics for one instruction — the top-level AST unit."""

    instruction_name: str
    pragma: ExecModel
    body: SemaNode
    enriched: bool = False

    @property
    def is_empty(self) -> bool:
        """True if this is a stub (empty semantics)."""
        return self.body.kind == SemaNodeKind.SEQ and len(self.body.children) == 0


def validate_types(node: SemaNode) -> None:
    """Assert that statement nodes carry no result type."""
    for n in node.walk():
        if n.kind in _STATEMENT_KINDS:
            assert (
                n.ty is None
            ), f"Statement node {n.kind} must have ty=None, got ty={n.ty}"


def validate_well_formed(node: SemaNode) -> None:
    """Assert structural well-formedness of the AST."""
    for n in node.walk():
        nc = len(n.children)
        if n.kind in (
            SemaNodeKind.ASSIGN,
            SemaNodeKind.ADD_ASSIGN,
            SemaNodeKind.SUB_ASSIGN,
            SemaNodeKind.ARRAYDEREF,
        ):
            assert nc == 2, f"{n.kind} must have 2 children, got {nc}"
        elif n.kind == SemaNodeKind.IF:
            assert nc >= 2, f"IF must have >= 2 children, got {nc}"
        elif n.kind == SemaNodeKind.FOR:
            assert (
                nc == 4
            ), f"FOR must have 4 children (init, cond, step, body), got {nc}"
        elif n.kind == SemaNodeKind.WHILE:
            assert nc == 2, f"WHILE must have 2 children (cond, body), got {nc}"
        elif n.kind == SemaNodeKind.BITCAT:
            assert nc >= 2, f"BITCAT must have >= 2 children, got {nc}"
        elif n.kind == SemaNodeKind.SIGNEXT_FROM_BIT:
            assert nc in (1, 2), f"SIGNEXT_FROM_BIT must have 1 or 2 children, got {nc}"
        elif n.kind == SemaNodeKind.WITHIN:
            assert nc in (2, 3), f"WITHIN must have 2 or 3 children, got {nc}"
        elif n.kind in _TERNARY_KINDS:
            assert nc == 3, f"{n.kind} must have 3 children, got {nc}"
        elif n.kind in _BINARY_KINDS:
            assert nc == 2, f"{n.kind} must have 2 children, got {nc}"
        elif n.kind == SemaNodeKind.RETURN:
            assert nc == 1, f"RETURN must have 1 child (return value), got {nc}"
        elif n.kind in _UNARY_KINDS:
            assert nc == 1, f"{n.kind} must have 1 child, got {nc}"
        elif n.kind == SemaNodeKind.LIT:
            assert nc == 0, f"LIT must have 0 children, got {nc}"
            assert n.lit_value is not None, "LIT must have lit_value set"
        elif n.kind == SemaNodeKind.ID:
            assert nc == 0, f"ID must have 0 children, got {nc}"
            assert n.id_name is not None, "ID must have id_name set"
        elif n.kind == SemaNodeKind.CALL:
            assert nc >= 1, f"CALL must have >= 1 child (callee id), got {nc}"

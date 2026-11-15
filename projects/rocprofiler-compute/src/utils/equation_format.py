# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Metric equations in one written form.

normalize_equation() parses an equation and writes it back with constants factored out
of the aggregation functions and only the parentheses it needs. tools/format_yaml.py
keeps the YAML files in this form. parse_equation() and format_equation() are its two
halves, so a caller can change the syntax tree in between. Standard library only: the
YAML formatting hook runs with only PyYAML installed.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

# Values that are text, not equations, such as the roofline column header
LITERAL_VALUES = {"Peak (Empirical)"}

# Aggregation functions where constants should be factored out
AGGREGATION_FUNCTIONS = {"SUM", "MIN", "MAX", "AVG"}

# Operator precedence for parentheses minimization
OPERATOR_PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2}

# Channel placeholder of a per-channel template, such as TCC_HIT[::_1]; analyze
# expands the template once per channel, so the formatter leaves it as written
CHANNEL_PLACEHOLDER = "::_"


def normalize_equation(equation: str) -> str:
    """Parse, normalize, and serialize a metric equation string.

    Raises ValueError when the equation cannot be parsed.
    """
    if equation in LITERAL_VALUES or CHANNEL_PLACEHOLDER in equation:
        return equation

    return format_equation(parse_equation(equation))


def parse_equation(equation: str) -> "ASTNode":
    """Return the syntax tree of an equation; raise ValueError if it does not parse."""
    return Parser(Tokenizer(equation).tokenize()).parse()


def format_equation(node: "ASTNode") -> str:
    """Write a syntax tree as an equation in the normalized form."""
    return _serialize_ast(_normalize_ast(node))


def _normalize_ast(node: "ASTNode") -> "ASTNode":
    """Return the tree with constants factored out of the aggregation functions."""
    if isinstance(node, UnaryOpNode):
        return UnaryOpNode(node.op, _normalize_ast(node.operand))
    if isinstance(node, BinaryOpNode):
        return BinaryOpNode(
            node.op, _normalize_ast(node.left), _normalize_ast(node.right)
        )
    if isinstance(node, FunctionCallNode):
        normalized_args = [_normalize_ast(arg) for arg in node.args]
        return _factor_constant(FunctionCallNode(node.name, normalized_args))
    return node


def _factor_constant(call: "FunctionCallNode") -> "ASTNode":
    """Return SUM(x * 2) as 2 * SUM(x), and the same for the other aggregations.

    Numbers are never negative here, since a minus sign is a unary operation, so the
    factor does not change MIN or MAX.
    """
    if call.name not in AGGREGATION_FUNCTIONS or len(call.args) != 1:
        return call
    product = call.args[0]
    if not isinstance(product, BinaryOpNode) or product.op != "*":
        return call
    for factor, rest in ((product.left, product.right), (product.right, product.left)):
        if isinstance(factor, NumberNode):
            inner_call = _factor_constant(FunctionCallNode(call.name, [rest]))
            return BinaryOpNode("*", factor, inner_call)
    return call


def _serialize_ast(
    node: "ASTNode", parent_op: Optional[str] = None, is_right: bool = False
) -> str:
    """Serialize AST to string with minimal parentheses."""
    if isinstance(node, NumberNode):
        return node.value
    if isinstance(node, (IdentifierNode, VariableNode)):
        return node.name
    if isinstance(node, UnaryOpNode):
        operand_text = _serialize_ast(node.operand)
        # -(a + b) keeps its parentheses: -a + b is a different equation
        if isinstance(node.operand, BinaryOpNode):
            operand_text = f"({operand_text})"
        return f"{node.op}{operand_text}"
    if isinstance(node, BinaryOpNode):
        left_text = _serialize_ast(node.left, node.op)
        right_text = _serialize_ast(node.right, node.op, is_right=True)
        operation_text = f"{left_text} {node.op} {right_text}"
        if _needs_parentheses(node.op, parent_op, is_right):
            return f"({operation_text})"
        return operation_text
    args_text = ", ".join(_serialize_ast(arg) for arg in node.args)
    return f"{node.name}({args_text})"


def _needs_parentheses(op: str, parent_op: Optional[str], is_right: bool) -> bool:
    """Return whether a binary operation under parent_op needs parentheses.

    A right operand keeps them even where the result would be equal, such as
    a + (b + c), so the evaluation order stays the same.
    """
    if parent_op is None:
        return False
    precedence = OPERATOR_PRECEDENCE[op]
    parent_precedence = OPERATOR_PRECEDENCE[parent_op]
    return precedence < parent_precedence or (
        precedence == parent_precedence and is_right
    )


class TokenType(Enum):
    NUMBER = auto()
    IDENTIFIER = auto()
    VARIABLE = auto()
    OPERATOR = auto()
    LPAREN = auto()
    RPAREN = auto()
    COMMA = auto()
    EOF = auto()


@dataclass
class Token:
    type: TokenType
    value: str
    pos: int


class Tokenizer:
    """Tokenize metric equations into tokens."""

    SINGLE_CHARACTER_TOKENS = {
        "+": TokenType.OPERATOR,
        "-": TokenType.OPERATOR,
        "*": TokenType.OPERATOR,
        "/": TokenType.OPERATOR,
        "(": TokenType.LPAREN,
        ")": TokenType.RPAREN,
        ",": TokenType.COMMA,
    }

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    def _skip_whitespace(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def _read_number(self) -> str:
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] in "0123456789.":
            self.pos += 1
        if self.pos < len(self.text) and self.text[self.pos] in "eE":
            self.pos += 1
            if self.pos < len(self.text) and self.text[self.pos] in "+-":
                self.pos += 1
            while self.pos < len(self.text) and self.text[self.pos].isdigit():
                self.pos += 1
        return self.text[start : self.pos]

    def _read_name(self) -> str:
        """Read an identifier, or a variable with its leading $."""
        start = self.pos
        self.pos += 1
        while self.pos < len(self.text) and (
            self.text[self.pos].isalnum() or self.text[self.pos] == "_"
        ):
            self.pos += 1
        return self.text[start : self.pos]

    def tokenize(self) -> List[Token]:
        tokens = []
        while self.pos < len(self.text):
            self._skip_whitespace()
            if self.pos >= len(self.text):
                break

            start_pos = self.pos
            char = self.text[self.pos]

            if char.isdigit() or (
                char == "."
                and self.pos + 1 < len(self.text)
                and self.text[self.pos + 1].isdigit()
            ):
                tokens.append(Token(TokenType.NUMBER, self._read_number(), start_pos))
            elif char == "$":
                tokens.append(Token(TokenType.VARIABLE, self._read_name(), start_pos))
            elif char.isalpha() or char == "_":
                tokens.append(Token(TokenType.IDENTIFIER, self._read_name(), start_pos))
            elif char in self.SINGLE_CHARACTER_TOKENS:
                self.pos += 1
                token_type = self.SINGLE_CHARACTER_TOKENS[char]
                tokens.append(Token(token_type, char, start_pos))
            else:
                raise ValueError(f"Unknown character '{char}' at position {start_pos}")

        tokens.append(Token(TokenType.EOF, "", self.pos))
        return tokens


@dataclass
class ASTNode:
    pass


@dataclass
class NumberNode(ASTNode):
    value: str


@dataclass
class IdentifierNode(ASTNode):
    name: str


@dataclass
class VariableNode(ASTNode):
    name: str


@dataclass
class BinaryOpNode(ASTNode):
    op: str
    left: ASTNode
    right: ASTNode


@dataclass
class UnaryOpNode(ASTNode):
    op: str
    operand: ASTNode


@dataclass
class FunctionCallNode(ASTNode):
    name: str
    args: List[ASTNode]


class Parser:
    """Recursive descent parser for metric equations."""

    def __init__(self, tokens: List[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    def _current(self) -> Token:
        return self.tokens[self.pos]

    def _advance(self) -> Token:
        token = self._current()
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
        return token

    def _expect(self, token_type: TokenType) -> Token:
        token = self._current()
        if token.type != token_type:
            raise ValueError(f"Expected {token_type}, got {token.type}")
        return self._advance()

    def parse(self) -> ASTNode:
        tree = self._parse_additive()
        token = self._current()
        if token.type != TokenType.EOF:
            raise ValueError(
                f"Unexpected token after expression: {token.type} at pos {token.pos}"
            )
        return tree

    def _parse_additive(self) -> ASTNode:
        left = self._parse_multiplicative()
        while (
            self._current().type == TokenType.OPERATOR and self._current().value in "+-"
        ):
            op = self._advance().value
            left = BinaryOpNode(op, left, self._parse_multiplicative())
        return left

    def _parse_multiplicative(self) -> ASTNode:
        left = self._parse_unary()
        while (
            self._current().type == TokenType.OPERATOR and self._current().value in "*/"
        ):
            op = self._advance().value
            left = BinaryOpNode(op, left, self._parse_unary())
        return left

    def _parse_unary(self) -> ASTNode:
        if self._current().type == TokenType.OPERATOR and self._current().value == "-":
            op = self._advance().value
            return UnaryOpNode(op, self._parse_unary())
        return self._parse_primary()

    def _parse_primary(self) -> ASTNode:
        token = self._current()

        if token.type == TokenType.NUMBER:
            self._advance()
            return NumberNode(token.value)

        if token.type == TokenType.VARIABLE:
            self._advance()
            return VariableNode(token.value)

        if token.type == TokenType.IDENTIFIER:
            name = self._advance().value
            if self._current().type == TokenType.LPAREN:
                self._advance()
                args = self._parse_args()
                self._expect(TokenType.RPAREN)
                return FunctionCallNode(name, args)
            return IdentifierNode(name)

        if token.type == TokenType.LPAREN:
            self._advance()
            expr = self._parse_additive()
            self._expect(TokenType.RPAREN)
            return expr

        raise ValueError(f"Unexpected token {token.type} at pos {token.pos}")

    def _parse_args(self) -> List[ASTNode]:
        args = [self._parse_additive()]
        while self._current().type == TokenType.COMMA:
            self._advance()
            args.append(self._parse_additive())
        return args

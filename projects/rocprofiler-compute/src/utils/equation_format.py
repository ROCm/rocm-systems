# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Metric equations in one written form.

normalize_equation() parses an equation and writes it back with constants factored out
of the aggregation functions and only the parentheses it needs. tools/format_yaml.py
keeps the YAML files in this form. Standard library only.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

# Literal string values that should not be parsed as equations
LITERAL_VALUES = {"N/A", "None", "null", "true", "false", "Peak (Empirical)"}

# Aggregation functions where constants should be factored out
AGGREGATION_FUNCTIONS = {"SUM", "MIN", "MAX", "AVG"}

# Operator precedence for parentheses minimization
OPERATOR_PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2}


def normalize_equation(equation: str) -> str:
    """Parse, normalize, and serialize a metric equation string."""
    stripped = equation.strip()
    if stripped in LITERAL_VALUES or stripped.startswith(('"', "'")):
        return equation

    try:
        tokens = Tokenizer(equation).tokenize()
        ast = Parser(tokens).parse()
        return _serialize_ast(_normalize_ast(ast))
    except (ValueError, IndexError):
        return equation


def _normalize_ast(node: "ASTNode") -> "ASTNode":
    """Recursively normalize AST: factor constants out of aggregation functions."""
    if isinstance(node, (NumberNode, IdentifierNode, VariableNode)):
        return node

    if isinstance(node, UnaryOpNode):
        return UnaryOpNode(node.op, _normalize_ast(node.operand))

    if isinstance(node, BinaryOpNode):
        return BinaryOpNode(
            node.op, _normalize_ast(node.left), _normalize_ast(node.right)
        )

    if isinstance(node, FunctionCallNode):
        normalized_args = [_normalize_ast(arg) for arg in node.args]
        func_node = FunctionCallNode(node.name, normalized_args)

        if node.name in AGGREGATION_FUNCTIONS and len(normalized_args) == 1:
            inner = normalized_args[0]
            if isinstance(inner, BinaryOpNode) and inner.op == "*":
                if isinstance(inner.left, NumberNode):
                    return BinaryOpNode(
                        "*", inner.left, FunctionCallNode(node.name, [inner.right])
                    )
                if isinstance(inner.right, NumberNode):
                    return BinaryOpNode(
                        "*", inner.right, FunctionCallNode(node.name, [inner.left])
                    )
        return func_node

    return node


def _serialize_ast(
    node: "ASTNode", parent_op: Optional[str] = None, is_right: bool = False
) -> str:
    """Serialize AST to string with minimal parentheses."""
    if isinstance(node, NumberNode):
        return node.value
    if isinstance(node, IdentifierNode):
        return node.name
    if isinstance(node, VariableNode):
        return node.name
    if isinstance(node, UnaryOpNode):
        operand_str = _serialize_ast(node.operand, node.op, False)
        # Parenthesize binary operands to preserve semantics: -(a + b) not -a + b
        if isinstance(node.operand, BinaryOpNode):
            operand_str = f"({operand_str})"
        return f"{node.op}{operand_str}"

    if isinstance(node, BinaryOpNode):
        left = _serialize_ast(node.left, node.op, False)
        right = _serialize_ast(node.right, node.op, True)
        result = f"{left} {node.op} {right}"

        if parent_op is not None:
            node_prec = OPERATOR_PRECEDENCE.get(node.op, 0)
            parent_prec = OPERATOR_PRECEDENCE.get(parent_op, 0)
            if node_prec < parent_prec:
                return f"({result})"
            if node_prec == parent_prec and is_right and parent_op in "-/":
                return f"({result})"
        return result

    if isinstance(node, FunctionCallNode):
        args = ", ".join(_serialize_ast(arg) for arg in node.args)
        return f"{node.name}({args})"

    return str(node)


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

    OPERATORS = {"+", "-", "*", "/"}

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    def _peek(self) -> str:
        return self.text[self.pos] if self.pos < len(self.text) else ""

    def _advance(self) -> str:
        ch = self._peek()
        self.pos += 1
        return ch

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

    def _read_identifier(self) -> str:
        start = self.pos
        while self.pos < len(self.text) and (
            self.text[self.pos].isalnum() or self.text[self.pos] == "_"
        ):
            self.pos += 1
        return self.text[start : self.pos]

    def _read_variable(self) -> str:
        start = self.pos
        self._advance()
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
            ch = self._peek()

            if ch.isdigit() or (
                ch == "."
                and self.pos + 1 < len(self.text)
                and self.text[self.pos + 1].isdigit()
            ):
                tokens.append(Token(TokenType.NUMBER, self._read_number(), start_pos))
            elif ch == "$":
                tokens.append(
                    Token(TokenType.VARIABLE, self._read_variable(), start_pos)
                )
            elif ch.isalpha() or ch == "_":
                tokens.append(
                    Token(TokenType.IDENTIFIER, self._read_identifier(), start_pos)
                )
            elif ch in self.OPERATORS:
                self._advance()
                tokens.append(Token(TokenType.OPERATOR, ch, start_pos))
            elif ch == "(":
                self._advance()
                tokens.append(Token(TokenType.LPAREN, "(", start_pos))
            elif ch == ")":
                self._advance()
                tokens.append(Token(TokenType.RPAREN, ")", start_pos))
            elif ch == ",":
                self._advance()
                tokens.append(Token(TokenType.COMMA, ",", start_pos))
            else:
                raise ValueError(f"Unknown character '{ch}' at position {start_pos}")

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
        ast = self._parse_additive()
        if self._current().type != TokenType.EOF:
            curr = self._current()
            raise ValueError(
                f"Unexpected token after expression: {curr.type} at pos {curr.pos}"
            )
        return ast

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
        if self._current().type == TokenType.RPAREN:
            return []
        args = [self._parse_additive()]
        while self._current().type == TokenType.COMMA:
            self._advance()
            args.append(self._parse_additive())
        return args

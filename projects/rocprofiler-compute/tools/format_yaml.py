#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""YAML metric equation formatter for rocprofiler-compute pre-commit hook."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any

import yaml

# =============================================================================
# Configuration - Edit these to accommodate future YAML updates
# =============================================================================

# Path patterns that contain metric equations to check (substrings matched against path)
EQUATION_FILE_PATTERNS = [
    "analysis_configs/gfx",
    "rocprof_compute_tui/utils/gfx",
]

# Path suffixes to exclude from equation checking
EQUATION_FILE_EXCLUDE_SUFFIXES = [
    "_template.yaml",
]

# Path patterns to skip entirely (substrings matched against path)
SKIP_FILE_PATTERNS = [
    "/build/",
    "profiling_config.yaml",
]

# YAML keys that contain metric equations
EQUATION_KEYS = {"value", "avg", "min", "max", "peak"}

# Literal string values that should not be parsed as equations
LITERAL_VALUES = {"N/A", "None", "null", "true", "false", "Peak (Empirical)"}

# Aggregation functions where constants should be factored out
AGGREGATION_FUNCTIONS = {"SUM", "MIN", "MAX", "AVG"}

# Operator precedence for parentheses minimization
OPERATOR_PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2}

# =============================================================================
# Tokenizer / Parser / AST
# =============================================================================


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

    def __init__(self, text: str):
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

    def tokenize(self) -> list[Token]:
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
    args: list[ASTNode]


class Parser:
    """Recursive descent parser for metric equations."""

    def __init__(self, tokens: list[Token]):
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

    def _parse_args(self) -> list[ASTNode]:
        if self._current().type == TokenType.RPAREN:
            return []
        args = [self._parse_additive()]
        while self._current().type == TokenType.COMMA:
            self._advance()
            args.append(self._parse_additive())
        return args


# =============================================================================
# Normalizer
# =============================================================================


def _normalize_ast(node: ASTNode) -> ASTNode:
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


# =============================================================================
# Serializer
# =============================================================================


def _serialize_ast(
    node: ASTNode, parent_op: str | None = None, is_right: bool = False
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


def extract_metric_equations(
    data: dict[str, Any], path: str = ""
) -> list[tuple[str, str, str]]:
    """Extract (path, key, value) tuples for equation fields from YAML data."""
    equations = []

    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            if key in EQUATION_KEYS and isinstance(value, str):
                equations.append((current_path, key, value))
            elif isinstance(value, dict):
                equations.extend(extract_metric_equations(value, current_path))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        equations.extend(
                            extract_metric_equations(item, f"{current_path}[{i}]")
                        )
    return equations


def _should_check_equations(path: Path) -> bool:
    """Return True if the file contains metric equations to check."""
    path_str = str(path)
    for exclude_suffix in EQUATION_FILE_EXCLUDE_SUFFIXES:
        if path_str.endswith(exclude_suffix):
            return False
    for pattern in EQUATION_FILE_PATTERNS:
        if pattern in path_str:
            return True
    return False


def _should_skip_file(path: Path) -> bool:
    """Return True if the file should be skipped entirely."""
    path_str = str(path)
    for pattern in SKIP_FILE_PATTERNS:
        if pattern in path_str:
            return True
    try:
        with open(path, encoding="utf-8") as f:
            if "AUTOGENERATED" in f.readline().upper():
                return True
    except (OSError, UnicodeDecodeError):
        return True
    return False


def _apply_fixes(content: str, equations: list[tuple[str, str, str]]) -> str:
    """Apply equation normalizations using regex to handle multi-line YAML."""
    result = content
    for _, _, original in equations:
        normalized = normalize_equation(original)
        if normalized != original:
            pattern = re.escape(original).replace(r"\ ", r"\s+")
            result = re.sub(pattern, normalized, result, count=1)
    return result


def check_file(
    path: Path, fix: bool = False, show_diff: bool = False
) -> tuple[bool, list[str]]:
    """Check a YAML file for formatting issues. Returns (is_valid, issues)."""
    if _should_skip_file(path):
        return True, []

    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
        data = yaml.safe_load(content)
    except (yaml.YAMLError, OSError) as e:
        return False, [f"Failed to parse YAML: {e}"]

    if data is None or not _should_check_equations(path):
        return True, []

    equations = extract_metric_equations(data)
    issues = []
    equations_to_fix = []

    for eq_path, key, value in equations:
        normalized = normalize_equation(value)
        if normalized != value:
            equations_to_fix.append((eq_path, key, value))
            issues.append(f"  {eq_path}: equation needs normalization")
            if show_diff:
                issues.append(f"    - {value}")
                issues.append(f"    + {normalized}")

    if fix and equations_to_fix:
        with open(path, "w", encoding="utf-8") as f:
            f.write(_apply_fixes(content, equations_to_fix))
        return True, [f"  Fixed {len(equations_to_fix)} equation(s)"]

    return len(issues) == 0, issues


def main() -> int:
    parser = argparse.ArgumentParser(
        description="YAML metric equation formatter for rocprofiler-compute"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Check without modifying")
    mode.add_argument("--fix", action="store_true", help="Auto-fix in place")
    mode.add_argument("--diff", action="store_true", help="Show proposed changes")
    parser.add_argument("files", nargs="*", type=Path, help="Files to check")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    files = args.files or [Path(line.strip()) for line in sys.stdin if line.strip()]

    if not files:
        print("No files to check")
        return 0

    all_valid = True
    total_issues = 0

    for file_path in files:
        if not file_path.exists() or file_path.suffix.lower() not in {".yaml", ".yml"}:
            continue

        is_valid, issues = check_file(file_path, fix=args.fix, show_diff=args.diff)

        if not is_valid:
            all_valid = False
            total_issues += len(issues)
            print(f"{file_path}:")
            for issue in issues:
                print(issue)
        elif args.verbose:
            print(f"{file_path}: OK")

    if not all_valid:
        print(f"\n{total_issues} formatting issue(s) found")
        return 1

    if args.verbose:
        print(f"\nAll {len(files)} files passed format checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())

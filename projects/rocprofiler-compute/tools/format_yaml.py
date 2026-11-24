#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Write the metric equations of the panel configs, panel collectables and metric
library YAML files in one form, which CONTRIBUTING.md describes."""

import argparse
import re
import sys
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# =============================================================================
# Configuration - Edit these to accommodate future YAML updates
# =============================================================================

# Path pattern of the panel collectables, whose top-level values are all equations
COLLECTABLES_FILE_PATTERN = "analysis_configs/collectables/"

# Path patterns that contain metric equations to check (substrings matched against path)
EQUATION_FILE_PATTERNS = [
    "analysis_configs/gfx",
    "metric_library/gfx",
    COLLECTABLES_FILE_PATTERN,
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

# YAML keys that contain metric equations ("formula" and "peak_empirical" in metric
# library files)
EQUATION_KEYS = {
    "value",
    "avg",
    "min",
    "max",
    "std dev",
    "expr",
    "peak",
    "formula",
    "peak_empirical",
}

# Key of the column titles of a metric_table. The titles use the equation keys, but
# they are not equations.
HEADER_KEY = "header"

# Aggregation functions whose constant factor moves out of the call
AGGREGATION_FUNCTIONS = {"SUM", "MIN", "MAX", "AVG"}

# Precedence of each binary operator, which decides where parentheses are needed
OPERATOR_PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2}

# Marks a per-channel template such as TCC_HIT[::_1]. Analyze expands the template
# once for each channel, so the formatter leaves it as written.
CHANNEL_PLACEHOLDER = "::_"


def normalize_equation(equation: str) -> str:
    """Return the equation in the form of the YAML files.

    Raises ValueError if the equation does not parse.
    """
    if CHANNEL_PLACEHOLDER in equation:
        return equation

    tree = Parser(Tokenizer(equation).tokenize()).parse()
    return _serialize_ast(_normalize_ast(tree))


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
    """Write a syntax tree as an equation with only the parentheses it needs."""
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


def extract_metric_equations(
    data: Dict[str, Any], path: str = ""
) -> List[Tuple[str, str]]:
    """Return (key path, equation) for each equation in YAML data. Table headers
    are skipped."""
    equations = []

    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            if key == HEADER_KEY:
                continue
            if key in EQUATION_KEYS and isinstance(value, str):
                equations.append((current_path, value))
            elif isinstance(value, dict):
                equations.extend(extract_metric_equations(value, current_path))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        equations.extend(
                            extract_metric_equations(item, f"{current_path}[{i}]")
                        )
    return equations


def file_equations(path: Path, data: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Return (key path, equation) of each equation in the YAML data of a file."""
    if COLLECTABLES_FILE_PATTERN in str(path):
        return [(name, value) for name, value in data.items() if isinstance(value, str)]
    return extract_metric_equations(data)


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
    return False


def _apply_fixes(content: str, fixes: List[Tuple[str, str]]) -> str:
    """Replace each original equation with its normalized form in the YAML text."""
    result = content
    for original, normalized in fixes:
        pattern = re.escape(original).replace(r"\ ", r"\s+")
        result = re.sub(pattern, normalized, result, count=1)
    return result


def check_file(
    path: Path, fix: bool = False, show_diff: bool = False
) -> Tuple[bool, List[str]]:
    """Check a YAML file for formatting issues. Returns (is_valid, issues)."""
    if _should_skip_file(path):
        return True, []

    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
        data = yaml.safe_load(content)
    except (yaml.YAMLError, OSError, UnicodeDecodeError) as e:
        return False, [f"Failed to parse YAML: {e}"]

    if data is None or not _should_check_equations(path):
        return True, []

    parse_errors = []
    format_issues = []
    fixes = []

    for eq_path, value in file_equations(path, data):
        try:
            normalized = normalize_equation(value)
        except ValueError as e:
            parse_errors.append(f"  {eq_path}: cannot parse equation: {e}\n    {value}")
            continue
        if normalized != value:
            fixes.append((value, normalized))
            format_issues.append(f"  {eq_path}: equation needs normalization")
            if show_diff:
                format_issues.append(f"    - {value}")
                format_issues.append(f"    + {normalized}")

    if fix and fixes:
        with open(path, "w", encoding="utf-8") as f:
            f.write(_apply_fixes(content, fixes))
        format_issues = []

    issues = parse_errors + format_issues
    return not issues, issues


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
    """Split a metric equation into numbers, names, operators, parentheses and
    commas."""

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
    """Build the syntax tree of a metric equation."""

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


if __name__ == "__main__":
    sys.exit(main())

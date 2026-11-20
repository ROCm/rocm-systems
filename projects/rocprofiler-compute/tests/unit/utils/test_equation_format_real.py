# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Check src/utils/equation_format.py on every equation of the YAML files that
tools/format_yaml.py formats."""

from typing import List, Set

import yaml
from format_yaml import extract_metric_equations

from rocprof_compute_soc.metric_library.loader import default_metric_library_dir
from utils.equation_format import (
    CHANNEL_PLACEHOLDER,
    ASTNode,
    BinaryOpNode,
    FunctionCallNode,
    NumberNode,
    UnaryOpNode,
    normalize_equation,
    parse_equation,
)
from utils.metrics.expression import build_eval_string


def real_equations() -> Set[str]:
    """Return every equation of the metric library."""
    equations = set()
    for path in default_metric_library_dir().glob("gfx*/*.yaml"):
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        equations.update(equation for _, equation in extract_metric_equations(document))
    return equations


def fully_parenthesized(node: ASTNode) -> str:
    """Write a syntax tree with parentheses around every operation and no spaces."""
    if isinstance(node, UnaryOpNode):
        return f"{node.op}({fully_parenthesized(node.operand)})"
    if isinstance(node, BinaryOpNode):
        left = fully_parenthesized(node.left)
        right = fully_parenthesized(node.right)
        return f"({left}{node.op}{right})"
    if isinstance(node, FunctionCallNode):
        return f"{node.name}({','.join(fully_parenthesized(arg) for arg in node.args)})"
    if isinstance(node, NumberNode):
        return node.value
    return node.name


def test_real_equations_round_trip():
    """Each real equation parses as Python reads it, and normalizing it with every
    parenthesis written out and no spaces gives back the equation."""
    failures: List[str] = []
    for equation in sorted(real_equations()):
        if CHANNEL_PLACEHOLDER in equation:
            continue
        written_out = fully_parenthesized(parse_equation(equation))
        normalized = normalize_equation(written_out)
        if build_eval_string(written_out) != build_eval_string(equation):
            failures.append(f"parsed as {written_out}: {equation}")
        elif normalized != equation:
            failures.append(f"normalized to {normalized}: {equation}")
    assert failures == []

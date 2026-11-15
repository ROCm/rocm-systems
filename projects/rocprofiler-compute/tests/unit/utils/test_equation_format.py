# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for src/utils/equation_format.py."""

from typing import List, Set

import pytest
import yaml
from format_yaml import extract_metric_equations

import config
from rocprof_compute_soc.metric_library.loader import default_metric_library_dir
from utils.equation_format import (
    CHANNEL_PLACEHOLDER,
    LITERAL_VALUES,
    ASTNode,
    BinaryOpNode,
    FunctionCallNode,
    NumberNode,
    UnaryOpNode,
    normalize_equation,
    parse_equation,
)
from utils.metrics.expression import build_eval_string
from utils.utils_counter_defs import COLLECTABLES_DIR

# (input, normalized) pairs; each normalized form is also its own normalized form
FACTORING_CASES = [
    # a constant on the right of the product
    (
        "SUM(TCP_REQ_sum * 64) / SUM(End_Timestamp - Start_Timestamp)",
        "64 * SUM(TCP_REQ_sum) / SUM(End_Timestamp - Start_Timestamp)",
    ),
    # every aggregation; a factored product on the right keeps its parentheses
    (
        "AVG(64 * x) / MIN(2 * y) + MAX(z * 4)",
        "64 * AVG(x) / (2 * MIN(y)) + 4 * MAX(z)",
    ),
    # left as written: a second argument, a function that is not an aggregation,
    # a sum, and a negative constant, which would swap MIN and MAX
    ("MAX(2 * x, 1)", "MAX(2 * x, 1)"),
    ("TO_INT(x * 2)", "TO_INT(x * 2)"),
    ("SUM(x + 128)", "SUM(x + 128)"),
    ("MIN(-2 * x)", "MIN(-2 * x)"),
]


def real_equations() -> Set[str]:
    """Return every equation of the panel configs, metric library and collectables."""
    files = list(config.analysis_configs_dir.glob("gfx*/*.yaml"))
    files += default_metric_library_dir().glob("gfx*/*.yaml")
    equations = set()
    for path in files:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        equations.update(equation for _, equation in extract_metric_equations(document))
    for path in COLLECTABLES_DIR.glob("*.yaml"):
        equations.update(yaml.safe_load(path.read_text(encoding="utf-8")).values())
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
        if equation in LITERAL_VALUES or CHANNEL_PLACEHOLDER in equation:
            continue
        written_out = fully_parenthesized(parse_equation(equation))
        normalized = normalize_equation(written_out)
        if build_eval_string(written_out) != build_eval_string(equation):
            failures.append(f"parsed as {written_out}: {equation}")
        elif normalized != equation:
            failures.append(f"normalized to {normalized}: {equation}")
    assert failures == []


@pytest.mark.parametrize(
    "equation", ["Peak (Empirical)", "SUM(TO_INT(TCC_ATOMIC[::_1])) / SUM($denom)"]
)
def test_text_and_channel_templates_left_as_written(equation):
    assert normalize_equation(equation) == equation


@pytest.mark.parametrize(("equation", "normalized"), FACTORING_CASES)
def test_constants_factored_out_of_aggregations(equation, normalized):
    assert normalize_equation(equation) == normalized
    assert normalize_equation(normalized) == normalized


def test_unary_minus_keeps_its_operand():
    """-(a - b) keeps its parentheses: -a - b is a different equation."""
    for equation in ["-(a - b)", "a - -b", "100 * -SUM(x) / y"]:
        assert normalize_equation(equation) == equation


@pytest.mark.parametrize(
    "equation",
    [
        "SUM(a",  # an unclosed call
        "a b",  # two operands without an operator
        "a +",  # an operator without its right operand
        "TCC_HIT[0]",  # a single channel, which no equation names
    ],
)
def test_unparseable_equation_raises(equation):
    with pytest.raises(ValueError):
        parse_equation(equation)

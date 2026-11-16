# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for src/utils/equation_format.py."""

import pytest

from utils.equation_format import normalize_equation, parse_equation

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
    # two constants come out one at a time
    ("SUM(x * 4 * 2)", "2 * (4 * SUM(x))"),
    # left as written: a second argument, a function that is not an aggregation,
    # a sum, and a negative constant, which would swap MIN and MAX
    ("MAX(2 * x, 1)", "MAX(2 * x, 1)"),
    ("TO_INT(x * 2)", "TO_INT(x * 2)"),
    ("SUM(x + 128)", "SUM(x + 128)"),
    ("MIN(-2 * x)", "MIN(-2 * x)"),
]


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

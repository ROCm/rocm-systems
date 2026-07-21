# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Restricted evaluator for metric expressions."""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable, Collection, Mapping
from typing import Any

import pandas as pd

MAX_EXPRESSION_LENGTH = 100_000
MAX_EXPRESSION_NODES = 10_000


class UnsafeExpressionError(ValueError):
    """Raised when an expression contains syntax outside the metric language."""


class _ExpressionEvaluator:
    _BINARY_OPERATORS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }
    _UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }
    _COMPARISON_OPERATORS: dict[type[ast.cmpop], Callable[[Any, Any], Any]] = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
    }

    def __init__(
        self,
        variables: Mapping[str, Any],
        functions: Mapping[str, Callable[..., Any]],
        subscriptable_names: Collection[str],
    ) -> None:
        self._variables = variables
        self._functions = functions
        self._subscriptable_names = frozenset(subscriptable_names)

    def evaluate(self, node: ast.AST) -> object:
        method = getattr(self, f"_evaluate_{type(node).__name__}", None)
        if method is None:
            raise UnsafeExpressionError(
                f"Unsupported expression element: {type(node).__name__}"
            )
        return method(node)

    def _evaluate_Expression(self, node: ast.Expression) -> object:
        return self.evaluate(node.body)

    def _evaluate_Constant(self, node: ast.Constant) -> object:
        if node.value is None or isinstance(node.value, (bool, int, float, str)):
            return node.value
        raise UnsafeExpressionError(
            f"Unsupported constant type: {type(node.value).__name__}"
        )

    def _evaluate_Name(self, node: ast.Name) -> object:
        try:
            return self._variables[node.id]
        except KeyError as error:
            raise NameError(f"name '{node.id}' is not defined") from error

    def _evaluate_BinOp(self, node: ast.BinOp) -> object:
        function = self._BINARY_OPERATORS.get(type(node.op))
        if function is None:
            raise UnsafeExpressionError(
                f"Unsupported binary operator: {type(node.op).__name__}"
            )
        return function(self.evaluate(node.left), self.evaluate(node.right))

    def _evaluate_UnaryOp(self, node: ast.UnaryOp) -> object:
        function = self._UNARY_OPERATORS.get(type(node.op))
        if function is None:
            raise UnsafeExpressionError(
                f"Unsupported unary operator: {type(node.op).__name__}"
            )
        return function(self.evaluate(node.operand))

    def _evaluate_Compare(self, node: ast.Compare) -> object:
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise UnsafeExpressionError("Chained comparisons are not supported")
        function = self._COMPARISON_OPERATORS.get(type(node.ops[0]))
        if function is None:
            raise UnsafeExpressionError(
                f"Unsupported comparison operator: {type(node.ops[0]).__name__}"
            )
        return function(self.evaluate(node.left), self.evaluate(node.comparators[0]))

    def _evaluate_IfExp(self, node: ast.IfExp) -> object:
        branch = node.body if bool(self.evaluate(node.test)) else node.orelse
        return self.evaluate(branch)

    def _evaluate_Subscript(self, node: ast.Subscript) -> object:
        if (
            not isinstance(node.value, ast.Name)
            or node.value.id not in self._subscriptable_names
        ):
            raise UnsafeExpressionError(
                "Subscripts are limited to approved metric data sources"
            )

        slice_node = node.slice
        if hasattr(ast, "Index") and isinstance(slice_node, ast.Index):
            slice_node = slice_node.value
        if not isinstance(slice_node, ast.Constant) or not isinstance(
            slice_node.value, str
        ):
            raise UnsafeExpressionError("Subscript keys must be string literals")

        source = self._variables[node.value.id]
        return source[slice_node.value]

    def _evaluate_Call(self, node: ast.Call) -> object:
        if node.keywords:
            raise UnsafeExpressionError("Keyword arguments are not supported")

        if isinstance(node.func, ast.Name):
            try:
                function = self._functions[node.func.id]
            except KeyError as error:
                raise UnsafeExpressionError(
                    f"Function '{node.func.id}' is not allowed"
                ) from error
            args = [self.evaluate(arg) for arg in node.args]
            return function(*args)

        if isinstance(node.func, ast.Attribute) and node.func.attr == "where":
            if not 1 <= len(node.args) <= 2:
                raise UnsafeExpressionError(
                    "The where method accepts only condition and other arguments"
                )
            source = self.evaluate(node.func.value)
            if not isinstance(source, pd.Series):
                raise UnsafeExpressionError(
                    "The where method is limited to pandas Series"
                )
            args = [self.evaluate(arg) for arg in node.args]
            if any(callable(arg) for arg in args):
                raise UnsafeExpressionError("Callable where arguments are not allowed")
            return source.where(*args)

        raise UnsafeExpressionError("Only approved metric functions may be called")


def evaluate_expression(
    expression: str,
    *,
    variables: Mapping[str, Any],
    functions: Mapping[str, Callable[..., Any]],
    subscriptable_names: Collection[str],
) -> object:
    """Evaluate a metric expression without compiling or executing Python code."""
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise UnsafeExpressionError("Metric expression is too long")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise UnsafeExpressionError(
            f"Invalid metric expression: {error.msg}"
        ) from error

    if sum(1 for _ in ast.walk(tree)) > MAX_EXPRESSION_NODES:
        raise UnsafeExpressionError("Metric expression is too complex")

    return _ExpressionEvaluator(
        variables=variables,
        functions=functions,
        subscriptable_names=subscriptable_names,
    ).evaluate(tree)

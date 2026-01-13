##############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
##############################################################################

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Protocol, Union

from .ui_model import MetricSnapshot, PredicateResult


class Predicate(Protocol):
    def evaluate(self, snap: MetricSnapshot) -> PredicateResult: ...


MetricId = str


@dataclass(frozen=True)
class EvalResult:
    value: Optional[float]
    inputs: dict[str, float]
    missing: bool


class ValueExpr(Protocol):
    def evaluate(self, snap: MetricSnapshot) -> EvalResult: ...
    def display(self, snap: MetricSnapshot) -> str: ...


@dataclass(frozen=True)
class ConstExpr:
    value: float

    def evaluate(self, snap: MetricSnapshot) -> EvalResult:
        return EvalResult(
            value=self.value,
            inputs={"<const>": self.value},
            missing=False,
        )

    def display(self, snap: MetricSnapshot) -> str:
        return f"{self.value:.4g}"


@dataclass(frozen=True)
class MetricExpr:
    metric_id: MetricId

    def evaluate(self, snap: MetricSnapshot) -> EvalResult:
        v = snap.values.get(self.metric_id)
        return EvalResult(
            value=v,
            inputs={self.metric_id: v if v is not None else float("nan")},
            missing=v is None,
        )

    def display(self, snap: MetricSnapshot) -> str:
        meta = snap.meta.get(self.metric_id)
        return f"{self.metric_id} ({meta.metric_name})" if meta else self.metric_id


_SUPPORTED_OPS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
}


@dataclass(frozen=True)
class BinaryExpr:
    left: ValueExpr
    op: str
    right: ValueExpr

    def evaluate(self, snap: MetricSnapshot) -> EvalResult:
        l = self.left.evaluate(snap)  # noqa: E741
        r = self.right.evaluate(snap)

        inputs = {**l.inputs, **r.inputs}

        if l.missing or r.missing:
            return EvalResult(None, inputs, True)

        try:
            value = _SUPPORTED_OPS[self.op](l.value, r.value)  # type: ignore[arg-type]
        except Exception:
            return EvalResult(None, inputs, True)

        return EvalResult(value, inputs, False)

    def display(self, snap: MetricSnapshot) -> str:
        return f"{self.left.display(snap)} {self.op} {self.right.display(snap)}"


_METRIC_RE = re.compile(r"\d+\.\d+\.\d+")

_TOKEN_RE = re.compile(
    r"""
    (?P<METRIC>\d+\.\d+\.\d+)
  | (?P<NUMBER>\d+(\.\d+)?)
  | (?P<OP>[+\-*/])
  | (?P<LPAREN>\()
  | (?P<RPAREN>\))
  | (?P<SPACE>\s+)
    """,
    re.VERBOSE,
)


class _Token:
    def __init__(self, typ: str, value: str) -> None:
        self.type = typ
        self.value = value


def _tokenize(expr: str) -> list[_Token]:
    tokens: list[_Token] = []
    pos = 0
    while pos < len(expr):
        m = _TOKEN_RE.match(expr, pos)
        if not m:
            raise ValueError(f"Invalid token in expression: '{expr[pos:]}'")
        pos = m.end()
        typ = m.lastgroup
        if typ != "SPACE":
            tokens.append(_Token(typ, m.group(typ)))
    return tokens


class _Parser:
    def __init__(self, tokens: list[_Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    def _peek(self) -> Optional[_Token]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _consume(self, expected: Optional[str] = None) -> _Token:
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of expression")
        if expected and tok.type != expected:
            raise ValueError(f"Expected {expected}, got {tok.type}")
        self.pos += 1
        return tok

    def parse(self) -> ValueExpr:
        expr = self._parse_expr()
        if self._peek() is not None:
            raise ValueError("Unexpected trailing tokens")
        return expr

    def _parse_expr(self) -> ValueExpr:
        node = self._parse_term()
        while self._peek() and self._peek().value in ("+", "-"):
            op = self._consume("OP").value
            rhs = self._parse_term()
            node = BinaryExpr(node, op, rhs)
        return node

    def _parse_term(self) -> ValueExpr:
        node = self._parse_factor()
        while self._peek() and self._peek().value in ("*", "/"):
            op = self._consume("OP").value
            rhs = self._parse_factor()
            node = BinaryExpr(node, op, rhs)
        return node

    def _parse_factor(self) -> ValueExpr:
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of expression")

        if tok.type == "NUMBER":
            self._consume()
            return ConstExpr(float(tok.value))

        if tok.type == "METRIC":
            self._consume()
            return MetricExpr(tok.value)

        if tok.type == "LPAREN":
            self._consume("LPAREN")
            node = self._parse_expr()
            self._consume("RPAREN")
            return node

        raise ValueError(f"Unexpected token: {tok.type}")


def parse_value_expr(expr: Union[str, int, float]) -> ValueExpr:
    if isinstance(expr, (int, float)):
        return ConstExpr(float(expr))

    if _METRIC_RE.fullmatch(expr):
        return MetricExpr(expr)

    tokens = _tokenize(expr)
    return _Parser(tokens).parse()


@dataclass(frozen=True)
class Compare:
    lhs: Union[str, int, float, ValueExpr]
    rhs: Union[str, int, float, ValueExpr]
    op: str  # > >= < <=

    def __post_init__(self) -> None:
        object.__setattr__(self, "lhs", _coerce_expr(self.lhs))
        object.__setattr__(self, "rhs", _coerce_expr(self.rhs))

    def evaluate(self, snap: MetricSnapshot) -> PredicateResult:
        l = self.lhs.evaluate(snap)  # noqa: E741
        r = self.rhs.evaluate(snap)

        expr = f"{self.lhs.display(snap)} {self.op} {self.rhs.display(snap)}"
        inputs = {**l.inputs, **r.inputs}

        if l.missing or r.missing:
            return PredicateResult(
                passed=False,
                expression=expr,
                details="Missing metric(s) for comparison.",
                inputs=inputs,
            )

        ok = {
            ">": l.value > r.value,
            ">=": l.value >= r.value,
            "<": l.value < r.value,
            "<=": l.value <= r.value,
        }[self.op]

        details = f"{l.value:.4g} {self.op} {r.value:.4g} => {'PASS' if ok else 'FAIL'}"
        return PredicateResult(ok, expr, details, inputs)


def _coerce_expr(v: Union[str, int, float, ValueExpr]) -> ValueExpr:
    if isinstance(v, (ConstExpr, MetricExpr, BinaryExpr)):
        return v
    return parse_value_expr(v)


@dataclass(frozen=True)
class AnyOf:
    preds: Sequence[Predicate]

    def evaluate(self, snap: MetricSnapshot) -> PredicateResult:
        results = [p.evaluate(snap) for p in self.preds]
        ok = any(r.passed for r in results)
        expr = "ANY(" + " OR ".join(r.expression for r in results) + ")"
        details = "\n".join(r.details for r in results)
        inputs: dict[str, float] = {}
        for r in results:
            inputs.update(r.inputs)
        return PredicateResult(ok, expr, details, inputs)


@dataclass(frozen=True)
class AllOf:
    preds: Sequence[Predicate]

    def evaluate(self, snap: MetricSnapshot) -> PredicateResult:
        results = [p.evaluate(snap) for p in self.preds]
        ok = all(r.passed for r in results)
        expr = "ALL(" + " AND ".join(r.expression for r in results) + ")"
        details = "\n".join(r.details for r in results)
        inputs: dict[str, float] = {}
        for r in results:
            inputs.update(r.inputs)
        return PredicateResult(ok, expr, details, inputs)


@dataclass(frozen=True)
class AlwaysTrue:
    reason: str = "Forced pass (testing / scaffolding)"

    def evaluate(self, snap: MetricSnapshot) -> PredicateResult:
        return PredicateResult(
            passed=True,
            expression="ALWAYS_TRUE",
            details=self.reason,
            inputs={},
        )

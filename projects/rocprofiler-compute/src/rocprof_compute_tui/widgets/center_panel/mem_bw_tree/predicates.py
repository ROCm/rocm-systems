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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, Union

from .ui_model import MetricSnapshot, PredicateResult


class Predicate(Protocol):
    def evaluate(self, snap: MetricSnapshot) -> PredicateResult: ...


def _name(snap: MetricSnapshot, mid: str) -> str:
    meta = snap.meta.get(mid)
    return f"{mid} ({meta.metric_name})" if meta else mid


_SUPPORTED_OPS = {
    "+": lambda a, b: a + b,
    "*": lambda a, b: a * b,
}


def _resolve_value_expr(
    expr: Union[str, int, float],
    snap: MetricSnapshot,
) -> tuple[float | None, str, dict[str, float], bool]:
    """
    Resolves a value expression into:
      value, display_expr, inputs, missing
    """

    # ---- constant ----
    if isinstance(expr, (int, float)):
        v = float(expr)
        return v, f"{v:.2f}", {"<const>": v}, False

    # ---- simple metric id ----
    if expr in snap.values:
        v = snap.values.get(expr)
        return (
            v,
            _name(snap, expr),
            {expr: v if v is not None else float("nan")},
            v is None,
        )

    # ---- binary metric expression (e.g. "11.7.2 + 11.7.1") ----
    tokens = expr.split()
    if len(tokens) != 3 or tokens[1] not in _SUPPORTED_OPS:
        raise ValueError(f"Unsupported value expression: {expr}")

    left, op, right = tokens

    a = snap.values.get(left)
    b = snap.values.get(right)

    inputs = {
        left: a if a is not None else float("nan"),
        right: b if b is not None else float("nan"),
    }

    if a is None or b is None:
        return None, expr, inputs, True

    value = _SUPPORTED_OPS[op](a, b)
    display = f"{_name(snap, left)} {op} {_name(snap, right)}"

    return value, display, inputs, False


@dataclass(frozen=True)
class Compare:
    lhs: Union[str, int, float]
    rhs: Union[str, int, float]
    op: str  # ">", ">=", "<", "<="

    def evaluate(self, snap: MetricSnapshot) -> PredicateResult:
        # ---- resolve both sides ----
        a, lhs_expr, lhs_inputs, lhs_missing = _resolve_value_expr(self.lhs, snap)
        b, rhs_expr, rhs_inputs, rhs_missing = _resolve_value_expr(self.rhs, snap)

        inputs = {
            **lhs_inputs,
            **rhs_inputs,
        }

        expr = f"{lhs_expr} {self.op} {rhs_expr}"

        # ---- missing handling ----
        if lhs_missing or rhs_missing:
            return PredicateResult(
                passed=False,
                expression=expr,
                details="Missing metric(s) for comparison.",
                inputs=inputs,
            )

        # ---- comparison ----
        ok = {
            ">": a > b,
            ">=": a >= b,
            "<": a < b,
            "<=": a <= b,
        }[self.op]

        details = f"{a:.4g} {self.op} {b:.4g} => {'PASS' if ok else 'FAIL'}"
        return PredicateResult(ok, expr, details, inputs)


@dataclass(frozen=True)
class Dominates:
    primary: str
    others: Sequence[str]
    # No constants: "dominates" means primary is greater than all others

    def evaluate(self, snap: MetricSnapshot) -> PredicateResult:
        p = snap.values.get(self.primary)
        inputs = {self.primary: p if p is not None else float("nan")}
        for o in self.others:
            inputs[o] = (
                snap.values.get(o) if snap.values.get(o) is not None else float("nan")
            )

        expr = f"{_name(snap, self.primary)} dominates " + ", ".join(
            _name(snap, o) for o in self.others
        )

        if p is None or any(snap.values.get(o) is None for o in self.others):
            return PredicateResult(
                False, expr, "Missing metric(s) for dominance check.", inputs
            )

        comparisons = [(o, p > snap.values[o]) for o in self.others]  # type: ignore[index]
        ok = all(x[1] for x in comparisons)
        parts = [f"{p:.4g} > {snap.values[o]:.4g} ({o})" for o, passed in comparisons]
        details = "; ".join(parts) + f" => {'PASS' if ok else 'FAIL'}"
        return PredicateResult(ok, expr, details, inputs)


@dataclass(frozen=True)
class RankedHigher:
    metric: str
    than: Sequence[str]

    def evaluate(self, snap: MetricSnapshot) -> PredicateResult:
        m = snap.values.get(self.metric)
        inputs = {self.metric: m if m is not None else float("nan")}
        for o in self.than:
            inputs[o] = (
                snap.values.get(o) if snap.values.get(o) is not None else float("nan")
            )

        expr = f"{_name(snap, self.metric)} ranked higher than " + ", ".join(
            _name(snap, o) for o in self.than
        )

        if m is None or any(snap.values.get(o) is None for o in self.than):
            return PredicateResult(
                False, expr, "Missing metric(s) for ranking check.", inputs
            )

        ok = all(m > snap.values[o] for o in self.than)  # type: ignore[index]
        details = (
            "; ".join([f"{m:.4g} > {snap.values[o]:.4g} ({o})" for o in self.than])
            + f" => {'PASS' if ok else 'FAIL'}"
        )
        return PredicateResult(ok, expr, details, inputs)


@dataclass(frozen=True)
class AllOf:
    preds: Sequence[Predicate]

    def evaluate(self, snap: MetricSnapshot) -> PredicateResult:
        results = [p.evaluate(snap) for p in self.preds]
        ok = all(r.passed for r in results)
        expr = "ALL(" + " AND ".join(r.expression for r in results) + ")"
        details = "\n".join([r.details for r in results])
        inputs: dict[str, float] = {}
        for r in results:
            inputs.update(r.inputs)
        return PredicateResult(ok, expr, details, inputs)


@dataclass(frozen=True)
class AnyOf:
    preds: Sequence[Predicate]

    def evaluate(self, snap: MetricSnapshot) -> PredicateResult:
        results = [p.evaluate(snap) for p in self.preds]
        ok = any(r.passed for r in results)
        expr = "ANY(" + " OR ".join(r.expression for r in results) + ")"
        details = "\n".join([r.details for r in results])
        inputs: dict[str, float] = {}
        for r in results:
            inputs.update(r.inputs)
        return PredicateResult(ok, expr, details, inputs)


@dataclass(frozen=True)
class AlwaysTrue:
    """
    Debug uses
    """

    reason: str = "Forced pass (testing / scaffolding)"

    def evaluate(self, snap: MetricSnapshot) -> PredicateResult:
        return PredicateResult(
            passed=True,
            expression="ALWAYS_TRUE",
            details=self.reason,
            inputs={},  # no metrics involved
        )

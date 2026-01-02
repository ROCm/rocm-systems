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
from typing import Protocol

from .ui_model import MetricSnapshot, PredicateResult


class Predicate(Protocol):
    def evaluate(self, snap: MetricSnapshot) -> PredicateResult: ...


def _name(snap: MetricSnapshot, mid: str) -> str:
    meta = snap.meta.get(mid)
    return f"{mid} ({meta.metric_name})" if meta else mid


@dataclass(frozen=True)
class Compare:
    lhs: str
    rhs: str
    op: str  # ">", ">=", "<", "<="

    def evaluate(self, snap: MetricSnapshot) -> PredicateResult:
        a = snap.values.get(self.lhs)
        b = snap.values.get(self.rhs)
        inputs = {
            self.lhs: a if a is not None else float("nan"),
            self.rhs: b if b is not None else float("nan"),
        }
        expr = f"{_name(snap, self.lhs)} {self.op} {_name(snap, self.rhs)}"

        if a is None or b is None:
            return PredicateResult(
                False, expr, "Missing metric(s) for comparison.", inputs
            )

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

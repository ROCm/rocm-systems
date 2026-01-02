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

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MetricMeta:
    metric_name: str
    panel: str
    table: str
    unit: Optional[str] = None


@dataclass
class MetricSnapshot:
    values: dict[str, float]  # Metric_ID -> value
    meta: dict[str, MetricMeta]  # Metric_ID -> meta
    missing: list[str]  # list[Metric_ID]


@dataclass(frozen=True)
class PredicateResult:
    passed: bool
    expression: str  # short human explanation
    details: str  # verbose detail
    inputs: dict[str, float]  # metric_id -> value used


@dataclass
class NodeEvaluation:
    node_id: str
    passed: bool
    predicate_results: list[PredicateResult]


@dataclass
class DecisionResult:
    active_path: list[str]  # list[node_id]
    node_eval: dict[str, NodeEvaluation]  # node_id -> evaluation

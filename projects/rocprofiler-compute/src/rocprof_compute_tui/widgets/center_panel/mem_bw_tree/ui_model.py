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
from enum import Enum
from typing import Optional


@dataclass(frozen=True)
class MetricMeta:
    metric_name: str
    panel: str
    table: str
    unit: Optional[str] = None


@dataclass
class MetricSnapshot:
    """
    A point-in-time view of metrics for a single kernel dispatch.
    """

    values: dict[str, dict[str, float]]  # Metric_ID -> { column_name -> value }
    meta: dict[str, MetricMeta]  # Metric_ID -> metadata
    missing: list[str]  # Metric_IDs not found in kernel data


@dataclass(frozen=True)
class PredicateResult:
    """
    Result of evaluating a single predicate.
    """

    passed: bool
    expression: str
    details: str
    inputs: dict[str, float]  # Metric_ID -> value used


class EvalStatus(str, Enum):
    """
    Final evaluation status for a node.
    """

    TRUE = "true"  # Predicate evaluated successfully and passed
    FALSE = "false"  # Predicate evaluated successfully and failed
    ERROR = "error"  # Exception during evaluation
    NO_RULE = "no_rule"  # Node reached but no predicate attached


@dataclass
class NodeEvaluation:
    """
    Evaluation result for a single tree node.
    """

    node_id: str
    status: EvalStatus
    predicate_results: list[PredicateResult]
    error: Optional[str] = None


@dataclass
class DecisionResult:
    """
    Output of the DecisionEngine BFS evaluation.
    """

    reached: list[str]  # BFS order of reached nodes
    node_eval: dict[str, NodeEvaluation]  # node_id -> evaluation

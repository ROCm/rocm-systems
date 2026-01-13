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

from collections import deque
from dataclasses import dataclass
from typing import Optional

from rocprof_compute_tui.widgets.mem_bw_tree_core import TreeNode

from .rules import RULES
from .ui_model import DecisionResult, EvalStatus, MetricSnapshot, NodeEvaluation


@dataclass(frozen=True)
class EngineConfig:
    """
    Engine behavior flags.

    block_children_on_error:
        If True, a node evaluating to ERROR prevents its children from
        being evaluated. Default False (per current requirement).
    """

    block_children_on_error: bool = False


class DecisionEngine:
    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        self.config = config or EngineConfig()

        # Pre-index rules by node_id for O(1) lookup during traversal
        self._rule_by_node: dict[str, object] = {r.node_id: r for r in RULES}

    def required_metric_ids(self) -> list[str]:
        """
        Collect all metric IDs required by predicates.
        """
        out: set[str] = set()
        for r in RULES:
            out.update(r.metric_ids)
        return sorted(out)

    def evaluate(
        self,
        snap: MetricSnapshot,
        root: TreeNode,
    ) -> DecisionResult:
        """
        Evaluate the decision tree using BFS traversal.

        Semantics:
          - Root is always evaluated as NO_RULE (for now).
          - Any reached node is highlighted.
          - Predicate TRUE / FALSE / ERROR are all terminal states
            for the node itself.
          - Children are evaluated if the parent is reached.
          - ERROR does not block children unless configured.
        """

        node_eval: dict[str, NodeEvaluation] = {}
        reached: list[str] = []

        q: deque[TreeNode] = deque([root])

        while q:
            node = q.popleft()
            node_id = node.node_id
            reached.append(node_id)

            rule = self._rule_by_node.get(node_id)

            # ----------------------------------------------------------
            # Case 1: No rule attached to this node
            # ----------------------------------------------------------
            if rule is None:
                node_eval[node_id] = NodeEvaluation(
                    node_id=node_id,
                    status=EvalStatus.NO_RULE,
                    predicate_results=[],
                    error=None,
                )

                # Root and non-rule nodes never block traversal
                q.extend(node.children)
                continue

            # ----------------------------------------------------------
            # Case 2: Rule exists, evaluate predicate
            # ----------------------------------------------------------
            try:
                pr = rule.predicate.evaluate(snap)

                status = EvalStatus.TRUE if pr.passed else EvalStatus.FALSE

                node_eval[node_id] = NodeEvaluation(
                    node_id=node_id,
                    status=status,
                    predicate_results=[pr],
                    error=None,
                )

                # Predicate outcome does not gate children
                q.extend(node.children)

            except Exception as e:
                node_eval[node_id] = NodeEvaluation(
                    node_id=node_id,
                    status=EvalStatus.ERROR,
                    predicate_results=[],
                    error=f"{type(e).__name__}: {e}",
                )

                if not self.config.block_children_on_error:
                    q.extend(node.children)

        return DecisionResult(
            reached=reached,
            node_eval=node_eval,
        )

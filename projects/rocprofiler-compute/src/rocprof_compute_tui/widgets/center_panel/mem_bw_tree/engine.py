from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .rules import RULES
from .ui_model import DecisionResult, MetricSnapshot, NodeEvaluation


@dataclass(frozen=True)
class EngineConfig:
    # For beta: “first passing rule wins”
    pass


class DecisionEngine:
    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        self.config = config or EngineConfig()

    def required_metric_ids(self) -> list[str]:
        out: set[str] = set()
        for r in RULES:
            out.update(r.metric_ids)
        return sorted(out)

    def evaluate(
        self, snap: MetricSnapshot, path_lookup: dict[str, list[str]]
    ) -> DecisionResult:
        node_eval: dict[str, NodeEvaluation] = {}
        passing: list[str] = []

        # 1) Evaluate all rules
        for rule in RULES:
            pr = rule.predicate.evaluate(snap)

            node_eval[rule.node_id] = NodeEvaluation(
                node_id=rule.node_id,
                passed=pr.passed,
                predicate_results=[pr],
            )

            if pr.passed:
                passing.append(rule.node_id)

        # 2) Choose deepest passing node (longest path)
        active_path: list[str] = []
        if passing:
            # Guard against missing PATHS entries
            best_node = max(
                passing,
                key=lambda nid: len(path_lookup.get(nid, [])),
            )
            active_path = path_lookup.get(best_node, [best_node])

        return DecisionResult(
            active_path=active_path,
            node_eval=node_eval,
        )

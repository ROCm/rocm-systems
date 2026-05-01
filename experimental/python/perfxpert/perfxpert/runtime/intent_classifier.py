"""Rule-based intent classifier for air-gap routing (spec §2, review C2).

This is a thin middleware wrapper around perfxpert.tools.intent.classify.
Lives in runtime/ because it is ALWAYS consulted — even in
LLM-enabled mode the deterministic verdict is computed first and the LLM
may only refine it. This preserves the air-gap parity invariant (§5):
handoff targets are byte-identical with and without LLM.

Used by agents/root.py:
    verdict = classify_intent(user_query)
    if verdict.intent == "analyze": handoff to Analysis
    elif verdict.intent == "verify": handoff to Correctness
    elif verdict.intent == "optimize": handoff to Recommendation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from perfxpert.tools import intent as intent_tool


Intent = Literal["analyze", "optimize", "verify", "explain", "help"]


@dataclass(frozen=True)
class IntentVerdict:
    intent: Intent
    confidence: float
    matched_pattern: Optional[str] = None
    warning: Optional[str] = None


def classify_intent(user_query: str) -> IntentVerdict:
    """Classify a user query; returns a frozen IntentVerdict.

    Air-gap mode and LLM mode BOTH call this first — the deterministic
    verdict is the source of truth for handoff routing.
    """
    r = intent_tool.classify(user_query or "")
    return IntentVerdict(
        intent=r["intent"],
        confidence=r.get("confidence", 0.0),
        matched_pattern=r.get("matched_pattern"),
        warning=r.get("warning"),
    )


__all__ = ["Intent", "IntentVerdict", "classify_intent"]

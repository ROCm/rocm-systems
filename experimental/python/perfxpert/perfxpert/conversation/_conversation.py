"""Conversation state + compaction + cost estimation.

A turn = one user message followed by one assistant message. When the
turn count crosses `compaction_every_n_turns`, older turns are replaced
by a single system summary message; the most recent
`compaction_keep_recent_turns` turns are preserved verbatim.

API keys are never serialized by to_dict(); callers must re-supply them
to from_dict() after deserialization.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _estimate_tokens(text: str) -> int:
    # Rough heuristic: 4 chars/token for English text + code mix.
    return max(0, len(text) // 4)


class Conversation:
    """In-memory conversation state with LLM compaction + cost estimation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        compaction_every_n_turns: int = 20,
        compaction_keep_recent_turns: int = 5,
    ) -> None:
        self._api_key = api_key  # held in memory only, never serialized
        self._compaction_every_n = compaction_every_n_turns
        self._compaction_keep = compaction_keep_recent_turns
        self._messages: List[Dict[str, Any]] = []
        self._turn_count: int = 0

    # ------------------------------------------------------------------
    # Message manipulation
    # ------------------------------------------------------------------

    def add_user(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        self._messages.append({"role": "assistant", "content": content})
        self._turn_count += 1
        self._maybe_compact()

    @property
    def messages(self) -> List[Dict[str, Any]]:
        return list(self._messages)

    @property
    def turn_count(self) -> int:
        return self._turn_count

    # ------------------------------------------------------------------
    # Serialization (api_key always excluded)
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "messages": list(self._messages),
            "turn_count": self._turn_count,
            "compaction_every_n_turns": self._compaction_every_n,
            "compaction_keep_recent_turns": self._compaction_keep,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *, api_key: Optional[str] = None) -> "Conversation":
        c = cls(
            api_key=api_key,
            compaction_every_n_turns=int(data.get("compaction_every_n_turns", 20)),
            compaction_keep_recent_turns=int(data.get("compaction_keep_recent_turns", 5)),
        )
        c._messages = list(data.get("messages", []))
        c._turn_count = int(data.get("turn_count", 0))
        return c

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------

    def _maybe_compact(self) -> None:
        if self._compaction_every_n <= 0:
            return
        if self._turn_count == 0:
            return
        if self._turn_count % self._compaction_every_n != 0:
            return
        keep_messages = self._compaction_keep * 2
        if len(self._messages) <= keep_messages:
            return
        to_summarize = self._messages[:-keep_messages] if keep_messages else self._messages[:]
        summary = (
            f"[summary of prior {len(to_summarize)} messages across "
            f"{self._turn_count - self._compaction_keep} turns — details compacted]"
        )
        preserved = self._messages[-keep_messages:] if keep_messages else []
        self._messages = [{"role": "system", "content": summary}] + preserved

    # ------------------------------------------------------------------
    # Cost estimation
    # ------------------------------------------------------------------

    def dry_run_cost_estimate(
        self,
        model: str,
        input_price_per_mtok: float,
        output_price_per_mtok: float,
        avg_output_tokens: int = 500,
    ) -> Dict[str, Any]:
        input_chars = sum(len(m.get("content", "")) for m in self._messages)
        input_tokens = _estimate_tokens("".join(m.get("content", "") for m in self._messages))
        cost_in = input_tokens * input_price_per_mtok / 1_000_000
        cost_out = avg_output_tokens * output_price_per_mtok / 1_000_000
        return {
            "model": model,
            "estimated_input_tokens": input_tokens,
            "estimated_output_tokens": avg_output_tokens,
            "estimated_cost_usd": round(cost_in + cost_out, 6),
            "input_chars": input_chars,
        }


__all__ = ["Conversation"]

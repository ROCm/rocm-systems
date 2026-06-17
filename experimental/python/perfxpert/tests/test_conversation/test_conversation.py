"""Tests for perfxpert.conversation."""

import pytest

from perfxpert.conversation import Conversation


def test_add_user_and_assistant():
    c = Conversation()
    c.add_user("hi")
    c.add_assistant("hello")
    msgs = c.messages
    assert len(msgs) == 2
    assert msgs[0] == {"role": "user", "content": "hi"}
    assert msgs[1] == {"role": "assistant", "content": "hello"}


def test_messages_returns_copy():
    c = Conversation()
    c.add_user("hi")
    m = c.messages
    m.append({"role": "user", "content": "mutate"})
    assert len(c.messages) == 1  # unaffected


def test_turn_count():
    c = Conversation()
    assert c.turn_count == 0
    c.add_user("a")
    c.add_assistant("b")
    assert c.turn_count == 1
    c.add_user("c")
    c.add_assistant("d")
    assert c.turn_count == 2


def test_api_key_never_serialized():
    c = Conversation(api_key="sk-secret")
    c.add_user("hi")
    c.add_assistant("ok")
    payload = c.to_dict()
    assert "api_key" not in payload
    assert "sk-secret" not in repr(payload)


def test_round_trip_preserves_messages():
    c = Conversation(api_key="sk-a")
    c.add_user("ping")
    c.add_assistant("pong")
    payload = c.to_dict()
    restored = Conversation.from_dict(payload, api_key="sk-b")
    assert restored.messages == c.messages
    assert restored.turn_count == c.turn_count


def test_compaction_triggers_at_threshold():
    c = Conversation(compaction_every_n_turns=3, compaction_keep_recent_turns=1)
    for i in range(5):
        c.add_user(f"u{i}")
        c.add_assistant(f"a{i}")
    msgs = c.messages
    # After compaction: summary message + 1 recent turn (2 messages) = 3 total
    assert any(m.get("role") == "system" and "summary" in m.get("content", "").lower() for m in msgs)
    assert len(msgs) < 10


def test_no_compaction_below_threshold():
    c = Conversation(compaction_every_n_turns=100)
    for i in range(3):
        c.add_user(f"u{i}")
        c.add_assistant(f"a{i}")
    assert len(c.messages) == 6  # untouched


def test_dry_run_cost_estimate_zero_tokens():
    c = Conversation()
    est = c.dry_run_cost_estimate(
        model="claude-3-5-sonnet-20241022",
        input_price_per_mtok=3.0,
        output_price_per_mtok=15.0,
        avg_output_tokens=500,
    )
    assert est["estimated_input_tokens"] == 0
    assert est["estimated_cost_usd"] >= 0.0


def test_dry_run_cost_with_messages():
    c = Conversation()
    c.add_user("x" * 400)
    c.add_assistant("y" * 400)
    est = c.dry_run_cost_estimate(
        model="claude-3-5-sonnet-20241022",
        input_price_per_mtok=3.0,
        output_price_per_mtok=15.0,
        avg_output_tokens=500,
    )
    # ~200 input tokens (800 chars / 4), 500 output tokens
    assert est["estimated_input_tokens"] > 0
    assert est["estimated_output_tokens"] == 500
    assert est["estimated_cost_usd"] > 0


def test_empty_conversation_to_dict():
    c = Conversation()
    payload = c.to_dict()
    assert payload["messages"] == []
    assert payload["turn_count"] == 0

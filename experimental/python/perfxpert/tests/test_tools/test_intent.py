"""Tests for perfxpert.tools.intent — air-gap routing without LLM."""

import pytest

from perfxpert.tools import intent
from perfxpert.tools._class import ToolClass


@pytest.mark.parametrize("query, expected", [
    ("Why is my kernel slow?", "analyze"),
    ("analyze this trace", "analyze"),
    ("what's the bottleneck", "analyze"),
    ("how do I fix the memcpy overhead", "optimize"),
    ("suggest optimizations for my matmul kernel", "optimize"),
    ("did my patch help", "verify"),
    ("is this better than before", "verify"),
    ("compare run 1 vs run 2", "verify"),
    ("explain what MFMA means", "explain"),
    ("what is the ridge point", "explain"),
    ("help", "help"),
    ("huh", "analyze"),  # ambiguous default
])
def test_intent_classification(query, expected):
    r = intent.classify(query)
    assert r["intent"] == expected


def test_returns_confidence():
    r = intent.classify("analyze my trace")
    assert 0 <= r["confidence"] <= 1


def test_ambiguous_gets_analyze_with_warning():
    r = intent.classify("uh?")
    assert r["intent"] == "analyze"
    assert r["confidence"] < 0.5
    assert "ambiguous" in r.get("warning", "").lower() or "default" in r.get("warning", "").lower()


def test_is_read_only_class():
    assert intent.classify.__tool_class__ == ToolClass.READ_ONLY

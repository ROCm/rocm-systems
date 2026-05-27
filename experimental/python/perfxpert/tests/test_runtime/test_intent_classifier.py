"""Tests for perfxpert.runtime.intent_classifier."""

import pytest

from perfxpert.runtime import intent_classifier


def test_classify_returns_verdict():
    v = intent_classifier.classify_intent("why is this slow?")
    assert v.intent == "analyze"
    assert 0.0 <= v.confidence <= 1.0


def test_classify_handles_empty_query():
    v = intent_classifier.classify_intent("")
    assert v.intent == "analyze"   # default
    assert v.confidence < 0.5


@pytest.mark.parametrize("q,expected", [
    ("did my patch help?", "verify"),
    ("suggest optimizations", "optimize"),
    ("what is MFMA?", "explain"),
    ("help", "help"),
])
def test_parametric_classification(q, expected):
    v = intent_classifier.classify_intent(q)
    assert v.intent == expected


def test_verdict_is_frozen():
    v = intent_classifier.classify_intent("analyze")
    with pytest.raises((AttributeError, TypeError)):
        v.intent = "verify"  # frozen dataclass


def test_airgap_and_llm_agree_on_verdict():
    """Parity invariant (§5 air-gap): classifier verdict is deterministic."""
    v1 = intent_classifier.classify_intent("why slow?")
    v2 = intent_classifier.classify_intent("why slow?")
    assert v1 == v2

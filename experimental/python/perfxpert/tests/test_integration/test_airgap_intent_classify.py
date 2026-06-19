"""Airgap-mode intent routing (spec C2): the rule-based intent classifier
MUST return the same intent label for the same query, regardless of LLM mode.
"""

import os
from contextlib import contextmanager

import pytest

from perfxpert.runtime.intent_classifier import classify_intent


QUERIES_AND_EXPECTED = [
    ("Why is my matmul slow?", "analyze"),
    ("How do I make this faster?", "analyze"),
    ("Did my edit regress perf?", "analyze"),
    ("What does MFMA mean?", "explain"),
    ("What can I do with rocpd?", "help"),
    # edge cases
    ("profile ./app", "analyze"),
    ("fix kernel xxx", "analyze"),
    ("check if results still match baseline", "analyze"),
]


@contextmanager
def airgap_env(on: bool):
    prev = os.environ.get("PERFXPERT_AIRGAP")
    if on:
        os.environ["PERFXPERT_AIRGAP"] = "1"
    else:
        os.environ.pop("PERFXPERT_AIRGAP", None)
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("PERFXPERT_AIRGAP", None)
        else:
            os.environ["PERFXPERT_AIRGAP"] = prev


@pytest.mark.parametrize("query,expected", QUERIES_AND_EXPECTED)
def test_intent_identical_with_and_without_llm(query: str, expected: str) -> None:
    with airgap_env(on=False):
        a = classify_intent(query)
    with airgap_env(on=True):
        b = classify_intent(query)
    assert a.intent == b.intent == expected, f"Mismatch: with_llm={a.intent!r}, airgap={b.intent!r}, expected={expected!r}"

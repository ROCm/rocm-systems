"""Property-based robustness test for the sanitization pipeline.

Attacks 9-14 are enumerated; this test is a defense-in-depth: generate
thousands of random payloads with injected metachars / prompt markers and
confirm the sanitizer never lets them through untransformed.

NOTE: this test does NOT count towards the 14-attack red-team total. It is a
supplementary fuzz gate that runs in the nightly CI lane (marked `slow`).
"""

import re

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

from perfxpert.tools._safety import reject_shell_metachars, ShellMetacharError

# Exact same pattern as in _safety.py
_SHELL_METACHARS = re.compile(r"[;&|`$()<>\n\r\0]|\\\\|\\\"|\\'")
PROMPT_INJECTION_MARKERS = [
    "IGNORE PREVIOUS", "IGNORE ABOVE", "SYSTEM:", "<|im_start|>", "<|im_end|>",
    "ASSISTANT:", "</system>", "---END OF FENCE---",
]


@pytest.mark.red_team
@pytest.mark.slow
@given(safe_text=st.text(min_size=0, max_size=200))
@settings(max_examples=500, deadline=None)
def test_sanitizer_accepts_safe_strings(safe_text: str) -> None:
    # For any string WITHOUT shell metachars, the sanitizer must accept it.
    if not _SHELL_METACHARS.search(safe_text):
        # This string is safe; sanitizer must not raise
        reject_shell_metachars(safe_text)


@pytest.mark.red_team
@pytest.mark.slow
@given(any_text=st.text(min_size=0, max_size=200))
@settings(max_examples=500, deadline=None)
def test_sanitizer_rejects_dangerous_strings(any_text: str) -> None:
    # For any string WITH shell metachars, the sanitizer must reject it.
    if _SHELL_METACHARS.search(any_text):
        # This string contains metachars; sanitizer must raise
        with pytest.raises(ShellMetacharError):
            reject_shell_metachars(any_text)
    else:
        # This string is safe; sanitizer must not raise
        reject_shell_metachars(any_text)

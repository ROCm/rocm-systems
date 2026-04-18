"""Parity: primary recommendation type (category) must match.

Signal is skipped (xfail-style pytest.skip()) when one path returns recs and
the other doesn't — the agentic path by design returns no recommendations in
the non-LLM wrapper.  Both-None and both-non-None-equal are passing.
"""

import pytest

from .diff_report import summarize_for_failure_message
from .parity_runner import ParityRunner


@pytest.mark.parity
def test_primary_rec_type_agrees(parity_runner: ParityRunner, fx) -> None:
    dual = parity_runner.run_both_paths(fx)
    result = dual.agree_rec_type()
    if result is None:
        pytest.skip(
            f"Fixture '{fx.id}': rec_type signal skipped — asymmetric None "
            f"(old={dual.old.primary_rec_type!r}, new={dual.new.primary_rec_type!r}). "
            "Agentic path does not populate recommendations without LLM. "
            "Signal is not evaluable for parity; bottleneck agreement is the ship gate."
        )
    assert result, summarize_for_failure_message(dual)

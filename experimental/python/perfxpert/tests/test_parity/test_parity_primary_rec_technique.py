"""Parity: primary recommendation technique (specific optimization) must match.

This is the strictest of the three parity signals — allows different-type
agreement (e.g. both say "compute") without same-technique agreement (e.g.
old says "mfma_enablement", new says "vgpr_reduction_compute_bound").
The ≥95% aggregate threshold accounts for the inherent LLM-in-loop variance
in technique selection.

Signal is skipped when one path returns recs and the other doesn't (asymmetric
None) — see test_parity_primary_rec_type.py for explanation.
"""

import pytest

from .diff_report import summarize_for_failure_message
from .parity_runner import ParityRunner


@pytest.mark.parity
def test_primary_rec_technique_agrees(parity_runner: ParityRunner, fx) -> None:
    dual = parity_runner.run_both_paths(fx)
    result = dual.agree_rec_technique()
    if result is None:
        pytest.skip(
            f"Fixture '{fx.id}': rec_technique signal skipped — asymmetric None "
            f"(old={dual.old.primary_rec_technique!r}, new={dual.new.primary_rec_technique!r}). "
            "Agentic path does not populate recommendations without LLM."
        )
    assert result, summarize_for_failure_message(dual)

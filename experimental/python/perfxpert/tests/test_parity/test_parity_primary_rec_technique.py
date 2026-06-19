"""Parity: primary recommendation technique (specific optimization) must match.

This is the strictest of the three parity signals — allows different-type
agreement (e.g. both say "compute") without same-technique agreement (e.g.
old says "mfma_enablement", new says "vgpr_reduction_compute_bound").
The ≥95% aggregate threshold accounts for the inherent LLM-in-loop variance
in technique selection.
"""

import pytest

from .diff_report import summarize_for_failure_message
from .parity_runner import ParityRunner


@pytest.mark.parity
def test_primary_rec_technique_agrees(parity_runner: ParityRunner, fx) -> None:
    result = parity_runner.run_fixture(fx)
    assert result.agree_rec_technique(), summarize_for_failure_message(result)

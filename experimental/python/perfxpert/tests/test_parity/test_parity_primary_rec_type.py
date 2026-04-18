"""Parity: primary recommendation type (category) must match."""

import pytest

from .diff_report import summarize_for_failure_message
from .parity_runner import ParityRunner


@pytest.mark.parity
def test_primary_rec_type_agrees(parity_runner: ParityRunner, fx) -> None:
    dual = parity_runner.run_both_paths(fx)
    assert dual.agree_rec_type(), summarize_for_failure_message(dual)

"""Parity: primary recommendation type (category) must match."""

import pytest

from .diff_report import summarize_for_failure_message
from .parity_runner import ParityRunner


@pytest.mark.parity
def test_primary_rec_type_agrees(parity_runner: ParityRunner, fx) -> None:
    result = parity_runner.run_fixture(fx)
    assert result.agree_rec_type(), summarize_for_failure_message(result)

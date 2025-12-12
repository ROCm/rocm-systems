"""Parity: primary bottleneck classification must match the fixture contract.

Per spec §7 Go/No-Go table: ≥ 95% agreement aggregated across all fixtures.
This file records per-fixture agreement; the aggregate lives in
test_parity_aggregate.py.
"""

import pytest

from .diff_report import summarize_for_failure_message
from .parity_runner import ParityRunner


@pytest.mark.parity
def test_bottleneck_classification_agrees(parity_runner: ParityRunner, fx) -> None:
    result = parity_runner.run_fixture(fx)
    assert result.agree_bottleneck(), summarize_for_failure_message(result)

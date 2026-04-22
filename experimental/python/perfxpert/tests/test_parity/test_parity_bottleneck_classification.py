"""Parity: primary bottleneck classification must match between old and new paths.

Per spec §7 Go/No-Go table: ≥ 95% agreement aggregated across all fixtures.
This file records per-fixture agreement; the aggregate lives in
test_parity_aggregate.py.
"""

import pytest

from .diff_report import summarize_for_failure_message
from .parity_runner import ParityRunner


@pytest.mark.parity
def test_bottleneck_classification_agrees(parity_runner: ParityRunner, fx) -> None:
    dual = parity_runner.run_both_paths(fx)
    if fx.expected_bottleneck is not None:
        # Additional sanity: at least one path gets the expected answer.
        paths_hit_expected = [
            dual.old.primary_bottleneck == fx.expected_bottleneck,
            dual.new.primary_bottleneck == fx.expected_bottleneck,
        ]
        assert any(paths_hit_expected), (
            f"Neither path matches expected bottleneck "
            f"{fx.expected_bottleneck!r}; investigate fixture before parity"
        )

    # The actual parity assertion:
    assert dual.agree_bottleneck(), summarize_for_failure_message(dual)

"""Parity: primary recommendation type (category) must match."""

import pytest

from .diff_report import summarize_for_failure_message
from .parity_runner import ParityRunner


@pytest.mark.parity
def test_primary_rec_type_agrees(parity_runner: ParityRunner, fx) -> None:
    dual = parity_runner.run_both_paths(fx)
    if fx.expected_rec_type is not None:
        paths_hit_expected = [
            dual.old.primary_rec_type == fx.expected_rec_type,
            dual.new.primary_rec_type == fx.expected_rec_type,
        ]
        assert any(paths_hit_expected), (
            f"Neither path matches expected recommendation type "
            f"{fx.expected_rec_type!r}; investigate fixture before parity"
        )
    assert dual.agree_rec_type(), summarize_for_failure_message(dual)

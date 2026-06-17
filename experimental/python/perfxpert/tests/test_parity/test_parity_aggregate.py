"""Aggregate parity test — enforces the Week-5 Go/No-Go ≥ 95% threshold.

Fails if fewer than 95% of (fixture × signal) combinations agree.
3 signals × N fixtures = 3N comparisons; spec target ≥ 0.95.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .diff_report import field_level_diffs
from .fixtures_inventory import available_parity_fixtures
from .parity_runner import ParityRunner

SNAPSHOTS_DIR = Path(__file__).parent / "parity_snapshots"

MIN_REQUIRED_FIXTURES = 3
MIN_ACTIONABLE_FIXTURES = 3
MIN_DISTINCT_BOTTLENECKS = 2
MIN_DISTINCT_REC_TYPES = 2
MIN_DISTINCT_TECHNIQUES = 2
PARITY_THRESHOLD = 0.95


@pytest.mark.parity
def test_minimum_viable_fixture_floor() -> None:
    """Week-5 PR CI must exercise non-trivial parity signals, not placeholder parity."""
    fixtures = available_parity_fixtures()
    assert len(fixtures) >= MIN_REQUIRED_FIXTURES, (
        f"Parity suite requires at least {MIN_REQUIRED_FIXTURES} fixtures in CI; "
        f"found {len(fixtures)}. Add hermetic parity fixtures instead of skipping."
    )
    actionable = [fx for fx in fixtures if fx.expected_rec_type is not None]
    assert len(actionable) >= MIN_ACTIONABLE_FIXTURES, (
        f"Parity suite requires at least {MIN_ACTIONABLE_FIXTURES} fixtures with "
        "non-null recommendation categories."
    )

    bottlenecks = {
        fx.expected_bottleneck
        for fx in fixtures
        if fx.expected_bottleneck is not None
    }
    assert len(bottlenecks) >= MIN_DISTINCT_BOTTLENECKS, (
        "Parity floor is too weak: present fixtures do not cover enough distinct "
        f"bottleneck classes ({sorted(bottlenecks)})."
    )

    rec_types = {
        fx.expected_rec_type
        for fx in fixtures
        if fx.expected_rec_type is not None
    }
    assert len(rec_types) >= MIN_DISTINCT_REC_TYPES, (
        "Parity floor is too weak: present fixtures do not cover enough distinct "
        f"recommendation categories ({sorted(rec_types)})."
    )

    techniques = {
        fx.expected_rec_technique
        for fx in fixtures
        if fx.expected_rec_technique is not None
    }
    assert len(techniques) >= MIN_DISTINCT_TECHNIQUES, (
        "Parity floor is too weak: present fixtures do not cover enough distinct "
        f"optimization techniques ({sorted(techniques)})."
    )


@pytest.mark.parity
def test_runner_runs_every_inventory_fixture_without_crash() -> None:
    """Contract: ParityRunner produces a result for every present fixture."""
    runner = ParityRunner()
    fixtures = available_parity_fixtures()
    for fx in fixtures:
        result = runner.run_fixture(fx)
        assert result.observed is not None


@pytest.mark.parity
def test_aggregate_agreement_at_or_above_95pct() -> None:
    runner = ParityRunner()
    fixtures = available_parity_fixtures()
    if not fixtures:
        pytest.skip("No fixtures available (Phase 13 backfill pending)")

    total_signals = 0
    agreements = 0
    per_fixture_report = []

    for fx in fixtures:
        result = runner.run_fixture(fx)
        signals = result.agreements()
        total_signals += len(signals)
        agreements += sum(1 for v in signals.values() if v)
        per_fixture_report.append(
            {
                "fixture_id": result.fixture_id,
                "agreements": signals,
                "diffs": field_level_diffs(result),
                "duration_s": result.observed.duration_s,
            }
        )

    agreement_rate = agreements / total_signals if total_signals else 0.0

    # Write the aggregate snapshot for exit_dashboard consumption
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    (SNAPSHOTS_DIR / "_aggregate.json").write_text(
        json.dumps(
            {
                "total_signals": total_signals,
                "agreements": agreements,
                "agreement_rate": round(agreement_rate, 4),
                "threshold": PARITY_THRESHOLD,
                "per_fixture": per_fixture_report,
            },
            indent=2,
        )
    )

    assert agreement_rate >= PARITY_THRESHOLD, (
        f"Parity failed: {agreements}/{total_signals} "
        f"({agreement_rate:.1%}) < required {PARITY_THRESHOLD:.0%}\n"
        f"Per-fixture diffs written to {SNAPSHOTS_DIR / '_aggregate.json'}"
    )

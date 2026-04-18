"""Aggregate parity test — enforces the Week-5 Go/No-Go ≥ 95% threshold.

Fails if fewer than 95% of (fixture × signal) combinations agree.
3 signals × N fixtures = 3N comparisons; spec target ≥ 0.95.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .diff_report import field_level_diffs
from .fixtures_inventory import available_fixtures
from .parity_runner import ParityRunner

SNAPSHOTS_DIR = Path(__file__).parent / "parity_snapshots"

PARITY_THRESHOLD = 0.95


@pytest.mark.parity
def test_runner_runs_every_inventory_fixture_without_crash() -> None:
    """Contract: ParityRunner produces DualResult for every present fixture."""
    runner = ParityRunner()
    fixtures = available_fixtures()
    if len(fixtures) < 10:
        pytest.skip(
            f"Parity suite needs ≥ 10 fixtures (spec §7 exit criteria); got {len(fixtures)}. "
            f"Fixtures backfilled in Phase 13."
        )
    for fx in fixtures:
        dual = runner.run_both_paths(fx)
        assert dual.old is not None
        assert dual.new is not None


@pytest.mark.parity
def test_aggregate_agreement_at_or_above_95pct() -> None:
    runner = ParityRunner()
    fixtures = available_fixtures()
    if not fixtures:
        pytest.skip("No fixtures available (Phase 13 backfill pending)")

    total_signals = 0
    agreements = 0
    per_fixture_report = []

    for fx in fixtures:
        dual = runner.run_both_paths(fx)
        signals = dual.agreements()  # {bottleneck, rec_type, rec_technique}
        total_signals += len(signals)
        agreements += sum(1 for v in signals.values() if v)
        per_fixture_report.append(
            {
                "fixture_id": dual.fixture_id,
                "agreements": signals,
                "diffs": field_level_diffs(dual),
                "old_duration_s": dual.old.duration_s,
                "new_duration_s": dual.new.duration_s,
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

"""Aggregate parity test — enforces the Phase-7 Go/No-Go ≥ 95% threshold.

Ship gate applies ONLY to PMC-enriched fixtures (has_pmc_counters=True).
Trace-only fixtures are excluded from the gate but included in informational
output so the divergence is visible.

Rationale: trace-only fixtures expose a known policy divergence between paths —
the legacy path uses heuristics; the agentic path correctly returns
data_insufficient.  Once re-profiled with --pmc (Phase 7 Part C) these will
be promoted to the PMC subset.

3 signals × N_pmc fixtures = 3*N_pmc comparisons; spec target ≥ 0.95 on PMC
subset.  Overall agreement (informational) is also computed and written to the
snapshot for visibility.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .diff_report import field_level_diffs
from .fixtures_inventory import available_fixtures, pmc_fixtures, trace_only_fixtures
from .parity_runner import ParityRunner

SNAPSHOTS_DIR = Path(__file__).parent / "parity_snapshots"

PARITY_THRESHOLD = 0.95


@pytest.mark.parity
def test_runner_runs_every_inventory_fixture_without_crash() -> None:
    """Contract: ParityRunner produces DualResult for every present fixture."""
    runner = ParityRunner()
    fixtures = available_fixtures()
    if not fixtures:
        pytest.skip("No fixtures available (Phase 13 backfill pending)")
    for fx in fixtures:
        dual = runner.run_both_paths(fx)
        assert dual.old is not None
        assert dual.new is not None


@pytest.mark.parity
def test_aggregate_agreement_at_or_above_95pct() -> None:
    """Ship-gated aggregate: ≥95% parity agreement on PMC-enriched fixtures only.

    Trace-only fixtures are excluded from the assertion but recorded in the
    snapshot as informational.  They are expected to diverge because the legacy
    path uses heuristics while the agentic path returns data_insufficient.
    """
    runner = ParityRunner()
    pmc_fxs = pmc_fixtures()
    trace_fxs = trace_only_fixtures()

    if not pmc_fxs:
        pytest.skip(
            "No PMC fixtures available — parity gate requires at least 1 PMC fixture. "
            "Re-profile with --pmc (Phase 7 Part C) or check fixtures_inventory.py."
        )

    # --- PMC subset (ship-gated) ---
    pmc_total = 0
    pmc_agreements = 0
    pmc_report = []

    for fx in pmc_fxs:
        dual = runner.run_both_paths(fx)
        signals = dual.agreements()  # {bottleneck, rec_type, rec_technique}
        pmc_total += len(signals)
        pmc_agreements += sum(1 for v in signals.values() if v)
        pmc_report.append(
            {
                "fixture_id": dual.fixture_id,
                "has_pmc_counters": True,
                "agreements": signals,
                "diffs": field_level_diffs(dual),
                "old_duration_s": dual.old.duration_s,
                "new_duration_s": dual.new.duration_s,
            }
        )

    # --- Trace-only subset (informational only; not gated) ---
    trace_total = 0
    trace_agreements = 0
    trace_report = []

    for fx in trace_fxs:
        dual = runner.run_both_paths(fx)
        signals = dual.agreements()
        trace_total += len(signals)
        trace_agreements += sum(1 for v in signals.values() if v)
        trace_report.append(
            {
                "fixture_id": dual.fixture_id,
                "has_pmc_counters": False,
                "note": (
                    "Excluded from gate: legacy uses heuristics, agentic returns "
                    "data_insufficient.  Policy divergence — not a bug. "
                    "Will be promoted once re-profiled with --pmc."
                ),
                "agreements": signals,
                "diffs": field_level_diffs(dual),
                "old_duration_s": dual.old.duration_s,
                "new_duration_s": dual.new.duration_s,
            }
        )

    pmc_rate = pmc_agreements / pmc_total if pmc_total else 0.0
    overall_total = pmc_total + trace_total
    overall_agreements = pmc_agreements + trace_agreements
    overall_rate = overall_agreements / overall_total if overall_total else 0.0

    # Write the aggregate snapshot for exit_dashboard consumption.
    # Both numbers are written: pmc_subset_agreement (ship-gated) and
    # overall_agreement (informational — shows the divergence).
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    (SNAPSHOTS_DIR / "_aggregate.json").write_text(
        json.dumps(
            {
                "pmc_subset_agreement": {
                    "total_signals": pmc_total,
                    "agreements": pmc_agreements,
                    "agreement_rate": round(pmc_rate, 4),
                    "threshold": PARITY_THRESHOLD,
                    "ship_gated": True,
                    "fixtures": pmc_report,
                },
                "overall_agreement": {
                    "total_signals": overall_total,
                    "agreements": overall_agreements,
                    "agreement_rate": round(overall_rate, 4),
                    "ship_gated": False,
                    "note": (
                        "Includes trace-only fixtures.  Low overall rate is expected: "
                        "legacy uses heuristics on trace-only data; agentic returns "
                        "data_insufficient (correct). Will improve after --pmc re-profile."
                    ),
                    "fixtures": pmc_report + trace_report,
                },
            },
            indent=2,
        )
    )

    print(
        f"\nParity summary:\n"
        f"  PMC subset (ship-gated):  {pmc_agreements}/{pmc_total} "
        f"({pmc_rate:.1%}) — threshold ≥{PARITY_THRESHOLD:.0%}\n"
        f"  Overall (informational):  {overall_agreements}/{overall_total} "
        f"({overall_rate:.1%}) — {len(trace_fxs)} trace-only fixture(s) excluded from gate"
    )

    assert pmc_rate >= PARITY_THRESHOLD, (
        f"Parity FAILED on PMC subset: {pmc_agreements}/{pmc_total} "
        f"({pmc_rate:.1%}) < required {PARITY_THRESHOLD:.0%}\n"
        f"Per-fixture diffs written to {SNAPSHOTS_DIR / '_aggregate.json'}"
    )

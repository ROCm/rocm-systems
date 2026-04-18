# Phase 5 Runbook — Parity + Red-Team + Go/No-Go

## Context

Phase 5 is the **audit gate** before Phase 6 deletion. Every exit criterion in
the spec §7 Go/No-Go table has a corresponding test suite here. If any gate
slips, Phase 6 is blocked until fixed.

## How to run the full gate locally

```bash
cd experimental/python/perfxpert

# 1. Install dev deps
pip install -e '.[dev]'

# 2. Run the parity suite
pytest -m parity -v

# 3. Run the red-team suite (14 attacks)
pytest -m red_team -v

# 4. Run the regression-gate false-positive suite
pytest -m regression_gate -v

# 5. Run the airgap parity test
pytest tests/test_integration/test_airgap.py tests/test_integration/test_airgap_intent_classify.py -v

# 6. Generate the exit dashboard
python scripts/exit_dashboard.py --output exit_dashboard.json --render
```

## How to interpret the dashboard

Three verdicts:

| Verdict | Meaning | Action |
|---------|---------|--------|
| GO | All 9 thresholds met (including nightly) | Proceed to Phase 6 |
| PARTIAL (pending) | All non-nightly green; nightly inputs not yet collected | Wait for nightly CI; do NOT merge Phase 6 yet |
| NO-GO | One or more PR-lane thresholds failed | Fix failing area, re-run |

## Interpreting a NO-GO

Each metric maps to a failure region:

| Metric | Owner (fix PR against) |
|--------|------------------------|
| `parity_agreement_rate < 0.95` | Phases 3/4 — new-path diverges from old |
| `red_team_pass_count < 14` | Phase 3/4 — sanitizer, allowlist, or gate_cascade bug |
| `airgap_identical_rate < 1.0` | Phase 3 — intent_classifier or gate_cascade dependency on LLM |
| `regression_gate_false_positive_rate > 0.05` | Phase 3 — regression.compare_runs threshold tuning |
| `per_agent_narrow_scope_violations > 0` | Phase 3 — fence exceeded 400 lines / agent exceeded 5 tools |
| `tool_class_split_violations > 0` | Phase 4 — an EXECUTION tool was registered with MCP |

## Adding a new parity fixture

1. Generate the synthetic DB (or record a real rocprofv3 run) in `tests/fixtures/<id>.db`.
2. Add an entry to `tests/test_parity/fixtures_inventory.py`.
3. Re-run `pytest -m parity -v`.

## Adding a new attack

The 14-attack count is normative (spec §5.8 + §7). Adding a 15th attack
requires an RFC + normative update to the spec. Prefer:

- Parameterizing an existing attack's test (more vectors, same attack class).
- Adding a fuzz property to `test_sanitizer_fuzz.py` (supplementary, not counted).

## Fixing a red-team failure

1. Locate which attack regressed: `cat tests/test_red_team/_attack_outcomes/*.json | jq`.
2. The `expected_rejection_site` field on the Attack entry tells you which
   production module should have caught it.
3. Write a failing unit test in the owning Phase 3/4 module's test dir.
4. Fix in the owning module; do NOT modify the red-team test (it is the spec).
5. Re-run the red-team suite.

## Rollback plan

If Phase 5 reveals a fundamental defect in Phase 3/4, the rollback is:
1. Revert the offending PR(s) from the work branch.
2. Re-run Phase 5 gate locally until GO.
3. Re-attempt with smaller diffs.

Phase 5 tests themselves are additive — they never need to be reverted.

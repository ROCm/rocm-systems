# Audit Gate Runbook — Parity + Red-Team + Go/No-Go

## Context

This runbook describes the audit gate that guards breaking changes to the
agentic runtime. Every exit criterion in the spec §7 Go/No-Go table has a
corresponding test suite here. If any gate slips, a breaking change is
blocked until fixed.

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
pytest tests/test_integration/test_airgap_parity.py tests/test_integration/test_airgap_intent_classify.py -v

# 6. Generate the exit dashboard
python scripts/exit_dashboard.py --output exit_dashboard.json --render
```

## How to interpret the dashboard

Three verdicts:

| Verdict | Meaning | Action |
|---------|---------|--------|
| GO | All 9 thresholds met (including nightly) | Change may proceed |
| PARTIAL (pending) | All non-nightly green; nightly inputs not yet collected | Wait for nightly CI; do NOT merge breaking changes yet |
| NO-GO | One or more PR-lane thresholds failed | Fix failing area, re-run |

## Interpreting a NO-GO

Each metric maps to a failure region:

| Metric | Likely root cause |
|--------|-------------------|
| `parity_agreement_rate < 0.95` | Parity/fixture drift — new path diverges from expected |
| `red_team_pass_count < 14` | Sanitizer, allowlist, or gate_cascade bug |
| `airgap_identical_rate < 1.0` | `intent_classifier` or `gate_cascade` grew a dependency on the LLM |
| `regression_gate_false_positive_rate > 0.05` | `regression.compare_runs` threshold tuning |
| `per_agent_narrow_scope_violations > 0` | Fence exceeded 400 lines / agent exceeded 5 tools |
| `tool_class_split_violations > 0` | An EXECUTION tool was registered with MCP |

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
3. Write a failing unit test in the owning module's test dir.
4. Fix in the owning module; do NOT modify the red-team test (it is the spec).
5. Re-run the red-team suite.

## Rollback plan

If the audit gate reveals a fundamental defect in the runtime or tools:
1. Revert the offending PR(s) from the work branch.
2. Re-run the audit gate locally until GO.
3. Re-attempt with smaller diffs.

Audit-gate tests themselves are additive — they never need to be reverted.

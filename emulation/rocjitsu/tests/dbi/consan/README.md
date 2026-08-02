# ConSan DBI validation

This directory contains the single ConSan validation runner, its fault runner,
shared coverage and provenance helpers, the live HIP fixtures, and focused
CPU-side contract tests. Generic rocJITsu DBI fixtures remain in the parent
`tests/dbi/` directory.

Keep this directory declarative. Add a workload, profile, or fault to
`consan_validation.py` and its checked-in reference data instead of creating a
new campaign-specific planner, executor, validator, and mirrored unit-test
module. A separate helper is justified only when it is reused by the validation
authority and has a distinct safety or parsing contract.

Run the CPU-only orchestration suite from the repository root with:

```sh
python3 -m unittest discover \
  -s emulation/rocjitsu/tests/dbi/consan -p 'test_consan*.py'
```

See [`VALIDATION.md`](../../../docs/consan/VALIDATION.md) for live-GPU workspace
requirements and the reproducible workload contract.

The physical-gfx1201 empirical study uses `consan_empirical_report.py` as a
read-only, fail-closed result reader. Its `consan_empirical_gfx1201.json`
manifest names the retained campaign artifacts and exact provenance, while
`consan_validation_faults_gfx1201_empirical.json` preserves the final fault
specification. The reader validates those inputs and generates
[`GFX1201_EMPIRICAL_RESULTS.md`](../../../docs/consan/GFX1201_EMPIRICAL_RESULTS.md);
it does not plan or execute validation campaigns.

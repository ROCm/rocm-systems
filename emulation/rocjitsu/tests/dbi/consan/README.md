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

Follow the [test setup instructions](../../../README.md#running-tests) to install
the CPU-only dependencies in a separate virtual environment and configure CMake.
Run the complete set of discovered ConSan Python suites from the repository root:

```sh
ctest --test-dir emulation/rocjitsu/build -L '^consan-python$' --output-on-failure
```

To run only this directory's orchestration tests, or without a CMake build, only
the virtual environment setup is needed. From the repository root:

```sh
emulation/rocjitsu/build/consan-python/bin/python -m unittest discover \
  -s emulation/rocjitsu/tests/dbi/consan -p 'test_consan*.py'
```

See [`VALIDATION.md`](../../../docs/consan/validation/VALIDATION.md) for live-GPU workspace
requirements and the reproducible workload contract.

The empirical campaign runner supports physical `gfx950` and `gfx1201`.

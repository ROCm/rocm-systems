# Init-pipeline warm-gate — corrected run commands (v11 CR-10)

Operates on the schema-valid committed config `configs/init_pipeline_warmgate.json`
(three real entries across two suites) and enforces a **positive `--go-timeout`**.
Run **in order**; **stop and diagnose at the first failed gate** — a wall-time
result is reportable only after every correctness gate is clean (v11 §Hardware
gate exit criteria). Record for every run: **commit SHA + dirty marker, config
SHA-256, persisted manifest, full command line, requested/effective pool, GO
timeout, host/GPU topology, binary paths, output paths.**

```bash
CFG=tools/scripts/test_runner/configs/init_pipeline_warmgate.json
R=tools/scripts/test_runner
GO=120            # production-safe GO timeout (>= worst-case queue wait)
SHA=$(git rev-parse HEAD); DIRTY=$(git diff --quiet || echo -dirty)
echo "commit=$SHA$DIRTY config_sha256=$(sha256sum $CFG | cut -c1-16)"
```
> First fill the three `test_filter` values in the config with real, verified
> `suite.case` identities on your build. The list-tests preflight (step 2) rejects
> any that match zero or multiple tests, so a wrong name fails fast.

## Pre-hardware
1. **Schema + unit tests:** `cd $R && python -m pytest tests/ -q`
2. **List-tests preflight + manifest** (proves each curated filter -> exactly one
   test; persists the manifest; exits nonzero on any rejection):
   ```bash
   python $R/test_runner.py -c $CFG --exec-mode init-pipeline --go-timeout $GO --no-build --emit-manifest
   ```
3. **Negative planner cases:** wildcard filter, NetIb-as-pipeline, runner-owned
   env override, `--go-timeout 0` without override — each must fail before spawn.

## Hardware (pool 1 and 2 only — never pool 8 for the first corrected gate)
4. **Direct serial smoke** of each committed filter (prove exactly one test runs,
   not just exit 0):
   ```bash
   python $R/test_runner.py -c $CFG --exec-mode serial --no-build --emit-results \
       --test-name '<one filter>' --report-suffix warmgate_smoke
   ```
5. **Serial per-config baseline:** `--exec-mode serial --expand-sweeps --emit-results`
6. **Isolated fork** (own manifest): `--exec-mode init-pipeline --init-pool 1 --go-timeout $GO --test-name 'warmgate_fork_*' --phase-timings --emit-results`
7. **Isolated MPI** (own manifest): `... --test-name 'warmgate_mpi_*' ...`
8. **Injected MPI failures** (test-only short timeout for the runner-death case):
   `... --test-name 'warmgate_mpi_*' --go-timeout 5 ...`
9. **Mixed cross-suite** run (fork + MPI + serial control, both suites, one scheduler):
   `--exec-mode init-pipeline --init-pool 2 --go-timeout $GO --phase-timings --emit-results`
10. **Continuous pool 1**, then **pool 2** (`--init-pool 1|2`).
11. **Automatic validation:** each run prints `interval validation ... ok=` and
    persists the manifest; the interval validator must report `max_executing=1`.
12. **Correctness diff** (exact stable-key subset for isolated runs; full curated
    set for pool runs; **no broad `--exclude`**):
    ```bash
    python $R/tools/compare_results.py <serial>/tests.jsonl <pool>/tests.jsonl
    ```
13. **Performance** — only after every step above is clean.

## Artifact bundle (collect for every gate run)
commit SHA + dirty marker; config + SHA-256; persisted `init_pipeline_manifest.json`;
exact commands; topology + binary paths; outer job log; **every per-entry log**;
`tests.jsonl` + summary metrics; interval-validation output; correctness diff;
retained failed rendezvous dirs; process/rank PID evidence. The collection fails
if any required artifact is missing or a job log ends without a terminal marker.

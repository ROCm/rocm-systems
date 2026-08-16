# Init-pipeline warm-gate — exact run commands (v10 §9/§10/§13)

Curated 3-entry gate: `configs/init_pipeline_warmgate.json`. Run **in this order**;
**stop and diagnose at the first failed gate** — do not interpret performance until
every correctness gate is clean (v10 §12). Record for every run: **commit SHA,
config hash, full command line, requested/effective `--init-pool`, host/GPU
topology, binary paths, output paths.**

```bash
CFG=tools/scripts/test_runner/configs/init_pipeline_warmgate.json
R=tools/scripts/test_runner            # runner dir
SHA=$(git rev-parse HEAD); echo "commit=$SHA config_hash=$(sha256sum $CFG | cut -c1-16)"
```

## Gate A — planner / negative (no hardware)
```bash
python $R/test_runner.py -c $CFG --exec-mode init-pipeline --emit-manifest   # manifest + planning summary, no spawn
python $R/test_runner.py -c $CFG --exec-mode init-pipeline --no-build --allow-serial-only --test-name '<serial-only-name>'  # serial-only accepted
cd $R && python -m pytest tests/ -q                                          # planner/negative unit tests green
```

## B — serial control (per-config baseline)
```bash
python $R/test_runner.py -c $CFG --exec-mode serial --expand-sweeps --no-build --emit-results --report-suffix warmgate_serial
```

## C — fork isolation (its own one-entry manifest)
```bash
python $R/test_runner.py -c $CFG --exec-mode init-pipeline --init-pool 1 --no-build \
    --test-name 'AllReduce.OutOfPlace' --phase-timings --emit-results --report-suffix warmgate_fork_iso
```

## D — MPI mechanism + injected failures (its own one-entry manifest)
```bash
python $R/test_runner.py -c $CFG --exec-mode init-pipeline --init-pool 1 --no-build \
    --test-name 'AllReduceMPITest.*' --phase-timings --emit-results --report-suffix warmgate_mpi_iso
# injected runner-death: TEST-ONLY short GO timeout (never in production)
python $R/test_runner.py -c $CFG --exec-mode init-pipeline --init-pool 1 --no-build \
    --test-name 'AllReduceMPITest.*' --go-timeout 5 --emit-results --report-suffix warmgate_mpi_gotimeout
```

## E — continuous loader (pool 1, then pool 2)
```bash
python $R/test_runner.py -c $CFG --exec-mode init-pipeline --init-pool 1 --loader-policy continuous \
    --phase-timings --no-build --emit-results --report-suffix warmgate_pool1
python $R/test_runner.py -c $CFG --exec-mode init-pipeline --init-pool 2 --loader-policy continuous \
    --phase-timings --no-build --emit-results --report-suffix warmgate_pool2
```

## F — per-config correctness (diff every pipeline run vs the expanded-serial baseline)
```bash
python $R/tools/compare_results.py <serial>/tests.jsonl <pool1>/tests.jsonl --exclude '*_CuMem1'
python $R/tools/compare_results.py <serial>/tests.jsonl <pool2>/tests.jsonl --exclude '*_CuMem1'
# or the whole pool sweep at once:
python $R/tools/init_pipeline_sweep.py -c $CFG --pools 1,2 --baseline <serial>/tests.jsonl -- --no-build --emit-results
```

## Interval / serialization check (E/F)
`max EXECUTING == 1` is derived from the combined `[exec_start, exec_end]` intervals
of **both** pipeline (`go_issued`) and serial (`serial_spawned`) entries in each run's
`tests.jsonl` (`lib.pipeline_runner.max_concurrent_execution`). A killed/timed-out
entry's interval ends only at confirmed reap.

**Performance is reportable only if every correctness gate above is clean** and the
run used only this committed config (no out-of-tree default-enabling patch).

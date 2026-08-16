# Init-pipeline execution mode

The init-pipeline overlaps each test entry's (dominant) RCCL device-code
initialization behind a READY/GO barrier, then executes entries **one at a time**
(no GPU co-tenancy). It is **opt-in** (`--exec-mode=init-pipeline`); the default
`serial` path is unchanged.

This is *not* co-tenancy: no two *tested executions* overlap. Only
*initialization* overlaps a currently-executing entry, so the collectives
themselves never run concurrently — the correctness contract is "identical
per-configuration pass/fail/skip vs the serial baseline."

See `test/common/ForkSafetyInvariant.md` for the hard rule the design rests on
(warm in the forked children/ranks, never the fork parent).

---

## How it works

1. **Plan** — each test's `warmup_profile` decides its path:
   - `fork_coll` — `rccl-UnitTests` fork sweeps. Expanded (via the pure
     `lib/sweep.py::enumerate_sweep`) into one **sub-entry per sweep point**, each
     pinned to a single child generation with `UT_MIN_GPUS/UT_MAX_GPUS`,
     `UT_POW2_GPUS`, `UT_PROCESS_MASK`, `UT_RANKS_PER_GPU`.
   - `mpi_coll` — `rccl-UnitTestsMPI`. One generation per process already.
   - `netib_plugin` — pure NetIb plugin tests (conservative boundary; **hardware
     classification still pending**, see below).
   - `none` / absent — runs on the **serial path** (also every test under
     `--exec-mode=serial`).
2. **Warm** — the C++ side loads the device-code object via a throwaway
   single-rank `ncclCommInitAll(1)` (no cross-rank/NIC connection), then
   atomically publishes a `ready` token (`test/common/Rendezvous.*`).
3. **Barrier** — the runner's `PipelineScheduler` (`lib/pipeline.py`) launches
   entries under a bounded init-pool, parks each at READY, and releases exactly
   one at a time (`go`), waiting for full process-tree exit before the next.
4. **Roll up** — split-sweep sub-entries are folded back into a `parent_summary`
   with `record_type` / `counts_toward_topline`, so totals never double-count.

Enabling env (set by the runner, never by hand): `RCCL_TEST_READY_GO=1` (warmup)
and an **absolute** per-entry `RCCL_TEST_RENDEZVOUS_DIR`.

---

## Config

```json
{
  "name": "AllReduce.InPlace",
  "warmup_profile": "fork_coll",
  "fork_expand": { "num_gpus": [8], "process_mask": 3, "max_ranks_per_gpu": 1 }
}
```
- `process_mask`: `1`=single-process, `2`=multi-process, `3`=both.
- `perf_sensitive: true` marks a measured test so `--loader-policy=quiescent_exec`
  pauses new warmups while it executes.
- MPI/NetIb tests need only `warmup_profile` (no `fork_expand`).

## CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--exec-mode {serial,init-pipeline}` | `serial` | enable the pipeline |
| `--init-pool N` | `2` | max INITIALIZING+READY (ceiling 8) |
| `--init-timeout S` | `600` | READY wait before failing an entry (`0`=indefinite, dangerous) |
| `--loader-policy {continuous,quiescent_exec}` | `continuous` | pause warmups during a perf entry |
| `--release-order {ready,config}` | `ready` | cross-entry execution order |
| `--fork-sweep-policy {legacy,independent}` | `legacy` | per-parent order + fail-fast (serial-equivalent) vs max overlap |
| `--phase-timings` | off | print per-suite init/queue-wait/exec aggregate |
| `--queue-wait-warn S` | `0` | warn if READY→GO wait exceeds S |
| `--emit-manifest` | off | print the sub-entry expansion and exit |

---

## Validation gates (run on hardware)

Always compare against a **same-settings serial baseline** (e.g. CuMem OFF both
sides). Totals across differently-configured runs are not comparable.

- **A1/A2 (warmup safe & valid):** run a suite `serial` vs `init-pipeline`;
  P/F/S identical, durations nonzero (no `0.000s`), no signal-11/exit-139, and the
  `[RCCL_TEST_WARMUP]` diagnostics show `warmup pid == executing pid` with the
  warmed device ⊇ the assigned device (and never the fork parent's pid).
- **A3 (one entry):** `RCCL_TEST_READY_GO=1 RCCL_TEST_RENDEZVOUS_DIR=<dir>` on one
  entry; it warms, writes `<dir>/ready`, blocks; `touch <dir>/go` releases it.
- **A4 (two-entry overlap):** two entries warm concurrently, execute serially
  (`--phase-timings` shows `time_to_ready` overlapping the prior `execution_time`).
- **A5 (scale & perf):** sweep `--init-pool ∈ {1,2,4,6,8}` × `--loader-policy`;
  correctness must equal serial per-config first, then read wall time.

**Per-config correctness diff** (the decisive gate):
```
python tools/compare_results.py SERIAL/tests.jsonl PIPELINE/tests.jsonl \
    --exclude '*_CuMem1'
```
Exits non-zero on any regression or dropped coverage.

**Scale sweep driver:**
```
python tools/init_pipeline_sweep.py -c configs/<cfg>.json \
    --pools 1,2,4,6,8 --policies continuous,quiescent_exec \
    --baseline SERIAL/tests.jsonl -- --no-build --emit-results
```
Runs each combination, prints a wall-time table, and diffs each run against the
serial baseline.

---

## Still pending (hardware-gated)

- **NetIb conservative boundary (§4.3/4.4):** where to place READY in the pure
  NetIb plugin call graph (before any tested `connect`/`accept`/QP creation) must
  be classified with hardware evidence before the boundary is fixed. Until then
  `netib_plugin` tests should stay on the serial path.
- **Liveness-fd propagation (§2.2):** the runner can pass a liveness pipe
  (`RCCL_TEST_LIVENESS_FD`, already read by `Rendezvous::WaitForGo`) so entries
  tear down if the runner dies; propagation through shell + mpirun is unproven, so
  process-group termination remains the primary mechanism and the pipe is not yet
  created runner-side.
- **Phase 5 numbers:** the ~15–20% wall-time improvement is a *hypothesis*; only a
  Gate-A5 run with all validity + per-config checks green makes any speedup real.

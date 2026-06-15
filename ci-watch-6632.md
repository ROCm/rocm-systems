# CI Watch — PR #6632

Automated watcher that reruns flaky CI failures for PR
[#6632](https://github.com/ROCm/rocm-systems/pull/6632)
("AIRUNTIME-2124: Add GPU-side signal/wait for imported external semaphores").

- **Repo:** ROCm/rocm-systems
- **Branch:** `jungx098/rocr-vulkan-extsem-signal-wait`
- **Base:** `develop`

## Scheduled jobs (session-only)

These are in-memory Claude cron jobs — they only fire while the Claude session
is idle and are lost when the session exits.

| Job ID | Schedule | Type | Purpose |
| --- | --- | --- | --- |
| `41818cd9` | `11,41,57 * * * *` (:11, :41, :57 each hour) | recurring | Run the watch cycle below |
| `1cbdc289` | `33 6 10 6 *` (2026-06-10 06:33 PDT) | one-shot | Delete `41818cd9` — ends the 10-hour window |

Window: ~20:33 PDT 2026-06-09 → 06:33 PDT 2026-06-10 (10 hours).

## What the watch cycle does

1. Find the latest **TheRock CI** run for the PR (run id from the
   `statusCheckRollup` detailsUrl).
2. If the run is not `completed` (queued/in_progress) → no action this cycle.
3. If `completed`, count failed jobs whose name does **not** contain `(xfail)`.
4. **Guard:** only rerun if at least one *non-xfail* job failed. If only
   `(xfail)` jobs failed, do nothing (they always fail by design — rerunning
   would loop forever).
5. Rerun all failures in parallel: `gh run rerun <id> --repo ROCm/rocm-systems --failed`.
   This also re-triggers the xfail shards, which is harmless for a one-shot.
6. Also rerun failed jobs in the **rocprofiler-sdk Code Coverage** and
   **Continuous Integration (Core …)** runs the same way.

## Why `--failed` (parallel) instead of per-job reruns

GitHub only allows individual-job reruns when the run is terminal; the first
`gh run rerun --job <id>` flips the run to `in_progress` and blocks the rest,
making per-job reruns effectively serial (one per cycle). `gh run rerun --failed`
reruns every failed job at once. Trade-off: it also reruns the 4 `(xfail)`
shards, but as a one-shot (not the auto-loop) that is just minor wasted runner
time.

## Known failing jobs (treated as flaky / infra)

Non-deterministic across reruns — different shards fail each run, including on
unrelated PRs, so attributed to CI flakiness rather than the PR changes:

- `Test hip-tests` (Linux) — single tests, e.g. `hipGetProcAddressIpcApis`,
  `Unit_hipStreamEndCapture_first_and_add_a_node_later` (1 of 1015)
- `Test hip-tests (PAL)` (Windows) — shards rotate
- `Test rocprofiler-sdk` — `test_kernel_trace_no_bubbles` (wall-clock timing
  assert; fails every run under runner load)
- `Test rocprofiler-systems`, `Test rocrtst` (queue alloc errors under load)
- `Core • mi325/mi355` — sometimes fail at the CMake dashboard stage (infra)
- `Test hip-tests (ROCR) (xfail)` shards — **expected to fail**, never rerun

## Required check to merge

Only `TheRock CI Summary` gates the merge (plus 1 approving review, code-owner
review, and resolved review threads). All other checks run but are non-gating.

## Manual commands

```bash
# Status snapshot
gh pr view 6632 --repo ROCm/rocm-systems --json statusCheckRollup

# Rerun all failed jobs in a run (parallel)
gh run rerun <runId> --repo ROCm/rocm-systems --failed

# List the cron jobs / cancel early
#   (via Claude CronList / CronDelete tools)
```

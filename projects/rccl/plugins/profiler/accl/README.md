# ACCL Profiler — RCCL Timing Decomposition Plugin

A profiler plugin that decomposes collective execution time into GPU kernel,
network, proxy, and enqueue delay components. Unlike the inspector plugin
(which only captures Coll + KernelCh events), the accl-profiler subscribes
to all four key event types for full root-cause triage.

## Event Subscriptions

```
ncclProfileColl | ncclProfileKernelCh | ncclProfileProxyOp | ncclProfileProxyStep
```

| Event Type | What it captures |
|---|---|
| `ncclProfileColl` | Collective lifecycle (host-side start/stop) |
| `ncclProfileKernelCh` | Per-channel GPU kernel timing (globaltimer) |
| `ncclProfileProxyOp` | Proxy operation lifecycle (CPU-side) |
| `ncclProfileProxyStep` | Network send/recv step state transitions |

## Output

Per-collective JSONL records with timing decomposition:

```json
{
  "header": {"rank": 0, "n_ranks": 8},
  "coll_perf": {
    "coll": "AllReduce", "coll_sn": 10, "coll_msg_size_bytes": 4194304,
    "coll_algo": "Ring", "coll_proto": "Simple",
    "coll_n_channels": 4, "coll_n_channels_reported": 4,
    "coll_exec_time_us": 150.30,
    "coll_algobw_gbs": 0.027907, "coll_busbw_gbs": 0.048836,
    "coll_timing_source": "gpu_globaltimer",
    "decomposition": {
      "enqueue_to_kernel_us": 5.20,
      "gpu_kernel_avg_us": 45.10, "gpu_kernel_min_us": 44.80, "gpu_kernel_max_us": 45.40,
      "proxy_gpu_wait_us": 12.80, "proxy_network_us": 1.20,
      "proxy_peer_wait_us": 38.70, "proxy_flush_us": 42.30,
      "proxy_gpu_recv_wait_us": 0.00,
      "n_proxy_ops": 8, "n_send_ops": 4, "n_recv_ops": 4
    },
    "event_trace_ts": {
      "kernel_events": [
        {"channel_id": 0, "kernel_start_ts": 123456, "kernel_stop_ts": 128006, "duration_us": 45}
      ]
    }
  }
}
```

`coll_n_channels` is the channel count the plugin uses as its completion target.
`coll_n_channels_reported` is the raw value from the profiler v5 descriptor, whose
`nChannels` field is a `uint8_t`: a collective using the full 256 channels
(`NCCL_MAX_NCHANNELS=256`) reports `0`. The plugin promotes a reported `0` to 256,
so the two fields differ only in that wrapped case.

The last line of every file is a summary, written on every clean finalize:

```json
{"summary": {"dropped_collectives": 0, "leaked_collectives": 0,
             "pool_size": 256, "complete": true}}
```

- `dropped_collectives` — never got a pool slot because the pool was full.
- `leaked_collectives` — got a slot but never completed, because RCCL's teardown
  drain skipped some of their kernel-channel events; reclaimed at finalize.
- `complete` — false if either counter is non-zero. **A file with no summary line
  at all means the process did not reach finalize**, so its data is also suspect.
  `accl_report.py` warns on stderr in both cases; do not compare an incomplete run
  against a full one.

## Build

```bash
# CMake (standalone)
cmake -S . -B build && cmake --build build

# CMake (as part of a Linux RCCL build — enabled by default)
cmake ...

# Optional: omit the plugin from an RCCL build/package
cmake -DBUILD_PROFILER_ACCL=OFF ...
```

An installed RCCL build places the plugin at `lib/librccl-profiler-accl.so`
and the report utility at `share/rccl/accl/accl_report.py`. The plugin is a
separate host-only shared library and is not linked into `librccl.so`; it is
loaded only when selected with `NCCL_PROFILER_PLUGIN`.

## Usage

```bash
export NCCL_PROFILER_PLUGIN=/path/to/librccl-profiler-accl.so
export ACCL_PROFILER_OUTPUT_DIR=/path/to/output
export ACCL_PROFILER_MIN_SIZE_BYTES=0       # optional, filter small messages

./all_reduce_perf -b 1K -e 256M -f 2 -g 1 -n 20 -w 5
```

Output files are written as `accl_profiler_rank<N>_<hostname>_pid<PID>_0x<commHash>.jsonl`
in the output directory.

## Report Script

```bash
# Single run analysis
python3 accl_report.py single --input /path/to/output/

# Compare baseline vs candidate
python3 accl_report.py compare --baseline /path/to/baseline/ --candidate /path/to/candidate/
```

The report classifies bottlenecks per message size (evaluated in this order):
- **unknown** — zero wall time (degenerate record)
- **gpu-compute (no proxy)** — kernel-only collective with no proxy ops
- **NETWORK** — network send/recv + peer wait dominates (>40% of wall time)
- **PROXY-FLUSH/GDR** — GDR flush + GPU recv wait dominates (>40%)
- **GPU-SCHEDULING** — proxy GPU wait dominates (>20% of wall time)
- **ENQUEUE-DELAY** — enqueue-to-first-kernel delay dominates (>40%)
- **GPU-COMPUTE** — kernel time dominates (>50% of wall time)
- **mixed** — no single component dominates

## Requirements

- RCCL v2.30+ (profiler v5 API)
- `LD_LIBRARY_PATH` must include the RCCL build with v5 support

## CI validation

From the repository root, run the host-only regression tests:

```bash
python3 -m pytest -q projects/rccl/test/test_accl_profiler_ci.py \
  projects/rccl/test/test_plan_rccl_ci.py
```

These exercise the planner, artifact discovery, rendered Bash preflight checks,
JSONL validation, and report/summary integration. They also run as a separate
CPU-only job in `rccl-host-unit-tests.yml`. They do not replace GPU qualification
or establish the numerical correctness of the profiler's output.

[PR #11784](https://github.com/ROCm/rocm-systems/pull/11784) owns the GitHub Actions
integration: package/build, suite selection, Ruby execution, and diagnostic
artifacts. [PR #11881](https://github.com/ROCm/rocm-systems/pull/11881) owns the
profiler/report correctness changes, including warmup filtering. The CI runs
the reporter shipped with its fetched artifact; report calculations are not
reimplemented here. That correctness PR also identifies remaining KernelCh
issues, so it must not be assumed to fix every AllGather coverage failure.

To qualify a PR branch with a fresh gfx950 build and the two-node Ruby sweep:

```bash
gh workflow run therock-rccl-ci.yml --ref YOUR_PR_BRANCH \
  -f test_suites=accl-profiler -f accl_profiler_nodes=2
```

`test_suites` accepts any comma-separated subset of `single-node`, `rocprof`,
`pytorch`, `jax`, `madengine`, and `accl-profiler`. The default `auto` preserves
the normal event-based policy. Explicit selections build only their GPU families.
After the fresh build finishes, repeat with `-f artifact_run_id=BUILD_RUN_ID`
to test the skipped-build/reuse path without rebuilding. Use artifacts from the
same commit when claiming runtime qualification of that commit.

The CI sweep uses five iterations and two initial warmups per size, with a
1K–256M factor-two sweep, to bound profiler pool pressure. It still rejects any
dropped/leaked records, incomplete summaries, missing ranks or requested sizes,
and report failures. AllGather/ReduceScatter descriptor bytes are per-rank
contributions, not rccl-tests' aggregate buffer bytes. A successful process exit
alone is not a passing output-validation run. Missing coverage remains a visible
failure to hand off to profiler work, not a reason to duplicate fixes in the CI
driver. Keep `manifest.json`, the reports, preflight log, and raw JSONL artifacts
as the execution and output-validation evidence. Report timings remain subject
to the correctness work above.

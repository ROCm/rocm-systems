# ACCL Profiler — RCCL Timing Decomposition Plugin

A profiler plugin that decomposes collective execution time into GPU kernel,
network, proxy, and enqueue delay components. Unlike the inspector plugin
(which only captures Coll + KernelCh events), the accl-profiler subscribes
to all four key event types for full root-cause triage.

## Event Subscriptions

```
ncclProfileColl | ncclProfileKernelCh | ncclProfileProxyOp | ncclProfileProxyStep
```

`ncclProfileKernelLaunch | ncclProfileNetPlugin` are added with `ACCL_NET_COUNTERS=1`
(see [Network Counters](#network-counters-opt-in)).

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

Each `proxy_*` field is a mean over the proxy ops that can produce it, not over
`n_proxy_ops`. A send op only passes through the send-side states and a recv op
only through the recv-side ones, so `proxy_gpu_wait_us` and `proxy_peer_wait_us`
are averaged over `n_send_ops`, and `proxy_flush_us` and `proxy_gpu_recv_wait_us`
over `n_recv_ops`. `proxy_network_us` covers both directions and is the sum of
the per-send-op and per-recv-op means. A ring collective posts one send and one
recv op per channel, so all five are per-channel costs on the same scale as
`gpu_kernel_avg_us`. A class with no ops contributes 0, since its total is 0 too.

The last line of every file is a summary, written on every clean finalize:

```json
{"summary": {"dropped_collectives": 0, "leaked_collectives": 0,
             "inflight_collectives": 0,
             "dropped_proxy_ops": 0, "dropped_proxy_steps": 0,
             "overflow_proxy_ops": 0, "stale_proxy_steps": 0,
             "outstanding_proxy_ops": 0, "outstanding_proxy_steps": 0,
             "write_error": false, "coll_pool_size": 256,
             "proxy_op_pool_size": 1024, "proxy_step_pool_size": 4096,
             "max_proxy_ops_per_coll": 256, "complete": true}}
```

- `dropped_collectives` — never got a pool slot because the pool was full.
- `leaked_collectives` — got a slot but never completed, because RCCL's teardown
  drain skipped some of their kernel-channel events; reclaimed at finalize.
- `inflight_collectives` — an owner was writing the record when finalize closed
  the file, so that record may be missing from the output.
- `dropped_proxy_ops` / `dropped_proxy_steps` — no free slot in the proxy op or
  proxy step pool; the affected records understate the proxy decomposition.
- `overflow_proxy_ops` — the op completed, but its collective already held
  `max_proxy_ops_per_coll` ops, so its timings were discarded.
- `stale_proxy_steps` — the step stopped after its parent op's pool slot had
  been reissued, so its timings were dropped rather than charged to the new
  tenant.
- `outstanding_proxy_ops` / `outstanding_proxy_steps` — still held at finalize
  because their stop had not arrived. Their slots are deliberately *not*
  reclaimed: proxy events are delivered straight from the proxy progress thread,
  which is shared across communicators and can outlive one comm's teardown, so
  releasing them could free the context under a live caller. The context is
  freed by whichever stop arrives last; if none ever does, one fixed-size
  context per affected communicator is retained until the process exits. The
  output file is always closed at finalize, so no descriptor is held.
- `write_error` — `ferror()` on the output file at finalize: the records on disk
  are truncated, and every counter above may still read 0.
- `complete` — false if any of the nine counters is non-zero, or if
  `write_error` is true. **A file with no summary line at all means the process
  did not reach finalize**, so its data is also suspect.
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

## Network Counters (opt-in)

For accuracy-first diagnostics, build RCCL and this plugin from the same tree:

```bash
ACCL_NET_COUNTERS=1 ACCL_NET_STRICT=1 ACCL_NET_DEVICES=rdma0 \
ACCL_PROFILER_MIN_SIZE_BYTES=0 NCCL_OOB_NET_ENABLE=0 NCCL_SOCKET_IFNAME=eno0 \
NCCL_NET=IB-CAST <serialized application>
```

Strict mode supports an eager, serialized application with one collective per
plan, one GPU per process, and one active communicator per process. Every rank
must enable it consistently. The measured NICs must be isolated from unrelated
producers. Graphs, host-callback plans, grouped/P2P plans, symmetric/GPU-initiated
operations, and CE/RMA paths are unsupported. Unsupported or unverified windows
cannot produce CLEAN. Aborted runs can lose the final snapshot and report it.

The sequence is:

```
previous GPU work retires
all registered proxy operations complete (including ones that start late)
all ranks establish a drained boundary through socket bootstrap
worker reads until two consecutive hardware snapshots match (up to six reads)
accept settled S1 (N's after and N+1's before)
all ranks finish reading
launch N+1
```

The ordinary eager `KernelLaunch` callback performs the wait before GPU launch.
Coll start never waits for reads. A final boundary runs during normal communicator
finalization while bootstrap and progress threads are still alive. Any failed
drain/read coordination invalidates attribution; it does not manufacture a delta.
`ACCL_NET_WAIT_MS` (default 1000) bounds local waits. Distributed bootstrap follows
RCCL's normal abort behavior, so diagnostic jobs should have an external timeout.

RCCL supplies a small optional protocol through existing v5 NetPlugin events:
registration begins before GPU completion events are posted, and each actual
proxy operation is counted before posting. Coll stop closes registration.
The completion predicate waits for the registered count as well as all GPU
channels and all observed proxy operations. Profiler ABI layouts are unchanged.
Older RCCL builds without this protocol produce unverified counter windows.

RDMA netlink STAT_GET reads bypass the sysfs lifespan cache. ethtool pause/PFC
counters are stale context on ionic and never determine a collective verdict.
Software counters require `RCCL_TELEMETRY_ENABLE=1`. AINIC ucast hardware bytes
include CTS control traffic; they are not identical to application payload bytes.

Each `net_counters` object contains:

- per-device deltas, overlap/unavailability flags, `completion_verified`, and
  `coordinated`;
- raw `snapshot_before` and `snapshot_after`, with IDs, every RDMA value, overall
  read intervals, each device's read interval, stability status, and read attempts;
- host-observed collective start/completion times and `launch_wait_us`;
- `counter_discontinuities`: decreasing/reset/wrapped counters invalidate the
  affected device's delta instead of silently erasing error evidence;
- software payload-byte/CQ-error deltas and explicitly stale ethtool context.

Snapshots are reused only across the coordinated quiescent boundary. The worker
does the hardware reads. Several configured NICs are read sequentially, so their
timestamps differ; no atomic multi-NIC latch is claimed. Two identical reads validate observed stability, not a firmware visibility fence.
Counter visibility still depends on driver/firmware semantics and must be checked against hardware.

Configuration retained for passive compatibility:

| Variable | Default | Meaning |
|---|---|---|
| `ACCL_NET_COUNTERS` | off | Enable counters |
| `ACCL_NET_STRICT` | off | Coordinate accuracy-first boundaries; required for CLEAN |
| `ACCL_NET_DEVICES` | required | Explicit RDMA devices, port 1 |
| `ACCL_NET_DEVICE_SELECT` | all | `local_rank` uses rank modulo uniform ranks/node; verify routing |
| `ACCL_NET_WAIT_MS` | 1000 | Local completion/read timeout |
| `ACCL_NET_GRACE_US` | 1000 | Passive-mode grace only |

`net_tel_report.py OUTPUT --warmup 0 --trace trace.json` reports per-communicator,
collective, and size. CLEAN requires registered/coordinated valid windows, all
expected ranks, and at least three complete collective instances. Recognized
loss/congestion produces NETWORK_SUSPECT; unclassified non-volume counter changes
produce INCONCLUSIVE. Missing rank files cannot imply a healthy network.

The Perfetto trace has separate lanes for host-observed collective activity,
NIC totals for that collective, and deduplicated raw hardware snapshots. Select
a collective slice to inspect all NIC deltas, raw-read metadata, payload counters,
and validity. GPU duration is a detail field; host callback placement is not
presented as an exactly aligned GPU timeline. Counter graphs show per-collective
totals, not instantaneous rates. Each host's CLOCK_MONOTONIC origin is normalized
independently: timelines from different hosts are not clock synchronized.

Measured GPU times are reported unchanged. `--estimate-read-skew` optionally
shows the older heuristic subtraction with an explicit warning; it does not
recover the uninstrumented execution time. Snapshot waits and coordination alter
the workload's natural overlap and congestion behavior.

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

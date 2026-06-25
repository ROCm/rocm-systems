# ACCL Profiler — RCCL Timing Decomposition Plugin

A profiler plugin that decomposes collective execution time into GPU kernel,
network, proxy, and launch overhead components. Unlike the inspector plugin
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
  "coll": "AllReduce",
  "msg_size": 4194304,
  "total_exec_us": 150.3,
  "launch_overhead_us": 5.2,
  "gpu_kernel_avg_us": 45.1,
  "proxy_peer_wait_us": 38.7,
  "proxy_network_us": 1.2,
  "proxy_flush_us": 42.3,
  "proxy_gpu_wait_us": 12.8,
  "bottleneck": "PROXY-FLUSH/GDR"
}
```

## Build

```bash
# From this directory (after building RCCL)
make

# Or with custom NCCL_HOME
make NCCL_HOME=/path/to/rccl/build
```

## Usage

```bash
export NCCL_PROFILER_PLUGIN=/path/to/librccl-profiler-accl.so
export ACCL_PROFILER_OUTPUT_DIR=/path/to/output
export ACCL_PROFILER_MIN_SIZE_BYTES=0       # optional, filter small messages
export ACCL_PROFILER_WARMUP_ITERS=5         # optional, skip warmup iterations
export NCCL_INSPECTOR_REQUIRE_KERNEL_TIMING=0  # required on some ROCm versions

./all_reduce_perf -b 1K -e 256M -f 2 -g 1 -n 20 -w 5
```

Output files are written as `accl_rank_<N>.jsonl` in the output directory.

## Report Script

```bash
# Single run analysis
python3 accl_report.py single --input /path/to/output/

# Compare baseline vs candidate
python3 accl_report.py compare --baseline /path/to/baseline/ --candidate /path/to/candidate/
```

The report classifies bottlenecks per message size:
- **GPU-COMPUTE** — kernel time dominates
- **PROXY-FLUSH/GDR** — GDR flush dominates
- **PROXY-PEER-WAIT** — remote FIFO wait dominates
- **NETWORK** — raw network I/O dominates
- **LAUNCH-OVERHEAD** — host-side launch latency dominates
- **GPU-SCHEDULING** — proxy GPU wait dominates

## Requirements

- RCCL v2.30+ (profiler v5 API)
- `LD_LIBRARY_PATH` must include the RCCL build with v5 support

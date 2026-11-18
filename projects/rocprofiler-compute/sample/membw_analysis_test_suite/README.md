# L2/L1 Bottleneck Workloads

HIP workloads targeting MI350 L2 backpressure and L1 stall metrics (≥10%).

More workloads are in the subfolders:

- [`l2/`](l2/README.md): L2 cache and memory workloads.
- [`ea/`](ea/README.md): EA (Efficiency Arbiter) memory bandwidth workloads.

## Workload Target

Each workload runs the baseline kernel by default and the optimized kernel with the `opt` argument.

| Workload | Baseline (intended effect) | Optimized (intended effect) |
|---|---|---|
| gl2_backpressure | Uncoalesced + large stride -> L2 misses -> fill LFIFO -> TCP stalls. | Shared mem, minimal L2 traffic. |
| L1_stall_microbenchmark | Uncoalesced loads/stores -> TA/UTCL1/L2 stalls. | Shared memory reduces VMEM FIFO pressure. |
| utcl1_stall | Rapid page hopping (>32 pages) -> exceed UTCL1 entries -> in-flight stall. | Stay in one page. |
| ta_tcp_stall | Random access, TA waits for TCP. | Sequential access, TCP serves TA quickly. |

`L1_stall_microbenchmark` also takes `sq1`, which runs a kernel that tries to fill the SQ-to-TA FIFOs ("VMEM stalled by L1 Cache"). It doesn't fill them yet.

> [!NOTE]
> Above workloads are still WIP, the profiled result may not reflect the intended results.

## Target Metrics

The metrics are in the "L1 Cache" table (3001) of the Memory Bandwidth Analysis panel.

| Workload | Metric | Expression |
|---|---|---|
| gl2_backpressure | L1 Cache - TCP stalled by L2 | `100 * SUM(TCP_TCR_TCP_STALL_CYCLES_sum) / SUM(TCP_GATE_EN2_sum)` |
| L1_stall_microbenchmark | L1 Cache - TA stalled by TCP (aggregated), L1 Cache - TCP stalled by UTCL1, L1 Cache - TCP stalled by L2 | See the other rows |
| L1_stall_microbenchmark `sq1` | L1 Cache - VMEM stalled by L1 Cache | `100 * SUM(SQ_VMEM_TA_CMD_FIFO_FULL) / SUM(SQ_BUSY_CYCLES)` |
| utcl1_stall | L1 Cache - TCP stalled by UTCL1 | `100 * SUM(TCP_UTCL1_STALL_INFLIGHT_MAX_sum) / SUM(TCP_GATE_EN2_sum)` |
| ta_tcp_stall | L1 Cache - TA stalled by TCP (aggregated) | `100 * SUM(TCP_TCP_TA_ADDR_STALL_CYCLES_sum) / SUM(TCP_GATE_EN2_sum)` |

## Build

```bash
hipcc -g <hip workload> -o <output>
# example: hipcc -g gl2_backpressure.hip -o gl2_backpressure
```

## Profiling

```bash
# Note: --no-roof is optional and skips the
# benchmarking that is unrelated to our purpose

# baseline profile
rocprof-compute profile -n gl2_backpressure_baseline --experimental --membw-analysis --no-roof -- ./gl2_backpressure
# optimized profile
rocprof-compute profile -n gl2_backpressure_optimized --experimental --membw-analysis --no-roof -- ./gl2_backpressure opt
```

## Analyzing

```bash
rocprof-compute analyze -p <path to profiled result>
```

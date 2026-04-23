# L2/L1 Bottleneck Workloads

HIP workloads targeting MI350 L2 backpressure and L1 stall metrics (≥10%).

## Workload Target

+--------------------------+--------------------------------------------------------------------------------------------------+-------------------------------------------+
| Workload                 | Baseline (intended effect)                                                                       | Optimized (intended effect)               |
+--------------------------+--------------------------------------------------------------------------------------------------+-------------------------------------------+
| gl2_backpressure         | Uncoalesced + large stride -> L2 misses -> fill LFIFO -> TCP stalls.                             | Shared mem, minimal L2 traffic.           |
+--------------------------+--------------------------------------------------------------------------------------------------+-------------------------------------------+
| L1_stall_microbenchmark  | Many scattered loads/stores -> VMEM issues many commands; TA/GL1 can't consume them fast enough  | Shared memory reduces VMEM FIFO pressure. |
|                          | -> VMEM FIFO fills.                                                                              |                                           |
+--------------------------+--------------------------------------------------------------------------------------------------+-------------------------------------------+
| utcl1_stall              | Rapid page hopping (>32 pages) -> exceed UTCL1 entries -> in-flight stall.                       | Stay in one page.                         |
+--------------------------+--------------------------------------------------------------------------------------------------+-------------------------------------------+
| ta_tcp_stall             | Random access, TA waits for TCP.                                                                 | Sequential access, TCP serves TA quickly. |
+--------------------------+--------------------------------------------------------------------------------------------------+-------------------------------------------+

> [!NOTE]
Note: Above workloads are still WIP, the profiled result may not reflect the intended results.

## Target Metrics

- **gl2_backpressure**: `TCP_TCR_TCP_STALL_CYCLES / tcp_busy`
- **utcl1_stall**: `TCP_UTCL1_STALL_INFLIGHT_MAX / tcp_busy`

## Build

```bash
hipcc -g <hip workload> -o <output>
# example: hipcc -g gl2_backpressure.hip -o gl2_backpressure
```

## Profiling

```bash
# baseline profile
rocprof-compute profile -n gl2_backpressure_baseline --membw-analysis --no-roof -- ./gl2_backpressure
# optimized profile
rocprof-compute profile -n gl2_backpressure_optimized --membw-analysis --no-roof -- ./gl2_backpressure opt
```

## Analyzing

```bash
# baseline profile
rocprof-compute analyze -p <path to profiled result> --membw-analysis
```

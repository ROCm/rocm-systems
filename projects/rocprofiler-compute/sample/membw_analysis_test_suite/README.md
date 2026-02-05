# L2/L1 Bottleneck Workloads

HIP workloads targeting MI350 L2 backpressure and L1 stall metrics (≥10%).

## Workload Target

| Workload | Intended effect |
|----------|-----------------|
| **gl2_backpressure** | Baseline: uncoalesced + large stride → L2 misses → fill LFIFO → TCP stalls. Optimized: shared mem, minimal L2 traffic. |
| **vmem_stall** | Baseline: Many scattered loads/stores -> VMEM issues many commands; TA/GL1 can’t consume them fast enough → VMEM FIFO fills. Optimized: Shared memory reduces VMEM FIFO pressure. |
| **utcl1_stall** | Baseline: rapid page hopping (>32 pages) → exceed UTCL1 entries → in-flight stall. Optimized: stay in one page. |
| **utcl2_stall** | Baseline: extreme page hopping → exceed UTCL2 capacity → page not present → LFIFO stall. Optimized: sequential access. |

> [!NOTE]
Note: Above workloads are still WIP, the profiled result may not reflect the intended results.

## Target Metrics

- **gl2_backpressure**: `TCP_TCR_TCP_STALL_CYCLES / tcp_busy`
- **utcl1_stall**: `TCP_UTCL1_STALL_INFLIGHT_MAX / tcp_busy`
- **utcl2_stall**: `TCP_UTCL1_STALL_LFIFO_NO_RES / tcp_busy`

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
rocprof-compute profile -n gl2_backpressure_opt --membw-analysis --no-roof -- ./gl2_backpressure opt
```

## Analyzing

```bash
# baseline profile
rocprof-compute analyze -p <path to profiled result> --membw-analysis
```

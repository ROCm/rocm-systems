# Memory Bandwidth Analysis Test Suite

HIP workloads targeting MI350 (gfx950) L1 and L2 cache/memory metrics for validation with `rocprof-compute --membw-analysis --experimental`.

## L1 Workloads (tables 3001-3003)

| Workload | Baseline (intended effect) | Optimized (intended effect) |
|---|---|---|
| gl2_backpressure | Uncoalesced + large stride -> L2 misses -> fill LFIFO -> TCP stalls. | Shared mem, minimal L2 traffic. |
| L1_stall_microbenchmark | Many scattered loads/stores -> VMEM FIFO fills. | Shared memory reduces VMEM FIFO pressure. |
| utcl1_stall | Rapid page hopping (>32 pages) -> exceed UTCL1 entries -> in-flight stall. | Stay in one page. |
| ta_tcp_stall | Random access, TA waits for TCP. | Sequential access, TCP serves TA quickly. |

> [!NOTE]
> L1 workloads are still WIP. Profiled results may not yet reflect intended results.

### L1 Target Metrics

- **gl2_backpressure**: `TCP_TCR_TCP_STALL_CYCLES / tcp_busy`
- **utcl1_stall**: `TCP_UTCL1_STALL_INFLIGHT_MAX / tcp_busy`

## L2 Workloads (tables 3007-3012)

Each workload targets specific TCC perfmon counters exposed in the new L2 metric tables added in PR 4091. All use runtime L2 cache size detection (`hipDeviceAttributeL2CacheSize`).

| Workload | Target Metrics | Baseline (intended effect) | Optimized (intended effect) |
|---|---|---|---|
| l2_hbm_read_bw_stress | L2-EA read credit stall (HBM), L2-EA read BW, HBM read fraction, L2 hit rate, L2 Cache Efficiency | Streaming reads over 4x L2 buffer. Every access misses L2 -> EA read -> HBM. Exhausts DRAM read credits. | Reads small tile fitting in L2 -> high hit rate, no credit stalls. |
| l2_hbm_write_bw_stress | L2-EA write credit stall (HBM), L2-EA write stall, TOO_MANY_EA_WRREQS_STALL, SRC_FIFO_FULL, writeback rate | Streaming writes over 4x L2 buffer. Write-allocate + eviction of dirty lines -> max EA write traffic. | Writes small region in L2 -> dirty lines stay cached, minimal writeback. |
| l2_cache_thrash | TAG_STALL, IB_STALL, LATENCY_FIFO_FULL, eviction rate, Back Pressure Indicator, Internal Resource Pressure | Scattered RMW on 1.5x L2 buffer (prime stride). Fills latency FIFO, causes tag stalls, IB stalls (backpressure). | Coalesced RMW on small working set -> all hits, no stalls. |
| l2_atomic_stress | EA0_ATOMIC, EA0_ATOMIC_LEVEL (atomic latency), TCC_ATOMIC | High-contention atomicAdd on 1024 cachelines across 2x L2 buffer. Forces EA atomic path with high per-atomic latency. | Regular RMW (no atomics) on L2-fitting tile. Each thread works on its own cacheline. |
| l2_coherence_traffic | NC_REQ, UC_REQ, CC_REQ, PROBE, PROBE_EVICT | `fg` mode: fine-grained memory (CC type) -> coherence protocol, internal probes. `nc` mode: `__builtin_nontemporal_store` -> NC traffic. | Coarse-grained memory with normal cached accesses. |
| l2_multigpu_fabric | EA0_RDREQ_GMI_CREDIT_STALL, EA0_WRREQ_GMI_CREDIT_STALL, Remote Access Pressure (IF) | GPU 0 reads/writes memory on GPU 1 via P2P -> Infinity Fabric (GMI) credit exhaustion. | N/A (requires 2 GPUs) |
| l2_io_stress | EA0_RDREQ_IO_CREDIT_STALL, EA0_WRREQ_IO_CREDIT_STALL | GPU kernel accesses host-pinned memory (hipHostMalloc) -> PCIe/IO credit exhaustion. | Same kernel on device-local memory. |
| l2_normalized_throughput | All Normalized Stall Metrics (table 30.10), All Throughput Metrics (table 30.11), Combined Credit Pressure (table 30.12) | Mixed R+W streaming over 4x L2 buffer, purely memory-bound. High stalls relative to GRBM_GUI_ACTIVE. | Compute-heavy FMA kernel with minimal memory -> near-zero normalized stalls. |

### Expected Validation Results

| Workload | Key Metric | Baseline | Optimized |
|---|---|---|---|
| l2_hbm_read_bw_stress | L2-EA read credit stall (HBM) | >10% | <1% |
| l2_hbm_read_bw_stress | L2 hit rate | <20% | >90% |
| l2_hbm_write_bw_stress | L2-EA write credit stall (HBM) | >10% | <1% |
| l2_hbm_write_bw_stress | L2 writeback rate | >30% | <5% |
| l2_cache_thrash | L2 tag stall rate | >10% | <5% |
| l2_cache_thrash | L2 Back Pressure Indicator | >10% | <5% |
| l2_atomic_stress | L2-EA atomic latency | >100 cyc | N/A (no atomics) |
| l2_coherence_traffic (fg) | Coherent cached req rate | >50% | <5% |
| l2_io_stress | L2-EA read credit stall (IO) | >5% | <1% |
| l2_normalized_throughput | Combined Credit Pressure | >10% | <1% |

## Build

```bash
# Build a single workload
hipcc -g <workload>.hip -o <workload> --offload-arch=gfx950

# Build all L2 workloads
for f in l2_*.hip; do
    hipcc -g "$f" -o "${f%.hip}" --offload-arch=gfx950
done
```

## Profiling

```bash
# Profile baseline
rocprof-compute profile -n <name>_baseline --membw-analysis --experimental --no-roof -- ./<workload>

# Profile optimized
rocprof-compute profile -n <name>_optimized --membw-analysis --experimental --no-roof -- ./<workload> opt
```

### Examples

```bash
# HBM read stress
rocprof-compute profile -n hbm_read_baseline --membw-analysis --experimental --no-roof -- ./l2_hbm_read_bw_stress
rocprof-compute profile -n hbm_read_optimized --membw-analysis --experimental --no-roof -- ./l2_hbm_read_bw_stress opt

# Cache thrash
rocprof-compute profile -n thrash_baseline --membw-analysis --experimental --no-roof -- ./l2_cache_thrash
rocprof-compute profile -n thrash_optimized --membw-analysis --experimental --no-roof -- ./l2_cache_thrash opt

# Coherence (fine-grained mode)
rocprof-compute profile -n coherence_fg --membw-analysis --experimental --no-roof -- ./l2_coherence_traffic
rocprof-compute profile -n coherence_nc --membw-analysis --experimental --no-roof -- ./l2_coherence_traffic nc
rocprof-compute profile -n coherence_opt --membw-analysis --experimental --no-roof -- ./l2_coherence_traffic opt

# IO stress (host-pinned vs device-local)
rocprof-compute profile -n io_baseline --membw-analysis --experimental --no-roof -- ./l2_io_stress
rocprof-compute profile -n io_optimized --membw-analysis --experimental --no-roof -- ./l2_io_stress opt

# Multi-GPU (requires 2 GPUs)
rocprof-compute profile -n fabric_read --membw-analysis --experimental --no-roof -- ./l2_multigpu_fabric read
rocprof-compute profile -n fabric_write --membw-analysis --experimental --no-roof -- ./l2_multigpu_fabric write
```

## Analyzing

```bash
rocprof-compute analyze -p <path to profiled result> --membw-analysis --experimental
```

## Hardware Requirements

| Workload | GPUs | Notes |
|---|---|---|
| l2_hbm_read_bw_stress | 1 | Single GPU |
| l2_hbm_write_bw_stress | 1 | Single GPU |
| l2_cache_thrash | 1 | Single GPU |
| l2_atomic_stress | 1 | Single GPU |
| l2_coherence_traffic | 1 | Single GPU; fine-grained alloc may not be supported on all platforms |
| l2_multigpu_fabric | 2 | Requires P2P access between GPUs |
| l2_io_stress | 1 | Single GPU; needs sufficient host memory for 1GB pinned alloc |
| l2_normalized_throughput | 1 | Single GPU |

# EA Bottleneck Detection Workloads

HIP workloads targeting MI350 EA (Efficiency Arbiter) memory bandwidth bottleneck
detection metrics (tables 30.13–30.18 in `3000_mem_bw.yaml`). Each workload has a
**baseline** (exhibits the bottleneck) and **optimized** variant (mitigates it),
selected via the `opt` CLI argument.

## Architecture Reference

```
TCC (L2, 16 channels) --[64B R + 64B W per channel]--> EA ---> DRAM FE --> HBM
                                                           |-> GMI FE  --> Infinity Fabric (remote XCD/AID)
                                                           |-> IO FE   --> PCIe (host)
```

- EA has 6 independent credit pools: 3 destinations (HBM/GMI/IO) x 2 directions (read/write)
- Atomics consume write credits (counted within `TCC_EA0_WRREQ`), but have higher latency (RMW)
- MI350: 16 TCC channels, 128 HBM channels (NPS1), L2 cacheline = 128B

## Workload Summary

| Workload               | Baseline (intended effect)                                       | Optimized (intended effect)                        |
|------------------------|------------------------------------------------------------------|----------------------------------------------------|
| ea_hbm_read_bw         | Streaming reads from 4GB buffer (>> L2) saturates HBM read BW.  | All workgroups re-read 2MB region (fits in L2).    |
| ea_hbm_write_bw        | Streaming writes to 4GB buffer saturates HBM write BW.          | Register accumulation, single write per thread.    |
| ea_write_backpressure  | Coalesced streaming writes at max rate flood EA write credits.   | ALU work between writes throttles write rate.      |
| ea_atomic_pressure     | All threads atomicAdd to 64K-bin global histogram.               | LDS-local histogram, merge with few global atomics.|
| ea_rw_balance          | Read-only streaming sum, single scalar output.                   | 1:1 vector copy (balanced read + write).           |
| ea_fabric_bw           | Streaming reads from remote GPU memory via peer access (GMI).    | Copy remote data to local first, read from HBM.    |
| ea_io_bw               | Streaming reads from host-pinned coherent memory (PCIe).         | Copy to device memory first, read from HBM.        |

> [!NOTE]
> `ea_fabric_bw` requires >= 2 GPUs with peer access. It skips gracefully on single-GPU systems.

## Target Metrics

| Workload               | Primary Metrics (table)                                                        |
|------------------------|--------------------------------------------------------------------------------|
| ea_hbm_read_bw         | EA-HBM read BW (30.14), EA utilization (30.13), Norm HBM read stall (30.16)   |
| ea_hbm_write_bw        | Write Credit Pressure (30.18), Write Backpressure (30.18), EA-HBM write BW    |
| ea_write_backpressure  | Write Backpressure (30.18), Norm Write stall (30.16)                           |
| ea_atomic_pressure     | HBM Atomic Pressure (30.18), atomic fraction of writes (30.13), atomic BW      |
| ea_rw_balance          | Read/Write Balance (30.18), EA-HBM read/write BW (30.14)                      |
| ea_fabric_bw           | IF BW Bound (30.18), read BW fraction IF (30.13), Stall Dominance IF (30.18)  |
| ea_io_bw               | IO BW Bound (30.18), read BW fraction IO (30.13), Stall Dominance IO (30.18)  |

## Build

```bash
cd sample/membw_analysis_test_suite/ea
make            # builds all workloads
make clean      # removes binaries
```

Or individually:

```bash
hipcc -g ea_hbm_read_bw.hip -o ea_hbm_read_bw
```

## Profiling

```bash
# From the rocprofiler-compute root directory:

# Profile baseline
src/rocprof-compute profile -n ea_test_baseline -b 30 \
    --membw-analysis --experimental --no-roof \
    --output-directory /tmp/ea_test_baseline \
    -- ./sample/membw_analysis_test_suite/ea/ea_hbm_read_bw

# Profile optimized
src/rocprof-compute profile -n ea_test_opt -b 30 \
    --membw-analysis --experimental --no-roof \
    --output-directory /tmp/ea_test_opt \
    -- ./sample/membw_analysis_test_suite/ea/ea_hbm_read_bw opt
```

## Analyzing

```bash
src/rocprof-compute analyze --path /tmp/ea_test_baseline \
    -b 30.13 30.14 30.16 30.18 --membw-analysis --experimental

src/rocprof-compute analyze --path /tmp/ea_test_opt \
    -b 30.13 30.14 30.16 30.18 --membw-analysis --experimental
```

## Validation Results (MI350X, single GPU)

All results collected on a single MI350X GPU. `ea_fabric_bw` requires multi-GPU
and is pending validation.

### ea_hbm_read_bw — HBM Read Traffic

| Metric                                  | Baseline       | Optimized     |
|-----------------------------------------|----------------|---------------|
| EA-HBM read BW (30.14.0)               | 6088 Gb/s      | 54 Gb/s       |
| EA-HBM write BW (30.14.1)              | 0.1 Gb/s       | 1.7 Gb/s      |
| EA utilization (30.13.0)                | 99.2%          | 12.7%         |
| EA read latency (30.14.10)             | 1586 cyc       | 1143 cyc      |
| Norm HBM read credit stall (30.16.0)   | 2.01%          | 0.62%         |
| Read Credit Pressure (30.18.0)         | 0.02%          | 0.10%         |
| R/W Balance (30.18.6)                  | 99.99%         | 89.05%        |
| Stall Dominance HBM (30.18.8)          | 100%           | 100%          |

**Key findings:**
- EA-HBM read BW differentiates at **113x** (6088 vs 54 Gb/s). The 2MB tiled
  region fits in L2, so subsequent passes are cache hits.
- EA utilization drops from 99% to 13%, confirming L2 reuse avoids EA.
- Read Credit Pressure (30.18.0) remains low even at near-peak BW. On MI350,
  the HBM read credit pool is deep enough that coalesced streaming reads do not
  exhaust credits. The normalized stall metric (30.16.0) shows a clearer signal
  at 2.01% vs 0.62% using `GRBM_GUI_ACTIVE_PER_XCD` as denominator.

### ea_hbm_write_bw — HBM Write Credit Pressure

| Metric                                  | Baseline       | Optimized     |
|-----------------------------------------|----------------|---------------|
| EA-HBM write BW (30.14.1)              | 4948 Gb/s      | 31 Gb/s       |
| EA utilization (30.13.0)                | 92.3%          | 4.8%          |
| EA write latency (30.14.11)            | 535 cyc        | 397 cyc       |
| Write Credit Pressure (30.18.1)        | **27.25%**     | 6.53%         |
| Write Backpressure (30.18.5)           | **25.13%**     | 6.53%         |
| Norm HBM write credit stall (30.16.1)  | 3203%          | 20.1%         |
| R/W Balance (30.18.6)                  | 0.00%          | 0.15%         |
| Stall Dominance HBM (30.18.8)          | 100%           | 100%          |

**Key findings:**
- Write Credit Pressure at **27.25%** clearly exceeds the 10% bottleneck threshold.
  The optimized kernel accumulates in registers and writes once, reducing write BW
  by 160x (4948 vs 31 Gb/s).
- Write Backpressure (25.13%) closely tracks Write Credit Pressure since both
  derive from write credit stalls on HBM.
- Normalized write credit stall at 3203% reflects per-channel stall accumulation
  across 16 TCC channels summed against a single-XCD activity counter. The 30.18
  metric using `TCC_BUSY_sum` as denominator gives the per-channel percentage.

### ea_write_backpressure — Write Path Backpressure

| Metric                                  | Baseline       | Optimized     |
|-----------------------------------------|----------------|---------------|
| EA-HBM write BW (30.14.1)              | 5774 Gb/s      | 818 Gb/s      |
| EA utilization (30.13.0)                | 92.3%          | 97.3%         |
| EA write latency (30.14.11)            | 429 cyc        | 261 cyc       |
| Write Credit Pressure (30.18.1)        | **13.14%**     | 0.03%         |
| Write Backpressure (30.18.5)           | **13.13%**     | 0.03%         |
| Norm HBM write credit stall (30.16.1)  | 1539%          | 3.5%          |
| Norm Write stall (30.16.6)             | 1538%          | 3.5%          |
| Norm Too many writes stall (30.16.7)   | 0.00%          | 0.00%         |
| Stall Dominance HBM (30.18.8)          | 100%           | 100%          |

**Key findings:**
- Coalesced streaming writes at maximum rate trigger **13.13% Write Backpressure**
  (above 10% threshold), while compute-gated writes drop to 0.03%.
- The ALU work between writes in the optimized kernel reduces the write request
  rate by ~7x, giving EA time to drain credits between bursts.
- `TCC_TOO_MANY_EA_WRREQS_STALL` (30.16.7) is 0% for both variants. This counter
  tracks global write credit pool exhaustion across all TCC channels, which does
  not trigger on MI350 with these patterns. The per-destination credit stall
  (`TCC_EA0_WRREQ_DRAM_CREDIT_STALL`) is the active mechanism.

### ea_atomic_pressure — HBM Atomic Contention

| Metric                                  | Baseline       | Optimized     |
|-----------------------------------------|----------------|---------------|
| EA-HBM atomic BW (30.14.2)             | 831 Gb/s       | 7 Gb/s        |
| EA-HBM read BW (30.14.0)               | 104 Gb/s       | 3597 Gb/s     |
| EA atomic latency (30.14.12)           | 829 cyc        | 3820 cyc      |
| HBM Atomic Pressure (30.18.7)          | **100%**       | **100%**      |
| EA atomic fraction of writes (30.13.6) | 100%           | 100%          |
| Write Credit Pressure (30.18.1)        | 3.17%          | 0.01%         |
| Norm HBM write credit stall (30.16.1)  | 350%           | 0.78%         |
| EA uncached write fraction (30.13.9)   | 100%           | 100%          |
| R/W Balance (30.18.6)                  | 3.03%          | 99.61%        |
| Stall Dominance HBM (30.18.8)          | 100%           | 100%          |

**Key findings:**
- EA-HBM atomic BW differentiates at **118x** (831 vs 7 Gb/s). The LDS histogram
  reduces global atomic traffic from millions of atomics to ~256 per workgroup.
- HBM Atomic Pressure is 100% for both variants. This is correct: the metric
  measures `WRREQ_ATOMIC_DRAM / WRREQ_DRAM` — what fraction of writes are atomics.
  In both kernels the only writes reaching EA are atomics (baseline: all global,
  optimized: the LDS merge atomics). The differentiation is in **volume**, not
  fraction.
- Write Credit Pressure drops from 3.17% to 0.01% because atomics consume write
  credits; fewer atomics = fewer stalls.
- R/W Balance flips from 3% (baseline: mostly atomic writes) to 99.6% (optimized:
  mostly input reads with negligible atomic writes).
- Atomic latency is higher in the optimized case (3820 vs 829 cyc) because fewer
  outstanding requests means less amortization of the per-request RMW round-trip.

### ea_rw_balance — Read/Write Traffic Balance

| Metric                                  | Baseline       | Optimized     |
|-----------------------------------------|----------------|---------------|
| EA-HBM read BW (30.14.0)               | 6001 Gb/s      | 2446 Gb/s     |
| EA-HBM write BW (30.14.1)              | 0.0 Gb/s       | 2446 Gb/s     |
| EA-HBM total BW (30.14.3)              | 6001 Gb/s      | 4892 Gb/s     |
| EA utilization (30.13.0)                | 109.6%         | 94.9%         |
| R/W Balance (30.18.6)                  | **100.00%**    | **33.33%**    |
| Read Credit Pressure (30.18.0)         | 0.03%          | 1.86%         |
| Write Credit Pressure (30.18.1)        | 0.00%          | 7.91%         |
| HBM BW Bound Combined (30.18.2)       | 0.03%          | 9.77%         |
| Stall Dominance HBM (30.18.8)          | 100%           | 100%          |

**Key findings:**
- R/W Balance metric correctly differentiates: **100%** for read-only reduction
  vs **33%** for 1:1 vector copy. The formula `RDREQ / (RDREQ + WRREQ)` reflects
  request-level balance including L2 write-allocate fetches.
- The vector copy shows perfectly symmetric BW: 2446 Gb/s read = 2446 Gb/s write.
- HBM BW Bound Combined reaches 9.77% for the copy (both read and write credit
  stalls contributing), compared to 0.03% for the read-only baseline where write
  credits are never consumed.

### ea_io_bw — IO/PCIe Bandwidth

| Metric                                  | Baseline       | Optimized     |
|-----------------------------------------|----------------|---------------|
| EA-IO read BW (30.14.7)                | 56.6 Gb/s      | 0.0 Gb/s      |
| EA-HBM read BW (30.14.0)               | 2.5 Gb/s       | 6214 Gb/s     |
| EA read BW fraction - IO (30.13.4)     | **95.80%**     | 0.00%         |
| EA read BW fraction - HBM (30.13.2)    | 4.20%          | 100.00%       |
| EA read latency (30.14.10)             | **27,251 cyc** | 1,464 cyc     |
| IO BW Bound (30.18.4)                  | **10.48%**     | 0.00%         |
| Stall Dominance - IO (30.18.10)        | **100%**       | 0%            |
| Stall Dominance - HBM (30.18.8)        | 0%             | 100%          |
| Norm IO read credit stall (30.16.4)    | 864.5%         | 0.00%         |
| EA utilization (30.13.0)                | 71.8%          | 94.2%         |
| Wall-clock time                         | 51 ms          | 2 ms          |

**Key findings:**
- IO BW Bound at **10.48%** exceeds the 5% threshold, while optimized is 0%. This
  is the cleanest bottleneck detection result across all workloads.
- EA read BW fraction - IO correctly identifies **95.8%** of read traffic going
  via PCIe in the baseline. After copying to device memory, it drops to 0%.
- Stall Destination Dominance perfectly flips: IO 100% (baseline) to HBM 100%
  (optimized). This cross-validates the dominance metric.
- EA-IO read BW at 56.6 Gb/s is consistent with PCIe Gen5 x16 bandwidth (~64 GB/s
  theoretical peak).
- Read latency is **18.6x higher** via PCIe (27,251 vs 1,464 cycles), reflecting
  the ~1us PCIe round-trip vs ~50-100ns HBM latency.

### ea_fabric_bw — IF/GMI Bandwidth (multi-GPU)

Not yet validated. Requires >= 2 GPUs with peer access. The workload correctly
detects single-GPU systems and skips with a message.

**Expected results on multi-GPU:**

| Metric                                  | Baseline (expected) | Optimized (expected) |
|-----------------------------------------|---------------------|----------------------|
| EA read BW fraction - IF (30.13.3)     | > 90%               | 0%                   |
| IF BW Bound (30.18.3)                  | > 10%               | ~ 0%                 |
| Stall Dominance - IF (30.18.9)         | > 80%               | 0%                   |
| Stall Dominance - HBM (30.18.8)        | ~ 0%                | 100%                 |

## Cross-Validation Summary

### Bottleneck Detection Thresholds (30.18)

| Workload               | Key Metric (30.18.x)     | Baseline  | Optimized | Threshold | Pass |
|------------------------|--------------------------|-----------|-----------|-----------|------|
| ea_hbm_write_bw        | Write Credit Pressure    | 27.25%    | 6.53%     | >= 10%    | Yes  |
| ea_write_backpressure  | Write Backpressure       | 13.13%    | 0.03%     | >= 10%    | Yes  |
| ea_rw_balance          | Read/Write Balance       | 100.00%   | 33.33%    | N/A       | Yes  |
| ea_atomic_pressure     | HBM Atomic Pressure      | 100%      | 100%      | N/A       | (1)  |
| ea_io_bw               | IO BW Bound              | 10.48%    | 0.00%     | >= 5%     | Yes  |
| ea_hbm_read_bw         | Read Credit Pressure     | 0.02%     | 0.10%     | >= 10%    | (2)  |

**(1)** Atomic Pressure measures fraction (not volume). Both variants are 100%
atomic because neither has non-atomic writes. Differentiation is via atomic BW
(831 vs 7 Gb/s, 118x) and write credit stall (350% vs 0.78%).

**(2)** Read Credit Pressure stays below threshold even at ~6 TB/s read BW.
MI350's deep HBM read credit pools do not exhaust under coalesced streaming
patterns. Differentiation is via EA-HBM read BW (6088 vs 54 Gb/s, 113x) and
EA utilization (99% vs 13%).

### Stall Destination Dominance Cross-Validation

| Workload (baseline)                        | HBM    | IF     | IO     |
|--------------------------------------------|--------|--------|--------|
| ea_hbm_read_bw, ea_hbm_write_bw           | 100%   | 0%     | 0%     |
| ea_write_backpressure, ea_atomic_pressure  | 100%   | 0%     | 0%     |
| ea_rw_balance                              | 100%   | 0%     | 0%     |
| ea_io_bw                                   | 0%     | 0%     | 100%   |
| ea_fabric_bw (expected)                    | ~0%    | >80%   | 0%     |

The Stall Destination Dominance metric correctly identifies the dominant stall
source across all validated workloads, including the HBM-to-IO flip in ea_io_bw.

## Observations on MI350 Credit Stall Behavior

1. **Write credits exhaust more readily than read credits.** Streaming writes at
   peak BW trigger 27% Write Credit Pressure, while streaming reads at similar BW
   show only 0.02% Read Credit Pressure. This suggests MI350 has a deeper read
   credit pool or faster read credit retirement.

2. **`TCC_TOO_MANY_EA_WRREQS_STALL` does not fire** for any workload tested. The
   global write credit pool exhaustion condition appears difficult to trigger on
   MI350. The per-destination credit stall (`TCC_EA0_WRREQ_DRAM_CREDIT_STALL`)
   is the primary write bottleneck indicator.

3. **30.16 normalized stalls can exceed 100%** because stall counters are summed
   across 16 TCC channels while the denominator (`GRBM_GUI_ACTIVE_PER_XCD`) is a
   single-XCD value. Values like 3203% mean each channel was stalled for ~32x the
   kernel duration on average. The 30.18 metrics using `TCC_BUSY_sum` give the
   more intuitive per-channel percentage.

4. **Scattered writes do not trigger EA backpressure.** They bottleneck upstream
   at L1/L2 coalescing, throttling the EA write request rate. Only coalesced
   streaming writes at maximum rate exhaust EA write credits.

# EA Bottleneck Detection Workloads

HIP workloads targeting MI350 EA (Efficiency Arbiter) memory bandwidth bottleneck
detection metrics (tables 30.13--30.18 in `3000_mem_bw.yaml`). Each workload has a
**baseline** (exhibits the bottleneck) and **optimized** variant (mitigates it),
selected via the `opt` CLI argument.

## Architecture Reference

```
TCC (L2, 16 channels) --[64B R + 64B W per channel]--> EA ---> DRAM FE --> HBM
                                                           |-> GMI FE  --> GMI (remote XCD/AID)
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

| Workload               | Primary Metrics (table)                                                                     |
|------------------------|---------------------------------------------------------------------------------------------|
| ea_hbm_read_bw         | EA to HBM read BW (30.14), EA utilization (30.13), Norm HBM read stall (30.16)             |
| ea_hbm_write_bw        | Write Credit Pressure (30.18), Write Backpressure (30.18), EA to HBM write BW (30.14)      |
| ea_write_backpressure  | Write Backpressure (30.18), Norm Write stall (30.16)                                        |
| ea_atomic_pressure     | HBM Atomic Pressure (30.18), atomic fraction of writes (30.13), EA to HBM atomic BW (30.14)|
| ea_rw_balance          | Read/Write Balance (30.18), EA to HBM read/write BW (30.14)                                |
| ea_fabric_bw           | GMI BW Bound (30.18), read BW fraction GMI (30.13), Stall Dominance GMI (30.18)            |
| ea_io_bw               | IO BW Bound (30.18), read BW fraction IO (30.13), Stall Dominance IO (30.18)               |

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
# Note: --no-roof is optional and skips the
# benchmarking that is unrelated to the our purpose

# From the rocprofiler-compute root directory:

# Profile baseline
src/rocprof-compute profile -n ea_test_baseline \
    -b 30.13 30.14 30.15 30.16 30.17 30.18 \
    --experimental --membw-analysis --no-roof \
    --output-directory /tmp/ea_test_baseline \
    -- ./sample/membw_analysis_test_suite/ea/ea_hbm_read_bw

# Profile optimized
src/rocprof-compute profile -n ea_test_opt \
    -b 30.13 30.14 30.15 30.16 30.17 30.18 \
    --experimental --membw-analysis --no-roof \
    --output-directory /tmp/ea_test_opt \
    -- ./sample/membw_analysis_test_suite/ea/ea_hbm_read_bw opt
```

## Analyzing

Side-by-side baseline vs. optimized comparison:

```bash
src/rocprof-compute analyze \
    -p /tmp/ea_test_baseline -p /tmp/ea_test_opt \
    -b 30.13 30.14 30.15 30.16 30.17 30.18 --experimental --membw-analysis
```

To analyze each run independently:

```bash
src/rocprof-compute analyze -p /tmp/ea_test_baseline \
    -b 30.13 30.14 30.15 30.16 30.17 30.18 --experimental --membw-analysis

src/rocprof-compute analyze -p /tmp/ea_test_opt \
    -b 30.13 30.14 30.15 30.16 30.17 30.18 --experimental --membw-analysis
```

## Validation Results (MI350X)

### Test Environment

| Component          | Configuration                                      |
|--------------------|----------------------------------------------------|
| GPU                | 2x AMD Instinct MI350X (gfx950)                    |
| GPU Architecture   | CDNA 4 (gfx950), 256 CUs, 32 SIMDs/CU, 8 XCDs     |
| GPU SCLK           | 2200 MHz (max)                                     |
| GPU MCLK           | 1900 MHz (HBM)                                     |
| HBM                | 288 GB per GPU, 128 HBM channels, NPS1              |
| L2 Cache           | 4096 KB, 128 channels, 16 banks                    |
| L1 Cache           | 32 KB per CU                                       |
| VBIOS              | 113-M350-01-1K5-020E                               |
| Compute Partition  | SPX                                                |
| Memory Partition   | NPS1                                                |
| CPU                | AMD EPYC 9655 96-Core (Turin)                      |
| OS                 | Ubuntu 24.04 LTS, kernel 6.8.0-31-generic          |
| ROCm               | 7.2.0                                              |
| GPU Driver         | amdgpu 6.16.13                                     |

All results collected on MI350X GPUs. `ea_fabric_bw` validated on the multi-GPU
system with peer access enabled between GPU 0 and GPU 1.

### ea_hbm_read_bw -- HBM Read Traffic

| Metric                                            | Baseline       | Optimized     |
|---------------------------------------------------|----------------|---------------|
| EA to HBM read BW (30.14.0)                       | 6079 Gb/s      | 53 Gb/s       |
| EA to HBM write BW (30.14.1)                      | 0 Gb/s         | 2 Gb/s        |
| EA utilization (30.13.0)                           | 99.42%         | 13.16%        |
| EA read average latency (30.14.10)                 | 1589 cyc       | 1147 cyc      |
| EA read latency (30.14.11)                         | 1589 cyc       | 1147 cyc      |
| Normalized EA - HBM read credit stall (30.16.0)   | 1.73%          | 0.72%         |
| EA HBM BW Bound - Read Credit Pressure (30.18.0)  | 0.01%          | 0.11%         |
| EA Read/Write Balance (30.18.6)                    | 99.99%         | 89.05%        |
| EA Stall Destination Dominance - HBM (30.18.8)    | 100.00%        | 100.00%       |

**Key findings:**
- EA to HBM read BW differentiates at **115x** (6079 vs 53 Gb/s). The 2MB tiled
  region fits in L2, so subsequent passes are cache hits.
- EA utilization drops from 99% to 13%, confirming L2 reuse avoids EA.
- Read Credit Pressure (30.18.0) remains low even at near-peak BW. On MI350,
  the HBM read credit pool is deep enough that coalesced streaming reads do not
  exhaust credits. The normalized stall metric (30.16.0) shows a clearer signal
  at 1.73% vs 0.72% using `GRBM_GUI_ACTIVE_PER_XCD` as denominator.

### ea_hbm_write_bw -- HBM Write Credit Pressure

| Metric                                             | Baseline       | Optimized     |
|----------------------------------------------------|----------------|---------------|
| EA to HBM write BW (30.14.1)                       | 4855 Gb/s      | 31 Gb/s       |
| EA utilization (30.13.0)                            | 89.96%         | 3.90%         |
| EA write average latency (30.14.12)                 | 530 cyc        | 404 cyc       |
| EA write latency (30.14.13)                         | 530 cyc        | 404 cyc       |
| EA HBM BW Bound - Write Credit Pressure (30.18.1)  | **25.06%**     | 6.68%         |
| EA Write Backpressure (30.18.5)                     | **25.21%**     | 6.77%         |
| Normalized EA - HBM write credit stall (30.16.1)   | 2962%          | 20.30%        |
| EA Read/Write Balance (30.18.6)                     | 0.00%          | 0.15%         |
| EA Stall Destination Dominance - HBM (30.18.8)     | 100.00%        | 100.00%       |

**Key findings:**
- Write Credit Pressure at **25.06%** clearly exceeds the 10% bottleneck threshold.
  The optimized kernel accumulates in registers and writes once, reducing write BW
  by 157x (4855 vs 31 Gb/s).
- Write Backpressure (25.21%) closely tracks Write Credit Pressure since both
  derive from write credit stalls on HBM.
- Normalized write credit stall at 2962% reflects per-channel stall accumulation
  across 16 TCC channels summed against a single-XCD activity counter. The 30.18
  metric using `TCC_BUSY_sum` as denominator gives the per-channel percentage.

### ea_write_backpressure -- Write Path Backpressure

| Metric                                             | Baseline       | Optimized     |
|----------------------------------------------------|----------------|---------------|
| EA to HBM write BW (30.14.1)                       | 5719 Gb/s      | 820 Gb/s      |
| EA utilization (30.13.0)                            | 90.17%         | 97.93%        |
| EA write average latency (30.14.12)                 | 428 cyc        | 258 cyc       |
| EA write latency (30.14.13)                         | 428 cyc        | 258 cyc       |
| EA HBM BW Bound - Write Credit Pressure (30.18.1)  | **13.20%**     | 0.03%         |
| EA Write Backpressure (30.18.5)                     | **13.13%**     | 0.03%         |
| Normalized EA - HBM write credit stall (30.16.1)   | 1524%          | 3.54%         |
| Normalized EA - Write stall (30.16.6)               | 1517%          | 3.56%         |
| Normalized EA - Too many writes stall (30.16.7)     | 0.00%          | 0.00%         |
| EA Stall Destination Dominance - HBM (30.18.8)     | 100.00%        | 100.00%       |

**Key findings:**
- Coalesced streaming writes at maximum rate trigger **13.13% Write Backpressure**
  (above 10% threshold), while compute-gated writes drop to 0.03%.
- The ALU work between writes in the optimized kernel reduces the write request
  rate by ~7x, giving EA time to drain credits between bursts.
- `TCC_TOO_MANY_EA_WRREQS_STALL` (30.16.7) is 0% for both variants. This counter
  tracks global write credit pool exhaustion across all TCC channels, which does
  not trigger on MI350 with these patterns. The per-destination credit stall
  (`TCC_EA0_WRREQ_DRAM_CREDIT_STALL`) is the active mechanism.

### ea_atomic_pressure -- HBM Atomic Contention

| Metric                                             | Baseline       | Optimized     |
|----------------------------------------------------|----------------|---------------|
| EA to HBM atomic BW (30.14.2)                      | 836 Gb/s       | 7 Gb/s        |
| EA to HBM read BW (30.14.0)                        | 105 Gb/s       | 3581 Gb/s     |
| EA atomic average latency (30.14.14)                | 747 cyc        | 3443 cyc      |
| EA atomic latency (30.14.15)                        | 747 cyc        | 3443 cyc      |
| EA HBM Atomic Pressure (30.18.7)                    | **100.00%**    | **100.00%**   |
| EA atomic fraction of writes (30.13.6)              | 100.00%        | 100.00%       |
| EA HBM BW Bound - Write Credit Pressure (30.18.1)  | 2.06%          | 0.03%         |
| Normalized EA - HBM write credit stall (30.16.1)   | 246%           | 4.05%         |
| EA uncached write fraction (30.13.9)                | 100.00%        | 100.00%       |
| EA Read/Write Balance (30.18.6)                     | 3.03%          | 99.61%        |
| EA Stall Destination Dominance - HBM (30.18.8)     | 100.00%        | 100.00%       |

**Key findings:**
- EA to HBM atomic BW differentiates at **119x** (836 vs 7 Gb/s). The LDS histogram
  reduces global atomic traffic from millions of atomics to ~256 per workgroup.
- HBM Atomic Pressure is 100% for both variants. This is correct: the metric
  measures `WRREQ_ATOMIC_DRAM / WRREQ_DRAM` -- what fraction of writes are atomics.
  In both kernels the only writes reaching EA are atomics (baseline: all global,
  optimized: the LDS merge atomics). The differentiation is in **volume**, not
  fraction.
- Write Credit Pressure drops from 2.06% to 0.03% because atomics consume write
  credits; fewer atomics = fewer stalls.
- R/W Balance flips from 3% (baseline: mostly atomic writes) to 99.6% (optimized:
  mostly input reads with negligible atomic writes).
- Atomic latency is higher in the optimized case (3443 vs 747 cyc) because fewer
  outstanding requests means less amortization of the per-request RMW round-trip.

### ea_rw_balance -- Read/Write Traffic Balance

| Metric                                             | Baseline       | Optimized      |
|----------------------------------------------------|----------------|----------------|
| EA to HBM read BW (30.14.0)                        | 6203 Gb/s      | 2580 Gb/s      |
| EA to HBM write BW (30.14.1)                       | 0 Gb/s         | 2580 Gb/s      |
| EA to HBM total BW (30.14.3)                       | 6203 Gb/s      | 5159 Gb/s      |
| EA utilization (30.13.0)                            | 95.15%         | 95.61%         |
| EA Read/Write Balance (30.18.6)                     | **100.00%**    | **33.33%**     |
| EA HBM BW Bound - Read Credit Pressure (30.18.0)   | 0.03%          | 2.12%          |
| EA HBM BW Bound - Write Credit Pressure (30.18.1)  | 0.00%          | 2.16%          |
| EA HBM BW Bound - Combined (30.18.2)               | 0.03%          | 4.28%          |
| EA Stall Destination Dominance - HBM (30.18.8)     | 100.00%        | 100.00%        |

**Key findings:**
- R/W Balance metric correctly differentiates: **100%** for read-only reduction
  vs **33%** for 1:1 vector copy. The formula `RDREQ / (RDREQ + WRREQ)` reflects
  request-level balance including L2 write-allocate fetches.
- The vector copy shows perfectly symmetric BW: 2580 Gb/s read = 2580 Gb/s write.
- HBM BW Bound Combined reaches 4.28% for the copy (both read and write credit
  stalls contributing), compared to 0.03% for the read-only baseline where write
  credits are never consumed.

### ea_io_bw -- IO/PCIe Bandwidth

| Metric                                             | Baseline       | Optimized     |
|----------------------------------------------------|----------------|---------------|
| EA to IO read BW (30.14.7)                          | 56.6 Gb/s      | 0.0 Gb/s      |
| EA to HBM read BW (30.14.0)                         | 2.6 Gb/s       | 6276 Gb/s     |
| EA read BW fraction - IO (30.13.4)                  | **95.59%**     | 0.00%         |
| EA read BW fraction - HBM (30.13.2)                 | 4.41%          | 100.00%       |
| EA read average latency (30.14.10)                   | **27533 cyc**  | 1448 cyc      |
| EA read latency (30.14.11)                           | **27533 cyc**  | 1448 cyc      |
| EA IO BW Bound - Combined (30.18.4)                 | **9.41%**      | 0.00%         |
| EA Stall Destination Dominance - IO (30.18.10)      | **100.00%**    | 0.00%         |
| EA Stall Destination Dominance - HBM (30.18.8)     | 0.00%          | 100.00%       |
| Normalized EA - IO read credit stall (30.16.4)      | 778.58%        | 0.00%         |
| EA utilization (30.13.0)                             | 72.11%         | 99.69%        |

**Key findings:**
- EA read BW fraction - IO correctly identifies **95.6%** of read traffic going
  via PCIe in the baseline. After copying to device memory, it drops to 0%.
- Stall Destination Dominance perfectly flips: IO 100% (baseline) to HBM 100%
  (optimized). This cross-validates the dominance metric.
- EA to IO read BW at 56.6 Gb/s is consistent with PCIe Gen5 x16 bandwidth (~64 GB/s
  theoretical peak).
- Read latency is **19x higher** via PCIe (27,533 vs 1,448 cycles), reflecting
  the ~1us PCIe round-trip vs ~50-100ns HBM latency.
- IO BW Bound at 9.41% is near the bottleneck threshold. The normalized stall
  metric (30.16.4 = 779%) and Stall Destination Dominance (100% IO) provide
  stronger bottleneck indicators for IO-bound workloads.

### ea_fabric_bw -- GMI Bandwidth (multi-GPU)

Validated on a multi-GPU MI350X system with peer access enabled.

| Metric                                             | Baseline       | Optimized     |
|----------------------------------------------------|----------------|---------------|
| EA to GMI read BW (30.14.4)                         | 61.8 Gb/s      | 0.0 Gb/s      |
| EA to HBM read BW (30.14.0)                         | 0.0 Gb/s       | 6244 Gb/s     |
| EA read BW fraction - GMI (30.13.3)                 | **100.00%**    | 0.00%         |
| EA read BW fraction - HBM (30.13.2)                 | 0.00%          | 100.00%       |
| EA read average latency (30.14.10)                   | **21858 cyc**  | 1492 cyc      |
| EA read latency (30.14.11)                           | **21858 cyc**  | 1492 cyc      |
| EA GMI BW Bound - Combined (30.18.3)                | **15.08%**     | 0.00%         |
| EA Stall Destination Dominance - GMI (30.18.9)      | **100.00%**    | 0.00%         |
| EA Stall Destination Dominance - HBM (30.18.8)     | 0.00%          | 100.00%       |
| Normalized EA - GMI read credit stall (30.16.2)     | 1347.62%       | 0.00%         |
| EA utilization (30.13.0)                             | 99.72%         | 96.43%        |

**Key findings:**
- GMI BW Bound at **15.08%** clearly exceeds the 10% threshold, confirming
  that streaming reads from remote GPU memory via GMI saturate GMI credits.
  After copying to local HBM, it drops to 0%.
- EA read BW fraction - GMI correctly identifies **100%** of read traffic going
  via GMI in the baseline. After local copy, it drops to 0%.
- Stall Destination Dominance perfectly flips: GMI 100% (baseline) to HBM 100%
  (optimized). This cross-validates the dominance metric for the GMI destination.
- EA to GMI read BW at 61.8 Gb/s reflects the inter-GPU GMI link bandwidth.
- Read latency is **14.6x higher** via GMI (21,858 vs 1,492 cycles), reflecting
  the remote memory access round-trip latency.

## Cross-Validation Summary

### Bottleneck Detection Thresholds (30.18)

| Workload               | Key Metric (30.18.x)     | Baseline  | Optimized | Threshold | Pass |
|------------------------|--------------------------|-----------|-----------|-----------|------|
| ea_hbm_write_bw        | Write Credit Pressure    | 25.06%    | 6.68%     | >= 10%    | Yes  |
| ea_write_backpressure  | Write Backpressure       | 13.13%    | 0.03%     | >= 10%    | Yes  |
| ea_rw_balance          | Read/Write Balance       | 100.00%   | 33.33%    | N/A       | Yes  |
| ea_atomic_pressure     | HBM Atomic Pressure      | 100%      | 100%      | N/A       | (1)  |
| ea_fabric_bw           | GMI BW Bound             | 15.08%    | 0.00%     | >= 10%    | Yes  |
| ea_io_bw               | IO BW Bound              | 9.41%     | 0.00%     | >= 5%     | Yes  |
| ea_hbm_read_bw         | Read Credit Pressure     | 0.01%     | 0.11%     | >= 10%    | (2)  |

**(1)** Atomic Pressure measures fraction (not volume). Both variants are 100%
atomic because neither has non-atomic writes. Differentiation is via atomic BW
(836 vs 7 Gb/s, 119x) and write credit stall (246% vs 4.05%).

**(2)** Read Credit Pressure stays below threshold even at ~6 TB/s read BW.
MI350's deep HBM read credit pools do not exhaust under coalesced streaming
patterns. Differentiation is via EA to HBM read BW (6079 vs 53 Gb/s, 115x) and
EA utilization (99% vs 13%).

### Stall Destination Dominance Cross-Validation

| Workload (baseline)                        | HBM    | GMI    | IO     |
|--------------------------------------------|--------|--------|--------|
| ea_hbm_read_bw, ea_hbm_write_bw           | 100%   | 0%     | 0%     |
| ea_write_backpressure, ea_atomic_pressure  | 100%   | 0%     | 0%     |
| ea_rw_balance                              | 100%   | 0%     | 0%     |
| ea_io_bw                                   | 0%     | 0%     | 100%   |
| ea_fabric_bw                               | 0%     | 100%   | 0%     |

The Stall Destination Dominance metric correctly identifies the dominant stall
source across all validated workloads, including the HBM-to-IO flip in ea_io_bw
and the GMI dominance in ea_fabric_bw.

## Observations on MI350 Credit Stall Behavior

1. **Write credits exhaust more readily than read credits.** Streaming writes at
   peak BW trigger 25% Write Credit Pressure, while streaming reads at similar BW
   show only 0.01% Read Credit Pressure. This suggests MI350 has a deeper read
   credit pool or faster read credit retirement.

2. **`TCC_TOO_MANY_EA_WRREQS_STALL` does not fire** for any workload tested. The
   global write credit pool exhaustion condition appears difficult to trigger on
   MI350. The per-destination credit stall (`TCC_EA0_WRREQ_DRAM_CREDIT_STALL`)
   is the primary write bottleneck indicator.

3. **30.16 normalized stalls can exceed 100%** because stall counters are summed
   across 16 TCC channels while the denominator (`GRBM_GUI_ACTIVE_PER_XCD`) is a
   single-XCD value. Values like 2962% mean each channel was stalled for ~30x the
   kernel duration on average. The 30.18 metrics using `TCC_BUSY_sum` give the
   more intuitive per-channel percentage.

4. **Scattered writes do not trigger EA backpressure.** They bottleneck upstream
   at L1/L2 coalescing, throttling the EA write request rate. Only coalesced
   streaming writes at maximum rate exhaust EA write credits.

5. **GMI latency is comparable to IO/PCIe latency.** Read latency via GMI
   (21,858 cyc) is similar to IO/PCIe latency (27,533 cyc), both being an order
   of magnitude higher than HBM latency (~1,500 cyc). This reflects the inherent
   cost of crossing chip/package boundaries.

6. **Latency metrics show both "average latency" (SUM/SUM) and "latency"
   (AVG/AVG) variants.** In all workloads tested on MI350, the two variants
   report identical values. This is expected for single-dispatch profiling since
   there is only one data point, making SUM/SUM and AVG/AVG equivalent.

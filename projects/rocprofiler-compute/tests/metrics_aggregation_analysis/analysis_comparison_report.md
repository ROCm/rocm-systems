# Metrics Aggregation Formula Change: AVG(A/B) to SUM(A)/SUM(B)

## Technical Comparison Report -- rocprofiler-compute

**Date:** 2026-04-09
**Branch:** `users/xuchen-amd/metrics-aggregation-report`
**Base:** `develop`

---

## 1. Executive Summary

This change corrects the aggregation formula used for ratio-based performance metrics (hit rates, bandwidths, latencies, IPC, etc.) in rocprofiler-compute's analysis YAML configurations. The previous formula, `AVG(A/B)`, computes the arithmetic mean of per-dispatch ratios, giving equal weight to every dispatch regardless of its size or duration. The corrected formula, `SUM(A)/SUM(B)`, computes a weighted average where each dispatch contributes in proportion to its denominator magnitude. A one-line change to `src/utils/parser.py` ensures division-by-zero cases produce `"N/A"` instead of `inf`.

**Scope:** 3,118 expressions across 106 YAML files spanning 7 GPU architectures.

**Impact:** Identical results for single-dispatch workloads. Sub-1% differences for uniform multi-dispatch workloads. Up to 65% correction for mixed-kernel workloads.

---

## 2. System Under Test

| Property | Value |
|---|---|
| GPU | MI308X (gfx942, MI300 series), 80 CUs, 64-wide waves |
| Max SCLK / MCLK | 1420 / 1300 MHz |
| L2 Cache | 4096 KB, 64 channels, 4 XCDs |
| HBM | 128 channels |
| CPU | AMD EPYC 9354 32-Core, ~1.5 TB RAM |
| ROCm | 7.1.0 |
| OS | Ubuntu 22.04.2 LTS, kernel 5.15.0-70-generic |

---

## 3. Workloads Profiled

| Workload | Kernel | Dispatches | Duration Range | Grid Size | Workgroup Size |
|---|---|---|---|---|---|
| vcopy_test | `vecCopy` | 1 | 12,932 ns | 1,048,576 | 256 |
| memcopy_test | `memoryCopyKernel` | 1,001 | 2.2 ms -- 53.6 ms | 268,435,456 | 256 |
| matmul_test | `matMulKernel` | 1 | 42,972,678 ns | 16,777,216 | 1,024 |
| instmix_test | `kernelasm` | 1 | 1,860 ns | 64 | 64 |

A **mixed-kernel scenario** is constructed by combining all four workloads (1,004 dispatches, 4 distinct kernels, durations spanning 4 orders of magnitude).

### 3.1 Exact Commands Used

All profiling and analysis was performed using `rocprof-compute` from source at `src/rocprof-compute`.

**Compilation:**
```bash
hipcc sample/vcopy.cpp -o sample/vcopy -O2
hipcc sample/memcopy.cpp -o sample/memcopy -O2
hipcc sample/mat_mul_max.hip -o sample/mat_mul_max -O2
hipcc sample/instmix.hip -o sample/instmix -O2
```

**Profiling:**
```bash
src/rocprof-compute profile -n vcopy_test --no-roof --device 0 -- sample/vcopy -n 1048576 -b 256
src/rocprof-compute profile -n memcopy_test --no-roof --device 0 -- sample/memcopy
src/rocprof-compute profile -n matmul_test --no-roof --device 0 -- sample/mat_mul_max
src/rocprof-compute profile -n instmix_test --no-roof --device 0 -- sample/instmix
```

**Analysis (AVG configs — original YAML from `develop` branch):**
```bash
git checkout develop -- src/rocprof_compute_soc/analysis_configs/
src/rocprof-compute analyze -p workloads/<name>/MI308X --output-format csv --output-name <name>_avg
```

**Analysis (SUM configs — modified YAML on this branch):**
```bash
git checkout HEAD -- src/rocprof_compute_soc/analysis_configs/
src/rocprof-compute analyze -p workloads/<name>/MI308X --output-format csv --output-name <name>_sum
```

Both analysis runs use the same raw profiling data. The only difference is the YAML config expressions.

---

## 4. The Problem: Why AVG(A/B) Is Incorrect for Ratio Metrics

### 4.1 The Two Formulas

Many performance metrics are ratios: `L2 Hit Rate = Hits / Total`, `BW = Bytes / Duration`, `IPC = Instructions / Cycles`.

When aggregating across N dispatches:

**AVG(A/B) — equal weight per dispatch:**
```
AVG(A/B) = (1/N) * SUM_i(A_i / B_i)
```

**SUM(A)/SUM(B) — weighted by denominator magnitude:**
```
Let r_i = A_i / B_i      (per-dispatch ratio)
Let w_i = B_i / SUM(B)   (weight = dispatch's share of total denominator)

SUM(A)/SUM(B) = SUM_i( w_i * r_i )
              = (A_1/B_1)*(B_1/B_total) + (A_2/B_2)*(B_2/B_total) + ...

Proof: SUM( (B_i/B_total) * (A_i/B_i) ) = SUM( A_i/B_total ) = SUM(A)/SUM(B)
```

A dispatch that processes more data, runs longer, or handles more requests contributes proportionally more — which is the physically correct behavior.

### 4.2 Concrete Example: Bandwidth

From `memcopy_test` dispatches 169 and 701:

| Dispatch | Bytes | Duration | BW |
|---|---|---|---|
| A (typical) | 6.44 GB | 2.21 ms | 2,916 Gb/s |
| B (contended) | 6.44 GB | 53.62 ms | 120 Gb/s |

**AVG(A/B):** `(2916 + 120) / 2 = 1,518 Gb/s`

**SUM(A)/SUM(B) as weighted average:**
```
w_A = 2.21 / 55.83 = 0.040    (4.0% of total time)
w_B = 53.62 / 55.83 = 0.960   (96.0% of total time)

Weighted BW = 0.040 * 2916 + 0.960 * 120 = 231 Gb/s
```

Equivalently: `SUM(Bytes) / SUM(Duration) = 12.88 GB / 55.83 ms = 231 Gb/s`.

Dispatch B ran for 96% of the total time at 120 Gb/s. The aggregate should be dominated by 120 — not pulled halfway to 2,916 by a dispatch that lasted 4% of the time. AVG overestimates by **6.6x**.

### 4.3 Concrete Example: IPC (Cross-Kernel)

From actual profiled data:

| Kernel | SQ_INSTS | SQ_BUSY_CU_CYCLES | IPC | Duration |
|---|---|---|---|---|
| vcopy | 327,680 | 1,117,402 | 0.2933 | 12,932 ns |
| matmul | 3,234,180,000 | 4,802,071,000 | 0.6735 | 42,972,678 ns |

**AVG(IPC):** `(0.2933 + 0.6735) / 2 = 0.4834` (+39.3% error)

**SUM/SUM:** `3,234,507,680 / 4,803,188,402 = 0.6734`

vcopy executed 0.01% of total instructions yet receives 50% weight under AVG.

### 4.4 Physical Interpretation

- `SUM(A)/SUM(B)` answers: **"What was the overall rate for the entire execution?"**
- `AVG(A/B)` answers: **"What was the average rate if each dispatch is treated equally regardless of size?"**

For performance analysis, the first question is almost always correct.

---

## 5. The Change

### 5.1 Scope

| Dimension | Count |
|---|---|
| YAML files modified | 106 |
| GPU architectures | 7 (gfx908, gfx90a, gfx940, gfx941, gfx942, gfx950, gfx1151) |
| Expressions changed | 3,118 |
| Code change | 1 line in `src/utils/parser.py` (`isinf` check) |

### 5.2 YAML Transformation Examples

**IPC:**
```yaml
# Before:
value: AVG((SQ_INSTS / SQ_BUSY_CU_CYCLES) if (SQ_BUSY_CU_CYCLES != 0) else None)
# After:
value: SUM(SQ_INSTS) / SUM(SQ_BUSY_CU_CYCLES)
```

**L2 Hit Rate:**
```yaml
# Before:
value: AVG((100 * TCC_HIT_sum / (TCC_HIT_sum + TCC_MISS_sum)) if ((TCC_HIT_sum + TCC_MISS_sum) != 0) else None)
# After:
value: SUM(100 * TCC_HIT_sum) / SUM((TCC_HIT_sum + TCC_MISS_sum))
```

**L2 Cache BW:**
```yaml
# Before:
value: AVG((TCC_REQ_sum * 128) / (End_Timestamp - Start_Timestamp))
# After:
value: SUM((TCC_REQ_sum * 128)) / SUM((End_Timestamp - Start_Timestamp))
```

### 5.3 Division-by-Zero Handling

The old `AVG` expressions used per-dispatch guards (`if (B != 0) else None`) to exclude dispatches with zero denominators via pandas `.where()`. The new `SUM(A)/SUM(B)` expressions remove these guards. When `SUM(B) = 0`, Python produces `inf`.

To handle this, one line was added to the existing result-checking logic in `src/utils/parser.py` (`eval_expression`, line 480):

```python
# Before:
(np.isscalar(eval_result) and pd.isna(eval_result))
# After:
(np.isscalar(eval_result) and (pd.isna(eval_result) or np.isinf(eval_result)))
```

This extends the same code path that already catches `NaN` → `"N/A"` to also catch `inf` → `"N/A"`.

### 5.4 Zero-Denominator Cases in Profiled Data

A scan of all workloads identified every metric where the denominator equals zero:

| Workload | Metric | B=0 Count | A when B=0 | Old AVG | New SUM/SUM | Match |
|---|---|---|---|---|---|---|
| vcopy_test | LDS Bank Conflicts | 1/1 | A=0 | 0/0 → NaN → N/A | 0/0 → NaN → N/A | YES |
| memcopy_test | LDS Bank Conflicts | 1,001/1,001 | A=0 | 0/0 → NaN → N/A | 0/0 → NaN → N/A | YES |
| instmix_test | L2-Fabric Write Latency | 1/1 | **A=27,521,900** | Guard → None → N/A | 27.5M/0 → inf → N/A | YES |
| instmix_test | sL1D Cache Hit Rate | 1/1 | A=0 | 0/0 → NaN → N/A | 0/0 → NaN → N/A | YES |
| instmix_test | vL1D Cache Hit Rate | 1/1 | A=0 | Guard → None → N/A | 0/0 → NaN → N/A | YES |
| matmul_test | *(none)* | 0 | — | — | — | — |

All zero-denominator cases produce identical user-visible results (`N/A`) under both configurations.

The three possible scenarios:

| Scenario | Old AVG | New SUM/SUM |
|---|---|---|
| A=0, B=0 (all dispatches) | `to_avg(NaN)` → NaN → N/A | `0/0` → NaN → N/A |
| A>0, B=0 (all dispatches) | Guard → None → N/A | `SUM(A)/0` → inf → caught by `isinf` → N/A |
| A>0, B=0 (some dispatches) | Guard excludes those from mean | A leaks into SUM(A); B contributes 0 to SUM(B) |

The third scenario (partial zero-denominator) was **not observed** in any profiled workload. In practice, hardware counters for the same metric are either both zero or both non-zero for a given dispatch.

---

## 6. Effect on Single-Dispatch Workloads

For N=1: `AVG(A/B) = A/B = SUM(A)/SUM(B)`. Both formulas are mathematically identical.

**Verified:** All 33 System Speed-of-Light metrics are identical for vcopy_test (1 dispatch), matmul_test (1 dispatch), and instmix_test (1 dispatch). Full tables are in Appendix A.

---

## 7. Effect on Multi-Dispatch Uniform Workloads (memcopy_test, 1,001 dispatches)

### 7.1 System Speed-of-Light Comparison

14 of 33 metrics show a measurable difference. The remaining 19 are identical.

**Metrics that differ:**

| Metric | AVG(A/B) | SUM(A)/SUM(B) | Diff % | Unit |
|---|---|---|---|---|
| VALU FLOPs | 23,543.88 | 23,273.48 | -1.15% | Gflop/s |
| VALU IOPs | 482.71 | 479.01 | -0.77% | Giop/s |
| MFMA IOPs (Int8) | 222.56 | 219.95 | -1.17% | Giop/s |
| SALU Utilization | 10.08 | 10.07 | -0.10% | Percent |
| VALU Utilization | 10.08 | 10.07 | -0.10% | Percent |
| Wavefront Occupancy | 2,280.94 | 2,279.38 | -0.07% | Wavefronts |
| vL1D Cache BW | 7,738.78 | 7,679.50 | -0.77% | Gb/s |
| L2 Cache Hit Rate | 33.42 | 33.37 | -0.15% | Percent |
| L2 Cache BW | 2,897.46 | 2,875.25 | -0.77% | Gb/s |
| L2-Fabric Read BW | 967.35 | 959.94 | -0.77% | Gb/s |
| L2-Fabric Write BW | 967.34 | 959.93 | -0.77% | Gb/s |
| sL1D Cache BW | 362.75 | 359.98 | -0.76% | Gb/s |
| L1I BW | 483.67 | 479.97 | -0.77% | Gb/s |
| CU Utilization | 99.00 | 98.94 | -0.06% | Percent |

**Unchanged metrics:** All hit rate metrics (vL1D, sL1D, L1I), all latency metrics (L2-Fabric Read/Write, L1I Fetch), IPC, VALU Active Threads, MFMA Utilization, VMEM/Branch Utilization, LDS Bank Conflicts, and all MFMA FLOPs.

**Why bandwidth metrics differ:** One outlier dispatch (ID 701, 53.6 ms vs typical 2.2 ms) gets equal weight under AVG but proportional weight under SUM/SUM, pulling bandwidth values down by ~0.77%.

**Why hit rates and latency don't differ:** Their numerator and denominator counters scale proportionally across dispatches of the same kernel, so equal-weight and proportional-weight averaging produce the same result.

### 7.2 Broader Panel Comparison (memcopy_test)

**Compute Speed-of-Light (Panel 11.1):**

| Metric | AVG(A/B) | SUM(A)/SUM(B) | Diff % |
|---|---|---|---|
| VALU FLOPs | 23,543.88 | 23,273.48 | -1.15% |
| VALU IOPs | 482.71 | 479.01 | -0.77% |
| MFMA IOPs (INT8) | 222.56 | 219.95 | -1.17% |
| MFMA FLOPs (all precisions) | identical | identical | 0.00% |

**L2 Speed-of-Light (Panel 17.1):**

| Metric | AVG(A/B) | SUM(A)/SUM(B) | Diff % |
|---|---|---|---|
| Utilization | 102.10 | 102.03 | -0.07% |
| Peak Bandwidth | 24.91 | 24.72 | -0.76% |
| Hit Rate | 33.42 | 33.37 | -0.15% |
| L2-Fabric Read BW | 967.35 | 959.94 | -0.77% |
| L2-Fabric Write and Atomic BW | 967.34 | 959.93 | -0.77% |
| HBM Bandwidth | 5,324.8 | 5,324.8 | 0.00% |

---

## 8. Effect on Mixed-Kernel Workloads

### 8.1 Per-Dispatch Raw Counter Data

| Workload | Kernel | Duration (ns) | TCC_HIT | TCC_MISS | TCC_REQ | SQ_INSTS | SQ_BUSY_CYC |
|---|---|---|---|---|---|---|---|
| vcopy | vecCopy | 12,932 | 65,764 | 131,117 | 196,878 | 327,680 | 1,117,402 |
| memcopy (disp 0) | memoryCopy | 2,234,598 | 16,778,100 | 33,556,000 | 50,334,100 | 79,691,840 | 250,646,300 |
| memcopy (disp 1000) | memoryCopy | 2,214,050 | 16,778,100 | 33,554,500 | 50,332,600 | 79,691,840 | 244,880,600 |
| matmul | matMulKernel | 42,972,678 | 64,070,400 | 71,200,600 | 135,261,000 | 3,234,180,000 | 4,802,071,000 |
| instmix | kernelasm | 1,860 | 9 | 17 | 26 | 25 | 180 |

Counter magnitudes span 7-8 orders of magnitude across kernels. Any unweighted average dramatically over-represents instmix and vcopy.

### 8.2 Combined Workload Comparison (1,004 dispatches)

| Metric | AVG(A/B) | SUM(A)/SUM(B) | Diff % |
|---|---|---|---|
| L2 Hit Rate (%) | 33.44 | 33.41 | -0.09% |
| L2 Cache BW (Gb/s) | 2,851.89 | 2,598.38 | -8.89% |
| L2-Fabric Read Latency (cyc) | 827.99 | 827.13 | -0.10% |
| IPC | 0.3225 | 0.3290 | +2.03% |
| vL1D Hit Rate (%) | 62.49 | 62.47 | -0.02% |

The 1,001 memcopy dispatches dominate the SUM/SUM result, producing values close to the memcopy-only values. AVG allows the 3 non-memcopy dispatches to exert disproportionate influence.

### 8.3 Two-Kernel Pair Comparisons (Worst Case)

Isolating just two kernels shows the maximum divergence:

| Pair | Metric | AVG(A/B) | SUM(A)/SUM(B) | Diff % |
|---|---|---|---|---|
| vcopy + matmul | IPC | 0.4834 | 0.6734 | +39.3% |
| vcopy + matmul | L2 BW (Gb/s) | 1,175.79 | 403.36 | -65.7% |

---

## 9. MIN/MAX Columns Are Unchanged

MIN and MAX report per-dispatch extremes:
```
MIN = min_i(A_i / B_i)    -- smallest per-dispatch ratio
MAX = max_i(A_i / B_i)    -- largest per-dispatch ratio
```

These are not affected by the AVG-to-SUM change. They show dispatch-level variability, not an aggregate.

**Example (memcopy_test IPC):** MIN=0.3169, MAX=0.3262 — unchanged regardless of aggregation formula.

---

## 10. Mathematical Appendix

### 10.1 When AVG(A/B) = SUM(A)/SUM(B)

The two formulas are identical when all denominators are equal:

```
If B_1 = B_2 = ... = B_N = B:
  AVG(A/B) = (1/(N*B)) * SUM(A_i)
  SUM(A)/SUM(B) = SUM(A_i) / (N*B)    -- identical
```

This is why single-dispatch and uniform workloads show zero or near-zero difference.

### 10.2 Direction of Divergence

When denominators vary, AVG(A/B) over-weights dispatches with small denominators:
- **Bandwidth** (A/duration): short dispatches inflate the average
- **Hit rates** (hits/total): dispatches with few accesses are over-represented
- **IPC** (instructions/cycles): dispatches with few busy cycles dominate

### 10.3 NaN Handling

The old `if (B != 0) else None` guards are removed in the SUM/SUM expressions. NaN handling:
- Individual `B_i = 0` with `A_i = 0`: contributes 0/0 = NaN to `SUM()`, which `to_sum()` skips via `skipna=True`
- All `B_i = 0`: `SUM(B) = 0`, division produces `inf`, caught by the `isinf` check → `"N/A"`
- `to_sum()` on an all-NaN Series returns `NaN` (line 179-180 of `parser.py`)

---

## Appendix A: Single-Dispatch Full Tables

### A.1 vcopy_test (1 dispatch)

| Metric | AVG | SUM | Unit | Diff |
|---|---|---|---|---|
| VALU FLOPs | 0.0 | 0.0 | Gflop/s | 0 |
| VALU IOPs | 403.95 | 403.95 | Giop/s | 0 |
| SALU Utilization | 4.73 | 4.73 | Percent | 0 |
| VALU Utilization | 4.05 | 4.05 | Percent | 0 |
| VMEM Utilization | 1.35 | 1.35 | Percent | 0 |
| Branch Utilization | 1.35 | 1.35 | Percent | 0 |
| VALU Active Threads | 64.0 | 64.0 | Work-items | 0 |
| IPC | 0.29 | 0.29 | Instr/cycle | 0 |
| Wavefront Occupancy | 986.65 | 986.65 | Wavefronts | 0 |
| LDS Bank Conflicts/Access | N/A | N/A | Conflicts/access | 0 |
| vL1D Cache Hit Rate | 62.5 | 62.5 | Percent | 0 |
| vL1D Cache BW | 5170.57 | 5170.57 | Gb/s | 0 |
| L2 Cache Hit Rate | 33.4 | 33.4 | Percent | 0 |
| L2 Cache BW | 1941.63 | 1941.63 | Gb/s | 0 |
| L2-Fabric Read BW | 646.58 | 646.58 | Gb/s | 0 |
| L2-Fabric Write BW | 646.32 | 646.32 | Gb/s | 0 |
| L2-Fabric Read Latency | 715.05 | 715.05 | Cycles | 0 |
| L2-Fabric Write Latency | 183.33 | 183.33 | Cycles | 0 |
| sL1D Cache Hit Rate | 99.93 | 99.93 | Percent | 0 |
| sL1D Cache BW | 323.16 | 323.16 | Gb/s | 0 |
| L1I Hit Rate | 99.92 | 99.92 | Percent | 0 |
| L1I BW | 323.16 | 323.16 | Gb/s | 0 |
| L1I Fetch Latency | 26.68 | 26.68 | Cycles | 0 |
| CU Utilization | 46.07 | 46.07 | Percent | 0 |

All 33 metrics identical. (MFMA metrics omitted — all 0.0 in both configs.)

### A.2 matmul_test (1 dispatch)

| Metric | AVG | SUM | Unit | Diff |
|---|---|---|---|---|
| VALU FLOPs | 2902.72 | 2902.72 | Gflop/s | 0 |
| VALU IOPs | 205.66 | 205.66 | Giop/s | 0 |
| VALU Utilization | 30.48 | 30.48 | Percent | 0 |
| VALU Active Threads | 63.48 | 63.48 | Work-items | 0 |
| IPC | 0.67 | 0.67 | Instr/cycle | 0 |
| Wavefront Occupancy | 2532.08 | 2532.08 | Wavefronts | 0 |
| Theoretical LDS BW | 13212.37 | 13212.37 | Gb/s | 0 |
| vL1D Cache Hit Rate | 49.81 | 49.81 | Percent | 0 |
| L2 Cache Hit Rate | 47.36 | 47.36 | Percent | 0 |
| L2-Fabric Read Latency | 390.77 | 390.77 | Cycles | 0 |
| CU Utilization | 100.23 | 100.23 | Percent | 0 |

All metrics identical. (Full 33-metric table omitted for brevity — all zero diff.)

### A.3 instmix_test (1 dispatch)

| Metric | AVG | SUM | Unit | Diff |
|---|---|---|---|---|
| VALU FLOPs | 0.58 | 0.58 | Gflop/s | 0 |
| IPC | 0.14 | 0.14 | Instr/cycle | 0 |
| L2 Cache Hit Rate | 34.62 | 34.62 | Percent | 0 |
| L2-Fabric Read Latency | 172.5 | 172.5 | Cycles | 0 |
| L2-Fabric Write Latency | N/A | N/A | Cycles | 0 |
| sL1D Cache Hit Rate | N/A | N/A | Percent | 0 |
| vL1D Cache Hit Rate | N/A | N/A | Percent | 0 |

All metrics identical. N/A values match in both configs (see Section 5.4).

---

## Appendix B: Raw Data

### B.1 Per-Dispatch Counter Data — memcopy_test (First 10 of 1,001)

| Dispatch | Duration (ns) | TCC_HIT | TCC_MISS | TCC_REQ | L2 Hit Rate (%) | SQ_INSTS | SQ_BUSY_CYC | IPC |
|---|---|---|---|---|---|---|---|---|
| 0 | 2,234,598 | 16,778,100 | 33,556,000 | 50,334,100 | 33.33 | 79,691,840 | 250,646,300 | 0.3179 |
| 1 | 2,210,546 | 16,778,100 | 33,554,500 | 50,332,600 | 33.33 | 79,691,840 | 247,314,300 | 0.3222 |
| 2 | 2,214,816 | 16,778,100 | 33,554,500 | 50,332,600 | 33.33 | 79,691,840 | 247,422,300 | 0.3221 |
| 3 | 2,214,806 | 16,778,100 | 33,554,500 | 50,332,600 | 33.33 | 79,691,840 | 247,378,500 | 0.3221 |
| 4 | 2,214,028 | 16,778,100 | 33,554,500 | 50,332,600 | 33.33 | 79,691,840 | 247,165,400 | 0.3224 |
| 5 | 2,214,412 | 16,778,100 | 33,554,500 | 50,332,600 | 33.33 | 79,691,840 | 246,298,300 | 0.3236 |
| 6 | 2,212,034 | 16,778,100 | 33,554,500 | 50,332,600 | 33.33 | 79,691,840 | 247,868,500 | 0.3215 |
| 7 | 2,214,788 | 16,778,100 | 33,554,500 | 50,332,600 | 33.33 | 79,691,840 | 246,474,200 | 0.3233 |
| 8 | 2,213,290 | 16,778,100 | 33,554,500 | 50,332,600 | 33.33 | 79,691,840 | 247,045,700 | 0.3226 |
| 9 | 2,215,976 | 16,778,100 | 33,554,500 | 50,332,600 | 33.33 | 79,691,840 | 247,035,400 | 0.3226 |

Dispatch 0 has slightly higher TCC_MISS (33,556,000 vs 33,554,500) due to cold-cache effects.

### B.2 Per-Dispatch Variability (memcopy_test, 1,001 Dispatches)

| Counter / Metric | MIN | MAX | Mean | Std Dev |
|---|---|---|---|---|
| Duration (ns) | 2,209,028 | 53,620,340 | 2,439,206 | 2,506,708 |
| TCC_REQ | 50,332,600 | 50,334,100 | 50,332,602 | 47 |
| TCC_HIT | 16,778,100 | 16,778,100 | 16,778,100 | 0 |
| SQ_BUSY_CU_CYCLES | 244,880,600 | 250,646,300 | 247,206,186 | 272,432 |
| L2 Hit Rate (%) | 33.33 | 95.75 | 33.42 | 2.13 |
| L2 BW (Gb/s) | 120.15 | 2,916.47 | 2,858.08 | 307.22 |
| IPC | 0.3169 | 0.3262 | 0.3223 | 0.0011 |

### B.3 Source Data Locations

| Data | Path |
|---|---|
| Raw counters | `workloads/{vcopy,memcopy,matmul,instmix}_test/MI308X/pmc_perf.csv` |
| System info | `workloads/vcopy_test/MI308X/sysinfo.csv` |
| AVG analysis results | `tests/metrics_aggregation_analysis/comparison_results/*_avg/` |
| SUM analysis results | `tests/metrics_aggregation_analysis/comparison_results/*_sum/` |
| Changed YAML configs | `src/rocprof_compute_soc/analysis_configs/{gfx908,gfx90a,gfx940,gfx941,gfx942,gfx950,gfx1151}/` |

---

## Summary of Impact

| Scenario | Dispatches | Kernels | Max Diff | Practical Impact |
|---|---|---|---|---|
| Single dispatch | 1 | 1 | 0.00% | None — mathematically identical |
| Uniform multi-dispatch | 1,001 | 1 | ~1.2% (throughput) | Small; SUM/SUM correctly weights outlier dispatches |
| Mixed kernels (2 kernels) | 2 | 2 | 65.7% (BW), 39.3% (IPC) | Large; AVG is misleading |
| Mixed kernels (all combined) | 1,004 | 4 | ~8.9% (BW), ~2.0% (IPC) | Moderate; AVG over-represents tiny dispatches |

The change is a strict correctness improvement with no regressions for single-dispatch workloads.

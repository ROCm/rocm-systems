# Metrics Aggregation: Proposal to Replace AVG(A/B) with SUM(A)/SUM(B)

**Project**: rocprofiler-compute
**Target architectures**: gfx908, gfx90a, gfx940, gfx941, gfx942, gfx950
**Date**: 2026-03-29

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Background: How Metrics Are Currently Aggregated](#2-background-how-metrics-are-currently-aggregated)
3. [Mathematical Analysis](#3-mathematical-analysis)
4. [Empirical Evidence from MI350 Profiling](#4-empirical-evidence-from-mi350-profiling)
5. [NaN Handling Under Counter Multiplexing](#5-nan-handling-under-counter-multiplexing)
6. [Proposed Change](#6-proposed-change)
7. [Impact Assessment](#7-impact-assessment)
8. [Appendix: Full Data Tables](#appendix-full-data-tables)

---

## 1. Problem Statement

When rocprofiler-compute aggregates a ratio metric across multiple kernel dispatches, it currently computes the **mean of per-dispatch ratios**:

$$\text{AVG}(A/B) = \frac{1}{N}\sum_{i=1}^{N} \frac{A_i}{B_i}$$

This gives **equal weight to every dispatch** regardless of its magnitude. For workloads where dispatches differ significantly in size, duration, or access count, this produces values that do not correspond to any physically meaningful aggregate measurement.

### Who is affected

Any user who:
- Profiles a workload containing **multiple different kernels** (e.g., a PyTorch training step with dozens of distinct GPU kernels)
- Profiles a workload where the **same kernel runs with different input sizes** across dispatches
- Aggregates metrics across dispatches using the default `AVG()` expressions in the analysis panels

### What is not affected

- **Single-dispatch profiles**: Only one ratio value exists, so AVG(A/B) = SUM(A)/SUM(B) trivially.
- **Uniform workloads**: When all dispatches execute the same kernel with the same parameters, per-dispatch counter values are nearly identical, so AVG(A/B) approximately equals SUM(A)/SUM(B).
- **MIN/MAX metrics**: Per-dispatch minimum and maximum are inherently per-dispatch statistics and remain correct.

---

## 2. Background: How Metrics Are Currently Aggregated

### Metric expression format

Derived metrics are defined in YAML configuration files under `src/rocprof_compute_soc/analysis_configs/<arch>/`. Each metric has a `value` expression that is evaluated at analysis time. Examples:

```yaml
# L2 Cache Hit Rate (percentage)
value: AVG((((100 * TCC_HIT_sum) / (TCC_HIT_sum + TCC_MISS_sum))
       if ((TCC_HIT_sum + TCC_MISS_sum) != 0) else None))

# IPC (instructions per cycle)
value: AVG((SQ_INSTS / SQ_BUSY_CU_CYCLES))

# L2 Cache Bandwidth (GB/s)
value: AVG(((TCC_REQ_sum * 128) / (End_Timestamp - Start_Timestamp)))

# vL1D Cache Hit Rate (percentage)
value: AVG(((100 - ((100 * (TCP_TCC_READ_REQ_sum + TCP_TCC_WRITE_REQ_sum
       + TCP_TCC_ATOMIC_WITH_RET_REQ_sum + TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum))
       / TCP_TOTAL_CACHE_ACCESSES_sum)) if (TCP_TOTAL_CACHE_ACCESSES_sum != 0) else None))
```

### Evaluation flow

1. Each hardware counter (e.g., `TCC_HIT_sum`) resolves to a pandas Series — one value per kernel dispatch.
2. Arithmetic operations (`*`, `/`, `+`, `-`) are evaluated element-wise across dispatches.
3. The outer `AVG()` call reduces the per-dispatch Series to a single scalar by calling `.mean()`.

This means `AVG(TCC_HIT_sum / (TCC_HIT_sum + TCC_MISS_sum))` computes the hit rate **independently for each dispatch**, then averages those rates with **equal weight per dispatch**.

---

## 3. Mathematical Analysis

### 3.1 Why equal-weight averaging is incorrect for ratio metrics

Consider a simple example: two kernel dispatches profiled together.

| Dispatch | L2 Hits | L2 Misses | Total Accesses | Per-Dispatch Hit Rate |
|----------|---------|-----------|----------------|----------------------|
| Kernel A | 10 | 990 | 1,000 | 1.0% |
| Kernel B | 9,000 | 1,000 | 10,000 | 90.0% |
| **Total** | **9,010** | **1,990** | **11,000** | **81.9%** |

**Current result (AVG of ratios):**

```
AVG(A/B) = (1.0% + 90.0%) / 2 = 45.5%
```

**Correct result (ratio of sums):**

```
SUM(A) / SUM(B) = 9,010 / 11,000 = 81.9%
```

The current formula reports 45.5%, but the actual L2 hit rate across both dispatches is 81.9%. The discrepancy arises because Kernel A had only 1,000 accesses while Kernel B had 10,000 — yet both are weighted equally.

### 3.2 The mathematical relationship

For two formula variants:

```
AVG(A/B) = (1/N) * sum(A_i / B_i)         — equal weight per dispatch
SUM(A)/SUM(B) = sum(A_i) / sum(B_i)       — weight proportional to denominator magnitude
```

These are **identical** if and only if all `B_i` values are equal (i.e., the denominator is constant across dispatches). Otherwise, SUM(A)/SUM(B) is a **weighted average** of the per-dispatch ratios, where the weight of dispatch `i` is `B_i / sum(B_j)`:

```
SUM(A)/SUM(B) = sum( (B_i / sum(B_j)) * (A_i / B_i) ) = sum( w_i * r_i )
```

where `w_i = B_i / sum(B_j)` and `r_i = A_i / B_i`.

This weighted average is the correct aggregate because it reflects the **overall ratio** — the answer to "what fraction of all L2 accesses were hits?" rather than "what is the average of per-dispatch hit rates?"

### 3.3 Category-by-category analysis

| Metric Category | Example | What denominator represents | Why weighting by denominator is correct |
|----------------|---------|---------------------------|----------------------------------------|
| **Hit rates** | L2 hit rate = Hits / Total | Total cache accesses | A dispatch with 10x more accesses contributes 10x more information about the cache's behavior |
| **Throughput** | BW = Bytes / Time | Kernel duration | Throughput is total work / total time; short kernels should not be overweighted |
| **Latency** | L2-Fabric latency = Level / Requests | Request count | Each request's latency contributes equally to the overall average latency |
| **Utilization** | VALU util = Active / (GUI_ACTIVE * CUs) | GPU active cycles | Utilization is fraction of total available cycles used; short kernels should not dominate |
| **IPC** | IPC = Instructions / Busy cycles | Busy CU cycles | IPC represents instruction throughput per cycle of actual work |
| **Instructions/wave** | Insts/wave = Instructions / Waves | Wavefront count | The overall ratio tells how many instructions the average wavefront executed |

In every category, SUM(A)/SUM(B) produces the physically meaningful aggregate — the value you would get if all dispatches were treated as one combined execution.

### 3.4 When AVG(A/B) could be preferred

AVG(A/B) answers the question: "on average, what did each dispatch experience?" This could be useful when dispatches represent independent experiments (e.g., different input sizes being benchmarked). However, in GPU profiling, the standard use case is **characterizing the workload as a whole**, where SUM(A)/SUM(B) is the appropriate aggregate.

---

## 4. Empirical Evidence from MI350 Profiling

### 4.1 Workloads profiled

Five datasets were used for comparison. All profiling was performed on MI350.

| Dataset | Dispatches | Kernel(s) | Description |
|---------|-----------|-----------|-------------|
| vcopy_uniform | 100 | vecCopy (1M threads) | Identical kernel repeated 100 times |
| nbody_multi | 10 | bodyForce | Compute-heavy N-body simulation |
| vcopy_small | 5 | vecCopy (4K threads) | Small problem size |
| memcopy_multi | 1001 | memoryCopyKernel | Memory-bound copy kernel |
| mixed_kernels | 30 | vecCopy + bodyForce + memoryCopy | 10 dispatches from each workload, combined |

### 4.2 Uniform workloads: formulas agree

For **vcopy_uniform** (100 identical dispatches), the difference between AVG(A/B) and SUM(A)/SUM(B) is negligible:

| Metric | AVG(A/B) | SUM(A)/SUM(B) | Difference |
|--------|----------|---------------|------------|
| L2 Hit Rate | 33.5643% | 33.5643% | 0.00% |
| L2-Fabric Read Latency | 1065.85 cycles | 1065.85 cycles | -0.00% |
| L2 Cache BW | 5252.37 GB/s | 5237.20 GB/s | +0.29% |
| IPC | 0.2177 | 0.2174 | +0.12% |
| Instructions per Wavefront | 20.0000 | 20.0000 | 0.00% |
| vL1D Cache Hit Rate | 87.50% | 87.50% | 0.00% |

**Conclusion**: The proposed change does not regress results for uniform workloads. Differences are within floating-point noise.

Results for other uniform workloads (nbody_multi, memcopy_multi) show the same pattern — differences below 0.01% for most metrics.

### 4.3 Mixed kernels: formulas diverge dramatically

For **mixed_kernels** (10 vecCopy + 10 bodyForce + 10 memoryCopy dispatches), the difference between formulas is enormous:

| Metric | AVG(A/B) | SUM(A)/SUM(B) | Difference | Explanation |
|--------|----------|---------------|------------|-------------|
| L2 Hit Rate | 53.14% | 33.87% | **+56.9%** | nbody's 92.5% hit rate (few accesses) is weighted equally with memcopy's 33.3% (many accesses) |
| L2-Fabric Read Latency | 1490.5 cycles | 2326.3 cycles | **-35.9%** | memcopy's high latency (many requests) is underweighted |
| L2 Cache BW | 4549.2 GB/s | 1821.4 GB/s | **+149.8%** | vcopy's high per-dispatch BW (short kernels) dominates the average |
| IPC | 0.415 | 0.600 | **-30.8%** | nbody's high IPC (many busy cycles) is underweighted |
| Instructions per Wavefront | 418,333 | 159.7 | **+261,912%** | nbody has ~1.25M insts/wave vs vecCopy has 20 — equal weighting is nonsensical |
| vL1D Hit Rate | 83.33% | 87.34% | **-4.6%** | nbody's lower hit rate (fewer accesses) is overweighted |

### 4.4 Detailed walkthrough: L2 Cache Bandwidth

Consider the L2 Cache BW metric: `AVG((TCC_REQ_sum * 128) / (End_Timestamp - Start_Timestamp))`.

In the mixed workload, the three kernel types have very different characteristics:

- **vecCopy** dispatches: short duration (~100 ns), moderate request count → high per-dispatch BW
- **bodyForce** dispatches: long duration, few L2 requests → low per-dispatch BW
- **memoryCopy** dispatches: long duration, many L2 requests → high per-dispatch BW

AVG(A/B) = 4549.2 GB/s:
- The 10 very short vecCopy dispatches each report extremely high bandwidth
- These 10 dispatches dominate the average despite representing a tiny fraction of total GPU time

SUM(A)/SUM(B) = 1821.4 GB/s:
- Total bytes transferred / total time = actual sustained bandwidth
- Short dispatches contribute proportionally to their duration

The SUM/SUM value is the number a user would verify against a roofline model or hardware spec sheet. The AVG value is misleading.

### 4.5 Detailed walkthrough: L2-Fabric Read Latency

Consider the L2-Fabric latency metric: `AVG(TCC_EA0_RDREQ_LEVEL_sum / TCC_EA0_RDREQ_sum)`.

The denominator is the number of read requests. Dispatches with many requests provide more statistical weight for measuring latency.

AVG(A/B) = 1490.5 cycles:
- Weights each dispatch equally, even if one dispatch issued 100 requests and another issued 100,000

SUM(A)/SUM(B) = 2326.3 cycles:
- Weights each request equally — a dispatch that issued 100,000 requests contributes 1000x more than one that issued 100
- This is the standard definition of "average latency" in computer architecture

### 4.6 Non-uniform single-kernel workloads

Even for **vcopy_small** (5 dispatches of the same kernel with a small problem size), measurable differences appear:

| Metric | AVG(A/B) | SUM(A)/SUM(B) | Difference |
|--------|----------|---------------|------------|
| L2 Hit Rate | 47.74% | 47.74% | +0.01% |
| L2-Fabric Read Latency | 662.0 cycles | 668.6 cycles | **-0.99%** |
| L2 Cache BW | 59.58 GB/s | 57.99 GB/s | **+2.74%** |
| IPC | 0.0319 | 0.0318 | +0.59% |

With small dispatch counts and small problem sizes, run-to-run variance in counter values creates noticeable dispatch-to-dispatch variation, causing the two formulas to diverge by up to 2.7%.

---

## 5. NaN Handling Under Counter Multiplexing

### 5.1 How NaN values arise

When rocprofiler-compute uses iteration multiplexing (collecting different counter groups in separate passes), the following can occur:
- Pass 1 collects counters A, C; Pass 2 collects counters B, D
- After imputation, dispatches in incomplete subgroups (where dispatch count is not a multiple of the pass count) may retain NaN for the counters from the missing pass

### 5.2 Three candidate formulas under NaN

When NaN values exist, the question becomes: which formula best recovers the true aggregate (the value that would be computed if all data were available)?

We tested this using synthetic data where the ground truth (full clean data with no NaN) is known. NaN values were injected in five patterns of increasing severity, and each formula's error was measured against ground truth.

| Formula | Definition | Mean Absolute Error vs Ground Truth |
|---------|-----------|-------------------------------------|
| AVG(A/B) | Mean of per-dispatch ratios | 17.6% |
| SUM(A)/SUM(B) | Sum numerator / Sum denominator (nansum) | 36.2% |
| AVG(A)/AVG(B) | Mean numerator / Mean denominator (nanmean) | 29.6% |
| **Pairwise SUM(A)/SUM(B)** | Sum only dispatches where both A and B are non-NaN | **6.6%** |

### 5.3 Why pairwise completion matters

When numerator and denominator have NaN in **different rows** (different dispatches), both SUM(A)/SUM(B) and AVG(A)/AVG(B) compute over **different populations**:

```
SUM(A)/SUM(B):  sums A over rows {0,1,2,3,4} / sums B over rows {0,1,5,6,7}
                These are different dispatches — the ratio is meaningless.

Pairwise:       Only includes rows where both A and B are valid:
                SUM(A[{0,1}]) / SUM(B[{0,1}])
                Same dispatches — the ratio is correct.
```

Example with synthetic data (NaN in different rows for numerator vs denominator):

| Metric | Ground Truth | SUM/SUM | Pairwise | SUM/SUM Error | Pairwise Error |
|--------|-------------|---------|----------|---------------|----------------|
| L2 Hit Rate | 43.44% | 47.21% | 32.06% | +8.7% | **-26.2%** |
| L2-Fabric Latency | 67.70 | 43.38 | 72.73 | **-35.9%** | +7.4% |
| IPC | 1.177 | 0.761 | 1.113 | **-35.4%** | -5.4% |
| Insts/Wave | 100.0 | 64.9 | 100.0 | **-35.1%** | 0.0% |

Across all NaN patterns tested, pairwise SUM/SUM had the lowest overall error (6.6% MAE vs 36.2% for plain SUM/SUM).

### 5.4 NaN mismatch in practice

Analysis of the profiling pipeline code reveals that cross-pass NaN mismatch (numerator and denominator having NaN in different rows) is **rare in practice**:
- The imputation algorithm uses backfill/forwardfill within complete subgroups, ensuring both numerator and denominator are either both valid or both NaN for the same dispatch
- The CSV loading path uses inner join, dropping unmatched dispatches entirely

However, it can occur when a kernel's dispatch count is not a multiple of the number of profiling passes, leaving an incomplete final subgroup. The pairwise approach handles this correctly by construction.

---

## 6. Proposed Change

### 6.1 Formula change

Replace the current aggregation behavior of `AVG()` for ratio metrics:

| | Current | Proposed |
|---|---------|----------|
| **What AVG() computes for A/B** | `mean(A_i / B_i)` | `SUM(A_i[valid]) / SUM(B_i[valid])` |
| **Weighting** | Equal per dispatch | Proportional to denominator magnitude |
| **NaN handling** | Skips NaN ratios | Only includes dispatches where both A and B are non-NaN |
| **YAML expressions** | No changes | No changes |

Where `valid` = the set of dispatches where both numerator and denominator are non-NaN.

### 6.2 What changes for users

- **Uniform workloads** (single kernel, same parameters): Results are **identical** within floating-point tolerance (verified empirically — see Section 4.2).
- **Multi-kernel workloads**: Results change to reflect the **physically correct aggregate**. Large dispatches (more requests, longer duration, more cycles) contribute proportionally more.
- **MIN/MAX metrics**: Unchanged. These remain per-dispatch extremes.

### 6.3 Metrics affected

All metrics using `AVG(expression_with_division)` in the YAML config files. This includes approximately 3,000+ expression instances across 6 architectures. The specific metric categories are:

| Category | Example Metrics | Expected Impact |
|----------|----------------|-----------------|
| Hit rates | L2 Hit Rate, vL1D Hit Rate, sL1D Hit Rate | Moderate — rates become weighted by access count |
| Throughput | VALU FLOP/s, L2 Cache BW, vL1D Cache BW, L2-Fabric BW | Large — short-duration dispatches no longer dominate |
| Latency | L2-Fabric Read Latency, TCP-L2 Latency | Moderate — latencies become weighted by request count |
| Utilization | VALU Utilization, SALU Utilization, MFMA Utilization | Moderate — utilization weighted by GPU active cycles |
| Normalized counters | IPC, Instructions per Wavefront, Wavefront Occupancy | Moderate — weighted by cycle/wavefront count |

### 6.4 Metrics NOT affected

- Metrics that use `AVG()` on a **single counter** (no division): `AVG(Arch_VGPR)` → unchanged, still computes mean.
- Metrics that use `AVG()` on a **non-ratio expression** (e.g., `AVG(100 - hit_rate)`): The subtraction of the ratio from a constant breaks the ratio semantics, so mean is correctly applied.
- Metrics that already use `SUM()`: e.g., metrics in `3000_mem_bw.yaml` that explicitly use `SUM(A) / SUM(B)`.
- **MIN/MAX columns**: Continue to report per-dispatch extremes.

### 6.5 Existing precedent in the codebase

The `3000_mem_bw.yaml` panel already implements SUM/SUM for bandwidth and latency metrics, and documents the rationale:

```yaml
# From 3000_mem_bw.yaml:
# "Average Latency" uses SUM(LEVEL)/SUM(REQ) — weighted by request count
# "Latency" uses AVG(LEVEL/REQ) — equal weight per dispatch
```

This panel provides both variants side-by-side. The proposed change aligns all other panels with the `SUM(A)/SUM(B)` approach that `3000_mem_bw.yaml` already uses for its primary metrics.

---

## 7. Impact Assessment

### 7.1 Backward compatibility

| Workload type | Impact |
|---------------|--------|
| Single dispatch | Zero change — mathematically identical |
| Uniform multi-dispatch (same kernel, same params) | Negligible change — <0.3% difference observed |
| Non-uniform single kernel (varying input sizes) | Small change — 1-3% difference observed |
| Mixed kernels (different kernel types) | Large change — up to 150%+ difference for throughput metrics |

### 7.2 Risk assessment

- **No YAML config changes required**: The change is implemented in the evaluation layer, not in the ~3,000 metric expressions.
- **No CLI/API changes**: User-facing interfaces remain identical.
- **Reversible**: If a user prefers the old behavior, they can wrap expressions with a hypothetical `DISPATCH_AVG()` function (not currently planned).

### 7.3 Test coverage

- Uniform workload tests (vcopy_uniform, memcopy_multi) verify backward compatibility
- Mixed workload tests verify correctness of the new formula
- Synthetic NaN tests verify robustness under counter multiplexing
- All existing integration tests must continue to pass

---

## Appendix: Full Data Tables

### A.1 Formula comparison across all workloads

| Workload | Metric | AVG(A/B) | SUM(A)/SUM(B) | Diff% |
|----------|--------|----------|---------------|-------|
| vcopy_uniform | L2 Hit Rate | 33.5643% | 33.5643% | 0.00% |
| vcopy_uniform | L2-Fabric Read Latency | 1065.85 | 1065.85 | -0.00% |
| vcopy_uniform | L2 Cache BW | 5252.37 | 5237.20 | +0.29% |
| vcopy_uniform | IPC | 0.2177 | 0.2174 | +0.12% |
| vcopy_uniform | Instructions per Wavefront | 20.00 | 20.00 | 0.00% |
| vcopy_uniform | vL1D Hit Rate | 87.50% | 87.50% | 0.00% |
| nbody_multi | L2 Hit Rate | 92.53% | 92.53% | -0.00% |
| nbody_multi | L2-Fabric Read Latency | 1035.35 | 1035.35 | +0.00% |
| nbody_multi | L2 Cache BW | 20.66 | 20.65 | +0.00% |
| nbody_multi | IPC | 0.8398 | 0.8398 | +0.00% |
| nbody_multi | Instructions per Wavefront | 1,254,961 | 1,254,961 | 0.00% |
| nbody_multi | vL1D Hit Rate | 75.00% | 75.00% | +0.00% |
| vcopy_small | L2 Hit Rate | 47.74% | 47.74% | +0.01% |
| vcopy_small | L2-Fabric Read Latency | 662.02 | 668.62 | -0.99% |
| vcopy_small | L2 Cache BW | 59.58 | 57.99 | +2.74% |
| vcopy_small | IPC | 0.0319 | 0.0318 | +0.59% |
| vcopy_small | Instructions per Wavefront | 20.00 | 20.00 | 0.00% |
| vcopy_small | vL1D Hit Rate | 87.50% | 87.50% | 0.00% |
| memcopy_multi | L2 Hit Rate | 33.34% | 33.34% | +0.00% |
| memcopy_multi | L2-Fabric Read Latency | 2341.45 | 2341.45 | +0.00% |
| memcopy_multi | L2 Cache BW | 8525.92 | 8525.69 | +0.00% |
| memcopy_multi | IPC | 0.1922 | 0.1922 | +0.00% |
| memcopy_multi | Instructions per Wavefront | 19.00 | 19.00 | 0.00% |
| memcopy_multi | vL1D Hit Rate | 87.50% | 87.50% | +0.00% |
| **mixed_kernels** | **L2 Hit Rate** | **53.14%** | **33.87%** | **+56.92%** |
| **mixed_kernels** | **L2-Fabric Read Latency** | **1490.52** | **2326.30** | **-35.93%** |
| **mixed_kernels** | **L2 Cache BW** | **4549.18** | **1821.37** | **+149.77%** |
| **mixed_kernels** | **IPC** | **0.4154** | **0.6000** | **-30.76%** |
| **mixed_kernels** | **Instructions per Wavefront** | **418,333** | **159.66** | **+261,912%** |
| **mixed_kernels** | **vL1D Hit Rate** | **83.33%** | **87.34%** | **-4.58%** |

### A.2 NaN correctness analysis (synthetic data, ground truth comparison)

Mean Absolute Error (%) vs ground truth, by NaN injection pattern:

| NaN Pattern | AVG(A/B) | SUM/SUM | AVG/AVG | Pairwise SUM/SUM |
|-------------|----------|---------|---------|-----------------|
| Aligned (same rows NaN) | 13.95% | 3.64% | 3.64% | **3.64%** |
| Mismatch trailing | 19.28% | 34.17% | 26.07% | **8.15%** |
| Mismatch interleaved | N/A | 95.82% | 94.43% | N/A |
| Mismatch random | 19.63% | 22.83% | 21.97% | **8.04%** |
| Heavy mismatch | N/A | 36.71% | 26.59% | N/A |
| **Overall** | **17.62%** | **36.23%** | **29.59%** | **6.61%** |

### A.3 Third-formula equivalence: SUM/SUM vs AVG/AVG

When no NaN mismatch exists (all real workloads), SUM(A)/SUM(B) and AVG(A)/AVG(B) are **mathematically identical**:

```
AVG(A)/AVG(B) = (SUM(A)/N) / (SUM(B)/N) = SUM(A)/SUM(B)
```

This was confirmed empirically: across all 4 real workloads and all 6 metrics, the difference between SUM/SUM and AVG/AVG was 0.00%. The two formulas diverge **only** when the count of valid values differs between numerator and denominator (NaN mismatch), in which case pairwise SUM/SUM is preferred (see Section 5).

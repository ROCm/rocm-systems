# Latency vs Average Latency: Analysis Report

## Executive Summary

**`AVG(X)/AVG(Y)` and `SUM(X)/SUM(Y)` are algebraically identical.** The N cancels:

```
AVG(X)/AVG(Y) = (SUM(X)/N) / (SUM(Y)/N) = SUM(X)/SUM(Y)
```

The newly added "Latency" metrics (`AVG()/AVG()`) will always produce the exact same
numerical result as the existing "Average Latency" metrics (`SUM()/SUM()`), regardless of
the input data. This was confirmed both mathematically and empirically across all available
MI350 profiling results.

If the intent was to create a metric that gives **equal weight to each dispatch** (or each
hardware instance), the correct formula would be `AVG(X_i / Y_i)` -- the **mean of
per-element ratios** -- not `AVG(X) / AVG(Y)`.

---

## 1. The Two Formulas Under Comparison

### Average Latency (SUM/SUM) -- Existing

```yaml
L1 average latency:
  value: SUM(TCP_TCP_LATENCY_sum) / SUM(TCP_TOTAL_ACCESSES_sum)
```

Aggregation: sum all per-dispatch latency-cycle accumulators, divide by sum of all
per-dispatch access counts.

### Latency (AVG/AVG) -- Newly Added

```yaml
L1 latency:
  value: AVG(TCP_TCP_LATENCY_sum) / AVG(TCP_TOTAL_ACCESSES_sum)
```

Aggregation: mean of per-dispatch latency-cycle accumulators, divide by mean of
per-dispatch access counts.

---

## 2. Detailed Mathematical Breakdown With Counter References

### 2.1 Data Structure: What Lands in pmc_perf.csv

For L1 metrics (TCP), a single dispatch `d` on MI350 (256 CUs across 8 XCDs):

```
TCP_TCP_LATENCY_sum_d = sum over all CUs c: TCP_TCP_LATENCY_{d,c}
                      = l_{d,1} + l_{d,2} + ... + l_{d,256}

TCP_TOTAL_ACCESSES_sum_d = sum over all CUs c: TCP_TOTAL_ACCESSES_{d,c}
                         = a_{d,1} + a_{d,2} + ... + a_{d,256}
```

The `_sum` suffix means the hardware/driver already summed across all CU instances
(for TCP) or all L2 channels (for TCC). What lands in `pmc_perf.csv` is **one scalar
per dispatch**:

```
Dispatch 0:    L_0 = TCP_TCP_LATENCY_sum,   A_0 = TCP_TOTAL_ACCESSES_sum
Dispatch 1:    L_1 = TCP_TCP_LATENCY_sum,   A_1 = TCP_TOTAL_ACCESSES_sum
...
Dispatch D-1:  L_{D-1},                     A_{D-1}
```

### 2.2 What SUM()/SUM() Computes

```
SUM(TCP_TCP_LATENCY_sum) / SUM(TCP_TOTAL_ACCESSES_sum)

= (L_0 + L_1 + ... + L_{D-1}) / (A_0 + A_1 + ... + A_{D-1})
```

### 2.3 What AVG()/AVG() Computes

```
AVG(TCP_TCP_LATENCY_sum) / AVG(TCP_TOTAL_ACCESSES_sum)

= ((L_0 + L_1 + ... + L_{D-1}) / D)  /  ((A_0 + A_1 + ... + A_{D-1}) / D)

= (L_0 + L_1 + ... + L_{D-1}) / (A_0 + A_1 + ... + A_{D-1})
```

The `/D` appears in both numerator and denominator and cancels. **Always. Regardless
of what `L_i` and `A_i` values are.**

### 2.4 Why They Are Always the Same

It is not a property of the data. It is a property of the algebra. For any values:

```
(X/N) / (Y/N) = X/Y
```

This is true whether D=1 or D=1000, whether the `A_i` are constant or wildly varying.
The `AVG()/AVG()` formulas in the codebase will never produce a different number from
the `SUM()/SUM()` formulas.

### 2.5 What WOULD Be Different: Mean of Per-Element Ratios

The formula that gives different results is the **mean of per-element ratios**:

```
(1/D) * SUM_d(L_d / A_d)

= (1/D) * (L_0/A_0  +  L_1/A_1  +  ...  +  L_{D-1}/A_{D-1})
```

The division happens **inside** the sum, per-element, before aggregation. This is NOT
what `AVG(X)/AVG(Y)` computes -- that divides **after** aggregation.

Concrete example with 2 dispatches:

```
D0:  L_0 = 5,000     A_0 = 100      (per-dispatch latency = 50 cyc/req)
D1:  L_1 = 300,000   A_1 = 10,000   (per-dispatch latency = 30 cyc/req)

SUM/SUM:          305,000 / 10,100              = 30.20
AVG/AVG:          152,500 / 5,050               = 30.20   (identical, D cancels)
MEAN(L_d/A_d):    (50 + 30) / 2                 = 40.00   (different)
```

### 2.6 The Same Principle Applies to Per-Channel L2 Counters

Replace "dispatches" with "L2 channels" and `_sum` with per-channel counters `[0..127]`:

```
SUM(TCC_EA0_RDREQ_LEVEL[i]) / SUM(TCC_EA0_RDREQ[i])    -- this IS what _sum/_sum gives
AVG(TCC_EA0_RDREQ_LEVEL[i]) / AVG(TCC_EA0_RDREQ[i])    -- identical (N cancels)
MEAN(TCC_EA0_RDREQ_LEVEL[i] / TCC_EA0_RDREQ[i])         -- different when req counts vary
```

The per-channel divergences in Section 4 of this report (e.g., atomic_baseline +8.1%)
are between the first and third formulas -- not between the first and second, which
are always equal.

### 2.7 Summary Table

| Formula | Notation | Equal to SUM/SUM? | When different? |
|---|---|---|---|
| `SUM(X) / SUM(Y)` | `SUM(X_i) / SUM(Y_i)` | -- (baseline) | -- |
| `AVG(X) / AVG(Y)` | `(SUM(X_i)/N) / (SUM(Y_i)/N)` | **Always equal** | Never |
| `AVG(X_i / Y_i)` | `(1/N) * SUM(X_i/Y_i)` | **Not always** | When `Y_i` varies across elements |

### 2.8 Implementation Confirmation

In the rocprofiler-compute codebase (`src/utils/parser.py`):
- `to_avg()` calls `pd.Series.mean()` on the per-dispatch column
- `to_sum()` calls `pd.Series.sum()` on the per-dispatch column
- Both operate on the same rows of the DataFrame

Since `Series.mean() = Series.sum() / len(Series)`, the N cancels in the ratio.

---

## 3. Empirical Verification

### 3.1 Multi-Dispatch Workloads

#### nbody-block-full (10 dispatches)

| Metric               | SUM/SUM (Avg Latency) | AVG/AVG (Latency) | Difference |
|----------------------|----------------------:|-------------------:|-----------:|
| L1 latency           |                  6.66 |               6.66 |       0.00 |
| L2 read latency      |                678.43 |             678.43 |       0.00 |
| L2 write latency     |                139.08 |             139.08 |       0.00 |
| UTCL1 latency        |                 22.23 |              22.23 |       0.00 |

All dispatches have identical denominator values (same kernel, same grid size = 1024,
workgroup = 256). The numerators vary slightly due to runtime conditions (e.g., UTCL1
latency ranges from 6.00 to 80.21 cycles/req across dispatches), but the ratio of sums
equals the ratio of means regardless.

#### test_l2_mem (1001 dispatches)

| Metric               | SUM/SUM (Avg Latency) | AVG/AVG (Latency) | Difference |
|----------------------|----------------------:|-------------------:|-----------:|
| L1 latency           |                 18.91 |              18.91 |      ~0.00 |
| L2 read latency      |               1442.29 |            1442.29 |      ~0.00 |
| L2 write latency     |                193.06 |             193.06 |      ~0.00 |
| L2-EA read latency   |               2356.69 |            2356.69 |      ~0.00 |
| L2-EA write latency  |                493.61 |             493.61 |      ~0.00 |

Even with 1001 dispatches and varying latency numerators (e.g., L1 latency ranges 18.15
to 19.36 across dispatches), the two formulas produce identical results to floating-point
precision (~10^-15 relative error).

### 3.2 Single-Dispatch Workloads

For all single-dispatch workloads (hbm_read_baseline, hbm_write_baseline, thrash_baseline,
atomic_baseline, etc.), `SUM(X) = AVG(X) = X` for a single value, so the results are
trivially identical.

---

## 4. What Would Actually Be Different

The meaningful alternative to `SUM(X)/SUM(Y)` is not `AVG(X)/AVG(Y)`, but rather
**`AVG(X_i / Y_i)`** -- the unweighted mean of per-element ratios:

```
Weighted Average:    SUM(X) / SUM(Y)      -- each request contributes equally
Unweighted Average:  MEAN(X_i / Y_i)      -- each element (dispatch/CU/channel) contributes equally
```

### Synthetic Example

| Dispatch | Latency Cycles | Accesses | Per-Dispatch Latency |
|----------|---------------:|---------:|---------------------:|
| A        |          5,000 |      100 |             50.0     |
| B        |        300,000 |   10,000 |             30.0     |

- `SUM/SUM` = 305,000 / 10,100 = **30.20** (dominated by Dispatch B, which has 99% of accesses)
- `AVG/AVG` = 152,500 / 5,050 = **30.20** (identical -- N cancels)
- `MEAN(ratio)` = (50 + 30) / 2 = **40.00** (each dispatch weighted equally -- +32.5% different)

### Per-Channel Analysis (Analog Within a Single Dispatch)

To demonstrate this principle with real data, I analyzed the per-L2-channel distribution
within single dispatches. The `_sum` counters aggregate across all 128 L2 channels, but
per-channel data (e.g., `TCC_EA0_RDREQ_LEVEL[0..127]`) reveals the distribution:

| Workload                     | Metric   | Active Ch | SUM/SUM    | MEAN(ratio) | Diff%    | Req CV% |
|------------------------------|----------|-----------|------------|-------------|----------|---------|
| atomic_baseline              | EA Read  | 112/128   | 569,691    | 615,776     | **+8.1%**| 34.8%   |
| l2_verify_hbm_write_baseline | EA Read  | 64/128    | 311.52     | 259.24      | **-16.8%**| 41.6% |
| l2_verify_atomic_baseline    | EA Read  | 112/128   | 319.96     | 312.65      | **-2.3%**| 30.1%  |
| hbm_write_baseline           | EA Read  | 112/128   | 280.14     | 282.93      | **+1.0%**| 52.4%  |
| io_baseline                  | EA Read  | 128/128   | 11,153     | 11,169      | +0.14%   | 0.7%   |
| hbm_read_baseline            | EA Read  | 128/128   | 2,049.84   | 2,049.91    | +0.00%   | 0.2%   |
| thrash_baseline              | EA Read  | 128/128   | 1,252.81   | 1,253.34    | +0.04%   | 0.5%   |

**Key observation:** The divergence between `SUM/SUM` and `MEAN(ratio)` is driven by the
**coefficient of variation (CV) of the denominator** (request count). When requests are
uniformly distributed across channels (low CV), the formulas converge. When requests are
sparse or highly skewed (high CV), they diverge significantly.

The largest divergence occurs in `atomic_baseline / EA Read` (+8.1%), where only 112 of
128 channels received any read requests, and the request count per channel ranged from
0 to 8 with a CV of 34.8%.

---

## 5. Hardware Perspective: Which Formula Makes Sense?

### For Latency Metrics, SUM(X)/SUM(Y) Is the Correct Formula

From a hardware perspective, `SUM(X)/SUM(Y)` (the existing "Average Latency") is the
correct way to compute latency because:

1. **It represents the true per-request average latency.** The numerator counters (e.g.,
   `TCP_TCP_LATENCY_sum`) are accumulators of total latency cycles across all requests.
   The denominator counters (e.g., `TCP_TOTAL_ACCESSES_sum`) count total requests.
   Dividing total cycles by total requests gives the true average cycles per request.

2. **It naturally weights by traffic volume.** A CU or L2 channel handling 10x more
   requests contributes 10x more to the average, which correctly reflects the system's
   aggregate behavior. A channel with 1 request experiencing 1000-cycle latency should
   not have equal influence with a channel processing 1 million requests at 30-cycle
   latency.

3. **It corresponds to Little's Law.** In queueing theory, average latency =
   total-time-in-system / total-requests. This is exactly `SUM(X)/SUM(Y)`.

4. **It is additive across hierarchy levels.** SUM-based latency can be decomposed:
   total L1-to-memory latency = L1 latency + L2 latency + HBM latency, because the sums
   of cycle accumulators are additive.

### When Would MEAN(X_i/Y_i) Be Useful?

An unweighted average of per-element ratios would be useful for:

- **Identifying outlier CUs/channels:** If one CU consistently sees 10x higher latency
  than others, MEAN(ratio) will surface this while SUM/SUM will dilute it.
- **Fairness analysis:** Understanding if all CUs/channels experience similar latency,
  regardless of their traffic volume.
- **Detecting hardware imbalance:** Non-uniform latency across channels could indicate
  defective HBM stacks, routing imbalances, or address mapping issues.

However, `AVG(X)/AVG(Y)` does **not** achieve this goal. Only `AVG(X_i/Y_i)` would.

---

## 6. Recommendations

1. **The current `AVG(X)/AVG(Y)` formulas are redundant** with the existing `SUM(X)/SUM(Y)`
   formulas. They will always produce identical numerical results.

2. **If a "ratio of averages" semantic is desired**, the formula should be changed to
   compute the **mean of per-dispatch ratios**, which would require a different
   expression syntax, e.g.:
   ```yaml
   L1 latency:
     value: AVG(TCP_TCP_LATENCY_sum / TCP_TOTAL_ACCESSES_sum)
   ```
   This computes `MEAN(X_i/Y_i)` -- the per-dispatch latency averaged with equal weight
   per dispatch. Whether the current parser supports this syntax would need verification.

3. **For the standard use case** (single kernel, single dispatch, or homogeneous repeated
   dispatches), both formulas and even `MEAN(X_i/Y_i)` produce identical or near-identical
   results, so the distinction is academic.

4. **For heterogeneous workloads** (multiple different kernels profiled together), the
   distinction matters. `SUM/SUM` gives the correct system-level latency, while
   `MEAN(ratio)` gives equal weight to each kernel dispatch regardless of its traffic
   contribution.

---

## Appendix A: Per-Dispatch Raw Data (nbody-block-full)

### L1 Latency: TCP_TCP_LATENCY_sum / TCP_TOTAL_ACCESSES_sum

| Dispatch | Latency Cycles | Accesses | Ratio (cyc/req) |
|----------|---------------:|---------:|----------------:|
| D0       |         68,416 |   10,240 |            6.68 |
| D1       |         68,241 |   10,240 |            6.66 |
| D2       |         69,845 |   10,240 |            6.82 |
| D3       |         67,639 |   10,240 |            6.61 |
| D4       |         70,784 |   10,240 |            6.91 |
| D5       |         67,421 |   10,240 |            6.58 |
| D6       |         69,743 |   10,240 |            6.81 |
| D7       |         68,098 |   10,240 |            6.65 |
| D8       |         66,178 |   10,240 |            6.46 |
| D9       |         66,050 |   10,240 |            6.45 |

**SUM/SUM** = 682,415 / 102,400 = **6.66**
**AVG/AVG** = 68,241.5 / 10,240 = **6.66** (identical)
**MEAN(ratio)** = mean([6.68, 6.66, ..., 6.45]) = **6.66** (identical because denominator is constant)

### UTCL1 Latency: TCP_CLIENT_UTCL1_INFLIGHT_sum / TCP_UTCL1_REQUEST_sum

| Dispatch | Inflight Cycles | Requests | Ratio (cyc/req) |
|----------|----------------:|---------:|----------------:|
| D0       |        205,333  |    2,560 |           80.21 |
| D1       |         42,966  |    2,560 |           16.78 |
| D2       |         45,248  |    2,560 |           17.68 |
| D3       |         45,409  |    2,560 |           17.74 |
| D4       |         47,986  |    2,560 |           18.74 |
| D5       |         48,787  |    2,560 |           19.06 |
| D6       |         49,575  |    2,560 |           19.37 |
| D7       |         52,936  |    2,560 |           20.68 |
| D8       |         15,360  |    2,560 |            6.00 |
| D9       |         15,360  |    2,560 |            6.00 |

Note: D0 has 4x higher UTCL1 latency (80.21 vs ~17) -- likely a cold-start TLB miss
penalty on the first dispatch. D8-D9 have the lowest latency (6.00) suggesting fully
warmed TLB. Despite this 13x range in per-dispatch latency, all three formulas produce
**22.23** because the denominator (2,560 requests) is constant across dispatches.

---

## Appendix B: Per-Channel Distribution (atomic_baseline, Dispatch 0)

### L2-to-EA Read Latency (TCC_EA0_RDREQ_LEVEL / TCC_EA0_RDREQ)

Only 112 of 128 channels active. Request count per channel ranges from 0 to 8.
Per-channel latency ranges from 230,893 to 2,715,160 cycles/request.

- **SUM/SUM** = 273,451,880 / 480 = **569,691** cycles/req
- **MEAN(per-channel ratio)** = mean of 112 active channel ratios = **615,776** cycles/req
- **Difference: +8.1%**

This divergence occurs because channels with more requests (up to 8) tend to have lower
per-request latency (due to request pipelining/batching), while channels with fewer
requests (1-2) have higher latency. SUM/SUM correctly weights toward the pipelined
channels; MEAN(ratio) gives equal weight to each channel's ratio regardless of traffic.

### L2-to-EA Write Latency (TCC_EA0_WRREQ_LEVEL / TCC_EA0_WRREQ)

All 128 channels active with identical request counts (1,048,580 each, CV = 0.00%).
Per-channel latency ranges from 2,124 to 3,696 cycles/request.

- **SUM/SUM** = **2,668.65** cycles/req
- **MEAN(per-channel ratio)** = **2,668.65** cycles/req
- **Difference: 0.00%**

When request distribution is perfectly uniform, the formulas converge.

---

## Appendix C: Test Configuration

- **GPU**: MI350 (gfx950), 256 CUs, 8 XCDs, 128 L2 channels
- **ROCm**: 7.2.0
- **Workloads analyzed**: nbody-block-full (10 dispatches), test_l2_mem (1001 dispatches),
  hbm_read_baseline, hbm_write_baseline, thrash_baseline, atomic_baseline, and 7 additional
  verification workloads
- **Analysis config**: `src/rocprof_compute_soc/analysis_configs/gfx950/3000_mem_bw.yaml`

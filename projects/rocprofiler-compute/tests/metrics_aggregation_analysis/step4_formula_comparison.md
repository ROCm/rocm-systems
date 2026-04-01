# Step 4: Formula Comparison Results

## Datasets

| Dataset | Dispatches | Kernel(s) | NaN | Purpose |
|---------|-----------|-----------|-----|---------|
| vcopy_uniform | 100 | vecCopy (1M threads) | 0 | Uniform kernel, same params — baseline |
| nbody_multi | 10 | bodyForce | 0 | Compute-heavy, different counter profile |
| vcopy_small | 5 | vecCopy (4K threads) | 0 | Small problem size, fewer dispatches |
| memcopy_multi | 1001 | memoryCopyKernel | 0 | Memory-bound, many dispatches |
| mixed_kernels | 30 | vecCopy + bodyForce + memoryCopy | 0 | **Simulated heterogeneous aggregation** |
| synthetic_clean | 10 | synthetic (3 variants) | 0 | Synthetic with large counter variation, no NaN |
| synthetic_aligned_nan | 10 | synthetic | 8 | NaN in same rows for num & denom |
| synthetic_mismatch_nan | 10 | synthetic | 20 | **NaN in different rows for num vs denom** |

---

## Key Finding 1: Uniform workloads show negligible differences

For **vcopy_uniform** (100 identical dispatches), all three formulas agree within 0.3%:

| Metric | AVG(A/B) | SUM/SUM | Diff% |
|--------|----------|---------|-------|
| L2 Hit Rate | 33.56% | 33.56% | 0.00% |
| L2-Fabric Read Latency | 1065.85 | 1065.85 | -0.00% |
| L2 Cache BW | 5252.37 | 5237.20 | 0.29% |
| IPC | 0.2177 | 0.2174 | 0.12% |

**Conclusion**: When dispatches are uniform (same kernel, same params), AVG(A/B) ≈ SUM/SUM. This confirms SUM/SUM does not regress for uniform workloads.

---

## Key Finding 2: Mixed kernels show massive divergence

For **mixed_kernels** (10 vcopy + 10 nbody + 10 memcopy dispatches), the differences are enormous:

| Metric | AVG(A/B) | SUM/SUM | Diff% | Interpretation |
|--------|----------|---------|-------|----------------|
| L2 Hit Rate | 53.14% | 33.87% | **+56.9%** | AVG overweights nbody's high hit rate (few accesses) vs memcopy's low hit rate (many accesses) |
| L2-Fabric Read Latency | 1490.5 | 2326.3 | **-35.9%** | AVG underweights memcopy's high latency (many requests) |
| L2 Cache BW | 4549.2 | 1821.4 | **+149.8%** | AVG overweights vcopy's high per-dispatch BW (short kernels) |
| IPC | 0.415 | 0.600 | **-30.8%** | AVG underweights nbody's high IPC (many busy cycles) |
| Insts per Wavefront | 418333 | 159.7 | **+261912%** | Extreme: nbody has ~1.25M insts/wave vs vcopy has 20 |
| vL1D Hit Rate | 83.33% | 87.34% | **-4.6%** | AVG overweights nbody's lower hit rate (fewer accesses) |

**This is the bug report's exact scenario.** When aggregating across different kernels (or same kernel with different inputs), AVG(A/B) produces values that don't correspond to any physically meaningful measurement.

---

## Key Finding 3: SUM/SUM vs AVG/AVG are identical when no NaN mismatch

Across all real workloads (zero NaN) and synthetic_clean (zero NaN):

| Dataset | SUM/SUM vs AVG/AVG diff | NaN mismatch |
|---------|------------------------|--------------|
| vcopy_uniform | 0.00% for all metrics | No |
| nbody_multi | 0.00% for all metrics | No |
| vcopy_small | 0.00% for all metrics | No |
| memcopy_multi | 0.00% for all metrics | No |
| mixed_kernels | 0.00% for all metrics | No |
| synthetic_clean | 0.00% for all metrics | No |
| synthetic_aligned_nan | 0.00% for all metrics | No |

**When NaN positions are aligned (same rows for both numerator and denominator), `SUM(A)/SUM(B)` = `AVG(A)/AVG(B)` exactly.** This is because `AVG(A)/AVG(B)` = `(SUM(A)/N_a) / (SUM(B)/N_b)` and when N_a = N_b, the N cancels out.

---

## Key Finding 4: NaN mismatch makes SUM/SUM ≠ AVG/AVG

For **synthetic_mismatch_nan** (numerator NaN in rows 7-9, denominator NaN in rows 5-6):

| Metric | SUM/SUM | AVG/AVG | Diff% | Explanation |
|--------|---------|---------|-------|-------------|
| L2 Hit Rate | 47.21% | 33.72% | **-28.6%** | SUM sums 7 num values and 8 denom values — different counts |
| L2-Fabric Read Latency | 43.38 | 49.57 | **+14.3%** | AVG divides by different N for num (7 valid) vs denom (8 valid) |
| IPC | 0.76 | 0.87 | **+14.3%** | Same pattern |
| Insts per Wavefront | 54.55 | 77.94 | **+42.9%** | Same pattern, amplified |

**When NaN is mismatched**:
- `SUM(A)/SUM(B)` sums over different subsets (only non-NaN values for each)
- `AVG(A)/AVG(B)` averages over different counts, introducing a `N_b/N_a` scaling factor

**Neither is clearly "correct" when NaN is mismatched** — both are computing ratios of quantities measured over different subsets of dispatches. However, from the real profiling data, **NaN mismatch does not occur** (all workloads had zero NaN after the rocpd join). This makes the SUM/SUM vs AVG/AVG distinction irrelevant for real data.

---

## Key Finding 5: Synthetic data confirms the theoretical analysis

For **synthetic_clean** (no NaN, intentionally varied counter magnitudes):

| Metric | AVG(A/B) | SUM/SUM | Diff% | Category |
|--------|----------|---------|-------|----------|
| L2 Hit Rate | 42.46% | 43.44% | -2.3% | rate |
| L2-Fabric Read Latency | 58.06 | 67.70 | -14.2% | latency |
| L2 Cache BW | 1279.36 | 328.00 | +290.1% | throughput |
| IPC | 0.88 | 1.18 | -25.1% | normalized_counter |
| vL1D Hit Rate | 68.40% | 94.97% | -28.0% | rate |

The synthetic data was designed with large counter variation between dispatches (simulating different kernels). The differences confirm:
- **Throughput**: AVG(work/time) hugely overestimates by overweighting short-duration dispatches (+290%)
- **Rates**: AVG(hits/total) gives a misleading value when access counts vary between dispatches (-28% for vL1D)
- **Latency**: AVG(LEVEL/REQ) underestimates because it underweights dispatches with many requests (-14%)
- **Normalized counters**: AVG(insts/cycles) is skewed by dispatches with few busy cycles (-25%)

---

## Key Finding 6: MIN/MAX spread as indicator of dispatch variability

For **mixed_kernels**, the MIN/MAX spread shows how different dispatches are:

| Metric | MIN(A/B) | SUM/SUM | MAX(A/B) | Spread |
|--------|----------|---------|----------|--------|
| L2 Hit Rate | 33.33% | 33.87% | 92.53% | 59.2 pp |
| L2-Fabric Read Latency | 1035.4 | 2326.3 | 2341.5 | 1306 cycles |
| IPC | 0.19 | 0.60 | 0.84 | 0.65 |

When the MIN/MAX spread is wide relative to the aggregate, it indicates the AVG(A/B) formula is likely to be misleading (since it treats these wildly different dispatches equally).

---

## Summary

| Formula | When to use | Real data agreement with SUM/SUM |
|---------|-------------|----------------------------------|
| `AVG(A/B)` | **Never for aggregate metrics** — gives equal weight per dispatch regardless of magnitude. Produces incorrect results for mixed kernels. | Diverges up to 261912% on mixed kernels |
| `SUM(A)/SUM(B)` | **Default for all ratio metrics** — weights by denominator magnitude (request count, busy cycles, time). Matches physical meaning. | Reference |
| `AVG(A)/AVG(B)` | **Identical to SUM/SUM when no NaN mismatch** (confirmed on all real data). Differs only with cross-pass NaN, which doesn't occur in current profiling pipeline. | 0.00% on all real data |

**Recommendation**: Use `SUM(A)/SUM(B)` for all ratio metrics. On current MI350 profiling pipeline (rocpd format), `AVG(A)/AVG(B)` gives identical results, but `SUM/SUM` is more explicit about intent and robust to any future NaN scenarios.

# Context: Metrics AVG(A/B) → SUM(A)/SUM(B) Fix

> **Purpose**: Session context file. Resume this work by pointing Claude Code at this file.
> **Branch**: `users/xuchen-amd/metrics_avg_fix`
> **Last updated**: 2026-03-29

---

## What This Is About

The `AVG()` aggregation function in rocprofiler-compute's metric evaluation computes the **mean of per-dispatch ratios** (`mean(A_i/B_i)`). This gives equal weight to every dispatch regardless of magnitude, producing incorrect aggregate values when dispatches differ in size. The fix is to change ratio metrics to use **pairwise SUM(A)/SUM(B)** — sum only dispatches where both numerator and denominator are valid.

---

## Completed Research

### Step 1: Metric Taxonomy
- **Output**: `step1_metric_taxonomy.csv`
- Parsed all 19 gfx950 YAML config files
- Catalogued 397 metric expressions, categorized as: rate, throughput, latency, utilization, normalized_counter, absolute, constant_denom, composite
- Identified which metrics contain division (ratio metrics) vs which are sums/single counters

### Step 2: Candidate Formulas
- **Output**: `step2_candidate_formulas.csv`, `step2_candidate_formulas.md`
- For each metric category, derived the mathematically correct aggregation formula
- Conclusion: SUM(A)/SUM(B) is correct for all ratio categories (rates, throughput, latency, utilization, normalized counters)

### Step 3: Profiled Workloads on MI350
- **Profiled datasets** (all under `workloads/*/MI350/pmc_perf.csv`):
  - `vcopy_uniform` — 100 dispatches, single kernel, identical params
  - `nbody_multi` — 10 dispatches, compute-heavy bodyForce kernel
  - `vcopy_small` — 5 dispatches, small problem size
  - `memcopy_multi` — 1001 dispatches, memory-bound copy kernel
- **Synthetic datasets** (created in step4 script):
  - `mixed_kernels` — 30 dispatches from 3 different kernels combined
  - `synthetic_clean` — 10 dispatches with intentional large variation, no NaN
  - `synthetic_aligned_nan` — NaN in same rows for num and denom
  - `synthetic_mismatch_nan` — NaN in different rows for num vs denom

### Step 4: Formula Comparison
- **Script**: `step4_compare_formulas.py`
- **Output**: `step4_formula_comparison.csv`, `step4_formula_comparison.md`
- Computed AVG(A/B), SUM(A)/SUM(B), AVG(A)/AVG(B) for 6 representative metrics across all datasets
- **Key findings**:
  - Uniform workloads: all formulas agree within 0.3%
  - Mixed kernels: divergence up to +261,912% (Instructions per Wavefront)
  - SUM/SUM = AVG/AVG when no NaN mismatch (confirmed on all real data)

### Step 4b: NaN Correctness Analysis
- **Script**: `step4b_nan_correctness.py`
- **Output**: `step4b_nan_correctness.csv`
- Tested 4 formulas against ground truth under 5 NaN injection patterns
- **Key finding**: Pairwise SUM/SUM (only include rows where both A and B are valid) has lowest error — 6.6% MAE vs 36.2% for plain SUM/SUM, 29.6% for AVG/AVG, 17.6% for AVG(A/B)

### Final Report
- **Output**: `metrics_aggregation_report.md`
- Professional report with problem statement, mathematical analysis, profiling evidence, NaN analysis, proposed change, and impact assessment
- Target audience: users of the existing AVG() metrics

---

## Proposed Implementation (Not Yet Started)

### Approach: Division-Intercepting Wrapper (zero YAML changes)

Intercept the `/` operator at the pandas Series level during `eval()` so that `A / B` produces a `PairwiseRatioSeries` that remembers its numerator and denominator. Then `to_avg()` detects this and computes `SUM(A[valid])/SUM(B[valid])`.

### Implementation plan file
- **Location**: `.claude/plans/staged-wobbling-barto.md`
- Contains complete design with 4 new classes, modification list for all aggregation functions, and expression pattern correctness matrix

### Key design points
1. **`DivisionTrackingSeries`**: Wraps `pd.Series` from DataFrame column lookups. All arithmetic (`+`, `-`, `*`) returns another `DivisionTrackingSeries` to preserve tracking. Only `__truediv__` produces `PairwiseRatioSeries`.
2. **`PairwiseRatioSeries`**: Stores ratios + numerator + denominator. Scalar multiplication preserves it (correct for `100 * A/B`). Addition/subtraction destroys it (correct for `100 - hit_rate`). `.where()` preserves it.
3. **`TrackedColumnAccessor` / `TrackedDataFrame`**: Wraps `raw_pmc_df` dict and DataFrames to return `DivisionTrackingSeries` on column access.
4. **`to_avg()` change**: If input is `PairwiseRatioSeries`, compute pairwise SUM/SUM. If `DivisionTrackingSeries` (no division happened), unwrap and use regular mean.

### Files to modify
- `src/utils/parser.py` — add 4 new classes, modify `to_avg` + 12 other aggregation functions, modify `eval_expression` (1 line)
- `src/rocprof_compute_analyze/analysis_db.py` — import `TrackedDataFrame`, wrap `pmc_df` in `evaluate()` (1 line)
- `tests/test_pairwise_ratio.py` — new unit test file

### Critical edge case (caught during planning)
The original plan agent's design had `DivisionTrackingSeries.__add__` return plain `pd.Series`, which broke tracking for expressions like `AVG((counter * 128) / (End_Timestamp - Start_Timestamp))` where intermediate arithmetic happens before division. The corrected design keeps `DivisionTrackingSeries` through all arithmetic ops — only division produces `PairwiseRatioSeries`.

---

## Codebase Key Locations

| File | What | Lines |
|------|------|-------|
| `src/utils/parser.py` | `to_avg()` function | 128-160 |
| `src/utils/parser.py` | `SUPPORTED_CALL` dict (AVG→to_avg mapping) | 78-99 |
| `src/utils/parser.py` | `CodeTransformer` (AST transformation) | 407-459 |
| `src/utils/parser.py` | `MetricEvaluator.eval_expression()` | 475-551 |
| `src/utils/parser.py` | `build_eval_string()` | 553-654 |
| `src/utils/parser.py` | `eval_metric()` (orchestration) | 1189-1271 |
| `src/rocprof_compute_analyze/analysis_db.py` | `db_analysis.evaluate()` (second eval path) | 458-529 |
| `src/utils/utils_analysis.py` | `process_rocpd_csv()` (NaN introduction point) | 597-637 |
| `src/utils/utils_analysis.py` | `impute_counters_iteration_multiplex()` | 395-492 |
| `src/utils/file_io.py` | `create_df_pmc()` | 214-268 |
| `src/rocprof_compute_soc/analysis_configs/gfx950/*.yaml` | Metric expressions (19 files) |  |
| `src/rocprof_compute_soc/analysis_configs/gfx950/3000_mem_bw.yaml` | Already uses SUM/SUM (reference) |  |

### How metric expressions flow through the code
```
YAML expression (e.g., "AVG(TCC_HIT_sum / (TCC_HIT_sum + TCC_MISS_sum))")
  → build_eval_string(): AST transform, AVG→to_avg, counters→raw_pmc_df['pmc_perf']['counter']
  → eval_metric(): iterates metric tables, calls MetricEvaluator.eval_expression()
  → eval(): Python evaluates the string, DataFrame subscripts return pd.Series, arithmetic is element-wise
  → to_avg(series): currently calls series.mean(), proposed: detect PairwiseRatioSeries → SUM/SUM
  → scalar result stored in metric display DataFrame
```

---

## Files Created During This Research

| File | Description |
|------|-------------|
| `step1_metric_taxonomy.csv` | All 397 gfx950 metrics with categories and formula analysis |
| `step2_candidate_formulas.csv` | Candidate formulas per metric category |
| `step2_candidate_formulas.md` | Detailed rationale for each category |
| `step4_compare_formulas.py` | Script: computes AVG/SUM/AVG-AVG across all datasets |
| `step4_formula_comparison.csv` | Raw results: 48 rows (8 datasets x 6 metrics) |
| `step4_formula_comparison.md` | Key findings summary |
| `step4b_nan_correctness.py` | Script: ground truth comparison under NaN patterns |
| `step4b_nan_correctness.csv` | Raw results: 25 rows (5 patterns x 5 metrics) |
| `metrics_aggregation_report.md` | Final professional report for stakeholders |
| `METRICS_AVG_FIX_CONTEXT.md` | This file |
| `.claude/plans/staged-wobbling-barto.md` | Implementation plan (approved but not executed) |

---

## What Remains

- [ ] **Implementation**: Write the `DivisionTrackingSeries`, `PairwiseRatioSeries`, `TrackedColumnAccessor`, `TrackedDataFrame` classes and modify `to_avg()` + other aggregation functions (plan is in `.claude/plans/staged-wobbling-barto.md`)
- [ ] **Unit tests**: `tests/test_pairwise_ratio.py` covering all 13 expression patterns from the correctness matrix
- [ ] **Integration tests**: Verify existing `tests/test_analyze_workloads.py` passes
- [ ] **Backward compat verification**: Run `rocprof-compute analyze` on vcopy_uniform and confirm results match pre-change values within float tolerance
- [ ] **Correctness verification**: Run on mixed_kernels and confirm results match SUM/SUM expectations
- [ ] **Other architectures**: gfx908/gfx90a/gfx940/gfx941/gfx942 use similar YAML expressions — no YAML changes needed since the fix is in `parser.py`, but integration tests should cover at least one other arch

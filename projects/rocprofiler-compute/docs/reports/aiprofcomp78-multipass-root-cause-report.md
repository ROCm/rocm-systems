# AIPROFCOMP-78 — Multi-Pass Root Cause Validation Report

**Branch:** `users/feizheng10/cpx_fix`  
**Date:** 2026-08-27  
**GPU:** MI300X (gfx942), CPX partition  
**Data:** `/home/feizheng/Downloads/aiprofcomp78-cpx-data/`

---

## 1. Question

Do Percent metric bound violations (>100%) come from **multi-pass counter stitching**, or from something else (single-pass HW bug, CPX partition alone, formula error)?

---

## 2. Method

1. **Analyze logs** — full-panel multi-pass `rocprof-compute analyze` output (`analyze_*_cpx.log`).
2. **Merged PMC** — per-dispatch ratios from stitched `pmc_perf.csv`.
3. **Per-pass PMC** — which `results_pmc_perf_*.csv` / perfmon YAML contains each counter partner.

If partners are in **different perfmon passes** but merged into one row, violations after merge are consistent with **multi-pass imputation**, not necessarily HW bugs.

---

## 3. Workloads

| Workload | Kernels (approx) | Perfmon passes | Partition |
|----------|------------------|----------------|-----------|
| occupancy_cpx | occupancy sample (`./sample/occupancy`) | 13 | CPX (`cu_per_gpu=38`) |
| mat_exp | GEMM-heavy | 13 | CPX |
| rocflop | rocBLAS flop test | 13 | CPX |

All profiles: full panel, multi-pass (default `rocprof-compute profile` behavior).

---

## 4. Analyze output — affected metrics

Values from **uncapped** analyze (pre-#10717) logs:

| Workload | Metric | Avg | Max | Violation |
|----------|--------|-----|-----|-----------|
| **occupancy_cpx** | HBM Read Traffic | **103.23%** | — | Avg >100% |
| **mat_exp** | HBM Read Traffic | 99.11% | **188.24%** | Max >100% |
| **mat_exp** | Workgroup Manager Utilization | 39.91% | **739.60%** | Max >100% |
| **mat_exp** | Data-Return Busy | 2.59% | **345.38%** | Max >100% |
| **mat_exp** | CPF Utilization | **101.05%** | 100% | Avg >100% |
| rocflop | HBM Read Traffic | 99.70% | 100% | None |
| rocflop | Workgroup Manager Utilization | 100.08% | 100.52% | Mild avg |

**Conclusion:** Violations are **sporadic** (not every workload/metric) and **worst on mat_exp** (many short dispatches).

---

## 5. Merged PMC — per-dispatch `a > b`

Computed from `pmc_perf.csv` (stitched multi-pass table):

| Workload | Metric | Dispatches | `a > b` rows | % rows | Max `100×a/b` | `100×SUM(a)/SUM(b)` |
|----------|--------|------------|--------------|--------|---------------|---------------------|
| mat_exp | HBM Read | 208 | 15 | 7.2% | 188.2% | 99.11% |
| mat_exp | Workgroup Manager Utilization (`GRBM_SPI/GUI`) | 206 | 2 | 1.0% | **739.6%** | 40.06% |
| occupancy | HBM Read | 8 | 2 | 25.0% | 103.9% | **103.23%** |
| occupancy | Workgroup Manager Utilization | 8 | 1 | 12.5% | 100.0% | 99.99% |
| rocflop | HBM Read | 7 | 0 | 0% | 100.0% | 99.70% |
| rocflop | Workgroup Manager Utilization | 7 | 1 | 14.3% | 100.5% | 100.08% |

**Test case occupancy** HBM avg inflation (103.23%) matches `SUM(a)/SUM(b)` on merged data with a minority of bad rows.

---

## 6. Smoking gun — partners in different perfmon passes (mat_exp)

| Metric | Numerator | Denominator | Numerator pass | Denominator pass |
|--------|-----------|-------------|----------------|------------------|
| HBM Read | `TCC_EA0_RDREQ_DRAM_sum` | `TCC_EA0_RDREQ_sum` | `SQ_INST_LEVEL_LDS_ACCUM` | `SQ_INST_LEVEL_SMEM_ACCUM` |
| Workgroup Manager Utilization | `GRBM_SPI_BUSY` | `GRBM_GUI_ACTIVE` | `SQC_ICACHE_INFLIGHT_LEVEL_ACCUM` | `SQC_DCACHE_INFLIGHT_LEVEL_ACCUM` |
| CPF Util | `CPF_CPF_STAT_BUSY` | `CPF_CPF_STAT_IDLE` | `SQC_DCACHE_INFLIGHT_LEVEL_ACCUM` | `SQC_ICACHE_INFLIGHT_LEVEL_ACCUM` |

No single perfmon pass contains **both** partners for these metrics. Analyze merges them by `Dispatch_ID`, so per-dispatch ratios can exceed 100% even when each counter is valid in its own pass.

---

## 7. Verdict

| Hypothesis | Supported? |
|------------|------------|
| Multi-pass stitching causes bound violations | **Yes** — strong (pass split + merged violations) |
| Systematic HW counter bug (all dispatches bad) | **No** — minority of rows; workload-dependent |
| CPX partition alone | **Partial** — CPX data shown; SPX occupancy log also had HBM max 111.76% |
| Formula error in YAML | **No** — `SUM(a)/SUM(b)` is correct when `a≤b` per row |

**Recommended next step:** Re-profile with `--block 17` (and 5, 6, 15 for other metrics) and confirm `a > b` rate drops on **single-pass** PMC files. Until then, analyze-time caps are **display guards** for known multi-pass noise, not proof of counter correctness.

---

## 8. Reproduction commands

```bash
# Inspect merged ratios (example: mat_exp HBM)
python3 -c "
import pandas as pd
pmc = pd.read_csv('mat_exp/0/pmc_perf.csv')
a = pmc[pmc.Counter_Name=='TCC_EA0_RDREQ_DRAM_sum']
b = pmc[pmc.Counter_Name=='TCC_EA0_RDREQ_sum']
m = a.merge(b, on=['Dispatch_ID','Kernel_Name'], suffixes=('_a','_b'))
m['pct'] = m.Counter_Value_a / m.Counter_Value_b * 100
print(m[m.pct>100][['Dispatch_ID','Counter_Value_a','Counter_Value_b','pct']].head())
"

# Count perfmon passes
ls mat_exp/0/perfmon/pmc_perf*.yaml | wc -l
```

---

## References

- Design: [Problem statement](../design/metric-counter-correction-problem-statement.md)
- Fix PR: #10717

# Percent Metric Bounds — Problem Statement

**Status:** Draft for team review  
**Audience:** All rocprofiler-compute stakeholders (profiler designers, perfmon counter designers)  
**JIRA:** AIPROFCOMP-78  
**Related:** [Correction methods (normative)](metric-counter-correction-methods.md) · [Single-pass grouping evaluation](single-pass-counter-grouping-evaluation.md) · [AIPROFCOMP-78 validation data](../reports/aiprofcomp78-multipass-root-cause-report.md)

---

## 1. What users see

During full-panel `analyze` on MI300-class GPUs (gfx940–942), some **Percent** metrics violate expected bounds:

| Symptom | Example metrics | Example values (before correction) |
|---------|-------------------|-------------------------------------|
| Avg slightly above 100% | HBM Read Traffic | **103.23%** avg (occupancy CPX) |
| Max far above 100% | Workgroup Manager Utilization | **739.60%** max (mat_exp CPX) |
| Max above 100% | HBM Read Traffic, Data-Return Busy | **188.24%**, **345.38%** max (mat_exp CPX) |
| Negative splits | Remote Read/Write Traffic | Already handled by shipped `NOISE_CLAMP` |

**Intentional exception:** VALU Utilization on gfx942 can exceed 100% (dual-issue). Do not cap — use `ValuDualIssueDetector` warnings.

---

## 2. Why it happens (hypotheses)

| Cause | Mechanism | Evidence strength |
|-------|-----------|-------------------|
| **Multi-pass profiling** | Perfmon slot limits require many kernel re-runs. Analyze **stitches** counters from different passes onto one dispatch row. | **Strong** — counter partners land in different perfmon YAML passes (see §3) |
| **Asynchronous sampling** | Counters in different IP blocks are not sample-aligned even within one pass. | Moderate — shows up on short dispatches |
| **Non-partition pairs** | Numerator/denominator are not strict subsets (e.g. SPI busy vs GUI active). | By design — capping is display-only for these |
| **Hardware / driver bug** | Persistent `a > b` under **single-pass** collection on stable workloads. | Must be ruled out before caps (escalate, do not clamp-only) |
| **Dynamic power management (DPM)** | Clock and power state may differ across kernel re-runs in multi-pass profiling, reducing comparability of counters stitched onto the same dispatch row. | Speculative — possible contributor; not yet validated in AIPROFCOMP-78 data |

**Key insight:** A row in merged `pmc_perf.csv` does **not** mean all counters on that row were collected in the same profiling pass.

---

## 3. Evidence from AIPROFCOMP-78 workloads (gfx942 CPX)

Data path: `/home/feizheng/Downloads/aiprofcomp78-cpx-data/` (also summarized in [validation report](../reports/aiprofcomp78-multipass-root-cause-report.md)).

**System:** MI300X, gfx942, CPX partition (`cu_per_gpu=38`, `num_xcd=1`), ROCm 7.15, full-panel profile (**13 perfmon passes** per workload).

### 3.1 Merged analyze output (multi-pass, full panel)

| Workload | Metric | Avg | Min | Max | Notes |
|----------|--------|-----|-----|-----|-------|
| occupancy CPX | HBM Read Traffic | **103.23%** | — | — | Matches classic avg inflation case |
| mat_exp CPX | HBM Read Traffic | 99.11% | 36.56% | **188.24%** | Avg OK; max exploded |
| mat_exp CPX | Workgroup Manager Utilization | 39.91% | 8.76% | **739.60%** | Extreme max on short dispatch |
| mat_exp CPX | Data-Return Busy | 2.59% | 0.05% | **345.38%** | Extreme max |
| mat_exp CPX | CPF Utilization | **101.05%** | 100% | 100% | Avg inflation |
| rocflop CPX | HBM Read Traffic / Workgroup Manager Utilization | ≤100.08% | — | ≤100.5% | Mild or no violation |

Violations are **workload-dependent** and **sporadic** (minority of dispatches), not uniform across every kernel.

### 3.2 Raw PMC: partners collected in different passes

For **mat_exp CPX**, perfmon pass assignment shows ratio partners are **never co-located**:

| Metric pair | Numerator counter | Denominator counter | Pass file (numerator) | Pass file (denominator) |
|-------------|-------------------|---------------------|------------------------|-------------------------|
| HBM Read | `TCC_EA0_RDREQ_DRAM_sum` | `TCC_EA0_RDREQ_sum` | `SQ_INST_LEVEL_LDS_ACCUM` | `SQ_INST_LEVEL_SMEM_ACCUM` |
| Workgroup Manager Utilization | `GRBM_SPI_BUSY` | `GRBM_GUI_ACTIVE` | `SQC_ICACHE_INFLIGHT_LEVEL_ACCUM` | `SQC_DCACHE_INFLIGHT_LEVEL_ACCUM` |
| CPF Utilization | `CPF_CPF_STAT_BUSY` | `CPF_CPF_STAT_IDLE` | `SQC_DCACHE_INFLIGHT_LEVEL_ACCUM` | `SQC_ICACHE_INFLIGHT_LEVEL_ACCUM` |

After analyze merges passes into one table, per-dispatch ratios can exceed 100% even when each counter is valid in isolation.

### 3.3 Per-dispatch violations on merged data (mat_exp CPX)

| Metric | Dispatches | Rows with `a > b` | % bad | Max ratio | `SUM(a)/SUM(b)` avg |
|--------|------------|-------------------|-------|-----------|---------------------|
| HBM Read | 208 | 15 | 7.2% | 188.2% | 99.11% |
| Workgroup Manager Utilization | 206 | 2 | 1.0% | **739.6%** | 40.06% |

Example HBM outlier (dispatch 13): `a=192`, `b=102` → 188% (GEMM kernel).

### 3.4 What this does **not** prove yet

- We do **not** yet have single-pass **re-profile** PMC CSVs in this dataset showing all violations disappear (validation recipe in correction-methods doc §5.3).
- SPX full-panel runs on the same workloads show smaller or no violations for some metrics — partition mode affects severity but is not the sole root cause.
- Systematic single-pass violations would indicate HW/driver investigation, not formula caps alone.

---

## 4. Problem scope by metric type

| Type | Example | Physical invariant? | Primary issue |
|------|---------|---------------------|---------------|
| **Partition ratio** | HBM Read = DRAM reads / total reads | Yes (`a ⊆ b`) | Multi-pass stitching → avg >100% or sporadic max |
| **Partition utilization** | CPF busy / (busy+idle) | Yes | Same |
| **Non-partition ratio** | Workgroup Manager Utilization (SPI busy / GUI active) | No | Max spikes; avg cap misleading |
| **Subtraction split** | Remote = total − DRAM | Yes (≥ 0) | Shipped `NOISE_CLAMP` |
| **Dual-issue VALU** | VALU Utilization | Can exceed 100% | Documented exception |

---

## 5. Response options (summary)

| Approach | Fixes root cause? | Accuracy | Cost |
|----------|-------------------|----------|------|
| **Single-pass / block-only profile** | Yes, when partners co-locate | Best | Fewer counters or more passes — see [grouping evaluation](single-pass-counter-grouping-evaluation.md) |
| **Grouping policy** | Partial — co-locates partners when allocator can pack them | Good when same pass achieved | Can **increase** full-panel pass count — see evaluation doc |
| **Analyze-time caps** | No — display guard | Bounded output; see methods doc | Zero profile cost |

**Proposed near-term path:** validate with block-only/single-pass profiles → prefer grouping or reduced `--block` where affordable → use analyze-time caps only for remaining multi-pass full-panel noise.

---

## 6. Open questions for the team

1. Do we require all Percent metrics ≤ 100% in product UI, or allow documented exceptions (VALU) plus optional warnings?
2. What single-pass evidence bar is required before merging formula caps (qualitative today — see methods doc §5.4)?
3. Should gfx942 `profiling_counter_grouping_policy.yaml` gain entries for HBM Read Traffic, Workgroup Manager Utilization, and CPF Utilization despite full-panel pass-count cost?
4. When single-pass validation still shows violations, what is the escalation path to driver/HW?

---

## References

- Local CPX workloads: `aiprofcomp78-cpx-data/` (occupancy, mat_exp, rocflop)
- Analyze logs: `analyze_*_cpx.log` in same directory
- Implementation PR: #10717 (`users/feizheng10/cpx_fix`)
- Design methods PR: #10655 (`users/feizheng10/metric-correction-design`)

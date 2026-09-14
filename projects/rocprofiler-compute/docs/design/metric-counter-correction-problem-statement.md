# Percent Metric Bounds — Problem Statement

**Status:** Draft for team review  
**Audience:** All rocprofiler-compute stakeholders (profiler designers, perfmon counter designers)  
**JIRA:** AIPROFCOMP-78  
**Related:** [Correction methods (normative)](metric-counter-correction-methods.md) · [Single-pass grouping evaluation](single-pass-counter-grouping-evaluation.md) · [AIPROFCOMP-78 validation data](../reports/aiprofcomp78-multipass-root-cause-report.md) · [Hardware validation runbook](../reports/aiprofcomp78-hardware-validation-runbook.md)

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
| **Hardware / driver bug** | Persistent `a > b` under **single-pass** collection on stable workloads. | Must be ruled out before caps (escalate, do not clamp-only). **HBM on occupancy ruled out** — see §3.5 |
| **Dynamic power management (DPM)** | Clock and power state may differ across kernel re-runs in multi-pass profiling, reducing comparability of counters stitched onto the same dispatch row. | Speculative — possible contributor; not yet validated in AIPROFCOMP-78 data |

**Key insight:** A row in merged `pmc_perf.csv` does **not** mean all counters on that row were collected in the same profiling pass.

---

## 3. Evidence from AIPROFCOMP-78 workloads (gfx942 CPX)

Data path: `/home/feizheng/Downloads/aiprofcomp78-cpx-data/` (also summarized in [validation report](../reports/aiprofcomp78-multipass-root-cause-report.md)).

**System:** MI300X, gfx942, CPX partition (`cu_per_gpu=38`, `num_xcd=1`), ROCm 7.15, full-panel profile (**13 perfmon passes** per workload).

### 3.1 Merged analyze output (multi-pass, full panel)

| Workload | Metric | Avg | Min | Max | Notes |
|----------|--------|-----|-----|-----|-------|
| occupancy_cpx | HBM Read Traffic | **103.23%** | 90.41% | **103.9%** | Matches classic avg inflation case |
| mat_exp CPX | HBM Read Traffic | 99.11% | 36.56% | **188.24%** | Avg OK; max exploded |
| mat_exp CPX | Workgroup Manager Utilization | 39.91% | 8.76% | **739.60%** | Extreme max on short dispatch |
| mat_exp CPX | Data-Return Busy | 2.59% | 0.05% | **345.38%** | Extreme max |
| mat_exp CPX | CPF Utilization | **101.05%** | 100% | 100% | Avg inflation |
| rocflop CPX | HBM Read Traffic | 99.70% | 96.98% | 100% | Mild or no violation |
| rocflop CPX | Workgroup Manager Utilization | 100.08% | 99.60% | 100.52% | Mild avg |

**Scope (all three workloads):** Each full-panel run reports **54 Percent metrics**. With bound violations defined as avg, min, or max **> 100%** (excluding intentional VALU dual-issue on gfx942), the three CPX test cases show **11 abnormal workload×metric combinations** in total — **occupancy_cpx 1**, **mat_exp CPX 7**, **rocflop CPX 3** (**9** distinct metric ids). The **7 rows above** are **representatives** chosen for §3.5 hardware validation, not an exhaustive inventory. Other abnormal metrics in the same logs include mat_exp CPC Stall Rate (max **199.67%**), CPC Packet Decoding Utilization (max **148.69%**), Uncached Read Traffic (max **126.88%**), and rocflop HBM Write and Atomic Traffic (avg **102.01%**, max **113.38%**).

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

- **HBM Read Traffic:** single-pass validation **done** for test case occupancy and **mat_exp** (§3.5). Build `mat_exp` with ROCm **7.15** + `hipcc` (rocblas is in that tree; earlier “no rocblas” referred to default `/opt/rocm`).
- **Workgroup Manager Utilization:** panel `--block 6` splits partners (mat_exp stitched max **605.9%** vs merged **739.6%**). **Metric-id `--block 6.1.2`** on mat_exp: **1 pass**, co-located, **0/208** `a > b`, max **100.0%** — **validated**.
- **CPF Utilization:** panel `--block 5` splits partners. **Metric-id `--block 5.1.0`** on mat_exp: **1 pass**, **0/208** `> 100%`, max **100.0%** — **validated** (merged uncapped avg **101.05%** not reproduced).
- **Data-Return Busy:** panel `--block 15` splits partners (stitched max **64.4%**). **Metric-id `--block 15.4.0`**: **1 pass**, **0/208** `a > b`, max **0.04%** vs merged **345.38%** — **validated**.
- SPX full-panel runs on the same workloads show smaller or no violations for some metrics — partition mode affects severity but is not the sole root cause.
- Systematic single-pass violations would indicate HW/driver investigation, not formula caps alone.

### 3.5 Single-pass hardware validation (darkstar, 2026-08-27)

Data path: `~/Downloads/aiprofcomp78-darkstar-data/` (and on-node workloads not yet copied — see **Validation** column). Conductor node `hpe-darkstar-ccs-aus-e12-03`, MI300X gfx942, ROCm 7.15, bundled rocprof-compute **3.8.0**. Step-by-step commands: [hardware runbook](../reports/aiprofcomp78-hardware-validation-runbook.md).

**Same rows as §3.1** — merged full-panel columns repeat §3.1 for side-by-side reading. Single-pass columns use darkstar profiles on `./sample/occupancy`, `./sample/rocflop`, or **HPCTrainingExamples `mat_exp`** (streams_sync). Use **metric-id `--block`** (e.g. `6.1.2`) when panel `--block` (e.g. `6`) still splits ratio partners across perfmon passes.

Per-dispatch check: partition metrics use raw `100 × a / b` (HBM: `DRAM / RDREQ_sum`; CPF: `busy / (busy + idle)`); Workgroup Manager Utilization uses `100 × SPI / GUI` (non-partition); Data-Return Busy uses `100 × TD_TD_BUSY_sum / (GRBM_GUI_ACTIVE × cu_per_gpu)`.

| Workload | Metric | Merged Avg (§3.1) | Merged Max (§3.1) | Single-pass profile | Partners co-located? | Dispatches | Rows `a > b` | Max ratio | Validation |
|----------|--------|-------------------|-------------------|---------------------|----------------------|------------|--------------|-----------|------------|
| occupancy_cpx | HBM Read Traffic | **103.23%** | **103.9%** | `--block 17` occupancy | **Yes** (`pmc_perf_2.yaml`) | 8 | **0 (0%)** | **100.0%** | **Validated** |
| mat_exp CPX | HBM Read Traffic | 99.11% | **188.24%** | `--block 17` mat_exp (streams_sync) | **Yes** (`pmc_perf_2.yaml`) | 208 | **0 (0%)** | **100.0%** | **Validated** |
| mat_exp CPX | Workgroup Manager Utilization | 39.91% | **739.60%** | **`--block 6.1.2`** mat_exp | **Yes** (`pmc_perf_0.yaml`, **1 pass**) | 208 | **0 (0%)** | **100.0%** | **Validated** |
| mat_exp CPX | Data-Return Busy | 2.59% | **345.38%** | **`--block 15.4.0`** mat_exp | **Yes** (`pmc_perf_0.yaml`, **1 pass**) | 208 | **0 (0%)** | **0.04%** | **Validated** |
| mat_exp CPX | CPF Utilization | **101.05%** | 100% | **`--block 5.1.0`** mat_exp | **Yes** (`pmc_perf_0.yaml`, **1 pass**) | 208 | **0 (0%)** | **100.0%** | **Validated** |
| rocflop CPX | HBM Read Traffic | 99.70% | 100% | `--block 17` rocflop | **Yes** (`pmc_perf_2.yaml`) | 7 | **0 (0%)** | **100.0%** | **Validated** |
| rocflop CPX | Workgroup Manager Utilization | 100.08% | 100.52% | **`--block 6.1.2`** rocflop | **Yes** (`pmc_perf_0.yaml`, **1 pass**) | 7 | **0 (0%)** | **100.0%** | **Validated** |

**Validation:** **Validated** = single-pass profile with ratio partners in the **same perfmon pass** on the same workload as §3.1. Use **metric-id `--block`** (e.g. `6.1.2`) when a panel `--block` would split partners; HBM on `--block 17` already co-locates. Convenience set: `--set aiprofcomp78_bounds` (gfx942: WGM + CPF + Data-Return metric ids, 1 pass).

**Primary A/B:** merged full-panel violations (HBM **7.2%** / max **188%**; WGM max **739.6%**; Data-Return max **345.38%**) vs co-located metric-id profiles (**0%** bad rows, bounded max) on **mat_exp** — consistent with multi-pass stitching, not persistent HW/driver error under co-located collection.

**Conclusion:** All §3.1 bound-violation metrics in the table **clear the single-pass bar** under co-located collection via metric-id `--block` where needed (hardware-validated on bundled **rocprof-compute 3.8.0**; the same `--block` path applies on **rocprof-compute 3.10+**). **Grouping policy** for full-panel profiles is a separate production trade-off — it requires **rocprof-compute 3.10+** to change perfmon layout on hardware (not yet validated; see grouping evaluation §8).

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

**Proposed near-term path:** block-only/single-pass validation **underway** (HBM/test case occupancy confirmed §3.5) → prefer grouping or reduced `--block` where affordable → use analyze-time caps only for remaining multi-pass full-panel noise. Grouping policy pass-count cost still requires rocprof-compute **3.10+** on hardware (see grouping evaluation §8).

---

## 6. Open questions for the team

1. Do we require all Percent metrics ≤ 100% in product UI, or allow documented exceptions (VALU) plus optional warnings?
2. What single-pass evidence bar is required before merging formula caps (qualitative today — see methods doc §5.4)? **Partial answer:** one partition-metric family (HBM on test case occupancy) clears the bar via `--block 17`; extend to mat_exp and non-partition metrics before generalizing.
3. Should gfx942 `profiling_counter_grouping_policy.yaml` gain entries for HBM Read Traffic, Workgroup Manager Utilization, and CPF Utilization despite full-panel pass-count cost?
4. When single-pass validation still shows violations, what is the escalation path to driver/HW?

---

## References

- Local CPX workloads (multi-pass baseline): `~/Downloads/aiprofcomp78-cpx-data/` (test case occupancy, mat_exp, rocflop)
- Darkstar hardware validation (single-pass): `~/Downloads/aiprofcomp78-darkstar-data/` · `analyze_darkstar_hardware.py`
- Analyze logs: `analyze_*_cpx.log` in CPX data directory
- [Hardware validation runbook](../reports/aiprofcomp78-hardware-validation-runbook.md)
- Implementation PR: #10717 (`users/feizheng10/cpx_fix`)
- Design methods PR: #10655 (`users/feizheng10/metric-correction-design`)

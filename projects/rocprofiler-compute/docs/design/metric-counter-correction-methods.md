# Metric Counter Correction Methods — Design Guidance

**Status:** Draft design guidance  
**Context:** Percent and ratio metrics that violate expected bounds (e.g. > 100%, negative splits) due to multi-pass profiling variance, counter pairing semantics, or potential hardware counter issues — observed across CDNA/RDNA architectures during full-panel analyze runs.  
**Audience:** Primary — designers and maintainers choosing correction methods for metric YAML; secondary — source material for enduser-facing explanations.  
**Related code (shipped):** `src/utils/metrics/noise_clamper.py`, `src/utils/utils_analysis.py`, `src/rocprof_compute_soc/soc_base.py`, `src/rocprof_compute_soc/analysis_configs/profiling_counter_grouping_policy.yaml`, `src/utils/metrics/common.py` (`ValuDualIssueDetector`)

**Related code (proposed):** `src/utils/metrics/aggregation.py` (`BOUND_RATIO` / `to_bound_ratio` — follow-up PR)

---

## 0. How to use this document (designers)

This guidance is **hybrid**: normative **MUST** / **MUST NOT** rules for PR review, plus **SHOULD** / **MAY** patterns where rollout scope is still evolving. **§0 is the authoritative rule set**; later sections explain and illustrate — they do not override §0.

| Label | Meaning |
|---|---|
| **MUST** / **MUST NOT** | Required for new or changed metric formulas; PR reviewers **MUST block** violations |
| **SHOULD** | Strong default; deviate only with rationale documented in the PR or a follow-up issue |
| **MAY** | Optional; designer judgment based on §5 validation |

### Terminology

| Term | Meaning in this doc |
|---|---|
| **Correction method** | End-to-end approach: collection (single-pass grouping) or analyze-time formula helper |
| **Clamp / cap** | Analyze-time adjustment that bounds a computed value (e.g. `NOISE_CLAMP` → 0, `BOUND_RATIO` → 100%) |
| **Ratio cap** | Upper-bound correction: `BOUND_RATIO` or `SUM(MIN(a,b))/SUM(b)` only — not `NOISE_CLAMP` |

### Definitions (used in MUST rules and §5.4)

| Term | Definition |
|---|---|
| **Partition metric** | `a` is a subset of `b` by construction (e.g. HBM Read Traffic: `TCC_EA0_RDREQ_DRAM_sum ⊆ TCC_EA0_RDREQ_sum`; CPF Utilization: `busy / (busy + idle)`) |
| **Non-partition pair** | Numerator and denominator are not a strict subset (e.g. Workgroup Manager Utilization: `GRBM_SPI_BUSY / GRBM_GUI_ACTIVE`) |
| **Sporadic violation** | Under **single-pass** validation (§5), ratio or subtraction anomalies affect a **minority** of dispatches, with small excess (often ≈1–5% on avg) or one outlier max on a short dispatch; violations shrink vs multi-pass full panel |
| **Systematic violation** | Under **single-pass** validation on a stable workload, a **majority** of dispatches show the same violation direction (e.g. most rows `a > b` by a similar margin), reproducible across runs |
| **Escalate** | Do **not** merge clamp-only YAML; investigate root cause (counter definition, driver/firmware, hardware) via JIRA or team process; attach single-pass evidence |

### Normative rules (MUST / MUST NOT)

1. **MUST** document the **co-collection strategy** in the PR for any new or changed ratio or subtraction formula (`--block`, `--set`, or grouping-policy entry).
2. **MUST** use `NOISE_CLAMP` (not raw `a − b` alone) in Percent metrics where a subtraction result must be ≥ 0 (e.g. remote traffic, cache splits).  
   *Rationale:* Shipped pattern with analyze-time warnings (≥ 1% relative error). Negative values are clearly invalid; ratio caps (Rules 3–6) stay silent today and require validate-first.
3. **MUST** run §5 **single-pass validation** before merging any PR that adds **`BOUND_RATIO`** or **`SUM(MIN(a,b))/SUM(b)`** (no other ratio cap helpers without updating this doc). **MUST** attach in the PR: profile command, workload, and validation conclusion (sporadic → cap acceptable / systematic → escalate / no cap needed).
4. **MUST NOT** merge **clamp-only** formula changes when single-pass validation shows **systematic violations** (§0 Definitions); **escalate** instead.
5. **Partition metrics only — MUST NOT** apply caps when single-pass data is already ≤ 100% (no change needed). **MUST NOT** cap-only when violations are **systematic** under single-pass.
6. **Non-partition pairs — MUST NOT** use `SUM(MIN(a,b))/SUM(b)` — falsely implies a subset. **MUST NOT** cap the **avg** column with ratio caps; min/max caps only per Rule 3 validation.
7. **MUST NOT** apply `BOUND_RATIO`, `SUM(MIN(a,b))/SUM(b)`, or similar ratio caps to metrics in `ValuDualIssueDetector.candidate_metrics` (`VALU Utilization`, `VALU FLOPs (F64)` on gfx942) — use existing detectors and user-facing warnings.

### Advisory guidance (SHOULD / MAY)

- **SHOULD** use **single-pass collection** when perfmon budget allows (`--block`, `--set`, grouping policy).
- **SHOULD** apply `BOUND_RATIO` (min/max) or `SUM(MIN(a,b))/SUM(b)` (partition avg only) **after** Rule 3 validation confirms **sporadic** noise — same validate-first gate for partition and non-partition metrics.
- **SHOULD** add unit tests in `tests/test_metric_utils.py` (or single-pass workload golden data) when introducing new cap helpers or capped YAML formulas.
- **SHOULD** add grouping-policy entries for high-priority ratio partners on specific arches.
- **MAY** adopt numeric thresholds in §5.4 later if the team agrees — **intentionally qualitative for now**.
- **MAY** add diagnostics when `BOUND_RATIO` or `SUM(MIN)` adjust values (open question §6).
- **MAY** derive public-facing explanations from §7 when product/docs are updated.

### PR review checklist (ratio / split metrics)

- [ ] **Rule 1:** Co-collection approach documented (`--block`, `--set`, or policy)
- [ ] **Rule 2:** Subtraction uses `NOISE_CLAMP` where result must be ≥ 0
- [ ] **Rule 3:** Single-pass validation attached (§5.3 recipe); cap decision stated
- [ ] **Rules 4–6:** Systematic single-pass violations escalated, not clamp-only; no `SUM(MIN)/SUM(b)` on non-partition; no avg cap on non-partition
- [ ] **Rule 7:** No ratio cap on `ValuDualIssueDetector.candidate_metrics`
- [ ] **SHOULD:** Tests or golden workload data for new caps

### Implementation status

§3.2 shows **target patterns** — some are shipped today, others are planned in a follow-up implementation PR. Use this table when reading examples or reviewing PRs against this doc.

| Component | Status | Where / notes |
|---|---|---|
| **`NOISE_CLAMP`** (`to_noise_clamp`) | **Shipped** | `noise_clamper.py`; Remote Read/Write and cache-split metrics (gfx940–950) |
| **Single-pass grouping** (allocator, `--set`, `--block`) | **Shipped** | `soc_base.py`, `profiling_counter_grouping_policy.yaml`; gfx115x has tier-0 entries, many arches (e.g. gfx942) empty |
| **Multi-pass imputation** | **Shipped** | `utils_analysis.py` — stitches counters across passes |
| **`ValuDualIssueDetector`** (VALU > 100% exception) | **Shipped** | `utils/metrics/common.py` — warnings, no clamp |
| **`BOUND_RATIO`** (`to_bound_ratio`) | **Proposed** | Helper + YAML registration — follow-up PR |
| **`SUM(MIN(a,b))/SUM(b)`** on partition avg (e.g. HBM Read Traffic) | **Proposed** | `1700_l2_cache.yaml` — follow-up PR |
| **`BOUND_RATIO` on min/max** (e.g. Workgroup Manager Utilization, CPC Stall) | **Proposed** | Panel YAML updates — follow-up PR |
| **Clamp diagnostics** (`BOUND_RATIO`, `SUM(MIN)`) | **Not started** | Open question §6 — only `NOISE_CLAMP` warns today |

---

## 1. Target Problems, Root Causes, and Correction Methods

### 1.1 Target problems

| Symptom | Typical metric types | User impact |
|---|---|---|
| **Percent avg > 100%** | HBM Read/Write Traffic, CPF Utilization | Aggregate fraction looks physically impossible |
| **Percent min/max > 100%** (often extreme) | Workgroup Manager Utilization, CPC Stall Rate, Data-Return Busy | Max column dominated by one bad dispatch (e.g. 739%) |
| **Negative derived values** | Remote Read/Write Traffic, cache split metrics | Subset subtraction goes negative |
| **Complementary parts do not sum to 100%** | HBM + Remote traffic | After independent corrections, totals may drift |

These symptoms appear across architectures more or less.

### 1.2 Root causes

| Root cause | Mechanism | When it dominates |
|---|---|---|
| **Multi-pass profiling** | Hardware perfmon slot limits require multiple replay passes; analyze **imputes** counters from different passes onto the same dispatch row (`utils_analysis.py`) | Default full analyze (`--block` all panels) |
| **Asynchronous counter sampling** | Counters in different IP blocks are not sample-aligned even within one pass | Short dispatches, low event counts |
| **Non-partition counter pairs** | Numerator and denominator measure different semantics (not strict subset) | Workgroup Manager Utilization: `GRBM_SPI_BUSY / GRBM_GUI_ACTIVE` |
| **Aggregate ratio inflation** | `SUM(a)/SUM(b)` exceeds 100% when some rows have `a > b`, even if most rows are valid | HBM avg on noisy workloads |
| **Potential hardware counter bugs** | Counter definition mismatch, overflow/wrap, incorrect event pairing, driver/firmware sampling defects, or IP-block timing that violates expected subset relationships | Persistent violations under **single-pass** collection on stable workloads; reproducible across runs, drivers, or partition modes |
| **Intentional hardware behavior** | gfx942 VALU dual-issue can exceed 100% utilization | VALU Utilization (documented exception) |

**Key insight:** A single PMC table row does **not** guarantee that all counters on that row were measured simultaneously unless they were collected in **one profiling pass**.

**Noise vs hardware bug:** See §0 Definitions (*sporadic* vs *systematic*). Multi-pass variance and async sampling usually produce sporadic violations. Suspect a **potential hardware counter bug** when violations remain systematic after single-pass validation. Correction methods address sporadic noise; they must not become the only signal for systematic issues.

### 1.3 Correction methods (overview)

| Method | Layer | What it fixes | Primary lever |
|---|---|---|---|
| **Single-pass counter grouping** | Profile (collection) | Prevents cross-pass stitching for related counters | `profiling_counter_grouping_policy.yaml`, `--set`, greedy coalescing in `soc_base.py` |
| **`NOISE_CLAMP`** | Analyze (formula) | Negative values from subtracting counters (`a − b < 0`) | Lower bound → 0, with diagnostics |
| **`BOUND_RATIO`** | Analyze (formula) | Per-row ratio > 100% before min/max aggregation | Upper bound → 100% (silent today) |
| **`SUM(MIN(a,b))/SUM(b)`** | Analyze (formula) | Aggregate avg inflation from `SUM(a)/SUM(b)` | Cap numerator per row before summing (avg columns only) |

Methods compose: **grouping** addresses the cause; **clamp/cap helpers** are downstream guards when multi-pass or noise remains unavoidable. None of the analyze-time helpers **prove** a violation is noise — see §2.5 (masking risk) and §5 (validation per §0 Rule 3).

---

## 2. Method Definitions, Pros, and Cons

### 2.1 Single-pass counter grouping

**Definition:** Place counters that appear together in ratio or subtraction formulas into the same PMC perfmon bucket so they are collected during the same kernel replay pass.

**Implementation today:**

- Greedy coalescing allocator in `soc_base.py` packs counters into `pmc_perf` files.
- `profiling_counter_grouping_policy.yaml` assigns **tier-0 priority** to metric IDs that should coalesce first (`same_bucket_priority_metric_ids`).
- User-facing shortcuts: `--set <name>` (predefined single-pass subsets), reduced `--block` selections.

| Pros | Cons |
|---|---|
| Fixes the problem at the source — row-level counters are co-temporal | Cannot fit all panel counters in one pass on most workloads |
| No analyze-time distortion | Requires ongoing policy maintenance per arch and metric |
| Best accuracy for partition counters (`dram ⊆ total`) | Does not help non-partition pairs (e.g. Workgroup Manager Utilization) |
| Reduces need for silent clamping | Many arches (e.g. gfx942) have empty `same_bucket_priority_metric_ids` today |

**Masking risk (HW bugs):** **Lowest among analyze-time methods.** Single-pass grouping does not alter counter values; violations visible in raw PMC data remain visible in analyze output. This is the preferred way to **validate** whether remaining anomalies are collection artifacts or potential hardware/driver issues.

---

### 2.2 `NOISE_CLAMP` (`to_noise_clamp`)

**Definition:** For a **difference** and its **reference**, clamp negative differences to zero and emit a warning when relative error ≥ 1%.

```python
# src/utils/metrics/noise_clamper.py
result = difference if difference >= 0 else 0.0
```

| Pros | Cons |
|---|---|
| Correct for physically impossible negatives (subset subtraction) | Does **not** fix ratios > 100% |
| Warns when variance is significant (≥ 1% relative) | Only handles **lower** bound, not upper |
| Already deployed for L2 cache split metrics across gfx940–950 | Complementary metrics corrected independently may not sum to exactly 100% |
| Aligns with CHANGELOG guidance on multi-pass variance | Assumes negative = noise; systematic bias could be masked |

**Masking risk (HW bugs):** **Moderate.** A warning fires at ≥ 1% relative error, which helps surface frequent subtraction failures. However, clamping still **replaces** the raw value in the metric output, so a **persistent** negative (bug or mis-defined counter pair) can look like a clean 0% with only a summary warning at end of analyze. Cross-check raw PMC rows under single-pass before trusting corrected Remote/split metrics alone.

**When to use:** Formulas that **subtract** counters where the result must be ≥ 0 (remote traffic, cache residency splits). Required by **§0 Rule 2** (no validate-first gate — shipped helper with warnings).

---

### 2.3 `BOUND_RATIO` (`to_bound_ratio`) *(proposed)*

**Definition:** Compute per-row `(numerator / denominator) × scale`, capping the result at `cap` (default 100). Used before `MIN()` / `MAX()` aggregation on min/max columns.

```python
# src/utils/metrics/aggregation.py (proposed)
return (num / safe_den * scale).clip(upper=cap)
```

| Pros | Cons |
|---|---|
| Fixes min/max column explosions (739% → 100%) | **Silent** today — no diagnostic when capping fires |
| Preserves per-dispatch min/max semantics | Assumes numerator should not exceed denominator — wrong for non-partition pairs if applied blindly to avg |
| Minimal impact when ratios are already ≤ 100% | Does not fix aggregate `SUM/SUM` avg inflation by itself |
| Composable with existing YAML expression engine | Can hide systematic counter bugs if over-applied |

**Masking risk (HW bugs):** **High when silent; moderate if diagnostics added.** A systematic `a > b` bug (e.g. subset counter always over-counts) produces capped 100% min/max with **no trace** in current implementation. Large one-off spikes (739% Workgroup Manager Utilization max) are clearly noise; **repeated** capping on the same counter pair under single-pass is a red flag — escalate to hardware/driver review, not more clamping.

**When to use:** Min/max columns for Percent metrics where `numerator / denominator` should represent a utilization or traffic **share**.

**Validate before cap:** Per **§0 Rules 3–6**. Run §5 single-pass cross-checks first. Apply `BOUND_RATIO` only when violations are **sporadic** (§0 Definitions). If **systematic**, **escalate** — do not cap.

**Non-partition pairs** (e.g. Workgroup Manager Utilization): capping is a display choice, not a physical invariant — min/max only; never avg (§0 Rule 6).

---

### 2.4 Aggregate numerator capping — `SUM(MIN(a,b))/SUM(b)`

**Definition:** For **avg** columns using volume-weighted fractions, cap each row's numerator at the denominator before summing:

```yaml
avg: 100 * SUM(MIN(a, b)) / SUM(b)
```

Mathematical guarantee: `MIN(aᵢ, bᵢ) ≤ bᵢ` for all rows → aggregate ratio ≤ 100%.

| Pros | Cons |
|---|---|
| Preserves volume weighting of `SUM(a)/SUM(b)` | Only valid when `a` is a **subset** of `b` |
| Fixes avg inflation (e.g. 103.23% → 100%) without changing min/max logic | Discards excess `a − b` on inflated rows (slight under-count) |
| Mathematically bounded | Not interchangeable with min/max column formulas |
| Same weighting semantics as uncapped aggregate | Silent — no warning when capping occurs |

**Masking risk (HW bugs):** **High when silent.** Discards `a − b` excess on every inflated row before summing, so a **biased** subset counter that consistently over-reports can still yield a plausible avg ≤ 100%. The under-count is invisible unless raw `a` and `b` are inspected. Always cross-check uncapped `SUM(a)/SUM(b)` under single-pass when validating new counter definitions.

**Validate before cap:** Per **§0 Rules 3–5**. Apply `SUM(MIN(a,b))/SUM(b)` only when single-pass data shows **sporadic** avg inflation. If single-pass raw data is already ≤ 100%, no cap is needed.

**When to use (if validation supports it):** Avg columns for **partition** Percent metrics only (HBM traffic, CPF Utilization `busy/(busy+idle)`).

---

### 2.5 Masking risk summary

How much each method can hide a **potential hardware counter bug** (persistent wrong counts, not one-off multi-pass noise):

| Method | Masking risk | What remains visible | Escalation signal |
|---|---|---|---|
| **Single-pass grouping** | Low | Raw ratios in analyze; same formula, cleaner inputs | Violations still > 100% or negative after single-pass |
| **`NOISE_CLAMP`** | Moderate | Summary warning count; raw PMC if inspected manually | Warnings on most dispatches; large rel. error under single-pass |
| **`BOUND_RATIO`** | High (today) | Nothing in metric output unless raw PMC reviewed | Same metric capped on many dispatches under single-pass |
| **`SUM(MIN(a,b))/SUM(b)`** | High (today) | Uncapped aggregate only if formula run separately | Avg always 100% while raw `SUM(a)/SUM(b)` still > 100% under single-pass |

**Principle:** Analyze-time corrections improve **user-facing plausibility** for known collection noise. They are **not** a substitute for validating counter correctness under stable collection conditions (§5).

---

## 3. Examples

### 3.1 Abstract examples

Assume three dispatches. All formulas use Percent scale (× 100).

#### Problem A — aggregate avg inflation (`SUM(a)/SUM(b)`)

| Dispatch | `a` | `b` |
|---:|---:|---:|
| 1 | 110 | 100 |
| 2 | 100 | 100 |
| 3 | 100 | 100 |

| Method | Formula | Result |
|---|---|---:|
| Raw | `SUM(a)/SUM(b) × 100` | `(110+100+100)/300 × 100` = **103.33%** |
| Aggregate cap | `SUM(MIN(a,b))/SUM(b) × 100` | `(100+100+100)/300 × 100` = **100.00%** |

#### Problem B — min/max column explosion

| Dispatch | `a` | `b` | `a/b` |
|---:|---:|---:|---:|
| 1 | 50 | 100 | 50% |
| 2 | 120 | 100 | 120% |

| Method | Formula | Result |
|---|---|---:|
| Raw max | `MAX(a/b) × 100` | **120%** |
| `BOUND_RATIO` max | `MAX(BOUND_RATIO(a,b))` | `MAX(50%, 100%)` = **100%** |
| Wrong tool: aggregate cap | `SUM(MIN(a,b))/SUM(b) × 100` | `(50+100)/200 × 100` = **75%** ← not a max |

#### Problem C — negative subtraction (remote traffic)

| Dispatch | `total` | `part` | `total − part` |
|---:|---:|---:|---:|
| 1 | 100 | 110 | **−10** |

| Method | Formula | Result |
|---|---|---:|
| Raw | `(total − part) / total × 100` | **−10%** |
| `NOISE_CLAMP` | `NOISE_CLAMP(total − part, total) / total × 100` | **0%** (+ warning if ≥ 1% rel. error) |

#### Problem D — single-pass ideal (partition counters)

If `a` and `b` are co-collected in one pass and `a ⊆ b` by construction:

| Dispatch | `a` | `b` | `a/b` |
|---:|---:|---:|---:|
| 1 | 95 | 100 | 95% |
| 2 | 88 | 100 | 88% |

All methods agree; clamping is a no-op. Residual violations (> 100%) should be rare and small. **If violations are large or frequent here, treat as potential HW counter bug** — not a clamping candidate.

#### Problem E — distinguishing noise from potential HW bug

**Scenario A — multi-pass full panel** (partition counters):

| Dispatch | `a` | `b` | Notes |
|---:|---:|---:|---|
| 1 | 110 | 100 | One inflated row among many |
| 2 | 98 | 100 | Normal |
| 3 | 99 | 100 | Normal |

| Interpretation | Action |
|---|---|
| Sporadic under multi-pass | Often shrinks with single-pass re-profile (§5.3); then apply §0 Rules 3–5 |

**Scenario B — single-pass** collection, stable workload (same counters):

| Dispatch | `a` | `b` | `a/b` |
|---:|---:|---:|---:|
| 1 | 115 | 100 | 115% |
| 2 | 112 | 100 | 112% |
| 3 | 118 | 100 | 118% |

| Interpretation | Action |
|---|---|
| **Systematic** under single-pass (§0 Definitions) | **Escalate** — do not cap-only; investigate counter/driver/hardware |
| After `BOUND_RATIO` / `SUM(MIN)` without validation | User sees ≤ 100%; **bug signal is hidden** unless raw PMC is checked |

---

### 3.2 rocprof-compute metric examples

Examples below include **shipped** and **proposed** patterns — see **Implementation status** in §0.

#### Single-pass grouping *(shipped infrastructure; arch policy varies)*

**Goal:** Collect `TCC_EA0_RDREQ_DRAM_sum` and `TCC_EA0_RDREQ_sum` in the same pass.

| Approach | Example |
|---|---|
| Policy (gfx115x pattern) | Add metric id to `same_bucket_priority_metric_ids` in `profiling_counter_grouping_policy.yaml` |
| User shortcut | `rocprof-compute profile --set <l2_subset> ...` when the set includes both counters |
| Today on gfx942 | Policy map is `{}` — no arch-specific coalescing priorities for HBM traffic metrics |

**Real metric:** *HBM Read Traffic* (panel 1700, gfx942 `1700_l2_cache.yaml`)

---

#### `NOISE_CLAMP` *(shipped)*

**Real metric:** *Remote Read Traffic* (gfx942)

```yaml
avg: 100 * SUM(NOISE_CLAMP(TCC_EA0_RDREQ_sum - TCC_EA0_RDREQ_DRAM_sum, TCC_EA0_RDREQ_sum)) / SUM(TCC_EA0_RDREQ_sum)
```

When `TCC_EA0_RDREQ_DRAM_sum > TCC_EA0_RDREQ_sum` on a dispatch (multi-pass variance), remote goes negative → clamped to 0 with optional warning.

**Also used in:** Uncached/cached L2 splits, TCP hit-rate residuals (`1600_vector_l1_data_cache.yaml`), gfx950 memory bandwidth formulas.

---

#### `BOUND_RATIO` *(proposed — apply only after single-pass validation)*

**Real metric:** *Workgroup Manager Utilization* min/max (gfx942 `0600_workgroup_manager_spi.yaml`)

**Validation-first workflow:** Profile with minimal blocks so `GRBM_SPI_BUSY` and `GRBM_GUI_ACTIVE` are co-collected (§5.3). Inspect per-dispatch ratios in raw PMC. Apply cap in YAML only if violations are sporadic (e.g. one short-dispatch max spike), not systematic across dispatches.

```yaml
min: 100 * MIN(BOUND_RATIO($GRBM_SPI_BUSY_PER_XCD, $GRBM_GUI_ACTIVE_PER_XCD))
max: 100 * MAX(BOUND_RATIO($GRBM_SPI_BUSY_PER_XCD, $GRBM_GUI_ACTIVE_PER_XCD))
```

Reproduction example (mat_exp workload, multi-pass analyze): raw max **739.60%** → capped **100%** if cap is applied. **Decision to cap pending single-pass validation.**

**Also proposed for (after validation):** CPC Stall Rate, CPC Packet Decoding, Data-Return Busy, HBM Read/Write min/max.

---

#### `SUM(MIN(a,b))/SUM(b)` *(proposed — apply only after single-pass validation)*

**Real metric:** *HBM Read Traffic* avg (gfx942)

**Validation-first workflow:** Profile block 17.1 (or `--set`) so `TCC_EA0_RDREQ_DRAM_sum` and `TCC_EA0_RDREQ_sum` are co-collected (§5.3). Compare uncapped `SUM(a)/SUM(b)` vs per-dispatch ratios in raw PMC. Apply `SUM(MIN)/SUM(b)` in YAML only if single-pass avg still exceeds 100% due to sporadic row inflation — not if violations are systematic.

```yaml
avg: 100 * SUM(MIN(TCC_EA0_RDREQ_DRAM_sum, TCC_EA0_RDREQ_sum)) / SUM(TCC_EA0_RDREQ_sum)
```

Reproduction example (occupancy workload, multi-pass analyze): raw avg **103.23%** → capped **100.00%** if cap is applied. **Decision to cap pending single-pass validation.**

**Note:** Min/max for the same metric use `BOUND_RATIO` (also after validation) — different column semantics (see §3.1 Problem B).

---

#### Intentional exception — no clamp *(shipped)*

**Real metric:** *VALU Utilization* (gfx942)

Dual-issue on gfx942 can legitimately report > 100% (up to ~200%). Listed in `ValuDualIssueDetector.candidate_metrics` — **§0 Rule 7** forbids ratio caps; use existing warnings.

---

## 4. Practical Mental Model

**Decision flow:** Follow the diagram below; all cap decisions defer to **§0** (Rules 3–7) and **§5** validation.

```mermaid
flowchart TD
    A["Percent metric formula"] --> B{"Are numerator and denominator<br/>from the same pass?"}
    B -->|No — multi-pass default| C["Imputation stitches counters<br/>onto same dispatch row"]
    C --> D["a > b or (total − part) < 0 possible"]
    D --> E{"Formula shape?"}
    E -->|Subtraction| F["NOISE_CLAMP<br/>(lower bound; shipped)"]
    E -->|Ratio min/max| G["Consider BOUND_RATIO<br/>after §5 validation"]
    E -->|Ratio avg, partition| H["Consider SUM(MIN)/SUM(b)<br/>after §5 validation"]
    B -->|Yes — single pass or --set| I{"Is a a subset of b?"}
    I -->|Yes — e.g. dram ⊆ total| J["Single-pass validate"]
    J --> J2{"Violations sporadic?"}
    J2 -->|Yes| K["Consider cap if needed<br/>(SUM(MIN) or BOUND_RATIO)"]
    J2 -->|No — systematic| P["Potential HW counter bug<br/>— escalate, do not clamp-only"]
    I -->|No — e.g. Workgroup Manager Utilization| L["Single-pass validate"]
    L --> L2{"Violations sporadic?"}
    L2 -->|Yes| M["Consider BOUND_RATIO on min/max"]
    L2 -->|No — systematic| P
    A --> N{"VALU dual-issue?"}
    N -->|Yes| O["§0 Rule 7<br/>— do not ratio-cap"]
```

**PR review:** Use the **§0 PR review checklist** — do not duplicate rules here.

---

## 5. Validation and Cross-Checking Guidance

Use this workflow when authoring a new corrected metric, reviewing a clamping PR, or investigating whether anomalies indicate **collection noise** vs **potential hardware counter bugs**.

### 5.1 Stable collection conditions

Reduce confounding factors before comparing raw vs corrected values:

| Control | Purpose |
|---|---|
| **Single-pass collection** | Profile only the blocks/counters needed for the metric (`--block`, `--set`, or minimal panel subset) so ratio partners are co-temporal |
| **Counter grouping policy** | Ensure formula counters land in the same PMC bucket (`profiling_counter_grouping_policy.yaml` + coalescing in `soc_base.py`) |
| **Stable synthetic workload** | Prefer repeatable kernels (e.g. occupancy, rocflop, mat_exp) with sufficient dispatch count — avoid tiny or one-off dispatches that dominate min/max |
| **Fixed system config** | Same GPU, driver/ROCm version, partition mode (CPX/SPX), and clock state across A/B runs |
| **Multiple runs** | Confirm violations are sporadic (noise) vs reproducible (investigate further) |

### 5.2 Cross-checks to run

| Check | How | Pass criterion | Fail → investigate |
|---|---|---|---|
| **Raw vs corrected** | Compare uncapped formula output to clamped output on same workload | Small delta on avg; max capped only on outlier dispatches | Large or systematic delta on avg under single-pass |
| **Single-pass vs multi-pass** | Re-profile same workload with minimal blocks vs full panel | Violations shrink or disappear with single-pass | Same violation magnitude single-pass → not imputation alone |
| **Complementary metrics** | HBM + Remote (+ other splits) from same counter family | Splits approximately sum to 100% (exact sum not required after independent clamp) | Systematic drift; one side always clamped |
| **Partition invariant** | For `a ⊆ b`, inspect per-dispatch `a/b` in raw PMC CSV | ≤ 100% on nearly all rows, small excess on few rows | Majority of rows > 100% under single-pass |
| **Warning counts** | Review `NOISE_CLAMP` summary at end of analyze | Few warnings, low max rel. error | Warnings on most dispatches or high max rel. error |
| **Cross-arch / cross-mode** | Same workload on another arch or partition mode if available | Qualitatively similar (some variance expected) | Isolated to one arch/mode → arch-specific counter or driver issue |

### 5.3 Recommended validation profile recipe

Example for validating *HBM Read Traffic* counter pair on gfx942 (adjust kernel name and paths for your environment; confirm flags via `rocprof-compute profile --help`):

```bash
# 1. Profile L2 block only (single-pass friendly subset)
rocprof-compute profile --block 17.1 --kernel-name <kernel> --output-directory workloads/validate_hbm/

# 2. Analyze (variance warnings enabled by default at kernel level)
rocprof-compute analyze --block 17.1 --path workloads/validate_hbm/

# 3. Inspect raw PMC: per-dispatch TCC_EA0_RDREQ_DRAM_sum vs TCC_EA0_RDREQ_sum
# 4. Compare uncapped SUM(a)/SUM(b) vs SUM(MIN(a,b))/SUM(b) on same data
# 5. Record conclusion in PR per §0 Rule 3
```

Repeat with `--set` when a predefined set already co-packages the needed counters.

### 5.4 When to clamp vs when to escalate

**Threshold policy:** Criteria use **§0 Definitions** (*sporadic* / *systematic* / *escalate*). Qualitative by design — not fixed numeric gates or CI enforcement until the team agrees after collecting single-pass data.

| Observation under single-pass + stable workload | Likely cause | Action |
|---|---|---|
| **Sporadic** ratio > 100% (partition or non-partition) | Short dispatch, async sampling, or residual multi-pass noise | **SHOULD** consider cap after Rule 3 validation (`SUM(MIN)` partition avg; `BOUND_RATIO` min/max) |
| **Systematic** partition metric > 100% | Potential HW counter bug or wrong event pairing | **MUST NOT** cap-only (**§0 Rule 4–5**); **escalate** |
| **Systematic** non-partition metric > 100% | Semantics or denominator collapse | Same gate; **SHOULD** cap min/max only if reclassified as sporadic after investigation |
| Single-dispatch max spike (e.g. >> 200%) with tiny denominator | Denominator collapse / short dispatch | After validation confirms **sporadic** spike: **SHOULD** `BOUND_RATIO` on max; document in PR |
| Most dispatches > 100% by similar margin | Potential HW counter bug | **Escalate** (**§0 Rule 4**) |
| Persistent negative subtraction | Bug or mis-defined subset | Fix formula or counter definition; `NOISE_CLAMP` is display-only |
| `ValuDualIssueDetector.candidate_metrics` | Intentional dual-issue | **§0 Rule 7** — warn user; no ratio cap |

---

## 6. Open Questions

1. **CDNA grouping policy:** Should `profiling_counter_grouping_policy.yaml` gain entries (e.g. gfx942) coalescing HBM, GRBM, and CPC counter pairs? What is the perfmon budget trade-off?

2. **Silent vs diagnostic capping:** Should `BOUND_RATIO` and `SUM(MIN)/SUM(b)` adopt `NOISE_CLAMP`-style warnings when values are adjusted (threshold, counter, summary at end of analyze)? This is critical for not masking potential HW counter bugs.

3. **Scope of `BOUND_RATIO` rollout:** Which Percent metrics across gfx940/942/950 (and other arches) should be updated, and in what priority order?

4. **Avg column consistency:** **§0 Rule 6** forbids `SUM(MIN)/SUM(b)` and avg ratio caps on non-partition pairs — confirm no exceptions needed.

5. **Complementary metric consistency:** After independent clamping on HBM (`SUM(MIN)` / `BOUND_RATIO`) and Remote (`NOISE_CLAMP`), should we enforce or document that splits may not sum to exactly 100%?

6. **Single-pass validation metadata:** Can we add analyze-time metadata (pass-id per counter group) to detect when ratio partners were not co-collected and warn before clamping?

7. **Source of truth:** Analysis YAML files are autogenerated — see the `# AUTOGENERATED FILE` header comments in `src/rocprof_compute_soc/analysis_configs/gfx*/*.yaml`. Those headers cite `utils/unified_config.yaml` / `split_config.py`, which are **not in this repository** (likely legacy boilerplate). Formulas are validated via `tools/config_management/` (see `CONTRIBUTING.md`). Should correction patterns be documented/enforced there instead?

8. **Acceptance criteria for Percent metrics:** Should product/docs list `ValuDualIssueDetector.candidate_metrics` and other exceptions rather than requiring all Percent ≤ 100%?

9. **Imputation alternatives:** Is forward/backward fill imputation the best merge strategy, or should ratio metrics skip rows where partners came from different passes?

10. **Testing strategy:** What reference workloads and thresholds should gate regression tests for clamping (before/after bounds, warning counts)?

11. **HW counter bug triage:** What process links repeated single-pass violations to driver/firmware/hardware teams vs rocprof-compute formula fixes?

12. **Automated validation:** Should CI include single-pass golden workloads that assert raw (uncapped) partition invariants before corrected metrics ship?

---

## 7. Public-facing notes (draft)

*Secondary audience — port to FAQ, conceptual docs, or analyze warnings when ready. Not user documentation yet.*

- **Why can some Percent metrics exceed 100%?** Multi-pass profiling merges counters from different replay passes; occasional misalignment can inflate ratios. When caps are applied in product, they follow internal single-pass validation (§0 Rule 3) — display values reflect known collection noise, not a substitute for counter correctness review.
- **Why is VALU Utilization sometimes > 100%?** On some GPUs (e.g. gfx942), dual-issue execution can exceed 100% — this is expected (`ValuDualIssueDetector.candidate_metrics`), not a bug; ratio caps are not applied.
- **What do “counter variance corrected” warnings mean?** A subtraction-based metric had a negative intermediate value (usually multi-pass noise) and was clamped to zero.
- **Do HBM + Remote always sum to 100%?** Not exactly after independent corrections; each split is sanitized separately.

---

## References

- **Motivating example:** Percent metrics > 100% on MI300X CPX (one reproduction case among several arches)
- `CHANGELOG.md` — Known issues: negative values and multi-pass variance (v3.3.x)
- `src/utils/utils_analysis.py` — Multi-pass data imputation
- `src/rocprof_compute_soc/analysis_configs/profiling_counter_grouping_policy.yaml` — Coalescing priorities
- `src/utils/metrics/noise_clamper.py` — `NOISE_CLAMP` implementation
- `src/utils/metrics/aggregation.py` — `BOUND_RATIO` (**proposed** in follow-up PR)

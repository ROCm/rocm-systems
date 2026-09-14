# Single-Pass Counter Grouping — Evaluation and Cost

**Status:** Draft  
**Audience:** Designers deciding collection vs analyze-time fixes  
**JIRA:** AIPROFCOMP-78  
**Related:** [Problem statement](metric-counter-correction-problem-statement.md) · [Correction methods](metric-counter-correction-methods.md) · [Validation data](../reports/aiprofcomp78-multipass-root-cause-report.md)

---

## 1. What single-pass grouping is

**Goal:** Place counters that appear together in a ratio or subtraction formula into the **same PMC perfmon bucket** so they are collected during the **same kernel re-run**.

**Levers today:**

| Lever | Location | User-facing |
|-------|----------|-------------|
| Greedy allocator | `soc_base.py` → `_allocate_perfmon_counter_files` | Automatic |
| Grouping policy | `profiling_counter_grouping_policy.yaml` → `same_bucket_priority_metric_ids` | Maintainer config |
| Metric-aware coalesce | `soc_base.py` → `_metric_aware_coalesce_pass` | Tier-0 metrics packed first |
| Reduced scope | `--block`, `--set` | User selects fewer panels |

**Shipped state for gfx942:** `same_bucket_priority_metric_ids: {}` (empty). gfx115x and gfx1250 have non-empty policies.

---

## 2. Why grouping is the most accurate fix (when it works)

When numerator and denominator are co-collected on the same dispatch:

- `SUM(a)/SUM(b)` is the correct volume-weighted fraction for partition metrics.
- No analyze-time cap substitutes a different estimator.
- Violations that remain are candidates for HW/driver escalation, not silent clamping.

**Limitation:** The allocator gives **accumulator** counters (`*_ACCUM`) dedicated perfmon files and **skips** `_ACCUM` buckets during metric-aware coalesce (`soc_base.py` lines 425–426). Many ratio partners are tied to level/accumulator passes, so co-location depends on whether the metric-aware pass can pack the whole metric group into one **non-accum** bucket or a shared accum slot.

---

## 3. Measured profile cost (full-panel CPX workloads)

From perfmon YAML count on AIPROFCOMP-78 CPX profiles:

| Workload | GPU | Partition | Perfmon passes | Notes |
|----------|-----|-----------|----------------|-------|
| occupancy | MI300X gfx942 | CPX | **13** | Full panel |
| mat_exp | MI300X gfx942 | CPX | **13** | Full panel |
| rocflop | MI300X gfx942 | CPX | **13** | Full panel |

Each pass re-runs the workload kernel(s). Total profile time scales roughly linearly with pass count (plus fixed startup).

---

## 4. Allocator simulation (gfx942, MI300)

Simulated with `OmniSoC_Base.detect_counters()` + `_allocate_perfmon_counter_files()` against shipped gfx942 YAML (Aug 2026). Pass counts are **allocator estimates**; measured CPX runs may differ by +1 when level accumulators split (13 measured vs 12 simulated baseline).

### 4.1 Pass count by profile scope

| Scenario | PMC passes | Accumulator files | Counter count |
|----------|------------|-------------------|---------------|
| Full panel (typical blocks) | **12** | 5 | 439 |
| Block 17 (L2 only) | **8** | 0 | 34 |
| Blocks 5+6 (CPC/SPI) | **5** | 0 | 39 |
| Block 15 (TA/TD data-return) | **8** | 0 | 28 |
| Blocks 5+6+15+17 (all cap-affected panels) | **8** | 0 | 97 |

**Validation recipe cost:** `--block 17` reduces passes from ~12–13 to **8** (~35–40% fewer re-runs vs full panel) while keeping L2/HBM counters.

### 4.2 Grouping policy typically increases passes

On gfx942 full panel, adding `same_bucket_priority_metric_ids` **typically increases** pass count. Metric-aware coalesce may open **new buckets** when priority metric groups cannot fit existing ones — co-locating ratio partners trades off against **more kernel re-runs**.

| Policy | Full-panel passes | Δ vs empty policy |
|--------|-------------------|---------------------|
| `{}` (today) | 12 | — |
| `17.2.1` (HBM Read Traffic) | **17** | **+5 (+42%)** |
| `6.1.2` (Workgroup Manager Utilization) | 17 | +5 |
| All cap metrics (`17.2.1`, `6.1.2`, `5.1.0`, `15.4.0`) | 17 | +5 |

Do **not** expect grouping policy to reduce passes on full-panel gfx942 profiles. gfx1250 is a documented exception where priority metrics can steer packing **without** increasing pass count — behavior is **arch- and counter-set-specific**.

### 4.3 Does grouping policy co-locate ratio partners?

Simulated counter → pass bucket (full panel):

| Counter | No policy | With `17.2.1` priority | With `6.1.2` priority |
|---------|-----------|------------------------|------------------------|
| `TCC_EA0_RDREQ_DRAM_sum` | `SQ_INST_LEVEL_VMEM_ACCUM` | **`0` (same as partner)** | `5` |
| `TCC_EA0_RDREQ_sum` | `SQ_LEVEL_WAVES_ACCUM` | **`0`** | `4` |
| `GRBM_SPI_BUSY` | `SQ_INST_LEVEL_LDS_ACCUM` | `0` | **`0` (with partner)** |
| `GRBM_GUI_ACTIVE` | `SQ_IFETCH_LEVEL_ACCUM` | `0` | **`0`** |
| `CPF_CPF_STAT_BUSY` | `SQ_IFETCH_LEVEL_ACCUM` | `0` | `0` |
| `CPF_CPF_STAT_IDLE` | `SQ_INST_LEVEL_LDS_ACCUM` | `0` | `0` |

**Without policy:** every affected pair is split across passes (matches mat_exp CPX perfmon files).  
**With targeted priority:** partners can co-locate (e.g. HBM both in bucket `0`), at the cost of **more total passes**.

**Block 17 only:** `TCC_EA0_RDREQ_DRAM_sum` and `TCC_EA0_RDREQ_sum` already share a pass **with or without** policy (8 passes). Block-only validation does not need policy entries for HBM co-location.

---

## 5. Evaluation summary

| Question | Answer |
|----------|--------|
| Is grouping the most accurate fix? | **Yes**, when counters truly share a pass on the same dispatch |
| Does empty gfx942 policy cause splitting? | **Yes** — partners land in different accum passes under default packing |
| Does adding policy increase passes? | **Typically yes** on gfx942 full panel — simulated **+42%** for priority metrics |
| Does `--block 17` help HBM validation? | **Yes** — partners already co-locate; **8 passes** vs ~12–13 |
| Does policy help block-only HBM? | **No extra benefit** in simulation (already co-located) |
| Can policy fix Workgroup Manager Utilization on full panel? | **Simulated co-location** with `6.1.2`, same pass-count penalty |

---

## 6. Recommendations

1. **Validation (Rule 3):** Use `--block` subsets (e.g. 17, 5+6, 15) before full-panel caps — lower pass count than full panel, partners often co-locate without policy changes.
2. **Grouping policy for gfx942:** Do **not** assume zero cost. Any PR adding `same_bucket_priority_metric_ids` must report **simulated and measured pass count** before/after (use allocator simulation + one profile run).
3. **Full-panel production:** Grouping policy is a **trade-off** — better ratio accuracy vs longer profile time. Consider user `--set` presets instead of default full-panel policy for all users.
4. **Analyze-time caps:** Remain appropriate when users run default multi-pass full panel and pass-count increase is unacceptable.

---

## 7. How to reproduce pass-count simulation

From `projects/rocprofiler-compute/`:

```bash
PYTHONPATH=src python3 - <<'PY'
# See commit aiprofcomp78 docs or soc_base unit tests for full script.
# Uses OmniSoC_Base.detect_counters + _allocate_perfmon_counter_files.
PY
```

Reference implementation: `tests/unit/rocprof_compute_soc/test_soc_base.py` (allocator tests).

---

## 8. Open follow-ups

1. **Measure** pass count and profile wall time on hardware after adding gfx942 policy entries (confirm simulation).
2. **Single-pass re-profile** mat_exp / occupancy with `--block 17` and compare raw `a/b` distributions to merged full-panel data.
3. **Investigate** whether pass count increase can be mitigated (allocator tuning vs selective priority metrics only).

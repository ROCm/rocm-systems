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
| Reduced scope | `--block` (panel or **metric id**), `--set` | User selects fewer panels/metrics |

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

Simulated with `tools/counter_grouping_inspector.py` (§7) against shipped gfx942 YAML. All pass counts below were **run through the inspector** (Aug 2026); measured CPX hardware profiles match the full-panel bucket count (**13**).

### 4.1 Pass count by profile scope

| Scenario | PMC passes | Accumulator YAMLs | Counter assignments |
|----------|------------|-------------------|---------------------|
| Full panel (typical blocks) | **13** | **7** | 274 |
| Block 17 (L2 only) | **8** | 0 | 34 |
| Blocks 5+6 (CPC/SPI) | **5** | 0 | 39 |
| Block 15 (TA/TD data-return) | **8** | 0 | 28 |
| Blocks 5+6+15+17 (all cap-affected panels) | **8** | 0 | 97 |
| Metric id `6.1.2` (Workgroup Manager Utilization only) | **1** | 0 | 3 |
| Metric ids `5.1.0`, `15.4.0`, `17.2.1` (each alone) | **1** | 0 | 3–4 |
| Metric ids `6.1.2` + `5.1.0` + `15.4.0` (combined) | **1** | 0 | 6 |

**Validation recipe cost:** Panel `--block 17` reduces passes from **13** to **8** (~38% fewer re-runs vs full panel) and co-locates HBM partners. **Metric-id `--block`** (e.g. `6.1.2`) co-locates a single ratio metric in **1 pass** — preferred for §3.5 hardware validation on **rocprof-compute 3.8.0+**.

### 4.2 Grouping policy typically increases passes

On gfx942 full panel, adding `same_bucket_priority_metric_ids` **increases** pass count in inspector runs. Metric-aware coalesce opens **new buckets** when priority metric groups cannot fit existing ones — co-locating ratio partners trades off against **more kernel re-runs**.

| Policy (`same_bucket_priority_metric_ids`) | Full-panel passes | Δ vs `{}` | Workgroup Manager Utilization | HBM Read Traffic |
|--------------------------------------------|-------------------|-----------|------------------------------|------------------|
| `{}` (shipped) | **13** | — | 2 buckets (split) | 2 buckets (split) |
| `6.1.2` | **19** | **+6 (+46%)** | **1 bucket** (`0`) | still 2 buckets |
| `17.2.1` | **19** | **+6** | 1 bucket (`0`) | **1 bucket** (`0`) |
| All cap metrics (`17.2.1`, `6.1.2`, `5.1.0`, `15.4.0`) | **19** | **+6** | 1 bucket (`0`) | **1 bucket** (`0`) |

**Flat cost:** Adding **any one** priority metric alone also yields **19 passes** (+6) — not cumulative per metric. Inspector re-run Aug 2026.

Reproduce: temporarily edit `profiling_counter_grouping_policy.yaml` (gfx942 section), run `python3 tools/counter_grouping_inspector.py --arch gfx942`, restore file. **Hardware:** empty policy **13 passes confirmed** on darkstar; policy **+6 passes** requires **rocprof-compute 3.10+** (ignored on bundled 3.8.0 — perfmon byte-identical) — see [hardware runbook](../reports/aiprofcomp78-hardware-validation-runbook.md).

Do **not** expect grouping policy to reduce passes on full-panel gfx942 profiles. gfx1250 is a documented exception where priority metrics can steer packing **without** increasing pass count — behavior is **arch- and counter-set-specific**.

### 4.3 Does grouping policy co-locate ratio partners?

Counter → bucket layout from `tools/counter_grouping_inspector.py` (full panel — §7):

| Counter | Bucket (`{}`) | With `17.2.1` | With `6.1.2` |
|---------|---------------|---------------|--------------|
| `TCC_EA0_RDREQ_DRAM_sum` | `SQ_INST_LEVEL_LDS_ACCUM`* | **`0`** | `4` |
| `TCC_EA0_RDREQ_sum` | `SQ_INST_LEVEL_SMEM_ACCUM`* | **`0`** | `5` |
| `GRBM_SPI_BUSY` | `SQC_ICACHE_INFLIGHT_LEVEL_ACCUM` | `0` | **`0`** |
| `GRBM_GUI_ACTIVE` | `SQC_DCACHE_INFLIGHT_LEVEL_ACCUM` | `0` | **`0`** |
| `CPF_CPF_STAT_BUSY` | (multi-bucket metric) | `0` | `0` |
| `CPF_CPF_STAT_IDLE` | (multi-bucket metric) | `0` | `0` |

\*HBM Read Traffic multi-bucket row under `{}`: `SQ_INST_LEVEL_LDS_ACCUM`, `SQ_INST_LEVEL_SMEM_ACCUM`.

**Without policy:** every affected pair is split across passes (matches mat_exp CPX perfmon files and the inspector multi-bucket metrics table).  
**With targeted priority:** partners co-locate (e.g. HBM in bucket `0` with `17.2.1`), at **+6 passes** on full panel.

**Block 17 only:** `TCC_EA0_RDREQ_DRAM_sum` and `TCC_EA0_RDREQ_sum` already share a pass **with or without** policy (8 passes). Block-only validation does not need policy entries for HBM co-location.

#### WGM example (Workgroup Manager Utilization)

Metric **`6.1.2`** — `100 × GRBM_SPI_BUSY / GRBM_GUI_ACTIVE` (non-partition; min/max caps only). On **mat_exp CPX** (gfx942, 13 perfmon passes, full panel):

**Measured pass split** (from shipped perfmon YAML on disk — no counter appears in both passes):

| Role | Counter | Perfmon pass file |
|------|---------|-------------------|
| Numerator | `GRBM_SPI_BUSY` | `pmc_perf_SQC_ICACHE_INFLIGHT_LEVEL_ACCUM.yaml` |
| Denominator | `GRBM_GUI_ACTIVE` | `pmc_perf_SQC_DCACHE_INFLIGHT_LEVEL_ACCUM.yaml` |

**Merged analyze output** (`pmc_perf.csv` stitched across passes):

| Dispatches | Rows with `a > b` | % bad | Max `100×a/b` | `100×SUM(a)/SUM(b)` |
|------------|-------------------|-------|---------------|---------------------|
| 206 | 2 | 1.0% | **739.6%** | 40.06% |

Analyze joins numerator and denominator by `Dispatch_ID`, so two mismatched passes can produce extreme per-dispatch ratios even though each counter is valid in its own pass.

**Inspector output** (full panel, gfx942, empty policy — `python3 tools/counter_grouping_inspector.py --arch gfx942`):

| Counter | Bucket |
|---------|--------|
| `GRBM_SPI_BUSY` | `SQC_ICACHE_INFLIGHT_LEVEL_ACCUM` |
| `GRBM_GUI_ACTIVE` | `SQC_DCACHE_INFLIGHT_LEVEL_ACCUM` |

Multi-bucket metrics row: `Workgroup Manager Utilization` → **2 buckets** (`SQC_DCACHE_INFLIGHT_LEVEL_ACCUM`, `SQC_ICACHE_INFLIGHT_LEVEL_ACCUM`). `Summary: 13 bucket(s)`.

**With `6.1.2` priority** (inspector with edited policy YAML): partners co-locate in bucket `0`; full panel **13 → 19 passes** (+6).

**Takeaway:** Empty gfx942 policy leaves Workgroup Manager Utilization partners in different passes → merged max **739.6%** on mat_exp. Priority metric `6.1.2` co-locates partners in the inspector at **+6 passes** on full panel. **Hardware validation (darkstar):** metric-id **`--block 6.1.2`** co-locates in **1 pass**, **0/208** `a > b`, max **100.0%** on mat_exp (**rocprof-compute 3.8.0**). Panel `--block 6` still splits partners (structural check only).

---

## 5. Evaluation summary

| Question | Answer |
|----------|--------|
| Is grouping the most accurate fix? | **Yes**, when counters truly share a pass on the same dispatch |
| Root cause of §3.1 bound violations (validated workloads)? | **Multi-pass stitching** — violations disappear under co-located collection |
| Does empty gfx942 policy cause splitting? | **Yes** — partners land in different accum passes under default packing |
| Does adding policy increase passes? | **Yes** on gfx942 full panel — inspector **+6 passes** (13 → 19) for any tested priority entry |
| Does `--block 17` help HBM validation? | **Yes** — partners already co-locate; **8 passes** vs **13** full panel |
| Does metric-id `--block` co-locate without policy? | **Yes** — **1 pass** per metric (`6.1.2`, `5.1.0`, `15.4.0`, `17.2.1`); hardware-validated on 3.8.0 |
| Does policy help block-only HBM? | **No extra benefit** in simulation (already co-located) |
| Can policy fix all cap metrics on full panel? | **Simulated co-location** with all four ids; **+6 passes** flat (requires **rocprof-compute 3.10+** on hardware) |

---

## 6. Recommendations

1. **Validation (Rule 3):** Prefer **metric-id `--block`** (e.g. `6.1.2`, `5.1.0`, `15.4.0`) or `--set aiprofcomp78_bounds` for co-located single-pass checks on **rocprof-compute 3.8.0+**. Use panel `--block 17` for HBM (8 passes). Panel `--block 5/6/15` confirms allocator splits but does not co-locate partners.
2. **Grouping policy for gfx942:** Do **not** assume zero cost. Any PR adding `same_bucket_priority_metric_ids` must report **inspector pass count** before/after (§7) and confirm on **rocprof-compute 3.10+** hardware (+6 passes, 13 → 19, flat for one or all four cap metrics).
3. **Full-panel production:** Grouping policy is a **trade-off** — better ratio accuracy vs **+46%** profile time. Users who need specific metrics can use metric-id `--block` or `--set` presets without full-panel policy.
4. **Analyze-time caps:** Remain appropriate when users run default multi-pass full panel and the +6-pass policy cost is unacceptable.

---

## 7. How to reproduce pass-count simulation

Use **`tools/counter_grouping_inspector.py`** — offline developer tool that runs the same SoC path as profiling (`detect_counters` → `perfmon_coalesce` → bucket allocation) without a GPU or rocprofiler.

From `projects/rocprofiler-compute/`:

```bash
# Full panel — pass count in Summary; split metrics listed at end
python3 tools/counter_grouping_inspector.py --arch gfx942

# Scoped profiles (--block accepts panel or metric IDs from analyze)
python3 tools/counter_grouping_inspector.py --arch gfx942 --block 17
python3 tools/counter_grouping_inspector.py --arch gfx942 --block 6.1.2
python3 tools/counter_grouping_inspector.py --arch gfx942 --block 5.1.0 15.4.0
python3 tools/counter_grouping_inspector.py --arch gfx942 --block 5 6
python3 tools/counter_grouping_inspector.py --arch gfx942 --block 5 6 15 17

# Emit perfmon YAML names (compare to measured profile perfmon/ directory)
python3 tools/counter_grouping_inspector.py --arch gfx942 -v

# Save plan + metric/bucket report for diffing
python3 tools/counter_grouping_inspector.py --arch gfx942 --output /tmp/plan.txt
grep -E 'Summary:|Workgroup Manager Utilization|HBM Read Traffic' /tmp/plan.txt
```

**How to read output**

| Output | Meaning |
|--------|---------|
| `Summary: N bucket(s), …` | Simulated perfmon pass count |
| Multi-bucket metrics table | Metrics whose formula counters land in **2+ buckets** — split ratio partners |
| `-v` perfmon file list | Names like `pmc_perf_SQC_ICACHE_INFLIGHT_LEVEL_ACCUM.yaml` (matches on-disk profile layout) |

Example rows on full-panel gfx942 today (empty grouping policy):

```
Summary: 13 bucket(s), 274 counter assignment(s).

| 0600 | 601   | 2   | Workgroup Manager Utilization | 2 | SQC_DCACHE_INFLIGHT_LEVEL_ACCUM, SQC_ICACHE_INFLIGHT_LEVEL_ACCUM |
| 1700 | 1702  | 1   | HBM Read Traffic              | 2 | … |
```

**Grouping policy what-if:** gfx942 policy is `{}` in `src/rocprof_compute_soc/analysis_configs/profiling_counter_grouping_policy.yaml`. Add `same_bucket_priority_metric_ids` entries (e.g. `6.1.2`), re-run the inspector, and compare `Summary` plus whether Workgroup Manager Utilization / HBM Read Traffic move to single-bucket rows.

Example (temporary edit — restore after):

```yaml
gfx942:
  same_bucket_priority_metric_ids:
    "6.1.2":
      name: "Workgroup Manager Utilization"
```

```bash
python3 tools/counter_grouping_inspector.py --arch gfx942 | rg '^Summary:|Workgroup Manager Utilization|HBM Read Traffic'
# Expect: 19 bucket(s); Workgroup Manager Utilization | 0 |
```

Smoke test: `python3 tools/test_counter_grouping_inspector_manual.py`

---

## 8. Hardware validation (Conductor + CPX)

**Executed 2026-08-27** on `hpe-darkstar-ccs-aus-e12-03` (MI300X gfx942, Conductor pool `MI300X-AIG-SW-ML-LIBRARIES`). ROCm **7.15** + bundled **rocprof-compute 3.8.0**; `HIP_VISIBLE_DEVICES=9` (CPX card). Artifacts: `~/Downloads/aiprofcomp78-darkstar-data/`.

| Follow-up | Inspector (offline) | Hardware (darkstar) |
|-----------|--------------------|---------------------|
| **1. Policy pass count + wall time** | `{}` → **13**; priority entries → **19** (+6) | Empty → **13 passes / 108 s** ✓. Policy patched → **13 passes / 104 s** — **no perfmon change** on 3.8.0 (policy allocator is 3.10+) |
| **2. Block 17 single-pass HBM** | `--block 17` → **8 passes**, partners co-located | **8 passes / 85 s** ✓. HBM in `pmc_perf_2.yaml`. Raw `a > b`: **0/8** dispatches (vs **25%** occupancy full-panel merged baseline) |
| **3. Block 5 CPF (split partners)** | `--block 5` → **5 passes**, busy/idle in buckets **0** / **1** | **5 passes / ~50 s** ✓. `CPF_CPF_STAT_BUSY` in `pmc_perf_0.yaml`, `CPF_CPF_STAT_IDLE` in `pmc_perf_1.yaml`. Stitched CPF Util: **0/8** `> 100%` on test case occupancy |
| **4. rocflop HBM + WGM** | `--block 17` HBM; **`--block 6.1.2`** WGM | HBM **0/7** co-located (`--block 17`); WGM **0/7**, max **100%** (`--block 6.1.2`, **1 pass**) |
| **5. mat_exp HBM + WGM** | `--block 17` HBM; **`--block 6.1.2`** WGM on streams_sync `mat_exp` | HBM **0/208**; WGM **0/208**, max **100%** (vs **739.6%** merged) |
| **6. mat_exp CPF + Data-Return (panel)** | `--block 5` / `--block 15` (split layout) | Structural only — partners in different passes |
| **7. Metric-id co-location (§3.5 validated)** | `--block 6.1.2`, `5.1.0`, `15.4.0` (or `--set aiprofcomp78_bounds`) | **1 pass** each; all cap metrics **0** bad rows; Data-Return max **0.04%** (vs **345.38%** merged) |

**Follow-up 2 detail (test case occupancy, per-dispatch `100 × DRAM / RDREQ`):**

| Source | Dispatches | `a > b` | Max |
|--------|------------|---------|-----|
| occupancy_cpx full panel (merged, `aiprofcomp78-cpx-data`) | 8 | 2 (**25%**) | 103.9% |
| mat_exp CPX full panel (merged) | 208 | 15 (**7.2%**) | 188.2% |
| darkstar `--block 17` occupancy (pass 2, single-pass) | 8 | **0 (0%)** | **100.0%** |

Step-by-step commands and environment notes: [aiprofcomp78-hardware-validation-runbook.md](../reports/aiprofcomp78-hardware-validation-runbook.md).

---

## 9. Open follow-ups

1. **Policy pass count on rocprof-compute 3.10+:** Re-run follow-up 1 on hardware to confirm inspector **13 → 19** (+6 passes, ~+46% wall time) when policy YAML is applied.
2. **Investigate** whether pass count increase can be mitigated (allocator tuning vs selective priority metrics only).
3. **Ship gfx942 grouping policy?** Single-pass validation is **complete** via metric-id `--block` on 3.8.0; policy remains optional for default full-panel users who accept **+6 passes**.

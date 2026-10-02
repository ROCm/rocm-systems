# AIPROFCOMP-865 — Collectables-scoped delivery plan

**JIRA:** AIPROFCOMP-865 (single-run collection; parent AIPROFCOMP-864)
**Branch:** `users/feizheng10/aiprofcomp-865-single-run-collection`
**Scope:** Reuse **Collectables (Layer 1.5)** *concepts* from the [metric library LLD](https://github.com/ROCm/rocm-systems/blob/develop/projects/rocprofiler-compute/docs/design/analysis-config-redesign/lld-phase1-metric-library.md) to fix multi-pass ratio errors on the **legacy analysis YAML path**.
**Out of scope:** Full LLD (MetricLibrary, `!inherit`, adapter, Stage 4 migration, SDK collectables, collectable-aware coalesce).

**Related docs**

- Stakeholder Q&A: `aiprofcomp-865-stakeholder-qa.md`
- Phase 1 & 2 overview: `aiprofcomp-865-single-run-collection-phases.md`
- Phase 2 mechanics: `aiprofcomp-865-phase2-weighted-avg-design.md`
- gfx942 impact numbers: `aiprofcomp-865-gfx942-single-pass-impact-report.md`

---

## 1. Problem statement and status evaluation

### 1.1 Problem statement

Ratio metrics in rocprofiler-compute are defined as expressions over PMC counters (e.g. \(M = A/B\) or sums of terms over a denominator). When profiling requires **multiple perfmon replays**, numerators and denominators for the **same logical metric** may come from **different passes**. Evaluating the full formula on merged counter data is **not** guaranteed to equal any single kernel execution’s true ratio (see LLD § “Why cross-pass evaluation is mathematically wrong”).

**Jira 865 asks to:**

1. Prefer **same replay** for all counters needed per metric (Phase 1: grouping policy + coalesce).
2. Where hardware slots make that impossible, use **decomposed submetrics** and **correct analyze-time composition** (Phase 2).

**Collectables (LLD Layer 1.5)** name the right abstraction: a **single-pass fragment** (formula + PMC set) that must be collected together, then **composed** into a display metric. This document scopes 865 to that abstraction on **today’s panel YAML**, without waiting for the full metric-library rollout.

### 1.2 Normative intent (unchanged)

- If \(M = A/B\), all PMCs for \(A\) and \(B\) on a dispatch should be collected in the **same perfmon replay** when possible.
- Exceptions: intentional \>100% (e.g. VALU dual-issue), **SLOT_LIMIT** (decompose), or HW defect—not silent analyze-time caps.

### 1.3 Status evaluation (current branch + evidence)

#### Phase 1 — Same-run contract (profiler / grouping)

| Item | Status | Notes |
|------|--------|--------|
| Conductor CPX script (`aiprofcomp865_phase1_conductor_cpx.sh`) | **Done** | `ROCR_GPU=9`, single `ROCM_ROOT`, SQG omit in metrics path |
| TheRock + rocpd stack validation | **Done** | e.g. `therock715d_20260911_231523`; HBM 17.2.1 / 17.2.5 at 100% on subset |
| P0 golden offline (13-pass + policy) | **Partial** | P0 ids single-bucket in inspector; **91** metrics still multi-bucket globally |
| P0 worksheet (all metrics × workloads) | **Not complete** | 3-workload CPX matrix not fully signed |
| Grouping coalesce #10912 on develop | **Track** | Re-baseline after merge |
| Sysinfo partition vs ROCR die | **Open** | `cu_per_gpu=38` vs `SPX` / `num_xcd=8` on Conductor |

#### Phase 2 — Collectables on legacy YAML (analyze)

| Item | Status | Notes |
|------|--------|--------|
| `WEIGHTED_AVG` parse + per-dispatch merge | **Done** | `expression.py`, `aggregation.py`, `weighted_avg.py` (aliases) |
| **Collectable graph** + `apply_composite_metrics()` | **Done** | `collectable.py` |
| `COLLECT_SUM` (\(M = h + i\)) | **Done** | Xuan-style sum of single-pass sub-collectables |
| `_collectable_id` on rows | **Done** | Parser → `collectable_ids` attr |
| Test pilot (`tests/fixtures/weighted_avg/`) | **Done** | WEIGHTED_AVG + COLLECT_SUM fixtures |
| Production gfx942 pilot metric | **Pending** | After Phase 1 **SLOT_LIMIT** sign-off |
| Inspector collectable hints | **Partial** | `WEIGHTED_AVG` parents in text output |

#### gfx942 default profile — totals and decomposition (inspector)

| What | Count |
|------|------:|
| Unique raw HW PMC counters (default profile set) | **274** |
| Perfmon replays (buckets) | **13** |
| Analysis YAML metrics (gfx942) | **408** |

**408 metrics decomposed:**

| Category | Count | % of 408 |
|----------|------:|---------:|
| Has ≥1 PMC in default profile set | **374** | 91.7% |
| No profile PMCs (peaks, derived-only, etc.) | **34** | 8.3% |

**Of the 374 with PMCs** (% of 374):

| Category | Count | % of 374 |
|----------|------:|---------:|
| Single-bucket today | 283 | 75.7% |
| Multi-bucket (impacted) | 91 | 24.3% |
| → Fixable by coalesce/repack (no composite) | 75 | 20.1% |
| → Needs decomposition (collectables / Phase 2) | 16 | 4.3% |

(283 + 91 = 374; 75 + 16 = 91. The 34 are outside the 374 denominator.)

**On the 75:** coalesce today minimizes **pass count** for **all** counters, not “one bucket per metric.” Those 75 are **repackable/policy** cases, not proof that 13 passes is wrong. See `aiprofcomp-865-gfx942-single-pass-impact-report.md` §2.

**Conclusion:** 865 can be **closed on the legacy path** once Phase 1 evidence and a **bounded** Phase 2 pilot (or explicit waivers) are accepted. Full LLD remains a **separate, longer** program.

### 1.4 What we are *not* solving in this scope

- Replacing `analysis_configs/gfx942/*.yaml` with Layer 2 + `collectables.yaml`
- Per-pass collectable evaluation inside the profiler (LLD “future enhancement”)
- Collectable-aware `perfmon_coalesce()`
- SDK integration of collectable definitions

---

## 2. Design

### 2.1 Mapping LLD Collectables → 865 delivery

| LLD (long term) | 865 (this scope) |
|-----------------|------------------|
| `CollectableDefinition` in `collectables.yaml` | Metric table **row** with normal `avg:` formula; optional `_collectable_id: collect.*` |
| Layer 2 metric references `$collect.xxx` | Parent row references sub-rows by **name** in `WEIGHTED_AVG(a,b)` or `COLLECT_SUM(a,b)` |
| `avg_mode: weighted` at Layer 3 | `WEIGHTED_AVG` + `_weighted_avg.weight_counter` metadata |
| Sum of same-denominator parts | `COLLECT_SUM(h, i)` |
| `get_counters_for_metrics()` | Unchanged profiler path; sub-rows drive PMC via existing counter extraction |
| Per-pass eval of collectable | **Analyze:** per-dispatch eval of cached sub-expression on `raw_pmc_df` (requires sub-row **single-bucket**) |

### 2.2 Analyze pipeline (implemented)

```text
build_dfs / parser
  → df.attrs: weighted_avg_specs, weighted_avg_subs, collect_sum_specs, collectable_ids

build_metric_value_string
  → built Avg strings for collectable rows; empty for composite parents

cache_collectable_expressions()
  → snapshot collectable built Avg before overwrite

eval_metric()
  → evaluate collectable rows (normal path)

apply_composite_metrics()
  → build_metric_eval_graph(); for each composite:
       WEIGHTED_AVG: per-dispatch weighted merge
       COLLECT_SUM: per-dispatch sum, then run-level avg
```

**Key modules:** `src/utils/metrics/collectable.py`, `evaluation_pipeline.py`, `parser.py`, `expression.py`, `aggregation.py`.

### 2.3 YAML contract (legacy panel)

**Collectable row (single-pass fragment):**

```yaml
hbm_read_sub:
  avg: 100 * SUM(TCC_EA0_RDREQ_DRAM_sum) / SUM(TCC_EA0_RDREQ_sum)
  unit: Percent
  _collectable_id: collect.hbm_read   # optional; stable id for future LLD import
```

**Parent — weighted (partitioned denominators):**

```yaml
hbm_combined:
  avg: WEIGHTED_AVG(hbm_read_sub, hbm_write_sub)
  unit: Percent
  _weighted_avg:
    hbm_read_sub:
      weight_counter: TCC_EA0_RDREQ_sum
    hbm_write_sub:
      weight_counter: TCC_EA0_WRREQ_sum
```

**Parent — sum (Xuan \(M = h + i\), shared structure):**

```yaml
hbm_total:
  avg: COLLECT_SUM(hbm_read_sub, hbm_write_sub)
  unit: Percent
```

**Rules**

- Collectable rows must be **single-bucket** in `counter_grouping_inspector` for the profile counter set.
- Parent composite rows do not need PMCs in profile list (analyze-only composite) if children supply counters.
- Do not use `WEIGHTED_AVG` when \(M = h + i\) is algebraically exact; use `COLLECT_SUM`.

### 2.4 Phase 1 vs Phase 2 division

| Verdict (worksheet) | Action |
|---------------------|--------|
| **PASS** | No collectable split |
| **POLICY_GAP** | `profiling_counter_grouping_policy.yaml` / coalesce (#10912); re-inspector |
| **SLOT_LIMIT** | Introduce collectable rows + composite parent; verify each collectable one bucket |
| **INTENTIONAL** | Document (e.g. VALU); no composite for inflation KPI |
| **HW_BUG** | Driver/firmware; no YAML cap |

### 2.5 Future LLD alignment (no work in 865 closure)

When `MetricLibrary` + adapter land:

- Import `_collectable_id` → `collectables.yaml` entries.
- Adapter emits same `df.attrs` / graph as today’s parser.
- `WEIGHTED_AVG` / `COLLECT_SUM` remain composite operators (or map to L3 `avg_mode`).

---

## 3. Plan and estimation

### 3.1 Definition of done (865 Jira closure)

1. **Phase 1:** Completed worksheet for **P0** metrics × agreed workloads (or documented subset + waivers); inspector evidence archived; Conductor re-run post-#10912 if applicable.
2. **Phase 2:** At least **one** SLOT_LIMIT (or agreed) metric converted to collectable rows + composite parent in **production** gfx942 YAML *or* written waiver with owner sign-off.
3. **Regression:** Unit/integration tests green on metrics without composite parents.
4. **Jira:** Summary linking impact report, pilot metric, and explicit **“LLD Stages 1–4 deferred”** statement.

### 3.2 Work breakdown and estimates

Assumes **one primary engineer**, reviews from Xuan/maintainer within 2–3 business days. Estimates are **working days**, not calendar days.

| # | Task | Dep | Est. |
|---|------|-----|-----:|
| P1-1 | Finish P0 worksheet template + fill from inspector + Conductor artifacts | — | 1–2 d |
| P1-2 | Re-run inspector full gfx942; save baseline; note POLICY_GAP vs SLOT_LIMIT | P1-1 | 0.5 d |
| P1-3 | Re-profile / analyze if #10912 merged or policy PR lands | P1-2 | 1–2 d |
| P2-1 | Pick pilot from SLOT_LIMIT list (identity + weights documented) | P1-2 | 0.5 d |
| P2-2 | Add production gfx942 YAML (collectables + parent); policy ids if needed | P2-1 | 1 d |
| P2-3 | Inspector: all collectable rows single-bucket; optional collectable id hint | P2-2 | 0.5 d |
| P2-4 | Hardware or workload analyze check vs hand calc (`pmc_perf.csv`) | P2-2 | 1 d |
| DOC-1 | Jira closure comment + PR series description | P2-4 | 0.5 d |
| ENG-1 | Commit/push collectable graph work (if not already on remote) | — | 0.5 d |

**Total to Jira closure:** **~5–8 working days** (about **1–1.5 calendar weeks** with reviews and Conductor access).

**Not included:** LLD Stage 1–4 (**several weeks to months**); migrating all 16 SLOT_LIMIT metrics (**optional backlog**).

### 3.3 Suggested sequence (one week aggressive)

| Day | Focus |
|-----|--------|
| 1 | Worksheet + inspector baselines; classify P0 rows |
| 2 | Policy/coalesce follow-ups or #10912 re-baseline decision |
| 3 | Pilot metric choice + YAML + unit tests |
| 4 | Inspector + analyze validation (local or Conductor) |
| 5 | PR polish, Jira closure, stakeholder review |

### 3.4 Risks

| Risk | Mitigation |
|------|------------|
| Pilot metric not true SLOT_LIMIT (coalesce fixes it) | Use worksheet + packability check before YAML |
| Conductor / partition noise | Document sysinfo limits; separate HW vs analyze bugs |
| Scope creep into full LLD | This doc + Jira explicitly bound to legacy path |
| `COLLECT_SUM` vs `WEIGHTED_AVG` wrong choice | Algebra checklist in design §2.3 |

### 3.5 Deliverables checklist

- [ ] `aiprofcomp-865-phase1-worksheet.md` (or equivalent) with P0 rows filled
- [ ] Saved inspector plan (`gfx942-full-plan.txt` or CI artifact)
- [ ] Production pilot YAML **or** signed waiver
- [ ] Tests: `test_weighted_avg*.py`, `test_collectable.py` passing in CI
- [ ] Jira AIPROFCOMP-865 resolution comment referencing this document

---

## Revision history

| Date | Change |
|------|--------|
| 2026-09-16 | Initial collectables-scoped plan for 865 closure vs long-term LLD |

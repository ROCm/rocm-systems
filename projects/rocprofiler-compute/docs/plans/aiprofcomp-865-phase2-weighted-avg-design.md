# AIPROFCOMP-865 Phase 2 — `WEIGHTED_AVG` design and implementation plan

**Status:** Design (handoff to implementation agent)
**JIRA:** AIPROFCOMP-865 (bullet 2–3), parent AIPROFCOMP-864
**Prerequisite:** Phase 1 worksheet classifies metrics as `SLOT_LIMIT` with inspector evidence
**Out of scope:** `BOUND_RATIO`, partition normalization (AIPROFCOMP-798), alola-only workflows

---

## 1. Problem statement

When a ratio metric \(M = (A+B)/C\) (or a sum of numerators over one denominator) cannot fit all required PMCs in **one perfmon bucket**, multi-pass profiling makes `SUM(A)/SUM(C)` and `MAX(A/C)` unreliable.

Phase 1 addresses the common case \(M = A/B\) via **metric-aware coalescing** and `profiling_counter_grouping_policy.yaml`. Phase 2 covers **hard slot limits** by:

1. Splitting into submetrics collected in **separate single-pass replays**:
   - \(M_0 = A/C_0\), \(M_1 = B/C_1\)
2. Recombining in analyze with a **weighted average**:
   - \(M = \dfrac{M_0 C_0 + M_1 C_1}{C_0 + C_1}\)

This matches the AIPROFCOMP-864/865 Jira note and avoids silent analyze-time caps.

---

## 2. Preconditions (from Phase 1)

| Gate | Owner | Artifact |
|------|--------|----------|
| Metric listed as `SLOT_LIMIT` | Phase 1 agent | Worksheet row + inspector multi-bucket output |
| Identity proof | Design reviewer | Short note showing \(C_0 + C_1 = C\) (or approved partition of denominator) |
| Submetrics single-bucket | Inspector | Each submetric id → 1 perfmon bucket |
| Xuan / maintainer sign-off | PR reviewer | Comment on this doc §4 schema |

Do **not** implement `WEIGHTED_AVG` for metrics that Phase 1 can fix with policy/coalesce alone.

---

## 3. YAML contract (proposed)

### 3.1 Parent metric (display metric)

```yaml
HBM Combined Traffic (example):
  avg: WEIGHTED_AVG(hbm_read_sub, hbm_write_sub)
  unit: Percent
  _weighted_avg:
    hbm_read_sub:
      weight_counter: TCC_EA0_RDREQ_sum
    hbm_write_sub:
      weight_counter: TCC_EA0_WRREQ_sum
```

Rules:

- `WEIGHTED_AVG(...)` appears only in **aggregation fields** (`avg`, optionally `min`/`max` TBD — see §6).
- `_weighted_avg` is **metadata** (ignored by autogen validators that only scan `value`/`avg` formulas) or moved to a parallel `metric_components:` block if validators require it.
- Submetrics (`hbm_read_sub`, …) are **normal metrics** with their own `avg: 100 * SUM(A)/SUM(C0)` definitions and their own PMC sets.

### 3.2 Alternative (explicit weights in function)

```yaml
avg: WEIGHTED_AVG(
  METRIC(3.1.63, weight=TCC_EA0_RDREQ_sum),
  METRIC(3.1.64, weight=TCC_EA0_WRREQ_sum)
)
```

Prefer **one** style after Xuan review; implementation agent should not ship both.

---

## 4. Analyze pipeline changes

| Component | Change |
|-----------|--------|
| `src/utils/metrics/expression.py` (or parser) | Parse `WEIGHTED_AVG(...)` and build AST node |
| `src/utils/metrics/aggregation.py` | `to_weighted_avg(sub_results, weights)` using per-dispatch aligned series |
| `utils_analysis.py` / metric eval | Evaluate submetrics first; pass PMC columns into weighted merge |
| `tools/counter_grouping_inspector.py` | Optional: flag parent metrics that declare `WEIGHTED_AVG` and verify each sub-id is single-bucket |
| Autogen / unified_config | If metrics are generated, extend split script or hand-author pilot in gfx942 YAML |

### 4.1 Merge semantics (normative)

For each dispatch row \(i\) where all submetrics and weights exist:

\[
M_i = \frac{\sum_k M_{k,i} \cdot C_{k,i}}{\sum_k C_{k,i}}
\]

Run-level `avg` aggregates dispatch rows using the same definition as today (sum across dispatches / kernel scope per existing metric `avg` rules). **Do not** implement `WEIGHTED_AVG` as `SUM(MIN)/SUM` caps.

### 4.2 Masking risk

Lower than `BOUND_RATIO` if weights are raw counters and submetrics are single-pass. Still document in metric description that parent metric is **derived**, not a direct HW ratio.

---

## 5. Profiler / grouping interaction

- Submetrics remain **separate metric ids** in grouping policy (`same_bucket_priority_metric_ids`).
- Profiler still emits **multiple perfmon files**; parent metric may not appear in profile PMC list (analyze-only composite).
- No change to `soc_base._metric_aware_coalesce_pass` unless submetrics need cross-metric packing hints.

---

## 6. Open design choices (resolve before coding)

1. **min / max** — Reject `WEIGHTED_AVG` for `max` initially, or define conservative bounds?
2. **Missing weight rows** — Skip dispatch vs propagate NaN?
3. **Pilot metric** — First candidate from Phase 1 `SLOT_LIMIT` list (likely **not** gfx942 HBM if Phase 1 + #10912 fixes coalesce).
4. **Interaction with `NOISE_CLAMP`** — Do not double-decompose Remote traffic metrics.

---

## 7. Implementation plan (task breakdown)

### Milestone A — Parser and unit math (1 PR)

| Task | Files | Tests |
|------|-------|-------|
| A1 AST + parser for `WEIGHTED_AVG` | `expression.py`, parser tests | Parse valid/invalid YAML snippets |
| A2 `to_weighted_avg` | `aggregation.py` | Numeric table: 2 submetrics, 3 dispatches |
| A3 Wire eval order (children before parent) | `utils_analysis.py` | Unit test with mock PMC frame |

### Milestone B — Pilot YAML (1 PR, stacks on A)

| Task | Files | Tests |
|------|-------|-------|
| B1 Pick pilot metric from Phase 1 `SLOT_LIMIT` | `gfx942/*.yaml` | Inspector single-bucket per sub-id |
| B2 Document in metric description | same | — |
| B3 Integration test or workload fixture | `tests/integration` or analyze golden | Optional if hardware fixture exists |

### Milestone C — Tooling and docs (1 PR)

| Task | Files |
|------|-------|
| C1 Inspector hint for `WEIGHTED_AVG` parents | `counter_grouping_inspector.py` |
| C2 Update Phase 1 plan cross-link | `aiprofcomp-865-single-run-collection-phases.md` |
| C3 Design review closure | Jira AIPROFCOMP-865 comment |

---

## 8. Verification

1. Inspector: each submetric → **one bucket** on full gfx942 panel.
2. Analyze: parent `avg` matches hand calculation from exported `pmc_perf.csv` on pilot workload.
3. CPX conductor run (Phase 1 host): parent metric ≤ 100% where submetrics are single-pass (no new caps).
4. Regression: metrics without `WEIGHTED_AVG` unchanged (diff analyze output on fixed workload).

---

## 9. Agent handoff checklist

Implementation agent should:

- [ ] Read Phase 1 `SLOT_LIMIT` rows in `docs/plans/aiprofcomp-865-single-run-collection-phases.md` worksheets (on branch or Conductor artifact dir).
- [ ] Confirm grouping fix (#10912) merge status on `rocprofiler-compute-develop` before blaming slot limits.
- [ ] Resolve §6 open choices with reviewer; default **avg-only**, **skip dispatch on missing weight**.
- [ ] Implement Milestone A → B → C as separate commits.
- [ ] Do **not** add `BOUND_RATIO` in the same PR series.

---

## 10. References

- Phase 1 plan: `docs/plans/aiprofcomp-865-single-run-collection-phases.md`
- Grouping policy: `src/rocprof_compute_soc/analysis_configs/profiling_counter_grouping_policy.yaml`
- Coalesce: `src/rocprof_compute_soc/soc_base.py` (`_metric_aware_coalesce_pass`)
- Correction methods (caps deferred): design PR #10655 / `docs/design/metric-counter-correction-methods.md` if present on branch

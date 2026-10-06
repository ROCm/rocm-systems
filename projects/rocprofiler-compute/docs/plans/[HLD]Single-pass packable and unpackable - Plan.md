# Single-pass packable and unpackable

**JIRA:** AIPROFCOMP-865 (parent AIPROFCOMP-864)

**Scope:** Fix multi-pass ratio errors on the existing analysis YAML structure.

**Out of scope:** Full metric-library migration (MetricLibrary, Stage 4, SDK collectables registry).

**Related design:**

- [Analysis config YAML redesign](../design/analysis-config-redesign/hld-analysis-config-redesign.md#layer-details) — Layer 1.5 collectables. This plan uses that idea on today’s panel YAML.
- [Three-layer analysis config LLD](../design/analysis-config-redesign/lld-index.md) and [metric library LLD](../design/analysis-config-redesign/lld-phase1-metric-library.md) — the migration this plan does not do.

This is the overall plan. The packing algorithm and flowchart are in [Metric grouping (SPP / SPU)](<[HLD]Single-pass packable and unpackable - Metrics Grouping Algorithm.md>).

Full historical plan drafts are preserved on branch
`users/feizheng10/aiprofcomp-865-docs-backup`.

---

## 1. Terminology

| Term | Meaning |
|------|---------|
| **Single-pass packable(SPP)** | Metric whose PMC set fits one hardware bucket. **Collection:** guarantee counter collection within a single pass (PMC duplication across passes allowed). **Analyze:** same-pass bind — each expression uses counters from one co-located pass (shadow columns `{counter}@pass:{key}`), so ratios are not evaluated on values merged across disjoint replays. |
| **Single-pass unpackable(SPU)** | Cannot fit one bucket even with global repack → decompose into collectables and recompose with `WEIGHTED_AVG` / `COLLECT_SUM` / `COLLECT_RATIO`. (Slot-budget limited under `perfmon_config`.) |
| **[Collectable](https://github.com/ROCm/rocm-systems/blob/users/feizheng10/aiprofcomp-865-docs-backup/projects/rocprofiler-compute/docs/plans/aiprofcomp-865-problem-decompose.html)** | A single-pass fragment (formula + PMC set) that must be collected together, then composed into a display metric ([Layer 1.5](../design/analysis-config-redesign/hld-analysis-config-redesign.md#layer-details) on today’s panel YAML). Used to solve **SPU** parents. |
| **POLICY_GAP** | Shipping-layout diagnosis: multi-bucket today but **is** SPP under the new packer (will be fixed by Phase 1 below). |
| **WEIGHTED_AVG** | Analyze composite: recombine single-pass submetrics as \((M_0 C_0 + M_1 C_1)/(C_0 + C_1)\). |
| **COLLECT_SUM / COLLECT_RATIO** | Analyze composites for sum-of-subcollectables and ratio-of-collectables parents. |

---

## 2. Current status

Ratio metrics are expressions over PMC counters. With **multiple perfmon replays**, numerators and denominators for the same logical metric can come from **different passes**. Merged-pass `SUM(A)/SUM(B)` is not guaranteed to match any single kernel execution’s true ratio.

**Shipping heuristic baseline (gfx942, offline inspector):**

| Quantity | Value |
|----------|------:|
| Unique HW PMCs in default profile | 274 |
| Perfmon passes | **12–13** |
| YAML metrics scanned | 408 |
| Metrics with profile PMCs | **374** |
| Single-bucket today | **283** |
| Multi-bucket (impacted) | **75** packable (**POLICY_GAP** → SPP) + **16** **SPU** |
| Derived metrics with no profile PMCs | 34 |

**Normative intent:** for \(M = A/B\), collect all PMCs for \(A\) and \(B\) in the same replay when possible. Exceptions: intentional >100% (e.g. VALU dual-issue), **SPU** (Phase 2), or HW defect — not silent analyze-time caps.

---

## 3. Target

```
408 YAML metrics (gfx942 default)
├── 374 with profile PMCs
│   ├── 358 Single-pass packable(SPP)     ← Phase 1 (~14 passes + same-pass bind)
│   └── 16 Single-pass unpackable(SPU)    ← Phase 2 collectables
└── 34 with no profile PMCs                ← out of packing scope
```

| Goal | Detail |
|------|--------|
| SPP metrics | Full PMC set in **one** replay + same-pass bind at analyze |
| SPU parents | Decompose into collectables; recompose with `WEIGHTED_AVG` / `COLLECT_SUM` / `COLLECT_RATIO` |
| Escape hatch | `ROCPROF_COMPUTE_ANALYZE_LEGACY_PASS_MERGE=1` / `ROCPROF_COMPUTE_PERFMON_LEGACY_HEURISTIC=1` during soak; removed in Phase 2b cleanup PR |

**SPP is collection + post-analysis.** Packing without same-pass bind still yields wrong ratios under PMC duplication; bind without packing still leaves packable metrics multi-bucket. Phase 1 ships both. Phase 2 is only SPU collectables / composites.

---

## 4. Solution

### 4.1 Phase 1 design — Single-pass packable(SPP)

#### Collection — single-pass packing

Replace the shipping heuristic as the default allocator. The normative steps, flowchart, TCC series affinity, and packing gates are in [Metric grouping (SPP / SPU)](<[HLD]Single-pass packable and unpackable - Metrics Grouping Algorithm.md>).

**Primary packing code:** `counter_grouping_single_pass.py`, `counter_grouping_buckets.py`, `soc_base.py`.

#### Analyze — same-pass bind (part of Single-pass packable(SPP))

When packing duplicates hub counters across passes:

1. Preserve per-pass PMC columns as `{counter}@pass:{key}` shadows at load time.
2. Bind each SPP metric’s expression to a co-located pass (`PassLayout` / `pass_provenance.py`).
3. Wire CLI (`eval_metric`) and DB (`calc_expressions` / `bind_expression_dataframe`).

Without this bind, SPP layouts can produce incorrect percent averages/maxes (e.g. CPC Utilization avg ≫ 100%).

**Primary bind code:** `pass_provenance.py`, `file_io` / `utils_analysis` shadow columns, analyze CLI+DB bind paths.

### 4.2 Phase 2 design — Collectables for Single-pass unpackable(SPU)

SPU parents cannot fit one bucket even with global repack → decompose into collectables and recompose with `WEIGHTED_AVG` / `COLLECT_SUM` / `COLLECT_RATIO`.

Pick the operator from the parent algebra:

- `COLLECT_SUM` adds already-single-pass submetrics. HBM bandwidth is read BW + write BW, so `COLLECT_SUM(hbm_read_sub, hbm_write_sub)`.
- `WEIGHTED_AVG` recombines ratio pieces whose weights were split across passes: \(M = (M_0 C_0 + M_1 C_1)/(C_0 + C_1) = (A+B)/(C_0+C_1)\). That is not the sum of the two ratios (\(A/C_0 + B/C_1\)). Do not use it to add two bandwidths.
- `COLLECT_RATIO` is a ratio of summed numerator pieces over summed denominator pieces.

YAML sketch (HBM bandwidth):

```yaml
value: COLLECT_SUM(hbm_read_sub, hbm_write_sub)
```

Weighted-ratio sketch (not a bandwidth sum):

```yaml
avg: WEIGHTED_AVG(read_ratio_sub, write_ratio_sub)
_weighted_avg:
  read_ratio_sub:
    weight_counter: TCC_EA0_RDREQ_sum
  write_ratio_sub:
    weight_counter: TCC_EA0_WRREQ_sum
```

Submetrics are normal single-pass (SPP) metrics. Parents are analyze-time composites (collectable graph / `apply_composite_metrics()`).

gfx942: **16** SPU parents ≈ **10** unique PMC sets (panel mirrors share sets).

#### gfx942 Single-pass unpackable(SPU) conversion summary

All **16** former SPU parents (**10** unique PMC sets) are analyze-time composites. Sub-collectables are single-bucket; offline eval reports `slot_limit_metrics: 0` (tool key for the SPU residual count).

| Unique set | Composite |
|------------|-----------|
| VALU FLOPs | `COLLECT_SUM` F16/F32/F64 rates |
| vL1D hit | `COLLECT_SUM` non-RW hit + atom correction |
| HBM Bandwidth | `COLLECT_SUM` rd/wr BW |
| AI HBM/L2/L1/LDS | `COLLECT_RATIO`(FLOP pieces, bytes) |
| Perf GFLOPs | `COLLECT_SUM` VALU+MFMA rates |
| IPC Issued | `COLLECT_SUM` two IPC slices |
| Read Instructions | `COLLECT_SUM` load + (−store−atomic) |

`min`/`max` on composite parents are `None` (avg-only) so they do not re-introduce the full PMC set into grouping.

Offline gate: `PYTHONPATH=src:tools python3 tools/eval_single_pass_packable.py --arch gfx942` → `slot_limit_metrics: 0`, `packable_multi: 0`, passes: 14.

**Primary code:** `collectable.py`, `weighted_avg.py`, aggregation/expression/evaluation_pipeline composite path; gfx942 SPU YAML conversions.

### 4.3 Phase 2 follow-on — Legacy cleanup (separate PR)

After SPP + SPU collectables land and verification (including blocking-ticket checks) is green, remove migration-only legacy paths in a **separate PR** (do not mix with collectables):

- Profile: drop `ROCPROF_COMPUTE_PERFMON_LEGACY_HEURISTIC` / `ROCPROF_COMPUTE_PERFMON_SINGLE_PASS_PACKABLE=0` escape hatches and the old heuristic allocator path once SPP is the sole default.
- Analyze: drop `ROCPROF_COMPUTE_ANALYZE_LEGACY_PASS_MERGE` and merged-pass analyze fallbacks once same-pass bind is mandatory.
- Dead priority-coalesce / obsolete grouping helpers that exist only for the pre-SPP layout.
- Drop the priority-tier sort in `_iter_metric_groups`. It still consults `_same_bucket_priority_metric_ids()`. SPP ignores that sort key and orders unions largest-first.
- Docs/tests that exercise only the legacy escape hatches.

Keep cleanup gated on Phase 1 + Phase 2 (collectables) acceptance so rollback via env vars remains available during soak.

---

## 5. Evaluation / Verification Plan

### 5.1 CTest / unit tests

Existing CTest suite still passes, plus new unit tests for each component:

- Phase 1: packing (`test_counter_grouping_single_pass`, buckets, `soc_base` packing path) and same-pass bind (`test_pass_provenance`, DB bind).
- Phase 2: collectables / `WEIGHTED_AVG` / `COLLECT_RATIO` (`test_collectable`, `test_weighted_avg*`, `test_collect_ratio`).

### 5.2 Manual offline grouping checks

Inspector + `eval_single_pass_packable.py`:

- `packable_multi == 0`, pass count ≈ **14**, SPU residual fill **+0** on gfx942.
- After Phase 2: `slot_limit_metrics == 0`.

### 5.3 End-to-end / multi-arch regression (3 workloads)

Run health runners + report on both Phase 1 and Phase 2 stacks across gfx908 / gfx90a / gfx942 / gfx950 / gfx115x / gfx1250 as available. **CPX mode only on MI300 (gfx942).**

Workloads:

- `vcopy`
- `nbody` (`mini-nbody`)
- `mega_kernel`

Each run should cover:

- **(a) High-level metric validation** — panel / SoL-style sanity (avg/max ≤ 100% where expected; no impossible ratios).
- **(b) Delta comparison before/after** — SPP (and Phase 2 where applicable) vs legacy packing / legacy pass-merge baselines (`compare_spp_legacy_medians` / health-report deltas).

### 5.4 SPU collectables (Phase 2 only)

gfx942 SLOT→composite conversions, `slot_limit_metrics == 0`, and before/after comparison (`compare_slot16_phase2.py` / health deltas) for the 16 SPU parents.

### 5.5 Blocking-ticket validation

Re-check the metric classes called out by these tickets after SPP (Phase 1) and again after SPU collectables (Phase 2) where applicable. AIPROFCOMP-865 is the umbrella; these are expected to clear once it lands:

   | Ticket | Symptom |
   |--------|---------|
   | [AIPROFCOMP-265](https://ontrack-internal.amd.com/browse/AIPROFCOMP-265) | Incorrect CPC metrics |
   | [AIPROFCOMP-90](https://ontrack-internal.amd.com/browse/AIPROFCOMP-90) | L1 bandwidth > 100% |
   | [AIPROFCOMP-268](https://ontrack-internal.amd.com/browse/AIPROFCOMP-268) | Incorrect cache metrics |
   | [AIPROFCOMP-267](https://ontrack-internal.amd.com/browse/AIPROFCOMP-267) | Incorrect TA/TD metrics |
   | [AIPROFCOMP-266](https://ontrack-internal.amd.com/browse/AIPROFCOMP-266) | Incorrect Workgroup Manager utilization |
   | [ROCM-31864](https://ontrack-internal.amd.com/browse/ROCM-31864) | gfx950 L2-Fabric HBM / remote read traffic incorrect (incl. >100%) |

**Stack:** Doc → Phase 1 (SPP packing + same-pass bind) → Health utils → Phase 2a (SPU collectables / `WEIGHTED_AVG` / `COLLECT_*`) → Phase 2b (legacy cleanup, separate PR).

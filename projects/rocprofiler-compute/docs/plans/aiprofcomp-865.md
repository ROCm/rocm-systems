# AIPROFCOMP-865 — Single-pass packable and collectables

**JIRA:** AIPROFCOMP-865 (parent AIPROFCOMP-864)

**Scope:** Fix multi-pass ratio errors on the legacy analysis YAML path.

**Out of scope:** Full metric-library LLD (MetricLibrary, `!inherit`, Stage 4 migration, SDK collectables registry).

Full historical plan drafts are preserved on branch
`users/feizheng10/aiprofcomp-865-docs-backup`.

---

## 1. Terminology

| Term | Meaning |
|------|---------|
| **Collectable** | A single-pass fragment (formula + PMC set) that must be collected together, then composed into a display metric (LLD Layer 1.5 concept on today’s panel YAML). |
| **SPP (Single-pass packable)** | Default packing objective: every metric whose PMC set fits one hardware bucket gets a co-located perfmon replay. |
| **SLOT_LIMIT** | Metric whose full PMC set cannot fit one hardware bucket under `perfmon_config` slot limits, even with global repack. |
| **POLICY_GAP** | Metric that is multi-bucket under the old shipping layout but **is** packable in one bucket with SPP (fixed by Phase 1, not Phase 2). |
| **Same-pass bind** | Analyze-time binding of each packable metric’s expression to counters from one co-located pass (shadow columns `{counter}@pass:{key}`), so ratios are not evaluated on values merged across disjoint replays. |
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
| Single-bucket today | **~283–299** |
| Multi-bucket (impacted) | **~75–91** packable (**POLICY_GAP**) + **16** **SLOT_LIMIT** |
| Metrics with no profile PMCs | 34 |

**Normative intent:** for \(M = A/B\), collect all PMCs for \(A\) and \(B\) in the same replay when possible. Exceptions: intentional >100% (e.g. VALU dual-issue), **SLOT_LIMIT** (Phase 2), or HW defect — not silent analyze-time caps.

---

## 3. Target

```
408 YAML metrics (gfx942 default)
├── 374 with profile PMCs
│   ├── 358 single-pass (all packable)     ← Phase 1 SPP (~14 passes)
│   └── 16 SLOT_LIMIT                      ← Phase 2 collectables
└── 34 with no profile PMCs                ← out of packing scope
```

| Goal | Detail |
|------|--------|
| Packable metrics | Full PMC set in **one** replay (duplication across passes allowed when unions conflict) |
| SLOT_LIMIT parents | Decompose into single-pass subcollectables; compose with `WEIGHTED_AVG` / `COLLECT_SUM` / `COLLECT_RATIO` |
| Analyze contract | Same-pass bind so duplicated hub counters are not cross-pass merged |
| Escape hatch | `ROCPROF_COMPUTE_ANALYZE_LEGACY_PASS_MERGE=1` restores legacy merged-pass analyze |

**SPP spans collection and post-analysis.** Packing without same-pass bind still yields wrong ratios under PMC duplication; bind without packing still leaves packable metrics multi-bucket. Phase 1 ships packing; Phase 2 ships bind + SLOT composites (stacked).

---

## 4. Solution

### 4.1 Phase 1 design — Single-pass packable (collection)

**Replace** the shipping heuristic + priority coalesce as the default allocator in
`_allocate_perfmon_counter_files`.

Algorithm (normative sketch):

1. Build unique **packable** PMC unions (skip SLOT_LIMIT parents).
2. Largest-first: ensure some bucket contains each union’s full set (duplicate PMCs across passes when needed).
3. Run **SLOT_LIMIT fill** so residual SLOT PMCs appear somewhere (+0 extra passes on gfx942).
4. Optional harden: TCC series affinity / ACCUM slot charging where required.

**Locked decisions:**

- Default path is SPP; optional `ROCPROF_COMPUTE_PERFMON_LEGACY_HEURISTIC=1` during migration.
- Do **not** use `WEIGHTED_AVG` for former POLICY_GAP metrics — packing covers them.
- Priority policy YAML is not required for the packable guarantee.
- gfx942 offline gates: `packable_multi == 0`, passes ≈ **14**, SLOT_LIMIT == **16**, SLOT extra passes == **0**.

**Primary code:** `counter_grouping_single_pass.py`, `counter_grouping_buckets.py`, `soc_base.py`.

### 4.2 Phase 2 design — Collectables + same-pass bind (analyze)

#### Slot-limited parents

When \(M = (A+B)/C\) cannot fit one bucket:

1. Collect submetrics in separate single-pass replays: \(M_0 = A/C_0\), \(M_1 = B/C_1\).
2. Recombine in analyze: \(M = (M_0 C_0 + M_1 C_1)/(C_0 + C_1)\) via `WEIGHTED_AVG`, or use `COLLECT_SUM` / `COLLECT_RATIO` where appropriate.

YAML sketch:

```yaml
avg: WEIGHTED_AVG(hbm_read_sub, hbm_write_sub)
_weighted_avg:
  hbm_read_sub:
    weight_counter: TCC_EA0_RDREQ_sum
  hbm_write_sub:
    weight_counter: TCC_EA0_WRREQ_sum
```

Submetrics are normal single-pass metrics. Parents are analyze-time composites (collectable graph / `apply_composite_metrics()`).

gfx942: **16** SLOT_LIMIT parents ≈ **10** unique PMC sets (panel mirrors share sets).

#### Same-pass bind (companion to SPP duplication)

When Phase 1 duplicates hub counters across passes:

1. Preserve per-pass PMC columns as `{counter}@pass:{key}` shadows at load time.
2. Bind each packable metric’s expression to a co-located pass (`PassLayout` / `pass_provenance.py`).
3. Wire CLI (`eval_metric`) and DB (`calc_expressions` / `bind_expression_dataframe`).

Without this bind, SPP layouts can produce impossible percent averages/maxes (e.g. CPC Utilization avg ≫ 100%).

**Primary code:** `collectable.py`, `weighted_avg.py`, aggregation/expression/evaluation_pipeline; `pass_provenance.py`, `file_io` / analyze CLI+DB bind; gfx942 SLOT YAML conversions.

---

## 5. Evaluation / Verification Plan

| Layer | Check |
|-------|--------|
| Offline packing | `eval_single_pass_packable.py` / inspector: `packable_multi == 0`, pass count, SLOT fill +0 |
| Unit tests | Packing (`test_counter_grouping_single_pass`); collectables / WA / `COLLECT_RATIO`; same-pass bind (`test_pass_provenance`, DB bind tests) |
| Health report | SPP vs legacy health runners + `generate_metric_health_report.py` on gfx942 (and other arches as needed) |
| Hardware spot-check | gfx942 CPX/SPX: P0 HBM / WGM / CPC ratios sane (avg/max ≤ 100% where expected) |
| Regression workload | MI-PATH `ns3d*` (AIPROFCOMP-265 class): SPP + same-pass bind vs legacy packing / legacy pass merge |
| Multi-arch | Re-baseline gfx908, gfx90a, gfx950, gfx115x, gfx1250 offline after Phase 1 |

**Stack pairing:** Phase 1 packing alone must not be treated as the full SPP product guarantee until Phase 2 same-pass bind is present.

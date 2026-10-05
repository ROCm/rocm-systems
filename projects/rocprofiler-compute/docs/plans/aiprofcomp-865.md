# AIPROFCOMP-865 — Single-pass packable and unpackable

**JIRA:** AIPROFCOMP-865 (parent AIPROFCOMP-864)

**Scope:** Fix multi-pass ratio errors on the existing analysis YAML structure.

**Out of scope:** Full metric-library LLD (MetricLibrary, Stage 4 migration, SDK collectables registry).

Full historical plan drafts are preserved on branch
`users/feizheng10/aiprofcomp-865-docs-backup`.

---

## 1. Terminology

| Term | Meaning |
|------|---------|
| **Single-pass packable(SPP)** | Metric whose PMC set fits one hardware bucket. **Collection:** guarantee counter collection within a single pass (PMC duplication across passes allowed). **Analyze:** same-pass bind — each expression uses counters from one co-located pass (shadow columns `{counter}@pass:{key}`), so ratios are not evaluated on values merged across disjoint replays. |
| **Single-pass unpackable(SPU)** | Cannot fit one bucket even with global repack → decompose into collectables and recompose with `WEIGHTED_AVG` / `COLLECT_SUM` / `COLLECT_RATIO`. (Slot-budget limited under `perfmon_config`.) |
| **[Collectable](https://github.com/ROCm/rocm-systems/blob/users/feizheng10/aiprofcomp-865-docs-backup/projects/rocprofiler-compute/docs/plans/aiprofcomp-865-problem-decompose.html)** | A single-pass fragment (formula + PMC set) that must be collected together, then composed into a display metric (LLD Layer 1.5 concept on today’s panel YAML). Used to solve **SPU** parents. |
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
| Escape hatch | `ROCPROF_COMPUTE_ANALYZE_LEGACY_PASS_MERGE=1` restores legacy merged-pass analyze |

**SPP is collection + post-analysis.** Packing without same-pass bind still yields wrong ratios under PMC duplication; bind without packing still leaves packable metrics multi-bucket. Phase 1 ships both. Phase 2 is only SPU collectables / composites.

---

## 4. Solution

### 4.1 Phase 1 design — Single-pass packable(SPP)

#### Collection — single-pass packing

**Replace** the shipping heuristic + priority coalesce as the default allocator in
`_allocate_perfmon_counter_files`.

Algorithm (normative sketch):

1. Build unique **SPP** PMC unions (skip SPU parents).
2. Largest-first: ensure some bucket contains each union’s full set (duplicate PMCs across passes when needed).
3. Run **SPU residual fill** so residual SPU PMC pieces appear somewhere (+0 extra passes on gfx942).
4. Harden **TCC series affinity + coverage** (and ACCUM slot charging where required).

#### TCC series affinity + coverage (SPP packing harden)

TCC channel series need packing rules beyond plain PMC-union co-location. On gfx942, TCC allows **4 event bases per pass** (channel instances `[i]` are dimensions of one base, not extra slots). Full policy (approved for design): [TCC series affinity + coverage](https://github.com/ROCm/rocm-systems/blob/users/feizheng10/aiprofcomp-865-docs-backup/projects/rocprofiler-compute/docs/plans/aiprofcomp-865-tcc-series-affinity-coverage.md).

Normative sketch:

1. Pack by **series base**; when a TCC series is selected, expand **all collectable channel instances** in that pass.
2. Keep affinity pairs in the **same pass** (e.g. `TCC_EA0_RDREQ_LEVEL` with `TCC_EA0_RDREQ`, and WR/ATOMIC analogues) so latency ratios are not joined across replays — L2 channel maps can remap between passes.
3. Cover every selected series from the profile/YAML set; do **not** prune to runtime-nonzero channels.
4. Do **not** duplicate the same per-channel REQ series into a second pass with a different channel map (orphan REQ copies invite wrong same-pass bind / cross-pass joins).

**Impact:** Enforcing this on gfx942 default SPP is a **layout** harden — offline eval stays at **14** passes / `packable_multi == 0` (not a Phase 2 / SPU concern).

**Locked decisions:**

- Default path is SPP packing; optional `ROCPROF_COMPUTE_PERFMON_LEGACY_HEURISTIC=1` during migration.
- Do **not** use `WEIGHTED_AVG` for former POLICY_GAP metrics — packing covers them (they are SPP).
- Priority policy YAML is not required for the packable guarantee.
- gfx942 offline packing gates: `packable_multi == 0`, passes ≈ **14**, SPU count == **16**, SPU extra passes == **0**.

**Primary packing code:** `counter_grouping_single_pass.py`, `counter_grouping_buckets.py`, `soc_base.py`.

#### Analyze — same-pass bind (part of Single-pass packable(SPP))

When packing duplicates hub counters across passes:

1. Preserve per-pass PMC columns as `{counter}@pass:{key}` shadows at load time.
2. Bind each SPP metric’s expression to a co-located pass (`PassLayout` / `pass_provenance.py`).
3. Wire CLI (`eval_metric`) and DB (`calc_expressions` / `bind_expression_dataframe`).

Without this bind, SPP layouts can produce impossible percent averages/maxes (e.g. CPC Utilization avg ≫ 100%).

**Primary bind code:** `pass_provenance.py`, `file_io` / `utils_analysis` shadow columns, analyze CLI+DB bind paths.

### 4.2 Phase 2 design — Collectables for Single-pass unpackable(SPU)

SPU parents cannot fit one bucket even with global repack → decompose into collectables and recompose with `WEIGHTED_AVG` / `COLLECT_SUM` / `COLLECT_RATIO`.

When \(M = (A+B)/C\) is SPU:

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

---

## 5. Evaluation / Verification Plan

| Layer | Check |
|-------|--------|
| Offline packing | `eval_single_pass_packable.py` / inspector: `packable_multi == 0`, pass count, SPU residual fill +0 |
| SPP analyze (same-pass bind) | Unit tests (`test_pass_provenance`, DB bind tests); health report P0 ratios ≤ 100% where expected |
| SPU collectables | WA / `COLLECT_RATIO` unit tests; gfx942 `slot_limit_metrics == 0` after Phase 2 |
| Health report | SPP vs legacy health runners + `generate_metric_health_report.py` on gfx942 (and other arches as needed) |
| Hardware spot-check | gfx942 CPX/SPX: P0 HBM / WGM / CPC ratios sane (avg/max ≤ 100% where expected) |
| Regression workload | MI-PATH `ns3d*` (AIPROFCOMP-265 class): SPP (packing + bind) vs legacy packing / legacy pass merge |
| Multi-arch | Re-baseline gfx908, gfx90a, gfx950, gfx115x, gfx1250 offline after Phase 1 |

**Stack:** Doc → Phase 1 (SPP packing + same-pass bind) → Health utils → Phase 2 (SPU collectables / `WEIGHTED_AVG` / `COLLECT_*` only).

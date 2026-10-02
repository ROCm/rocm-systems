# gfx942 single-pass impact report (AIPROFCOMP-865)

**Date:** 2026-09-14
**Scope:** Default `rocprof-compute profile` counter set on **gfx942** (all analysis blocks except `3000` membw), offline via `counter_grouping_inspector.py`
**Coalesce:** Same path as profiling, including **`profiling_counter_grouping_policy.yaml`** (`same_bucket_priority_metric_ids` → `_metric_aware_coalesce_pass` in `soc_base.py`)
**Branch context:** `users/feizheng10/aiprofcomp-865-single-run-collection` (local inspector run)

This report answers:

1. With the **current 13 perfmon passes**, how many metrics are **impacted** by multi-pass PMC collection?
2. What **fraction** of that gap is addressable by **coalesce / policy / repack** vs **`WEIGHTED_AVG`** (Phase 2)?

---

## 0. Totals (read this first)

| What | Count | Notes |
|------|------:|-------|
| **Unique raw HW PMC counters** in default profile set | **274** | Distinct counter names coalesced into perfmon replays (not 274 replays) |
| **Perfmon replays (buckets)** | **13** | How those 274 counters are collected today |
| **Analysis YAML metrics** (gfx942, inspector scan) | **408** | Every metric row in arch YAML under `analysis_configs/gfx942/` |

### Decomposition of the **408** metrics

| Category | Count | % of 408 | In “% of 374” columns? |
|----------|------:|---------:|-------------------------|
| **Has ≥1 PMC in the default profile set** | **374** | 91.7% | **Yes** — denominator for single- vs multi-bucket |
| **No profile PMCs** (peaks, derived-only, no formula HW counter in this plan) | **34** | 8.3% | **No** — pass / bucket assignment N/A |

```text
408 metrics (YAML)
├── 374  use ≥1 of the 274 profile PMCs
│    ├── 283  single-bucket today (OK for same-pass ratio)
│    └──  91   multi-bucket (impacted)
│         ├── 75  packable in one bucket with better coalesce/repack
│         └── 16  cannot fit one bucket (collectables / Phase 2)
└── 34   no profile PMCs in this plan
```

---

## 1. Baseline (today)

| Quantity | Value |
|----------|------:|
| Perfmon replays (buckets) | **13** |
| Unique PMC counters in default profile set | **274** |
| Analysis YAML metrics scanned (gfx942) | **408** |
| Metrics with ≥1 formula PMC in this profile set | **374** |
| Metrics with no in-profile PMCs (peaks, derived-only, etc.) | **34** |

**“Impacted”** = metric has in-profile PMCs that map to **2+ perfmon buckets** under the current 13-pass plan (same-run ratio semantics are broken or unreliable).

| Bucket assignment (current 13 passes) | Count | % of 408 | % of 374 (with PMCs) |
|----------------------------------------|------:|---------:|---------------------:|
| **Single-bucket (OK today)** | **283** | **69.4%** | **75.7%** |
| **Multi-bucket (impacted)** | **91** | **22.3%** | **24.3%** |
| No in-profile PMCs (out of scope for pass count) | 34 | 8.3% | — |

---

## 2. Remediation split (of the 91 impacted)

For each impacted metric, all required PMCs **can** be packed into **one hardware bucket** (gfx942 `perfmon_config` slot limits), or **cannot**.

| Class | Count | % of 91 impacted | % of 374 with PMCs | Remediation |
|-------|------:|-----------------:|-------------------:|-------------|
| **A. Multi-bucket but packable** | **75** | **82.4%** | **20.1%** | Stronger **global repack** / coalesce / policy (#10912-style fixes). Greedy repack experiment: **12–13 passes** total (no need for +55 additive passes). |
| **B. Cannot fit one bucket** | **16** | **17.6%** | **4.3%** | **`WEIGHTED_AVG`** decomposition, metric redesign, or **accept multi-pass** |

**The 16 (class B)** — whole-metric single-pass is impossible at any pass count without changing what is measured:

- Examples: `VALU FLOPs`, `AI HBM` / `AI L2` / `AI L1` / `AI LDS`, `FLOPs (Total)`, roofline-style bundles with many IPs and counters.

### What “75 coalesce/repack can fix” means (and does not mean)

It does **not** mean the profiler is globally broken or that 13 passes is wrong.

- **Today’s coalesce optimizes for:** pack **all 274 counters** into **few replays** (13), sharing counters across many metrics. That is a **global** bin-packing goal.
- **Single-pass per metric** is a **different** goal: for each metric \(M\), all of \(M\)’s PMCs must lie in **one** replay.
- The **75** are metrics that are **multi-bucket under today’s packing** but whose PMC set **could** fit in **one hardware bucket** if we prioritized that metric (policy entries, metric-aware coalesce improvements such as #10912, or a full repack). Offline experiment: all **358** packable metrics can still fit in **~12–13** replays—so fixing the 75 is often **reassignment**, not necessarily **more** passes.

So the gap is often **POLICY_GAP / packing objective**, not “hardware cannot do it.” Remaining **16** are the cases where **even one bucket per metric** is impossible (true **SLOT_LIMIT**).

### Class B — existing 13 passes and per-bucket slices (offline, 2026-09-16)

For each class **B** metric, PMCs were split by **today’s perfmon bucket** (one slice per replay the parent touches). No refill, no new counters.

| Check | Result |
|-------|--------|
| Slices per metric | 2–5 (matches parent bucket span) |
| Each slice single-bucket in current plan | **Yes** (by construction) |
| Each slice fits gfx942 `perfmon_config` slots | **Yes** (0 failures on all 16) |
| Extra collection passes for slice-based submetrics | **No** — same **274** PMCs, **13** replays |

| Metric id | Metric name | #Buckets | #PMCs |
|-----------|-------------|--------:|------:|
| 0200.201.0 | VALU FLOPs | 3 | 12 |
| 0200.201.17 | vL1D Cache Hit Rate | 2 | 5 |
| 0300.301.28 | VL1 Hit | 2 | 5 |
| 0400.401.9 | HBM Bandwidth | 2 | 5 |
| 0400.402.0 | AI HBM | 5 | 22 |
| 0400.402.1 | AI L2 | 4 | 21 |
| 0400.402.2 | AI L1 | 5 | 18 |
| 0400.402.3 | AI LDS | 5 | 19 |
| 0400.402.4 | Performance (GFLOPs) | 4 | 17 |
| 1100.1101.0 | VALU FLOPs | 3 | 12 |
| 1100.1102.1 | IPC (Issued) | 4 | 9 |
| 1100.1103.0 | FLOPs (Total) | 4 | 17 |
| 1500.1504.4 | Read Instructions | 2 | 3 |
| 1600.1601.0 | Hit rate | 2 | 5 |
| 1600.1603.5 | Cache Hit Rate | 2 | 5 |
| 1600.1603.7 | Cache Hits | 2 | 5 |

**Packing ≠ algebra:** bucket-aligned slices are valid **collectable boundaries** for grouping; they are **not** automatically valid **`WEIGHTED_AVG`** terms. Each parent still needs designed submetrics and **proved weights** (below).

#### What “prove weights” means

Analyze recomposes the parent with (per dispatch \(i\)):

\[
M_i = \frac{\sum_k M_{k,i} \cdot W_{k,i}}{\sum_k W_{k,i}}
\]

where \(M_{k,i}\) is submetric \(k\)’s value and \(W_{k,i}\) comes from the YAML **`weight_counter`** for that sub (raw PMC, same dispatch row). **Proving weights** means showing that this merge **matches the intended parent definition**—not merely that submetrics are single-bucket.

| You must show | Why |
|---------------|-----|
| **Identity:** With ideal single-pass PMCs, \(M_i\) equals the parent formula (or an agreed definition). | Wrong weights re-weight ratios and bias the parent (not a silent cap—often a **wrong** number). |
| **Partition of activity:** Weights usually track **how much each submetric’s denominator contributes** to the whole. Classic split: parent \(\frac{A+B}{C}\) → \(M_0=\frac{A}{C_0}\), \(M_1=\frac{B}{C_1}\) with weights \(W_0=C_0\), \(W_1=C_1\) and **\(C_0+C_1=C\)** (same dispatch). | If \(\sum_k W_{k,i} \neq C_i\), you are not averaging the same quantity the YAML `value`/`avg` formula describes. |
| **Pilot check:** Hand-calc or exported `pmc_perf.csv` on a workload: parent `WEIGHTED_AVG` vs reference. | Phase 2 gate in `aiprofcomp-865-phase2-weighted-avg-design.md` §8. |

**Example (test pilot):** `hbm_combined_traffic = WEIGHTED_AVG(hbm_read_sub, hbm_write_sub)` with `weight_counter` = `TCC_EA0_RDREQ_sum` and `TCC_EA0_WRREQ_sum`. Read/write submetrics are **percent of DRAM traffic within each request class**; weights are **request counts**, so the parent is “DRAM bytes as % of total requests,” not a naive average of the two percents.

**Reviewer artifact:** A short note per pilot metric (worksheet row): submetric formulas, chosen `weight_counter`(s), algebra or numeric identity proof, inspector single-bucket per sub-id.

---

## 3. Cumulative “solved” by method (% of metrics with PMCs)

Denominator: **374** metrics that participate in default profiling.

| Stage | Method | Metrics fixed / OK | Cumulative OK | Cumulative % |
|-------|--------|-------------------:|--------------:|-------------:|
| 0 | **Current 13 passes + grouping policy** | 283 already single-bucket | 283 | **75.7%** |
| 1 | **+ Coalesce / repack / policy** (class A) | +75 | **358** | **95.7%** |
| 2 | **+ `WEIGHTED_AVG` or redesign** (class B) | +16 | **374** | **100%** |

As **% of all 408 YAML metrics** (including the 34 without profile PMCs):

| Stage | Cumulative OK | % of 408 |
|-------|--------------:|---------:|
| Current plan | 283 | 69.4% |
| After coalesce/repack (class A) | 358 | 87.7% |
| After Phase 2 path for class B | 374 | 91.7% |
| No profile PMCs (unchanged) | 34 | 8.3% (N/A for pass semantics) |

**Note:** Stage 2 does **not** mean “one profile replay” for class **B** on gfx942 default: submetrics can use **the same 13 replays** (§2 class B table). It means **analyze-time** reconstructs the parent via **`WEIGHTED_AVG`** with proved weights.

---

## 4. P0 golden metrics (CPX checklist)

Under **this** 13-pass plan + policy, inspector classifies **all listed P0 ids as single-bucket**:

| Metric ID | Name | Status (current plan) |
|-----------|------|------------------------|
| 3.1.63 | HBM Read Traffic (mem chart) | Single-bucket |
| 3.1.64 | HBM Write and Atomic Traffic | Single-bucket |
| 17.2.1 | HBM Read Traffic (L2 panel) | Single-bucket |
| 17.2.5 | HBM Write and Atomic Traffic | Single-bucket |
| 6.1.2 | Workgroup Manager Utilization | Single-bucket |

Phase 1 hardware worksheets still need CPX analyze numbers and post-#10912 confirmation; this table is **offline coalesce only**.

---

## 5. Methods compared (what each % means)

| Method | What it fixes | Share of **91 impacted** | Share of **374 with PMCs** |
|--------|---------------|-------------------------:|---------------------------:|
| **Already OK** (13 passes + policy as shipped) | — | — (not in the 91) | **75.7%** |
| **Coalesce / policy / repack** | Class A (75 metrics) | **82.4%** of impacted | **+20.1%** → **95.7%** total |
| **`WEIGHTED_AVG` / redesign** | Class B (16 metrics) | **17.6%** of impacted | **+4.3%** → **100%** total |
| **Accept multi-pass** | Alternative for B | — | Leaves analyze wrong for same-run ratios |

**Additive-pass models (not repack):** Keeping 13 and adding dedicated replays only for packable unions is **~+55 passes (model B)** or **+91 (model A)**; **full repack** targets **~12–13 passes** for all 358 packable metrics. See Phase 1 plan discussion / chat notes.

---

## 6. How to reproduce

```bash
cd projects/rocprofiler-compute
PYTHONPATH=src python3 tools/counter_grouping_inspector.py --arch gfx942 -o /tmp/gfx942-plan.txt
# Summary line: 13 bucket(s); multi-bucket: 91 of 408 (22.3%)
```

Classification script (packable vs slot-limited): same logic as `CounterFile` packing used in `soc_base.py` (see agent session 2026-09-14).

---

## 7. Limitations

- Offline only; no GPU; SQG omitted on some Conductor paths (gfx942).
- Default profile excludes block `3000` unless membw analysis.
- **INTENTIONAL** \>100% metrics (e.g. VALU dual-issue) are a separate worksheet verdict, not counted in the 16 unless they are also slot-limited whole-metric packs.
- Inspector YAML scan is full gfx942; counter **plan** is default profile set.

---

## 8. One-paragraph summary

With **13 perfmon passes** and **forced grouping policy**, **91 of 408** metrics (**24.3%** of those with profile PMCs) still see formula counters split across buckets. **82.4%** of that impacted set (**75** metrics) can be fixed by **better coalesce/repack** without `WEIGHTED_AVG`, likely within **~13 passes**. The remaining **17.6%** (**16** metrics) cannot fit one hardware bucket and are the natural **`WEIGHTED_AVG` / redesign** set (**4.3%** of PMC-bearing metrics). **P0 golden HBM/WGM ids are single-bucket** in this offline plan; Phase 1 sign-off still requires hardware + #10912 alignment.

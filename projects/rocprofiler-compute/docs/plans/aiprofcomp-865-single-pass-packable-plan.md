# AIPROFCOMP-865 — Single-pass packable plan (replace shipping heuristic)

**Status:** Draft plan (experimental path exists behind env flag)
**JIRA:** AIPROFCOMP-865 (parent AIPROFCOMP-864)
**Related:**
- [problem-decompose HTML](aiprofcomp-865-problem-decompose.html)
- [Phase 2 WEIGHTED_AVG design](aiprofcomp-865-phase2-weighted-avg-design.md)
- [impact report](aiprofcomp-865-gfx942-single-pass-impact-report.md)
- [collectables scope](aiprofcomp-865-collectables-scope.md)

**Supersedes (default packing path):** “Fewer passes — heuristic + prioritized grouping (shipping)” as the production allocator.

---

## 1. Goal

Make **single-pass collection the default packing objective** for every metric whose PMC set fits one hardware bucket, then finish residual parents with Phase 2 `WEIGHTED_AVG`.

| Phase | What | gfx942 target |
|-------|------|----------------|
| **Phase 1** | Ship **Single-pass packable** as default; remove shipping heuristic + priority coalesce from the default path | **358** / 374 PMC metrics single-pass · **14** passes · SLOT fill **+0** |
| **Phase 2** | `WEIGHTED_AVG` collectables for **16** `SLOT_LIMIT` parents (**10** unique PMC sets) | Parent-level accuracy without one bucket for the full PMC set |
| **Validation** | Re-baseline + hardware on other arches | gfx908, gfx90a, gfx950, gfx115x, gfx1250 |
| **Cleanup** | Delete heuristic + prioritized grouping; update inspector / docs | One packing story in code and tools |

```
408 YAML metrics (gfx942 default)
├── 374 with profile PMCs
│   ├── 358 single-pass (all packable)     ← Phase 1, 14 passes
│   └── 16 SLOT_LIMIT                      ← Phase 2 WEIGHTED_AVG
└── 34 with no profile PMCs                ← out of packing scope
```

Compare shipping today: **12** passes, **299** single-pass, **59** POLICY_GAP leftovers, **16** SLOT_LIMIT.

### 1.1 Live offline baseline (2026-09-28, `eval_single_pass_packable.py --arch gfx942`)

| | Shipping (heuristic+priority+refill) | Single-pass packable + SLOT fill |
|--|--------------------------------------|----------------------------------|
| Passes | 12 | **14** (+2) |
| PMC total | 274 | 274 |
| Packable metrics (fit one bucket) | 368 | 368 |
| Packable unions lacking full bucket | 45 | **0** |
| SLOT_LIMIT metrics / unique unions | — | 16 / 10 |
| Extra passes for SLOT_LIMIT PMC presence | — | **+0** (all SLOT PMCs already in layout) |

**Number reconcile (Phase 1 task P1-0):** Product narrative uses **358** = 374 − 16 (impact report / HTML). Eval currently reports **368** packable metrics. Before cutover, lock one definition (same `detect_counters` + YAML scan as inspector) and update HTML/plan/acceptance to match.

---

## 2. Decisions (locked for this plan)

1. **Replace**, do not dual-run: Single-pass packable becomes the only default path in `_allocate_perfmon_counter_files`. No env flag required for production once Phase 1 lands (optional `LEGACY_HEURISTIC=1` only during migration if needed).
2. **Allow PMC duplication across passes** when packable unions conflict — required for the 358 guarantee; differs from shipping’s unique-assignment rule.
3. **Priority policy YAML is not required** for the 358 guarantee; optional only as tie-break for *which* metric gets a duplicate when unions conflict (defer unless multi-arch shows need).
4. **Do not** use `WEIGHTED_AVG` for the former 59 POLICY_GAP metrics — Phase 1 packing covers them.
5. **Refill** (`apply_metric_coalesce_refill_pass`) is obsolete once Single-pass packable is default — remove from allocate after cutover.
6. **SLOT_LIMIT fill** must run inside the production allocate path (today it exists only in `tools/eval_single_pass_packable.py`).

---

## 3. Phase 1 — Implement Single-pass packable (default)

### 3.1 Objectives

| # | Objective | Done when |
|---|-----------|-----------|
| P1-0 | Reconcile 358 vs 368 packable count | One locked number in plan + HTML + eval asserts |
| P1-1 | Default allocator = SPP + SLOT_LIMIT fill | No env gate for default profile |
| P1-2 | gfx942 offline: all packable metrics have a bucket containing their full PMC set | `packable_multi == 0` via eval / inspector |
| P1-3 | gfx942 pass count | **14** (or updated if merge improves); vs shipping 12 |
| P1-4 | SLOT_LIMIT fill in allocate | gfx942 **+0** extra passes; all SLOT PMCs present somewhere |
| P1-5 | Unit + integration tests | CI green |
| P1-6 | Heuristic off default path | `_metric_aware_coalesce_pass` not called in production allocate |
| P1-7 | Hardware spot-check gfx942 (CPX + SPX) | P0 HBM ratios sane; pass count matches offline |

### 3.2 Algorithm (normative)

Already prototyped in `src/rocprof_compute_soc/counter_grouping_single_pass.py`:

```
Profile PMC set
    │
    ▼
Unique packable PMC unions (skip SLOT_LIMIT parents)
    │  largest-first
    ▼
For each union: ensure some bucket has full set
    │  extend existing (best overlap) else open new (may duplicate PMCs)
    ▼
First-fit still-unplaced counters (TCC channel map preserved)
    │
    ▼
Merge bucket pairs while union fits AND packable_multi stays 0
    │
    ▼
SLOT_LIMIT fill: place missing SLOT PMCs into existing buckets
    │  open new only if a PMC cannot fit any existing bucket
    ▼
pmc_perf_*.txt  (gfx942: 14 passes, +0 for SLOT fill)
```

| Step | Function | Notes |
|------|----------|--------|
| Collect packable unions | `collect_unique_packable_unions` | Metric PMC ∩ profile; `counters_fit_one_bucket` |
| Ensure full bucket | `_ensure_packable_union` | Additive duplication when unions conflict |
| Leftovers | `_first_fit_unplaced` | Same TCC channel rules as shipping |
| Reduce passes | `_reduce_passes` | Pairwise merge; reject if any packable union loses coverage |
| SLOT fill | `fill_slot_limit_into_existing_passes` | **Wire into `try_allocate_single_pass_packable`** (today eval-only) |

**Out of Phase 1:** forcing SLOT_LIMIT *parents* into one bucket (impossible by definition).

### 3.3 Code changes (file-level)

| File | Action |
|------|--------|
| `src/rocprof_compute_soc/counter_grouping_single_pass.py` | (1) Call `fill_slot_limit_into_existing_passes` at end of `try_allocate_single_pass_packable` before success return. (2) Invert env: default **on**; `ROCPROF_COMPUTE_PERFMON_LEGACY_HEURISTIC=1` restores old path if kept. (3) Update module docstring (no longer “experiment”). Extend `SinglePassPackableStats` with SLOT fill fields. |
| `src/rocprof_compute_soc/soc_base.py` | `_allocate_perfmon_counter_files`: SPP is default; skip `_metric_aware_coalesce_pass`, CP-SAT, and refill when SPP succeeds. Update docstring. Optional: delete CP-SAT from default story (leave behind flag). |
| `src/rocprof_compute_soc/counter_grouping_refill.py` | Stop calling from allocate; keep shared helpers (`_iter_metric_groups`, `counters_fit_one_bucket`, `rebuild_counter_file`) used by SPP / inspector. |
| `…/profiling_counter_grouping_policy.yaml` | Stop using `same_bucket_priority_metric_ids` for packing steer in Phase 1; deprecate until cleanup deletes. |
| `tests/unit/rocprof_compute_soc/test_counter_grouping_single_pass.py` | Default-on; assert duplication allowed; assert SLOT fill integrated; assert `packable_multi==0` on fixture. Update env tests for inverted gate. |
| Tests that assume 12-pass / priority coalesce | Update expected pass counts / bucket layouts (grep for coalesce / priority / refill). |
| `tools/eval_single_pass_packable.py` | Assert gfx942 gates: `packable_multi==0`, passes==14 (or locked budget), SLOT `additional_passes==0`. Add `--arch` matrix mode for §5. |
| `tools/build_aiprofcomp865_problem_decompose_html.py` | Plan tab: mark SPP as **shipping path** (post cutover); shipping heuristic tab → historical. |

### 3.4 Concrete cutover patch (order of operations)

1. **Integrate SLOT fill** into `try_allocate_single_pass_packable` (behavior match eval today).
2. **Flip default:** `single_pass_packable_enabled_from_env()` → True unless `LEGACY_HEURISTIC=1` (or remove gate entirely and branch legacy only when env set).
3. **Tests** for gfx942 offline numbers + unit fixtures.
4. **Multi-arch offline sweep** (§5) before deleting heuristic code.
5. **HW spot-check** gfx942.
6. Cleanup milestone (§6) after sign-off.

### 3.5 gfx942 acceptance numbers (offline)

| Metric | Shipping (heuristic+refill) | Phase 1 plan |
|--------|----------------------------|--------------|
| Passes | 12 | **14** (+2) |
| Single-pass (product: 374−16) | 299 | **358** (reconcile vs eval 368 — P1-0) |
| Packable unions lacking full bucket | 45 (eval) | **0** |
| POLICY_GAP leftovers | 59 | **0** |
| SLOT_LIMIT parents | 16 | 16 (unchanged; Phase 2) |
| Extra passes for SLOT_LIMIT PMC presence | — | **+0** |

### 3.6 Risks / mitigations

| Risk | Mitigation |
|------|------------|
| +2 passes increases profile time | Document; compare wall-clock on CPX; optional later merge/CP-SAT to claw back passes without breaking packable coverage |
| Duplicate PMCs change analyze assumptions | Audit analyze joins by counter name across files; add test if any path assumes uniqueness |
| Filtered `--block` / `--set` profiles | Allocate only on detected counters; re-eval inspector on subset profiles |
| gfx1250 anti-affinity (VALU vs VMEM) | Multi-arch validation (§5); may need `never_same_bucket` constraints later |
| SLOT fill not in allocate today | P1-1 wires it; regression if eval and allocate diverge |

### 3.7 Phase 1 exit criteria

- [ ] P1-0 number locked (358 vs 368) and reflected in HTML/eval asserts.
- [ ] Default `profile` on gfx942 uses Single-pass packable + SLOT fill (no env).
- [ ] Offline: `packable_multi == 0`; SLOT additional passes == 0; passes ≤ 14 (or justified).
- [ ] Problem-decompose HTML “Single-pass packable plan” tab matches shipped behavior.
- [ ] Shipping heuristic not on default path.
- [ ] Unit tests updated; CI green.

---

## 4. Phase 2 — `WEIGHTED_AVG` for 16 SLOT_LIMIT

Follow [aiprofcomp-865-phase2-weighted-avg-design.md](aiprofcomp-865-phase2-weighted-avg-design.md). Milestone A (parser / aggregation / analyze wiring) is already on branch; this plan covers **YAML pilots + remaining unique sets** under the new SPP packing world.

**Important:** Phase 2 design doc still says “Phase 1 = metric-aware coalescing.” Update that preamble when Phase 1 lands: Phase 1 = Single-pass packable; Phase 2 = SLOT_LIMIT only.

### 4.1 Unique HW problems ≈ 10 (not 16)

Panel mirrors share PMC sets — implement once per unique set.

| # | Unique PMC set (size) | Parent ids | Phase 2 approach |
|---|----------------------|------------|------------------|
| 1 | VALU FLOPs (12) | `0200.201.0` ≡ `1100.1101.0` | Compose from F16/F32/F64 pieces already single-pass under SPP |
| 2 | vL1D hit (5) | `0200.201.17` ≡ `0300.301.28` ≡ `1600.1601.0` ≡ `1600.1603.5` ≡ `1600.1603.7` | One submetric design; 5 panel mirrors |
| 3 | HBM Bandwidth (5) | `0400.401.9` | Fabric Rd/Wr pieces + proved weights |
| 4 | AI HBM (22) | `0400.402.0` | Large composite; needs proved weights |
| 5 | AI L2 (21) | `0400.402.1` | Same |
| 6 | AI L1 (18) | `0400.402.2` | Same |
| 7 | AI LDS (19) | `0400.402.3` | Same |
| 8 | Perf GFLOPs (17) | `0400.402.4` ≡ `1100.1103.0` | F16/F32/F64 OPs + MFMA pieces |
| 9 | IPC Issued (9) | `1100.1102.1` | **Partial** reuse; unique PMCs: `SQ_INSTS_SENDMSG`, `SQ_INSTS_VSKIPPED` |
| 10 | Read Instructions (3) | `1500.1504.4` | **Partial**; unique: `TD_LOAD_WAVEFRONT_sum` |

Derivation check (prior session): **8 / 10** unique sets are fully PMC-coverable from other single-pass metrics’ counters; **2 / 10** need unique PMCs that only appear on the SLOT parent (IPC Issued, Read Instructions) — still Phase 2 candidates, but submetric splits must include those PMCs in some bucket (already true after SLOT fill on gfx942).

### 4.2 Suggested pilot order

1. **HBM Bandwidth** (`0400.401.9`) — clearest Rd/Wr split; matches existing Phase 2 design examples.
2. **VALU FLOPs** — high visibility; pieces already packable.
3. **vL1D hit** — one design × five mirrors.
4. Remaining AI_* / GFLOPs / IPC / Read Instructions — with waivers if identity proof fails.

### 4.3 Phase 2 objectives

| # | Objective | Done when |
|---|-----------|-----------|
| P2-1 | Pilot ≥1 SLOT_LIMIT parent with sub-collectables + `WEIGHTED_AVG` | YAML + inspector: each sub single-bucket |
| P2-2 | Cover all **10** unique PMC sets (16 parent rows) | Or written waiver per unique set |
| P2-3 | No +passes for sub-collectables on gfx942 | Sub slices fit existing 14-pass plan |
| P2-4 | Inspector flags `WEIGHTED_AVG` parents / sub single-bucket | Tooling updated |

### 4.4 Phase 2 exit criteria

- [ ] All 16 SLOT_LIMIT parents either `WEIGHTED_AVG`-backed or waived.
- [ ] Analyze identity proofs for pilots documented.
- [ ] gfx942 default: no POLICY_GAP; SLOT_LIMIT handled at analyze.

---

## 5. Multi-arch validation

Run the same offline eval (and spot hardware where available) on:

| Arch | Offline allocate + inspector | Hardware note |
|------|------------------------------|---------------|
| **gfx942** | Gate (358 / 14 / +0 SLOT) | CPX + SPX Conductor |
| **gfx908** | Pass count + packable_multi=0 | If host available |
| **gfx90a** | Same | If host available |
| **gfx950** | Same (policy YAML empty today) | |
| **gfx115x** | Same (priority ids exist for other reasons) | Watch VALU / channel rules |
| **gfx1250** | Same | Watch SP_VALU vs SQ_VMEM anti-affinity comment in policy YAML |

### 5.1 Commands

```bash
# Per arch (extend script if needed for --json / exit codes)
PYTHONPATH=src:tools python3 tools/eval_single_pass_packable.py --arch gfx942
PYTHONPATH=src:tools python3 tools/eval_single_pass_packable.py --arch gfx908
PYTHONPATH=src:tools python3 tools/eval_single_pass_packable.py --arch gfx90a
PYTHONPATH=src:tools python3 tools/eval_single_pass_packable.py --arch gfx950
PYTHONPATH=src:tools python3 tools/eval_single_pass_packable.py --arch gfx115x
PYTHONPATH=src:tools python3 tools/eval_single_pass_packable.py --arch gfx1250

# Inspector (post-cutover semantics)
PYTHONPATH=src:tools python3 tools/counter_grouping_inspector.py --arch <arch> ...
```

### 5.2 Per-arch checklist

For each arch:

1. Run eval; record passes, packable covered, `packable_multi`, SLOT_LIMIT count, SLOT +passes.
2. Diff vs old heuristic+refill on that arch (pass delta, coverage delta).
3. Note policy YAML: today priority lists exist for gfx908/90a/942/115x/1250; gfx950 empty — after SPP these lists should not affect packing.
4. Fail Phase 1 cutover if any arch regresses packable coverage (`packable_multi > 0`) or blows past an agreed pass budget (set per-arch budgets after first offline sweep).

### 5.3 Deliverable table (fill during validation)

| Arch | Shipping passes | SPP passes | Packable covered | packable_multi | SLOT_LIMIT | SLOT +passes |
|------|-----------------|------------|------------------|----------------|------------|--------------|
| gfx942 | 12 | 14 | 358* | 0 | 16 | 0 |
| gfx908 | | | | | | |
| gfx90a | | | | | | |
| gfx950 | | | | | | |
| gfx115x | | | | | | |
| gfx1250 | | | | | | |

\*After P1-0 reconcile.

---

## 6. Cleanup — remove heuristic + prioritized grouping

Do **after** Phase 1 default cutover and multi-arch offline sign-off (can parallel Phase 2 pilots).

| # | Task | Detail |
|---|------|--------|
| C1 | Delete `_metric_aware_coalesce_pass` | And callers / helpers only used by coalesce |
| C2 | Remove `same_bucket_priority_metric_ids` | From `profiling_counter_grouping_policy.yaml` (or shrink file to anti-affinity only if introduced for gfx1250) |
| C3 | Remove refill call path | Delete or slim `counter_grouping_refill.py`; move shared helpers next to SPP if needed |
| C4 | Update inspector | See §6.1 |
| C5 | Update docs | HTML (shipping tab → historical), impact report, `single-run-collection-phases.md`, stakeholder Q&A, Phase 2 design preamble |
| C6 | Remove env gate | Drop `ROCPROF_COMPUTE_PERFMON_SINGLE_PASS_PACKABLE`; drop `LEGACY_HEURISTIC` after one release if kept for bisect |
| C7 | CP-SAT / three-way compare tools | Either delete, mark obsolete, or retarget vs SPP (not vs heuristic) |

### 6.1 Inspector tool updates (detail)

File: `tools/counter_grouping_inspector.py` (+ any HTML builders that embed inspector numbers).

| Change | Why |
|--------|-----|
| Coverage OK iff **some** `pmc_perf_*.txt` contains all in-profile PMCs for the metric | Matches SPP / duplication (today some paths assume unique assignment) |
| Report pass count + **duplicate-PMC** stats (counter appears in N files) | New failure mode to watch |
| Classify leftovers: `SLOT_LIMIT` vs unexpected packable gap | POLICY_GAP should be **zero** under SPP; non-zero is a bug |
| Drop “fix via priority policy / coalesce / refill” suggestions for packable leftovers | Those paths are gone |
| Keep SLOT_LIMIT packability trial; link Phase 2 `WEIGHTED_AVG` | Unchanged intent |
| Optional: flag parents with `WEIGHTED_AVG` and verify each sub-id is single-bucket | Aligns with Phase 2 design §C1 |
| Refill before/after sections | Remove or replace with “SPP allocate summary” |

### 6.2 Docs / HTML updates

| Artifact | Change |
|----------|--------|
| problem-decompose HTML | Heuristic tab → “Historical (pre-SPP)”; SPP tab → default shipping story |
| impact report | Add post-SPP appendix or mark superseded for packing method |
| Phase 2 design | Replace “Phase 1 = coalesce” with “Phase 1 = SPP” |

---

## 7. Effort estimate and sequencing

Estimates assume one engineer familiar with the packing path; calendar days include review/CI, not wall-clock coding only. Parallelism noted where useful.

| Workstream | Calendar days | Notes |
|------------|---------------|--------|
| **Phase 1** — SPP default + SLOT fill + gfx942 tests/HW | **4–6 days** | P1-0 reconcile (0.5d); wire SLOT fill + default-on + unit tests (1.5–2d); test/fixture updates (1d); gfx942 HW spot-check CPX/SPX (1–1.5d); PR polish (0.5–1d) |
| **Multi-arch validation** (§5) | **2–3 days** | Offline sweep all 5 arches (1d); budgets + any arch-specific fixes e.g. gfx1250 anti-affinity (1–2d); HW only where hosts exist |
| **Cleanup** — remove heuristic/inspector/docs | **2–3 days** | Inspector rewrite (1–1.5d); delete coalesce/refill/policy + doc/HTML (1–1.5d); after multi-arch sign-off |
| **Phase 2** — `WEIGHTED_AVG` for 16 SLOT_LIMIT | **8–12 days** | Pilot HBM (2–3d YAML+proof+analyze); VALU/vL1D (2–3d); remaining 7 unique sets + waivers (3–5d); inspector WA flags (0.5–1d). Milestone A parser already done — not in this estimate |

**Total (serial):** ~**16–24 days**. With Phase 2 pilot overlapping cleanup after Phase 1 lands: ~**14–20 days**.

```text
Days 1–5      Phase 1 (P1-0 … P1-7) — SPP default, gfx942
Days 4–7      §5 multi-arch offline (+ overlap end of Phase 1)
Days 6–9      Cleanup C1–C7 (after offline sign-off; C4 inspector early-ok)
Days 8–18     Phase 2 pilots → remaining unique SLOT sets
```

---

## 8. Open items

1. **P1-0:** Lock packable count (**358** product vs **368** eval).
2. Per-arch **pass budget** (e.g. gfx1250 may need more than +2 vs its shipping baseline).
3. Whether to keep a **legacy heuristic env** through one release for bisect.
4. Priority-policy **tie-break** for duplicate placement — needed or not after multi-arch.
5. Analyze paths that assume **unique** PMC→file mapping — audit before cutover.
6. gfx1250 **anti-affinity** — express in SPP or separate constraint.
7. Whether CP-SAT remains a debug alternative or is deleted with the heuristic.

---

## 9. References

- Experimental code: `src/rocprof_compute_soc/counter_grouping_single_pass.py`
- Allocate entry: `OmniSoC_Base._allocate_perfmon_counter_files` in `soc_base.py`
- Eval: `tools/eval_single_pass_packable.py`
- Inspector: `tools/counter_grouping_inspector.py`
- HTML plan tab: `docs/plans/aiprofcomp-865-problem-decompose.html` → **Single-pass packable plan**
- Phase 2 design: `docs/plans/aiprofcomp-865-phase2-weighted-avg-design.md`
- Policy (to deprecate): `src/rocprof_compute_soc/analysis_configs/profiling_counter_grouping_policy.yaml`

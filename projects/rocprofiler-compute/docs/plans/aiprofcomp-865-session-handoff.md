# AIPROFCOMP-865 — Session handoff (for remote / other-laptop agents)

**Generated:** 2026-09-28
**JIRA:** AIPROFCOMP-865 (parent AIPROFCOMP-864)
**Purpose:** Resume this work without the full Cursor chat. Prefer this file + the plan over the raw agent transcript.

---

## 1. Where to work

| Item | Value |
|------|--------|
| **Implementation branch** | `users/feizheng10/aiprofcomp-865-cp-sat-local` |
| **Base** | `rocprofiler-compute-develop` (rebased 2026-09-28; merge-base ≈ `338c9ff0ba`) |
| **Docs-only draft PR** | https://github.com/ROCm/rocm-systems/pull/12410 (`users/feizheng10/aiprofcomp-865-spp-plan` → develop) — **draft** |
| **Pre-rebase backup tip** | `users/feizheng10/aiprofcomp-865-cp-sat-local-pre-rebase` (local/remote safety tip; ignore unless recovering) |

```bash
git fetch origin
git checkout users/feizheng10/aiprofcomp-865-cp-sat-local
git pull --ff-only
cd projects/rocprofiler-compute   # if monorepo root is rocm-systems
```

---

## 2. Goal (one paragraph)

Replace the shipping **heuristic + prioritized grouping + refill** allocator with **Single-pass packable (SPP)** as default: every metric whose PMC set fits one hardware bucket gets a full-bucket collection (PMC duplication across passes allowed). Then Phase 2 handles **16 SLOT_LIMIT** parents via `WEIGHTED_AVG`. Multi-arch validate; delete heuristic/inspector assumptions later.

Canonical plan: [`aiprofcomp-865-single-pass-packable-plan.md`](aiprofcomp-865-single-pass-packable-plan.md)
HTML overview: [`aiprofcomp-865-problem-decompose.html`](aiprofcomp-865-problem-decompose.html)

---

## 3. Locked decisions

1. **Replace** shipping packing path (not dual-run forever). Legacy only via env during migration.
2. **Allow PMC duplication** across `pmc_perf_*.txt` when packable unions conflict.
3. Priority YAML **not required** for the packable guarantee (defer tie-break).
4. Do **not** use `WEIGHTED_AVG` for former POLICY_GAP (59) — SPP covers those.
5. **SLOT_LIMIT fill** belongs inside production allocate (not eval-only).
6. Acceptance gates (gfx942): `packable_multi == 0`, **14** passes, SLOT_LIMIT **16**, SLOT **+0** passes.
   Product narrative “358” = 374 − 16; allocator `packable_metrics` eval = **368** — different scans; do not fail on that delta (see plan P1-0).

---

## 4. Status as of handoff

### Done (on branch tip `766193e810` and earlier)

- Experimental SPP allocator module + unit tests (was env-gated)
- CP-SAT optional path, refill pass, WEIGHTED_AVG analyze Milestone A, docs/HTML builders, eval harness
- Branch rebased onto current `rocprofiler-compute-develop` (18 unique 865 commits)
- Draft docs PR #12410

### Done on branch (commit after this handoff — verify with `git log -1`)

Phase 1 cutover: SPP default **on**, SLOT fill inside allocate, eval/unit tests, plan P1-0 lock, session handoff.

| File | Change |
|------|--------|
| `src/rocprof_compute_soc/counter_grouping_single_pass.py` | Default **on**; SLOT fill inside `try_allocate_single_pass_packable`; stats extended; `LEGACY_HEURISTIC` gate |
| `src/rocprof_compute_soc/soc_base.py` | Docstring: SPP default; legacy = coalesce → first-fit → refill |
| `tests/unit/rocprof_compute_soc/test_counter_grouping_single_pass.py` | Default-on + legacy + integrated SLOT fill tests |
| `tools/eval_single_pass_packable.py` | Compare legacy vs default SPP; gfx942 gate exit code |
| `docs/plans/aiprofcomp-865-single-pass-packable-plan.md` | P1-0 number lock + day estimates |
| `docs/plans/aiprofcomp-865-session-handoff.md` | This file |

**Env contract:**

- Default: SPP + SLOT fill
- `ROCPROF_COMPUTE_PERFMON_LEGACY_HEURISTIC=1` → old path
- `ROCPROF_COMPUTE_PERFMON_SINGLE_PASS_PACKABLE=0` → also disables SPP

### Verified (Phase 1)

```text
pytest tests/unit/rocprof_compute_soc/test_counter_grouping_single_pass.py  → 12 passed
eval_single_pass_packable.py --arch gfx942 → gates PASS
  legacy: 12 passes, packable_multi=45
  SPP:    14 passes, packable_multi=0, slot_limit=16, slot_+passes=0
```

### Not done yet

- [ ] Multi-arch offline sweep (§5): gfx908, gfx90a, gfx950, gfx115x, gfx1250
- [ ] HW spot-check gfx942 CPX/SPX (laptop / Conductor testing)
- [ ] Cleanup: remove heuristic/refill/priority from default story; update inspector
- [ ] Phase 2 YAML pilots for 10 unique SLOT_LIMIT sets (parser/aggregation largely already on branch)
- [ ] Promote draft docs PR #12410 or fold into implementation PR when ready

---

## 5. Effort estimates (calendar days, 1 engineer)

| Workstream | Days |
|------------|------|
| Phase 1 (remaining: review, commit, multi-arch, HW) | ~2–4 left of original 4–6 |
| Multi-arch validation | 2–3 |
| Cleanup (heuristic + inspector) | 2–3 |
| Phase 2 all 10 unique SLOT sets | 8–12 (HBM pilot alone ~2–3) |

---

## 6. Kickoff prompt (paste into new agent)

```text
Continue AIPROFCOMP-865 on branch users/feizheng10/aiprofcomp-865-cp-sat-local
(base: rocprofiler-compute-develop).

Read first:
- docs/plans/aiprofcomp-865-session-handoff.md
- docs/plans/aiprofcomp-865-single-pass-packable-plan.md

Phase 1 SPP-as-default + SLOT fill is on this branch (see handoff §4).
Do not invent WEIGHTED_AVG for POLICY_GAP.

Next priorities:
1. Multi-arch / HW testing (offline eval + Conductor)
2. Inspector cleanup only after multi-arch sign-off
3. Phase 2 starts with HBM Bandwidth SLOT_LIMIT pilot

Draft docs PR: https://github.com/ROCm/rocm-systems/pull/12410
```

---

## 7. Commands

```bash
# Unit tests
PYTHONPATH=src python3 -m pytest \
  tests/unit/rocprof_compute_soc/test_counter_grouping_single_pass.py -q

# Offline gate (gfx942)
PYTHONPATH=src:tools python3 tools/eval_single_pass_packable.py --arch gfx942

# Other arches
for a in gfx908 gfx90a gfx950 gfx115x gfx1250; do
  PYTHONPATH=src:tools python3 tools/eval_single_pass_packable.py --arch "$a"
done
```

---

## 8. Key code entry points

| Role | Path |
|------|------|
| Allocate entry | `src/rocprof_compute_soc/soc_base.py` → `_allocate_perfmon_counter_files` |
| SPP + SLOT fill | `src/rocprof_compute_soc/counter_grouping_single_pass.py` → `try_allocate_single_pass_packable` |
| Legacy coalesce | `soc_base._metric_aware_coalesce_pass` (legacy path only) |
| Refill helpers | `src/rocprof_compute_soc/counter_grouping_refill.py` |
| Inspector | `tools/counter_grouping_inspector.py` |
| Phase 2 design | `docs/plans/aiprofcomp-865-phase2-weighted-avg-design.md` |

---

## 9. Do not do

- Do not open a PR of the old unrebased tip (use current branch after 2026-09-28 rebase).
- Do not enable `WEIGHTED_AVG` for packable / former POLICY_GAP metrics.
- Do not force SLOT_LIMIT **parents** into one bucket in Phase 1.
- Do not assume Cursor chat sync — this handoff + git are the portable context.
- Raw transcript (`~/.cursor/projects/.../agent-transcripts/29512118-...jsonl`, ~3.3 MB) is optional local archaeology only; **not** required for remote agents.

---

## 10. Related artifacts

- Plan: [`aiprofcomp-865-single-pass-packable-plan.md`](aiprofcomp-865-single-pass-packable-plan.md)
- HTML: [`aiprofcomp-865-problem-decompose.html`](aiprofcomp-865-problem-decompose.html)
- Impact report: [`aiprofcomp-865-gfx942-single-pass-impact-report.md`](aiprofcomp-865-gfx942-single-pass-impact-report.md)
- Phase 2 WA: [`aiprofcomp-865-phase2-weighted-avg-design.md`](aiprofcomp-865-phase2-weighted-avg-design.md)

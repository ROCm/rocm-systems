## Context

rocprofiler-compute already exposes memory-path metrics and mem-bandwidth analysis (including gfx950 YAML such as `3000_mem_bw.yaml`), an ASCII **gfx9** memory chart for CDNA-class GPUs, and a **gfx115x** Rich-based memory chart path. Kernel authors and performance engineers need **prioritized bottleneck pointers** with **optional hints**, while **Optiq** needs a **stable topology + bindings** contract and metric streams. Prior discussion fixed **Q8 Option A** (category-native impact, no single forced “pain scalar” as source of truth) and **Q9 Option 1** (fixed exclusion trees / display families in YAML). Thresholds remain **cheap, explainable highlights**; **confidence** and **explicit uncertainty** product language accompany rankings.

## Goals / Non-Goals

**Goals:**

- Ship a **backend data contract** (serialization shape) for bottleneck **candidates**, native **impact** per category, **confidence**, **display_family** resolution, and **threshold highlights**—usable by CLI, TUI, and Optiq without forked logic.
- **gfx950-only** v1 rules and copy (engineering-owned YAML).
- CLI/TUI: **Pattern D** (BW emphasis on key **edges**) + **compressed Pattern A** (latency / hit-util / stall slots on **nodes**) plus **post-chart** numeric table and **hints** for surfaced reasons.
- Joint **topology + bindings JSON** with Optiq: versioned, arch-tagged, stable node/edge IDs; metric slots for block vs link.
- **Reference test cases** (workloads + expected invariants / acceptable top sets) to support **user study** as success metric.

**Non-Goals:**

- Single canonical “omniscient” bottleneck score as the only stored impact (normalized sort keys may exist only as **derived**, optional fields).
- General **causality graph** engine (v1 uses **fixed trees** only).
- **i18n** product framework (strings are engineering YAML for v1).
- **Smoothing / hysteresis** across runs (user controls aggregation over dispatches).
- Proving completeness of hard thresholds for all GPU corner cases.

## Decisions

1. **Category-native impact (Q8A)**  
   Each candidate stores `impact_kind` ∈ {`bw`,`latency`,`hit_util`,`stall`}, `impact_value` (float), and `unit`. Optional `impact_norm` MAY be computed solely for cross-category merge/sort, MUST remain derivable from native fields in logs.

2. **Fixed exclusion trees (Q9 / Option 1)**  
   Each candidate belongs to a `display_family_id`. YAML defines an **ordered** winner rule (typically **most specific true leaf** first, e.g. TCP UTCL2 branch before UTCL1 before “other”). At most **one displayed winner** per family per aggregation scope.

3. **Priority score (conceptual)**  
   `rank_score ≈ impact × confidence × (1 - redundancy)` with redundancy implemented by **family winner + tree rules**, not dynamic graph inference.

4. **Uncertainty language**  
   Expose states such as **primary (high confidence)**, **possible contributor**, **insufficient data** when inputs missing (v1 may assume full counters when profiling complete; state still required for partial future captures).

5. **Thresholds**  
   Remain in perf analysis YAML (or per-arch tuning tables). They gate **labels / highlights**, not the sole existence of a candidate in the raw candidate list where continuous severity is still stored.

6. **Presentation split**  
   Backend emits **structured records**; CLI/TUI renders ASCII + tables; Optiq consumes **same records** + **topology JSON**. Borrow **gfx115x-style** presentation patterns (layout, grouping, Rich affordances) in TUI where stack permits without merging gfx9 and gfx11 code blindly.

7. **Hints**  
   Engineering-owned strings keyed by stable `reason_id` / `metric_id`; optional in product (“plus”).

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| ASCII chart overcrowding | Cap lines per block; truncate; push detail to post-chart table |
| Drift between topology JSON and Python diagram | Joint schema ownership; version field; codegen or shared metadata long-term |
| Over-confident “primary bottleneck” in user study | Copy framed as hypothesis; show confidence + native impact |
| Duplicate metrics in DB | Family winner applied before top-N export to Optiq |

## Migration Plan

1. Publish **JSON schema draft** with Optiq (topology + bindings); bump when breaking.  
2. Add backend **candidate export** behind feature flag or arch gate (`gfx950`).  
3. Wire CLI/TUI to consume export; keep legacy chart path until parity reviewed.  
4. Add **reference test cases** in repo or internal suite; run in CI where feasible.  
5. Run **user study** on frozen build; iterate thresholds/trees in YAML.

## Open Questions

- Exact **merge policy** for cross-category Top-N (caps, tie order) pending perf team review.  
- Whether **optional** `impact_norm` formulas live in YAML or code for v1.  
- Minimal **topology** node set for gfx950 vs full CDNA diagram Optiq expects.  
- Whether SQLite needs **new tables/columns** or metric rows + JSON sidecar suffice for Optiq import.

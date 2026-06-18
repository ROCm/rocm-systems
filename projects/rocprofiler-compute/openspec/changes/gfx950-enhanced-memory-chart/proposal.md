## Why

Kernel authors and performance engineers need a **structured, scannable** view of memory-path behavior on **gfx950** that goes beyond raw metrics: prioritized bottleneck hypotheses, **category-native** severity, and optional hints—without pretending fixed thresholds capture every GPU corner case. Today, memory insight is spread across tables, the ASCII **gfx9** memory chart, and mem-bandwidth analysis rules; we lack a **single backend contract** and a **CLI/TUI presentation** aligned with a future **Optiq** topology while **borrowing richer front-end patterns from gfx115x**.

## What Changes

- Define a **backend-first data model** for “enhanced memory” insight: four analysis lenses (**BW**, **latency**, **hit/util**, **stall**), **category-native impact** values (Q8: Option A), **confidence**, and **YAML-owned** display strings (including generic hints as a plus).
- Introduce **display families** with **fixed exclusion trees** (Q9: Option 1) so redundant stall/bottleneck flags collapse to **at most one winner per family** for ranking and UI.
- Specify **Pattern D + compressed Pattern A** for CLI/TUI: **BW on key edges**, **latency / hit-util / stall** as compact **per-node** slots on the diagram where feasible, plus a **post-chart** section for **full numbers and hints** (narrow-terminal friendly).
- **gfx950-only** analysis and prioritization rules in v1; chart presentation **reuses or adapts gfx115x-style memory chart UX** where the stack allows, without coupling backend to a single front-end.
- Prepare **companion payloads** for Optiq: (1) **per-kernel / per-dispatch metric values** for bottleneck booleans and native impacts; (2) **joint topology + bindings JSON** with Optiq (blocks, edges, metric-to-node/edge mapping)—rocprofiler-compute co-owns schema review, not a unilateral dump.
- **Validation**: documented **test workloads + reference expectations** (acceptable top-N sets, family-winner invariants—not claimed omniscient “ground truth”) to support a planned **user study** as the v1 success criterion.

## Capabilities

### New Capabilities

- `memory-insight-backend`: Contract for bottleneck **candidates**, **category-native impact**, **confidence**, **display_family** tree resolution, optional **rank_score** derivation for merged lists, and serialization for analysis DB / export—**independent of CLI/TUI/Optiq rendering**.
- `memory-chart-cli-tui`: CLI/TUI behavior for gfx950 enhanced memory presentation (diagram + post-chart table + uncertainty language), reusing gfx115x-style patterns where applicable.
- `memory-topology-optiq`: Joint **topology + bindings** JSON with Optiq (node/edge IDs, connections, metric slots per block/edge); versioning and arch metadata.

### Modified Capabilities

- _(none — no existing `openspec/specs/` requirements in-repo yet)_

## Impact

- **Analysis**: `gfx950` mem-bandwidth YAML and related analysis pipelines; possible new derived metrics or structured export alongside existing `Metric`/`Value` flows.
- **Visualization**: `mem_chart_gfx9` / TUI `MemoryChart` / `tty.py` formatting paths; alignment or reuse with **gfx115x** memory chart modules where feasible.
- **Data consumers**: Optiq (layout, column width, formatting agreements); analysis SQLite schema only if new columns/types are strictly required (prefer additive JSON or metric rows first).
- **Process**: coordination with Optiq on JSON schema; engineering-owned copy in YAML; user-study protocol and golden-ish test cases.

# Skill taxonomy & capability tree

Discovery for **all agents**: visual tree, **category IDs**, composition hints, and navigation rules. Flat index: [`.ai/skills/index.md`](../skills/index.md).

## Capability tree

```text
rocprofiler-compute
├── product / cli          [P]
│   ├── add_feature.md
│   └── add_experimental_cli.md
├── domain / gpu & metrics [D]
│   └── update_soc_or_counters.md
├── analysis               [A]
│   └── analyze_or_roofline.md
├── execution graph & trace [G]
│   └── execution_graph_trace.md
├── platform / native      [N]
│   ├── native_or_cmake.md
│   └── fix_build_failure.md
├── quality                [Q]
│   ├── fix_bug.md
│   └── write_test.md
└── performance            [Perf]
    └── optimize_performance.md
```

## Category IDs (formal)

| ID | Category | Typical concerns | Skills |
|----|----------|------------------|--------|
| `P` | Product / CLI | argparse, experimental gating | `add_feature`, `add_experimental_cli` |
| `D` | Domain / GPU & metrics | SoC, counter YAML | `update_soc_or_counters` |
| `A` | Analysis | roofline, stats, analyze paths | `analyze_or_roofline` |
| `G` | Execution graph & trace | kernels, markers, CSV shape | `execution_graph_trace` |
| `N` | Native / build | C++, CMake, rocprofiler-sdk, **broken build** | `native_or_cmake`, `fix_build_failure` |
| `Q` | Quality | bugs, tests | `fix_bug`, `write_test` |
| `Perf` | Performance | hot paths, allocations | `optimize_performance` |

## Navigation rules

1. Pick the **deepest** match (e.g. trace columns → `G` before `A`).
2. If two apply, use **primary outcome** (correctness → `Q`; speed → `Perf`).
3. **CMake/build failure** → `fix_build_failure` before `native_or_cmake` unless doing new native **feature** work.
4. Use [`.ai/skills/index.md`](../skills/index.md) for phrase → file map.

## Composition (manual DAGs)

Chain skills without an automated runner (see [`.ai/ROADMAP.md`](../ROADMAP.md) for future DAG tooling):

| Task shape | Typical chain |
|------------|----------------|
| Bugfix + regression | `fix_bug` → `write_test` |
| Feature + CLI | `add_feature` or `add_experimental_cli` → `write_test` |
| SoC change | `update_soc_or_counters` → `write_test` |
| Trace / parser | `execution_graph_trace` → `write_test` |
| Speedups | `fix_bug` → `optimize_performance` or `optimize_performance` alone |
| Native feature | `native_or_cmake` → smoke |
| Build broken | `fix_build_failure` → `native_or_cmake` if needed |

Always apply [`.ai/rules/core.md`](../rules/core.md) and relevant standards first.

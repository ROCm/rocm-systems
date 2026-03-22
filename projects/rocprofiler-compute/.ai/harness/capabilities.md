# Capability tree → skills (AgentSkillOS-style)

Hierarchical map for **skill discovery** (all agents). Leaf nodes are playbooks under `.ai/skills/`. **Formal taxonomy (IDs):** [skill_taxonomy.md](skill_taxonomy.md). Flat index: [`.ai/skills/index.md`](../skills/index.md).

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
│   └── execution_graph_trace.md   # kernels, markers, CSV/trace shape
├── platform / native      [N]
│   └── native_or_cmake.md
├── quality                [Q]
│   ├── fix_bug.md
│   └── write_test.md
└── performance            [Perf]
    └── optimize_performance.md
```

## Composition (small DAGs)

Multi-step tasks should chain **existing** skills without new architecture (automated DAG runner is **reserved**: [future.md](future.md)):

| Task shape | Typical chain |
|------------|----------------|
| Bugfix with regression | `fix_bug` → `write_test` |
| Feature with CLI | `add_feature` or `add_experimental_cli` → `write_test` |
| SoC change | `update_soc_or_counters` → `write_test` |
| Trace / column / parser change | `execution_graph_trace` → `write_test` |
| Speedups after correctness | `fix_bug` → `optimize_performance` (or `optimize_performance` alone if behavior frozen) |
| Native change | `native_or_cmake` → integration smoke (manual or CI) |

Always apply [`.ai/rules/core.md`](../rules/core.md) and relevant **standards** before specializing.

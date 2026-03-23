# Skill taxonomy (canonical tree)

Formal **classification** for `.ai/skills/*.md`. The diagram mirrors the [capability tree](capabilities.md); this file adds **stable category IDs** and **selection hints** for agents (AgentSkillOS-style retrieval).

| ID | Category | Typical concerns | Skills (leaves) |
|----|----------|------------------|-----------------|
| `P` | **Product / CLI** | User-facing behavior, argparse, experimental gating | `add_feature`, `add_experimental_cli` |
| `D` | **Domain / GPU & metrics** | SoC classes, counter YAML, hardware-specific defs | `update_soc_or_counters` |
| `A` | **Analysis** | Roofline, stats, CLI/Web analysis paths | `analyze_or_roofline` |
| `G` | **Execution graph & trace** | Kernels, markers, CSV/trace shape, dispatch narrative | `execution_graph_trace` |
| `N` | **Native / build** | C++ tool, CMake, rocprofiler-sdk, **broken configure/build** | `native_or_cmake`, `fix_build_failure` |
| `Q` | **Quality** | Bugs, tests, regressions | `fix_bug`, `write_test` |
| `Perf` | **Performance engineering** | Hot paths, complexity, allocations (no behavior change unless agreed) | `optimize_performance` |

## Navigation rules

1. Pick the **deepest matching category** (e.g. trace column semantics → `G` before `A`).
2. If two categories apply, choose **primary user outcome** (correctness → `Q`; speed → `Perf`).
3. **CMake/build failure** → `fix_build_failure` before `native_or_cmake` unless the task is new native **feature** work.
4. Use [`.ai/skills/index.md`](../skills/index.md) for a flat intent → file map.

## Relation to composition

Small chains stay manual (see [capabilities.md](capabilities.md)). **DAG orchestration, MCP, subagents** are reserved in [future.md](future.md) until implemented.

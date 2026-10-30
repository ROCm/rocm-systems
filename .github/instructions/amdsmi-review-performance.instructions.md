---
applyTo: "projects/amdsmi/**"
excludeAgent: "cloud-agent"
---

<!-- Generated from projects/amdsmi/.github/agents/amdsmi-review-performance.agent.md by projects/amdsmi/tools/update_copilot_review_instructions.py. Edit the agent and rerun the script. -->

> Copilot code review: this is the amd-smi `amdsmi-review-performance` review agent. Apply its
> review checks to changes under `projects/amdsmi/`. Its instructions about
> tools, subagents, builds and output format apply only when it runs as an
> agent.

# Performance Review — amd-smi

You review performance and efficiency for the amd-smi project.

## Project Layout

Project structure is stored in repo memories.

## High-Churn Hotspots

Key areas to watch: core C library hot paths (`src/amd_smi/`), Python ctypes overhead (`py-interface/`), CLI device iteration (`amdsmi_cli/`).

## Your Job

1. Identify performance regressions in changed code
2. Check hot paths for unnecessary allocations, copies, or syscalls
3. Flag O(n²) or worse patterns where linear would work
4. Verify resource cleanup (file handles, memory, GPU resources)
5. Check for unnecessary repeated work (regex compilation in loops, redundant device queries)
6. If CI evidence is provided, check for timing regressions

## CI Evidence (when available)

If the orchestrator provides CI run data, use it to:
- Compare **step timings** between PR run and baseline `develop` run
- Flag steps that took significantly longer (>20% regression)
- Identify **cache misses** or changed cache behavior
- Correlate timing anomalies with code changes in the diff
- Note any new resource-intensive steps added

## Severity

| Marker | Use for |
|--------|---------|
| **❌ BLOCKING** | Performance regressions, resource leaks, O(n²) in hot paths |
| **⚠️ IMPORTANT** | Suboptimal patterns that scale poorly, missing cleanup |
| **💡 SUGGESTION** | Minor optimizations, caching opportunities |
| **📋 FUTURE WORK** | Performance improvements in untouched code |

## Output

Return findings as a markdown list:

**[F-N] [Severity]: [Issue Title]** (`file:line`)
- Explanation and impact
- **Fix:** [fix] or **Option A/B** with recommendation

## Test-only rule

- In CHANGELOG.md, flag every use of the word "provide".

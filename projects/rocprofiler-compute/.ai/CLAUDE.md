# `.ai/` entry (navigation)

**Claude Code’s primary project brain is the repo-root [`CLAUDE.md`](../CLAUDE.md).** This file maps the **`.ai/`** tree.

## Quick map

| Need | Path |
|------|------|
| Layers + execution loop | [`.ai/guide/workflow.md`](guide/workflow.md) |
| Skill tree + category IDs | [`.ai/guide/taxonomy.md`](guide/taxonomy.md) |
| All agents (tool table) | Same as `workflow.md` (§ tool → entry points) |
| Rules | [`rules/core.md`](rules/core.md), [`rules/security.md`](rules/security.md) |
| Tools (bash / git / MCP) | [`rules/tools_policy.md`](rules/tools_policy.md) |
| Skill deliverables (strict) | [`standards/agent_output.md`](standards/agent_output.md) |
| Skills | [`skills/index.md`](skills/index.md) |
| Code review (shared) | [`skills/code_review.md`](skills/code_review.md) · checklist [`review/checklist.md`](review/checklist.md) |
| Future (DAG / MCP / subagents) | [`.ai/ROADMAP.md`](ROADMAP.md) |
| One-shot prompt | [`prompts/run_session.md`](prompts/run_session.md) |

## Build (this subproject)

Out-of-source CMake from **`projects/rocprofiler-compute`** (`CMakeLists.txt`, `CONTRIBUTING.md`):

```bash
cmake -S . -B build
cmake --build build
```

Set `ROCM_PATH` as needed. No in-source builds.

Do not add dependencies or break configure without maintainer alignment ([`rules/core.md`](rules/core.md)).

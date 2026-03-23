# `.ai/` entry (navigation)

**Claude Code’s primary project brain is the repo-root [`CLAUDE.md`](../CLAUDE.md)** (standard discovery path). This file exists so anyone browsing **`.ai/`** still sees how the tree fits together.

## Quick map

| Need | Path |
|------|------|
| Full context + four-layer harness | [`../CLAUDE.md`](../CLAUDE.md) |
| All agents (Codex, Cursor, Copilot, …) | [`.ai/harness/multi_model.md`](harness/multi_model.md) |
| Execution loop (step order) | [`.ai/harness/execution_flow.md`](harness/execution_flow.md) |
| Rules | [`rules/core.md`](rules/core.md), [`rules/security.md`](rules/security.md) |
| Skills | [`skills/index.md`](skills/index.md) |
| Strict deliverables for skills | [`harness/skill_output_contract.md`](harness/skill_output_contract.md) |
| One-shot combined prompt | [`prompts/run_session.md`](prompts/run_session.md) |

## Build (this subproject)

Out-of-source CMake from **`projects/rocprofiler-compute`** (see root `CMakeLists.txt` / `CONTRIBUTING.md`). Typical pattern:

```bash
cmake -S . -B build
cmake --build build
```

Set `ROCM_PATH` as needed. Do **not** use in-source builds (root `CMakeLists.txt` forbids it).

Do not add new dependencies or break default configure without maintainer alignment ([`rules/core.md`](rules/core.md)).

# Claude Code — project brain (rocprofiler-compute)

**Keep this file relatively stable** — deep detail lives under **`.ai/`**. **Guide** (workflow + taxonomy): [`.ai/guide/workflow.md`](.ai/guide/workflow.md), [`.ai/guide/taxonomy.md`](.ai/guide/taxonomy.md). **Tools policy:** [`.ai/rules/tools_policy.md`](.ai/rules/tools_policy.md). **Claude-only hooks:** `.claude/settings.json` + `.claude/hooks/`.

OpenCode may read this file; **Codex** → [`AGENTS.md`](AGENTS.md).

## Repo identity

- **Project:** ROCm Compute Profiler — `src/`, `tests/`, native helper `src/lib/`.
- **Super-repo:** `projects/rocprofiler-compute` in [rocm-systems](https://github.com/ROCm/rocm-systems). Pre-commit may assume that layout.
- **Product:** profiling / trace / counter semantics **deterministic** in tests; experimental CLI per `CONTRIBUTING.md`.

## Four layers (summary)

| Layer | Load |
|-------|------|
| **1. Context** | This file + [`.ai/rules/core.md`](.ai/rules/core.md) + [`.ai/rules/security.md`](.ai/rules/security.md) |
| **2. Skills** | [`.ai/skills/index.md`](.ai/skills/index.md), [`.ai/guide/taxonomy.md`](.ai/guide/taxonomy.md) |
| **3. Tools** | [`.ai/rules/tools_policy.md`](.ai/rules/tools_policy.md) |
| **4. Hooks** | `.claude/settings.json`; pre-commit + `ai_dev_guide.py` for everyone |

**Full loop + tool table:** [`.ai/guide/workflow.md`](.ai/guide/workflow.md).

## AgentSkillOS (optional)

*Organizing, Orchestrating, and Benchmarking Agent Skills at Ecosystem Scale* (arXiv [2603.02176](https://arxiv.org/abs/2603.02176), 2026), [AgentSkillOS](https://github.com/ynulihao/AgentSkillOS). Implemented as markdown: [`.ai/guide/taxonomy.md`](.ai/guide/taxonomy.md) + `.ai/skills/*.md`.

## Mandatory reads before editing

1. [`.ai/rules/core.md`](.ai/rules/core.md)
2. [`.ai/rules/security.md`](.ai/rules/security.md)
3. [`.ai/rules/anti_patterns.md`](.ai/rules/anti_patterns.md)
4. If traces / counters / roofline / CSV: [`.ai/rules/profiling_infra.md`](.ai/rules/profiling_infra.md)

## Standards

- [`.ai/standards/python.md`](.ai/standards/python.md), [`cpp.md`](.ai/standards/cpp.md), [`cmake.md`](.ai/standards/cmake.md)

## Prompts & review

- [`.ai/prompts/default.md`](.ai/prompts/default.md) · [`.ai/prompts/run_session.md`](.ai/prompts/run_session.md)
- Deliverables: [`.ai/standards/agent_output.md`](.ai/standards/agent_output.md)
- PR: [`.ai/review/checklist.md`](.ai/review/checklist.md)

## Claude rule injection

- [`.claude/rules/claude-guide.md`](.claude/rules/claude-guide.md)

## Humans

- [docs/AI_GUIDE.md](docs/AI_GUIDE.md) · [CONTRIBUTING.md](CONTRIBUTING.md) · [`.ai/README.md`](.ai/README.md)

## Browsing only `.ai/`?

Start [`.ai/CLAUDE.md`](.ai/CLAUDE.md).

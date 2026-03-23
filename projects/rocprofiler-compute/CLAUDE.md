# Claude Code — project brain (rocprofiler-compute)

This file is the **context layer** for Claude Code. **Skills** and shared harness docs are **tool-agnostic**: capability tree [`.ai/harness/capabilities.md`](.ai/harness/capabilities.md), tools policy [`.ai/harness/tools-policy.md`](.ai/harness/tools-policy.md), multi-model map [`.ai/harness/multi_model.md`](.ai/harness/multi_model.md), **execution loop** [`.ai/harness/execution_flow.md`](.ai/harness/execution_flow.md). Browsing only `.ai/`? Start [`.ai/CLAUDE.md`](.ai/CLAUDE.md). **Claude-only hooks** live in `.claude/settings.json` + `.claude/hooks/`.

OpenCode may also read this file; **Codex** should use [`AGENTS.md`](AGENTS.md) instead.

## Repo identity

- **Project:** ROCm Compute Profiler CLI and libraries under `src/`, tests under `tests/`, optional native helper in `src/lib/`.
- **Super-repo:** `projects/rocprofiler-compute` inside [rocm-systems](https://github.com/ROCm/rocm-systems). Pre-commit hooks may assume that layout.
- **Product constraints:** profiling / trace / counter semantics must stay **deterministic** in tests; experimental CLI behind `--experimental` per `CONTRIBUTING.md`.

## Four-layer harness (use every session)

| Layer | What to load | Path |
|-------|----------------|------|
| **1. Context** | This file + invariants | `CLAUDE.md`, `.ai/rules/core.md` |
| **2. Skills** | One primary playbook; compose with a tiny DAG if needed | `.ai/skills/index.md`, `.ai/harness/capabilities.md` |
| **3. Tools** | Bash / git / build policy | `.ai/harness/tools-policy.md` |
| **4. Hooks** | PreToolUse guard on `Bash` (+ shared pre-commit) | `.claude/settings.json`; pre-commit / `ai_dev_harness.py` for everyone |

## AgentSkillOS alignment (optional reading)

For **hierarchical skill organization** and **composition** at scale, see *Organizing, Orchestrating, and Benchmarking Agent Skills at Ecosystem Scale* (arXiv [2603.02176](https://arxiv.org/abs/2603.02176), 2026) and the [AgentSkillOS](https://github.com/ynulihao/AgentSkillOS) project. This repo implements the idea lightly via `.ai/harness/capabilities.md` + `.ai/skills/*.md`, not a separate skill runtime.

## Mandatory reads before editing

1. [`.ai/rules/core.md`](.ai/rules/core.md)
2. [`.ai/rules/security.md`](.ai/rules/security.md)
3. [`.ai/rules/anti_patterns.md`](.ai/rules/anti_patterns.md)
4. If traces, counters, roofline, or CSV shapes change: [`.ai/rules/profiling_infra.md`](.ai/rules/profiling_infra.md)

## Standards (pick by language)

- [`.ai/standards/python.md`](.ai/standards/python.md)
- [`.ai/standards/cpp.md`](.ai/standards/cpp.md)
- [`.ai/standards/cmake.md`](.ai/standards/cmake.md)

## Prompts and review

- Default user prefix: [`.ai/prompts/default.md`](.ai/prompts/default.md)
- Combined session block: [`.ai/prompts/run_session.md`](.ai/prompts/run_session.md)
- Before PR: [`.ai/review/checklist.md`](.ai/review/checklist.md)

## Claude-specific rule file

- [`.claude/rules/claude-harness.md`](.claude/rules/claude-harness.md) — condensed harness instructions for rule injection.

## Human docs

- [docs/AI_GUIDE.md](docs/AI_GUIDE.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)
- Canonical index: [`.ai/README.md`](.ai/README.md)

# AI workflow (all models)

**Context → Skills → Tools → Hooks**, then an explicit **execution loop**. Applies to Claude, Codex, Cursor, Copilot, OpenCode, etc.; only **entry files** differ per product.

## Four layers (model-agnostic)

| Layer | Purpose | What to do |
|-------|---------|------------|
| **1. Context** | Repo identity, invariants | Read your tool’s **brain file** (table below) + [`.ai/rules/core.md`](../rules/core.md) + [`.ai/rules/security.md`](../rules/security.md). |
| **2. Skills** | Task playbooks + discovery | Pick **one** primary skill from [`.ai/skills/index.md`](../skills/index.md); use [taxonomy.md](taxonomy.md) (tree + IDs). |
| **3. Tools** | Shell / git / build | [`.ai/rules/tools_policy.md`](../rules/tools_policy.md). |
| **4. Hooks** | Safety / validation | **Everyone:** [pre-commit](https://github.com/pre-commit/pre-commit) + [`scripts/ai_dev_guide.py`](../../scripts/ai_dev_guide.py). **Claude Code:** [`.claude/settings.json`](../../.claude/settings.json) `PreToolUse` → `bash_guard.py`. |

## Tool → entry points (this repo)

| Product | Context (brain) | Rules / skills pointer | Hooks (extra) |
|---------|-----------------|-------------------------|---------------|
| **Claude Code** | [`CLAUDE.md`](../../CLAUDE.md) | [`.claude/rules/claude-guide.md`](../../.claude/rules/claude-guide.md) | `.claude/settings.json` |
| **Codex / OpenCode / generic** | [`AGENTS.md`](../../AGENTS.md) | Same `.ai/` tree | pre-commit + `ai_dev_guide.py` |
| **Cursor** | `.cursor/rules/*.mdc` + optional `AGENTS.md` | [`.cursor/rules/rocprofiler-compute-ai.mdc`](../../.cursor/rules/rocprofiler-compute-ai.mdc) | pre-commit + `ai_dev_guide.py` |
| **GitHub Copilot** | [`.github/copilot-instructions.md`](../../.github/copilot-instructions.md) | Points at `.ai/` | pre-commit + `ai_dev_guide.py` |
| **Humans** | [docs/AI_GUIDE.md](../../docs/AI_GUIDE.md) | [`.ai/README.md`](../README.md) | CI + pre-commit |

**Canonical corpus:** [`.ai/rules/`](../rules/), [`.ai/skills/`](../skills/), [`.ai/prompts/`](../prompts/), [`.ai/guide/`](.) (this folder). **Skill deliverables:** [`.ai/standards/agent_output.md`](../standards/agent_output.md). **Future automation:** [`.ai/ROADMAP.md`](../ROADMAP.md).

## Execution loop (order)

1. **Identify the task** — Bug, feature, perf, trace/schema, or **build failure** ([`fix_build_failure.md`](../skills/fix_build_failure.md) when CMake/native fails).
2. **Select one primary skill** — [`.ai/skills/index.md`](../skills/index.md) phrase map; refine with [taxonomy.md](taxonomy.md).
3. **Load invariants** — Always `core.md` + `security.md`; add `anti_patterns.md`, `profiling_infra.md` when relevant.
4. **Load standards** — [`.ai/standards/`](../standards/) for languages touched.
5. **Execute** — Numbered steps in the chosen `.ai/skills/*.md`.
6. **Validate** — `ruff` / `pytest`; out-of-source CMake for native; `python3 scripts/ai_dev_guide.py`; optional `pre-commit run` (super-repo per `CONTRIBUTING.md`).
7. **Review** — [`.ai/review/checklist.md`](../review/checklist.md); fill PR template (AI + validation).

## Outputs

Meet [`.ai/standards/agent_output.md`](../standards/agent_output.md).

## Integration

| Stage | Mechanism |
|-------|-----------|
| Layout / skills integrity | `ai_dev_guide.py` + pre-commit **AI guide / layout validator** hook |
| Python style | pre-commit **ruff** |
| Merge gate | PR review + PR template |

## AgentSkillOS (optional)

Hierarchical skills + composition: arXiv [2603.02176](https://arxiv.org/abs/2603.02176), [AgentSkillOS](https://github.com/ynulihao/AgentSkillOS). Implemented here as markdown ([taxonomy.md](taxonomy.md) + skills), not a separate runtime.

## Adding a new tool

1. Add a thin entry file if required.
2. Point at `.ai/rules/`, `.ai/skills/`, this file, [taxonomy.md](taxonomy.md), [`.ai/rules/tools_policy.md`](../rules/tools_policy.md).
3. Register paths in `REQUIRED_PATHS` in `scripts/ai_dev_guide.py`.

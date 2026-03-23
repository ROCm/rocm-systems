# AI-assisted development (rocprofiler-compute)

This project uses shared **rules, standards, skills, and review checklists** so AI tools produce consistent, reviewable changes.

## Where things live

| Location | Contents |
|----------|----------|
| [`.ai/`](../.ai/README.md) | Rules (incl. [security](../.ai/rules/security.md)), standards, skills, prompts, review checklist (canonical) |
| [`.ai/guide/`](../.ai/guide/workflow.md) | **Multi-model guide:** workflow (four layers + execution loop), skill taxonomy |
| [`CONTRIBUTING.md`](../CONTRIBUTING.md) | Human workflow, experimental CLI, pre-commit |
| [`pyproject.toml`](../pyproject.toml) | Ruff and pytest configuration |

## Tool entry points (all point at `.ai/`)

| Tool | How it loads project context |
|------|------------------------------|
| Cursor | [`.cursor/rules/rocprofiler-compute-ai.mdc`](../.cursor/rules/rocprofiler-compute-ai.mdc) |
| GitHub Copilot | [`.github/copilot-instructions.md`](../.github/copilot-instructions.md) |
| Claude Code | [`CLAUDE.md`](../CLAUDE.md) |
| OpenAI Codex | [`AGENTS.md`](../AGENTS.md) (see [Codex + AGENTS.md](https://developers.openai.com/codex/guides/agents-md)) |
| OpenCode | [`AGENTS.md`](../AGENTS.md) first; [`CLAUDE.md`](../CLAUDE.md) if you rely on Claude-style discovery (see [OpenCode rules](https://open-code.ai/docs/en/rules)) |
| Other agents | [`AGENTS.md`](../AGENTS.md) |

### Multi-model guide (Context → Skills → Tools → Hooks)

All products share the same **four layers**. **Canonical** workflow and taxonomy: [`.ai/guide/workflow.md`](../.ai/guide/workflow.md) and [`.ai/guide/taxonomy.md`](../.ai/guide/taxonomy.md). **Tools policy:** [`.ai/rules/tools_policy.md`](../.ai/rules/tools_policy.md). **Roadmap / future automation:** [`.ai/ROADMAP.md`](../.ai/ROADMAP.md).

| Layer | Shared (every model) | Claude-only extra |
|-------|----------------------|-------------------|
| Context | Tool’s brain file (table above) + [`.ai/rules/core.md`](../.ai/rules/core.md) + [`.ai/rules/security.md`](../.ai/rules/security.md) | [`CLAUDE.md`](../CLAUDE.md) |
| Skills | [`.ai/skills/`](../.ai/skills/), [`.ai/guide/taxonomy.md`](../.ai/guide/taxonomy.md) | — |
| Tools | [`.ai/rules/tools_policy.md`](../.ai/rules/tools_policy.md) | — |
| Hooks | pre-commit + [`scripts/ai_dev_guide.py`](../scripts/ai_dev_guide.py) | [`.claude/settings.json`](../.claude/settings.json), [`bash_guard.py`](../.claude/hooks/bash_guard.py) |

**Tool → file map** is in [`.ai/guide/workflow.md`](../.ai/guide/workflow.md) (§ Tool → entry points). Claude wiring: [`.claude/README.md`](../.claude/README.md). Skill ecosystem: arXiv [2603.02176](https://arxiv.org/abs/2603.02176) (AgentSkillOS).

OpenCode can also set extra paths via **`opencode.json`** (`instructions` / project config); optional if `AGENTS.md` is enough.

## Quick start

1. Skim [`.ai/guide/workflow.md`](../.ai/guide/workflow.md) once to see how your tool maps to the four layers and the execution loop.
2. Open [`.ai/rules/core.md`](../.ai/rules/core.md) and [`.ai/rules/security.md`](../.ai/rules/security.md).
3. Paste or attach [`.ai/prompts/default.md`](../.ai/prompts/default.md) (or summarize it) in your assistant.
4. Choose a skill from [`.ai/skills/index.md`](../.ai/skills/index.md) (use [`.ai/guide/taxonomy.md`](../.ai/guide/taxonomy.md) if needed). Planned **DAG / MCP / subagent** work is in [`.ai/ROADMAP.md`](../.ai/ROADMAP.md).
5. Before opening a PR, skim [`.ai/review/checklist.md`](../.ai/review/checklist.md) and fill the AI section in the PR template. **Reviewing** someone else’s change: use [`.ai/skills/code_review.md`](../.ai/skills/code_review.md) (Claude: [`.claude/skills/code-reviewer/SKILL.md`](../.claude/skills/code-reviewer/SKILL.md)).

## Validation

- Python: `ruff check`, `ruff format`, `pytest` (see `pyproject.toml`).
- Native: out-of-source CMake build for targets under `src/lib/`.
- Hooks: pre-commit setup is described in `CONTRIBUTING.md`.
- **AI layout:** from `projects/rocprofiler-compute`, run `python3 scripts/ai_dev_guide.py` (see [`.ai/README.md`](../.ai/README.md)); pre-commit runs the same check as **AI guide / layout validator**.

For repository-wide contribution rules, see the [rocm-systems CONTRIBUTING](https://github.com/ROCm/rocm-systems/blob/develop/CONTRIBUTING.md) guide.

# AI-assisted development (rocprofiler-compute)

This project uses shared **rules, standards, skills, and review checklists** so AI tools produce consistent, reviewable changes.

## Where things live

| Location | Contents |
|----------|----------|
| [`.ai/`](../.ai/README.md) | Rules (incl. [security](../.ai/rules/security.md)), standards, skills, prompts, review checklist (canonical) |
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

### Multi-model harness (Context → Skills → Tools → Hooks)

All products share the same **four layers**. **Canonical** skills tree and tools policy: **`.ai/harness/`** (not Claude-only).

| Layer | Shared (every model) | Claude-only extra |
|-------|----------------------|-------------------|
| Context | Tool’s brain file (table above) + [`.ai/rules/core.md`](../.ai/rules/core.md) + [`.ai/rules/security.md`](../.ai/rules/security.md) | [`CLAUDE.md`](../CLAUDE.md) |
| Skills | [`.ai/skills/`](../.ai/skills/), [`.ai/harness/capabilities.md`](../.ai/harness/capabilities.md) | — |
| Tools | [`.ai/harness/tools-policy.md`](../.ai/harness/tools-policy.md) | — |
| Hooks | pre-commit + [`scripts/ai_dev_harness.py`](../scripts/ai_dev_harness.py) | [`.claude/settings.json`](../.claude/settings.json), [`bash_guard.py`](../.claude/hooks/bash_guard.py) |

**Map (which file each tool reads):** [`.ai/harness/multi_model.md`](../.ai/harness/multi_model.md). Claude wiring: [`.claude/README.md`](../.claude/README.md). Skill ecosystem: arXiv [2603.02176](https://arxiv.org/abs/2603.02176) (AgentSkillOS).

OpenCode can also set extra paths via **`opencode.json`** (`instructions` / project config); optional if `AGENTS.md` is enough.

## Quick start

1. Skim [`.ai/harness/multi_model.md`](../.ai/harness/multi_model.md) once to see how your tool maps to the four layers; follow [`.ai/harness/execution_flow.md`](../.ai/harness/execution_flow.md) for the step-by-step loop.
2. Open [`.ai/rules/core.md`](../.ai/rules/core.md) and [`.ai/rules/security.md`](../.ai/rules/security.md).
3. Paste or attach [`.ai/prompts/default.md`](../.ai/prompts/default.md) (or summarize it) in your assistant.
4. Choose a skill from [`.ai/skills/index.md`](../.ai/skills/index.md) (use [`.ai/harness/skill_taxonomy.md`](../.ai/harness/skill_taxonomy.md) or [`.ai/harness/capabilities.md`](../.ai/harness/capabilities.md) if needed). Planned **DAG / MCP / subagent** work is sketched in [`.ai/harness/future.md`](../.ai/harness/future.md).
5. Before opening a PR, skim [`.ai/review/checklist.md`](../.ai/review/checklist.md) and fill the AI section in the PR template.

## Validation

- Python: `ruff check`, `ruff format`, `pytest` (see `pyproject.toml`).
- Native: out-of-source CMake build for targets under `src/lib/`.
- Hooks: pre-commit setup is described in `CONTRIBUTING.md`.
- **AI framework:** from the project root, run `python3 scripts/ai_dev_harness.py` (see [`.ai/harness/README.md`](../.ai/harness/README.md)); included in pre-commit as **AI dev framework harness**.

For repository-wide contribution rules, see the [rocm-systems CONTRIBUTING](https://github.com/ROCm/rocm-systems/blob/develop/CONTRIBUTING.md) guide.

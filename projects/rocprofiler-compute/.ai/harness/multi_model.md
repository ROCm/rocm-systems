# Multi-model harness (Context → Skills → Tools → Hooks)

The same **four-layer** pattern applies to **any** coding agent (Claude, GPT / Codex, Gemini, Cursor, Copilot, OpenCode, etc.). Only **where** each layer is loaded differs by product.

## The four layers (model-agnostic)

| Layer | Purpose | What to do (any model) |
|-------|---------|-------------------------|
| **1. Context** | Repo identity, invariants, “start here” | Read the **context file** for your tool (see table below) + `.ai/rules/core.md` + `.ai/rules/security.md`. |
| **2. Skills** | Task playbooks + discovery | Pick **one** primary skill from [`.ai/skills/index.md`](../skills/index.md); use the [capability tree](capabilities.md) to navigate. |
| **3. Tools** | Shell / git / build expectations | Follow [tools-policy.md](tools-policy.md) for bash, git, pytest, CMake, MCP. |
| **4. Hooks** | Automated safety / validation | **Everyone:** [pre-commit](https://github.com/pre-commit/pre-commit) + [`scripts/ai_dev_harness.py`](../../scripts/ai_dev_harness.py). **Claude Code only:** [`.claude/settings.json`](../../.claude/settings.json) `PreToolUse` → `bash_guard.py`. |

## Tool → entry points (this repo)

| Product | Context (brain) | Rules / skills pointer | Hooks (extra) |
|---------|-----------------|-------------------------|---------------|
| **Claude Code** | [`CLAUDE.md`](../../CLAUDE.md) | [`.claude/rules/claude-harness.md`](../../.claude/rules/claude-harness.md) | `.claude/settings.json` |
| **Codex / OpenCode / generic agents** | [`AGENTS.md`](../../AGENTS.md) | Same `.ai/` tree | pre-commit + `ai_dev_harness.py` |
| **Cursor** | `.cursor/rules/*.mdc` + optional `AGENTS.md` | [`.cursor/rules/rocprofiler-compute-ai.mdc`](../../.cursor/rules/rocprofiler-compute-ai.mdc) | pre-commit + `ai_dev_harness.py` |
| **GitHub Copilot** | [`.github/copilot-instructions.md`](../../.github/copilot-instructions.md) | Points at `.ai/` | pre-commit + `ai_dev_harness.py` |
| **Humans** | [docs/AI_GUIDE.md](../../docs/AI_GUIDE.md) | [`.ai/README.md`](../README.md) | CI + pre-commit |

**Single source of truth** for rules, standards, skills, prompts: **`.ai/`** (especially [`.ai/rules/`](../rules/), [`.ai/skills/`](../skills/), [`.ai/prompts/default.md`](../prompts/default.md), [`.ai/prompts/run_session.md`](../prompts/run_session.md)). **Execution loop:** [execution_flow.md](execution_flow.md). **Skill deliverables:** [skill_output_contract.md](skill_output_contract.md). **Skill taxonomy (tree):** [skill_taxonomy.md](skill_taxonomy.md). **Reserved advanced templates (DAG / MCP / subagents):** [future.md](future.md).

## AgentSkillOS (optional theory)

Hierarchical skills + composed workflows match the ecosystem-scale view in *Organizing, Orchestrating, and Benchmarking Agent Skills at Ecosystem Scale* (arXiv [2603.02176](https://arxiv.org/abs/2603.02176), 2026) and [AgentSkillOS](https://github.com/ynulihao/AgentSkillOS). This repo implements that **in markdown** via [capabilities.md](capabilities.md) + `.ai/skills/*.md`, not a vendor-specific runtime.

## Adding a new tool

1. Add a **thin** project file at the path that product reads (if required).
2. Point it at **`.ai/rules/`**, **`.ai/skills/`**, [multi_model.md](multi_model.md), [capabilities.md](capabilities.md), [tools-policy.md](tools-policy.md).
3. Register any new committed paths in `REQUIRED_PATHS` in `scripts/ai_dev_harness.py`.

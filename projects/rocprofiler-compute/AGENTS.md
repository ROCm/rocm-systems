# Agent instructions (rocprofiler-compute)

Used by tools that read **`AGENTS.md`** at the project root (**Codex**, **OpenCode**, many generic agents). **Claude Code** may prefer [`CLAUDE.md`](CLAUDE.md); **Cursor** / **Copilot** use their own entry files but the **same harness**.

## Four-layer harness (any model)

| Layer | Where (shared) |
|-------|----------------|
| **1. Context** | This file + [`CLAUDE.md`](CLAUDE.md) (Claude) + [`.ai/rules/core.md`](.ai/rules/core.md) |
| **2. Skills** | [`.ai/skills/index.md`](.ai/skills/index.md) + [`.ai/harness/capabilities.md`](.ai/harness/capabilities.md) |
| **3. Tools** | [`.ai/harness/tools-policy.md`](.ai/harness/tools-policy.md) |
| **4. Hooks** | pre-commit + [`scripts/ai_dev_harness.py`](scripts/ai_dev_harness.py); Claude adds [`.claude/settings.json`](.claude/settings.json) |

**Tool → file map:** [`.ai/harness/multi_model.md`](.ai/harness/multi_model.md).

## Workflow

1. Read [`.ai/rules/core.md`](.ai/rules/core.md) and [`.ai/rules/security.md`](.ai/rules/security.md) (and [`.ai/rules/anti_patterns.md`](.ai/rules/anti_patterns.md) when editing behavior).
2. Use [`.ai/prompts/default.md`](.ai/prompts/default.md) or summarize it in your first message.
3. Pick **one** primary skill from [`.ai/skills/index.md`](.ai/skills/index.md) (use the [capability tree](.ai/harness/capabilities.md) if unsure).
4. Follow [`.ai/standards/`](.ai/standards/) for Python, C++, or CMake as applicable.
5. Observe [`.ai/harness/tools-policy.md`](.ai/harness/tools-policy.md) for shell/git/build.

**Canonical tree:** [`.ai/README.md`](.ai/README.md) · **Humans:** [docs/AI_GUIDE.md](docs/AI_GUIDE.md) · [CONTRIBUTING.md](CONTRIBUTING.md)

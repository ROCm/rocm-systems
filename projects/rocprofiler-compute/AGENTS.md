# Agent instructions (rocprofiler-compute)

Used by tools that read **`AGENTS.md`** at the project root (**Codex**, **OpenCode**, many generic agents). **Claude Code** may prefer [`CLAUDE.md`](CLAUDE.md); **Cursor** / **Copilot** use their own entry files but the **same** `.ai/` guide.

## Project context (pointers only)

Do **not** duplicate tool config here — follow the linked files.

| Topic | Where |
|-------|--------|
| **Stack** | Python (Ruff + pytest per [`pyproject.toml`](pyproject.toml)), C++/CMake under [`src/lib/`](src/lib/), app under [`src/`](src/). |
| **Layout** | Tests mirror `src/` under [`tests/`](tests/); native out-of-source CMake per [`CONTRIBUTING.md`](CONTRIBUTING.md). |
| **Style** | [`.ai/standards/python.md`](.ai/standards/python.md) → `pyproject.toml` for Ruff rules and line length. |
| **Profiling / traces / experimental CLI** | [`.ai/rules/profiling_infra.md`](.ai/rules/profiling_infra.md), [`CONTRIBUTING.md`](CONTRIBUTING.md). |
| **SoC / GPU configs** | [`tools/config_management/README.md`](tools/config_management/README.md), `src/rocprof_compute_soc/analysis_configs/`. |
| **Layering** | Profiler vs analysis packages — [`.ai/rules/core.md`](.ai/rules/core.md) (lazy imports; no new heavy analysis deps in hot profiler paths without intent). |
| **Code review** | [`.ai/skills/code_review.md`](.ai/skills/code_review.md) + [`.ai/review/checklist.md`](.ai/review/checklist.md); PRs: [`.github/pull_request_template.md`](.github/pull_request_template.md). |

## Four layers (any model)

| Layer | Where (shared) |
|-------|----------------|
| **1. Context** | This file + [`CLAUDE.md`](CLAUDE.md) (Claude) + [`.ai/rules/core.md`](.ai/rules/core.md) |
| **2. Skills** | [`.ai/skills/index.md`](.ai/skills/index.md) + [`.ai/guide/taxonomy.md`](.ai/guide/taxonomy.md) |
| **3. Tools** | [`.ai/rules/tools_policy.md`](.ai/rules/tools_policy.md) |
| **4. Hooks** | pre-commit + [`scripts/ai_dev_guide.py`](scripts/ai_dev_guide.py); Claude adds [`.claude/settings.json`](.claude/settings.json) |

**Tool → file map:** [`.ai/guide/workflow.md`](.ai/guide/workflow.md).

## Workflow

1. Read [`.ai/rules/core.md`](.ai/rules/core.md) and [`.ai/rules/security.md`](.ai/rules/security.md) (and [`.ai/rules/anti_patterns.md`](.ai/rules/anti_patterns.md) when editing behavior).
2. Use [`.ai/prompts/default.md`](.ai/prompts/default.md) or summarize it in your first message.
3. Pick **one** primary skill from [`.ai/skills/index.md`](.ai/skills/index.md) (use [`.ai/guide/taxonomy.md`](.ai/guide/taxonomy.md) if unsure).
4. Follow [`.ai/standards/`](.ai/standards/) for Python, C++, or CMake as applicable.
5. Observe [`.ai/rules/tools_policy.md`](.ai/rules/tools_policy.md) for shell/git/build.

**Canonical tree:** [`.ai/README.md`](.ai/README.md) · **Humans:** [docs/AI_GUIDE.md](docs/AI_GUIDE.md) · [CONTRIBUTING.md](CONTRIBUTING.md)

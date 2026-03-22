# AI development assets (rocprofiler-compute)

Use this tree for **consistent AI-assisted changes** across tools (Cursor, Copilot, Claude Code, Codex, OpenCode, etc.). **Codex** and **OpenCode** pick up **[`AGENTS.md`](../AGENTS.md)** at the repo root; it forwards here.

## Layout

```text
.ai/
├── README.md                 # This file — start here
├── rules/                    # Global constraints
│   ├── core.md
│   ├── anti_patterns.md
│   └── profiling_infra.md
├── standards/                # Coding + build references
│   ├── python.md
│   ├── cpp.md
│   └── cmake.md
├── skills/                   # Task playbooks (pick one per change)
│   ├── index.md
│   ├── add_feature.md
│   ├── fix_bug.md
│   ├── write_test.md
│   ├── add_experimental_cli.md
│   ├── update_soc_or_counters.md
│   ├── analyze_or_roofline.md
│   └── native_or_cmake.md
├── prompts/
│   └── default.md            # Shared prompt prefix
└── review/
    └── checklist.md          # AI-aware code review
docs/
└── AI_GUIDE.md               # Contributor entry (links here)
.github/
├── copilot-instructions.md   # GitHub Copilot → points here
└── pull_request_template.md  # Includes AI usage section
AGENTS.md                     # Codex, OpenCode, other agents → points here
CLAUDE.md                     # Claude Code; OpenCode fallback → points here
.cursor/rules/
└── rocprofiler-compute-ai.mdc # Cursor → points here
```

## How to use

| Path | Purpose |
|------|---------|
| [rules/](rules/) | Hard constraints — always follow |
| [standards/](standards/) | Style and project conventions |
| [skills/](skills/) | Task playbooks — pick one per change |
| [prompts/](prompts/) | Pasteable prompt prefix |
| [review/](review/) | PR / review checklist |

**Workflow:** read `rules/core.md` → paste or follow `prompts/default.md` → open the skill from `skills/index.md` → before PR use `review/checklist.md`.

**Humans:** [docs/AI_GUIDE.md](../docs/AI_GUIDE.md) · [CONTRIBUTING.md](../CONTRIBUTING.md)

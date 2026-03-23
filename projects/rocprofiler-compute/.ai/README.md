# AI development assets (rocprofiler-compute)

Use this tree for **consistent AI-assisted changes** across tools (Cursor, Copilot, Claude Code, Codex, OpenCode, etc.). **Codex** and **OpenCode** pick up **[`AGENTS.md`](../AGENTS.md)** at the repo root; it forwards here.

## Layout

```text
.ai/
├── README.md                 # This file — start here
├── CLAUDE.md                 # Pointer when browsing .ai/ only (root CLAUDE.md is primary)
├── rules/                    # Global constraints
│   ├── core.md
│   ├── security.md           # Prompt injection / tools / MCP risk
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
│   ├── default.md            # Shared prompt prefix
│   └── run_session.md        # One-shot: flow + rules + skill slot
├── review/
│   └── checklist.md          # AI-aware code review
└── harness/
    ├── README.md             # Harness design + validation
    ├── multi_model.md        # Four layers for every AI product
    ├── skill_taxonomy.md     # Category IDs + skill mapping
    ├── capabilities.md       # Skill tree (diagram)
    ├── future.md             # Reserved: DAG, MCP, subagents
    ├── execution_flow.md     # Prompt → skill → validate loop
    ├── skill_output_contract.md  # Strict deliverables for skills
    └── tools-policy.md       # Bash / git / build (shared)
scripts/
└── ai_dev_harness.py         # Layout / skill template checks
docs/
└── AI_GUIDE.md               # Contributor entry (links here)
.github/
├── copilot-instructions.md   # GitHub Copilot → points here
└── pull_request_template.md  # Includes AI usage section
AGENTS.md                     # Codex, OpenCode, other agents → points here
CLAUDE.md                     # Claude Code; OpenCode fallback → points here
.cursor/rules/
└── rocprofiler-compute-ai.mdc # Cursor → points here
.claude/                        # Claude Code: hooks + rule stub (shared docs in .ai/harness/)
├── README.md
├── settings.json
├── capabilities.md           # stub → .ai/harness/capabilities.md
├── tools-policy.md           # stub → .ai/harness/tools-policy.md
├── rules/claude-harness.md
└── hooks/bash_guard.py
```

## How to use

| Path | Purpose |
|------|---------|
| [rules/](rules/) | Hard constraints — always follow |
| [standards/](standards/) | Style and project conventions |
| [skills/](skills/) | Task playbooks — pick one per change |
| [prompts/](prompts/) | Pasteable prompt prefix |
| [review/](review/) | PR / review checklist |

**Workflow:** read `rules/core.md` → paste or follow `prompts/default.md` → open the skill from `skills/index.md` (see `harness/capabilities.md`) → before PR use `review/checklist.md`.

**Harness:** [harness/multi_model.md](harness/multi_model.md) (all models) · [harness/README.md](harness/README.md) (validation via `python3 scripts/ai_dev_harness.py`, pre-commit).

**Humans:** [docs/AI_GUIDE.md](../docs/AI_GUIDE.md) · [CONTRIBUTING.md](../CONTRIBUTING.md)

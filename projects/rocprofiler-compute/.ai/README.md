# AI development assets (rocprofiler-compute)

Consistent **AI-assisted** changes across Claude, Copilot, Cursor, Codex, OpenCode, etc. **Codex / OpenCode:** root [`AGENTS.md`](../AGENTS.md). **Claude:** root [`CLAUDE.md`](../CLAUDE.md).

## Layout

```text
.ai/
├── README.md              # This file
├── CLAUDE.md              # Quick map when browsing .ai/ only
├── ROADMAP.md             # Reserved: DAG, MCP, subagents
├── guide/                 # How the system runs (workflow + skill discovery)
│   ├── workflow.md        # Four layers + tool table + execution loop
│   └── taxonomy.md        # Capability tree + category IDs + composition
├── rules/
│   ├── core.md
│   ├── security.md
│   ├── tools_policy.md    # Bash / git / build / MCP
│   ├── anti_patterns.md
│   └── profiling_infra.md
├── standards/
│   ├── python.md
│   ├── cpp.md
│   ├── cmake.md
│   └── agent_output.md    # Strict skill deliverables
├── skills/                # Task playbooks (see index.md); code_review.md = shared review playbook
├── prompts/
│   ├── default.md
│   └── run_session.md
└── review/
    └── checklist.md
scripts/
└── ai_dev_guide.py      # Validates .ai/ + entry files (pre-commit + CI)
docs/
└── AI_GUIDE.md
AGENTS.md · CLAUDE.md · .cursor/rules/*.mdc · .github/copilot-instructions.md
.claude/                   # Hooks, rules, skills (e.g. code-reviewer → .ai/skills/code_review.md)
```

## Workflow (short)

1. [`.ai/guide/workflow.md`](guide/workflow.md) — layers + ordered loop.
2. [`.ai/rules/core.md`](rules/core.md) + [`security.md`](rules/security.md).
3. [`.ai/skills/index.md`](skills/index.md) + [`.ai/guide/taxonomy.md`](guide/taxonomy.md) to pick a skill.
4. Deliverables: [`.ai/standards/agent_output.md`](standards/agent_output.md).
5. Before merge: [`.ai/review/checklist.md`](review/checklist.md).

## Validation (source of truth)

`scripts/ai_dev_guide.py` **`REQUIRED_PATHS`** + [`.ai/skills/index.md`](skills/index.md) define what must exist. **Run:**

```bash
python3 scripts/ai_dev_guide.py        # from projects/rocprofiler-compute
python3 scripts/ai_dev_guide.py -v     # verbose
```

Also runs via **pre-commit** (**AI guide / layout validator** hook).

### CI (rocm-systems monorepo)

GitHub only loads workflows from the **repository root** `.github/workflows/`, not from `projects/rocprofiler-compute/.github/workflows/`. To run `ai_dev_guide.py` in CI, add a job at the **super-repo root** (paths and working directory adjusted as needed):

```yaml
jobs:
  rocprofiler-compute-ai-guide:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: projects/rocprofiler-compute
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: python3 scripts/ai_dev_guide.py
```

**Extend:** add paths to `REQUIRED_PATHS`; new skills need `index.md` links + `Goal`/`Steps`/`Constraints`/`Output` headings.

**Humans:** [docs/AI_GUIDE.md](../docs/AI_GUIDE.md) · [CONTRIBUTING.md](../CONTRIBUTING.md)

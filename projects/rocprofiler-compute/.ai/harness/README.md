# AI dev framework harness

Lightweight checks that the **`.ai/`** tree and tool entry points stay complete and skills stay structured.

**Multi-model spec:** [multi_model.md](multi_model.md) (Context → Skills → Tools → Hooks for Claude, Codex, Cursor, Copilot, OpenCode, etc.). **Skill taxonomy (tree):** [skill_taxonomy.md](skill_taxonomy.md) · **Capability diagram:** [capabilities.md](capabilities.md) · **Tools policy:** [tools-policy.md](tools-policy.md). **Advanced (reserved templates):** [future.md](future.md).

## Layers

| Layer | Purpose |
|-------|---------|
| **`scripts/ai_dev_harness.py`** | Validates required paths, index ↔ skill files, skill section headings. Exit non-zero on failure. |
| **pre-commit** | Runs the script from `projects/rocprofiler-compute` (same pattern as `hash_checker.py`). |
| **CI (optional)** | Run `python3 scripts/ai_dev_harness.py` in a lint job when `.ai/`, `scripts/`, or entry-point files change. |

## What is validated

1. **Inventory** — Expected files under `.ai/` plus `AGENTS.md`, `CLAUDE.md`, `.cursor/rules/rocprofiler-compute-ai.mdc`, `.github/copilot-instructions.md`, `docs/AI_GUIDE.md`.
2. **Skill index** — Every `*.md` link in `.ai/skills/index.md` resolves to a file in `.ai/skills/`.
3. **Skill template** — Each skill (except `index.md`) contains headings: `## Goal`, `## Steps`, `## Constraints`, `## Output`.

## Claude-only wiring

**Claude Code** adds `PreToolUse` hooks under [`.claude/`](../.claude/README.md) ([hooks docs](https://code.claude.com/docs/en/hooks)). The **same four layers** for every product are described in [multi_model.md](multi_model.md). AgentSkillOS / arXiv [2603.02176](https://arxiv.org/abs/2603.02176) inform the shared [capability tree](capabilities.md).

## Extending

- Add new required paths to `REQUIRED_PATHS` in `scripts/ai_dev_harness.py`.
- Add new skills: update `index.md` and ensure the new file includes the four `##` sections (or relax `SKILL_HEADINGS` in the script if you rename sections deliberately).

## Run manually

From this project root:

```bash
python3 scripts/ai_dev_harness.py
```

Verbose:

```bash
python3 scripts/ai_dev_harness.py -v
```

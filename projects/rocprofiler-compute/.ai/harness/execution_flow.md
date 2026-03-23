# Execution flow (Prompt → Skill → Rules → Work → Validate → Review)

This is the **explicit loop** agents and humans should follow. It connects static markdown to **what to do in order**; enforcement at **runtime** is still pre-commit / CI / Claude hooks ([multi_model.md](multi_model.md)).

## Loop

1. **Identify the task** — Bug, feature, perf, trace/schema, or **build failure** (use [`.ai/skills/fix_build_failure.md`](../skills/fix_build_failure.md) when CMake/native fails).
2. **Select one primary skill** — [`.ai/skills/index.md`](../skills/index.md) + phrase map; refine with [skill_taxonomy.md](skill_taxonomy.md) / [capabilities.md](capabilities.md).
3. **Load invariants**
   - Always: [`.ai/rules/core.md`](../rules/core.md), [`.ai/rules/security.md`](../rules/security.md)
   - If behavior/traces: [`.ai/rules/anti_patterns.md`](../rules/anti_patterns.md), [`.ai/rules/profiling_infra.md`](../rules/profiling_infra.md) as relevant
4. **Load standards** — Python / C++ / CMake under [`.ai/standards/`](../standards/) for touched languages.
5. **Execute the skill steps** — Deterministic numbered steps in the chosen `.ai/skills/*.md`.
6. **Validate**
   - Python: `ruff check`, `ruff format`, `pytest` (see `pyproject.toml`)
   - Native: out-of-source CMake configure + build for affected targets (`src/lib/`, etc.)
   - Repo: `python3 scripts/ai_dev_harness.py` (layout/skills integrity)
   - Optional: `pre-commit run` from super-repo per `CONTRIBUTING.md`
7. **Review gate** — [`.ai/review/checklist.md`](../review/checklist.md) before requesting merge; PR template AI + validation sections filled.

## Outputs

Every skill delivery must satisfy [skill_output_contract.md](skill_output_contract.md).

## Integration (not optional for contributors)

| Stage | Mechanism |
|-------|-----------|
| Layout / skill files present | `scripts/ai_dev_harness.py` (also pre-commit **AI dev framework harness**) |
| Style / Python | pre-commit **ruff** |
| Human gate | PR review + [`.github/pull_request_template.md`](../../.github/pull_request_template.md) |

**Advanced automation** (DAG runner, MCP, subagents): reserved in [future.md](future.md).

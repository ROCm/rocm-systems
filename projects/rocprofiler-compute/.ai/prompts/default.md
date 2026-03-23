# Default prompt prefix (paste or attach)

You are contributing to **rocprofiler-compute** in the ROCm rocm-systems tree (`projects/rocprofiler-compute`). This applies to **any** model (Claude, GPT/Codex, Gemini, Copilot, etc.).

Use the four layers: **Context** (rules) → **Skills** (one playbook + [`.ai/guide/taxonomy.md`](../guide/taxonomy.md) if needed) → **Tools** ([`.ai/rules/tools_policy.md`](../rules/tools_policy.md)) → **Hooks** (pre-commit / `ai_dev_guide.py`). **Ordered loop:** [`.ai/guide/workflow.md`](../guide/workflow.md). **Combined paste:** [`.ai/prompts/run_session.md`](run_session.md). **Deliverables:** [`.ai/standards/agent_output.md`](../standards/agent_output.md).

Follow:

- `.ai/rules/core.md`, `.ai/rules/security.md`, `.ai/rules/anti_patterns.md`, and `.ai/rules/profiling_infra.md` when relevant
- `.ai/standards/python.md`, `.ai/standards/cpp.md`, or `.ai/standards/cmake.md` as applicable
- One playbook from `.ai/skills/` (see `.ai/skills/index.md`)

Requirements:

- Minimal, focused change; no new dependencies unless explicitly requested
- Match existing module layout under `src/`
- Code must pass **ruff** (see `pyproject.toml`) and **pytest** for Python changes; **CMake build** for native changes
- Add or update tests when behavior changes

If anything is ambiguous, **ask** instead of guessing (APIs, trace formats, GPU details).

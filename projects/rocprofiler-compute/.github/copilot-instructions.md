# GitHub Copilot — rocprofiler-compute

Follow the **`.ai/`** tree and the **multi-model guide** (Context → Skills → Tools → Hooks). Map for all products: [`.ai/guide/workflow.md`](../.ai/guide/workflow.md).

| Layer | Path |
|-------|------|
| Context + hard constraints | `.ai/rules/core.md`, `.ai/rules/security.md`, `.ai/rules/anti_patterns.md`, `.ai/rules/profiling_infra.md` |
| Skills + taxonomy | `.ai/skills/index.md` → one skill; `.ai/guide/taxonomy.md` to pick |
| Tools (bash / git / build) | `.ai/rules/tools_policy.md` |
| Style | `.ai/standards/python.md`, `.ai/standards/cpp.md`, `.ai/standards/cmake.md` |
| Opening prompt | `.ai/prompts/default.md` |
| Validation (everyone) | pre-commit, `python3 scripts/ai_dev_guide.py` |

Validate edits with **ruff** + **pytest** (Python) or **CMake** out-of-source build (native). See `pyproject.toml` and `CONTRIBUTING.md`.

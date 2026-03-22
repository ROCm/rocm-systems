# GitHub Copilot — rocprofiler-compute

Follow the **`.ai/`** tree and the **multi-model harness** (Context → Skills → Tools → Hooks). Map for all products: [`.ai/harness/multi_model.md`](../.ai/harness/multi_model.md).

| Layer | Path |
|-------|------|
| Context + hard constraints | `.ai/rules/core.md`, `.ai/rules/security.md`, `.ai/rules/anti_patterns.md`, `.ai/rules/profiling_infra.md` |
| Skills + capability tree | `.ai/skills/index.md` → one skill; `.ai/harness/capabilities.md` to pick |
| Tools (bash / git / build) | `.ai/harness/tools-policy.md` |
| Style | `.ai/standards/python.md`, `.ai/standards/cpp.md`, `.ai/standards/cmake.md` |
| Opening prompt | `.ai/prompts/default.md` |
| Validation (everyone) | pre-commit, `python3 scripts/ai_dev_harness.py` |

Validate edits with **ruff** + **pytest** (Python) or **CMake** out-of-source build (native). See `pyproject.toml` and `CONTRIBUTING.md`.

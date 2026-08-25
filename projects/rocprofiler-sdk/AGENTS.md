# AI Agent Guidelines — rocprofiler-sdk

## Project Docs

- **[`README.md`](README.md)** — user-facing overview.
- **[`CONTRIBUTING.md`](CONTRIBUTING.md)** — the contributor guide, and the source of truth for
  formatting, style, and test-layout requirements. Read it before changing code; the coding style
  guidelines (procedural interfaces across translation units, one variable per line) are enforced
  in review, not by a linter.

## Formatting

C and C++ use **clang-format 11** (`clang-format>=11.0.0,<12.0.0` in
[`requirements.txt`](requirements.txt)), CMake uses **cmake-format**, and Python uses **black**.
These are enforced by `rocprofiler-sdk-formatting.yml` under the monorepo root's
`.github/workflows/` (this project has no `.github` directory of its own), which also rejects any
file not ending in a newline. The monorepo root's `.pre-commit-config.yaml` pins different tools
(clang-format 18, gersemi) but never runs against this project, so do not format with it.

Compiler warnings are a merge gate. Configure with `-DROCPROFILER_BUILD_CI=ON` to get
`-Werror` alongside tests and samples.

## Skills

Reusable agent guidance lives under **[`.claude/skills/`](.claude/skills/)**, one directory per
skill. Load them on demand:

| Skill | Load when |
|-------|-----------|
| [`verifying-test-signal`](.claude/skills/verifying-test-signal/SKILL.md) | A suite reports green — especially a first run, a custom harness, or tests added to existing code |
| [`rocprof-sdk-cpu-only-testing`](.claude/skills/rocprof-sdk-cpu-only-testing/SKILL.md) | Iterating without a full ROCm/HIP build or a GPU |
| [`rocprof-sdk-queue-hook-invariants`](.claude/skills/rocprof-sdk-queue-hook-invariants/SKILL.md) | Touching HSA queue interception, activation predicates, or client ids |

New skills follow the format defined in
[`projects/amdsmi/.claude/skills/writing-skills/SKILL.md`](../amdsmi/.claude/skills/writing-skills/SKILL.md).

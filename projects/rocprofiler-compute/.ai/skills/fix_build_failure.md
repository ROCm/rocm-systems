# Skill: Fix build failure (CMake / native / ROCm)

## Goal

Restore a **failing** CMake configure or build with the **smallest** change. Typical for `src/lib/`, rocprofiler-sdk linkage, or ROCm path issues.

## Steps

1. **Reproduce** from `projects/rocprofiler-compute` with an **out-of-source** tree (never in-source per root `CMakeLists.txt`):
   ```bash
   cmake -S . -B build
   cmake --build build
   ```
   Capture the **first** error (compiler or CMake), not only the last line.
2. **Identify** the failing **target** or configure step (e.g. `rocprofiler-compute-tool`, missing `rocprofiler-sdk`, wrong `ROCM_PATH`).
3. **Isolate** — Change only `CMakeLists.txt`, `src/lib/*.cpp`, or toolchain-related env docs if the issue is path/docs; avoid unrelated refactors.
4. **Fix** minimal CMake or C++ to match existing patterns (`find_package(rocprofiler-sdk)`, `target_link_libraries`, C++17).
5. **Rebuild** the **narrowest** target if possible (`cmake --build build --target <name>`), then full build if needed.
6. **Python-only PRs** — If the failure is pre-commit/CI unrelated to CMake, say so and point to the correct skill (`fix_bug`, `write_test`).

## Constraints

- No in-source build directories; respect `ROCM_PATH` / `CMAKE_PREFIX_PATH` conventions already in project CMake files.
- Do not add new third-party dependencies without approval ([`.ai/rules/core.md`](../rules/core.md)).
- If you cannot reproduce (no ROCm machine), state that and suggest the exact commands/logs for the reporter.

## Output

- **Error summary** — command + key error lines (quoted or paraphrased).
- **Root cause** — one paragraph.
- **Patch** — files touched; unified diff or file list.
- **Build confirmation** — `cmake`/`cmake --build` lines run and **pass** (or blocked-with-reason).

**Output format (strict):** [`.ai/harness/skill_output_contract.md`](../harness/skill_output_contract.md).

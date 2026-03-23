# Skill: Native code / CMake

## Goal

Change the C++ tool or build system.

## Steps

1. Edit `src/lib/*.cpp` and/or `src/lib/CMakeLists.txt` (and root CMake only if required).
2. Keep `rocprofiler-sdk` usage aligned with existing `find_package` / `target_link_libraries`.
3. Configure **out-of-source** build; compile affected targets.
4. If behavior visible to Python/CLI changes, coordinate tests at the integration level.

## Constraints

- C++17 only; no new third-party libraries without approval.
- Follow `standards/cmake.md` for target-based CMake edits.

## Output

- Native/CMake diff, build commands used (out-of-source), and integration impact if visible to CLI/Python.

**Output format (strict):** [`.ai/harness/skill_output_contract.md`](../harness/skill_output_contract.md).

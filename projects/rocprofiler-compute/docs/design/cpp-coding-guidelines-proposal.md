# Proposal: port the rocprofiler-systems C++ skills into rocprofiler-compute

## Motivation

- We have no C++ coding standard in this repo.
- Python has one. `AGENTS.md` points at it, so everyone writes Python the same
  way.
- C++ has nothing. No naming rules. No agreed C++ version. No rule that code
  must be testable. No CMake rules.
- Reviewers argue the same points on every C++ PR.
- Agents make up their own rules each session.
- The [rocprofiler-systems-skills](https://github.com/ROCm/rocprofiler-systems-skills)
  repo already has C++ skills, and someone maintains them.
- Copying them is cheaper than writing our own.
- It also puts both profiler projects on the same style.

## Observation

- Our C++ layer is much smaller than theirs.
- A rule that forces us to restructure code is not worth it at this size.
- So we copy the skills and edit them. We do not point at their repo.
- We drop the naming skill. Renaming every file and namespace gains us nothing.
- We drop the policy-based DI skill. Our test mocks hook into the virtual
  classes. Replacing those with templates would break the mocks.

## C++ skills in the upstream repo

| Skill | What it covers |
| --- | --- |
| [programming-cpp](https://github.com/ROCm/rocprofiler-systems-skills/blob/main/skills/programming-cpp/SKILL.md) | Umbrella rules: C++ Core Guidelines as the reference, C++17 only, performance as a first-class concern, all code must be unit testable. |
| [programming-cpp-naming-rules](https://github.com/ROCm/rocprofiler-systems-skills/blob/main/skills/programming-cpp-naming-rules/SKILL.md) | File and class naming: folder structure mirrors the namespace, one class per file, no redundant `amd_smi_` style prefixes. |
| [programming-cpp-stl-algorithms](https://github.com/ROCm/rocprofiler-systems-skills/blob/main/skills/programming-cpp-stl-algorithms/SKILL.md) | Replace hand-written loops with STL algorithms and pick the right container. |
| [programming-cpp-design-patterns](https://github.com/ROCm/rocprofiler-systems-skills/blob/main/skills/programming-cpp-design-patterns/SKILL.md) | Catalogue of creational, structural, and behavioural patterns with the problem signals that suggest each one. |
| [programming-cpp-constexpr](https://github.com/ROCm/rocprofiler-systems-skills/blob/main/skills/programming-cpp-constexpr/SKILL.md) | Move computation to compile time: `constexpr` values and functions, `if constexpr` dispatch, precomputed tables. |
| [programming-cpp-policy-based-di](https://github.com/ROCm/rocprofiler-systems-skills/blob/main/skills/programming-cpp-policy-based-di/SKILL.md) | Compile-time dependency injection through template policy parameters, for mockable code without virtual dispatch. |
| [programming-cmake-best-practices](https://github.com/ROCm/rocprofiler-systems-skills/blob/main/skills/programming-cmake-best-practices/SKILL.md) | Modern CMake 3.15+: target-based commands, `PUBLIC`/`PRIVATE`/`INTERFACE` scoping, no global variable soup. |

## What we port, and with what edits

- **programming-cpp**: port, edited.
  - Keep Core Guidelines as the reference.
  - Keep the C++17 target, the performance rules, and the testability
    requirement.
  - Remove the line that says these rules beat the existing code style.
  - Add an exception so `torch_trace_collector` can use C++20. libtorch needs
    it.
  - Allow globals and singletons. Ask for a comment saying why. The tool is
    loaded with `LD_PRELOAD` and needs them.
- **programming-cpp-stl-algorithms**: port as-is.
- **programming-cpp-constexpr**: port as-is. Use it for new code. Do not go back
  and rewrite old code.
- **programming-cpp-design-patterns**: port as-is. It only lists patterns.
- **programming-cmake-best-practices**: port as-is. We already follow it.
- **programming-cpp-naming-rules**: drop. It would rename every file and
  namespace we have.
- **programming-cpp-policy-based-di**: drop. It would replace the virtual
  classes our test mocks plug into.

## How we would port it

- Put everything in one file, `.ai/rules/cpp-style.md`, with a table of contents
  and one section per ported skill.
- Apply the edits listed above while copying.
- Add `CPP_CODING_STYLE.md` at the project root. Keep it short. Each entry links
  to its section in `.ai/rules/cpp-style.md`.
- Add a "C++ Code Style" section to `AGENTS.md`. Every AI tool shim routes
  through it.
- Add a Cursor rule beside the Python one, scoped to C++ globs.
- Add a C++ row to the guidelines table in `CONTRIBUTING.md`.

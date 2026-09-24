# LLD: C++ coding guidelines

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
- We drop the policy-based DI skill. Our test mocks hook into the virtual
  classes. Replacing those with templates would break the mocks.
- We keep naming, cut down to a single rule. See below.

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

## One file per topic, not one file for everything

- The umbrella skill alone is over a thousand lines. A reviewer cannot cite a
  line of it, and nobody rereads it.
- So we split it by topic. Each file is short enough to read in one sitting and
  short enough to link to in a review comment.
- The files live under `.ai/rules/cpp/`. `CPP_CODING_STYLE.md` at the project
  root is the table of contents.

| File | Source | What it covers |
| --- | --- | --- |
| `core.md` | umbrella skill | Core Guidelines as the reference, C++17 baseline, interfaces, functions, classes, RAII, early return, `const`, `constexpr`, `noexcept`, `[[nodiscard]]`. |
| `performance.md` | umbrella skill | Allocation, copies, move semantics, cache behaviour, hot path rules. |
| `testability.md` | umbrella skill | Dependency injection through virtual interfaces, mockable seams, gtest and gmock conventions. |
| `naming.md` | naming skill | Cut down to one rule. See below. |
| `comments-and-docs.md` | umbrella skill | Doxygen on public interfaces, when an inline comment earns its place. |
| `callback-boundaries.md` | new, ours | See below. |
| `stl-algorithms.md` | STL skill | Ported as is. |
| `design-patterns.md` | patterns skill | Trimmed. Calls out Adapter and Strategy, which other rules already depend on. |
| `cmake.md` | CMake skill | Ported as is. We already follow it. |

One flat set of topic files, no `reference/` subfolder. Each file owns its
subject. Detail that belongs to an existing rule goes in that rule's file, which
is why the C++17 constexpr specifics live in `core.md`. A subject nothing else
owns gets its own file, which is why design patterns do.

Dropped:

- The policy-based DI skill, for the reason given above.

## New file: `callback-boundaries.md`

- Upstream has nothing like it, because their code is not shaped like ours.
- Every C++ library we ship runs inside somebody else's callback. The
  rocprofiler-sdk tool callbacks call us. The torch dispatcher calls us. Python
  calls the extension module.
- That shapes three things at once, so they belong in one file:
  - What thread we are on and what is allowed there. No blocking and no
    allocation on the sampling path. `synchronized_t`, atomics, `thread_local`.
  - What must not cross the boundary. No exceptions out of a rocprofiler-sdk,
    torch, or Python callback. Status codes instead, and how to report an error
    from a callback.
  - Process-lifetime state. See below.

## Globals and singletons

- We need process-lifetime state in places, and the reason is the callback API,
  not how the library is loaded. The host calls us. There is no object of ours
  that owns the state and no call we can thread it through.
- `LD_PRELOAD` is not the right test for this. `torch_trace_collector` is a
  Python extension module and it already keeps a process-wide `ProcessState`
  and a `thread_local ThreadState`.
- So the rule keys off the cause:
  - Allowed for process-lifetime state that a host callback API leaves us no
    place to own: rocprofiler-sdk tool callbacks, torch `RecordFunction`
    observers, module init.
  - It must never be a namespace scope object whose destructor runs at
    shutdown. Two forms satisfy that, and the choice turns on whether the host
    calls us after our static destructors would have run:
    - No, or unsure: a function-local static behind an accessor. This is the
      default.
    - Yes: a never-destroyed heap object behind a namespace scope reference,
      because a destructor at all is the bug. `rocprofiler_compute_tool.cpp`
      needs this, since rocprofiler-sdk calls `tool_fini` from its own
      `_dl_fini`. Only use it when you can name the teardown path that forces
      it.
  - It must stay reachable from tests, either resettable or handed to consumers
    by reference.
  - A comment says which host API forces it.
  - Banned anywhere else, including as a shortcut for passing state between our
    own functions.
- The rule lives in `callback-boundaries.md`, next to its cause.
  `testability.md` only requires that the accessor stays injectable.

## Naming

- We have two type naming styles in the tree today.
  `pc_sampling_collector/` uses `*_t` snake case, following rocprofiler-sdk's
  own style. Everything written since uses PascalCase.
- Without a rule the next author guesses. So we keep the naming skill, cut to
  one rule:

  > Types are PascalCase.

- Nothing else. No file renames, no namespace rules, no inventory of existing
  violations.
- New files in `pc_sampling_collector/` will look mixed for a while. That is the
  expected outcome and not something to fix.

## Other edits while porting

- Keep the C++17 baseline, and add an exception for `torch_trace_collector`,
  which needs C++20 because libtorch does.
- Remove the line saying these rules beat the existing code style.

## How we wire it in

- Add `.ai/rules/cpp/`, one file per row in the table above. That is where the
  detail lives.
- Add `CPP_CODING_STYLE.md` at the project root. It is both the table of
  contents and the version a person reads.
  - One line per topic, each linking to its file under `.ai/rules/cpp/`.
  - Short enough to skim and to keep current.
  - No second index inside `.ai/rules/cpp/`. Two tables of contents drift.
- Add a "C++ Code Style" section to `AGENTS.md` pointing at
  `CPP_CODING_STYLE.md`. Every AI tool shim routes through `AGENTS.md`.
- Add a Cursor rule beside the Python one, scoped to C++ globs.
- Add a C++ row to the guidelines table in `CONTRIBUTING.md`, pointing at
  `CPP_CODING_STYLE.md`.

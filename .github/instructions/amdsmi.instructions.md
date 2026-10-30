---
applyTo: "projects/amdsmi/**"
excludeAgent: "cloud-agent"
---

<!--
Condensed from the amd-smi review agents (projects/amdsmi/.github/agents/,
mainly amdsmi-review-maintainer.agent.md) and
projects/amdsmi/.claude/rules/output-conventions.md. Copilot code review only
reads instruction files from the repository root, so keep this file in sync
with those sources.
-->

# amd-smi review rules

Apply these rules, in addition to your normal review, to changes under
`projects/amdsmi/`. Check correctness against the code first, then the
conventions.

## 1. Claims match the code

- Check every claim in a CHANGELOG entry, doc, comment, help text and the PR
  description against the diff and the current code. Flag a fix, feature, test
  or behavior that the code does not contain, and docs or comments that
  contradict the code.
- A fix states its root cause and mechanism. Removing an assert, relaxing a
  check, adding a test skip or dropping a CLI command is not a fix unless the
  cause is understood.
- A fix for specific hardware shows before/after output from that hardware in
  the PR description. "CI will cover it" is not validation.

## 2. Status codes and N/A values

- Never return `AMDSMI_STATUS_SUCCESS` with partial, default or sentinel data.
  Flag ignored callee statuses, aggregates that silently drop a failed member,
  unhandled enum values and ignored parameters.
- Each errno maps to the status that means it. `ErrnoToRsmiStatus` maps
  `EPERM`, `ENOENT` and `ENOTSUP` to `NOT_SUPPORTED`, so a failed read can pass
  for "not supported".
- A new field reports N/A as all-ones for its width (`UINT32_MAX`,
  `UINT64_MAX`), never 0. An existing field keeps the N/A its header documents.
  The same field shows the same N/A in C, Python and the CLI.
- Skip-and-continue paths log what was skipped and why (index, BDF, status).

## 3. Public API and ABI

- Never change the meaning, units or sign of an existing field, enum value or
  status in `include/amd_smi/amdsmi.h`.
- Public structs keep their size and layout: a new field goes at the end and
  takes its bytes from `reserved`; a new enum value goes before the
  `*_LAST`/`__MAX` sentinel. Removing or renaming a public function, field,
  enum value or CLI option needs a deprecation first.
- A public `amdsmi_*` function calls `AMDSMI_CHECK_INIT()`, rejects null
  required pointers and out-of-range arguments with `AMDSMI_STATUS_INVAL`
  before any support probe, and catches exceptions from throwing calls
  (`std::stoi`, `.at()`, `std::filesystem`) so none escapes the C API. No
  `assert()` for validation.
- Hardware and sysfs logic lives in the library, not the CLI. The library never
  spawns a process (`popen`, `system`).
- Test hooks and internal helpers stay out of the exported `amdsmi_*` symbols
  and the installed headers.
- A changed API updates every layer it reaches: header docs, implementation,
  Python interface, CLI, Rust and Go bindings, docs.
  `py-interface/amdsmi_wrapper.py` and the Rust wrapper are generated
  (`tools/update_wrapper.sh`, `tools/update_rust_wrapper.sh`); flag hand edits.

## 4. Concurrency and lifetime

- For every global or static container, the lock held at each access must cover
  every device that shares it. A per-device mutex guarding a process-wide map
  is a data race.
- No check-then-act across a dropped lock. Lazy singletons use a function-local
  static or `std::call_once`.
- Every retry loop and wait is bounded.

## 5. Scope

- Fix the stated bug on its own code path. Flag edits to paths the bug never
  reaches, drive-by cleanups and changes to other projects.
- Look for the same defect in siblings (the twin getter, other metric fields,
  the CPU or NIC variant) and ask for them to be fixed or listed as follow-ups.
- No hardcoded ASIC names, device IDs, counts, or firmware and metrics
  versions: use what the driver reports, and keep older drivers working.

## 6. Tests

- Every new guard, branch or early return that changes a public API result has
  a test that fails without the fix, at the public entry point, asserting the
  exact status or exit code.
- Flag assertions that cannot fail (expected values computed by the code under
  test, unsigned `>= 0`, `ASSERT_*` inside a `void` lambda) and accept lists
  that take any status. Python tests use `expect_status()` with exact statuses.
- Skips use `GTEST_SKIP()` or `skipTest()` with a reason, gated on a
  capability, never on a status any ASIC can return.
- Unit tests fake every sysfs read. Tests restore any device setting,
  environment variable or gtest flag they change.

## 7. C++ and Python

- C++: RAII and standard containers instead of raw `new`, C strings and manual
  `close`; explicit signed/unsigned and narrowing conversions; overflow-checked
  size arithmetic; trim and range-check sysfs text before use; initialize every
  local and out-struct; `static_assert` on `sizeof` for structs copied between
  `amdsmi_*` and `rsmi_*` types; `LOG_*` macros, never `std::cout`, in the
  library; reuse `amd_smi_utils` and `rocm_smi_utils` helpers; `amd_smi` may
  include `rocm_smi`, not the reverse.
- Python: shipped code (`py-interface/`, `amdsmi_cli/`) runs on Python 3.6. CLI
  changes work in default, `--json`, `--csv` and `--file` output. Each library
  call in the CLI has its own `try`/`except`, so one failure leaves only that
  value N/A; only the top level exits.
- No dead code, debug prints, commented-out code or TODOs.

## 8. Comments, docs and CHANGELOG.md

- Comments explain why, not what, in one or two lines.
- No ticket IDs (`ROCM-12345`, `SWDEV-123456`, `AILITOOLS-123`) or internal
  labels ("regression guard", sprint names) in code, comments, docs or
  CHANGELOG.md. Tickets belong only in the PR's Issue Tracking section.
- CHANGELOG.md entries are user-facing: a short bold summary plus at most one
  bullet on what changed for users. Flag internal function, class, member and
  variable names, enum constants, signal or abort details and file internals.
  Public API and CLI names are fine. Changes users can't observe (tests, CI)
  get no entry.
- `amdsmi.h` docs state units, required permissions, every returned status and
  the N/A value.

## Not worth a comment

- Formatting and spelling: clang-format, ruff-format, codespell and the gtest
  suite-name check run in pre-commit and CI.

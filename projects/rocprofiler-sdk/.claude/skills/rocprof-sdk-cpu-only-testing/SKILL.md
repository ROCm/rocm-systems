---
name: rocprof-sdk-cpu-only-testing
description: "Use when iterating on rocprofiler-sdk without a full ROCm/HIP build — no GPU, no configured CMake tree, missing submodules, or errors like 'hsa/hsa.h: No such file or directory', 'amd_comgr/amd_comgr.h: No such file or directory', or 'PTL/TaskManager.hh: No such file or directory'. Covers out-of-tree syntax checks, linking a CPU-only GTest binary, which translation units cannot build this way, and the format and warning gates CI enforces."
---

# CPU-Only Testing — rocprofiler-sdk

A full configure-and-build needs ROCm, HIP, and initialized submodules. Without them you can still get real feedback on logic that does not call into HSA at runtime.

Run every command below from this project root, `projects/rocprofiler-sdk` in the rocm-systems monorepo, with `rocr-runtime`, `hip`, and `clr` checked out as its siblings. Non-default locations go in `SIBLINGS` or `ROCR`, and the generated tree in `OUT`.

## Tier 1 — Syntax Only

Seconds per file, and enough for the majority of edits. [`stub-includes.sh`](stub-includes.sh) builds the include tree a configured build would have generated and prints the flags:

```bash
FLAGS=$(.claude/skills/rocprof-sdk-cpu-only-testing/stub-includes.sh)
g++ -std=c++17 -fsyntax-only $FLAGS source/lib/rocprofiler-sdk/<path>.cpp
g++ -std=c++17 -fsyntax-only -Wall -Wextra -Wshadow $FLAGS source/lib/rocprofiler-sdk/<path>.cpp
```

What the script has to fake, because none of it exists in a plain checkout:

| Piece | Why |
|-------|-----|
| An `hsa` symlink to `rocr-runtime/runtime/hsa-runtime/inc`, plus `-I` on `hsa-runtime` itself and on `libhsakmt/include` | Sources include `<hsa/hsa.h>` but rocr keeps those headers in `inc/`; `hsa_api_trace.h` then includes its siblings as `<inc/...>`; and `agent.h` needs `<hsakmt/hsakmttypes.h>` |
| Version headers generated from the `*version*.h.in` templates | CMake configures these. Zeros compile, but a header selecting layout on a version macro takes the version-0 branch |
| `hip/hip_version.h` | Emitted inline by `clr/hipamd/CMakeLists.txt`, not from a template |
| `amd_comgr/amd_comgr.h` | Not a version header but a real API surface, with no comgr checkout here. Reached by anything including `context/context.hpp`, via `cxx/codeobj/disassembly.hpp` |

## Tier 2 — Compile and Link a GTest Binary

**REQUIRED SUB-SKILL:** `verifying-test-signal`. A hand-rolled harness fails silently by default; that skill is what keeps a green run meaningful. Tier 1 executes nothing, so it does not apply there.

Add `-pthread`, build against a prebuilt abseil and gtest, and supply one stub translation unit for the symbols that normally come from `rocprofiler-sdk-static-library`: `hsa::get_core_table`, `hsa::get_amd_ext_table`, `code_object::iterate_loaded_code_objects`, `common::cxx_demangle`, `common::register_static_dtor`, `registration::get_fini_status`. Link with `-Wl,--start-group $ABSL_LIBS -Wl,--end-group -pthread -ldl -lrt`.

## What Will Not Build This Way

`PTL/TaskManager.hh: No such file or directory` means the translation unit reaches `lib/rocprofiler-sdk/internal_threading.hpp`, and `external/ptl` is empty until submodules are initialized. `thread_trace/core.cpp` is the common case. The same applies to `gotcha`, `perfetto`, `pybind11`, and `sqlite`, which are all empty in a plain checkout. Do not force these; pick a different translation unit or use the real build.

**Include-chain surprise:** `context/context.hpp` includes `spm/core.hpp`, so any file touching contexts drags SPM's headers in and a break in SPM surfaces as a compile error somewhere unrelated — `kernel_replay/memory_snapshot.cpp`, for instance. When an error names a subsystem you did not touch, walk the include chain before assuming your edit caused it.

## Gates Before Pushing

```bash
clang-format --dry-run -Werror <changed .cpp/.hpp>   # must be 11.x
cmake-format --check <changed CMakeLists.txt>
black --check <changed .py>
tail -c 1 <each changed file> | xxd                  # must end in a newline
```

**Which formatter version:** clang-format 11 and `cmake-format`, per [CONTRIBUTING.md](../../../CONTRIBUTING.md) and [requirements.txt](../../../requirements.txt) (`clang-format>=11.0.0,<12.0.0`). The monorepo root's `.pre-commit-config.yaml` pins clang-format 18 and gersemi and does not exclude this project, but the workflow that runs it sparse-checks out only `emulation/rocjitsu`, so it never sees these files. The enforcing job is `rocprofiler-sdk-formatting.yml`, also under the monorepo root's `.github/workflows/`. Formatting with 18 produces a diff CI rejects.

Warnings are a merge gate (CONTRIBUTING requirement 6; `ROCPROFILER_BUILD_DEVELOPER` adds `-Werror`), so treat a new warning as a failure even when your own command line is not using `-Werror`.

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Adding a class with private members to share state across translation units | CONTRIBUTING rule 8: plain structs plus free functions for cross-TU internal interfaces |
| `int x, y;` or `auto v = std::vector<T>()` | Rule 10: one variable per line, `auto x = T{}` |
| Concluding a design is untestable because there is no GPU | Admission control, bookkeeping, and predicate logic are all CPU-testable; separate them from the HSA calls |

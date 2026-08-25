---
name: rocprof-sdk-cpu-only-testing
description: "Use when iterating on rocprofiler-sdk without a full ROCm/HIP build — no GPU, no configured CMake tree, missing submodules, or 'PTL/TaskManager.hh: No such file or directory'. Covers out-of-tree syntax checks, linking a CPU-only GTest binary, which translation units cannot build this way, and the format and warning gates CI actually enforces."
---

# CPU-Only Testing — rocprofiler-sdk

A full configure-and-build needs ROCm, HIP, and initialized submodules. Without them you can still get real feedback on logic that does not call into HSA at runtime.

**REQUIRED SUB-SKILL:** `verifying-test-signal` — a hand-rolled harness fails silently by default.

## Two Tiers

**Tier 1 — syntax only, seconds per file.** Catches the majority of edits:

```bash
g++ -std=c++17 -fsyntax-only -D__HIP_PLATFORM_AMD__ \
  -I source -I source/include -I <generated-stub-dir> -I <hsa-stub-dir> \
  -I <rocr>/libhsakmt/include -I <rocr>/runtime/hsa-runtime \
  -I ../hip/include -I ../clr/hipamd/include \
  -I external/abseil-cpp -I external/fmt/include \
  -I external/googletest/googletest/include \
  source/lib/rocprofiler-sdk/<path>.cpp
```

**Tier 2 — compile and link a GTest binary.** Add `-DFMT_HEADER_ONLY=1 -pthread`, build against a prebuilt abseil and gtest, and supply one stub translation unit for the symbols that normally come from `rocprofiler-sdk-static-library` (`hsa::get_core_table`, `hsa::get_amd_ext_table`, `code_object::iterate_loaded_code_objects`, `common::cxx_demangle`, `common::register_static_dtor`, `registration::get_fini_status`). Link with `-Wl,--start-group $ABSL_LIBS -Wl,--end-group -pthread -ldl -lrt`.

Two things the build system would otherwise generate must be hand-stubbed into a private include directory: the `version.h` / `ext_version.h` family (`rocprofiler-sdk`, `-roctx`, `-rocpd`, `-rocattach`, `lib/aqlprofile`) and `hip/hip_version.h` plus `amd_comgr/amd_comgr.h`.

## What Will Not Build This Way

| Symptom | Cause |
|---------|-------|
| `PTL/TaskManager.hh: No such file or directory` | The TU reaches `lib/rocprofiler-sdk/internal_threading.hpp`. `external/ptl` is empty until submodules are initialized, so anything on that include path is out of scope — including `thread_trace/core.cpp` |
| Missing headers from `gotcha`, `perfetto`, `pybind11`, `sqlite` | Same: those submodule directories are empty in a plain checkout |

Do not spend time forcing these; pick a different TU or the real build.

**Include-chain surprise:** `context/context.hpp` includes `spm/core.hpp`. Any TU that touches contexts drags SPM's headers in, so a break in SPM surfaces as a compile error in an unrelated suite — for example `kernel_replay/memory_snapshot.cpp`. When a failure names a subsystem you did not touch, check the include chain before assuming your edit caused it.

## Gates Before Pushing

```bash
clang-format --dry-run -Werror <changed .cpp/.hpp>   # must be 11.x
cmake-format --check <changed CMakeLists.txt>
black --check <changed .py>
g++ ... -Wall -Wextra -Wshadow -fsyntax-only <changed .cpp>
tail -c 1 <each changed file> | xxd   # must end in a newline
```

**Which formatter version:** clang-format 11 and `cmake-format`, per [CONTRIBUTING.md](../../../CONTRIBUTING.md) and [requirements.txt](../../../requirements.txt) (`clang-format>=11.0.0,<12.0.0`). The repository root `.pre-commit-config.yaml` pins clang-format 18 and gersemi and does not exclude this project, but the workflow that runs it sparse-checks out only `emulation/rocjitsu`, so it never sees these files. The enforcing job is `.github/workflows/rocprofiler-sdk-formatting.yml`. Formatting with 18 will produce a diff that CI rejects.

Warnings are a merge gate (CONTRIBUTING requirement 6, `ROCPROFILER_BUILD_DEVELOPER` adds `-Werror`), so treat a new warning as a failure even when your harness is not using `-Werror`.

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Adding a class with private members to share state across TUs | CONTRIBUTING rule 8: plain structs plus free functions for cross-TU internal interfaces |
| `int x, y;` or `auto v = std::vector<T>()` | Rule 10: one variable per line, `auto x = T{}` |
| Concluding a design is untestable because there is no GPU | Admission control, bookkeeping, and predicate logic are all CPU-testable; split them out from the HSA calls |

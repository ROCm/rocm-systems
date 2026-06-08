# api_trace

Runtime dispatch-table layer that lets `rocprofiler` (and any other tool that
speaks the `rocprofiler-register` protocol) intercept hipFile's public C API.

## How it fits together

The runtime implementations of the public hipFile API live in
`hipFile::` (in `src/amd_detail/hipfile.cpp` and
`src/common/hipfile-common.cpp`) — they no longer have `extern "C"` linkage and
are not directly exported from the shared library. This directory wraps them:

1. A `hipFileDispatchTable` struct (declared in
   `include/hipfile-api-trace.h`) holds a function pointer for every public
   API.
2. `api-trace.cpp` constructs a single static dispatch table, populates it
   with the `hipFile::` runtime functions via `UpdateDispatchTable`, and —
   when built with `HIPFILE_ROCPROFILER_REGISTER > 0` — hands the table to
   `rocprofiler_register_library_api_table` so a tool can wrap the pointers.
3. `api-dispatch-interface.cpp` defines the actual exported `extern "C"`
   symbols. Each one is a thin shim that fetches the (possibly tool-wrapped)
   dispatch table via `hipFile::GetHipFileDispatchTable()` and forwards
   through the matching function pointer.

Net effect: every public call goes
`user → extern "C" shim → dispatch table → (tool wrapper, if any) → runtime impl`.
When no profiling tool is loaded, the wrap step is a no-op.

## Files

| File | Role |
| --- | --- |
| `api-trace.cpp` | Defines the singleton dispatch table, the `UpdateDispatchTable` populator, the `rocprofiler-register` `ToolInit` glue, and the compile-time `HIPFILE_ENFORCE_ABI` / `HIPFILE_ENFORCE_ABI_VERSIONING` static asserts. |
| `api-dispatch-interface.cpp` | The `extern "C"` entry points exported from the shared library. Each one dispatches through the table. |
| `api-trace-internal.h` | Internal header. Declares `GetHipFileDispatchTable` and the in-namespace prototypes of the runtime implementations. Included by `api-trace.cpp`, `api-dispatch-interface.cpp`, and `hipfile.cpp`. **Do not include from tests** — the in-namespace declarations collide with the `extern "C"` declarations from `<hipfile.h>` when both become visible through `using namespace hipFile;`. |
| `../../../include/hipfile-api-trace.h` | Public header. Declares the `hipFileDispatchTable` struct, the `PfnHipFile*` function-pointer typedefs, and the table major/step version macros. |

## Versioning

`HIPFILE_RUNTIME_API_TABLE_MAJOR_VERSION` and `..._STEP_VERSION` in
`hipfile-api-trace.h` track the ABI of `hipFileDispatchTable`. Rules:

- Increment **STEP** whenever a new function pointer is appended to the table.
- Increment **MAJOR** (and reset STEP to 0) only for incompatible changes:
  renaming or changing the type of an existing member.
- Never reorder existing members. New members go at the end, under a new
  `STEP_VERSION == N` comment block.

The `HIPFILE_ENFORCE_ABI` and `HIPFILE_ENFORCE_ABI_VERSIONING` static asserts
at the bottom of `api-trace.cpp` make a forgotten version bump a compile
error. If you add a new member without updating them, the build will fail
with a message pointing you here.

## Build

The sources are wired into the hipfile library unconditionally by
`src/amd_detail/CMakeLists.txt` — there is no opt-out. The dispatch table is
always built and always used; the only thing the `HIPFILE_ROCPROFILER_REGISTER`
preprocessor define controls is whether `ToolInit` actually calls
`rocprofiler_register_library_api_table`. With it undefined or zero,
`ToolInit` is a no-op and the table is used as-is.

When `HIPFILE_ROCPROFILER_REGISTER` is enabled, you must also supply
`HIPFILE_ROCP_REG_VERSION_{MAJOR,MINOR,PATCH}` at compile time and link
against `rocprofiler-register`.

## Adding a new public API

1. Add the `PfnHipFile*` typedef and the matching `pfn_*` member to
   `hipFileDispatchTable` in `hipfile-api-trace.h`, under the current step
   version block. Bump `HIPFILE_RUNTIME_API_TABLE_STEP_VERSION`.
2. Implement the runtime function in `namespace hipFile` (in `hipfile.cpp` or
   a sibling source file) and add a matching declaration to
   `api-trace-internal.h`.
3. Assign the function pointer in `UpdateDispatchTable`.
4. Add the matching `extern "C"` shim in `api-dispatch-interface.cpp`.
5. Add a `HIPFILE_ENFORCE_ABI` line for the new member and bump the count
   passed to `HIPFILE_ENFORCE_ABI_VERSIONING` (it must equal
   `last_index + 1`).

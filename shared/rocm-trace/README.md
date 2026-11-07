# rocm-trace

The values every ROCm library uses to write trace events, and every tool uses to
enable and decode them: provider names and GUIDs, keywords, levels, and an
enable check that works on both TraceLogging transports (LTTng on Linux, ETW on
Windows). Header-only.

| Header | For | Needs |
|---|---|---|
| `rocm/trace/values.h` | runtimes and tools | nothing (C99, C++) |
| `rocm/trace/trace.h` | runtimes that write events | a TraceLogging implementation (C11, C++) |

The values are a contract between runtimes and tools. Once a ROCm release ships a
value, it never changes; later releases may add values, and each addition
increments `ROCM_TRACE_VALUES_VERSION` and the package's minor version.

## Values

**Providers.** One per library, named `rocm_<library>`: `rocm_hip` (CLR, HRX),
`rocm_hsa` (ROCr), `rocm_rccl`, `rocm_rocdecode`, `rocm_rocjpeg`, `rocm_roctx`,
`rocm_rocshmem`, `rocm_hipfile`. Each GUID is ETW's standard hash of the name, so
ETW tools also accept `*<name>`:

```bat
tracelog -start rocm -f rocm.etl -guid *rocm_hip -level 4 -matchanykw 0x1
```

**Keywords.** Every event carries the keywords of its group. ETW sessions select
by keyword mask; on LTTng the keyword becomes the event-name suffix (`;k0;` for
bit 0). Control events (state, serialize, copy time, code bytes) change what the
runtime does when a session enables them; ETW cannot enable single events, so
each has its own keyword.

| Keyword | Bit | Events |
|---|---|---|
| `ROCM_TRACE_KW_API` | 0x01 | API calls |
| `ROCM_TRACE_KW_DISPATCH` | 0x02 | `kernel_dispatch`, `graph_dispatch`, `kernel_symbol`, `cmd_handoff`, `cmd_submit_*` |
| `ROCM_TRACE_KW_STATE` | 0x04 | `state_request` (control) |
| `ROCM_TRACE_KW_SERIALIZE` | 0x08 | `serialize_request` (control) |
| `ROCM_TRACE_KW_COPY_TIME` | 0x10 | `copy_complete` (control) |
| `ROCM_TRACE_KW_CODE_BYTES` | 0x20 | `code_object_bytes` (control) |
| `ROCM_TRACE_KW_ALLOC` | 0x40 | allocation and free API calls, in addition to `API` |

**Levels.** Information for every event except hot paths; verbose for functions
called in tight loops (HSA signal and queue-index functions). ETW: 4 and 5.
LTTng: `TRACE_INFO` (6) and `TRACE_DEBUG` (14).

**Arguments.** `ROCM_TRACE_CAPTURE_NONE`, `_SCALARS` and `_FULL` (0, 1, 2) are the
levels of the `ROCM_TRACE_CAPTURE` environment variable (`none`, `scalars`, `full`;
default `full`). Events keep their fields at every level and record 0 or "" for what
is not captured: `scalars` drops strings and values written through output
pointers, `none` keeps only return values. `ROCM_TRACE_STRING_CAP` (256) is the
most bytes a string field records, except URIs and kernel names.

## Use

In rocm-systems, add the directory and link a target:

```cmake
add_subdirectory(${ROCM_SYSTEMS_ROOT}/shared/rocm-trace ${CMAKE_BINARY_DIR}/rocm-trace)
target_link_libraries(amdhip64 PRIVATE rocm::trace)        # a runtime
target_link_libraries(mytool PRIVATE rocm::trace_values)   # a tool
```

Elsewhere, use the installed package:

```cmake
find_package(rocm-trace 1.0 CONFIG REQUIRED)
```

In a runtime:

```c
#include <rocm/trace/trace.h>

ROCM_TRACE_DEFINE_PROVIDER(rocm_hip, HIP);

if (ROCM_TRACE_ENABLED(rocm_hip, "serialize_request", ROCM_TRACE_LEVEL_INFORMATION,
                       ROCM_TRACE_KW_SERIALIZE)) {
  /* serialize this dispatch */
}
```

`rocm::trace` links no TraceLogging implementation: on Windows it is the Windows
SDK; on Linux the runtime links the LTTng flavour of microsoft/tracelogging and
liblttng-ust. Plain data events need no `ROCM_TRACE_ENABLED`: `TraceLoggingWrite`
checks its own event.

## Build and test

```sh
cmake -S . -B build -DROCM_TRACE_BUILD_TESTS=ON \
      -DROCM_TRACE_TRACELOGGING_INCLUDE_DIR=<microsoft/tracelogging>/LTTng/include
cmake --build build && ctest --test-dir build
```

The tests check that every GUID is ETW's hash of its provider name, that keywords
are distinct single bits, and that `values.h` compiles as C99. With a TraceLogging
header available, they also compile `trace.h` as C11 and C++ and check the level
mapping.

| Option | Default | |
|---|---|---|
| `ROCM_TRACE_BUILD_TESTS` | `OFF` | build the tests |
| `ROCM_TRACE_INSTALL` | `ON` at top level, `OFF` as a subdirectory | install the headers and the CMake package |

# rocprofv3-qi: Queue-interception kernel-trace demo

A minimal Windows ROCm profiler tool that demonstrates kernel-dispatch tracing
using the HSA AMD intercept-queue extension end-to-end on Windows (WSL DXG),
without any doorbell-store hooking or HIP-side wrappers.

## What it does

- Loaded by `rocprofiler-register.dll` via the standard rocprofiler-sdk
  registration ABI (`rocprofiler_configure`, `rocprofiler_set_api_table`).
- Substitutes `core_->hsa_queue_create_fn` so every newly created HW queue
  becomes a soft intercept queue:
  - `hsa_amd_queue_intercept_create`
  - `hsa_amd_queue_intercept_register(queue, our_packet_writer, qs)`
  - `hsa_amd_profiling_set_profiler_enabled(queue, 1)`
- The registered packet writer:
  - For each `HSA_PACKET_TYPE_KERNEL_DISPATCH` packet, allocates a per-dispatch
    completion signal, swaps it into the AQL packet, and registers an
    async handler.
  - When the handler fires, calls
    `hsa_amd_profiling_get_dispatch_time` for start/end timestamps,
    records a `KernelRow` into the shared `TraceBuffers`, and chains the
    caller's original completion signal.
- Resolves kernel names via `hsa_executable_freeze` →
  `hsa_executable_iterate_agent_symbols` →
  `hsa_executable_symbol_get_info(KERNEL_OBJECT/NAME)`.
- Flushes a CSV (`rocprofv3_<pid>_kernel_trace.csv`) at process exit.

No doorbell-store hook. No HIP API trace. No memory copy trace.

## Why it exists

Companion to `projects/rocprofv3-min`, which uses both the AMD intercept
extension *and* a doorbell-store fallback. This project proves that on
Windows DXG the intercept extension alone is sufficient for kernel-trace,
and serves as a smaller reference for that path.

## Build

```bash
export CMAKE_EXE="C:/Program Files/Microsoft Visual Studio/2022/Enterprise/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe"
"$CMAKE_EXE" -S . -B build -G "Visual Studio 17 2022" -A x64
"$CMAKE_EXE" --build build --config Release --target install
```

The DLL `rocprofv3-qi.dll` installs to the stage `bin/` directory next to
`amdhip64_7.dll` and `rocprofiler-register.dll`.

## Run

```bash
export GPU_ENABLE_PAL=0
export ROCP_TOOL_LIBRARIES=rocprofv3-qi.dll
export ROCPROFV3_OUTPUT_DIR=.
./hip_hello.exe
ls rocprofv3_*_kernel_trace.csv
```

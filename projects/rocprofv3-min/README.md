# rocprofv3-min

Minimal Windows port of `rocprofv3` for HIP applications using the
statically-linked rocr-runtime that ships inside `amdhip64_7.dll`.

Produces the same three CSVs as Linux `rocprofv3`:

* `*_hip_api_trace.csv`
* `*_kernel_trace.csv`
* `*_memory_copy_trace.csv`

```
rocprofv3.exe --hip-trace --kernel-trace --memory-copy-trace -- <hip_app>
```

## Build

```
cmake -B build -S . -G "Visual Studio 17 2022" -A x64 \
      -DROCPROFV3_MIN_STAGE_DIR=C:/path/to/your/hip/stage
cmake --build build --config Release --target rocprofv3-min rocprofv3
cmake --install build --config Release
```

`ROCPROFV3_MIN_STAGE_DIR` must point at a HIP install layout containing
`include/hip/...` and `bin/` (where the artifacts will land). HSA headers
are picked up from the sibling `rocr-runtime` project in this repo.

## Runtime

```
set GPU_ENABLE_PAL=0
rocprofv3.exe --hip-trace --kernel-trace --memory-copy-trace -- hip_hello.exe
```

`GPU_ENABLE_PAL=0` is required: with PAL enabled, rocclr bypasses HSA
entirely and the wrappers are never invoked.

## Windows-specific divergence from Linux rocprofv3

* **Doorbell-store interception** for kernel dispatches. The Linux path
  (`hsa_amd_queue_intercept_create` substitute for `hsa_queue_create`)
  is incompatible with the WSL DXG thunk's `AqlToPm4Thread`, which
  exceptions on the soft-queue indirection (`HSA_STATUS_ERROR_EXCEPTION`,
  status `0x1016`). Instead we record each queue's doorbell signal and
  hook `hsa_signal_store_screlease` / `hsa_signal_store_relaxed`. When
  the store targets a registered doorbell we walk the AQL ring slots
  released by the store, swap each `KERNEL_DISPATCH` packet's
  `completion_signal` for one of ours, register an async handler, and
  forward the doorbell store unchanged.
* **`GPU_ENABLE_PAL=0` is mandatory** on Windows. Linux does not have a
  PAL backend choice for ROCr.
* **Blit-kernel name prefixes** include `__amd_rocclr_copy*` and
  `__amd_rocclr_fill*` (Windows rocclr) in addition to `__amd_copy*` /
  `__amd_fill*` (historical / Linux). Both are routed to
  `memory_copy_trace.csv` as `Direction = BLIT_KERNEL`.

The async-copy path, agent enumeration, kernel-name resolution
(`hsa_executable_freeze` + `hsa_executable_iterate_agent_symbols`),
profiling-enable plumbing, and CSV schema are otherwise identical to
Linux.

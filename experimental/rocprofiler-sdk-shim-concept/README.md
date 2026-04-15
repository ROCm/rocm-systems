# rocprofiler-sdk-shim-concept

This directory hosts an isolated proof-of-concept for the current shim design in
`docs/dispatch-tracer/SHIM_MEMFD_SOCK_DESIGN.md`.

It is intentionally standalone so you can build and test it without patching the
existing `rocprofiler-register`, `rocprofiler-sdk`, CLR/HIP, or ROCR/HSA projects.

What this concept targets for real:

- the real `rocprofiler-register` API-table hook ABI via
  `rocprofiler_set_api_table` and `rocprofiler_attach_set_api_table`
- the real `rocprofiler-sdk` tool ABI via `rocprofiler_configure`
- the real HIP/CLR registration path using `hip_api_trace.hpp`
- the real ROCR/HSA registration path using `hsa.h`

What it does today:

- captures real `hip` and `hsa` dispatch-table registrations
- allocates real per-table slot ranges and op metadata
- wraps a tiny real subset of ops:
  - HIP: `hipGetDeviceCount`, `hipMalloc`, `hipFree`
  - HSA: `hsa_init`, `hsa_iterate_agents`, `hsa_shut_down`
- exports the shim/SDK layering helpers:
  - `shim_get_runtime_original`
  - `shim_get_next_in_chain`
  - `shim_set_next_in_chain`
- exposes the memfd + abstract-socket + SCM_RIGHTS + ring-buffer transport
- provides external attach/stream/detach validation via `shim_consumer_test`

Current limitation:

- this does not yet patch `projects/rocprofiler-sdk` to make the in-tree SDK wrapper
  installation shim-aware. The concept exports the layering seam, but the actual SDK
  integration work is still separate.

## Build

```bash
cmake -S experimental/rocprofiler-sdk-shim-concept -B build/shim-concept -DCMAKE_BUILD_TYPE=Release
cmake --build build/shim-concept -j
```

## Run: HIP

```bash
export LD_LIBRARY_PATH=$PWD/build/shim-concept/lib:$LD_LIBRARY_PATH
export ROCPROFILER_REGISTER_FORCE_LOAD=1
export ROCPROFILER_REGISTER_LIBRARY=$PWD/build/shim-concept/lib/librocprofiler-sdk-shim-concept.so

./build/shim-concept/bin/real_hip_probe_test &
TARGET_PID=$!
./build/shim-concept/bin/shim_consumer_test $TARGET_PID 5
wait $TARGET_PID
```

## Run: HSA

```bash
export LD_LIBRARY_PATH=$PWD/build/shim-concept/lib:$LD_LIBRARY_PATH
export ROCPROFILER_REGISTER_FORCE_LOAD=1
export ROCPROFILER_REGISTER_LIBRARY=$PWD/build/shim-concept/lib/librocprofiler-sdk-shim-concept.so

./build/shim-concept/bin/real_hsa_probe_test &
TARGET_PID=$!
./build/shim-concept/bin/shim_consumer_test $TARGET_PID 5
wait $TARGET_PID
```

The HSA probe intentionally uses the shim's test-only helper export to call through the
registered `HsaApiTable`, because direct public `hsa_*` entry points do not reliably
exercise that table on every runtime path.

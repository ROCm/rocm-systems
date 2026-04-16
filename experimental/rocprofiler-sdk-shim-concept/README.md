# rocprofiler-sdk-shim-concept

This directory hosts a transport-only shim concept built around the real
`rocprofiler-sdk` tool architecture.

What is real in this concept:

- the tool is loaded through the normal `ROCP_TOOL_LIBRARIES` path
- the tool is inert unless `ROCP_SHIM_CONCEPT_ENABLE=1` is set
- `rocprofiler-sdk` creates the context, buffer, and tracing services
- `rocprofiler-sdk` produces the HIP/HSA/runtime-initialization records
- the shim only snapshots those SDK buffer records into a memfd-backed ring
- the consumer decodes the transported records with the real `rocprofiler-sdk`

What this concept is meant to validate:

- a target-side SDK tool can export records to an external consumer without the
  consumer living in the target process
- the shim can stay transport-focused instead of owning dispatch wrappers
- real HIP and real HSA activity can be observed through SDK-produced records
- runtime-load metadata is published from real
  `ROCPROFILER_BUFFER_TRACING_RUNTIME_INITIALIZATION` records

Services configured by the prototype:

- `ROCPROFILER_BUFFER_TRACING_RUNTIME_INITIALIZATION`
- `ROCPROFILER_BUFFER_TRACING_HSA_CORE_API`
- `ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT`

## Build

```bash
cmake -S experimental/rocprofiler-sdk-shim-concept -B build/shim-concept -DCMAKE_BUILD_TYPE=Release
cmake --build build/shim-concept -j
```

## Run: HIP

```bash
export LD_LIBRARY_PATH=$PWD/build/shim-concept/lib:/opt/rocm/lib:$LD_LIBRARY_PATH
export ROCP_TOOL_LIBRARIES=$PWD/build/shim-concept/lib/librocprofiler-sdk-shim-concept.so
export ROCP_SHIM_CONCEPT_ENABLE=1

./build/shim-concept/bin/real_hip_probe_test &
TARGET_PID=$!
./build/shim-concept/bin/shim_consumer_test $TARGET_PID 5
wait $TARGET_PID
```

## Run: HSA

```bash
export LD_LIBRARY_PATH=$PWD/build/shim-concept/lib:/opt/rocm/lib:$LD_LIBRARY_PATH
export ROCP_TOOL_LIBRARIES=$PWD/build/shim-concept/lib/librocprofiler-sdk-shim-concept.so
export ROCP_SHIM_CONCEPT_ENABLE=1

./build/shim-concept/bin/real_hsa_probe_test &
TARGET_PID=$!
./build/shim-concept/bin/shim_consumer_test $TARGET_PID 5
wait $TARGET_PID
```

The external consumer can attach after the target has started. Once attached, it
enables transport capture by flipping a single shared-memory flag. The tracing
remains owned by `rocprofiler-sdk`; the shim only forwards already-produced
records.

If `ROCP_SHIM_CONCEPT_ENABLE` is unset, the library returns `NULL` from
`rocprofiler_configure` and does not initialize the transport layer, so existing
in-process tools and normal application runs are unaffected.

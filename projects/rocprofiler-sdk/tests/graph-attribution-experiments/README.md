# Graph capture attribution experiments

This branch builds on ROCm/rocm-systems PRs #14204 and #14211 (head
`2e35762550e20b13efb9637619654955ca6507ff`). It adds:

- Capture-plan invalidation after successful graph-exec mutations, including memcpy parameter updates.
- Module-function name lookup for graph nodes captured by module and ext-module launches.
- Opt-in one-dimensional ROCclr copyBuffer attribution using a unique source/destination/byte-count match. Duplicate and unsupported copies remain unattributed.
- JSON test-tool controls for starting dispatch tracing before graph capture and finalizing idle workers from a control thread.
- Native positive/negative tests and a SGLang long-context reproduction harness.

The copyBuffer decoder uses a private ROCclr ABI and is **off by default**.
Enable it with `ROCPROFILER_EXPERIMENTAL_GRAPH_COPYBUFFER=1`. It validates
kernel name, argument-segment size, alignment, remainder, endpoint and workgroup
fields before matching a captured copy. It does not implement general SDMA,
array, 2D, batch or split-copy attribution. Runtime changes require revalidation. Copy attribution is disabled for mixed graphs
containing any node outside the recorded capture, including manually added nodes.
Graph node ordinals and capture matching still inherit the upstream ordering
constraints. Resolving a HIP capture call does not establish Python layer labels.

## Experimental memset attribution (native rocprofv3)

`ROCPROFILER_EXPERIMENTAL_GRAPH_FILLBUFFER=1` enables a separate, default-off
ROCclr `fillBufferUnAligned` decoder. It supports captured one-dimensional
Memset nodes with element sizes 1, 2 and 4, and the tested 384-byte runtime
argument segment. It checks destination, complete byte length, region geometry,
workgroup fields and rotated by-value pattern payloads before requiring a unique
capture-time parameter match. It never dereferences the runtime pattern pointer.
The private runtime ABI must be revalidated for other ROCm builds.

Successful graph-exec memset changes invalidate the whole capture plan; failed
changes preserve it. Source-graph memset changes, duplicate matches, unrecorded
nodes, child graphs and unsupported fill shapes refuse fill attribution. A zero
ID is a coverage gap, not permission to infer a source from kernel name or order.

The stdlib validator reads the **native rocprofv3 JSON**, checks exact source API
IDs and every launch/dispatch count, and expects refusals for negative cases.
The HIP probe independently checks all GPU bytes, including offset and typed
patterns. Run with a compatible SDK/native tool build and runtime library paths:

```bash
SDK_BUILD=/absolute/sdk-build bash projects/rocprofiler-sdk/tests/graph-attribution-experiments/run_memset_native.sh /absolute/new-results
```

## Native reproduction

Build the SDK shared library and `rocprofiler-sdk-json-tool` with tests enabled.
Run in a ROCm environment with the target GPU available:

```bash
export SDK_BUILD=/absolute/path/to/sdk-build
export ROCM_PATH=/opt/rocm
export GRAPH_TEST_ARCH=gfx950
bash projects/rocprofiler-sdk/tests/graph-attribution-experiments/run_native.sh /absolute/path/to/new-results
```

The module test checks the GPU result and 18 exact capture IDs, including repeated
module names and ext-module launches. The memcpy test checks GPU buffer contents
and 30 dispatches covering offsets, unaligned sizes, duplicate rejection,
mutation invalidation, fresh-exec recovery and manually added duplicates. The existing CTest graph suites
also cover multi-stream replay and kernel-node mutations.

## SGLang reproduction

Validated with SGLang `0.5.23.dev20261101+g1ad0149814`, ROCm 10 image
`rocm/sgl-dev:v0.5.23-rocm10-mi35x-20261101`, four MI350-series GPUs,
DeepSeek-V4.1-Flash, DSPARK and AMD monokernel. The workload uses 524,288 exact
input token IDs and requests 128 output tokens. It flushes the dedicated
server's cache before each request; do not target a shared production server.
Chunked prefill is eager; decode uses CUDA Graph.

The scripts must run in the same PID namespace as the server. Supply fresh,
absolute paths and a compatible model/runtime installation:

```bash
export SDK_BUILD=/absolute/path/to/sdk-build
export MODEL_PATH=/absolute/path/to/model
export GRAPH_TRACE_DIR=/absolute/path/to/new-results
export EXPERIMENT_DIR="$PWD/projects/rocprofiler-sdk/tests/graph-attribution-experiments"
export HIP_VISIBLE_DEVICES=0,1,2,3
bash "$EXPERIMENT_DIR/launch_sglang.sh" &
export SGLANG_SERVER_PID=$!
python3 "$EXPERIMENT_DIR/run_when_ready.py"
# Wait for finalized-<pid> files for all four scheduler workers.
python3 "$EXPERIMENT_DIR/summarize_run.py" "$GRAPH_TRACE_DIR"
kill -TERM "$SGLANG_SERVER_PID"
```

The launcher fixes the experimental copyBuffer flag on, queue interposition off,
and uses port 31241. Edit these explicitly for other experiments. It collects
code-object symbols throughout and enables HIP/dispatch tracing immediately
before the first PyTorch graph capture. `finalize` is written only after the
requests complete; do not reuse a directory containing that marker.

Some ROCm Python wheels load their own SDK through an absolute path. Verify the
server log reports SDK 1.5.0 and the intended binary, and avoid mixing core/devel
LLVM libraries. In the tested disposable container the Python core SDK reference
was redirected to the custom build, with the original library retained. The
harness intentionally does not rewrite installed libraries automatically.

## Observed validation

See `validation-summary.json`. Per-rank capture-call resolution improved from
2,682/7,085 (37.85%) to 6,618/6,805 (97.25%) after module naming, and to
6,270/6,270 (100%) with experimental copyBuffer support. All four ranks agreed;
176 copyBuffer dispatches per rank resolved to `hipMemcpyAsync` in the final run.
Existing graph CTests passed 18/18. Each full run returned the exact prompt/output
length and saved all worker traces. The different dispatch totals reflect
speculative acceptance differences, not a throughput comparison.

The summary intentionally excludes model weights, generated response text,
machine-specific paths and bulk traces. The percentages describe the observed
request windows, not a guarantee for arbitrary graphs or ROCm versions.

The post-review rerun at `8bc27cb0` used the same 524,288/128-token workload
with CUDA Graph decode and zero cached tokens or retractions. Each of four ranks
resolved 6,513/6,513 graph dispatches, including 179 copyBuffer dispatches mapped
to `hipMemcpyAsync`; all worker traces finalized and no kernel names were missing.
This verifies observed coverage for this workload, not arbitrary-graph accuracy.

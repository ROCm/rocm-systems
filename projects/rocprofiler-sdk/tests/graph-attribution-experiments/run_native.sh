#!/bin/bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${SDK_BUILD:?Set SDK_BUILD to the absolute SDK build directory}"
rocm=${ROCM_PATH:-/opt/rocm}
out=${1:?Pass an absolute output directory}
mkdir -p "$out"
"$rocm/bin/hipcc" --genco --offload-arch="${GRAPH_TEST_ARCH:-gfx950}" "$here/graph_module_kernels.hip" -o "$out/module.hsaco"
"$rocm/bin/hipcc" "$here/graph_module_probe.hip" -o "$out/module-probe"
"$rocm/bin/hipcc" "$here/graph_memcpy_probe.hip" -o "$out/memcpy-probe"
profile() {
 env LD_LIBRARY_PATH="$SDK_BUILD/lib:${LD_LIBRARY_PATH:-}" \
 LD_PRELOAD="$SDK_BUILD/lib/rocprofiler-sdk/librocprofiler-sdk-json-tool.so" \
 ROCPROFILER_REGISTER_LIBRARY="$SDK_BUILD/lib/librocprofiler-sdk.so" \
 ROCPROFILER_QUEUE_INTERPOSITION=0 ROCPROFILER_EXPERIMENTAL_GRAPH_COPYBUFFER=1 \
 ROCPROFILER_TOOL_CONTEXTS=HIP_API_BUFFERED,KERNEL_DISPATCH_BUFFERED,CODE_OBJECT \
 ROCPROFILER_TOOL_OUTPUT_FILE="$1" "${@:2}"
}
profile "$out/module.json" "$out/module-probe" "$out/module.hsaco" > "$out/module.log" 2>&1
python3 "$here/validate_module_graph.py" "$out/module.json"
profile "$out/memcpy.json" "$out/memcpy-probe" > "$out/memcpy.log" 2>&1
python3 "$here/validate_memcpy_graph.py" "$out/memcpy.json"

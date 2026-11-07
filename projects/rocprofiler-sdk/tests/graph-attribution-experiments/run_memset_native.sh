#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${SDK_BUILD:?Set SDK_BUILD to the absolute SDK build directory}"
out=${1:?Pass an absolute empty result directory}
mkdir "$out"
"${ROCM_PATH:-/opt/rocm}/bin/hipcc" "$here/graph_memset_probe.hip" -o "$out/probe"
for mode in disabled enabled; do
 flag=0; [[ "$mode" == enabled ]] && flag=1
 env ROCPROFILER_EXPERIMENTAL_GRAPH_FILLBUFFER="$flag" \
 ROCPROFILER_QUEUE_INTERPOSITION=0 \
 ROCPROFILER_REGISTER_LIBRARY="$SDK_BUILD/lib/librocprofiler-sdk.so" \
 LD_LIBRARY_PATH="$SDK_BUILD/lib:${LD_LIBRARY_PATH:-}" \
 python3 "$SDK_BUILD/bin/rocprofv3" --kernel-trace --hip-trace \
 --output-format json --output-directory "$out/$mode" --output-file probe \
 -- "$out/probe" > "$out/$mode.log" 2>&1
 args=(); [[ "$mode" == disabled ]] && args+=(--disabled)
 python3 "$here/validate_memset_native.py" "$out/$mode/probe_results.json" "${args[@]}" | tee "$out/$mode-validation.json"
done

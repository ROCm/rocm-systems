#!/bin/bash
set -eu
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${SDK_BUILD:?Set SDK_BUILD}"
: "${MODEL_PATH:?Set MODEL_PATH}"
: "${GRAPH_TRACE_DIR:?Use a fresh absolute output directory}"
mkdir -p "$GRAPH_TRACE_DIR"
if [ -e "$GRAPH_TRACE_DIR/finalize" ]; then echo "Use a fresh output directory" >&2; exit 1; fi
export GRAPH_TRACE_ENABLE=1
export PYTHONPATH="$here:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$SDK_BUILD/lib:${LD_LIBRARY_PATH:-}"
export ROCPROFILER_REGISTER_LIBRARY=$SDK_BUILD/lib/librocprofiler-sdk.so
export ROCP_TOOL_LIBRARIES=$SDK_BUILD/lib/rocprofiler-sdk/librocprofiler-sdk-json-tool.so
export ROCPROFILER_QUEUE_INTERPOSITION=0
export ROCPROFILER_EXPERIMENTAL_GRAPH_COPYBUFFER=1
export ROCPROFILER_TOOL_CONTEXTS=HIP_API_BUFFERED,KERNEL_DISPATCH_BUFFERED,CODE_OBJECT
exec python3 -m sglang.launch_server --model-path "$MODEL_PATH" --trust-remote-code --tp 4 --speculative-algorithm DSPARK --speculative-dspark-block-size 5 --fp8-gemm-backend aiter --enforce-shared-experts-fusion --enable-amd-monokernel --context-length 528384 --mem-fraction-static 0.8 --max-running-requests 1 --cuda-graph-max-bs-decode 1 --chunked-prefill-size 8192 --host 127.0.0.1 --decode-log-interval 1 --random-seed 38559850 --port 31241 > "$GRAPH_TRACE_DIR/server.log" 2>&1

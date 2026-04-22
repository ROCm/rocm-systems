#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="${1:-/tmp/src}"
DEMO_ROOT="${2:-/tmp/perfxpert-vhs-demo}"
MOCK_SRC_ROOT="${3:-/tmp/mock_src}"

mkdir -p "$DEMO_ROOT" "$MOCK_SRC_ROOT"

cp "$SRC_ROOT/experimental/python/perfxpert/tests/fixtures/compute_bound.db" \
  "$DEMO_ROOT/trace.db"

cat > "$DEMO_ROOT/_python_api_demo.py" <<'PY'
from pathlib import Path
from pprint import pprint

from perfxpert import api

db_path = Path(__file__).resolve().parent / "trace.db"
doc = api.agent_root(database_path=str(db_path), airgap=True)

summary = {
    "primary_bottleneck": doc.get("primary_bottleneck"),
    "hotspot_count": len(doc.get("hotspots") or []),
    "recommendation_count": len(doc.get("recommendations") or []),
}

pprint(summary, sort_dicts=False)
PY

cat > "$MOCK_SRC_ROOT/kernel.hip" <<'EOF'
#include <hip/hip_runtime.h>

__global__ void add(float* a, float* b) { *a += *b; }

int main() {
  float *a, *b;
  hipMemcpy(a, b, 64, hipMemcpyHostToDevice);
  hipMemcpy(a, b, 64, hipMemcpyHostToDevice);
  hipMemcpy(a, b, 64, hipMemcpyHostToDevice);
  hipMemcpy(a, b, 64, hipMemcpyDeviceToHost);
  hipLaunchKernelGGL(add, dim3(1), dim3(1), 0, 0, nullptr, nullptr);
  hipDeviceSynchronize();
  return 0;
}
EOF

touch "$DEMO_ROOT/.ready"

#!/bin/bash
# AegisBit smoke test — verifies the binary release works on this machine.
#
# Checks: dependencies present, library loads, kernel compiles,
#         profiling runs end-to-end with correct output.
#
# Usage: ./tools/smoke-test.sh
#   Run from the extracted release directory.
#   Requires: ROCm (hipcc), AMD GPU

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LIB_PATH="${ROOT_DIR}/lib/libaegisbit.so"
CLI_PATH="${ROOT_DIR}/tools/aegisbit"
TMPDIR=$(mktemp -d /tmp/aegisbit_smoke.XXXXXX)
PASS=0
FAIL=0

cleanup() { rm -rf "$TMPDIR"; }
trap cleanup EXIT

pass() { echo "  PASS  $1"; PASS=$((PASS + 1)); }
fail() { echo "  FAIL  $1: $2"; FAIL=$((FAIL + 1)); }

echo "=== AegisBit Smoke Test ==="
echo "  Library: ${LIB_PATH}"
echo "  CLI:     ${CLI_PATH}"
echo ""

# --- Check 1: Library exists ---
if [ -f "$LIB_PATH" ]; then
  pass "library exists ($(du -h "$LIB_PATH" | cut -f1))"
else
  fail "library exists" "not found at $LIB_PATH"
  echo "FATAL: Cannot continue without library."
  exit 1
fi

# --- Check 2: No LLVM dynamic deps ---
LLVM_DEPS=$(ldd "$LIB_PATH" 2>/dev/null | grep -i "llvm" || true)
if [ -z "$LLVM_DEPS" ]; then
  pass "LLVM statically linked"
else
  fail "LLVM statically linked" "found dynamic LLVM deps: $LLVM_DEPS"
fi

# --- Check 3: ROCm libs resolve ---
MISSING=$(ldd "$LIB_PATH" 2>/dev/null | grep "not found" || true)
if [ -z "$MISSING" ]; then
  pass "all shared library deps resolve"
else
  fail "shared library deps" "missing: $MISSING"
  echo "FATAL: ROCm libraries not found. Is ROCm installed?"
  exit 1
fi

# --- Check 4: CLI dry-run ---
DRY_OUT=$("$CLI_PATH" --dry-run -- echo hello 2>&1) || true
if echo "$DRY_OUT" | grep -q "LD_PRELOAD=.*libaegisbit.so"; then
  pass "CLI --dry-run finds library"
else
  fail "CLI --dry-run" "unexpected output: $DRY_OUT"
fi

# --- Check 5: hipcc available ---
if command -v /opt/rocm/bin/hipcc &>/dev/null; then
  pass "hipcc found"
else
  fail "hipcc found" "/opt/rocm/bin/hipcc not available"
  echo "Cannot run profiling tests without hipcc. Skipping GPU tests."
  echo ""
  echo "=== Results: $PASS passed, $FAIL failed (GPU tests skipped) ==="
  exit $([[ $FAIL -eq 0 ]] && echo 0 || echo 1)
fi

# --- Check 6: Compile test kernel ---
cat > "$TMPDIR/kernel.cpp" << 'KERNEL_EOF'
#include <hip/hip_runtime.h>
#include <cstdio>
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
int main() {
    const int N = 1024;
    float h_a[1024], h_b[1024], h_c[1024];
    for (int i = 0; i < N; i++) { h_a[i] = 1.0f; h_b[i] = 2.0f; }
    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, N*sizeof(float));
    hipMalloc(&d_b, N*sizeof(float));
    hipMalloc(&d_c, N*sizeof(float));
    hipMemcpy(d_a, h_a, N*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, N*sizeof(float), hipMemcpyHostToDevice);
    vector_add<<<(N+255)/256, 256>>>(d_a, d_b, d_c, N);
    hipDeviceSynchronize();
    hipMemcpy(h_c, d_c, N*sizeof(float), hipMemcpyDeviceToHost);
    int err = 0;
    for (int i = 0; i < N; i++) if (h_c[i] != 3.0f) err++;
    printf(err == 0 ? "PASS\n" : "FAIL\n");
    hipFree(d_a); hipFree(d_b); hipFree(d_c);
    return err > 0 ? 1 : 0;
}
KERNEL_EOF

if /opt/rocm/bin/hipcc -O2 -o "$TMPDIR/kernel" "$TMPDIR/kernel.cpp" 2>/dev/null; then
  pass "HIP kernel compiles"
else
  fail "HIP kernel compiles" "hipcc failed"
  echo "Cannot run profiling test. Stopping."
  echo ""
  echo "=== Results: $PASS passed, $FAIL failed ==="
  exit 1
fi

# --- Check 7: Kernel runs without profiling ---
PLAIN_OUT=$("$TMPDIR/kernel" 2>&1)
if echo "$PLAIN_OUT" | grep -q "PASS"; then
  pass "kernel runs correctly (baseline)"
else
  fail "kernel baseline" "expected PASS, got: $PLAIN_OUT"
fi

# --- Check 8: Profile the kernel and get VMEM report ---
JSON_FILE="$TMPDIR/report.json"
PROFILE_OUT=$("$CLI_PATH" --filter="*vector*" -o "$JSON_FILE" -- "$TMPDIR/kernel" 2>&1)

if echo "$PROFILE_OUT" | grep -q "VMEM Coalescing"; then
  pass "profiling produces VMEM coalescing report"
else
  fail "VMEM report" "no 'VMEM Coalescing' in output"
fi

# --- Check 9: Kernel still correct under instrumentation ---
if echo "$PROFILE_OUT" | grep -q "PASS"; then
  pass "kernel correct under instrumentation"
else
  fail "kernel correctness" "PASS not found in profiled output"
fi

# --- Check 10: JSON output is valid ---
if [ -f "$JSON_FILE" ] && python3 -c "import json; d=json.load(open('$JSON_FILE')); assert d.get('version')==1; assert len(d.get('kernels',[])) > 0" 2>/dev/null; then
  pass "JSON output valid (version=1, kernels present)"
else
  fail "JSON output" "missing, empty, or malformed"
fi

# --- Check 11: Coalescing efficiency is 100% for stride-1 kernel ---
if python3 -c "
import json, sys
d = json.load(open('$JSON_FILE'))
for k in d['kernels']:
  for s in k.get('vmem_coalescing',{}).get('sites',[]):
    if s['avg_efficiency_pct'] < 90:
      sys.exit(1)
" 2>/dev/null; then
  pass "coalescing efficiency >= 90% (stride-1 access)"
else
  fail "coalescing efficiency" "expected >= 90% for stride-1 kernel"
fi

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
exit $([[ $FAIL -eq 0 ]] && echo 0 || echo 1)

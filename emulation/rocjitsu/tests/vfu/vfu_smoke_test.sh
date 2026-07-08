#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# vfu_smoke_test.sh: End-to-end smoke test for the rocjitsu vfio-user GPU.
#
# Prerequisites:
#   - rocjitsu-vfu binary built and on PATH (or pass --vfu-bin)
#   - qemu-system-x86_64 on PATH with vfio-user-pci support
#   - A guest VM image with ROCm and amdgpu.ko installed (pass --guest-image)
#   - SSH access to guest via forwarded port (pass --ssh-port, default 2222)
#   - hipcc available in the guest (part of the ROCm installation)
#
# Usage:
#   ./vfu_smoke_test.sh [options]
#   Options:
#     --vfu-bin PATH      Path to rocjitsu-vfu binary
#     --config PATH       rocjitsu topology JSON config
#     --guest-image PATH  QEMU disk image with ROCm guest
#     --socket PATH       UNIX socket path (default: /tmp/rocjitsu-vfu-0.sock)
#     --ssh-port PORT     SSH port forwarding for guest (default: 2222)
#     --ssh-key PATH      SSH private key for guest access
#     --no-cleanup        Do not kill rocjitsu-vfu and QEMU after test
#
# Exit codes:
#   0  All checks passed
#   1  One or more checks failed

set -euo pipefail

VFU_BIN="${VFU_BIN:-$(dirname "$0")/../../build/tools/rocjitsu-vfu/rocjitsu-vfu}"
CONFIG="${CONFIG:-$(dirname "$0")/../../configs/gfx950_mi350p_kmd.json}"
GUEST_IMAGE="${GUEST_IMAGE:-}"
SOCKET="${SOCKET:-/tmp/rocjitsu-vfu-0.sock}"
SSH_PORT="${SSH_PORT:-2222}"
SSH_KEY="${SSH_KEY:-}"
CLEANUP=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vfu-bin)    VFU_BIN="$2";    shift 2 ;;
    --config)     CONFIG="$2";     shift 2 ;;
    --guest-image) GUEST_IMAGE="$2"; shift 2 ;;
    --socket)     SOCKET="$2";     shift 2 ;;
    --ssh-port)   SSH_PORT="$2";   shift 2 ;;
    --ssh-key)    SSH_KEY="$2";    shift 2 ;;
    --no-cleanup) CLEANUP=0;       shift   ;;
    *)            echo "Unknown: $1"; exit 1 ;;
  esac
done

if [[ -z "$GUEST_IMAGE" ]]; then
  echo "ERROR: --guest-image is required"
  exit 1
fi

if [[ ! -x "$VFU_BIN" ]]; then
  echo "ERROR: rocjitsu-vfu not found or not executable: $VFU_BIN"
  exit 1
fi

rm -f "$SOCKET"

VFU_PID=""
QEMU_PID=""

cleanup() {
  [[ -n "$QEMU_PID" ]] && kill "$QEMU_PID" 2>/dev/null || true
  [[ -n "$VFU_PID"  ]] && kill "$VFU_PID"  2>/dev/null || true
  rm -f "$SOCKET"
}

if [[ $CLEANUP -eq 1 ]]; then
  trap cleanup EXIT
fi

echo "[vfu-smoke] Starting rocjitsu-vfu..."
"$VFU_BIN" --socket "$SOCKET" --config "$CONFIG" &
VFU_PID=$!

# Give the server a moment to bind the socket.
sleep 1
if ! kill -0 "$VFU_PID" 2>/dev/null; then
  echo "ERROR: rocjitsu-vfu exited prematurely"
  exit 1
fi

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
SSH_OPTS+=" -o ConnectTimeout=120"
[[ -n "$SSH_KEY" ]] && SSH_OPTS+=" -i $SSH_KEY"

QEMU_VFIO_DEV='{"driver":"vfio-user-pci","socket":{"path":"'"$SOCKET"'","type":"unix"}}'

echo "[vfu-smoke] Starting QEMU..."
qemu-system-x86_64 \
  -accel kvm \
  -m 16G \
  -smp 4 \
  -drive "file=$GUEST_IMAGE,format=qcow2,if=virtio" \
  -netdev "user,id=net0,hostfwd=tcp::${SSH_PORT}-:22" \
  -device "virtio-net-pci,netdev=net0" \
  -device "$QEMU_VFIO_DEV" \
  -display none \
  -serial mon:stdio &
QEMU_PID=$!

echo "[vfu-smoke] Waiting for guest SSH to be ready (up to 120s)..."
for i in $(seq 1 60); do
  if ssh $SSH_OPTS -p "$SSH_PORT" root@localhost true 2>/dev/null; then
    break
  fi
  sleep 2
done

ssh $SSH_OPTS -p "$SSH_PORT" root@localhost true || {
  echo "ERROR: Could not SSH into guest after 120s"
  exit 1
}

PASS=0
FAIL=0

run_check() {
  local desc="$1"
  local cmd="$2"
  echo -n "[vfu-smoke] CHECK: $desc ... "
  if ssh $SSH_OPTS -p "$SSH_PORT" root@localhost "$cmd" >/dev/null 2>&1; then
    echo "PASS"
    ((PASS++)) || true
  else
    echo "FAIL"
    ((FAIL++)) || true
  fi
}

# Check 1: lspci shows AMD 0x75C8
run_check "lspci shows AMD MI350P (0x75C8)" \
  "lspci | grep -i '1002:75c8'"

# Check 2: amdgpu driver loaded
run_check "amdgpu driver loaded (no FATAL in dmesg)" \
  "dmesg | grep -v 'FATAL' | grep -q 'amdgpu'"

# Check 3: rocminfo sees the GPU
run_check "rocminfo sees GPU" \
  "rocminfo | grep -q 'gfx950'"

# Check 4: Compile and run HIP vector_add
run_check "HIP vector_add kernel dispatches" "
  cat > /tmp/vector_add.hip << 'EOF'
#include <hip/hip_runtime.h>
#include <stdio.h>
__global__ void vadd(float *a, float *b, float *c, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
int main() {
  const int N = 1024;
  float *da, *db, *dc;
  hipMalloc(&da, N*sizeof(float));
  hipMalloc(&db, N*sizeof(float));
  hipMalloc(&dc, N*sizeof(float));
  float ha[N], hb[N], hc[N];
  for (int i = 0; i < N; i++) { ha[i] = i; hb[i] = N-i; }
  hipMemcpy(da, ha, N*sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(db, hb, N*sizeof(float), hipMemcpyHostToDevice);
  vadd<<<N/64,64>>>(da, db, dc, N);
  hipMemcpy(hc, dc, N*sizeof(float), hipMemcpyDeviceToHost);
  for (int i = 0; i < N; i++) {
    if ((int)hc[i] != N) { printf(\"MISMATCH at %d: %f\\n\",i,hc[i]); return 1; }
  }
  printf(\"PASS\\n\");
  return 0;
}
EOF
  hipcc /tmp/vector_add.hip -o /tmp/vector_add && /tmp/vector_add | grep -q PASS
"

echo ""
echo "[vfu-smoke] Results: PASS=$PASS FAIL=$FAIL"
[[ $FAIL -eq 0 ]]

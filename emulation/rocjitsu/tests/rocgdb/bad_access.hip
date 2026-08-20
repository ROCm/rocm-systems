// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Minimal HIP workload that deliberately faults, used by the rocgdb
// wave-debugging CI harness (tests/rocgdb/debug_kernel.sh) to exercise GPU
// memory-violation reporting. The kernel stores through a pointer to an
// unmapped device address, which the emulator reports to the debugger as a
// memory access fault (rocm-dbgapi WAVE_STOP_REASON_MEMORY_VIOLATION -> SIGSEGV).
//
// Build with device debug info for the emulated target, e.g.:
//   hipcc --offload-arch=gfx950 -g -O0 -o bad_access bad_access.hip
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void bad_access(int *data, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // A wild, never-mapped device address. The store below faults.
  int *wild = reinterpret_cast<int *>(0x0000dead0000ULL);
  if (i < n)
    wild[i] = data[i] + 1;
}

int main() {
  const int n = 64;
  int *d = nullptr;
  if (hipMalloc(&d, n * sizeof(int)) != hipSuccess)
    return 1;
  if (hipMemset(d, 0, n * sizeof(int)) != hipSuccess)
    return 2;
  bad_access<<<dim3(1), dim3(64)>>>(d, n);
  (void)hipDeviceSynchronize();
  hipFree(d);
  std::printf("bad_access done\n");
  return 0;
}

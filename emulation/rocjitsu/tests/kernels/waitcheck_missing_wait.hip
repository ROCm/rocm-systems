// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Deliberately malformed real HIP kernel used by the offline waitcheck E2E
// tests. Keep the producer and consumer in one assembly block so LLVM cannot
// insert the missing scalar-memory wait between them.
#include <hip/hip_runtime.h>

extern "C" __global__ void waitcheck_missing_smem_wait() {
  asm volatile("s_load_dword s4, s[0:1], 0\n\t"
               "s_mov_b32 s5, s4"
               :
               :
               : "s4", "s5", "memory");
}

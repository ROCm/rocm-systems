// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

// The deliberately missing scalar wait must stay between these two instructions.
extern "C" __global__ void waitcheck_missing_smem_fixture() {
  asm volatile("s_load_dword s4, s[0:1], 0\n\t"
               "s_mov_b32 s5, s4"
               :
               :
               : "s4", "s5", "memory");
}

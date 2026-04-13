// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

/// Deliberate SGPR race via GLOBAL_TO_SGPR: s_load_dword loads into an SGPR,
/// then that SGPR is used without s_waitcnt lgkmcnt(0).
extern "C" __global__ void sgpr_race(int *out, const int *in) {
  int sval;
  asm volatile(
      "s_load_dword %0, %1, 0x0\n"
      "s_add_u32 %0, %0, 1\n"
      : "=s"(sval)
      : "s"(in)
      : "memory");
  out[threadIdx.x] = sval;
}

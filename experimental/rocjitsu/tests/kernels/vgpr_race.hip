// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

/// Deliberate VGPR data race: global load followed by use without s_waitcnt.
/// The compiler would normally insert a waitcnt, but inline asm bypasses that.
/// Used to test that the race detector catches missing waitcnt hazards.
extern "C" __global__ void vgpr_race(int *out, const int *in) {
  unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;

  int val;
  asm volatile(
      "global_load_dword %0, %1, off\n"
      "v_add_u32 %0, %0, 1\n"
      : "=v"(val)
      : "v"(&in[gid])
      : "memory");
  out[gid] = val;
}

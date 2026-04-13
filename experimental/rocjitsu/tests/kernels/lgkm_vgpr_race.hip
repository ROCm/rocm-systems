// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

/// Reverse array via LDS: out[i] = in[63-i].
/// All LDS access is C++. The race is injected by using inline asm to issue
/// a second ds_read (without lgkmcnt) that reads the same VGPR destination
/// as the compiler-generated ds_read. This creates a LDS_TO_VGPR race on
/// that VGPR.
extern "C" __global__ void lgkm_vgpr_race(const int *in, int *out) {
  __shared__ int lds[64];
  unsigned tid = threadIdx.x;

  // Write to LDS at reversed position (C++).
  lds[63 - tid] = in[tid];
  __syncthreads();

  // Read from LDS (C++ -- compiler generates ds_read + lgkmcnt).
  int val = lds[tid];

  // Inject a second ds_read into the same VGPR without lgkmcnt.
  // The v_add_u32 reads the VGPR before this ds_read completes.
  asm volatile(
      "ds_read_b32 %0, %1\n"
      "v_add_u32 %0, %0, 0\n"
      : "+v"(val)
      : "v"(tid * 4)
      : "memory");

  out[tid] = val;
}

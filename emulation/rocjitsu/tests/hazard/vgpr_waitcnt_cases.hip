// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "vgpr_waitcnt_cases.hpp"

__global__ void vgpr_no_race_kernel(const float *src, float *dst) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float val;
  asm volatile("global_load_dword %0, %1, off\n"
               "s_waitcnt vmcnt(0)\n"
               : "=v"(val)
               : "v"(&src[tid]));
  dst[tid] = val;
}

__global__ void vgpr_race_kernel(const float *src, float *dst) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float val;
  asm volatile("global_load_dword %0, %1, off\n"
               // BUG: missing s_waitcnt vmcnt(0).
               "s_nop 0\n"
               : "=v"(val)
               : "v"(&src[tid]));
  dst[tid] = val;
}

// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

extern "C" __global__ void lds_roundtrip(const float *__restrict__ A, float *__restrict__ C) {
  __shared__ float tile[256];
  const unsigned lane = threadIdx.x;

  tile[lane] = A[lane] + 1.0f;
  asm volatile("s_waitcnt lgkmcnt(0)\n\t"
               "s_barrier"
               :
               :
               : "memory");

  const unsigned src = (lane + 17) & 255;
  C[lane] = tile[src] + tile[lane];
}

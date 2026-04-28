// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

static __device__ __forceinline__ void lds_loop_barrier() {
  asm volatile("s_waitcnt lgkmcnt(0)\n\t"
               "s_barrier"
               :
               :
               : "memory");
}

/// LDS read/write loop fixture for DBT control-flow tests.
///
/// Each iteration writes a lane-local value to LDS, synchronizes, then reads a
/// shifted lane's value back into the loop-carried accumulator. This keeps the
/// loop small while forcing translated code to preserve scalar loop control,
/// LDS operations, and barriers across the backedge.
extern "C" __global__ void lds_loop(const float *__restrict__ A, float *__restrict__ C) {
  __shared__ float tile[256];
  const unsigned lane = threadIdx.x;

  float value = A[lane];
  for (unsigned iter = 0; iter < 4; ++iter) {
    tile[lane] = value + static_cast<float>(iter);
    lds_loop_barrier();

    const unsigned src = (lane + 17 + iter * 3) & 255;
    value = tile[src] * 0.5f + static_cast<float>(iter + 1);
    lds_loop_barrier();
  }

  C[lane] = value;
}

// Copyright (c) 2026 Advanced Micro Devices, Inc.
// All rights reserved.

#include <hip/hip_runtime.h>

/// Deliberate LDS data race: wave 0 writes, wave 1 reads, no barrier.
/// Used to test that the race detector catches inter-wavefront LDS races.
extern "C" __global__ void lds_race(int *out) {
  __shared__ int lds[64];
  unsigned tid = threadIdx.x;
  unsigned wf = tid / 64;
  unsigned lane = tid % 64;

  if (wf == 0) {
    lds[lane] = static_cast<int>(lane);
  }
  // Missing __syncthreads() -- deliberate race.
  if (wf == 1) {
    out[lane] = lds[lane];
  }
}

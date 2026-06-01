// Copyright (c) 2026 Advanced Micro Devices, Inc.
// All rights reserved.

#include <hip/hip_runtime.h>

/// Element-wise vector addition kernel for multi-workgroup stress testing.
///
/// Each workgroup computes 64 contiguous elements of C = A + B.
/// The global thread ID determines which element each lane computes:
///   global_id = workgroup_id * 64 + thread_id
///
/// With 256 workgroups of 64 threads each, all 16384 elements are covered,
/// distributing one workgroup per compute unit across 256 CUs (8 XCDs).
///
/// Kernel arguments (passed via the argument buffer):
///   A - pointer to input vector A  (N floats)
///   B - pointer to input vector B  (N floats)
///   C - pointer to output vector C (N floats)
///   N - number of elements

extern "C" __global__ void vector_add(const float *__restrict__ A,
                                      const float *__restrict__ B,
                                      float *__restrict__ C, unsigned N) {
  constexpr unsigned BLOCK_SIZE = 64;
  unsigned gid = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  if (gid < N)
    C[gid] = A[gid] + B[gid];
}

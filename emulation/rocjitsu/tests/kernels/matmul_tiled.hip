// Copyright (c) 2026 Advanced Micro Devices, Inc.
// All rights reserved.

#include <hip/hip_runtime.h>

/// Tiled matrix multiplication kernel for multi-workgroup stress testing.
///
/// Each workgroup computes 64 contiguous elements of the output matrix C = A * B.
/// The global thread ID determines which element each lane computes:
///   global_id = workgroup_id * 64 + thread_id
///   row = global_id / N,  col = global_id % N
///
/// With N=128 and 256 workgroups of 64 threads each, all 16384 output elements
/// are covered, distributing one workgroup per compute unit across 256 CUs.
///
/// Kernel arguments (passed via the argument buffer):
///   A        - pointer to input matrix A  (N*N floats, row-major)
///   B        - pointer to input matrix B  (N*N floats, row-major)
///   C        - pointer to output matrix C (N*N floats, row-major)
///   N        - matrix dimension

extern "C" __global__ void matmul_tiled(const float *__restrict__ A,
                                        const float *__restrict__ B,
                                        float *__restrict__ C, unsigned N) {
  // blockDim.x is always 64 (one wavefront per workgroup).
  constexpr unsigned BLOCK_SIZE = 64;
  unsigned gid = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  if (gid < N * N) {
    unsigned row = gid / N;
    unsigned col = gid % N;
    float sum = 0.0f;
    for (unsigned k = 0; k < N; ++k)
      sum += A[row * N + k] * B[k * N + col];
    C[row * N + col] = sum;
  }
}

// Copyright (c) 2026 Advanced Micro Devices, Inc.
// All rights reserved.

#include <hip/hip_runtime.h>

/// Naive matrix multiplication kernel for a small NxN matrix.
///
/// Each wavefront lane computes one element of the output matrix C = A * B.
/// Matrix elements are stored in row-major order.  The kernel is designed for
/// N <= 8 so that all output elements fit within a single wavefront (64 lanes).
///
/// Kernel arguments (passed via the argument buffer):
///   A        - pointer to input matrix A  (N*N floats, row-major)
///   B        - pointer to input matrix B  (N*N floats, row-major)
///   C        - pointer to output matrix C (N*N floats, row-major)
///   N        - matrix dimension

extern "C" __global__ void matmul_naive(const float *__restrict__ A,
                                        const float *__restrict__ B,
                                        float *__restrict__ C, unsigned N) {
  unsigned tid = threadIdx.x;
  unsigned row = tid / N;
  unsigned col = tid % N;

  if (tid < N * N) {
    float sum = 0.0f;
    for (unsigned k = 0; k < N; ++k)
      sum += A[row * N + k] * B[k * N + col];
    C[row * N + col] = sum;
  }
}

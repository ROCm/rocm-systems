/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
 This code object should be automatically built via "make build_tests".
 In case it's missing, please type the following to generate it,
   /opt/rocm/hip/bin/hipcc --cuda-device-only hipMatMul.cc -o hipMatMul.code
*/
#include <hip/hip_runtime.h>
__device__ int deviceGlobal = 1;

extern "C" __global__ void matmulK(int* A, int* B, int* C, int N) {
  int ROW = blockIdx.y * blockDim.y + threadIdx.y;
  int COL = blockIdx.x * blockDim.x + threadIdx.x;
  int tmpSum = 0;
  if ((ROW < N) && (COL < N)) {
    // each thread computes one element of the block sub-matrix
    for (int i = 0; i < N; i++) {
      tmpSum += A[ROW * N + i] * B[i * N + COL];
    }
    C[ROW * N + COL] = tmpSum;
  }
}

extern "C" __global__ void KernelandExtraParams(int* A, int* B, int* C, int* D, int N) {
  int ROW = blockIdx.y * blockDim.y + threadIdx.y;
  int COL = blockIdx.x * blockDim.x + threadIdx.x;
  int tmpSum = 0;
  if (ROW < N && COL < N) {
    // each thread computes one element of the block sub-matrix
    for (int i = 0; i < N; i++) {
      tmpSum += A[ROW * N + i] * B[i * N + COL];
    }
  }
  C[ROW * N + COL] = tmpSum;
  D[ROW * N + COL] = tmpSum;
}

extern "C" __global__ void dummyKernel() {}

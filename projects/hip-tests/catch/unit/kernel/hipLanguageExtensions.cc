/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_kernels.hh>
#include <hip_test_common.hh>
#include <hip_test_checkers.hh>

#include <hip/math_functions.h>

#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wuninitialized"

// Simple tests for variable type qualifiers:
__device__ int deviceVar;

// TODO-HCC __constant__ not working yet.
__constant__ int constantVar1;

__constant__ __device__ int constantVar2;

// Test HOST space:
__host__ void foo() { printf("foo!\n"); }

__device__ __noinline__ int sum1_noinline(int a) { return a + 1; }
__device__ __forceinline__ int sum1_forceinline(int a) { return a + 1; }


__device__ __host__ float PlusOne(float x) { return x + 1.0; }

__global__ void MyKernel(const float* a, const float* b, float* c, unsigned N) {
  unsigned gid = threadIdx.x;
  if (gid < N) {
    c[gid] = a[gid] + PlusOne(b[gid]);
  }
}

void callMyKernel() {
  float *a, *b, *c;
  const unsigned blockSize = 256;
  unsigned N = blockSize;
  hipLaunchKernelGGL(MyKernel, dim3(N / blockSize), dim3(blockSize), 0, 0, a, b, c, N);
}

template <typename T> __global__ void vectorADD(T __restrict__* A_d, T* B_d, T* C_d, size_t N) {
#ifdef NOT_YET
  int a = __shfl_up(x, 1);
#endif
  float x = 1.0;
#ifdef NOT_YET
  float fastZ = __sin(x);
#endif
  __syncthreads();

  size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = offset; i < N; i += stride) {
    C_d[i] = A_d[i] + B_d[i];
  }
}

/**
* @addtogroup hipLaunchKernelGGL hipLaunchKernelGGL
* @{
* @ingroup KernelTest
* `void hipLaunchKernelGGL(F kernel, const dim3& numBlocks, const dim3& dimBlocks,
   std::uint32_t sharedMemBytes, hipStream_t stream, Args... args)` -
* Method to invocate kernel functions
*/

/**
 * Test Description
 * ------------------------
 *    - Collection of code to make sure that various features
 * in the hip kernel language compile.

 * Test source
 * ------------------------
 *    - catch/unit/kernel/hipLanguageExtensions.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.5
 */

TEST_CASE(Unit_hipLanguageExtensions) { REQUIRE(true); }

/**
 * End doxygen group KernelTest.
 * @}
 */

/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#define NUM 1024
#define SIZE (NUM * sizeof(float))

__global__ static void vAdd(float* In1, float* In2, float* In3, float* In4, float* Out) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  In4[tid] = In1[tid] + In2[tid];
  __threadfence();
  In3[tid] = In3[tid] + In4[tid];
  __threadfence_block();
  Out[tid] = In4[tid] + In3[tid];
}

HIP_TEST_CASE(Unit_hipThreadFence) {
  float* In1 = new float[NUM];
  float* In2 = new float[NUM];
  float* In3 = new float[NUM];
  float* In4 = new float[NUM];
  float* Out = new float[NUM];
  // Initialization
  for (uint32_t i = 0; i < NUM; i++) {
    In1[i] = 1.0f;
    In2[i] = 1.0f;
    In3[i] = 1.0f;
    In4[i] = 1.0f;
  }

  float *In1d, *In2d, *In3d, *In4d, *Outd;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&In1d), SIZE));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&In2d), SIZE));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&In3d), SIZE));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&In4d), SIZE));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Outd), SIZE));

  HIP_CHECK(hipMemcpy(In1d, In1, SIZE, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(In2d, In2, SIZE, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(In3d, In3, SIZE, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(In4d, In4, SIZE, hipMemcpyHostToDevice));

  hipLaunchKernelGGL(vAdd, dim3(32, 1, 1), dim3(32, 1, 1), 0, 0, In1d, In2d, In3d, In4d, Outd);
  HIP_CHECK(hipMemcpy(Out, Outd, SIZE, hipMemcpyDeviceToHost));
  for (uint32_t i = 0; i < NUM; i++) {
    REQUIRE(Out[i] == 2 * In1[i] + 2 * In2[i] + In3[i]);
  }
  delete[] In1;
  delete[] In2;
  delete[] In3;
  delete[] In4;
  delete[] Out;
  HIP_CHECK(hipFree(In1d));
  HIP_CHECK(hipFree(In2d));
  HIP_CHECK(hipFree(In3d));
  HIP_CHECK(hipFree(In4d));
  HIP_CHECK(hipFree(Outd));
}

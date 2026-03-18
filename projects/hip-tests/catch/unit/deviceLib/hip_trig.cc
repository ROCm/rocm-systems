/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip/device_functions.h>
#define LEN 512
#define SIZE (LEN << 2)
__global__ static void kernel_trig(float* In, float* sin_d, float* cos_d, float* tan_d,
                                   float* sin_pd, float* cos_pd) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  sin_d[tid] = sinf(In[tid]);
  cos_d[tid] = cosf(In[tid]);
  tan_d[tid] = tanf(In[tid]);
  sincosf(In[tid], &sin_pd[tid], &cos_pd[tid]);
}
TEST_CASE(Unit_kernel_trigger) {
  float *In, *sin_h, *cos_h, *tan_h, *sin_ph, *cos_ph;
  float *In_d, *sin_d, *cos_d, *tan_d, *sin_pd, *cos_pd;
  int errors = 0;
  In = new float[LEN];
  sin_h = new float[LEN];
  cos_h = new float[LEN];
  tan_h = new float[LEN];
  sin_ph = new float[LEN];
  cos_ph = new float[LEN];
  for (int i = 0; i < LEN; i++) {
    In[i] = 1.0f;
    sin_h[i] = 0.0f;
    cos_h[i] = 0.0f;
    tan_h[i] = 0.0f;
    sin_ph[i] = 0.0f;
    cos_ph[i] = 0.0f;
  }
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&In_d), SIZE));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&sin_d), SIZE));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&cos_d), SIZE));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&tan_d), SIZE));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&sin_pd), SIZE));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&cos_pd), SIZE));

  HIP_CHECK(hipMemcpy(In_d, In, SIZE, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(kernel_trig, dim3(LEN, 1, 1), dim3(1, 1, 1), 0, 0, In_d, sin_d, cos_d, tan_d,
                     sin_pd, cos_pd);
  HIP_CHECK(hipMemcpy(sin_h, sin_d, SIZE, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(cos_h, cos_d, SIZE, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(tan_h, tan_d, SIZE, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(sin_ph, sin_pd, SIZE, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(cos_ph, cos_pd, SIZE, hipMemcpyDeviceToHost));
  // Validation
  for (int i = 0; i < LEN; i++) {
    if (sin_h[i] != sin_ph[i] || cos_h[i] != cos_ph[i] || tan_h[i] * cos_h[i] != sin_h[i]) {
      errors++;
    }
  }
  REQUIRE(errors == 0);
  delete[] In;
  delete[] sin_h;
  delete[] cos_h;
  delete[] tan_h;
  delete[] sin_ph;
  delete[] cos_ph;
  HIP_CHECK(hipFree(In_d));
  HIP_CHECK(hipFree(sin_d));
  HIP_CHECK(hipFree(cos_d));
  HIP_CHECK(hipFree(tan_d));
  HIP_CHECK(hipFree(sin_pd));
  HIP_CHECK(hipFree(cos_pd));
}

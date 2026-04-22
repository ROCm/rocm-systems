/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma clang diagnostic ignored "-Wunused-parameter"
#include <hip_test_common.hh>

#define N 512

static __global__ void tex1dKernel(float* val, hipTextureObject_t obj) {
#if !__HIP_NO_IMAGE_SUPPORT
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < N) {
    val[k] = tex1Dfetch<float>(obj, k);
  }
#endif
}


HIP_TEST_CASE(Unit_hipCreateTextureObject_tex1DfetchVerification) {
  CHECK_IMAGE_SUPPORT
  (void) hipGetLastError();  // Prevent negative tests affecting this

  // Allocating the required buffer on gpu device
  float *texBuf, *texBufOut;
  float val[N], output[N];

  for (int i = 0; i < N; i++) {
    val[i] = (i + 1) * (i + 1);
    output[i] = 0.0;
  }

  HIP_CHECK(hipMalloc(&texBuf, N * sizeof(float)));
  HIP_CHECK(hipMalloc(&texBufOut, N * sizeof(float)));
  HIP_CHECK(hipMemcpy(texBuf, val, N * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(texBufOut, 0, N * sizeof(float)));
  hipResourceDesc resDescLinear;

  memset(&resDescLinear, 0, sizeof(resDescLinear));
  resDescLinear.resType = hipResourceTypeLinear;
  resDescLinear.res.linear.devPtr = texBuf;
  resDescLinear.res.linear.desc = hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindFloat);
  resDescLinear.res.linear.sizeInBytes = N * sizeof(float);

  hipTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = hipReadModeElementType;

  // Creating texture object
  hipTextureObject_t texObj = 0;
  HIP_CHECK(hipCreateTextureObject(&texObj, &resDescLinear, &texDesc, NULL));

  dim3 dimBlock(64, 1, 1);
  dim3 dimGrid(N / dimBlock.x, 1, 1);

  hipLaunchKernelGGL(tex1dKernel, dim3(dimGrid), dim3(dimBlock), 0, 0, texBufOut, texObj);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(output, texBufOut, N * sizeof(float), hipMemcpyDeviceToHost));

  for (int i = 0; i < N; i++) {
    if (output[i] != val[i]) {
      INFO("Mismatch at index : " << i << ", output[i] " << output[i] << ", val[i] " << val[i]);
      REQUIRE(false);
    }
  }

  HIP_CHECK(hipDestroyTextureObject(texObj));
  HIP_CHECK(hipFree(texBuf));
  HIP_CHECK(hipFree(texBufOut));
}

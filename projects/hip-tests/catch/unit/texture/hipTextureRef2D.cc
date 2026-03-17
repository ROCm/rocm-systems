/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma clang diagnostic ignored "-Wunused-parameter"
#include <hip_test_common.hh>


#if CUDA_VERSION < CUDA_12000

texture<float, 2, hipReadModeElementType> tex;

__global__ void tex2DKernel(float* outputData, int width) {
#if !__HIP_NO_IMAGE_SUPPORT
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  outputData[y * width + x] = tex2D(tex, x, y);
#endif
}

TEST_CASE(Unit_hipTextureRef2D_Check) {
  CHECK_IMAGE_SUPPORT

  constexpr int SIZE = 256;
  constexpr unsigned int width = SIZE;
  constexpr unsigned int height = SIZE;
  constexpr unsigned int size = width * height * sizeof(float);
  unsigned int i, j;

  float* hData = reinterpret_cast<float*>(malloc(size));
  REQUIRE(hData != nullptr);
  memset(hData, 0, size);
  for (i = 0; i < height; i++) {
    for (j = 0; j < width; j++) {
      hData[i * width + j] = i * width + j;
    }
  }

  hipChannelFormatDesc channelDesc = hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindFloat);
  hipArray_t hipArray;
  HIP_CHECK(hipMallocArray(&hipArray, &channelDesc, width, height));
  HIP_CHECK(hipMemcpyToArray(hipArray, 0, 0, hData, size, hipMemcpyHostToDevice));

  tex.addressMode[0] = hipAddressModeWrap;
  tex.addressMode[1] = hipAddressModeWrap;
  tex.filterMode = hipFilterModePoint;
  tex.normalized = 0;

  HIP_CHECK(hipBindTextureToArray(tex, hipArray, channelDesc));

  float* dData = nullptr;
  HIP_CHECK(hipMalloc(&dData, size));
  REQUIRE(dData != nullptr);

  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);
  hipLaunchKernelGGL(tex2DKernel, dim3(dimGrid), dim3(dimBlock), 0, 0, dData, width);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  float* hOutputData = reinterpret_cast<float*>(malloc(size));
  REQUIRE(hOutputData != nullptr);
  memset(hOutputData, 0, size);
  HIP_CHECK(hipMemcpy(hOutputData, dData, size, hipMemcpyDeviceToHost));

  for (i = 0; i < height; i++) {
    for (j = 0; j < width; j++) {
      if (hData[i * width + j] != hOutputData[i * width + j]) {
        INFO("Difference found at [ " << i << j << " ]: " << hData[i * width + j]
                                      << hOutputData[i * width + j]);
        REQUIRE(false);
      }
    }
  }
  HIP_CHECK(hipUnbindTexture(tex));
  HIP_CHECK(hipFree(dData));
  HIP_CHECK(hipFreeArray(hipArray));
}


#endif  // CUDA_VERSION < CUDA_12000

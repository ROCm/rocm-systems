/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma clang diagnostic ignored "-Wunused-parameter"
#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_texture_helper.hh>

/**
 * @addtogroup hipCreateTextureObject hipCreateTextureObject
 * @{
 * @ingroup TextureTest
 */

template <bool normalizedCoords> __global__ void tex1DKernel(float* outputData,
                                                             hipTextureObject_t textureObject,
                                                             int width, float offsetX) {
#if !__HIP_NO_IMAGE_SUPPORT
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  outputData[x] =
      tex1D<float>(textureObject, normalizedCoords ? (x + offsetX) / width : x + offsetX);
#endif
}

template <hipTextureAddressMode addressMode, hipTextureFilterMode filterMode, bool normalizedCoords>
static void runTest(const int width, const float offsetX) {
  // printf("%s(addressMode=%d, filterMode=%d, normalizedCoords=%d, width=%d, offsetX=%f)\n",
  // __FUNCTION__,
  //        addressMode, filterMode, normalizedCoords, width, offsetX);
  unsigned int size = width * sizeof(float);
  float* hData = (float*)malloc(size);
  memset(hData, 0, size);
  for (int j = 0; j < width; j++) {
    hData[j] = j;
  }

  hipChannelFormatDesc channelDesc = hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindFloat);
  hipArray_t hipArray;
  HIP_CHECK(hipMallocArray(&hipArray, &channelDesc, width));

  HIP_CHECK(hipMemcpy2DToArray(hipArray, 0, 0, hData, width * sizeof(float), width * sizeof(float),
                               1, hipMemcpyHostToDevice));

  hipResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = hipResourceTypeArray;
  resDesc.res.array.array = hipArray;

  // Specify texture object parameters
  hipTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = addressMode;
  texDesc.filterMode = filterMode;
  texDesc.readMode = hipReadModeElementType;
  texDesc.normalizedCoords = normalizedCoords;

  // Create texture object
  hipTextureObject_t textureObject = 0;
  HIP_CHECK(hipCreateTextureObject(&textureObject, &resDesc, &texDesc, NULL));

  float* dData = nullptr;
  HIP_CHECK(hipMalloc((void**)&dData, size));

  dim3 dimBlock(16, 1, 1);
  dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, 1, 1);

  hipLaunchKernelGGL(tex1DKernel<normalizedCoords>, dimGrid, dimBlock, 0, 0, dData, textureObject,
                     width, offsetX);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipDeviceSynchronize());

  float* hOutputData = (float*)malloc(size);
  memset(hOutputData, 0, size);
  HIP_CHECK(hipMemcpy(hOutputData, dData, size, hipMemcpyDeviceToHost));

  bool result = true;
  for (int j = 0; j < width; j++) {
    float expectedValue =
        getExpectedValue<float, addressMode, filterMode>(width, offsetX + j, hData);
    if (!hipTextureSamplingVerify<float, filterMode>(hOutputData[j], expectedValue)) {
      INFO("Mismatch at " << offsetX + j << ":" << hOutputData[j] << " expected:" << expectedValue);
      result = false;
      break;
    }
  }

  HIP_CHECK(hipDestroyTextureObject(textureObject));
  HIP_CHECK(hipFree(dData));
  HIP_CHECK(hipFreeArray(hipArray));
  free(hData);
  free(hOutputData);
  REQUIRE(result);
}

/**
 * Test Description
 * ------------------------
 *  - Uses different addressing and filtering modes for 1D array.
 * Test source
 * ------------------------
 *  - unit/texture/hipTextureObj1DCheckModes.cc
 * Test requirements
 * ------------------------
 *  - Textures supported on device
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_hipTextureObj1DCheckModes) {
  CHECK_IMAGE_SUPPORT

  SECTION("hipAddressModeClamp, hipFilterModePoint, regularCoords") {
    runTest<hipAddressModeClamp, hipFilterModePoint, false>(256, -3);
    runTest<hipAddressModeClamp, hipFilterModePoint, false>(256, 4);
  }

  SECTION("hipAddressModeBorder, hipFilterModePoint, regularCoords") {
    runTest<hipAddressModeBorder, hipFilterModePoint, false>(256, -8.5);
    runTest<hipAddressModeBorder, hipFilterModePoint, false>(256, 12.5);
  }

  SECTION("hipAddressModeClamp, hipFilterModeLinear, regularCoords") {
    runTest<hipAddressModeClamp, hipFilterModeLinear, false>(256, -3);
    runTest<hipAddressModeClamp, hipFilterModeLinear, false>(256, 4);
  }

  SECTION("hipAddressModeBorder, hipFilterModeLinear, regularCoords") {
    runTest<hipAddressModeBorder, hipFilterModeLinear, false>(256, -8.5);
    runTest<hipAddressModeBorder, hipFilterModeLinear, false>(256, 12.5);
  }

  SECTION("hipAddressModeClamp, hipFilterModePoint, normalizedCoords") {
    runTest<hipAddressModeClamp, hipFilterModePoint, true>(256, -3);
    runTest<hipAddressModeClamp, hipFilterModePoint, true>(256, 4);
  }

  SECTION("hipAddressModeBorder, hipFilterModePoint, normalizedCoords") {
    runTest<hipAddressModeBorder, hipFilterModePoint, true>(256, -8.5);
    runTest<hipAddressModeBorder, hipFilterModePoint, true>(256, 12.5);
  }

  SECTION("hipAddressModeClamp, hipFilterModeLinear, normalizedCoords") {
    runTest<hipAddressModeClamp, hipFilterModeLinear, true>(256, -3);
    runTest<hipAddressModeClamp, hipFilterModeLinear, true>(256, 4);
  }

  SECTION("hipAddressModeBorder, hipFilterModeLinear, normalizedCoords") {
    runTest<hipAddressModeBorder, hipFilterModeLinear, true>(256, -8.5);
    runTest<hipAddressModeBorder, hipFilterModeLinear, true>(256, 12.5);
  }
}

/**
 * End doxygen group TextureTest.
 * @}
 */

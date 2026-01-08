/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma clang diagnostic ignored "-Wunused-parameter"
#include <hip_test_common.hh>
#include <hip_array_common.hh>
#include <hip_test_checkers.hh>
#include <hip_texture_helper.hh>
#include "texture_common.h"
// #define DEBUG_DATA

template <typename TestType>
__global__ void simpleKernelLayered2DArray(hipTextureObject_t tex, TestType* outputData,
                                           unsigned int width, unsigned int height,
                                           unsigned int layer) {
#if !__HIP_NO_IMAGE_SUPPORT
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  outputData[layer * width * height + y * width + x] = tex2DLayered<TestType>(tex, x, y, layer);
#endif
}

enum class BufferType { Host, Device };

template <typename TestType>
void runLayered2DTextureTest(BufferType bufferType) {
  constexpr int SIZE = 512;
  constexpr int num_layers = 5;
  constexpr unsigned int width = SIZE;
  constexpr unsigned int height = SIZE;
  constexpr unsigned int size = width * height * num_layers * sizeof(TestType);

  // Allocate and initialize host data
  TestType* hData = reinterpret_cast<TestType*>(malloc(size));
  REQUIRE(hData != nullptr);
  memset(hData, 0, size);
  for (unsigned int layer = 0; layer < num_layers; layer++) {
    for (unsigned int i = 0; i < width * height; i++) {
      initVal(hData[layer * width * height + i]);
    }
  }

  // Allocate device buffer if needed
  TestType* dData = nullptr;
  TestType* srcData = nullptr;  // Points to the source buffer (host or device)
  hipMemcpyKind srcToArrayKind, arrayToBufferKind;

  if (bufferType == BufferType::Device) {
    HIP_CHECK(hipMalloc(&dData, size));
    HIP_CHECK(hipMemcpy(dData, hData, size, hipMemcpyHostToDevice));
    srcData = dData;
    srcToArrayKind = hipMemcpyDeviceToDevice;
    arrayToBufferKind = hipMemcpyDeviceToDevice;
  } else {
    srcData = hData;
    srcToArrayKind = hipMemcpyHostToDevice;
    arrayToBufferKind = hipMemcpyDeviceToHost;
  }

  // Create layered array
  hipChannelFormatDesc channelDesc = hipCreateChannelDesc<TestType>();
  hipArray_t arr;
  HIP_CHECK(hipMalloc3DArray(&arr, &channelDesc, make_hipExtent(width, height, num_layers),
                             hipArrayLayered));
  hipMemcpy3DParms myparms{};

  SECTION("hipMemcpy3D whole layers") {
    myparms.srcPos = make_hipPos(0, 0, 0);
    myparms.dstPos = make_hipPos(0, 0, 0);
    myparms.srcPtr = make_hipPitchedPtr(srcData, width * sizeof(TestType), width, height);
    myparms.dstArray = arr;
    myparms.extent = make_hipExtent(width, height, num_layers);
    myparms.kind = srcToArrayKind;
    HIP_CHECK(hipMemcpy3D(&myparms));
  }

  SECTION("hipMemcpy3D layer by layer") {
    constexpr unsigned int layerSize = width * height * sizeof(TestType);
    TestType* hLayerData = reinterpret_cast<TestType*>(malloc(layerSize));
    REQUIRE(hLayerData != nullptr);

    TestType* dData1 = nullptr;
    if (bufferType == BufferType::Device) {
      HIP_CHECK(hipMalloc(&dData1, size));
      HIP_CHECK(hipMemset(dData1, 0, size));
    }

    for (unsigned int layer = 0; layer < num_layers; layer++) {
      // Copy buffer layer to image layer
      memset(hLayerData, 0, layerSize);
      memset(&myparms, 0, sizeof(myparms));
      myparms.srcPos = make_hipPos(0, 0, layer);
      myparms.dstPos = make_hipPos(0, 0, layer);
      myparms.srcPtr = make_hipPitchedPtr(srcData, width * sizeof(TestType), width, height);
      myparms.dstArray = arr;
      myparms.extent = make_hipExtent(width, height, 1);
      myparms.kind = srcToArrayKind;
      HIP_CHECK(hipMemcpy3D(&myparms));

      // Copy image layer to buffer layer
      memset(&myparms, 0, sizeof(myparms));
      myparms.srcPos = make_hipPos(0, 0, layer);
      if (bufferType == BufferType::Device) {
        myparms.dstPos = make_hipPos(0, 0, layer);
        myparms.srcArray = arr;
        myparms.dstPtr = make_hipPitchedPtr(dData1, width * sizeof(TestType), width, height);
        myparms.extent = make_hipExtent(width, height, 1);
        myparms.kind = arrayToBufferKind;
        HIP_CHECK(hipMemcpy3D(&myparms));
        HIP_CHECK(hipMemcpy(hLayerData, dData1 + layer * width * height, layerSize,
                           hipMemcpyDeviceToHost));
      } else {
        myparms.dstPos = make_hipPos(0, 0, 0);
        myparms.srcArray = arr;
        myparms.dstPtr = make_hipPitchedPtr(hLayerData, width * sizeof(TestType), width, height);
        myparms.extent = make_hipExtent(width, height, 1);
        myparms.kind = arrayToBufferKind;
        HIP_CHECK(hipMemcpy3D(&myparms));
      }

      // Compare layer
#ifdef DEBUG_DATA
      for (unsigned int i = 0; i < width * height; i++) {
        fprintf(stderr, "%4u: %u: %s -- %s\n", layer, i,
                getString(hData[layer * width * height + i]).c_str(),
                getString(hLayerData[i]).c_str());
      }
#endif
      REQUIRE(HipTest::checkArray(hData + layer * width * height, hLayerData, width, height, 1));
    }
    free(hLayerData);
    if (dData1 != nullptr) {
      HIP_CHECK(hipFree(dData1));
    }
  }

  // Create texture object
  hipTextureObject_t tex;
  hipResourceDesc texRes;
  memset(&texRes, 0, sizeof(hipResourceDesc));
  texRes.resType = hipResourceTypeArray;
  texRes.res.array.array = arr;

  hipTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(hipTextureDesc));
  texDescr.normalizedCoords = 0;
  texDescr.filterMode = hipFilterModePoint;
  texDescr.addressMode[0] = hipAddressModeClamp;
  texDescr.addressMode[1] = hipAddressModeClamp;
  texDescr.addressMode[2] = hipAddressModeClamp;
  texDescr.readMode = hipReadModeElementType;
  HIP_CHECK(hipCreateTextureObject(&tex, &texRes, &texDescr, NULL));

  // Allocate device memory for result
  TestType* dOutputData = nullptr;
  HIP_CHECK(hipMalloc(&dOutputData, size));
  if (bufferType == BufferType::Device) {
    HIP_CHECK(hipMemset(dOutputData, 0, size));
  }

  // Launch kernel
  dim3 dimBlock(32, 32);
  dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
  for (unsigned int layer = 0; layer < num_layers; layer++) {
    hipLaunchKernelGGL(simpleKernelLayered2DArray<TestType>, dimGrid, dimBlock, 0, 0, tex,
                       dOutputData, width, height, layer);
    HIP_CHECK(hipGetLastError());
  }
  HIP_CHECK(hipDeviceSynchronize());

  // Allocate mem for the result on host side
  TestType* hOutputData = reinterpret_cast<TestType*>(malloc(size));
  REQUIRE(hOutputData != nullptr);
  memset(hOutputData, 0, size);

  // Copy result from device to host
  HIP_CHECK(hipMemcpy(hOutputData, dOutputData, size, hipMemcpyDeviceToHost));
  REQUIRE(HipTest::checkArray(hData, hOutputData, width, height, num_layers) == true);

  // Cleanup
  HIP_CHECK(hipFree(dOutputData));
  HIP_CHECK(hipFreeArray(arr));
  free(hData);
  free(hOutputData);
  HIP_CHECK(hipDestroyTextureObject(tex));
  if (dData != nullptr) {
    HIP_CHECK(hipFree(dData));
  }
}

/**
 * Test Description
 * ------------------------
 * - The suite will test host buffer copied to/from layered 2D array in following steps,
       allocating a host buffer,
       creating layered array,
       copying host buffer to the layered array in two ways,
         copying whole host buffer to the layered array,
         copying host buffer layer by layer to the layered array and then
         copying & verifying layer data,
       creating a texture object on the layered array,
       getting the data from the texture object in kernel,
       verifing the data in host
 * Test source
 * ------------------------
 * - catch\unit\texture\hipSimpleTexture2DLayered.cc
 * Test requirements
 * ------------------------
 *  - Host specific (WINDOWS and LINUX)
 *  - Layered 2D array supported on device
 *  - Textures supported on device
 *  - HIP_VERSION >= 6.0
 */
TEMPLATE_TEST_CASE("Unit_Layered2DTexture_Check_HostBufferToFromLayered2DArray", "", char,
                   unsigned char, short, unsigned short, int, unsigned int, float) {
  CHECK_IMAGE_SUPPORT
  runLayered2DTextureTest<vector_type_helper_t<TestType, 0>>(BufferType::Host);
  runLayered2DTextureTest<vector_type_helper_t<TestType, 1>>(BufferType::Host);
  runLayered2DTextureTest<vector_type_helper_t<TestType, 2>>(BufferType::Host);
  runLayered2DTextureTest<vector_type_helper_t<TestType, 4>>(BufferType::Host);
}

/**
 * Test Description
 * ------------------------
 * - The suite will test device buffer copied to/from layered 2D array in following steps,
       allocating host buffer,
       allocating device buffer,
       copying host buffer to device buffer,
       creating layered array,
       copying device buffer to the layered array in two ways,
         copying whole device buffer to the layered array,
         copying device buffer layer by layer to the layered array and then
         copying & verifying layer data,
       creating a texture object on the layered array,
       getting the data from the texture object in kernel,
       verifing the data in host
 * Test source
 * ------------------------
 * - catch\unit\texture\hipSimpleTexture2DLayered.cc
 * Test requirements
 * ------------------------
 *  - Host specific (WINDOWS and LINUX)
 *  - Layered 2D array supported on device
 *  - Textures supported on device
 *  - HIP_VERSION >= 6.0
 */
TEMPLATE_TEST_CASE("Unit_Layered2DTexture_Check_DeviceBufferToFromLayered2DArray", "", char,
                   unsigned char, short, unsigned short, int, unsigned int, float) {
  CHECK_IMAGE_SUPPORT
  runLayered2DTextureTest<vector_type_helper_t<TestType, 0>>(BufferType::Device);
  runLayered2DTextureTest<vector_type_helper_t<TestType, 1>>(BufferType::Device);
  runLayered2DTextureTest<vector_type_helper_t<TestType, 2>>(BufferType::Device);
  runLayered2DTextureTest<vector_type_helper_t<TestType, 4>>(BufferType::Device);
}

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
// #define DEBUG_DATA

template <typename TestType>
__global__ void simpleKernelLayered1DArray(hipTextureObject_t tex, TestType* outputData,
                                           unsigned int width, unsigned int layer) {
#if !__HIP_NO_IMAGE_SUPPORT
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  outputData[layer * width + x] = tex1DLayered<TestType>(tex, x, layer);
#endif
}

/**
 * Test Description
 * ------------------------
 * - The suite will test host buffer copied to/from layered 1D array in following steps,
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
 * - catch\unit\texture\hipSimpleTexture1DLayered.cc
 * Test requirements
 * ------------------------
 *  - Host specific (WINDOWS and LINUX)
 *  - Layered 1D array supported on device
 *  - Textures supported on device
 *  - HIP_VERSION >= 6.0
 */
TEMPLATE_TEST_CASE("Unit_Layered1DTexture_Check_HostBufferToFromLayered1DArray", "", char,
                   unsigned char, short, ushort, int, uint, float, char1, uchar1, short1, ushort1,
                   int1, uint1, float1, char2, uchar2, short2, ushort2, int2, uint2, float2, char4,
                   uchar4, short4, ushort4, int4, uint4, float4) {
  CHECK_IMAGE_SUPPORT

#if __HIP_NO_IMAGE_SUPPORT
  HipTest::HIP_SKIP_TEST("__HIP_NO_IMAGE_SUPPORT is set");
  return;
#endif

  constexpr int SIZE = 512;
  constexpr int num_layers = 5;
  constexpr unsigned int width = SIZE;
  constexpr unsigned int size = width * num_layers * sizeof(TestType);
  TestType* hData = reinterpret_cast<TestType*>(malloc(size));
  REQUIRE(hData != nullptr);
  memset(hData, 0, size);
  for (unsigned int layer = 0; layer < num_layers; layer++) {
    for (unsigned int i = 0; i < width; i++) {
      initVal(hData[layer * width + i]);
    }
  }
  hipChannelFormatDesc channelDesc;
  // Allocate array and copy image data
  channelDesc = hipCreateChannelDesc<TestType>();
  hipArray_t arr;
  HIP_CHECK(
      hipMalloc3DArray(&arr, &channelDesc, make_hipExtent(width, 0, num_layers), hipArrayLayered));
  hipMemcpy3DParms myparms{};

  SECTION("hipMemcpy3D whole layers") {
    myparms.srcPos = make_hipPos(0, 0, 0);
    myparms.dstPos = make_hipPos(0, 0, 0);
    myparms.srcPtr = make_hipPitchedPtr(hData, width * sizeof(TestType), width, 1);
    myparms.dstArray = arr;
    myparms.extent = make_hipExtent(width, 1, num_layers);
    myparms.kind = hipMemcpyHostToDevice;
    HIP_CHECK(hipMemcpy3D(&myparms));
  }

  SECTION("hipMemcpy3D layer by layer") {
    constexpr unsigned int layerSize = width * sizeof(TestType);
    TestType* hLayerData = reinterpret_cast<TestType*>(malloc(layerSize));
    REQUIRE(hLayerData != nullptr);
    for (unsigned int layer = 0; layer < num_layers; layer++) {
      // Copy buffer layer to image layer
      memset(hLayerData, 0, layerSize);
      memset(&myparms, 0, sizeof(myparms));
      myparms.srcPos = make_hipPos(0, 0, layer);
      myparms.dstPos = make_hipPos(0, 0, layer);
      myparms.srcPtr = make_hipPitchedPtr(hData, width * sizeof(TestType), width, 1);
      myparms.dstArray = arr;
      myparms.extent = make_hipExtent(width, 1, 1);
      myparms.kind = hipMemcpyHostToDevice;
      HIP_CHECK(hipMemcpy3D(&myparms));

      // Copy image layer to buffer layer
      memset(&myparms, 0, sizeof(myparms));
      myparms.srcPos = make_hipPos(0, 0, layer);
      myparms.dstPos = make_hipPos(0, 0, 0);
      myparms.srcArray = arr;
      myparms.dstPtr = make_hipPitchedPtr(hLayerData, width * sizeof(TestType), width, 1);
      myparms.extent = make_hipExtent(width, 1, 1);
      myparms.kind = hipMemcpyDeviceToHost;
      HIP_CHECK(hipMemcpy3D(&myparms));

      // Compare layer
#ifdef DEBUG_DATA
      for (unsigned int i = 0; i < width; i++) {
        fprintf(stderr, "%4u: %u: %s -- %s\n", layer, i, getString(hData[layer * width + i]),
                getString(hLayerData[i]));
      }
#endif
      REQUIRE(HipTest::checkArray(hData + layer * width, hLayerData, width, 1, 1));
    }
    free(hLayerData);
  }

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
  TestType* dData = nullptr;
  HIP_CHECK(hipMalloc(&dData, size));

  dim3 dimBlock(8);
  dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x);
  for (unsigned int layer = 0; layer < num_layers; layer++) {
    hipLaunchKernelGGL(simpleKernelLayered1DArray<TestType>, dimGrid, dimBlock, 0, 0, tex, dData,
                       width, layer);
    HIP_CHECK(hipGetLastError());
  }
  HIP_CHECK(hipDeviceSynchronize());

  // Allocate mem for the result on host side
  TestType* hOutputData = reinterpret_cast<TestType*>(malloc(size));
  REQUIRE(hOutputData != nullptr);

  // Copy result from device to host
  HIP_CHECK(hipMemcpy(hOutputData, dData, size, hipMemcpyDeviceToHost));
  REQUIRE(HipTest::checkArray(hData, hOutputData, width, 1, num_layers));

  HIP_CHECK(hipFree(dData));
  HIP_CHECK(hipFreeArray(arr));
  free(hData);
  free(hOutputData);
  HIP_CHECK(hipDestroyTextureObject(tex));
}

/**
 * Test Description
 * ------------------------
 * - The suite will test device buffer copied to/from layered 1D array in following steps,
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
 * - catch\unit\texture\hipSimpleTexture1DLayered.cc
 * Test requirements
 * ------------------------
 *  - Host specific (WINDOWS and LINUX)
 *  - Layered 1D array supported on device
 *  - Textures supported on device
 *  - HIP_VERSION >= 6.0
 */
TEMPLATE_TEST_CASE("Unit_Layered1DTexture_Check_DeviceBufferToFromLayered1DArray", "", char,
                   unsigned char, short, ushort, int, uint, float, char1, uchar1, short1, ushort1,
                   int1, uint1, float1, char2, uchar2, short2, ushort2, int2, uint2, float2, char4,
                   uchar4, short4, ushort4, int4, uint4, float4) {
  CHECK_IMAGE_SUPPORT

#if __HIP_NO_IMAGE_SUPPORT
  HipTest::HIP_SKIP_TEST("__HIP_NO_IMAGE_SUPPORT is set");
  return;
#endif

  constexpr int SIZE = 512;
  constexpr int num_layers = 5;
  constexpr unsigned int width = SIZE;
  constexpr unsigned int size = width * num_layers * sizeof(TestType);
  TestType* hData = reinterpret_cast<TestType*>(malloc(size));
  memset(hData, 0, size);
  for (unsigned int layer = 0; layer < num_layers; layer++) {
    for (unsigned int i = 0; i < width; i++) {
      initVal(hData[layer * width + i]);
    }
  }

  TestType* dData = nullptr;
  HIP_CHECK(hipMalloc(&dData, size));

  HIP_CHECK(hipMemcpy(dData, hData, size, hipMemcpyHostToDevice));

  hipChannelFormatDesc channelDesc;
  // Allocate array and copy image data
  channelDesc = hipCreateChannelDesc<TestType>();
  hipArray_t arr;
  HIP_CHECK(
      hipMalloc3DArray(&arr, &channelDesc, make_hipExtent(width, 0, num_layers), hipArrayLayered));
  hipMemcpy3DParms myparms{};

  SECTION("hipMemcpy3D whole layers") {
    myparms.srcPos = make_hipPos(0, 0, 0);
    myparms.dstPos = make_hipPos(0, 0, 0);
    myparms.srcPtr = make_hipPitchedPtr(dData, width * sizeof(TestType), width, 1);
    myparms.dstArray = arr;
    myparms.extent = make_hipExtent(width, 1, num_layers);
    myparms.kind = hipMemcpyDeviceToDevice;
    HIP_CHECK(hipMemcpy3D(&myparms));
  }

  SECTION("hipMemcpy3D layer by layer") {
    constexpr unsigned int layerSize = width * sizeof(TestType);
    TestType* hLayerData = reinterpret_cast<TestType*>(malloc(layerSize));
    REQUIRE(hLayerData != nullptr);
    TestType* dData1 = nullptr;
    HIP_CHECK(hipMalloc(&dData1, size));
    HIP_CHECK(hipMemset(dData1, 0, size));
    for (unsigned int layer = 0; layer < num_layers; layer++) {
      // Copy buffer layer to image layer
      memset(hLayerData, 0, layerSize);
      memset(&myparms, 0, sizeof(myparms));
      myparms.srcPos = make_hipPos(0, 0, layer);
      myparms.dstPos = make_hipPos(0, 0, layer);
      myparms.srcPtr = make_hipPitchedPtr(dData, width * sizeof(TestType), width, 1);
      myparms.kind = hipMemcpyDeviceToDevice;
      myparms.dstArray = arr;
      myparms.extent = make_hipExtent(width, 1, 1);
      HIP_CHECK(hipMemcpy3D(&myparms));

      // Copy image layer to buffer layer
      memset(&myparms, 0, sizeof(myparms));
      myparms.srcPos = make_hipPos(0, 0, layer);
      myparms.dstPos = make_hipPos(0, 0, layer);
      myparms.srcArray = arr;
      myparms.dstPtr = make_hipPitchedPtr(dData1, width * sizeof(TestType), width, 1);
      myparms.extent = make_hipExtent(width, 1, 1);
      myparms.kind = hipMemcpyDeviceToDevice;
      HIP_CHECK(hipMemcpy3D(&myparms));
      HIP_CHECK(hipMemcpy(hLayerData, dData1 + layer * width, layerSize, hipMemcpyDeviceToHost));
      // Compare layer
#ifdef DEBUG_DATA
      for (unsigned int i = 0; i < width; i++) {
        fprintf(stderr, "%4u: %u: %s -- %s\n", layer, i,
                getString(hData[layer * width + i]).c_str(), getString(hLayerData[i]).c_str());
      }
#endif
      REQUIRE(HipTest::checkArray(hData + layer * width, hLayerData, width, 1, 1));
    }
    free(hLayerData);
    HIP_CHECK(hipFree(dData1));
  }

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
  HIP_CHECK(hipMemset(dData, 0, size));

  dim3 dimBlock(8);
  dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x);
  for (unsigned int layer = 0; layer < num_layers; layer++) {
    hipLaunchKernelGGL(simpleKernelLayered1DArray<TestType>, dimGrid, dimBlock, 0, 0, tex, dData,
                       width, layer);
    HIP_CHECK(hipGetLastError());
  }
  HIP_CHECK(hipDeviceSynchronize());

  // Allocate mem for the result on host side
  TestType* hOutputData = reinterpret_cast<TestType*>(malloc(size));
  REQUIRE(hOutputData != nullptr);

  // Copy result from device to host
  HIP_CHECK(hipMemcpy(hOutputData, dData, size, hipMemcpyDeviceToHost));
  REQUIRE(HipTest::checkArray(hData, hOutputData, width, 1, num_layers));

  HIP_CHECK(hipFree(dData));
  HIP_CHECK(hipFreeArray(arr));
  free(hData);
  free(hOutputData);
  HIP_CHECK(hipDestroyTextureObject(tex));
}

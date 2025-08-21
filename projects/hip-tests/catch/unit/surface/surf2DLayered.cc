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

/**
 * @addtogroup surf2DLayered surf2DLayered
 * @{
 * @ingroup SurfaceTest
 */

#include <hip_array_common.hh>
#include <hip_test_common.hh>
#include <hip_texture_helper.hh>

#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-parameter"

#define LOG_DATA 0

template <typename T> __global__ void surf2DLayeredKernelR(hipSurfaceObject_t surfaceObject,
                                                           T* outputData, int width, int height) {
#if !defined(__HIP_NO_IMAGE_SUPPORT) || !__HIP_NO_IMAGE_SUPPORT
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    surf2DLayeredread<T>(outputData + y * width + x, surfaceObject, x * sizeof(T), y, 0);
  }
#endif
}

template <typename T> __global__ void surf2DLayeredKernelW(hipSurfaceObject_t surfaceObject,
                                                           T* inputData, int width, int height) {
#if !defined(__HIP_NO_IMAGE_SUPPORT) || !__HIP_NO_IMAGE_SUPPORT
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    surf2DLayeredwrite<T>(inputData[y * width + x], surfaceObject, x * sizeof(T), y, 0);
  }
#endif
}

template <typename T> __global__ void surf2DLayeredKernelRW(hipSurfaceObject_t surfaceObject,
                                                            hipSurfaceObject_t outputSurfObj,
                                                            int width, int height) {
#if !defined(__HIP_NO_IMAGE_SUPPORT) || !__HIP_NO_IMAGE_SUPPORT
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    T data;
    surf2DLayeredread<T>(&data, surfaceObject, x * sizeof(T), y, 0);
    surf2DLayeredwrite<T>(data, outputSurfObj, x * sizeof(T), y, 0);
  }
#endif
}

template <typename T> static void runTestR(const int width, const int height) {
  unsigned int size = width * height * sizeof(T);
  T* hData = (T*)malloc(size);
  memset(hData, 0, size);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      initVal(hData[i * width + j]);
    }
  }

  hipChannelFormatDesc channelDesc = hipCreateChannelDesc<T>();
  hipArray_t hipArray = nullptr;
  HIP_CHECK(hipMallocArray(&hipArray, &channelDesc, width, height, hipArraySurfaceLoadStore));

  // Need set source pitch, but we don't have any padding here
  const size_t spitch = width * sizeof(T);
  HIP_CHECK(
      hipMemcpy2DToArray(hipArray, 0, 0, hData, spitch, spitch, height, hipMemcpyHostToDevice));

  hipResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = hipResourceTypeArray;
  resDesc.res.array.array = hipArray;

  // Create surface object
  hipSurfaceObject_t surfaceObject = 0;
  HIP_CHECK(hipCreateSurfaceObject(&surfaceObject, &resDesc));

  T* hOutputData = nullptr;
  HIP_CHECK(hipHostMalloc((void**)&hOutputData, size));
  memset(hOutputData, 0, size);

  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y, 1);
  surf2DLayeredKernelR<T><<<dimGrid, dimBlock>>>(surfaceObject, hOutputData, width, height);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int index = i * width + j;
      if (!isEqual(hData[index], hOutputData[index])) {
        printf("Difference [ %d %d ]:%s ----%s\n", i, j, getString(hData[index]).c_str(),
               getString(hOutputData[index]).c_str());
        REQUIRE(false);
      }
    }
  }

  HIP_CHECK(hipDestroySurfaceObject(surfaceObject));
  HIP_CHECK(hipFreeArray(hipArray));
  free(hData);
  HIP_CHECK(hipHostFree(hOutputData));
}

template <typename T> static void runTestW(const int width, const int height) {
  unsigned int size = width * height * sizeof(T);
  T* hData = nullptr;
  HIP_CHECK(hipHostMalloc((void**)&hData, size));
  memset(hData, 0, size);

  hipChannelFormatDesc channelDesc = hipCreateChannelDesc<T>();
  hipArray_t hipArray = nullptr;
  HIP_CHECK(hipMallocArray(&hipArray, &channelDesc, width, height, hipArraySurfaceLoadStore));

  // Need set source pitch, but we don't have any padding here
  const size_t spitch = width * sizeof(T);
  HIP_CHECK(
      hipMemcpy2DToArray(hipArray, 0, 0, hData, spitch, spitch, height, hipMemcpyHostToDevice));

  hipResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = hipResourceTypeArray;
  resDesc.res.array.array = hipArray;

  // Create surface object
  hipSurfaceObject_t surfaceObject = 0;
  HIP_CHECK(hipCreateSurfaceObject(&surfaceObject, &resDesc));

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      initVal(hData[i * width + j]);
    }
  }

  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y, 1);
  surf2DLayeredKernelW<T><<<dimGrid, dimBlock>>>(surfaceObject, hData, width, height);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  T* hOutputData = (T*)malloc(size);

  memset(hOutputData, 0, size);
  HIP_CHECK(hipMemcpy2DFromArray(hOutputData, spitch, hipArray, 0, 0, spitch, height,
                                 hipMemcpyDeviceToHost));

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int index = i * width + j;
      if (!isEqual(hData[index], hOutputData[index])) {
        printf("Difference [ %d %d ]:%s ----%s\n", i, j, getString(hData[index]).c_str(),
               getString(hOutputData[index]).c_str());
        REQUIRE(false);
      }
    }
  }

  HIP_CHECK(hipDestroySurfaceObject(surfaceObject));
  HIP_CHECK(hipFreeArray(hipArray));
  HIP_CHECK(hipHostFree(hData));
  free(hOutputData);
}

template <typename T> static void runTestRW(const int width, const int height) {
  unsigned int size = width * height * sizeof(T);
  T* hData = (T*)malloc(size);
  memset(hData, 0, size);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      initVal(hData[i * width + j]);
    }
  }
#if LOG_DATA
  printf("hData: ");
  for (int i = 0; i < 32; i++) {
    printf("%s  ", getString(hData[i]).c_str());
  }
  printf("\n");
#endif

  hipChannelFormatDesc channelDesc = hipCreateChannelDesc<T>();
  hipArray_t hipArray = nullptr, hipOutArray = nullptr;
  HIP_CHECK(hipMallocArray(&hipArray, &channelDesc, width, height, hipArraySurfaceLoadStore));

  // Need set source pitch, but we don't have any padding here
  const size_t spitch = width * sizeof(T);
  HIP_CHECK(
      hipMemcpy2DToArray(hipArray, 0, 0, hData, spitch, spitch, height, hipMemcpyHostToDevice));

  hipResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = hipResourceTypeArray;
  resDesc.res.array.array = hipArray;

  // Create surface object
  hipSurfaceObject_t surfaceObject = 0;
  HIP_CHECK(hipCreateSurfaceObject(&surfaceObject, &resDesc));

  HIP_CHECK(hipMallocArray(&hipOutArray, &channelDesc, width, height, hipArraySurfaceLoadStore));

  hipResourceDesc resOutDesc;
  memset(&resOutDesc, 0, sizeof(resOutDesc));
  resOutDesc.resType = hipResourceTypeArray;
  resOutDesc.res.array.array = hipOutArray;

  hipSurfaceObject_t outSurfaceObject = 0;
  HIP_CHECK(hipCreateSurfaceObject(&outSurfaceObject, &resOutDesc));

  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y, 1);
  surf2DLayeredKernelRW<T><<<dimGrid, dimBlock>>>(surfaceObject, outSurfaceObject, width, height);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  T* hOutputData = (T*)malloc(size);

  memset(hOutputData, 0, size);
  HIP_CHECK(hipMemcpy2DFromArray(hOutputData, spitch, hipOutArray, 0, 0, spitch, height,
                                 hipMemcpyDeviceToHost));

#if LOG_DATA
  printf("dData: ");
  for (int i = 0; i < 32; i++) {
    printf("%s  ", getString(hOutputData[i]).c_str());
  }
  printf("\n");
#endif

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int index = i * width + j;
      if (!isEqual(hData[index], hOutputData[index])) {
        printf("Difference [ %d %d ]:%s ----%s\n", i, j, getString(hData[index]).c_str(),
               getString(hOutputData[index]).c_str());
        REQUIRE(false);
      }
    }
  }

  HIP_CHECK(hipDestroySurfaceObject(surfaceObject));
  HIP_CHECK(hipDestroySurfaceObject(outSurfaceObject));
  HIP_CHECK(hipFreeArray(hipArray));
  HIP_CHECK(hipFreeArray(hipOutArray));
  free(hData);
  free(hOutputData);
}

/**
 * Test Description
 * ------------------------
 *    - Basic test for `surf2DLayeredread` with different types and dimensions.
 * Test source
 * ------------------------
 *    - unit/surface/surf2DLayered.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.7
 */
TEMPLATE_TEST_CASE("Unit_surf2DLayeredread_Positive_Basic", "", char, uchar, short, ushort, int,
                   uint, float, char1, uchar1, short1, ushort1, int1, uint1, float1, char2, uchar2,
                   short2, ushort2, int2, uint2, float2, char4, uchar4, short4, ushort4, int4,
                   uint4, float4) {
  CHECK_IMAGE_SUPPORT;

  const int width = GENERATE(31, 67);
  const int height = GENERATE(131, 263);
  runTestR<TestType>(width, height);
}

/**
 * Test Description
 * ------------------------
 *    - Basic test for `surf2DLayeredwrite` with different types and dimensions.
 * Test source
 * ------------------------
 *    - unit/surface/surf2DLayered.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.7
 */
TEMPLATE_TEST_CASE("Unit_surf2DLayeredwrite_Positive_Basic", "", char, uchar, short, ushort, int,
                   uint, float, char1, uchar1, short1, ushort1, int1, uint1, float1, char2, uchar2,
                   short2, ushort2, int2, uint2, float2, char4, uchar4, short4, ushort4, int4,
                   uint4, float4) {
  CHECK_IMAGE_SUPPORT;

  const int width = GENERATE(31, 67);
  const int height = GENERATE(131, 263);
  runTestW<TestType>(width, height);
}

/**
 * Test Description
 * ------------------------
 *    - Basic test for `surf2DLayeredread` and `surf2DLayeredwrite` together, with different types
 * and dimensions. Test source
 * ------------------------
 *    - unit/surface/surf2DLayered.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.7
 */
TEMPLATE_TEST_CASE("Unit_surf2DLayered_Positive_ReadWrite", "", char, uchar, short, ushort, int,
                   uint, float, char1, uchar1, short1, ushort1, int1, uint1, float1, char2, uchar2,
                   short2, ushort2, int2, uint2, float2, char4, uchar4, short4, ushort4, int4,
                   uint4, float4) {
  CHECK_IMAGE_SUPPORT;

  const int width = GENERATE(31, 67);
  const int height = GENERATE(131, 263);
  runTestRW<TestType>(width, height);
}

/**
 * End doxygen group SurfaceTest.
 * @}
 */

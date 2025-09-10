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
 * @addtogroup surf3D surf3D
 * @{
 * @ingroup SurfaceTest
 */

#include <hip_array_common.hh>
#include <hip_test_common.hh>
#include <hip_texture_helper.hh>

#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-parameter"

template <typename T> __global__ void surf3DKernelR(hipSurfaceObject_t surfaceObject, T* outputData,
                                                    int width, int height, int depth) {
#if !__HIP_NO_IMAGE_SUPPORT
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < width && y < height && z < depth) {
    surf3Dread(outputData + z * width * height + y * width + x, surfaceObject, x * sizeof(T), y, z);
  }
#endif
}

template <typename T> __global__ void surf3DKernelW(hipSurfaceObject_t surfaceObject, T* inputData,
                                                    int width, int height, int depth) {
#if !__HIP_NO_IMAGE_SUPPORT
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < width && y < height && z < depth) {
    surf3Dwrite(inputData[z * width * height + y * width + x], surfaceObject, x * sizeof(T), y, z);
  }
#endif
}

template <typename T> __global__ void surf3DKernelRW(hipSurfaceObject_t surfaceObject,
                                                     hipSurfaceObject_t outputSurfObj, int width,
                                                     int height, int depth) {
#if !__HIP_NO_IMAGE_SUPPORT
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < width && y < height && z < depth) {
    T data;
    surf3Dread(&data, surfaceObject, x * sizeof(T), y, z);
    surf3Dwrite(data, outputSurfObj, x * sizeof(T), y, z);
  }
#endif
}

template <typename T> static void runTestR(const int width, const int height, const int depth) {
  unsigned int size = width * height * depth * sizeof(T);
  T* hData = (T*)malloc(size);
  memset(hData, 0, size);
  for (int i = 0; i < depth; i++) {
    for (int j = 0; j < height; j++) {
      for (int k = 0; k < width; k++) {
        initVal(hData[i * width * height + j * width + k]);
      }
    }
  }

  // Allocate array and copy image data
  hipChannelFormatDesc channelDesc = hipCreateChannelDesc<T>();
  hipArray_t hipArray = nullptr;
  HIP_CHECK(hipMalloc3DArray(&hipArray, &channelDesc, make_hipExtent(width, height, depth),
                             hipArraySurfaceLoadStore));

  hipMemcpy3DParms myparms;
  memset(&myparms, 0, sizeof(myparms));
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.srcPtr = make_hipPitchedPtr(hData, width * sizeof(T), width, height);
  myparms.dstArray = hipArray;
  myparms.extent = make_hipExtent(width, height, depth);
  myparms.kind = hipMemcpyHostToDevice;

  HIP_CHECK(hipMemcpy3D(&myparms));

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

  dim3 dimBlock(8, 8, 8);  // 512 threads
  dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y,
               (depth + dimBlock.z - 1) / dimBlock.z);

  surf3DKernelR<T><<<dimGrid, dimBlock>>>(surfaceObject, hOutputData, width, height, depth);

  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  for (int i = 0; i < depth; i++) {
    for (int j = 0; j < height; j++) {
      for (int k = 0; k < width; k++) {
        int index = i * width * height + j * width + k;
        if (!isEqual(hData[index], hOutputData[index])) {
          printf("Difference [ %d %d %d]:%s ----%s\n", i, j, k, getString(hData[index]).c_str(),
                 getString(hOutputData[index]).c_str());
          REQUIRE(false);
        }
      }
    }
  }

  HIP_CHECK(hipDestroySurfaceObject(surfaceObject));
  HIP_CHECK(hipFreeArray(hipArray));
  free(hData);
  HIP_CHECK(hipHostFree(hOutputData));
}

template <typename T> static void runTestW(const int width, const int height, const int depth) {
  unsigned int size = width * height * depth * sizeof(T);
  T* hData = nullptr;
  HIP_CHECK(hipHostMalloc((void**)&hData, size));
  memset(hData, 0, size);

  // Allocate array and copy image data
  hipChannelFormatDesc channelDesc = hipCreateChannelDesc<T>();
  hipArray_t hipArray = nullptr;
  HIP_CHECK(hipMalloc3DArray(&hipArray, &channelDesc, make_hipExtent(width, height, depth),
                             hipArraySurfaceLoadStore));

  hipMemcpy3DParms myparms;
  memset(&myparms, 0, sizeof(myparms));
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.srcPtr = make_hipPitchedPtr(hData, width * sizeof(T), width, height);
  myparms.dstArray = hipArray;
  myparms.extent = make_hipExtent(width, height, depth);
  myparms.kind = hipMemcpyHostToDevice;

  HIP_CHECK(hipMemcpy3D(&myparms));

  hipResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = hipResourceTypeArray;
  resDesc.res.array.array = hipArray;

  // Create surface object
  hipSurfaceObject_t surfaceObject = 0;
  HIP_CHECK(hipCreateSurfaceObject(&surfaceObject, &resDesc));

  for (int i = 0; i < depth; i++) {
    for (int j = 0; j < height; j++) {
      for (int k = 0; k < width; k++) {
        initVal(hData[i * width * height + j * width + k]);
      }
    }
  }

  dim3 dimBlock(8, 8, 8);  // 512 threads
  dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y,
               (depth + dimBlock.z - 1) / dimBlock.z);

  surf3DKernelW<T><<<dimGrid, dimBlock>>>(surfaceObject, hData, width, height, depth);

  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  T* hOutputData = (T*)malloc(size);
  memset(hOutputData, 0, size);

  memset(&myparms, 0, sizeof(myparms));
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.srcArray = hipArray;
  myparms.dstPtr = make_hipPitchedPtr(hOutputData, width * sizeof(T), width, height);
  myparms.extent = make_hipExtent(width, height, depth);
  myparms.kind = hipMemcpyDeviceToHost;

  HIP_CHECK(hipMemcpy3D(&myparms));

  for (int i = 0; i < depth; i++) {
    for (int j = 0; j < height; j++) {
      for (int k = 0; k < width; k++) {
        int index = i * width * height + j * width + k;
        if (!isEqual(hData[index], hOutputData[index])) {
          printf("Difference [ %d %d %d]:%s ----%s\n", i, j, k, getString(hData[index]).c_str(),
                 getString(hOutputData[index]).c_str());
          REQUIRE(false);
        }
      }
    }
  }

  HIP_CHECK(hipDestroySurfaceObject(surfaceObject));
  HIP_CHECK(hipFreeArray(hipArray));
  HIP_CHECK(hipHostFree(hData));
  free(hOutputData);
}

template <typename T> static void runTestRW(const int width, const int height, const int depth) {
  unsigned int size = width * height * depth * sizeof(T);
  T* hData = (T*)malloc(size);
  memset(hData, 0, size);
  for (int i = 0; i < depth; i++) {
    for (int j = 0; j < height; j++) {
      for (int k = 0; k < width; k++) {
        initVal(hData[i * width * height + j * width + k]);
      }
    }
  }

  // Allocate array and copy image data
  hipChannelFormatDesc channelDesc = hipCreateChannelDesc<T>();
  hipArray_t hipArray = nullptr, hipOutArray = nullptr;
  HIP_CHECK(hipMalloc3DArray(&hipArray, &channelDesc, make_hipExtent(width, height, depth),
                             hipArraySurfaceLoadStore));

  hipMemcpy3DParms myparms;
  memset(&myparms, 0, sizeof(myparms));
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.srcPtr = make_hipPitchedPtr(hData, width * sizeof(T), width, height);
  myparms.dstArray = hipArray;
  myparms.extent = make_hipExtent(width, height, depth);
  myparms.kind = hipMemcpyHostToDevice;

  HIP_CHECK(hipMemcpy3D(&myparms));

  hipResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = hipResourceTypeArray;
  resDesc.res.array.array = hipArray;

  // Create surface object
  hipSurfaceObject_t surfaceObject = 0;
  HIP_CHECK(hipCreateSurfaceObject(&surfaceObject, &resDesc));

  HIP_CHECK(hipMalloc3DArray(&hipOutArray, &channelDesc, make_hipExtent(width, height, depth),
                             hipArraySurfaceLoadStore));

  hipResourceDesc resOutDesc;
  memset(&resOutDesc, 0, sizeof(resOutDesc));
  resOutDesc.resType = hipResourceTypeArray;
  resOutDesc.res.array.array = hipOutArray;

  hipSurfaceObject_t outSurfaceObject = 0;
  HIP_CHECK(hipCreateSurfaceObject(&outSurfaceObject, &resOutDesc));

  dim3 dimBlock(8, 8, 8);  // 512 threads
  dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y,
               (depth + dimBlock.z - 1) / dimBlock.z);

  surf3DKernelRW<T><<<dimGrid, dimBlock>>>(surfaceObject, outSurfaceObject, width, height, depth);

  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  T* hOutputData = (T*)malloc(size);
  memset(hOutputData, 0, size);

  memset(&myparms, 0, sizeof(myparms));
  myparms.srcPos = make_hipPos(0, 0, 0);
  myparms.dstPos = make_hipPos(0, 0, 0);
  myparms.srcArray = hipOutArray;
  myparms.dstPtr = make_hipPitchedPtr(hOutputData, width * sizeof(T), width, height);
  myparms.extent = make_hipExtent(width, height, depth);
  myparms.kind = hipMemcpyDeviceToHost;

  HIP_CHECK(hipMemcpy3D(&myparms));

  for (int i = 0; i < depth; i++) {
    for (int j = 0; j < height; j++) {
      for (int k = 0; k < width; k++) {
        int index = i * width * height + j * width + k;
        if (!isEqual(hData[index], hOutputData[index])) {
          printf("Difference [ %d %d %d]:%s ----%s\n", i, j, k, getString(hData[index]).c_str(),
                 getString(hOutputData[index]).c_str());
          REQUIRE(false);
        }
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
 *    - Basic test for `surf3Dread` with different types and dimensions.
 * Test source
 * ------------------------
 *    - unit/surface/surf3D.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.7
 */
TEMPLATE_TEST_CASE("Unit_surf3Dread_Positive_Basic", "", char, uchar, short, ushort, int, uint,
                   float, char1, uchar1, short1, ushort1, int1, uint1, float1, char2, uchar2,
                   short2, ushort2, int2, uint2, float2, char4, uchar4, short4, ushort4, int4,
                   uint4, float4) {
  CHECK_IMAGE_SUPPORT;

#if __HIP_NO_IMAGE_SUPPORT
  HipTest::HIP_SKIP_TEST("__HIP_NO_IMAGE_SUPPORT is set");
  return;
#endif

  const int width = GENERATE(31, 67);
  const int height = GENERATE(131, 263);
  const int depth = GENERATE(4, 11);
  runTestR<TestType>(width, height, depth);
}

/**
 * Test Description
 * ------------------------
 *    - Basic test for `surf3Dwrite` with different types and dimensions.
 * Test source
 * ------------------------
 *    - unit/surface/surf3D.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.7
 */
TEMPLATE_TEST_CASE("Unit_surf3Dwrite_Positive_Basic", "", char, uchar, short, ushort, int, uint,
                   float, char1, uchar1, short1, ushort1, int1, uint1, float1, char2, uchar2,
                   short2, ushort2, int2, uint2, float2, char4, uchar4, short4, ushort4, int4,
                   uint4, float4) {
  CHECK_IMAGE_SUPPORT;

#if __HIP_NO_IMAGE_SUPPORT
  HipTest::HIP_SKIP_TEST("__HIP_NO_IMAGE_SUPPORT is set");
  return;
#endif

  const int width = GENERATE(31, 67);
  const int height = GENERATE(131, 263);
  const int depth = GENERATE(4, 11);
  runTestR<TestType>(width, height, depth);
}

/**
 * Test Description
 * ------------------------
 *    - Basic test for `surf3Dread` and `surf3Dwrite` together, with different types and dimensions.
 * Test source
 * ------------------------
 *    - unit/surface/surf3D.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.7
 */
TEMPLATE_TEST_CASE("Unit_surf3D_Positive_ReadWrite", "", char, uchar, short, ushort, int, uint,
                   float, char1, uchar1, short1, ushort1, int1, uint1, float1, char2, uchar2,
                   short2, ushort2, int2, uint2, float2, char4, uchar4, short4, ushort4, int4,
                   uint4, float4) {
  CHECK_IMAGE_SUPPORT;

#if __HIP_NO_IMAGE_SUPPORT
  HipTest::HIP_SKIP_TEST("__HIP_NO_IMAGE_SUPPORT is set");
  return;
#endif

  const int width = GENERATE(31, 67);
  const int height = GENERATE(131, 263);
  const int depth = GENERATE(4, 11);
  runTestR<TestType>(width, height, depth);
}

/**
 * End doxygen group SurfaceTest.
 * @}
 */

/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma clang diagnostic ignored "-Wunused-parameter"
#include <hip_test_common.hh>
#include <hip_test_checkers.hh>

/**
 * @addtogroup hipCreateTextureObject hipCreateTextureObject
 * @{
 * @ingroup TextureTest
 * `hipCreateTextureObject(hipTextureObject_t* pTexObject, const hipResourceDesc* pResDesc,
 * const hipTextureDesc* pTexDesc, const struct hipResourceViewDesc* pResViewDesc)` -
 * Creates a texture object.
 */

#define SIZE_H 20
#define SIZE_W 179

// texture object is a kernel argument
template <typename TYPE_t>
static __global__ void texture2dCopyKernel(hipTextureObject_t texObj, TYPE_t* dst) {
#if !__HIP_NO_IMAGE_SUPPORT
  for (int i = 0; i < SIZE_H; i++)
    for (int j = 0; j < SIZE_W; j++) dst[SIZE_W * i + j] = tex2D<TYPE_t>(texObj, j, i);
  __syncthreads();
#endif
}

/**
 * Test Description
 * ------------------------
 *  - Basic test where resource type is 2D pitch.
 * Test source
 * ------------------------
 *  - unit/texture/hipTexObjPitch.cc
 * Test requirements
 * ------------------------
 *  - Textures supported on device
 *  - HIP_VERSION >= 5.2
 */
HIP_TEMPLATE_TEST_CASE(Unit_hipTexObjPitch_texture2D, char, unsigned char, short, unsigned short,
                   int, unsigned int, float) {
  CHECK_IMAGE_SUPPORT
  (void)hipGetLastError();  // Prevent negative tests affecting this

  TestType* B;
  TestType* A;
  TestType* devPtrB;
  TestType* devPtrA;

  B = new TestType[SIZE_H * SIZE_W];
  A = new TestType[SIZE_H * SIZE_W];
  for (size_t i = 1; i <= (SIZE_H * SIZE_W); i++) {
    A[i - 1] = i;
  }

  size_t devPitchA;
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&devPtrA), &devPitchA,
                           SIZE_W * sizeof(TestType), SIZE_H));
  HIP_CHECK(hipMemcpy2D(devPtrA, devPitchA, A, SIZE_W * sizeof(TestType), SIZE_W * sizeof(TestType),
                        SIZE_H, hipMemcpyHostToDevice));

  // Use the texture object
  hipResourceDesc texRes;
  memset(&texRes, 0, sizeof(texRes));
  texRes.resType = hipResourceTypePitch2D;
  texRes.res.pitch2D.devPtr = devPtrA;
  texRes.res.pitch2D.height = SIZE_H;
  texRes.res.pitch2D.width = SIZE_W;
  texRes.res.pitch2D.pitchInBytes = devPitchA;
  texRes.res.pitch2D.desc = hipCreateChannelDesc<TestType>();

  hipTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(texDescr));
  texDescr.normalizedCoords = false;
  texDescr.filterMode = hipFilterModePoint;
  texDescr.mipmapFilterMode = hipFilterModePoint;
  texDescr.addressMode[0] = hipAddressModeClamp;
  texDescr.addressMode[1] = hipAddressModeClamp;
  texDescr.addressMode[2] = hipAddressModeClamp;
  texDescr.readMode = hipReadModeElementType;

  hipTextureObject_t texObj;
  HIP_CHECK(hipCreateTextureObject(&texObj, &texRes, &texDescr, NULL));

  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&devPtrB), SIZE_W * sizeof(TestType) * SIZE_H));

  hipLaunchKernelGGL(texture2dCopyKernel, dim3(1, 1, 1), dim3(1, 1, 1), 0, 0, texObj, devPtrB);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipMemcpy2D(B, SIZE_W * sizeof(TestType), devPtrB, SIZE_W * sizeof(TestType),
                        SIZE_W * sizeof(TestType), SIZE_H, hipMemcpyDeviceToHost));

  HipTest::checkArray(A, B, SIZE_H, SIZE_W);
  delete[] A;
  delete[] B;
  HIP_CHECK(hipFree(devPtrA));
  HIP_CHECK(hipFree(devPtrB));
  HIP_CHECK(hipDestroyTextureObject(texObj));
}

/**
 * End doxygen group TextureTest.
 * @}
 */

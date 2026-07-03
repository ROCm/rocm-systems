/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

namespace {
constexpr size_t kLinearElements = 64;
constexpr size_t kLinearBytes = kLinearElements * sizeof(uint8_t);
constexpr size_t kArrayWidth = 8;
constexpr size_t kArrayHeight = 4;

hipChannelFormatDesc ByteChannelDesc() { return hipCreateChannelDesc<uint8_t>(); }

// Builds a zero-initialized resource descriptor backed by a small linear
// device buffer of unsigned char elements.
hipResourceDesc MakeLinearResourceDesc(void* dev_ptr) {
  hipResourceDesc res{};
  res.resType = hipResourceTypeLinear;
  res.res.linear.devPtr = dev_ptr;
  res.res.linear.desc = ByteChannelDesc();
  res.res.linear.sizeInBytes = kLinearBytes;
  return res;
}

// Builds a zero-initialized resource descriptor backed by a small HIP array.
hipResourceDesc MakeArrayResourceDesc(hipArray_t array) {
  hipResourceDesc res{};
  res.resType = hipResourceTypeArray;
  res.res.array.array = array;
  return res;
}

// Default texture descriptor used by the texture object contracts: element-type
// read mode with unnormalized coordinates.
hipTextureDesc MakeTextureDesc() {
  hipTextureDesc tex{};
  tex.readMode = hipReadModeElementType;
  tex.normalizedCoords = 0;
  return tex;
}
}  // namespace

HIP_TEST_CASE(Contract_Texture_CreateAndDestroy_LinearResource_Succeeds) {
  CHECK_IMAGE_SUPPORT;

  void* dev_ptr = nullptr;
  HIP_CHECK(hipMalloc(&dev_ptr, kLinearBytes));

  const hipResourceDesc res = MakeLinearResourceDesc(dev_ptr);
  const hipTextureDesc tex = MakeTextureDesc();

  hipTextureObject_t tex_obj = 0;
  const hipError_t status = hipCreateTextureObject(&tex_obj, &res, &tex, nullptr);
  if (status == hipErrorNotSupported) {
    HIP_CHECK(hipFree(dev_ptr));
    HIP_SKIP_TEST(HipTest::SkipReason::kTextureImageUnsupported);
  }
  HIP_CHECK(status);

  REQUIRE(tex_obj != 0);

  HIP_CHECK(hipDestroyTextureObject(tex_obj));
  HIP_CHECK(hipFree(dev_ptr));
}

HIP_TEST_CASE(Contract_Texture_GetResourceDesc_RoundTripsLinearResource) {
  CHECK_IMAGE_SUPPORT;

  void* dev_ptr = nullptr;
  HIP_CHECK(hipMalloc(&dev_ptr, kLinearBytes));

  const hipResourceDesc res = MakeLinearResourceDesc(dev_ptr);
  const hipTextureDesc tex = MakeTextureDesc();

  hipTextureObject_t tex_obj = 0;
  const hipError_t status = hipCreateTextureObject(&tex_obj, &res, &tex, nullptr);
  if (status == hipErrorNotSupported) {
    HIP_CHECK(hipFree(dev_ptr));
    HIP_SKIP_TEST(HipTest::SkipReason::kTextureImageUnsupported);
  }
  HIP_CHECK(status);

  hipResourceDesc returned{};
  HIP_CHECK(hipGetTextureObjectResourceDesc(&returned, tex_obj));

  REQUIRE(returned.resType == hipResourceTypeLinear);
  REQUIRE(returned.res.linear.devPtr == dev_ptr);
  REQUIRE(returned.res.linear.sizeInBytes == kLinearBytes);

  HIP_CHECK(hipDestroyTextureObject(tex_obj));
  HIP_CHECK(hipFree(dev_ptr));
}

HIP_TEST_CASE(Contract_Texture_GetTextureDesc_RoundTripsReadMode) {
  CHECK_IMAGE_SUPPORT;

  void* dev_ptr = nullptr;
  HIP_CHECK(hipMalloc(&dev_ptr, kLinearBytes));

  const hipResourceDesc res = MakeLinearResourceDesc(dev_ptr);
  const hipTextureDesc tex = MakeTextureDesc();

  hipTextureObject_t tex_obj = 0;
  const hipError_t status = hipCreateTextureObject(&tex_obj, &res, &tex, nullptr);
  if (status == hipErrorNotSupported) {
    HIP_CHECK(hipFree(dev_ptr));
    HIP_SKIP_TEST(HipTest::SkipReason::kTextureImageUnsupported);
  }
  HIP_CHECK(status);

  hipTextureDesc returned{};
  HIP_CHECK(hipGetTextureObjectTextureDesc(&returned, tex_obj));

  REQUIRE(returned.readMode == hipReadModeElementType);
  REQUIRE(returned.normalizedCoords == 0);

  HIP_CHECK(hipDestroyTextureObject(tex_obj));
  HIP_CHECK(hipFree(dev_ptr));
}

HIP_TEST_CASE(Contract_Texture_CreateAndDestroy_ArrayResource_Succeeds) {
  CHECK_IMAGE_SUPPORT;

  hipArray_t array = nullptr;
  const hipChannelFormatDesc channel = ByteChannelDesc();
  HIP_CHECK(hipMallocArray(&array, &channel, kArrayWidth, kArrayHeight));

  const hipResourceDesc res = MakeArrayResourceDesc(array);
  const hipTextureDesc tex = MakeTextureDesc();

  hipTextureObject_t tex_obj = 0;
  const hipError_t status = hipCreateTextureObject(&tex_obj, &res, &tex, nullptr);
  if (status == hipErrorNotSupported) {
    HIP_CHECK(hipFreeArray(array));
    HIP_SKIP_TEST(HipTest::SkipReason::kTextureImageUnsupported);
  }
  HIP_CHECK(status);

  REQUIRE(tex_obj != 0);

  HIP_CHECK(hipDestroyTextureObject(tex_obj));
  HIP_CHECK(hipFreeArray(array));
}

HIP_TEST_CASE(Contract_Texture_GetChannelDesc_MatchesArrayFormat) {
  CHECK_IMAGE_SUPPORT;

  hipArray_t array = nullptr;
  const hipChannelFormatDesc channel = ByteChannelDesc();
  HIP_CHECK(hipMallocArray(&array, &channel, kArrayWidth, kArrayHeight));

  hipChannelFormatDesc returned{};
  HIP_CHECK(hipGetChannelDesc(&returned, array));

  REQUIRE(returned.x == channel.x);
  REQUIRE(returned.y == channel.y);
  REQUIRE(returned.z == channel.z);
  REQUIRE(returned.w == channel.w);
  REQUIRE(returned.f == channel.f);

  HIP_CHECK(hipFreeArray(array));
}

HIP_TEST_CASE(Contract_Surface_CreateAndDestroy_ArrayResource_Succeeds) {
  CHECK_IMAGE_SUPPORT;

  hipArray_t array = nullptr;
  const hipChannelFormatDesc channel = ByteChannelDesc();
  HIP_CHECK(hipMallocArray(&array, &channel, kArrayWidth, kArrayHeight,
                           hipArraySurfaceLoadStore));

  const hipResourceDesc res = MakeArrayResourceDesc(array);

  hipSurfaceObject_t surf_obj = 0;
  const hipError_t status = hipCreateSurfaceObject(&surf_obj, &res);
  if (status == hipErrorNotSupported) {
    HIP_CHECK(hipFreeArray(array));
    HIP_SKIP_TEST(HipTest::SkipReason::kTextureImageUnsupported);
  }
  HIP_CHECK(status);

  REQUIRE(surf_obj != 0);

  HIP_CHECK(hipDestroySurfaceObject(surf_obj));
  HIP_CHECK(hipFreeArray(array));
}

HIP_TEST_CASE(Contract_Surface_GetChannelDesc_RoundTripsArrayResource) {
  CHECK_IMAGE_SUPPORT;

  hipArray_t array = nullptr;
  const hipChannelFormatDesc channel = ByteChannelDesc();
  HIP_CHECK(hipMallocArray(&array, &channel, kArrayWidth, kArrayHeight,
                           hipArraySurfaceLoadStore));

  const hipResourceDesc res = MakeArrayResourceDesc(array);

  hipSurfaceObject_t surf_obj = 0;
  const hipError_t status = hipCreateSurfaceObject(&surf_obj, &res);
  if (status == hipErrorNotSupported) {
    HIP_CHECK(hipFreeArray(array));
    HIP_SKIP_TEST(HipTest::SkipReason::kTextureImageUnsupported);
  }
  HIP_CHECK(status);

  hipChannelFormatDesc returned{};
  HIP_CHECK(hipGetChannelDesc(&returned, array));

  REQUIRE(returned.x == channel.x);
  REQUIRE(returned.y == channel.y);
  REQUIRE(returned.z == channel.z);
  REQUIRE(returned.w == channel.w);
  REQUIRE(returned.f == channel.f);

  HIP_CHECK(hipDestroySurfaceObject(surf_obj));
  HIP_CHECK(hipFreeArray(array));
}

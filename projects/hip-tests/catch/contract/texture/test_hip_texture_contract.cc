/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <contract_cleanup.hh>

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
  hip::contract::ContractCleanup cleanup;

  void* dev_ptr = nullptr;
  HIP_CHECK(hipMalloc(&dev_ptr, kLinearBytes));
  cleanup.Add([&] { (void)hipFree(dev_ptr); });

  const hipResourceDesc res = MakeLinearResourceDesc(dev_ptr);
  const hipTextureDesc tex = MakeTextureDesc();

  hipTextureObject_t tex_obj = 0;
  const hipError_t status = hipCreateTextureObject(&tex_obj, &res, &tex, nullptr);
  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST(HipTest::SkipReason::kTextureImageUnsupported);
  }
  HIP_CHECK(status);
  cleanup.Add([&] { (void)hipDestroyTextureObject(tex_obj); });

  REQUIRE(tex_obj != 0);
}

HIP_TEST_CASE(Contract_Texture_GetResourceDesc_RoundTripsLinearResource) {
  CHECK_IMAGE_SUPPORT;
  hip::contract::ContractCleanup cleanup;

  void* dev_ptr = nullptr;
  HIP_CHECK(hipMalloc(&dev_ptr, kLinearBytes));
  cleanup.Add([&] { (void)hipFree(dev_ptr); });

  const hipResourceDesc res = MakeLinearResourceDesc(dev_ptr);
  const hipTextureDesc tex = MakeTextureDesc();

  hipTextureObject_t tex_obj = 0;
  const hipError_t status = hipCreateTextureObject(&tex_obj, &res, &tex, nullptr);
  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST(HipTest::SkipReason::kTextureImageUnsupported);
  }
  HIP_CHECK(status);
  cleanup.Add([&] { (void)hipDestroyTextureObject(tex_obj); });

  hipResourceDesc returned{};
  HIP_CHECK(hipGetTextureObjectResourceDesc(&returned, tex_obj));

  REQUIRE(returned.resType == hipResourceTypeLinear);
  REQUIRE(returned.res.linear.devPtr == dev_ptr);
  REQUIRE(returned.res.linear.sizeInBytes == kLinearBytes);
}

HIP_TEST_CASE(Contract_Texture_GetTextureDesc_RoundTripsReadMode) {
  CHECK_IMAGE_SUPPORT;
  hip::contract::ContractCleanup cleanup;

  void* dev_ptr = nullptr;
  HIP_CHECK(hipMalloc(&dev_ptr, kLinearBytes));
  cleanup.Add([&] { (void)hipFree(dev_ptr); });

  const hipResourceDesc res = MakeLinearResourceDesc(dev_ptr);
  const hipTextureDesc tex = MakeTextureDesc();

  hipTextureObject_t tex_obj = 0;
  const hipError_t status = hipCreateTextureObject(&tex_obj, &res, &tex, nullptr);
  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST(HipTest::SkipReason::kTextureImageUnsupported);
  }
  HIP_CHECK(status);
  cleanup.Add([&] { (void)hipDestroyTextureObject(tex_obj); });

  hipTextureDesc returned{};
  HIP_CHECK(hipGetTextureObjectTextureDesc(&returned, tex_obj));

  REQUIRE(returned.readMode == hipReadModeElementType);
  REQUIRE(returned.normalizedCoords == 0);
}

HIP_TEST_CASE(Contract_Texture_CreateAndDestroy_ArrayResource_Succeeds) {
  CHECK_IMAGE_SUPPORT;
  hip::contract::ContractCleanup cleanup;

  hipArray_t array = nullptr;
  const hipChannelFormatDesc channel = ByteChannelDesc();
  HIP_CHECK(hipMallocArray(&array, &channel, kArrayWidth, kArrayHeight));
  cleanup.Add([&] { (void)hipFreeArray(array); });

  const hipResourceDesc res = MakeArrayResourceDesc(array);
  const hipTextureDesc tex = MakeTextureDesc();

  hipTextureObject_t tex_obj = 0;
  const hipError_t status = hipCreateTextureObject(&tex_obj, &res, &tex, nullptr);
  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST(HipTest::SkipReason::kTextureImageUnsupported);
  }
  HIP_CHECK(status);
  cleanup.Add([&] { (void)hipDestroyTextureObject(tex_obj); });

  REQUIRE(tex_obj != 0);
}

HIP_TEST_CASE(Contract_Texture_GetChannelDesc_MatchesArrayFormat) {
  CHECK_IMAGE_SUPPORT;
  hip::contract::ContractCleanup cleanup;

  hipArray_t array = nullptr;
  const hipChannelFormatDesc channel = ByteChannelDesc();
  HIP_CHECK(hipMallocArray(&array, &channel, kArrayWidth, kArrayHeight));
  cleanup.Add([&] { (void)hipFreeArray(array); });

  hipChannelFormatDesc returned{};
  HIP_CHECK(hipGetChannelDesc(&returned, array));

  REQUIRE(returned.x == channel.x);
  REQUIRE(returned.y == channel.y);
  REQUIRE(returned.z == channel.z);
  REQUIRE(returned.w == channel.w);
  REQUIRE(returned.f == channel.f);
}

HIP_TEST_CASE(Contract_Surface_CreateAndDestroy_ArrayResource_Succeeds) {
  CHECK_IMAGE_SUPPORT;
  hip::contract::ContractCleanup cleanup;

  hipArray_t array = nullptr;
  const hipChannelFormatDesc channel = ByteChannelDesc();
  HIP_CHECK(hipMallocArray(&array, &channel, kArrayWidth, kArrayHeight,
                           hipArraySurfaceLoadStore));
  cleanup.Add([&] { (void)hipFreeArray(array); });

  const hipResourceDesc res = MakeArrayResourceDesc(array);

  hipSurfaceObject_t surf_obj = 0;
  const hipError_t status = hipCreateSurfaceObject(&surf_obj, &res);
  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST(HipTest::SkipReason::kTextureImageUnsupported);
  }
  HIP_CHECK(status);
  cleanup.Add([&] { (void)hipDestroySurfaceObject(surf_obj); });

  REQUIRE(surf_obj != 0);
}

HIP_TEST_CASE(Contract_Surface_GetChannelDesc_RoundTripsArrayResource) {
  CHECK_IMAGE_SUPPORT;
  hip::contract::ContractCleanup cleanup;

  hipArray_t array = nullptr;
  const hipChannelFormatDesc channel = ByteChannelDesc();
  HIP_CHECK(hipMallocArray(&array, &channel, kArrayWidth, kArrayHeight,
                           hipArraySurfaceLoadStore));
  cleanup.Add([&] { (void)hipFreeArray(array); });

  const hipResourceDesc res = MakeArrayResourceDesc(array);

  hipSurfaceObject_t surf_obj = 0;
  const hipError_t status = hipCreateSurfaceObject(&surf_obj, &res);
  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST(HipTest::SkipReason::kTextureImageUnsupported);
  }
  HIP_CHECK(status);
  cleanup.Add([&] { (void)hipDestroySurfaceObject(surf_obj); });

  hipChannelFormatDesc returned{};
  HIP_CHECK(hipGetChannelDesc(&returned, array));

  REQUIRE(returned.x == channel.x);
  REQUIRE(returned.y == channel.y);
  REQUIRE(returned.z == channel.z);
  REQUIRE(returned.w == channel.w);
  REQUIRE(returned.f == channel.f);
}

HIP_TEST_CASE(Contract_Texture_GetResourceViewDesc_RoundTripsArrayResource) {
  CHECK_IMAGE_SUPPORT;
  hip::contract::ContractCleanup cleanup;

  hipArray_t array = nullptr;
  const hipChannelFormatDesc channel = ByteChannelDesc();
  HIP_CHECK(hipMallocArray(&array, &channel, kArrayWidth, kArrayHeight));
  cleanup.Add([&] { (void)hipFreeArray(array); });

  const hipResourceDesc res = MakeArrayResourceDesc(array);
  const hipTextureDesc tex = MakeTextureDesc();

  // Create the texture object with a fully specified resource view so the
  // runtime resource-view query has every field defined to return. A single
  // non-mipmapped, single-layer array view is used, so the mipmap-level and
  // layer bounds are all zero.
  hipResourceViewDesc view{};
  view.format = hipResViewFormatUnsignedChar1;
  view.width = kArrayWidth;
  view.height = kArrayHeight;
  view.depth = 0;
  view.firstMipmapLevel = 0;
  view.lastMipmapLevel = 0;
  view.firstLayer = 0;
  view.lastLayer = 0;

  hipTextureObject_t tex_obj = 0;
  const hipError_t create_status = hipCreateTextureObject(&tex_obj, &res, &tex, &view);
  if (create_status == hipErrorNotSupported) {
    HIP_SKIP_TEST(HipTest::SkipReason::kTextureImageUnsupported);
  }
  HIP_CHECK(create_status);
  cleanup.Add([&] { (void)hipDestroyTextureObject(tex_obj); });

  // The runtime resource-view query must return the view the object was created
  // with (or report the query unsupported on this runtime path). Every field the
  // descriptor carries is asserted so a dropped or corrupted field is caught,
  // not just format and width.
  hipResourceViewDesc returned{};
  const hipError_t status = hipGetTextureObjectResourceViewDesc(&returned, tex_obj);
  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST("hipGetTextureObjectResourceViewDesc is not supported by this runtime path.");
  }
  HIP_CHECK(status);

  REQUIRE(returned.format == view.format);
  REQUIRE(returned.width == view.width);
  REQUIRE(returned.height == view.height);
  REQUIRE(returned.depth == view.depth);
  REQUIRE(returned.firstMipmapLevel == view.firstMipmapLevel);
  REQUIRE(returned.lastMipmapLevel == view.lastMipmapLevel);
  REQUIRE(returned.firstLayer == view.firstLayer);
  REQUIRE(returned.lastLayer == view.lastLayer);
}

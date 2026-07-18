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
constexpr size_t kLinearBytes = 256;
constexpr size_t kArrayWidth = 8;
constexpr size_t kArrayHeight = 4;

HIP_RESOURCE_DESC LinearResourceDesc(void* device_ptr) {
  HIP_RESOURCE_DESC desc{};
  desc.resType = HIP_RESOURCE_TYPE_LINEAR;
  desc.res.linear.devPtr = reinterpret_cast<hipDeviceptr_t>(device_ptr);
  desc.res.linear.format = HIP_AD_FORMAT_UNSIGNED_INT8;
  desc.res.linear.numChannels = 1;
  desc.res.linear.sizeInBytes = kLinearBytes;
  return desc;
}

HIP_RESOURCE_DESC ArrayResourceDesc(hipArray_t array) {
  HIP_RESOURCE_DESC desc{};
  desc.resType = HIP_RESOURCE_TYPE_ARRAY;
  desc.res.array.hArray = array;
  return desc;
}

HIP_TEXTURE_DESC TextureDesc() {
  HIP_TEXTURE_DESC desc{};
  desc.addressMode[0] = HIP_TR_ADDRESS_MODE_CLAMP;
  desc.addressMode[1] = HIP_TR_ADDRESS_MODE_CLAMP;
  desc.addressMode[2] = HIP_TR_ADDRESS_MODE_CLAMP;
  desc.filterMode = HIP_TR_FILTER_MODE_POINT;
  desc.flags = 0;
  return desc;
}

bool CreateTextureOrSkip(hipTextureObject_t* texture, const HIP_RESOURCE_DESC* resource,
                         const HIP_TEXTURE_DESC* texture_desc,
                         const HIP_RESOURCE_VIEW_DESC* view_desc) {
  const hipError_t status = hipTexObjectCreate(texture, resource, texture_desc, view_desc);
  if (status == hipErrorNotSupported) {
    return false;
  }
  HIP_CHECK(status);
  return true;
}
}  // namespace

HIP_TEST_CASE(Contract_DriverTexture_CreateAndDestroy_LinearResource_Succeeds) {
  CHECK_IMAGE_SUPPORT;
  hip::contract::ContractCleanup cleanup;

  void* device_ptr = nullptr;
  HIP_CHECK(hipMalloc(&device_ptr, kLinearBytes));
  cleanup.Add([&] { (void)hipFree(device_ptr); });

  const auto resource = LinearResourceDesc(device_ptr);
  const auto texture_desc = TextureDesc();
  hipTextureObject_t texture = 0;

  if (!CreateTextureOrSkip(&texture, &resource, &texture_desc, nullptr)) {
    HIP_SKIP_TEST(HipTest::SkipReason::kTextureImageUnsupported);
  }
  cleanup.Add([&] { (void)hipTexObjectDestroy(texture); });

  REQUIRE(texture != 0);
}

HIP_TEST_CASE(Contract_DriverTexture_GetResourceDesc_RoundTripsLinearResource) {
  CHECK_IMAGE_SUPPORT;
  hip::contract::ContractCleanup cleanup;

  void* device_ptr = nullptr;
  HIP_CHECK(hipMalloc(&device_ptr, kLinearBytes));
  cleanup.Add([&] { (void)hipFree(device_ptr); });

  const auto resource = LinearResourceDesc(device_ptr);
  const auto texture_desc = TextureDesc();
  hipTextureObject_t texture = 0;

  if (!CreateTextureOrSkip(&texture, &resource, &texture_desc, nullptr)) {
    HIP_SKIP_TEST(HipTest::SkipReason::kTextureImageUnsupported);
  }
  cleanup.Add([&] { (void)hipTexObjectDestroy(texture); });

  HIP_RESOURCE_DESC returned{};
  HIP_CHECK(hipTexObjectGetResourceDesc(&returned, texture));

  REQUIRE(returned.resType == resource.resType);
  REQUIRE(returned.res.linear.devPtr == resource.res.linear.devPtr);
  REQUIRE(returned.res.linear.format == resource.res.linear.format);
  REQUIRE(returned.res.linear.numChannels == resource.res.linear.numChannels);
  REQUIRE(returned.res.linear.sizeInBytes == resource.res.linear.sizeInBytes);
}

HIP_TEST_CASE(Contract_DriverTexture_GetTextureDesc_RoundTripsFlags) {
  CHECK_IMAGE_SUPPORT;
  hip::contract::ContractCleanup cleanup;

  void* device_ptr = nullptr;
  HIP_CHECK(hipMalloc(&device_ptr, kLinearBytes));
  cleanup.Add([&] { (void)hipFree(device_ptr); });

  const auto resource = LinearResourceDesc(device_ptr);
  auto texture_desc = TextureDesc();
  texture_desc.flags = HIP_TRSF_READ_AS_INTEGER;
  hipTextureObject_t texture = 0;

  if (!CreateTextureOrSkip(&texture, &resource, &texture_desc, nullptr)) {
    HIP_SKIP_TEST(HipTest::SkipReason::kTextureImageUnsupported);
  }
  cleanup.Add([&] { (void)hipTexObjectDestroy(texture); });

  HIP_TEXTURE_DESC returned{};
  HIP_CHECK(hipTexObjectGetTextureDesc(&returned, texture));

  REQUIRE(returned.addressMode[0] == texture_desc.addressMode[0]);
  REQUIRE(returned.addressMode[1] == texture_desc.addressMode[1]);
  REQUIRE(returned.addressMode[2] == texture_desc.addressMode[2]);
  REQUIRE(returned.filterMode == texture_desc.filterMode);
  REQUIRE(returned.flags == texture_desc.flags);
}

HIP_TEST_CASE(Contract_DriverTexture_CreateAndGetResourceDesc_ArrayResource) {
  CHECK_IMAGE_SUPPORT;
  hip::contract::ContractCleanup cleanup;

  hipArray_t array = nullptr;
  const auto channel = hipCreateChannelDesc<uint8_t>();
  HIP_CHECK(hipMallocArray(&array, &channel, kArrayWidth, kArrayHeight));
  cleanup.Add([&] { (void)hipFreeArray(array); });

  const auto resource = ArrayResourceDesc(array);
  const auto texture_desc = TextureDesc();
  hipTextureObject_t texture = 0;

  if (!CreateTextureOrSkip(&texture, &resource, &texture_desc, nullptr)) {
    HIP_SKIP_TEST(HipTest::SkipReason::kTextureImageUnsupported);
  }
  cleanup.Add([&] { (void)hipTexObjectDestroy(texture); });

  HIP_RESOURCE_DESC returned{};
  HIP_CHECK(hipTexObjectGetResourceDesc(&returned, texture));

  REQUIRE(returned.resType == HIP_RESOURCE_TYPE_ARRAY);
  REQUIRE(returned.res.array.hArray == array);
}

// BACKEND-DIFF: The resource-view path is exercised only on AMD. It requires the
// driver-style resource-view format enumerator HIP_RES_VIEW_FORMAT_FLOAT_1X32,
// which the NVIDIA backend does not define (its hipResViewFormat* aliases map to
// the runtime-style cudaResViewFormat* enum instead, used by the higher-level
// texture object API rather than hipTexObjectCreate's HIP_RESOURCE_VIEW_DESC).
// Parity would require NVIDIA to define the driver-style resource-view format
// enumerators.
#if HT_AMD
HIP_TEST_CASE(Contract_DriverTexture_GetResourceViewDesc_IsQueryableWhenSupported) {
  CHECK_IMAGE_SUPPORT;
  hip::contract::ContractCleanup cleanup;

  hipArray_t array = nullptr;
  const auto channel = hipCreateChannelDesc<float>();
  HIP_CHECK(hipMallocArray(&array, &channel, kArrayWidth, 1));
  cleanup.Add([&] { (void)hipFreeArray(array); });

  const auto resource = ArrayResourceDesc(array);
  const auto texture_desc = TextureDesc();
  HIP_RESOURCE_VIEW_DESC view{};
  view.format = HIP_RES_VIEW_FORMAT_FLOAT_1X32;
  view.width = kArrayWidth;
  view.height = 0;
  view.depth = 0;
  hipTextureObject_t texture = 0;

  if (!CreateTextureOrSkip(&texture, &resource, &texture_desc, &view)) {
    HIP_SKIP_TEST(HipTest::SkipReason::kTextureImageUnsupported);
  }
  cleanup.Add([&] { (void)hipTexObjectDestroy(texture); });

  HIP_RESOURCE_VIEW_DESC returned{};
  const hipError_t status = hipTexObjectGetResourceViewDesc(&returned, texture);
  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST("hipTexObjectGetResourceViewDesc is not supported by this runtime path.");
  }
  HIP_CHECK(status);

  REQUIRE(returned.format == view.format);
  REQUIRE(returned.width == view.width);
}
#endif  // HT_AMD

HIP_TEST_CASE(Contract_DriverTexture_Create_InvalidArgs_AreRejected) {
  CHECK_IMAGE_SUPPORT;
  hip::contract::ContractCleanup cleanup;

  void* device_ptr = nullptr;
  HIP_CHECK(hipMalloc(&device_ptr, kLinearBytes));
  cleanup.Add([&] { (void)hipFree(device_ptr); });

  const auto resource = LinearResourceDesc(device_ptr);
  const auto texture_desc = TextureDesc();
  hipTextureObject_t texture = 0;

  REQUIRE(hipTexObjectCreate(nullptr, &resource, &texture_desc, nullptr) != hipSuccess);
  REQUIRE(hipTexObjectCreate(&texture, nullptr, &texture_desc, nullptr) != hipSuccess);
  REQUIRE(hipTexObjectCreate(&texture, &resource, nullptr, nullptr) != hipSuccess);
}

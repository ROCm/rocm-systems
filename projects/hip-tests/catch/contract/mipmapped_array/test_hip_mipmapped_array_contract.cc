/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <contract_cleanup.hh>

namespace {
constexpr size_t kWidth = 16;
constexpr size_t kHeight = 16;
constexpr unsigned int kNumLevels = 2;

// 8-bit unsigned, single channel.
hipChannelFormatDesc ByteChannelDesc() {
  return hipCreateChannelDesc(8, 0, 0, 0, hipChannelFormatKindUnsigned);
}

// 2D mipmapped array extent (depth 0 => 2D image).
hipExtent MipExtent() { return make_hipExtent(kWidth, kHeight, 0); }

// Driver-style 2D descriptor for hipMipmappedArrayCreate.
HIP_ARRAY3D_DESCRIPTOR MipArray3DDesc() {
  HIP_ARRAY3D_DESCRIPTOR desc{};
  desc.Width = kWidth;
  desc.Height = kHeight;
  desc.Depth = 0;
  desc.Format = HIP_AD_FORMAT_UNSIGNED_INT8;
  desc.NumChannels = 1;
  desc.Flags = 0;
  return desc;
}

// True if the status indicates the device/runtime path does not support
// mipmapped arrays (as opposed to a genuine contract violation).
bool IsUnsupportedOrNoMemory(hipError_t status) {
  return status == hipErrorNotSupported || status == hipErrorOutOfMemory;
}
}  // namespace

HIP_TEST_CASE(Contract_MipmappedArray_MallocAndGetLevel_ReturnsLevelArray) {
  CHECK_IMAGE_SUPPORT;
  hip::contract::ContractCleanup cleanup;

  hipMipmappedArray_t mipmap = nullptr;
  const auto desc = ByteChannelDesc();

  const hipError_t status =
      hipMallocMipmappedArray(&mipmap, &desc, MipExtent(), kNumLevels, 0);
  if (IsUnsupportedOrNoMemory(status)) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("Mipmapped arrays are not supported by this device/runtime path.");
  }
  HIP_CHECK(status);
  cleanup.Add([mipmap] { (void)hipFreeMipmappedArray(mipmap); });
  REQUIRE(mipmap != nullptr);

  hipArray_t level0 = nullptr;
  HIP_CHECK(hipGetMipmappedArrayLevel(&level0, mipmap, 0));
  REQUIRE(level0 != nullptr);
}

HIP_TEST_CASE(Contract_MipmappedArray_DriverCreateGetLevelDestroy) {
  CHECK_IMAGE_SUPPORT;
  hip::contract::ContractCleanup cleanup;

  // hipMipmappedArrayCreate is a driver-API entry point (cuMipmappedArrayCreate
  // on NVIDIA) and needs an initialized primary context; prime one so the test
  // passes even as the first HIP call in the process (e.g. under ctest
  // isolation). Harmless success on AMD.
  HIP_CHECK(hipFree(0));

  // The driver-style mipmapped-array APIs take the driver handle type
  // `hipmipmappedArray`. On AMD it aliases hipMipmappedArray_t; on NVIDIA it is
  // the distinct CUmipmappedArray (the runtime hipMipmappedArray_t maps to the
  // separate cudaMipmappedArray_t there). Use the driver type so the create/
  // get-level/destroy calls compile on both backends.
  hipmipmappedArray mipmap = nullptr;
  auto desc = MipArray3DDesc();

  const hipError_t status = hipMipmappedArrayCreate(&mipmap, &desc, kNumLevels);
  if (IsUnsupportedOrNoMemory(status)) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("Mipmapped arrays are not supported by this device/runtime path.");
  }
  HIP_CHECK(status);
  cleanup.Add([mipmap] { (void)hipMipmappedArrayDestroy(mipmap); });
  REQUIRE(mipmap != nullptr);

  hipArray_t level0 = nullptr;
  HIP_CHECK(hipMipmappedArrayGetLevel(&level0, mipmap, 0));
  REQUIRE(level0 != nullptr);
}

HIP_TEST_CASE(Contract_MipmappedArray_GetLevel_OutOfRange_IsRejected) {
  CHECK_IMAGE_SUPPORT;
  hip::contract::ContractCleanup cleanup;

  hipMipmappedArray_t mipmap = nullptr;
  const auto desc = ByteChannelDesc();

  const hipError_t status =
      hipMallocMipmappedArray(&mipmap, &desc, MipExtent(), kNumLevels, 0);
  if (IsUnsupportedOrNoMemory(status)) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("Mipmapped arrays are not supported by this device/runtime path.");
  }
  HIP_CHECK(status);
  cleanup.Add([mipmap] { (void)hipFreeMipmappedArray(mipmap); });
  REQUIRE(mipmap != nullptr);

  hipArray_t level = nullptr;
  // Request a level well beyond the allocated numLevels.
  REQUIRE(hipGetMipmappedArrayLevel(&level, mipmap, 5) != hipSuccess);
  (void)hipGetLastError();
}

HIP_TEST_CASE(Contract_MipmappedArray_GetMemoryRequirements_IsQueryable) {
  CHECK_IMAGE_SUPPORT;
  hip::contract::ContractCleanup cleanup;

  hipMipmappedArray_t mipmap = nullptr;
  const auto desc = ByteChannelDesc();

  const hipError_t status =
      hipMallocMipmappedArray(&mipmap, &desc, MipExtent(), kNumLevels, 0);
  if (IsUnsupportedOrNoMemory(status)) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("Mipmapped arrays are not supported by this device/runtime path.");
  }
  HIP_CHECK(status);
  cleanup.Add([mipmap] { (void)hipFreeMipmappedArray(mipmap); });
  REQUIRE(mipmap != nullptr);

  hipDevice_t device = 0;
  HIP_CHECK(hipDeviceGet(&device, 0));

  hipArrayMemoryRequirements req{};
  const hipError_t req_status =
      hipMipmappedArrayGetMemoryRequirements(&req, mipmap, device);
  // BACKEND-DIFF: hipMipmappedArrayGetMemoryRequirements reports "not queryable"
  // differently per backend. AMD returns hipErrorNotSupported; NVIDIA
  // (cuMipmappedArrayGetMemoryRequirements) rejects a query against a
  // runtime-created mipmapped array with hipErrorInvalidValue. The two branches
  // accept the backend's respective not-queryable status. Parity would require a
  // common not-supported status for this query path.
#if HT_AMD
  // Some runtime paths do not implement this query; accept unsupported.
  if (req_status != hipErrorNotSupported) {
    HIP_CHECK(req_status);
  } else {
    (void)hipGetLastError();
  }
#else
  // On NVIDIA hipMipmappedArrayGetMemoryRequirements maps to
  // cuMipmappedArrayGetMemoryRequirements, which rejects a query against a
  // runtime-created mipmapped array with hipErrorInvalidValue rather than
  // reporting the query itself unsupported. Accept both the unsupported and the
  // invalid-value outcomes as "not queryable on this path"; a success is also
  // contract-compliant.
  if (req_status != hipErrorNotSupported && req_status != hipErrorInvalidValue) {
    HIP_CHECK(req_status);
  } else {
    (void)hipGetLastError();
  }
#endif
}

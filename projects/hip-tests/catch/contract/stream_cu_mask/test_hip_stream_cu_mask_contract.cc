/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstdint>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

#if HT_AMD
namespace {
constexpr uint32_t kMaskWordBits = 32;

void RequireDevice() {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count <= 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kNoGpuDevice);
  }
}

std::vector<uint32_t> DefaultCuMask() {
  RequireDevice();

  int device = 0;
  HIP_CHECK(hipGetDevice(&device));

  hipDeviceProp_t properties{};
  HIP_CHECK(hipGetDeviceProperties(&properties, device));
  REQUIRE(properties.multiProcessorCount > 0);

  const size_t words = (static_cast<size_t>(properties.multiProcessorCount) + kMaskWordBits - 1) /
                       kMaskWordBits;
  std::vector<uint32_t> mask(words, 0);
  for (int cu = 0; cu < properties.multiProcessorCount; ++cu) {
    mask[static_cast<size_t>(cu) / kMaskWordBits] |=
        uint32_t{1} << (static_cast<uint32_t>(cu) % kMaskWordBits);
  }
  return mask;
}

std::vector<uint32_t> QueryCuMask(hipStream_t stream, size_t words) {
  std::vector<uint32_t> mask(words, 0);
  HIP_CHECK(hipExtStreamGetCUMask(stream, static_cast<uint32_t>(mask.size()), mask.data()));
  return mask;
}

bool CreateStreamWithMaskOrSkip(hipStream_t* stream, const std::vector<uint32_t>& mask) {
  const hipError_t status =
      hipExtStreamCreateWithCUMask(stream, static_cast<uint32_t>(mask.size()), mask.data());
  if (status == hipErrorNotSupported) {
    return false;
  }
  HIP_CHECK(status);
  return true;
}
}  // namespace

HIP_TEST_CASE(Contract_StreamCuMask_DefaultMaskRoundTrips_AllCUsActive) {
  const auto default_mask = DefaultCuMask();
  hipStream_t stream = nullptr;

  if (!CreateStreamWithMaskOrSkip(&stream, default_mask)) {
    HIP_SKIP_TEST("hipExtStreamCreateWithCUMask is not supported by this runtime path.");
  }

  const auto returned_mask = QueryCuMask(stream, default_mask.size());
  REQUIRE(returned_mask == default_mask);

  HIP_CHECK(hipStreamDestroy(stream));
}

HIP_TEST_CASE(Contract_StreamCuMask_CreateWithDefaultMask_Succeeds) {
  const auto default_mask = DefaultCuMask();
  hipStream_t stream = nullptr;

  if (!CreateStreamWithMaskOrSkip(&stream, default_mask)) {
    HIP_SKIP_TEST("hipExtStreamCreateWithCUMask is not supported by this runtime path.");
  }

  REQUIRE(stream != nullptr);

  HIP_CHECK(hipStreamDestroy(stream));
}

HIP_TEST_CASE(Contract_StreamCuMask_CreateRejectsInvalidArgs) {
  const auto default_mask = DefaultCuMask();
  hipStream_t stream = nullptr;

  REQUIRE(hipExtStreamCreateWithCUMask(nullptr, static_cast<uint32_t>(default_mask.size()),
                                       default_mask.data()) != hipSuccess);
  REQUIRE(hipExtStreamCreateWithCUMask(&stream, 0, default_mask.data()) != hipSuccess);
  REQUIRE(hipExtStreamCreateWithCUMask(&stream, static_cast<uint32_t>(default_mask.size()),
                                       nullptr) != hipSuccess);
}

HIP_TEST_CASE(Contract_StreamCuMask_GetRejectsInvalidArgs) {
  const auto default_mask = DefaultCuMask();
  std::vector<uint32_t> mask(default_mask.size(), 0);
  hipStream_t stream = nullptr;

  if (!CreateStreamWithMaskOrSkip(&stream, default_mask)) {
    HIP_SKIP_TEST("hipExtStreamCreateWithCUMask is not supported by this runtime path.");
  }

  const hipError_t null_mask_status =
      hipExtStreamGetCUMask(stream, static_cast<uint32_t>(mask.size()), nullptr);
  const hipError_t zero_size_status = hipExtStreamGetCUMask(stream, 0, mask.data());

  HIP_CHECK(hipStreamDestroy(stream));

  REQUIRE(null_mask_status == hipErrorInvalidValue);
  REQUIRE(zero_size_status == hipErrorInvalidValue);
}
#endif  // HT_AMD

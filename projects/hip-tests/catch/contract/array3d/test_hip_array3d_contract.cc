/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <array>
#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <contract_cleanup.hh>

namespace {
constexpr size_t kWidth = 4;
constexpr size_t kHeight = 3;
constexpr size_t kDepth = 2;

std::array<uint8_t, kWidth * kHeight * kDepth> MakePattern(uint8_t seed) {
  std::array<uint8_t, kWidth * kHeight * kDepth> pattern{};
  for (size_t i = 0; i < pattern.size(); ++i) {
    pattern[i] = static_cast<uint8_t>(seed + i);
  }
  return pattern;
}

hipChannelFormatDesc ByteChannelDesc() { return hipCreateChannelDesc<uint8_t>(); }

hipPitchedPtr HostPitchedPtr(void* ptr) {
  return make_hipPitchedPtr(ptr, kWidth, kWidth, kHeight);
}

hipExtent ArrayExtent() { return make_hipExtent(kWidth, kHeight, kDepth); }
}

// @asserts: hipMalloc3DArray - allocating a 3D array with a valid descriptor and extent yields a non-null handle
HIP_TEST_CASE(Contract_Array3D_Malloc3DArray_ReturnsUsableArray) {
  CHECK_IMAGE_SUPPORT;
  hip::contract::ContractCleanup cleanup;
  hipArray_t array = nullptr;
  const auto desc = ByteChannelDesc();

  HIP_CHECK(hipMalloc3DArray(&array, &desc, ArrayExtent(), 0));
  cleanup.Add([array] { (void)hipFreeArray(array); });

  REQUIRE(array != nullptr);
}

// @asserts: hipMemcpy3D - copying host bytes into a 3D array and back reproduces the original bytes exactly
HIP_TEST_CASE(Contract_Array3D_Memcpy3DToArrayAndBack_RoundTripsBytes) {
  CHECK_IMAGE_SUPPORT;
  hip::contract::ContractCleanup cleanup;
  const auto src = MakePattern(0x6a);
  std::array<uint8_t, kWidth * kHeight * kDepth> dst{};
  hipArray_t array = nullptr;
  const auto desc = ByteChannelDesc();

  HIP_CHECK(hipMalloc3DArray(&array, &desc, ArrayExtent(), 0));
  cleanup.Add([array] { (void)hipFreeArray(array); });

  hipMemcpy3DParms h2a{};
  h2a.srcPtr = HostPitchedPtr(const_cast<uint8_t*>(src.data()));
  h2a.dstArray = array;
  h2a.extent = ArrayExtent();
  h2a.kind = hipMemcpyHostToDevice;
  HIP_CHECK(hipMemcpy3D(&h2a));

  hipMemcpy3DParms a2h{};
  a2h.srcArray = array;
  a2h.dstPtr = HostPitchedPtr(dst.data());
  a2h.extent = ArrayExtent();
  a2h.kind = hipMemcpyDeviceToHost;
  HIP_CHECK(hipMemcpy3D(&a2h));

  REQUIRE(dst == src);
}

// @asserts: hipFreeArray - freeing an allocated 3D array succeeds
HIP_TEST_CASE(Contract_Array3D_FreeArray_Succeeds) {
  CHECK_IMAGE_SUPPORT;
  hipArray_t array = nullptr;
  const auto desc = ByteChannelDesc();

  HIP_CHECK(hipMalloc3DArray(&array, &desc, ArrayExtent(), 0));
  HIP_CHECK(hipFreeArray(array));
}

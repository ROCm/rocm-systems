/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include <hip_test_process.hh>
#include <resource_guards.hh>
#include <utils.hh>

template <typename> struct get_test_values_unsupported_type : std::false_type {};

template <typename TestType> constexpr std::pair<TestType, TestType> get_test_values() {
  if constexpr (std::is_same_v<TestType, char>) {
    return {'a', 'b'};
  } else if constexpr (std::is_integral_v<TestType>) {
    return {static_cast<TestType>(10), static_cast<TestType>(4)};
  } else if constexpr (std::is_floating_point_v<TestType>) {
    return {static_cast<TestType>(2.25f), static_cast<TestType>(0.25f)};
  } else {
    static_assert(get_test_values_unsupported_type<TestType>::value,
                  "get_test_values: add an explicit branch for this TestType");
  }
}

inline hipError_t getSwapExpectedReturn(const LinearAllocs allocTypeSrc,
                                        const LinearAllocs allocTypeDst, const int srcDevice = 0,
                                        const int dstDevice = 0) {
  // The swap endpoints are peer-to-peer when they live on different devices.
  const bool is_p2p = srcDevice != dstDevice;

  // Support for H2H will be implemented later.
  if (allocTypeSrc == LinearAllocs::malloc || allocTypeDst == LinearAllocs::malloc) {
    return hipErrorNotSupported;
  }

  if (allocTypeSrc == LinearAllocs::hipHostMalloc && allocTypeDst == LinearAllocs::hipHostMalloc) {
    return hipErrorNotSupported;
  }

  // Support for D2D will be implemented later.
  if (allocTypeSrc == LinearAllocs::hipMalloc && allocTypeDst == LinearAllocs::hipMalloc &&
      !is_p2p) {
    return hipErrorNotSupported;
  }

  int major, minor;
  HIP_CHECK(hipDeviceComputeCapability(&major, &minor, 0));
  if (major == 9 && minor >= 4) {
    return hipSuccess;
  }

  return hipErrorNotSupported;
}

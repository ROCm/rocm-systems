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

// A swap exchanges both endpoints, so the two sides are symmetric and named a/b rather than src/dst.
inline hipError_t getSwapExpectedReturn(const LinearAllocs allocTypeA, const LinearAllocs allocTypeB,
                                        const int deviceA = 0, const int deviceB = 0) {
  // The swap endpoints are peer-to-peer when they live on different devices.
  const bool is_p2p = deviceA != deviceB;

  // Support for H2H will be implemented later.
  if (allocTypeA == LinearAllocs::malloc || allocTypeB == LinearAllocs::malloc) {
    return hipErrorNotSupported;
  }

  if (allocTypeA == LinearAllocs::hipHostMalloc && allocTypeB == LinearAllocs::hipHostMalloc) {
    return hipErrorNotSupported;
  }

  // Support for D2D will be implemented later.
  if (allocTypeA == LinearAllocs::hipMalloc && allocTypeB == LinearAllocs::hipMalloc && !is_p2p) {
    return hipErrorNotSupported;
  }

  const auto supportsSwap = [](int device) {
    int major, minor;
    HIP_CHECK(hipDeviceComputeCapability(&major, &minor, device));
    return major == 9 && minor >= 4;
  };

  if (supportsSwap(deviceA) && supportsSwap(deviceB)) {
    return hipSuccess;
  }

  return hipErrorNotSupported;
}

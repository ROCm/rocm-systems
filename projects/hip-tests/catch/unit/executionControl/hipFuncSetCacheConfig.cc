/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "execution_control_common.hh"

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>

/**
 * @addtogroup hipFuncSetCacheConfig hipFuncSetCacheConfig
 * @{
 * @ingroup ExecutionTest
 * `hipFuncSetCacheConfig(const void* func, hipFuncCache_t config)` -
 * Set Cache configuration for a specific function.
 */

namespace {
constexpr std::array<hipFuncCache_t, 4> kCacheConfigs{
    hipFuncCachePreferNone, hipFuncCachePreferShared, hipFuncCachePreferL1,
    hipFuncCachePreferEqual};
}  // anonymous namespace

/**
 * Test Description
 * ------------------------
 *  - Sets cache config for every cache config enumeration.
 * Test source
 * ------------------------
 *  - unit/executionControl/hipFuncSetCacheConfig.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipFuncSetCacheConfig_Positive_Basic) {
  const auto cache_config = GENERATE(from_range(begin(kCacheConfigs), end(kCacheConfigs)));

  HIP_CHECK(hipFuncSetCacheConfig(reinterpret_cast<void*>(kernel), cache_config));

  kernel<<<1, 1>>>();
  HIP_CHECK(hipDeviceSynchronize());
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When pointer to the kernel function is `nullptr`
 *      - Expected output: return `hipErrorInvalidDeviceFunction`
 *    -# When cache config enumeration is invalid
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/executionControl/hipFuncSetCacheConfig.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipFuncSetCacheConfig_Negative_Parameters) {
  SECTION("func == nullptr") {
    HIP_CHECK_ERROR(hipFuncSetCacheConfig(nullptr, hipFuncCachePreferNone),
                    hipErrorInvalidDeviceFunction);
  }
  SECTION("invalid cache config") {
    HIP_CHECK_ERROR(
        hipFuncSetCacheConfig(reinterpret_cast<void*>(kernel), static_cast<hipFuncCache_t>(-1)),
        hipErrorInvalidValue);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Verifies that hipFuncSetCacheConfig maps cache config hints to the
 *    expected preferredShmemCarveout percentages:
 *    -# hipFuncCachePreferShared  -> 100%
 *    -# hipFuncCachePreferL1      -> 1%
 *    -# hipFuncCachePreferEqual   -> 50%
 *    -# hipFuncCachePreferNone    -> 0% (no preference)
 * Test source
 * ------------------------
 *  - unit/executionControl/hipFuncSetCacheConfig.cc
 */
HIP_TEST_CASE(Unit_hipFuncSetCacheConfig_Positive_VerifyCarveoutMapping) {
#if HT_NVIDIA
  HIP_SKIP_TEST(HipTest::SkipReason::kApiUnsupportedOnNvidia);
#endif

  struct {
    hipFuncCache_t config;
    int expectedCarveout;
  } testCases[] = {
      {hipFuncCachePreferNone, 0},
      {hipFuncCachePreferShared, 100},
      {hipFuncCachePreferL1, 1},
      {hipFuncCachePreferEqual, 50},
  };

  for (const auto& tc : testCases) {
    HIP_CHECK(hipFuncSetCacheConfig(reinterpret_cast<void*>(kernel), tc.config));

    hipFuncAttributes attributes;
    HIP_CHECK(hipFuncGetAttributes(&attributes, reinterpret_cast<void*>(kernel)));

    REQUIRE(attributes.preferredShmemCarveout == tc.expectedCarveout);
  }
}

/**
 * End doxygen group ExecutionTest.
 * @}
 */

/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

/**
 * @addtogroup hipDeviceSetCacheConfig hipDeviceSetCacheConfig
 * @{
 * @ingroup DeviceTest
 * `hipDeviceSetCacheConfig(hipFuncCache_t cacheConfig)` -
 * Set L1/Shared cache partition.
 */

namespace {
constexpr std::array<hipFuncCache_t, 4> kCacheConfigs{
    hipFuncCachePreferNone, hipFuncCachePreferShared, hipFuncCachePreferL1,
    hipFuncCachePreferEqual};
}  // anonymous namespace

/**
 * Test Description
 * ------------------------
 *  - Check that `hipSuccess` is returned for all enumerators of `hipFuncCache_t`
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceSetGetCacheConfig.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_hipDeviceSetCacheConfig_Positive_Basic) {
  const auto device = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipSetDevice(device));
  INFO("Current device is: " << device);

  const auto cache_config =
      GENERATE(from_range(std::begin(kCacheConfigs), std::end(kCacheConfigs)));
  HIP_CHECK(hipDeviceSetCacheConfig(cache_config));
}

/**
 * End doxygen group hipDeviceSetCacheConfig.
 * @}
 */

/**
 * @addtogroup hipDeviceGetCacheConfig hipDeviceGetCacheConfig
 * @{
 * @ingroup DeviceTest
 * `hipDeviceGetCacheConfig(hipFuncCache_t* cacheConfig)` -
 * Get Cache configuration for a specific Device.
 */

/**
 * Test Description
 * ------------------------
 *  - Check that default cache config is returned if set
 *    has not been called previously.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceSetGetCacheConfig.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_hipDeviceGetCacheConfig_Positive_Default) {
  const auto device = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipSetDevice(device));
  INFO("Current device is: " << device);

  hipFuncCache_t cache_config;
  HIP_CHECK(hipDeviceGetCacheConfig(&cache_config));
  REQUIRE(cache_config == hipFuncCachePreferNone);
}

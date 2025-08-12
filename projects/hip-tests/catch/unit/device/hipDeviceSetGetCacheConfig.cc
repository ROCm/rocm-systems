/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
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
TEST_CASE("Unit_hipDeviceSetCacheConfig_Positive_Basic") {
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
TEST_CASE("Unit_hipDeviceGetCacheConfig_Positive_Default") {
  const auto device = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipSetDevice(device));
  INFO("Current device is: " << device);

  hipFuncCache_t cache_config;
  HIP_CHECK(hipDeviceGetCacheConfig(&cache_config));
  REQUIRE(cache_config == hipFuncCachePreferNone);
}

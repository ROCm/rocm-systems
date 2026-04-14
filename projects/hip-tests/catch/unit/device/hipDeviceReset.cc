/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>

/**
 * @addtogroup hipDeviceReset hipDeviceReset
 * @{
 * @ingroup DeviceTest
 * `hipDeviceReset(void)` -
 * The state of current device is discarded and updated to a fresh state.
 *
 * Calling this function deletes all streams created, memory allocated, kernels running, events
 * created. Make sure that no other thread is using the device or streams, memory, kernels, events
 * associated with the current device.
 */

/**
 * Test Description
 * ------------------------
 *  - Validates that device reset frees allocated memory and
 *    reverts modified flags and configs to its default values.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceReset.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_hipDeviceReset_Positive_Basic) {
  const auto device = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipSetDevice(device));

  INFO("Current device is: " << device);
  HIP_CHECK(hipDeviceReset());

  unsigned int flags_before = 0u;
  HIP_CHECK(hipGetDeviceFlags(&flags_before));
  hipSharedMemConfig mem_config_before;
  HIP_CHECK(hipDeviceGetSharedMemConfig(&mem_config_before));

  void* ptr = nullptr;
  HIP_CHECK(hipMalloc(&ptr, 500));

  const auto cache_config_ret = hipDeviceSetCacheConfig(hipFuncCachePreferL1);
  REQUIRE((cache_config_ret == hipSuccess || cache_config_ret == hipErrorNotSupported));

  const auto shared_mem_config_ret = hipDeviceSetSharedMemConfig(
      mem_config_before == hipSharedMemBankSizeFourByte ? hipSharedMemBankSizeEightByte
                                                        : hipSharedMemBankSizeFourByte);
  REQUIRE((shared_mem_config_ret == hipSuccess || shared_mem_config_ret == hipErrorNotSupported));

  HIP_CHECK(hipSetDeviceFlags(hipDeviceScheduleBlockingSync));

  HIP_CHECK(hipDeviceReset());

  unsigned int flags_after = 0u;
  CHECK(hipGetDeviceFlags(&flags_after) == hipSuccess);
  CHECK(flags_after == flags_before);

  // This will faill in ASAN due to how we handle free
#if !defined(ENABLE_ADDRESS_SANITIZER)
  CHECK(hipFree(ptr) == hipErrorInvalidValue);
#endif

  if (cache_config_ret == hipSuccess) {
    hipFuncCache_t cache_config;
    CHECK(hipDeviceGetCacheConfig(&cache_config) == hipSuccess);
    CHECK(cache_config == hipFuncCachePreferNone);
  }

  if (shared_mem_config_ret == hipSuccess) {
    hipSharedMemConfig mem_config_after;
    CHECK(hipDeviceGetSharedMemConfig(&mem_config_after) == hipSuccess);
    CHECK(mem_config_after == mem_config_before);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Resets device from another thread
 *  - Validates that device reset frees allocated memory from the main
 *    thread, and reverts modified flags and configs to its default values.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceReset.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_hipDeviceReset_Positive_Threaded) {
  HIP_CHECK(hipSetDevice(0));
  INFO("Current device is: " << 0);
  HIP_CHECK(hipDeviceReset());

  unsigned int flags_before = 0u;
  HIP_CHECK(hipGetDeviceFlags(&flags_before));
  hipSharedMemConfig mem_config_before;
  HIP_CHECK(hipDeviceGetSharedMemConfig(&mem_config_before));

  void* ptr = nullptr;
  HIP_CHECK(hipMalloc(&ptr, 500));

  const auto cache_config_ret = hipDeviceSetCacheConfig(hipFuncCachePreferL1);
  REQUIRE((cache_config_ret == hipSuccess || cache_config_ret == hipErrorNotSupported));

  const auto shared_mem_config_ret = hipDeviceSetSharedMemConfig(
      mem_config_before == hipSharedMemBankSizeFourByte ? hipSharedMemBankSizeEightByte
                                                        : hipSharedMemBankSizeFourByte);
  REQUIRE((shared_mem_config_ret == hipSuccess || shared_mem_config_ret == hipErrorNotSupported));


  HIP_CHECK(hipSetDeviceFlags(hipDeviceScheduleBlockingSync));

  std::thread([] {
    HIP_CHECK_THREAD(hipSetDevice(0));
    HIP_CHECK_THREAD(hipDeviceReset());
  }).join();
  HIP_CHECK_THREAD_FINALIZE();

  unsigned int flags_after = 0u;
  CHECK(hipGetDeviceFlags(&flags_after) == hipSuccess);
  CHECK(flags_after == flags_before);

#if !defined(ENABLE_ADDRESS_SANITIZER)
  CHECK(hipFree(ptr) == hipErrorInvalidValue);
#endif

  if (cache_config_ret == hipSuccess) {
    hipFuncCache_t cache_config;
    CHECK(hipDeviceGetCacheConfig(&cache_config) == hipSuccess);
    CHECK(cache_config == hipFuncCachePreferNone);
  }

  if (shared_mem_config_ret == hipSuccess) {
    hipSharedMemConfig mem_config_after;
    CHECK(hipDeviceGetSharedMemConfig(&mem_config_after) == hipSuccess);
    CHECK(mem_config_after == mem_config_before);
  }
}

/**
 * End doxygen group DeviceTest.
 * @}
 */

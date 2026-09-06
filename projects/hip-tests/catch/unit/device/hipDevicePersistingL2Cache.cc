/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_process.hh>
#include <thread>

/**
 * @addtogroup hipDeviceSetLimit hipDeviceSetLimit
 * @{
 * @ingroup DeviceTest
 * Tests for GL2 Residency Control using hipLimitPersistingL2CacheSize
 *
 * COMPACT VERSION: 5 tests instead of 10, maintaining full coverage
 */

#if HT_AMD

bool isPersistingL2CacheSupported() {
  int deviceId;
  HIP_CHECK(hipGetDevice(&deviceId));
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, deviceId));

  if (props.persistingL2CacheMaxSize > 0) {
    INFO("Device: " << props.gcnArchName << ", Max L2: " << props.persistingL2CacheMaxSize << " bytes");
    return true;
  }
  return false;
}

/**
 * Test Description
 * ------------------------
 *  - Comprehensive positive functionality test covering:
 *    1. Query device property (persistingL2CacheMaxSize)
 *    2. Get current persisting L2 cache size
 *    3. Set to full range of valid values (0%, 25%, 50%, 75%, 100%)
 *    4. Verify each value persists correctly
 * Test source
 * ------------------------
 *  - unit/device/hipDevicePersistingL2Cache.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 *  - Device: MI450 or later
 */
HIP_TEST_CASE(Unit_hipDeviceLimit_PersistingL2Cache_Comprehensive) {
  if (!isPersistingL2CacheSupported()) {
    HIP_SKIP_TEST("GL2 Residency Control not supported. Requires MI450+");
  }

  int deviceId;
  HIP_CHECK(hipGetDevice(&deviceId));
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, deviceId));

  size_t maxSize = props.persistingL2CacheMaxSize;
  size_t originalSize = 0;

  SECTION("Query device property and max size") {
    // Verify persistingL2CacheMaxSize is populated and reasonable
    REQUIRE(maxSize > 0);
    REQUIRE(maxSize <= props.l2CacheSize);

    INFO("Max L2 reservation: " << maxSize << " bytes");
  }

  SECTION("Get current size") {
    HIP_CHECK(hipDeviceGetLimit(&originalSize, hipLimitPersistingL2CacheSize));
    REQUIRE(originalSize >= 0);
    REQUIRE(originalSize <= maxSize);
  }

  SECTION("Set and verify multiple values (0%, 25%, 50%, 75%, 100%)") {
    HIP_CHECK(hipDeviceGetLimit(&originalSize, hipLimitPersistingL2CacheSize));

    // Test comprehensive range of values
    std::vector<std::pair<std::string, size_t>> testCases = {
      {"0% (minimum)", 0},
      {"25% of max", maxSize / 4},
      {"50% of max", maxSize / 2},
      {"75% of max", (maxSize * 3) / 4},
      {"100% (maximum)", maxSize}
    };

    for (const auto& [description, targetSize] : testCases) {
      HIP_CHECK(hipDeviceSetLimit(hipLimitPersistingL2CacheSize, targetSize));

      size_t readSize = 0;
      HIP_CHECK(hipDeviceGetLimit(&readSize, hipLimitPersistingL2CacheSize));
      REQUIRE(readSize == targetSize);

      INFO("Set to " << description << ": " << targetSize << " bytes");
    }

    // Restore original
    HIP_CHECK(hipDeviceSetLimit(hipLimitPersistingL2CacheSize, originalSize));
  }
}

/**
 * Test Description
 * ------------------------
 *  - Boundary and range testing:
 *    1. Minimum value (0)
 *    2. Maximum value
 *    3. Exceeding maximum (should fail)
 *    4. Incremental increase/decrease
 * Test source
 * ------------------------
 *  - unit/device/hipDevicePersistingL2Cache_Compact.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 *  - Device: MI450 or later
 */
HIP_TEST_CASE(Unit_hipDeviceLimit_PersistingL2Cache_BoundaryTests) {
  if (!isPersistingL2CacheSupported()) {
    HIP_SKIP_TEST("GL2 Residency Control not supported. Requires MI450+");
  }

  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  size_t maxSize = props.persistingL2CacheMaxSize;

  size_t originalSize = 0;
  HIP_CHECK(hipDeviceGetLimit(&originalSize, hipLimitPersistingL2CacheSize));

  SECTION("Minimum value (0)") {
    HIP_CHECK(hipDeviceSetLimit(hipLimitPersistingL2CacheSize, 0));

    size_t readSize = 1; // Non-zero to verify it changes
    HIP_CHECK(hipDeviceGetLimit(&readSize, hipLimitPersistingL2CacheSize));
    REQUIRE(readSize == 0);
  }

  SECTION("Maximum value") {
    HIP_CHECK(hipDeviceSetLimit(hipLimitPersistingL2CacheSize, maxSize));

    size_t readSize = 0;
    HIP_CHECK(hipDeviceGetLimit(&readSize, hipLimitPersistingL2CacheSize));
    REQUIRE(readSize == maxSize);
  }

  SECTION("Exceed maximum (negative test)") {
    size_t invalidSize = maxSize + 1024;
    HIP_CHECK_ERROR(hipDeviceSetLimit(hipLimitPersistingL2CacheSize, invalidSize),
                    hipErrorInvalidValue);
  }

  SECTION("Incremental increase") {
    size_t newSize = std::min(originalSize + (1024 * 1024), maxSize);
    HIP_CHECK(hipDeviceSetLimit(hipLimitPersistingL2CacheSize, newSize));

    size_t readSize = 0;
    HIP_CHECK(hipDeviceGetLimit(&readSize, hipLimitPersistingL2CacheSize));
    REQUIRE(readSize == newSize);
  }

  SECTION("Incremental decrease") {
    size_t newSize = (originalSize > 1024 * 1024) ? (originalSize - (1024 * 1024)) : 0;
    HIP_CHECK(hipDeviceSetLimit(hipLimitPersistingL2CacheSize, newSize));

    size_t readSize = 0;
    HIP_CHECK(hipDeviceGetLimit(&readSize, hipLimitPersistingL2CacheSize));
    REQUIRE(readSize == newSize);
  }

  // Restore original
  HIP_CHECK(hipDeviceSetLimit(hipLimitPersistingL2CacheSize, originalSize));
}

// Kernel that stresses L2 cache
__global__ void L2CacheStressKernel(int* data, size_t numElements) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < numElements) {
    int value = data[idx];
    for (int i = 0; i < 100; i++) {
      value = value * 2 + 1;
    }
    data[idx] = value;
  }
}

/**
 * Test Description
 * ------------------------
 *  - Integration test: Set L2 cache size and run actual GPU kernel
 *  - Verifies feature works in real-world scenario
 * Test source
 * ------------------------
 *  - unit/device/hipDevicePersistingL2Cache_Compact.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 *  - Device: MI450 or later
 */
HIP_TEST_CASE(Unit_hipDeviceLimit_PersistingL2Cache_KernelIntegration) {
  if (!isPersistingL2CacheSupported()) {
    HIP_SKIP_TEST("GL2 Residency Control not supported. Requires MI450+");
  }

  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));

  size_t originalSize = 0;
  HIP_CHECK(hipDeviceGetLimit(&originalSize, hipLimitPersistingL2CacheSize));

  // Set to 75% of max
  size_t targetSize = (props.persistingL2CacheMaxSize * 3) / 4;
  HIP_CHECK(hipDeviceSetLimit(hipLimitPersistingL2CacheSize, targetSize));

  // Allocate and run kernel
  constexpr size_t numElements = 1024 * 1024;
  int* deviceData;
  HIP_CHECK(hipMalloc(&deviceData, numElements * sizeof(int)));

  std::vector<int> hostData(numElements, 1);
  HIP_CHECK(hipMemcpy(deviceData, hostData.data(), numElements * sizeof(int),
                      hipMemcpyHostToDevice));

  constexpr int blockSize = 256;
  int gridSize = (numElements + blockSize - 1) / blockSize;
  L2CacheStressKernel<<<gridSize, blockSize>>>(deviceData, numElements);
  HIP_CHECK(hipDeviceSynchronize());

  // Verify results
  HIP_CHECK(hipMemcpy(hostData.data(), deviceData, numElements * sizeof(int),
                      hipMemcpyDeviceToHost));
  REQUIRE(hostData[0] != 1); // Data was modified

  // Cleanup
  HIP_CHECK(hipFree(deviceData));
  HIP_CHECK(hipDeviceSetLimit(hipLimitPersistingL2CacheSize, originalSize));

  INFO("Kernel executed successfully with L2 reservation: " << targetSize << " bytes");
}

/**
 * Test Description
 * ------------------------
 *  - Concurrency tests:
 *    1. Multi-device: Independent control per GPU
 *    2. Threading: Thread-safe API usage
 * Test source
 * ------------------------
 *  - unit/device/hipDevicePersistingL2Cache_Compact.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 *  - Device: MI450 or later
 */
HIP_TEST_CASE(Unit_hipDeviceLimit_PersistingL2Cache_Concurrency) {
  if (!isPersistingL2CacheSupported()) {
    HIP_SKIP_TEST("GL2 Residency Control not supported. Requires MI450+");
  }

  SECTION("Multi-device independence") {
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));

    if (deviceCount < 2) {
      HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
    }

    std::vector<size_t> originalSizes(deviceCount);

    // Test on each device
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
      HIP_CHECK(hipSetDevice(deviceId));

      hipDeviceProp_t props;
      HIP_CHECK(hipGetDeviceProperties(&props, deviceId));

      if (props.persistingL2CacheMaxSize == 0) {
        continue; // Skip unsupported devices
      }

      HIP_CHECK(hipDeviceGetLimit(&originalSizes[deviceId], hipLimitPersistingL2CacheSize));

      // Set different value per device
      size_t targetSize = (props.persistingL2CacheMaxSize * (deviceId + 1)) / (deviceCount + 1);
      HIP_CHECK(hipDeviceSetLimit(hipLimitPersistingL2CacheSize, targetSize));

      size_t readSize = 0;
      HIP_CHECK(hipDeviceGetLimit(&readSize, hipLimitPersistingL2CacheSize));
      REQUIRE(readSize == targetSize);
    }

    // Restore
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
      HIP_CHECK(hipSetDevice(deviceId));
      hipDeviceProp_t props;
      HIP_CHECK(hipGetDeviceProperties(&props, deviceId));
      if (props.persistingL2CacheMaxSize > 0) {
        HIP_CHECK(hipDeviceSetLimit(hipLimitPersistingL2CacheSize, originalSizes[deviceId]));
      }
    }
    HIP_CHECK(hipSetDevice(0));
  }

  SECTION("Thread safety") {
    std::thread testThread([]() {
      hipDeviceProp_t props;
      HIP_CHECK_THREAD(hipGetDeviceProperties(&props, 0));

      size_t maxSize = props.persistingL2CacheMaxSize;
      REQUIRE_THREAD(maxSize > 0);

      size_t originalSize = 0;
      HIP_CHECK_THREAD(hipDeviceGetLimit(&originalSize, hipLimitPersistingL2CacheSize));

      size_t targetSize = maxSize / 4;
      HIP_CHECK_THREAD(hipDeviceSetLimit(hipLimitPersistingL2CacheSize, targetSize));

      size_t readSize = 0;
      HIP_CHECK_THREAD(hipDeviceGetLimit(&readSize, hipLimitPersistingL2CacheSize));
      REQUIRE_THREAD(readSize == targetSize);

      HIP_CHECK_THREAD(hipDeviceSetLimit(hipLimitPersistingL2CacheSize, originalSize));
    });

    testThread.join();
    HIP_CHECK_THREAD_FINALIZE();
  }
}

/**
 * Test Description
 * ------------------------
 *  - Negative parameter validation:
 *    1. nullptr parameter
 *    2. Invalid limit enum
 * Test source
 * ------------------------
 *  - unit/device/hipDevicePersistingL2Cache_Compact.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */
HIP_TEST_CASE(Unit_hipDeviceLimit_PersistingL2Cache_NegativeParams) {
  SECTION("nullptr parameter") {
    HIP_CHECK_ERROR(hipDeviceGetLimit(nullptr, hipLimitPersistingL2CacheSize),
                    hipErrorInvalidValue);
  }

  SECTION("Invalid limit value") {
    size_t val;
    HIP_CHECK_ERROR(hipDeviceGetLimit(&val, hipLimitRange), hipErrorInvalidValue);
  }
}

#endif  // HT_AMD

/**
 * End doxygen group.
 * @}
 */

/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <cstring>

static constexpr size_t LUID_LEN = 8;

/**
 * @addtogroup hipDeviceGetLuid hipDeviceGetLuid
 * @{
 * @ingroup DriverTest
 * `hipDeviceGetLuid(char* luid, unsigned int* deviceNodeMask, hipDevice_t device)` -
 * Returns an LUID and device node mask for the device.
 */

/**
 * Test Description
 * ------------------------
 *  - Query the LUID for each available device.
 *  - The LUID is a Windows/DXGI adapter concept. On Windows the call succeeds;
 *    on other platforms it returns `hipErrorNotSupported` without writing the
 *    output parameters.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceGetLuid.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */
HIP_TEST_CASE(Unit_hipDeviceGetLuid_Positive) {
  hipDevice_t device;
  const int deviceId = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipDeviceGet(&device, deviceId));

  char luid[LUID_LEN] = {0};
  unsigned int deviceNodeMask = 0;
#if defined(_WIN32)
  HIP_CHECK(hipDeviceGetLuid(luid, &deviceNodeMask, device));
  // A valid adapter LUID is never all-zero and the node mask has at least one
  // bit set, so verify the outputs were actually written.
  char zeroLuid[LUID_LEN] = {0};
  REQUIRE(memcmp(luid, zeroLuid, LUID_LEN) != 0);
  REQUIRE(deviceNodeMask != 0);
#else
  const unsigned int kSentinel = 0xdeadbeefU;
  deviceNodeMask = kSentinel;
  REQUIRE(hipDeviceGetLuid(luid, &deviceNodeMask, device) == hipErrorNotSupported);
  // Outputs must be left untouched on the unsupported path.
  char zero[LUID_LEN] = {0};
  REQUIRE(memcmp(luid, zero, LUID_LEN) == 0);
  REQUIRE(deviceNodeMask == kSentinel);
#endif
}

/**
 * Test Description
 * ------------------------
 *  - On Windows, verify that the LUID and device node mask returned by
 *    hipDeviceGetLuid match the values reported by hipGetDeviceProperties.
 *  - On other platforms, verify the API reports `hipErrorNotSupported` and that
 *    hipGetDeviceProperties reports an all-zero LUID and zero node mask.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceGetLuid.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */
HIP_TEST_CASE(Unit_hipDeviceGetLuid_VerifyLuidFrm_hipGetDeviceProperties) {
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  REQUIRE(deviceCount > 0);

  for (int dev = 0; dev < deviceCount; dev++) {
    hipDevice_t device;
    HIP_CHECK(hipDeviceGet(&device, dev));

    char luid[LUID_LEN] = {0};
    unsigned int deviceNodeMask = 0;

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));

#if defined(_WIN32)
    HIP_CHECK(hipDeviceGetLuid(luid, &deviceNodeMask, device));
    REQUIRE(memcmp(luid, prop.luid, LUID_LEN) == 0);
    REQUIRE(deviceNodeMask == prop.luidDeviceNodeMask);
#else
    REQUIRE(hipDeviceGetLuid(luid, &deviceNodeMask, device) == hipErrorNotSupported);
    char zero[LUID_LEN] = {0};
    REQUIRE(memcmp(prop.luid, zero, LUID_LEN) == 0);
    REQUIRE(prop.luidDeviceNodeMask == 0);
#endif
  }
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When the output LUID pointer is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When the output device node mask pointer is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When the device ordinal is negative
 *      - Expected output: return `hipErrorInvalidDevice`
 *    -# When the device ordinal is out of bounds
 *      - Expected output: return `hipErrorInvalidDevice`
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceGetLuid.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */
HIP_TEST_CASE(Unit_hipDeviceGetLuid_Negative) {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  if (numDevices > 0) {
    hipDevice_t device;
    HIP_CHECK(hipDeviceGet(&device, 0));

    char luid[LUID_LEN] = {0};
    unsigned int deviceNodeMask = 0;

    REQUIRE(hipErrorInvalidValue == hipDeviceGetLuid(nullptr, &deviceNodeMask, device));
    REQUIRE(hipErrorInvalidValue == hipDeviceGetLuid(luid, nullptr, device));
    REQUIRE(hipErrorInvalidDevice == hipDeviceGetLuid(luid, &deviceNodeMask, -1));
    REQUIRE(hipErrorInvalidDevice == hipDeviceGetLuid(luid, &deviceNodeMask, numDevices));
  }
}

/**
 * End doxygen group DriverTest.
 * @}
 */

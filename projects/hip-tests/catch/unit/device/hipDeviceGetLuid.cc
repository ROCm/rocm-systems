/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <cstring>

static constexpr size_t LUID_LEN = 8;
static constexpr unsigned int kSentinel = 0xdeadbeefU;

// Only Windows always has an adapter; elsewhere it depends on the backend (WSL has one).
#if defined(_WIN32)
static constexpr bool kLuidAlwaysSupported = true;
#else
static constexpr bool kLuidAlwaysSupported = false;
#endif

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
 *  - Where an adapter backs the device (Windows, or Linux under WSL) the call
 *    succeeds and writes a non-zero LUID and node mask; otherwise it returns
 *    `hipErrorNotSupported` without touching the output parameters.
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
  unsigned int deviceNodeMask = kSentinel;
  const hipError_t err = hipDeviceGetLuid(luid, &deviceNodeMask, device);

  char zeroLuid[LUID_LEN] = {0};
  if (err == hipSuccess) {
    // A valid adapter LUID is never all-zero and the node mask has at least one
    // bit set, so verify the outputs were actually written.
    REQUIRE(memcmp(luid, zeroLuid, LUID_LEN) != 0);
    REQUIRE(deviceNodeMask != 0);
  } else {
    // Windows must never land here.
    REQUIRE_FALSE(kLuidAlwaysSupported);
    REQUIRE(err == hipErrorNotSupported);
    // Outputs must be left untouched on the unsupported path.
    REQUIRE(memcmp(luid, zeroLuid, LUID_LEN) == 0);
    REQUIRE(deviceNodeMask == kSentinel);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Verify hipDeviceGetLuid and hipGetDeviceProperties agree for every device:
 *    on success the LUID and node mask must match; on `hipErrorNotSupported`
 *    the properties must report an all-zero LUID and zero node mask.
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

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));

    char luid[LUID_LEN] = {0};
    unsigned int deviceNodeMask = kSentinel;
    const hipError_t err = hipDeviceGetLuid(luid, &deviceNodeMask, device);

    if (err == hipSuccess) {
      REQUIRE(memcmp(luid, prop.luid, LUID_LEN) == 0);
      REQUIRE(deviceNodeMask == prop.luidDeviceNodeMask);
    } else {
      // Windows must never land here.
      REQUIRE_FALSE(kLuidAlwaysSupported);
      REQUIRE(err == hipErrorNotSupported);
      // Properties must not report a LUID either.
      char zero[LUID_LEN] = {0};
      REQUIRE(memcmp(prop.luid, zero, LUID_LEN) == 0);
      REQUIRE(prop.luidDeviceNodeMask == 0);
    }
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

/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cctype>
#include <string>

#include <hip_test_common.hh>

namespace {
// The gfx major version of a device, parsed from gcnArchName. Returns 0 for a name that
// does not parse, so an unrecognised device carries no expectation.
int GfxMajor(const char* gcnArchName) {
  const std::string arch(gcnArchName);

  const auto start = arch.find("gfx");
  if (start == std::string::npos) {
    return 0;
  }

  auto end = start + 3;
  while (end < arch.size() && std::isdigit(static_cast<unsigned char>(arch[end]))) {
    ++end;
  }

  const std::string digits = arch.substr(start + 3, end - (start + 3));
  if (digits.empty()) {
    return 0;
  }

  // A generic target spells the version out: "gfx9-4-generic", "gfx12-5-generic".
  if (end < arch.size() && arch[end] == '-') {
    return std::stoi(digits);
  }

  // Otherwise it is packed, the stepping already dropped: "gfx90a" and "gfx950" are 9.
  return digits.size() >= 4 ? std::stoi(digits.substr(0, 2)) : digits[0] - '0';
}
}  // namespace

/**
 * @addtogroup hipDeviceGetExecAffinitySupport hipDeviceGetExecAffinitySupport
 * @{
 * @ingroup DriverTest
 * `hipDeviceGetExecAffinitySupport(int* pi, hipExecAffinityType type, hipDevice_t dev)` -
 * Reports whether a given execution-affinity type is supported on a device.
 *  - `hipExecAffinityTypeCUCount` is the CU-count affinity equivalent of CUDA's
 *    `CU_EXEC_AFFINITY_TYPE_SM_COUNT`; CU masking is available on all AMD GPUs so this is
 *    always supported.
 *  - `hipExtExecAffinityTypeGranularityCU` / `hipExtExecAffinityTypeGranularityWGP` report the
 *    granularity the runtime will actually apply to a CU mask, so exactly one is supported
 *    per device. It is an effective runtime property, not a fixed hardware one: per-WGP
 *    (2-CU pairs) on RDNA gfx10-gfx12.4 unless `GPU_ENABLE_WGP_MODE=0` is set
 */

/**
 * Test Description
 * ------------------------
 *  - Query every supported affinity type on each available device and verify:
 *    -# `hipExecAffinityTypeCUCount` is always supported (`*pi == 1`).
 *    -# Each granularity query returns a boolean (0 or 1).
 *    -# The two granularity types are mutually exclusive: exactly one of
 *       `GranularityCU` / `GranularityWGP` is supported on any given device.
 *    -# The supported one is `GranularityCU` on pre-gfx10 devices, which never enable WGP
 *       mode. On gfx10+ it is configurable, so only mutual exclusion is checked there.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceGetExecAffinitySupport.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */
HIP_TEST_CASE(Unit_hipDeviceGetExecAffinitySupport_Positive) {
  const int deviceId = GENERATE(range(0, HipTest::getDeviceCount()));
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, deviceId));

  // CU-count affinity is the SM_COUNT equivalent and is supported on every AMD GPU.
  int cuCountSupported = -1;
  HIP_CHECK(hipDeviceGetExecAffinitySupport(&cuCountSupported, hipExecAffinityTypeCUCount, device));
  REQUIRE(cuCountSupported == 1);

  // Granularity queries are boolean.
  int cuGranularity = -1;
  int wgpGranularity = -1;
  HIP_CHECK(
      hipDeviceGetExecAffinitySupport(&cuGranularity, hipExtExecAffinityTypeGranularityCU, device));
  HIP_CHECK(hipDeviceGetExecAffinitySupport(&wgpGranularity, hipExtExecAffinityTypeGranularityWGP,
                                            device));
  REQUIRE((cuGranularity == 0 || cuGranularity == 1));
  REQUIRE((wgpGranularity == 0 || wgpGranularity == 1));

  // A device masks either per-CU or per-WGP, never both and never neither.
  REQUIRE((cuGranularity + wgpGranularity) == 1);

  // That sum is symmetric, so pin which one where it is not configurable: WGP mode is only
  // ever enabled for gfx10+, so anything older masks per-CU.
  hipDeviceProp_t props{};
  HIP_CHECK(hipGetDeviceProperties(&props, deviceId));
  const int gfxMajor = GfxMajor(props.gcnArchName);
  if (gfxMajor > 0 && gfxMajor < 10) {
    REQUIRE(cuGranularity == 1);
    REQUIRE(wgpGranularity == 0);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When the output pointer is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When the affinity type is the `hipExecAffinityTypeMax` sentinel
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When the affinity type is out of range
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When the device ordinal is negative
 *      - Expected output: return `hipErrorInvalidDevice`
 *    -# When the device ordinal is out of bounds
 *      - Expected output: return `hipErrorInvalidDevice`
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceGetExecAffinitySupport.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */
HIP_TEST_CASE(Unit_hipDeviceGetExecAffinitySupport_Negative) {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  REQUIRE(numDevices > 0);

  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  int support = 0;

  // Null output pointer.
  HIP_CHECK_ERROR(hipDeviceGetExecAffinitySupport(nullptr, hipExecAffinityTypeCUCount, device),
                  hipErrorInvalidValue);

  // Sentinel and out-of-range affinity types.
  HIP_CHECK_ERROR(hipDeviceGetExecAffinitySupport(&support, hipExecAffinityTypeMax, device),
                  hipErrorInvalidValue);
  HIP_CHECK_ERROR(
      hipDeviceGetExecAffinitySupport(
          &support, static_cast<hipExecAffinityType>(hipExecAffinityTypeMax + 1), device),
      hipErrorInvalidValue);

  // Invalid device ordinals.
  HIP_CHECK_ERROR(hipDeviceGetExecAffinitySupport(&support, hipExecAffinityTypeCUCount, -1),
                  hipErrorInvalidDevice);
  HIP_CHECK_ERROR(hipDeviceGetExecAffinitySupport(&support, hipExecAffinityTypeCUCount, numDevices),
                  hipErrorInvalidDevice);
}

/**
 * End doxygen group DriverTest.
 * @}
 */

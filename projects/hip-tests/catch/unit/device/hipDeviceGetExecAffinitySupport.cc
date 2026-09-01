/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

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
 *    device's CU-mask granularity. Exactly one is supported per device: per-CU on GCN/CDNA
 *    (gfx9) and gfx12.5+, per-WGP (2-CU pairs) on RDNA gfx10-gfx12.4.
 */

/**
 * Test Description
 * ------------------------
 *  - Query every supported affinity type on each available device and verify:
 *    -# `hipExecAffinityTypeCUCount` is always supported (`*pi == 1`).
 *    -# Each granularity query returns a boolean (0 or 1).
 *    -# The two granularity types are mutually exclusive: exactly one of
 *       `GranularityCU` / `GranularityWGP` is supported on any given device.
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

  // CU-count affinity is the SM_COUNT equivalent.
  int cuCountSupported = -1;
  HIP_CHECK(hipDeviceGetExecAffinitySupport(&cuCountSupported, hipExecAffinityTypeCUCount, device));
#if HT_AMD
  // CU masking is available on every AMD GPU, so this is always supported.
  REQUIRE(cuCountSupported == 1);
#else
  // On NVIDIA, CU_EXEC_AFFINITY_TYPE_SM_COUNT is only supported on Volta+ under MPS, so the
  // result is device-dependent; only require a valid boolean.
  REQUIRE((cuCountSupported == 0 || cuCountSupported == 1));
#endif

  // Granularity queries are boolean.
  int cuGranularity = -1;
  int wgpGranularity = -1;
  HIP_CHECK(
      hipDeviceGetExecAffinitySupport(&cuGranularity, hipExtExecAffinityTypeGranularityCU, device));
  HIP_CHECK(hipDeviceGetExecAffinitySupport(&wgpGranularity, hipExtExecAffinityTypeGranularityWGP,
                                            device));
  REQUIRE((cuGranularity == 0 || cuGranularity == 1));
  REQUIRE((wgpGranularity == 0 || wgpGranularity == 1));

#if HT_AMD
  // A device masks either per-CU or per-WGP, never both and never neither.
  REQUIRE((cuGranularity + wgpGranularity) == 1);
#else
  // CU-mask granularity is a ROCm-specific concept with no CUDA equivalent; both report
  // unsupported on the NVIDIA backend.
  REQUIRE(cuGranularity == 0);
  REQUIRE(wgpGranularity == 0);
#endif
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

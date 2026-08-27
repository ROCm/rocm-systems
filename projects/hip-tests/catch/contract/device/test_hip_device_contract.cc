/*
 * Copyright Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstring>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

namespace {
int CurrentDevice() {
  int device = -1;
  HIP_CHECK(hipGetDevice(&device));
  return device;
}

hipDeviceProp_t CurrentDeviceProperties() {
  hipDeviceProp_t properties{};
  HIP_CHECK(hipGetDeviceProperties(&properties, CurrentDevice()));
  return properties;
}
}

// @asserts: hipGetDeviceProperties - succeeds in populating properties for the current device
HIP_TEST_CASE(Contract_Device_HipGetDeviceProperties_GetProperties_SucceedsForCurrentDevice) {
  hipDeviceProp_t properties{};

  HIP_CHECK(hipGetDeviceProperties(&properties, CurrentDevice()));
}

// @asserts: hipGetDeviceProperties - the device name string is non-empty
HIP_TEST_CASE(Contract_Device_HipGetDeviceProperties_Name_IsNonEmpty) {
  const auto properties = CurrentDeviceProperties();

  REQUIRE(std::strlen(properties.name) > 0);
}

// @asserts: hipGetDeviceProperties - reported total global memory is positive
HIP_TEST_CASE(Contract_Device_HipGetDeviceProperties_TotalGlobalMem_IsPositive) {
  const auto properties = CurrentDeviceProperties();

  REQUIRE(properties.totalGlobalMem > 0);
}

// @asserts: hipGetDeviceProperties - reported multiprocessor count is positive
HIP_TEST_CASE(Contract_Device_HipGetDeviceProperties_MultiProcessorCount_IsPositive) {
  const auto properties = CurrentDeviceProperties();

  REQUIRE(properties.multiProcessorCount > 0);
}

// @asserts: hipGetDeviceProperties - reported warp size is positive
HIP_TEST_CASE(Contract_Device_HipGetDeviceProperties_WarpSize_IsPositive) {
  const auto properties = CurrentDeviceProperties();

  REQUIRE(properties.warpSize > 0);
}

// @asserts: hipDeviceGetAttribute - hipDeviceAttributeWarpSize matches the warp size from hipGetDeviceProperties
HIP_TEST_CASE(Contract_Device_HipDeviceGetAttribute_WarpSize_MatchesProperties) {
  const auto properties = CurrentDeviceProperties();
  int attribute_warp_size = 0;

  HIP_CHECK(hipDeviceGetAttribute(&attribute_warp_size, hipDeviceAttributeWarpSize, CurrentDevice()));

  REQUIRE(attribute_warp_size == properties.warpSize);
}

// @asserts: hipGetDevice - the current device ordinal is in range [0, device_count)
HIP_TEST_CASE(Contract_Device_HipGetDevice_CurrentOrdinal_IsWithinDeviceCount) {
  int device_count = 0;
  const int current_device = CurrentDevice();

  HIP_CHECK(hipGetDeviceCount(&device_count));

  REQUIRE(device_count > 0);
  REQUIRE(current_device >= 0);
  REQUIRE(current_device < device_count);
}

// @asserts: hipDeviceGetExecAffinitySupport - the CU-count affinity query succeeds and reports a boolean
HIP_TEST_CASE(Contract_Device_HipDeviceGetExecAffinitySupport_CuCountType_ReportsBoolean) {
  hipDevice_t device{};
  HIP_CHECK(hipDeviceGet(&device, CurrentDevice()));

  int supported = -1;
  HIP_CHECK(hipDeviceGetExecAffinitySupport(&supported, hipExecAffinityTypeCUCount, device));

  REQUIRE((supported == 0 || supported == 1));
#if HT_AMD
  // BACKEND-DIFF: CU masking is available on every AMD GPU, so CU-count affinity is always
  // supported. On NVIDIA, CU_EXEC_AFFINITY_TYPE_SM_COUNT is only supported on Volta+ under MPS,
  // so the result is device-dependent and only the boolean invariant is portable. Parity would
  // require an MPS-enabled NVIDIA device in CI.
  REQUIRE(supported == 1);
#endif
}

// @asserts: hipDeviceGetExecAffinitySupport - each CU-mask granularity query reports a boolean
HIP_TEST_CASE(Contract_Device_HipDeviceGetExecAffinitySupport_GranularityTypes_ReportBoolean) {
  hipDevice_t device{};
  HIP_CHECK(hipDeviceGet(&device, CurrentDevice()));

  int cu_granularity = -1;
  int wgp_granularity = -1;
  HIP_CHECK(
      hipDeviceGetExecAffinitySupport(&cu_granularity, hipExtExecAffinityTypeGranularityCU, device));
  HIP_CHECK(hipDeviceGetExecAffinitySupport(&wgp_granularity, hipExtExecAffinityTypeGranularityWGP,
                                            device));

  REQUIRE((cu_granularity == 0 || cu_granularity == 1));
  REQUIRE((wgp_granularity == 0 || wgp_granularity == 1));
#if HT_AMD
  // BACKEND-DIFF: the CU-mask granularity is a ROCm-specific property; exactly one of per-CU
  // (gfx9 / gfx12.5+) or per-WGP (RDNA gfx10-12.4) holds on any AMD device. NVIDIA has no
  // equivalent and reports both types as unsupported, so this mutual-exclusivity invariant is
  // asserted AMD-only.
  REQUIRE((cu_granularity + wgp_granularity) == 1);
#endif
}

// @asserts: hipDeviceGetExecAffinitySupport - a null output pointer is rejected with hipErrorInvalidValue
HIP_TEST_CASE(Contract_Device_HipDeviceGetExecAffinitySupport_NullOutput_ReturnsInvalidValue) {
  hipDevice_t device{};
  HIP_CHECK(hipDeviceGet(&device, CurrentDevice()));

  const hipError_t status =
      hipDeviceGetExecAffinitySupport(nullptr, hipExecAffinityTypeCUCount, device);

  REQUIRE(status == hipErrorInvalidValue);
  (void)hipGetLastError();
}

/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipMemGetAllocationGranularity hipMemGetAllocationGranularity
 * @{
 * @ingroup VirtualMemoryManagementTest
 * `hipError_t hipMemGetAllocationGranularity (size_t* granularity,
 *                                             const hipMemAllocationProp* prop,
 *                                             hipMemAllocationGranularity_flags option)` -
 * Calculates either the minimal or recommended granularity.
 */

#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>
#include <hip_test_common.hh>

#include "hip_vmm_common.hh"

TEST_CASE("Unit_hipMemGetAllocationGranularity_RDNA4", "[vmm]") {
  HIP_CHECK(hipFree(0));
  int deviceId = 0;
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, deviceId));
  checkVMMSupported(device);

  size_t min_granularity = 0;
  size_t recommended_granularity = 0;

  hipMemAllocationProp allocProp{};
  allocProp.type = hipMemAllocationTypePinned;
  allocProp.location.type = hipMemLocationTypeDevice;
  allocProp.location.id = deviceId;

  HIP_CHECK(hipMemGetAllocationGranularity(&min_granularity, &allocProp,
                                          hipMemAllocationGranularityMinimum));
  HIP_CHECK(hipMemGetAllocationGranularity(&recommended_granularity, &allocProp,
                                          hipMemAllocationGranularityRecommended));

  REQUIRE(min_granularity > 0);
  REQUIRE(recommended_granularity >= min_granularity);

  // Power-of-two check
  REQUIRE((min_granularity & (min_granularity - 1)) == 0);

#if HT_AMD
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, deviceId));

  // Architecture-specific check for RDNA4 (gfx1201)
  if (std::string(prop.gcnArchName).find("gfx1201") != std::string::npos) {
    constexpr size_t k2MB = 2 * 1024 * 1024;
    REQUIRE(min_granularity >= k2MB);
    REQUIRE(recommended_granularity >= k2MB);
    REQUIRE(min_granularity % k2MB == 0);
  }
#endif
}

/**
 * End doxygen group VirtualMemoryManagementTest.
 * @}
 */

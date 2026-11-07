/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipLocalityDomain hipLocalityDomain
 * @{
 * @ingroup LocalityDomainTest
 * Device attribute queries for locality domains:
 * `hipDeviceAttributeLocalityDomainCount` and
 * `hipDeviceAttributeLocalityDomainMultiprocessorCount`.
 */

#include "hip_locality_domain_common.hh"

/**
 * Test Description
 * ------------------------
 *  - Queries both locality domain attributes on every visible device and checks
 *    the basic invariants: the count is always at least 1, and each domain owns
 *    at least one compute unit.
 * Test requirements
 * ------------------------
 *  - Runtime implements the locality domain device attributes
 */
HIP_TEST_CASE(Unit_hipLocalityDomain_Attributes_Basic) {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  REQUIRE(device_count > 0);

  int probe = 0;
  if (!LocalityDomain::GetDomainCount(0, &probe)) {
    HIP_SKIP_TEST(LocalityDomain::kUnsupported);
    return;
  }

  for (int dev = 0; dev < device_count; ++dev) {
    int domains = 0;
    int domain_cus = 0;
    INFO("device " << dev);
    REQUIRE(LocalityDomain::GetDomainCount(dev, &domains));
    REQUIRE(LocalityDomain::GetDomainCuCount(dev, &domain_cus));

    // A device always has at least one locality domain; a device with no
    // exploitable NUMA structure reports exactly 1.
    REQUIRE(domains >= 1);
    REQUIRE(domain_cus >= 1);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Checks the relationship between per-domain compute units, the domain count
 *    and the device-wide compute unit count. Equality is only asserted for a
 *    single-domain device: on multi-domain parts some compute units may not be
 *    attributed to any domain, so only the upper bound is guaranteed.
 * Test requirements
 * ------------------------
 *  - Runtime implements the locality domain device attributes
 */
HIP_TEST_CASE(Unit_hipLocalityDomain_Attributes_Consistency) {
  int domains = 0;
  if (!LocalityDomain::GetDomainCount(0, &domains)) {
    HIP_SKIP_TEST(LocalityDomain::kUnsupported);
    return;
  }

  int domain_cus = 0;
  REQUIRE(LocalityDomain::GetDomainCuCount(0, &domain_cus));

  int total_cus = 0;
  HIP_CHECK(hipDeviceGetAttribute(&total_cus, hipDeviceAttributeMultiprocessorCount, 0));

  INFO("domains=" << domains << " cusPerDomain=" << domain_cus << " totalCus=" << total_cus);
  REQUIRE(domain_cus <= total_cus);

  if (domains == 1) {
    // The single domain owns the whole device.
    REQUIRE(domain_cus == total_cus);
  }

  // The reported SM resource must agree with the device attribute.
  hipDevResource resource = LocalityDomain::DeviceSmResource(0);
  REQUIRE(resource.sm.smCount == static_cast<unsigned int>(total_cus));
}

/**
 * Test Description
 * ------------------------
 *  - Negative: the locality domain attributes must follow the standard device
 *    attribute error contract when handed an invalid device ordinal.
 * Test requirements
 * ------------------------
 *  - Runtime implements the locality domain device attributes
 */
HIP_TEST_CASE(Unit_hipLocalityDomain_Attributes_Negative) {
  int domains = 0;
  if (!LocalityDomain::GetDomainCount(0, &domains)) {
    HIP_SKIP_TEST(LocalityDomain::kUnsupported);
    return;
  }

  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  const int invalid_device = device_count + 32;
  int value = 0;

  SECTION("invalid device ordinal - domain count") {
    HIP_CHECK_ERROR(
        hipDeviceGetAttribute(&value, hipDeviceAttributeLocalityDomainCount, invalid_device),
        hipErrorInvalidDevice);
  }

  SECTION("invalid device ordinal - per-domain CU count") {
    HIP_CHECK_ERROR(hipDeviceGetAttribute(
                        &value, hipDeviceAttributeLocalityDomainMultiprocessorCount,
                        invalid_device),
                    hipErrorInvalidDevice);
  }

  SECTION("negative device ordinal") {
    HIP_CHECK_ERROR(hipDeviceGetAttribute(&value, hipDeviceAttributeLocalityDomainCount, -1),
                    hipErrorInvalidDevice);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Basic: the driver-style pointer attribute reports -1 for an allocation that
 *    is not localized to a locality domain, and the query itself succeeds. A -1
 *    result is data, not an error.
 * Test requirements
 * ------------------------
 *  - Runtime implements the locality domain device attributes
 */
HIP_TEST_CASE(Unit_hipLocalityDomain_PointerOrdinal_Basic) {
  int domains = 0;
  if (!LocalityDomain::GetDomainCount(0, &domains)) {
    HIP_SKIP_TEST(LocalityDomain::kUnsupported);
    return;
  }

  void* ptr = nullptr;
  HIP_CHECK(hipMalloc(&ptr, 1024));

  int ordinal = -99;
  HIP_CHECK(hipPointerGetAttribute(&ordinal, HIP_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL,
                                   reinterpret_cast<hipDeviceptr_t>(ptr)));
  INFO("ordinal for a plain device allocation");
  REQUIRE(ordinal == -1);

  HIP_CHECK(hipFree(ptr));
}

/**
 * End doxygen group hipLocalityDomain.
 * @}
 */

/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipLocalityDomain hipLocalityDomain
 * @{
 * @ingroup LocalityDomainTest
 * Memory pool locality domain placement: `hipMemLocationTypeDeviceLocalityDomain`,
 * `hipMemPoolAttrLocalityDomainId` and the APIs that must reject a localized
 * memory location.
 */

#include "hip_locality_domain_common.hh"

/**
 * Test Description
 * ------------------------
 *  - Basic: creates a memory pool backed by locality domain 0 and destroys it.
 * Test requirements
 * ------------------------
 *  - Runtime implements the locality domain device attributes
 */
HIP_TEST_CASE(Unit_hipLocalityDomain_MemPool_Basic) {
  int domains = 0;
  if (!LocalityDomain::GetDomainCount(0, &domains)) {
    HIP_SKIP_TEST(LocalityDomain::kUnsupported);
    return;
  }
  HIP_CHECK(hipSetDevice(0));

  hipMemPoolProps props = LocalityDomain::DomainPoolProps(0, 0);
  hipMemPool_t pool = nullptr;
  HIP_CHECK(hipMemPoolCreate(&pool, &props));
  REQUIRE(pool != nullptr);
  HIP_CHECK(hipMemPoolDestroy(pool));
}

/**
 * Test Description
 * ------------------------
 *  - Functional: allocates from a locality-domain pool on every available
 *    domain, writes to the memory from a kernel, reads it back and frees it.
 * Test requirements
 * ------------------------
 *  - Runtime implements the locality domain device attributes
 */
HIP_TEST_CASE(Unit_hipLocalityDomain_MemPool_Functional) {
  int domains = 0;
  if (!LocalityDomain::GetDomainCount(0, &domains)) {
    HIP_SKIP_TEST(LocalityDomain::kUnsupported);
    return;
  }
  HIP_CHECK(hipSetDevice(0));

  for (int domain = 0; domain < domains; ++domain) {
    INFO("locality domain " << domain);

    hipMemPoolProps props =
        LocalityDomain::DomainPoolProps(0, static_cast<unsigned char>(domain));
    hipMemPool_t pool = nullptr;
    HIP_CHECK(hipMemPoolCreate(&pool, &props));

    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    constexpr size_t kN = 4096;
    constexpr size_t kBytes = kN * sizeof(int);
    int* d_a = nullptr;
    int* d_b = nullptr;
    int* d_c = nullptr;
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&d_a), kBytes, pool, stream));
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&d_b), kBytes, pool, stream));
    HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&d_c), kBytes, pool, stream));

    std::vector<int> h_a(kN), h_b(kN), h_c(kN, 0);
    for (size_t i = 0; i < kN; ++i) {
      h_a[i] = static_cast<int>(i);
      h_b[i] = static_cast<int>(i * 3);
    }
    HIP_CHECK(hipMemcpyAsync(d_a, h_a.data(), kBytes, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_b, h_b.data(), kBytes, hipMemcpyHostToDevice, stream));

    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((kN + kThreads - 1) / kThreads);
    HipTest::vectorADD<<<blocks, kThreads, 0, stream>>>(d_a, d_b, d_c, kN);
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipMemcpyAsync(h_c.data(), d_c, kBytes, hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    for (size_t i = 0; i < kN; ++i) {
      REQUIRE(h_c[i] == h_a[i] + h_b[i]);
    }

    HIP_CHECK(hipFreeAsync(d_a, stream));
    HIP_CHECK(hipFreeAsync(d_b, stream));
    HIP_CHECK(hipFreeAsync(d_c, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipMemPoolDestroy(pool));
  }
}

/**
 * Test Description
 * ------------------------
 *  - Functional: on a device with a single locality domain, a pool created with
 *    hipMemLocationTypeDeviceLocalityDomain and ordinal 0 is equivalent to a
 *    full-device pool, so it reports hipMemLocationTypeDevice for the location
 *    type and -1 for the domain id. On a multi-domain device the pool must
 *    report the ordinal it was created with.
 * Test requirements
 * ------------------------
 *  - Runtime implements the locality domain device attributes
 */
HIP_TEST_CASE(Unit_hipLocalityDomain_MemPool_SingleDomainEquivalence) {
  int domains = 0;
  if (!LocalityDomain::GetDomainCount(0, &domains)) {
    HIP_SKIP_TEST(LocalityDomain::kUnsupported);
    return;
  }
  HIP_CHECK(hipSetDevice(0));

  SECTION("non-localized pool reports -1") {
    hipMemPoolProps props{};
    props.allocType = hipMemAllocationTypePinned;
    props.location.type = hipMemLocationTypeDevice;
    props.location.id = 0;
    hipMemPool_t pool = nullptr;
    HIP_CHECK(hipMemPoolCreate(&pool, &props));

    int domain_id = -99;
    int location_type = -99;
    HIP_CHECK(hipMemPoolGetAttribute(pool, hipMemPoolAttrLocalityDomainId, &domain_id));
    HIP_CHECK(hipMemPoolGetAttribute(pool, hipMemPoolAttrLocationType, &location_type));
    REQUIRE(domain_id == -1);
    REQUIRE(location_type == static_cast<int>(hipMemLocationTypeDevice));
    HIP_CHECK(hipMemPoolDestroy(pool));
  }

  SECTION("localized pool attribute readback") {
    hipMemPoolProps props = LocalityDomain::DomainPoolProps(0, 0);
    hipMemPool_t pool = nullptr;
    HIP_CHECK(hipMemPoolCreate(&pool, &props));

    int domain_id = -99;
    int location_type = -99;
    HIP_CHECK(hipMemPoolGetAttribute(pool, hipMemPoolAttrLocalityDomainId, &domain_id));
    HIP_CHECK(hipMemPoolGetAttribute(pool, hipMemPoolAttrLocationType, &location_type));
    INFO("domains=" << domains << " domainId=" << domain_id << " locType=" << location_type);

    if (domains == 1) {
      // Documented equivalence: the request is erased on a single-domain device.
      REQUIRE(domain_id == -1);
      REQUIRE(location_type == static_cast<int>(hipMemLocationTypeDevice));
    } else {
      REQUIRE(domain_id == 0);
      REQUIRE(location_type == static_cast<int>(hipMemLocationTypeDeviceLocalityDomain));
    }

    // An allocation out of the pool must agree with the pool's own report.
    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    void* ptr = nullptr;
    HIP_CHECK(hipMallocFromPoolAsync(&ptr, 1024, pool, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    int ordinal = -99;
    HIP_CHECK(hipPointerGetAttribute(&ordinal, HIP_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL,
                                     reinterpret_cast<hipDeviceptr_t>(ptr)));
    REQUIRE(ordinal == domain_id);

    HIP_CHECK(hipFreeAsync(ptr, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipMemPoolDestroy(pool));
  }
}

/**
 * Test Description
 * ------------------------
 *  - Negative: the memory path rejects an out-of-range locality domain ordinal
 *    and a managed allocation type with hipErrorNotSupported, but an invalid
 *    device ordinal with hipErrorInvalidValue. Device and domain are validated
 *    separately and do not share an error code.
 * Test requirements
 * ------------------------
 *  - Runtime implements the locality domain device attributes
 */
HIP_TEST_CASE(Unit_hipLocalityDomain_MemPool_Negative) {
  int domains = 0;
  if (!LocalityDomain::GetDomainCount(0, &domains)) {
    HIP_SKIP_TEST(LocalityDomain::kUnsupported);
    return;
  }
  HIP_CHECK(hipSetDevice(0));

  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));

  SECTION("ordinal == domain count") {
    hipMemPoolProps props =
        LocalityDomain::DomainPoolProps(0, static_cast<unsigned char>(domains));
    hipMemPool_t pool = nullptr;
    HIP_CHECK_ERRORS(hipMemPoolCreate(&pool, &props), hipErrorNotSupported, hipErrorInvalidValue);
  }

  SECTION("ordinal far out of range") {
    hipMemPoolProps props = LocalityDomain::DomainPoolProps(0, 255);
    hipMemPool_t pool = nullptr;
    HIP_CHECK_ERRORS(hipMemPoolCreate(&pool, &props), hipErrorNotSupported, hipErrorInvalidValue);
  }

  SECTION("managed allocation type cannot be localized") {
    hipMemPoolProps props =
        LocalityDomain::DomainPoolProps(0, 0, hipMemAllocationTypeManaged);
    hipMemPool_t pool = nullptr;
    HIP_CHECK_ERROR(hipMemPoolCreate(&pool, &props), hipErrorNotSupported);
  }

  SECTION("invalid device ordinal is a different error") {
    hipMemPoolProps props =
        LocalityDomain::DomainPoolProps(static_cast<unsigned char>(device_count + 8), 0);
    hipMemPool_t pool = nullptr;
    HIP_CHECK_ERROR(hipMemPoolCreate(&pool, &props), hipErrorInvalidValue);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Negative: locality domains are a placement concept, not an isolation
 *    boundary. APIs that take a memory location for access control or residency
 *    hints must reject a localized location.
 * Test requirements
 * ------------------------
 *  - Runtime implements the locality domain device attributes
 */
HIP_TEST_CASE(Unit_hipLocalityDomain_Location_Rejected_Negative) {
  int domains = 0;
  if (!LocalityDomain::GetDomainCount(0, &domains)) {
    HIP_SKIP_TEST(LocalityDomain::kUnsupported);
    return;
  }
  HIP_CHECK(hipSetDevice(0));

  const hipMemLocation domain_location = LocalityDomain::DomainLocation(0, 0);

  SECTION("hipMemSetAccess rejects a localized location") {
    // VMM access control is whole-device only: a locality domain describes where
    // memory is placed, not who may reach it, so the location type is invalid here.
    size_t granularity = 0;
    hipMemAllocationProp prop{};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = 0;
    HIP_CHECK(hipMemGetAllocationGranularity(&granularity, &prop,
                                             hipMemAllocationGranularityMinimum));
    REQUIRE(granularity > 0);

    void* va = nullptr;
    HIP_CHECK(hipMemAddressReserve(&va, granularity, 0, nullptr, 0));

    hipMemAccessDesc desc{};
    desc.location = domain_location;
    desc.flags = hipMemAccessFlagsProtReadWrite;
    HIP_CHECK_ERROR(hipMemSetAccess(va, granularity, &desc, 1), hipErrorInvalidValue);

    HIP_CHECK(hipMemAddressFree(va, granularity));
  }

  SECTION("hipMemPoolSetAccess accepts a localized location but validates the ordinal") {
    // Unlike the VMM path above, pool access control does accept a locality
    // location - it resolves to the device - but an out-of-range ordinal is
    // still rejected.
    hipMemPoolProps props{};
    props.allocType = hipMemAllocationTypePinned;
    props.location.type = hipMemLocationTypeDevice;
    props.location.id = 0;
    hipMemPool_t pool = nullptr;
    HIP_CHECK(hipMemPoolCreate(&pool, &props));

    hipMemAccessDesc desc{};
    desc.location = domain_location;
    desc.flags = hipMemAccessFlagsProtReadWrite;
    HIP_CHECK(hipMemPoolSetAccess(pool, &desc, 1));

    hipMemAccessDesc bad{};
    bad.location = LocalityDomain::DomainLocation(0, 255);
    bad.flags = hipMemAccessFlagsProtReadWrite;
    HIP_CHECK_ERRORS(hipMemPoolSetAccess(pool, &bad, 1), hipErrorNotSupported, hipErrorInvalidValue);

    HIP_CHECK(hipMemPoolDestroy(pool));
  }

  SECTION("hipMemPrefetchAsync_v2 rejects a localized location") {
    int managed_support = 0;
    HIP_CHECK(hipDeviceGetAttribute(&managed_support, hipDeviceAttributeManagedMemory, 0));
    if (!managed_support) {
      HIP_SKIP_TEST("Managed memory is not supported on this device");
      return;
    }
    void* ptr = nullptr;
    HIP_CHECK(hipMallocManaged(&ptr, 4096));
    HIP_CHECK_ERROR(hipMemPrefetchAsync_v2(ptr, 4096, domain_location, 0, 0),
                    hipErrorNotSupported);
    HIP_CHECK(hipFree(ptr));
  }
}

/**
 * End doxygen group hipLocalityDomain.
 * @}
 */

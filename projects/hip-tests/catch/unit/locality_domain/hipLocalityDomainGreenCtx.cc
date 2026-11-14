/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipLocalityDomain hipLocalityDomain
 * @{
 * @ingroup LocalityDomainTest
 * Green context locality domain placement: `hipDevSmResourceSplit` with
 * `hipDevSmResourceGroupLocalityDomainId`, and the descriptor matching rule.
 */

#include "hip_locality_domain_common.hh"

/**
 * Test Description
 * ------------------------
 *  - Basic: splits the device SM resource into a group pinned to locality
 *    domain 0 and builds a green context and stream from it.
 * Test requirements
 * ------------------------
 *  - Runtime implements the locality domain device attributes
 */
HIP_TEST_CASE(Unit_hipLocalityDomain_GreenCtx_Basic) {
  int domains = 0;
  if (!LocalityDomain::GetDomainCount(0, &domains)) {
    HIP_SKIP_TEST(LocalityDomain::kUnsupported);
    return;
  }
  HIP_CHECK(hipSetDevice(0));

  hipDevResource input = LocalityDomain::DeviceSmResource(0);
  hipDevResource result{};
  hipDevSmResourceGroupParams params = LocalityDomain::LocalityParams(0);

  HIP_CHECK(hipDevSmResourceSplit(&result, 1, &input, nullptr, 0, &params));
  REQUIRE(result.sm.smCount > 0);
  REQUIRE(result.sm.smCount <= input.sm.smCount);

  hipDevResourceDesc_t desc{};
  HIP_CHECK(hipDevResourceGenerateDesc(&desc, &result, 1));

  hipExecutionCtx_t ctx = nullptr;
  HIP_CHECK(hipGreenCtxCreate(&ctx, desc, 0, 0));
  REQUIRE(ctx != nullptr);

  hipStream_t stream = nullptr;
  HIP_CHECK(hipExecutionCtxStreamCreate(&stream, ctx, hipStreamNonBlocking, 0));
  REQUIRE(stream != nullptr);

  // The context must report back the SM count it was provisioned with.
  hipDevResource readback{};
  HIP_CHECK(hipExecutionCtxGetDevResource(ctx, &readback, hipDevResourceTypeSm));
  REQUIRE(readback.sm.smCount == result.sm.smCount);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipExecutionCtxDestroy(ctx));
}

/**
 * Test Description
 * ------------------------
 *  - Functional: launches a vectorADD kernel on a stream belonging to a green
 *    context pinned to each available locality domain and verifies the results.
 * Test requirements
 * ------------------------
 *  - Runtime implements the locality domain device attributes
 */
HIP_TEST_CASE(Unit_hipLocalityDomain_GreenCtx_Functional) {
  int domains = 0;
  if (!LocalityDomain::GetDomainCount(0, &domains)) {
    HIP_SKIP_TEST(LocalityDomain::kUnsupported);
    return;
  }
  HIP_CHECK(hipSetDevice(0));

  for (int domain = 0; domain < domains; ++domain) {
    INFO("locality domain " << domain);

    hipDevResource input = LocalityDomain::DeviceSmResource(0);
    hipDevResource result{};
    hipDevSmResourceGroupParams params =
        LocalityDomain::LocalityParams(static_cast<unsigned int>(domain));
    HIP_CHECK(hipDevSmResourceSplit(&result, 1, &input, nullptr, 0, &params));

    hipDevResourceDesc_t desc{};
    HIP_CHECK(hipDevResourceGenerateDesc(&desc, &result, 1));
    hipExecutionCtx_t ctx = nullptr;
    HIP_CHECK(hipGreenCtxCreate(&ctx, desc, 0, 0));
    hipStream_t stream = nullptr;
    HIP_CHECK(hipExecutionCtxStreamCreate(&stream, ctx, hipStreamNonBlocking, 0));

    constexpr size_t kN = 4096;
    constexpr size_t kBytes = kN * sizeof(int);
    std::vector<int> h_a(kN), h_b(kN), h_c(kN, 0);
    for (size_t i = 0; i < kN; ++i) {
      h_a[i] = static_cast<int>(i);
      h_b[i] = static_cast<int>(i * 2);
    }

    int *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    HIP_CHECK(hipMalloc(&d_a, kBytes));
    HIP_CHECK(hipMalloc(&d_b, kBytes));
    HIP_CHECK(hipMalloc(&d_c, kBytes));
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

    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipExecutionCtxDestroy(ctx));
  }
}

/**
 * Test Description
 * ------------------------
 *  - Functional: localityDomainId must be ignored unless the caller opts in with
 *    hipDevSmResourceGroupLocalityDomainId. A nonsense ordinal with the flag
 *    cleared must still produce a valid whole-device split. This guards against
 *    over-validation of the field.
 * Test requirements
 * ------------------------
 *  - Runtime implements the locality domain device attributes
 */
HIP_TEST_CASE(Unit_hipLocalityDomain_GreenCtx_FlagOptIn) {
  int domains = 0;
  if (!LocalityDomain::GetDomainCount(0, &domains)) {
    HIP_SKIP_TEST(LocalityDomain::kUnsupported);
    return;
  }
  HIP_CHECK(hipSetDevice(0));

  hipDevResource input = LocalityDomain::DeviceSmResource(0);
  hipDevResource result{};
  hipDevSmResourceGroupParams params{};
  params.flags = hipDevSmResourceGroupDefault;  // flag deliberately NOT set
  params.localityDomainId = 7;                  // garbage, must be ignored
  params.smCount = 0;

  HIP_CHECK(hipDevSmResourceSplit(&result, 1, &input, nullptr, 0, &params));
  REQUIRE(result.sm.smCount == input.sm.smCount);
}

/**
 * Test Description
 * ------------------------
 *  - Negative: an out-of-range locality domain ordinal must be rejected with
 *    hipErrorInvalidResourceConfiguration, at the boundary, mid-range and at
 *    unsigned overflow. The backfill flag relaxes which SMs are selected, not
 *    whether the ordinal is legal, so it must not rescue an invalid ordinal.
 * Test requirements
 * ------------------------
 *  - Runtime implements the locality domain device attributes
 */
HIP_TEST_CASE(Unit_hipLocalityDomain_GreenCtx_Negative) {
  int domains = 0;
  if (!LocalityDomain::GetDomainCount(0, &domains)) {
    HIP_SKIP_TEST(LocalityDomain::kUnsupported);
    return;
  }
  HIP_CHECK(hipSetDevice(0));

  const unsigned int out_of_range = static_cast<unsigned int>(domains);

  SECTION("ordinal == domain count") {
    hipDevResource input = LocalityDomain::DeviceSmResource(0);
    hipDevResource result{};
    hipDevSmResourceGroupParams params = LocalityDomain::LocalityParams(out_of_range);
    HIP_CHECK_ERROR(hipDevSmResourceSplit(&result, 1, &input, nullptr, 0, &params),
                    hipErrorInvalidResourceConfiguration);
  }

  SECTION("ordinal far out of range") {
    hipDevResource input = LocalityDomain::DeviceSmResource(0);
    hipDevResource result{};
    hipDevSmResourceGroupParams params = LocalityDomain::LocalityParams(99u);
    HIP_CHECK_ERROR(hipDevSmResourceSplit(&result, 1, &input, nullptr, 0, &params),
                    hipErrorInvalidResourceConfiguration);
  }

  SECTION("ordinal at unsigned overflow") {
    hipDevResource input = LocalityDomain::DeviceSmResource(0);
    hipDevResource result{};
    hipDevSmResourceGroupParams params = LocalityDomain::LocalityParams(0xFFFFFFFFu);
    HIP_CHECK_ERROR(hipDevSmResourceSplit(&result, 1, &input, nullptr, 0, &params),
                    hipErrorInvalidResourceConfiguration);
  }

  SECTION("backfill does not rescue an invalid ordinal") {
    hipDevResource input = LocalityDomain::DeviceSmResource(0);
    hipDevResource result{};
    hipDevSmResourceGroupParams params =
        LocalityDomain::LocalityParams(out_of_range, 0, hipDevSmResourceGroupBackfill);
    HIP_CHECK_ERROR(hipDevSmResourceSplit(&result, 1, &input, nullptr, 0, &params),
                    hipErrorInvalidResourceConfiguration);
  }

  SECTION("smCount larger than the device") {
    hipDevResource input = LocalityDomain::DeviceSmResource(0);
    hipDevResource result{};
    hipDevSmResourceGroupParams params =
        LocalityDomain::LocalityParams(0, input.sm.smCount * 4);
    HIP_CHECK_ERROR(hipDevSmResourceSplit(&result, 1, &input, nullptr, 0, &params),
                    hipErrorInvalidResourceConfiguration);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Negative: hipDevResourceGenerateDesc must reject SM resources that disagree
 *    on localityDomainId. Resources sharing an ordinal must still be accepted.
 * Test requirements
 * ------------------------
 *  - Runtime implements the locality domain device attributes
 *  - A locality domain has enough compute units to split into two aligned groups
 */
HIP_TEST_CASE(Unit_hipLocalityDomain_GreenCtx_GenerateDesc_Negative) {
  int domains = 0;
  if (!LocalityDomain::GetDomainCount(0, &domains)) {
    HIP_SKIP_TEST(LocalityDomain::kUnsupported);
    return;
  }
  HIP_CHECK(hipSetDevice(0));

  hipDevResource input = LocalityDomain::DeviceSmResource(0);
  const unsigned int alignment =
      input.sm.smCoscheduledAlignment ? input.sm.smCoscheduledAlignment : 1u;

  int domain_cus = 0;
  if (!LocalityDomain::GetDomainCuCount(0, &domain_cus) || domain_cus < 1) {
    HIP_SKIP_TEST(LocalityDomain::kUnsupported);
    return;
  }

  const unsigned int half =
      (static_cast<unsigned int>(domain_cus) / 2u / alignment) * alignment;
  if (half == 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kSmCountTooSmall);
    return;
  }

  hipDevResource results[2]{};
  hipDevSmResourceGroupParams params[2]{};
  params[0] = LocalityDomain::LocalityParams(0, half);
  params[1] = LocalityDomain::LocalityParams(0, half);
  HIP_CHECK(hipDevSmResourceSplit(results, 2, &input, nullptr, 0, params));

  SECTION("matching locality domain is accepted") {
    hipDevResourceDesc_t desc{};
    HIP_CHECK(hipDevResourceGenerateDesc(&desc, results, 2));
  }

  SECTION("mismatched locality domain is rejected") {
    // Tamper with the reported ordinal so the two resources disagree.
    results[1].sm.localityDomainId = 7;
    hipDevResourceDesc_t desc{};
    HIP_CHECK_ERROR(hipDevResourceGenerateDesc(&desc, results, 2),
                    hipErrorInvalidResourceConfiguration);
  }
}

/**
 * End doxygen group hipLocalityDomain.
 * @}
 */

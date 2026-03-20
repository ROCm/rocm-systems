/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "mem_pools_performance_common.hh"

/**
 * @addtogroup stream stream
 * @{
 * @ingroup PerformanceTest
 */

class MemPoolSetAttributeBenchmark : public Benchmark<MemPoolSetAttributeBenchmark> {
 public:
  void operator()(const hipMemPoolAttr attribute) {
    hipMemPool_t mem_pool{nullptr};
    hipMemPoolProps pool_props = CreateMemPoolProps(0, hipMemHandleTypeNone);
    HIP_CHECK(hipMemPoolCreate(&mem_pool, &pool_props));

    if (attribute == hipMemPoolAttrReleaseThreshold) {
      uint64_t value{0};
      TIMED_SECTION(kTimerTypeCpu) {
        HIP_CHECK(hipMemPoolSetAttribute(mem_pool, attribute, &value));
      }
    } else {
      int value{0};
      TIMED_SECTION(kTimerTypeCpu) {
        HIP_CHECK(hipMemPoolSetAttribute(mem_pool, attribute, &value));
      }
    }

    HIP_CHECK(hipMemPoolDestroy(mem_pool));
  }
};

static void RunBenchmark(const hipMemPoolAttr attribute) {
  MemPoolSetAttributeBenchmark benchmark;
  benchmark.AddSectionName(GetMemPoolAttrSectionName(attribute));
  benchmark.Run(attribute);
}

/**
 * @warning **MemPool APIs are not fully implemented within current version
 *          or HIP and therefore they cannot be appropriately executed on AMD and NVIDIA platforms.
 *          Therefore, all tests related to MemPool APIs are implemented without formal
 *          verification and will be verified once HIP fully supports MemPool APIs.**
 * Test Description
 * ------------------------
 *  - Executes `hipMemPoolSetAttribute`:
 *    -# Supported attributes:
 *      - `hipMemPoolAttrReleaseThreshold`
 *      - `hipMemPoolReuseFollowEventDependencies`
 *      - `hipMemPoolReuseAllowOpportunistic`
 *      - `hipMemPoolReuseAllowInternalDependencies`
 * Test source
 * ------------------------
 *  - performance/stream/hipMemPoolSetAttribute.cc
 * Test requirements
 * ------------------------
 *  - Device supports memory pools
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipMemPoolSetAttribute) {
  if (!AreMemPoolsSupported(0)) {
    HipTest::HIP_SKIP_TEST(
        "GPU 0 doesn't support hipDeviceAttributeMemoryPoolsSupported "
        "attribute. Hence skipping the testing with Pass result.\n");
    return;
  }
  hipMemPoolAttr attribute =
      GENERATE(hipMemPoolAttrReleaseThreshold, hipMemPoolReuseFollowEventDependencies,
               hipMemPoolReuseAllowOpportunistic, hipMemPoolReuseAllowInternalDependencies);
  RunBenchmark(attribute);
}

/**
 * End doxygen group PerformanceTest.
 * @}
 */

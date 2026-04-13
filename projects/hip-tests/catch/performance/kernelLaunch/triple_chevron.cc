/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "kernel_launch_common.hh"

#include <hip_test_common.hh>

/**
 * @addtogroup kernelLaunch kernel launch
 * @{
 * @ingroup PerformanceTest
 * Contains performance tests for kernel launch overhead benchmarking.
 */

template <KernelType kernel_type, bool timer_type> class TripleChevronBenchmark
    : public KernelLaunchBenchmark<TripleChevronBenchmark<kernel_type, timer_type>, timer_type> {
 public:
  constexpr void LaunchKernel() {
    if constexpr (kernel_type == KernelType::kNull) {
      NullKernel<<<1, 1>>>();
    } else if constexpr (kernel_type == KernelType::kSmall) {
      KernelWithSmallArgs<<<1, 1>>>(small_kernel_args, nullptr);
    } else if constexpr (kernel_type == KernelType::kMedium) {
      KernelWithMediumArgs<<<1, 1>>>(medium_kernel_args, nullptr);
    } else if constexpr (kernel_type == KernelType::kLarge) {
      KernelWithLargeArgs<<<1, 1>>>(large_kernel_args, nullptr);
    } else
      ;
  }
};

template <KernelType kernel_type, bool timer_type> static void RunBenchmark(bool sync) {
  TripleChevronBenchmark<kernel_type, timer_type> benchmark;
  benchmark.AddSectionName(GetSynchronizationSectionName(sync));
  benchmark.AddSectionName(GetKernelTypeSectionName<kernel_type>());
  benchmark.AddSectionName(GetTimerTypeSectionName<timer_type>());
  benchmark.Run(sync);
  HIP_CHECK(hipGetLastError());
}

/**
 * Test Description
 * ------------------------
 *  - Calls an empty kernel using triple chevron annotation:
 *    -# With different timing methods:
 *      - CPU-based
 *      - Event-based
 *    -# With different synchronization behavior:
 *      - Using a stream synchronization between each iteration
 *      - Without any synchronization between iterations
 *    -# With different kernel argument sizes
 * Test source
 * ------------------------
 *  - performance/kernelLaunch/triple_chevron.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_Triple_Chevron) {
  bool sync = GENERATE(true, false);

  SECTION("null kernel") {
    SECTION("cpu-based timing") { RunBenchmark<KernelType::kNull, kTimerTypeCpu>(sync); }

    SECTION("event-based timing") { RunBenchmark<KernelType::kNull, kTimerTypeEvent>(sync); }
  }

  SECTION("small kernel") {
    SECTION("cpu-based timing") { RunBenchmark<KernelType::kSmall, kTimerTypeCpu>(sync); }

    SECTION("event-based timing") { RunBenchmark<KernelType::kSmall, kTimerTypeEvent>(sync); }
  }

  SECTION("medium kernel") {
    SECTION("cpu-based timing") { RunBenchmark<KernelType::kMedium, kTimerTypeCpu>(sync); }

    SECTION("event-based timing") { RunBenchmark<KernelType::kMedium, kTimerTypeEvent>(sync); }
  }

  SECTION("large kernel") {
    SECTION("cpu-based timing") { RunBenchmark<KernelType::kLarge, kTimerTypeCpu>(sync); }

    SECTION("event-based timing") { RunBenchmark<KernelType::kLarge, kTimerTypeEvent>(sync); }
  }
}

/**
 * End doxygen group kernelLaunch.
 * @}
 */

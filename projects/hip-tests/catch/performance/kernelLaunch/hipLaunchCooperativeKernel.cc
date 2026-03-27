/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "kernel_launch_common.hh"

#include <hip_test_common.hh>
#include <utils.hh>

/**
 * @addtogroup kernelLaunch kernel launch
 * @{
 * @ingroup PerformanceTest
 * Contains performance tests for kernel launch overhead benchmarking.
 */

template <KernelType kernel_type, bool timer_type> class LaunchCooperativeKernelBenchmark
    : public KernelLaunchBenchmark<LaunchCooperativeKernelBenchmark<kernel_type, timer_type>,
                                   timer_type> {
 public:
  constexpr void LaunchKernel() {
    if constexpr (kernel_type == KernelType::kNull) {
      error_ = hipLaunchCooperativeKernel(reinterpret_cast<void*>(NullKernel), dim3{1, 1, 1},
                                          dim3{1, 1, 1}, nullptr, 0, nullptr);
    } else if constexpr (kernel_type == KernelType::kSmall) {
      error_ =
          hipLaunchCooperativeKernel(reinterpret_cast<void*>(KernelWithSmallArgs), dim3{1, 1, 1},
                                     dim3{1, 1, 1}, small_kernel_args_, 0, nullptr);
    } else if constexpr (kernel_type == KernelType::kMedium) {
      error_ =
          hipLaunchCooperativeKernel(reinterpret_cast<void*>(KernelWithMediumArgs), dim3{1, 1, 1},
                                     dim3{1, 1, 1}, medium_kernel_args_, 0, nullptr);
    } else if constexpr (kernel_type == KernelType::kLarge) {
      error_ =
          hipLaunchCooperativeKernel(reinterpret_cast<void*>(KernelWithLargeArgs), dim3{1, 1, 1},
                                     dim3{1, 1, 1}, large_kernel_args_, 0, nullptr);
    } else
      ;
  }

  hipError_t GetError() { return error_; }

 private:
  hipError_t error_;

  char* out_ = nullptr;
  void* small_kernel_args_[2] = {&small_kernel_args, &out_};
  void* medium_kernel_args_[2] = {&medium_kernel_args, &out_};
  void* large_kernel_args_[2] = {&large_kernel_args, &out_};
};

template <KernelType kernel_type, bool timer_type> static void RunBenchmark(bool sync) {
  LaunchCooperativeKernelBenchmark<kernel_type, timer_type> benchmark;
  benchmark.AddSectionName(GetSynchronizationSectionName(sync));
  benchmark.AddSectionName(GetKernelTypeSectionName<kernel_type>());
  benchmark.AddSectionName(GetTimerTypeSectionName<timer_type>());
  benchmark.Run(sync);
  HIP_CHECK(benchmark.GetError());
}

/**
 * Test Description
 * ------------------------
 *  - Calls an empty kernel using hipLaunchCooperativeKernel:
 *    -# With different timing methods:
 *      - CPU-based
 *      - Event-based
 *    -# With different synchronization behavior:
 *      - Using a stream synchronization between each iteration
 *      - Without any synchronization between iterations
 *    -# With different kernel argument sizes
 * Test source
 * ------------------------
 *  - performance/kernelLaunch/hipLaunchCooperativeKernel.cc
 * Test requirements
 * ------------------------
 *  - Device supports CooperativeLaunch
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Performance_hipLaunchCooperativeKernel) {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
    HipTest::HIP_SKIP_TEST("CooperativeLaunch not supported");
    return;
  }

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

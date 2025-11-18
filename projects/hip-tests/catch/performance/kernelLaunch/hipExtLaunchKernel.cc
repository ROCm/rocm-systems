/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "kernel_launch_common.hh"

#include <hip_test_common.hh>

/**
 * @addtogroup kernelLaunch kernel launch
 * @{
 * @ingroup PerformanceTest
 * Contains performance tests for kernel launch overhead benchmarking.
 */

template <KernelType kernel_type, bool timer_type> class ExtLaunchKernelBenchmark
    : public KernelLaunchBenchmark<ExtLaunchKernelBenchmark<kernel_type, timer_type>, timer_type> {
 public:
  constexpr void LaunchKernel() {
    if constexpr (kernel_type == KernelType::kNull) {
      error_ = hipExtLaunchKernel(reinterpret_cast<void*>(NullKernel), 1, 1, nullptr, 0, nullptr,
                                  events_[0], events_[1], 0u);
    } else if constexpr (kernel_type == KernelType::kSmall) {
      error_ = hipExtLaunchKernel(reinterpret_cast<void*>(KernelWithSmallArgs), 1, 1,
                                  small_kernel_args_, 0, nullptr, events_[0], events_[1], 0u);
    } else if constexpr (kernel_type == KernelType::kMedium) {
      error_ = hipExtLaunchKernel(reinterpret_cast<void*>(KernelWithMediumArgs), 1, 1,
                                  medium_kernel_args_, 0, nullptr, events_[0], events_[1], 0u);
    } else if constexpr (kernel_type == KernelType::kLarge) {
      error_ = hipExtLaunchKernel(reinterpret_cast<void*>(KernelWithLargeArgs), 1, 1,
                                  large_kernel_args_, 0, nullptr, events_[0], events_[1], 0u);
    } else
      ;
  }

  hipError_t GetError() { return error_; }

 private:
  EventsGuard events_{2};
  hipError_t error_;

  char* out_ = nullptr;
  void* small_kernel_args_[2] = {&small_kernel_args, &out_};
  void* medium_kernel_args_[2] = {&medium_kernel_args, &out_};
  void* large_kernel_args_[2] = {&large_kernel_args, &out_};
};

template <KernelType kernel_type, bool timer_type> static void RunBenchmark(bool sync) {
  ExtLaunchKernelBenchmark<kernel_type, timer_type> benchmark;
  benchmark.AddSectionName(GetSynchronizationSectionName(sync));
  benchmark.AddSectionName(GetKernelTypeSectionName<kernel_type>());
  benchmark.AddSectionName(GetTimerTypeSectionName<timer_type>());
  benchmark.Run(sync);
  HIP_CHECK(benchmark.GetError());
}

/**
 * Test Description
 * ------------------------
 *  - Calls an empty kernel using hipExtLaunchKernel:
 *    -# With different timing methods:
 *      - CPU-based
 *      - Event-based
 *    -# With different synchronization behavior:
 *      - Using a stream synchronization between each iteration
 *      - Without any synchronization between iterations
 *    -# With different kernel argument sizes
 * Test source
 * ------------------------
 *  - performance/kernelLaunch/hipExtLaunchKernel.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Performance_hipExtLaunchKernel") {
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

/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <hip_test_common.hh>
#include <performance_common.hh>

struct SmallKernelArgs {
  char args[16];
};

struct MediumKernelArgs {
  char args[256];
};

struct LargeKernelArgs {
  char args[4080];
};

extern SmallKernelArgs small_kernel_args;
extern MediumKernelArgs medium_kernel_args;
extern LargeKernelArgs large_kernel_args;

__global__ void NullKernel();

__global__ void KernelWithSmallArgs(SmallKernelArgs, char*);

__global__ void KernelWithMediumArgs(MediumKernelArgs, char*);

__global__ void KernelWithLargeArgs(LargeKernelArgs, char*);

enum class KernelType { kNull = 0, kSmall, kMedium, kLarge };

template <typename Derived, bool timer_type>
class KernelLaunchBenchmark : public Benchmark<KernelLaunchBenchmark<Derived, timer_type>> {
 public:
  void operator()(bool sync = true) {
    auto& derived = static_cast<Derived&>(*this);

    if (sync) {
      TIMED_SECTION(timer_type) { derived.LaunchKernel(); }
    } else {
      if (this->current() != this->kWarmup)  // if not warmup
        RunWithoutSynchronization();
    }
  }

 private:
  void RunWithoutSynchronization() {
    auto iterations = this->iterations();
    auto warmups = this->warmups();

    // manually handle iterations here to avoid synchronization after each iteration
    this->Configure(1, 0);

    this->RegisterModifier([iterations](float time) { return time / iterations; });

    auto& derived = static_cast<Derived&>(*this);

    for (size_t i = 0u; i < warmups; ++i) {
      derived.LaunchKernel();
    }

    TIMED_SECTION(timer_type) {
      for (size_t i = 0u; i < iterations; ++i) {
        derived.LaunchKernel();
      }
    }
  }
};

static std::string GetSynchronizationSectionName(bool sync) {
  return sync ? "with synchronization" : "without synchronization";
}

template <KernelType kernel_type> std::string GetKernelTypeSectionName() {
  if constexpr (kernel_type == KernelType::kNull) {
    return "null kernel";
  } else if constexpr (kernel_type == KernelType::kSmall) {
    return "small kernel";
  } else if constexpr (kernel_type == KernelType::kMedium) {
    return "medium kernel";
  } else if constexpr (kernel_type == KernelType::kLarge) {
    return "large kernel";
  } else {
    return "unknown kernel type";
  }
}

template <bool timer_type> std::string GetTimerTypeSectionName() {
  if constexpr (timer_type == kTimerTypeEvent) {
    return "event based";
  } else {
    return "cpu based";
  }
}

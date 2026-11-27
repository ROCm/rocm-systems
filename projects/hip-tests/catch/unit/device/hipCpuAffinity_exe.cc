/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>
#include <sched.h>
#include <unistd.h>
#include <cerrno>
#include <iostream>

// Exit codes, also used by Unit_hipGetDeviceCount_KeepsCpuAffinity.
constexpr int kMaskKept = 0;
constexpr int kMaskChanged = 1;
constexpr int kMaskRestricted = 2;
constexpr int kError = 3;

// Allocates a CPU set large enough for the kernel's CPU mask. sysconf(_SC_NPROCESSORS_CONF) can
// be smaller than the number of possible CPUs, so grow the set until sched_getaffinity accepts it.
static cpu_set_t* GetAffinity(int* cpus) {
  for (int count = CPU_SETSIZE;; count *= 2) {
    cpu_set_t* set = CPU_ALLOC(count);
    if (set == nullptr) {
      return nullptr;
    }
    if (sched_getaffinity(0, CPU_ALLOC_SIZE(count), set) == 0) {
      *cpus = count;
      return set;
    }
    CPU_FREE(set);
    if (errno != EINVAL) {
      return nullptr;
    }
  }
}

// Reports whether initializing HIP changes the CPU mask of the calling thread. A launcher that
// calls hipGetDeviceCount and then forks workers passes this mask on to every worker.
int main() {
  int cpus = 0;
  cpu_set_t* before = GetAffinity(&cpus);
  if (before == nullptr) {
    std::cout << "sched_getaffinity failed" << std::endl;
    return kError;
  }
  const size_t size = CPU_ALLOC_SIZE(cpus);

  // The parent may already run on a narrowed mask (e.g. after its own HIP calls), which this
  // process inherits. Ask for every CPU; the kernel limits the request to the cpuset.
  for (int cpu = 0; cpu < cpus; ++cpu) {
    CPU_SET_S(cpu, size, before);
  }
  if (sched_setaffinity(0, size, before) != 0 || sched_getaffinity(0, size, before) != 0) {
    std::cout << "sched_setaffinity failed" << std::endl;
    CPU_FREE(before);
    return kError;
  }

  int result = kMaskKept;
  int deviceCount = 0;
  cpu_set_t* after = CPU_ALLOC(cpus);
  if (CPU_COUNT_S(size, before) != sysconf(_SC_NPROCESSORS_ONLN)) {
    // The runtime leaves a restricted mask alone, so the check needs a cpuset with every online
    // CPU.
    result = kMaskRestricted;
  } else if (hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount == 0) {
    std::cout << "hipGetDeviceCount failed" << std::endl;
    result = kError;
  } else if (after == nullptr || sched_getaffinity(0, size, after) != 0) {
    std::cout << "sched_getaffinity failed" << std::endl;
    result = kError;
  } else if (!CPU_EQUAL_S(size, before, after)) {
    std::cout << "CPU mask changed from " << CPU_COUNT_S(size, before) << " to "
              << CPU_COUNT_S(size, after) << " CPUs" << std::endl;
    result = kMaskChanged;
  }

  CPU_FREE(before);
  if (after != nullptr) {
    CPU_FREE(after);
  }
  return result;
}

/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once
#include "hip_test_common.hh"

#ifdef __linux__
  #include <sys/sysinfo.h>
#else
  #include <windows.h>
  #include <sysinfoapi.h>
#endif

namespace HipTest {
static inline int getGeviceCount() {
  int dev = 0;
  HIP_CHECK(hipGetDeviceCount(&dev));
  return dev;
}

// Get available system memory in MB
static inline size_t getAvailableSystemMemoryInMB() {
#ifdef __linux__
  struct sysinfo info{};
  sysinfo(&info);
  return info.freeram / (1024 * 1024);  // MB
#elif defined(_WIN32)
  MEMORYSTATUSEX statex;
  statex.dwLength = sizeof(statex);
  GlobalMemoryStatusEx(&statex);
  return (statex.ullAvailPhys / (1024 * 1024));  // MB
#endif
}

// Get total system memory in MB
static inline size_t getTotalSystemMemoryInMB() {
#ifdef __linux__
  struct sysinfo info{};
  sysinfo(&info);
  return info.totalram / (1024 * 1024);  // MB
#elif defined(_WIN32)
  MEMORYSTATUSEX statex;
  statex.dwLength = sizeof(statex);
  GlobalMemoryStatusEx(&statex);
  return (statex.ullTotalPhys / (1024 * 1024));  // MB
#endif
}

static inline size_t getHostThreadCount(const size_t memPerThread, const size_t maxThreads) {
  if (memPerThread == 0) return 0;
  auto memAmount = getAvailableSystemMemoryInMB();
  const auto processor_count = std::thread::hardware_concurrency();
  if (processor_count == 0 || memAmount == 0) return 0;
  size_t thread_count = 0;
  if ((processor_count * memPerThread) < memAmount)
    thread_count = processor_count;
  else
    thread_count = reinterpret_cast<size_t>(memAmount / memPerThread);
  if (maxThreads > 0) {
    return (thread_count > maxThreads) ? maxThreads : thread_count;
  }
  return thread_count;
}

}  // namespace HipTest

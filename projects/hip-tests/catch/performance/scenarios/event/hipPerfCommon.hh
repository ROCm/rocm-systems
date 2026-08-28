/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

#define HIP_CHECK_PERF(a)                                                                          \
  {                                                                                                \
    auto err = a;                                                                                  \
    if ((err != hipSuccess) && (err != hipErrorNotReady)) {                                        \
      printf(#a "= Error! %s\n", hipGetErrorString(err));                                          \
      exit(1);                                                                                     \
    }                                                                                              \
  }

// Padded atomic timestamp slot. Each slot holds nanoseconds-since-epoch and is
// aligned/padded to a cache line so that adjacent worker threads writing their
// own slots do not invalidate each other's cache lines (false sharing).
#ifdef __cpp_lib_hardware_interference_size
static constexpr size_t kCacheLineSize = std::hardware_destructive_interference_size;
#else
static constexpr size_t kCacheLineSize = 64;
#endif
struct alignas(kCacheLineSize) PaddedTimestamp {
  std::atomic<int64_t> ns{0};
  // Pad out the rest of the cache line so neighboring slots live on
  // independent lines and writes from one thread don't ping-pong another's
  // cache line.
  char pad[kCacheLineSize - sizeof(std::atomic<int64_t>)];
};

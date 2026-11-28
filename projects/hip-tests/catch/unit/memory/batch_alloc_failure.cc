/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <dlfcn.h>

namespace {
// The public ROCr C ABI uses a single 64-bit pool handle. This host-only shim
// needs no ROCr development headers or production failure-injection setting.
struct Pool {
  uint64_t handle;
};
using Allocate = uint32_t (*)(Pool, size_t, uint32_t, void**);
constexpr uint32_t kOutOfResources = 0x1008;
std::atomic<size_t> rejected_size{0};
std::atomic<unsigned> failures{0};
}  // namespace

extern "C" void hip_test_fail_pool_allocation(size_t size) {
  if (size != 0) failures.store(0);
  rejected_size.store(size);
}
extern "C" unsigned hip_test_pool_allocation_failures() { return failures.load(); }
extern "C" uint32_t hsa_amd_memory_pool_allocate(Pool pool, size_t size, uint32_t flags,
                                                 void** pointer) {
  static const auto allocate =
      reinterpret_cast<Allocate>(dlvsym(RTLD_NEXT, "hsa_amd_memory_pool_allocate", "ROCR_1"));
  if (size != 0 && size == rejected_size.load()) {
    failures.fetch_add(1);
    *pointer = nullptr;
    return kOutOfResources;
  }
  return allocate ? allocate(pool, size, flags, pointer) : kOutOfResources;
}

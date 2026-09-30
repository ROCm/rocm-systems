// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file asan_hsa_runtime_forwarder_mock.cpp
/// @brief Supplies an RTLD_LOCAL HSA allocation target for forwarder tests.

#include <hsa/hsa_ext_amd.h>

namespace {

int allocation_storage;

} // namespace

#if defined(__GNUC__)
#define RJ_TEST_EXPORT __attribute__((visibility("default")))
#else
#define RJ_TEST_EXPORT
#endif

extern "C" RJ_TEST_EXPORT hsa_status_t HSA_API hsa_amd_memory_pool_allocate(hsa_amd_memory_pool_t,
                                                                            size_t, uint32_t,
                                                                            void **ptr) {
  *ptr = &allocation_storage;
  return HSA_STATUS_SUCCESS;
}

#undef RJ_TEST_EXPORT

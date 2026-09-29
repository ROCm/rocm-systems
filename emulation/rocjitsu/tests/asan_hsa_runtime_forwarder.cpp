// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file asan_hsa_runtime_forwarder.cpp
/// @brief Keeps ASan's HSA callbacks valid across a dynamically loaded ROCr runtime.

#include <hsa/hsa_ext_amd.h>

#include <dlfcn.h>

#include <cstring>

namespace {

// Do not cache these pointers: ROCr may unload and later reload while this
// forwarding DSO stays resident for ASan's process lifetime.
template <typename Function> Function resolve_next(const char *name) {
  static_assert(sizeof(Function) == sizeof(void *));
  void *symbol = dlsym(RTLD_NEXT, name);
  Function function = nullptr;
  std::memcpy(&function, &symbol, sizeof(function));
  return function;
}

} // namespace

#if defined(__GNUC__)
#define RJ_TEST_EXPORT __attribute__((visibility("default")))
#else
#define RJ_TEST_EXPORT
#endif

// Give the test executable a concrete reference that prevents --as-needed
// from dropping this DSO before ASan performs its early HSA symbol lookup.
extern "C" RJ_TEST_EXPORT void rj_test_keep_asan_hsa_runtime_forwarder() {}

extern "C" RJ_TEST_EXPORT hsa_status_t HSA_API hsa_amd_memory_pool_allocate(
    hsa_amd_memory_pool_t memory_pool, size_t size, uint32_t flags, void **ptr) {
  const auto function =
      resolve_next<decltype(&hsa_amd_memory_pool_allocate)>("hsa_amd_memory_pool_allocate");
  return function != nullptr ? function(memory_pool, size, flags, ptr) : HSA_STATUS_ERROR;
}

extern "C" RJ_TEST_EXPORT hsa_status_t HSA_API hsa_amd_memory_pool_free(void *ptr) {
  const auto function =
      resolve_next<decltype(&hsa_amd_memory_pool_free)>("hsa_amd_memory_pool_free");
  return function != nullptr ? function(ptr) : HSA_STATUS_ERROR;
}

extern "C" RJ_TEST_EXPORT hsa_status_t HSA_API hsa_amd_pointer_info(const void *ptr,
                                                                    hsa_amd_pointer_info_t *info,
                                                                    void *(*alloc)(size_t),
                                                                    uint32_t *num_agents_accessible,
                                                                    hsa_agent_t **accessible) {
  const auto function = resolve_next<decltype(&hsa_amd_pointer_info)>("hsa_amd_pointer_info");
  return function != nullptr ? function(ptr, info, alloc, num_agents_accessible, accessible)
                             : HSA_STATUS_ERROR;
}

extern "C" RJ_TEST_EXPORT hsa_status_t HSA_API
hsa_amd_register_system_event_handler(hsa_amd_system_event_callback_t callback, void *data) {
  const auto function = resolve_next<decltype(&hsa_amd_register_system_event_handler)>(
      "hsa_amd_register_system_event_handler");
  return function != nullptr ? function(callback, data) : HSA_STATUS_ERROR;
}

extern "C" RJ_TEST_EXPORT hsa_status_t hsa_amd_vmem_address_reserve_align(void **va, size_t size,
                                                                          uint64_t address,
                                                                          uint64_t alignment,
                                                                          uint64_t flags) {
  const auto function = resolve_next<decltype(&hsa_amd_vmem_address_reserve_align)>(
      "hsa_amd_vmem_address_reserve_align");
  return function != nullptr ? function(va, size, address, alignment, flags) : HSA_STATUS_ERROR;
}

extern "C" RJ_TEST_EXPORT hsa_status_t hsa_amd_vmem_address_free(void *va, size_t size) {
  const auto function =
      resolve_next<decltype(&hsa_amd_vmem_address_free)>("hsa_amd_vmem_address_free");
  return function != nullptr ? function(va, size) : HSA_STATUS_ERROR;
}

#undef RJ_TEST_EXPORT

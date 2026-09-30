// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file asan_hsa_runtime_forwarder.cpp
/// @brief Keeps ASan's HSA callbacks valid across a dynamically loaded ROCr runtime.

#include <hsa/hsa_ext_amd.h>

#include <dlfcn.h>
#include <link.h>

#include <cstring>
#include <memory>
#include <mutex>
#include <utility>

namespace {

class RuntimeHandle {
public:
  static std::shared_ptr<RuntimeHandle> retain(void *handle) {
    link_map *map = nullptr;
    if (dlinfo(handle, RTLD_DI_LINKMAP, &map) != 0 || map == nullptr || map->l_name == nullptr ||
        map->l_name[0] == '\0') {
      return nullptr;
    }

    // Take an independent reference so an in-flight forwarded call remains
    // valid if the test replaces or clears the configured lookup handle.
    void *retained = dlopen(map->l_name, RTLD_NOW | RTLD_LOCAL);
    return retained != nullptr ? std::shared_ptr<RuntimeHandle>(new RuntimeHandle(retained))
                               : nullptr;
  }

  RuntimeHandle(const RuntimeHandle &) = delete;
  RuntimeHandle &operator=(const RuntimeHandle &) = delete;

  ~RuntimeHandle() { dlclose(handle_); }

  [[nodiscard]] void *get() const { return handle_; }

private:
  explicit RuntimeHandle(void *handle) : handle_(handle) {}

  void *handle_;
};

std::mutex runtime_handle_mutex;
std::shared_ptr<RuntimeHandle> runtime_handle;

std::shared_ptr<RuntimeHandle> get_runtime_handle() {
  const std::lock_guard lock(runtime_handle_mutex);
  return runtime_handle;
}

template <typename Function> Function resolve(void *handle, const char *name) {
  static_assert(sizeof(Function) == sizeof(void *));
  void *symbol = dlsym(handle, name);
  Function function = nullptr;
  std::memcpy(&function, &symbol, sizeof(function));
  return function;
}

// Resolve on every call: the test may unload one runtime and load another
// while this forwarding DSO stays resident for ASan's process lifetime.
template <typename Function, typename... Args>
hsa_status_t forward(const char *name, Args &&...args) {
  const std::shared_ptr<RuntimeHandle> runtime = get_runtime_handle();
  if (runtime == nullptr) {
    return HSA_STATUS_ERROR;
  }
  const auto function = resolve<Function>(runtime->get(), name);
  return function != nullptr ? function(std::forward<Args>(args)...) : HSA_STATUS_ERROR;
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

extern "C" RJ_TEST_EXPORT int rj_test_set_asan_hsa_runtime_handle(void *handle) {
  std::shared_ptr<RuntimeHandle> replacement;
  if (handle != nullptr) {
    replacement = RuntimeHandle::retain(handle);
    if (replacement == nullptr) {
      return 0;
    }
  }

  {
    const std::lock_guard lock(runtime_handle_mutex);
    replacement.swap(runtime_handle);
  }
  return 1;
}

extern "C" RJ_TEST_EXPORT hsa_status_t HSA_API hsa_amd_memory_pool_allocate(
    hsa_amd_memory_pool_t memory_pool, size_t size, uint32_t flags, void **ptr) {
  return forward<decltype(&hsa_amd_memory_pool_allocate)>("hsa_amd_memory_pool_allocate",
                                                          memory_pool, size, flags, ptr);
}

extern "C" RJ_TEST_EXPORT hsa_status_t HSA_API hsa_amd_memory_pool_free(void *ptr) {
  return forward<decltype(&hsa_amd_memory_pool_free)>("hsa_amd_memory_pool_free", ptr);
}

extern "C" RJ_TEST_EXPORT hsa_status_t HSA_API hsa_amd_pointer_info(const void *ptr,
                                                                    hsa_amd_pointer_info_t *info,
                                                                    void *(*alloc)(size_t),
                                                                    uint32_t *num_agents_accessible,
                                                                    hsa_agent_t **accessible) {
  return forward<decltype(&hsa_amd_pointer_info)>("hsa_amd_pointer_info", ptr, info, alloc,
                                                  num_agents_accessible, accessible);
}

extern "C" RJ_TEST_EXPORT hsa_status_t HSA_API
hsa_amd_register_system_event_handler(hsa_amd_system_event_callback_t callback, void *data) {
  return forward<decltype(&hsa_amd_register_system_event_handler)>(
      "hsa_amd_register_system_event_handler", callback, data);
}

extern "C" RJ_TEST_EXPORT hsa_status_t hsa_amd_vmem_address_reserve_align(void **va, size_t size,
                                                                          uint64_t address,
                                                                          uint64_t alignment,
                                                                          uint64_t flags) {
  return forward<decltype(&hsa_amd_vmem_address_reserve_align)>(
      "hsa_amd_vmem_address_reserve_align", va, size, address, alignment, flags);
}

extern "C" RJ_TEST_EXPORT hsa_status_t hsa_amd_vmem_address_free(void *va, size_t size) {
  return forward<decltype(&hsa_amd_vmem_address_free)>("hsa_amd_vmem_address_free", va, size);
}

#undef RJ_TEST_EXPORT

// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file asan_hsa_runtime_forwarder_test.cpp
/// @brief Verifies HSA forwarding into a local, reloadable runtime DSO.

#include <hsa/hsa_ext_amd.h>

#include <dlfcn.h>

#include <cstdio>
#include <cstring>

namespace {

using SetRuntimeHandle = int (*)(void *);
using MemoryPoolAllocate = decltype(&hsa_amd_memory_pool_allocate);

extern "C" void rj_test_keep_asan_hsa_runtime_forwarder();

template <typename Function> Function resolve(void *library, const char *name) {
  static_assert(sizeof(Function) == sizeof(void *));
  void *symbol = dlsym(library, name);
  Function function = nullptr;
  std::memcpy(&function, &symbol, sizeof(function));
  return function;
}

int run_load_cycle(const char *forwarder_path, const char *runtime_path) {
  void *runtime = dlopen(runtime_path, RTLD_NOW | RTLD_LOCAL);
  if (runtime == nullptr) {
    std::fprintf(stderr, "dlopen(%s) failed: %s\n", runtime_path, dlerror());
    return 3;
  }

  void *forwarder = dlopen(forwarder_path, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
  if (forwarder == nullptr) {
    std::fprintf(stderr, "forwarder is not resident: %s\n", dlerror());
    dlclose(runtime);
    return 4;
  }

  const auto set_runtime_handle =
      resolve<SetRuntimeHandle>(forwarder, "rj_test_set_asan_hsa_runtime_handle");
  const auto allocate = resolve<MemoryPoolAllocate>(forwarder, "hsa_amd_memory_pool_allocate");
  if (set_runtime_handle == nullptr || allocate == nullptr) {
    std::fprintf(stderr, "forwarder control or allocation symbol is missing: %s\n", dlerror());
    dlclose(forwarder);
    dlclose(runtime);
    return 5;
  }

  if (set_runtime_handle(runtime) == 0) {
    std::fprintf(stderr, "forwarder failed to retain the runtime lookup handle\n");
    dlclose(forwarder);
    dlclose(runtime);
    return 6;
  }
  void *allocation = nullptr;
  const hsa_status_t status = allocate({}, 4096, 0, &allocation);
  if (set_runtime_handle(nullptr) == 0) {
    std::fprintf(stderr, "forwarder failed to clear the runtime lookup handle\n");
    dlclose(forwarder);
    dlclose(runtime);
    return 7;
  }

  if (dlclose(forwarder) != 0) {
    std::fprintf(stderr, "dlclose(forwarder probe) failed: %s\n", dlerror());
    dlclose(runtime);
    return 8;
  }
  if (dlclose(runtime) != 0) {
    std::fprintf(stderr, "dlclose(runtime) failed: %s\n", dlerror());
    return 9;
  }
  void *unload_probe = dlopen(runtime_path, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
  if (unload_probe != nullptr) {
    std::fprintf(stderr, "runtime remained resident after its forwarder handle was cleared\n");
    dlclose(unload_probe);
    return 10;
  }
  if (status != HSA_STATUS_SUCCESS || allocation == nullptr) {
    std::fprintf(stderr, "forwarded allocation failed: status=%u allocation=%p\n",
                 static_cast<unsigned>(status), allocation);
    return 11;
  }
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  rj_test_keep_asan_hsa_runtime_forwarder();
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s /path/to/forwarder.so /path/to/runtime-mock.so\n", argv[0]);
    return 2;
  }

  const int first_status = run_load_cycle(argv[1], argv[2]);
  if (first_status != 0) {
    return first_status;
  }
  return run_load_cycle(argv[1], argv[2]);
}

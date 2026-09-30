// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hip_runtime_dlopen_test.cpp
/// @brief Exercises HIP initialization when the runtime has no link-time owner.

#include <dlfcn.h>

#include <cstdio>
#include <cstring>

namespace {

using HipInit = int (*)(unsigned int);

#if defined(RJ_TEST_ASAN_HSA_RUNTIME_FORWARDER)
extern "C" void rj_test_keep_asan_hsa_runtime_forwarder();
extern "C" int rj_test_set_asan_hsa_runtime_handle(void *handle);

// HIP retains its HSA kernarg pools until process-lifetime teardown, which can
// run after LeakSanitizer's exit check for this dlopen-only runtime. Restrict
// the suppression to allocations made through the HSA pool allocator; ordinary
// client allocations reached through runtime callbacks remain visible.
extern "C" __attribute__((visibility("default"))) const char *__lsan_default_suppressions() {
  return "leak:hsa_amd_memory_pool_allocate\n";
}
#endif

template <typename Function> Function resolve(void *library, const char *name) {
  static_assert(sizeof(Function) == sizeof(void *));
  void *symbol = dlsym(library, name);
  Function function = nullptr;
  std::memcpy(&function, &symbol, sizeof(function));
  return function;
}

int initialize_and_close_runtime(const char *hip_path) {
  void *hip = dlopen(hip_path, RTLD_NOW | RTLD_LOCAL);
  if (hip == nullptr) {
    std::fprintf(stderr, "dlopen(%s) failed: %s\n", hip_path, dlerror());
    return 3;
  }

#if defined(RJ_TEST_ASAN_HSA_RUNTIME_FORWARDER)
  constexpr const char *kHsaRuntimeSoname = "libhsa-runtime64.so.1";
  void *hsa = dlopen(kHsaRuntimeSoname, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
  if (hsa == nullptr) {
    std::fprintf(stderr, "the dynamic HIP runtime did not load %s: %s\n", kHsaRuntimeSoname,
                 dlerror());
    dlclose(hip);
    return 4;
  }
  if (rj_test_set_asan_hsa_runtime_handle(hsa) == 0) {
    std::fprintf(stderr, "failed to retain the loaded HSA runtime lookup handle\n");
    dlclose(hsa);
    dlclose(hip);
    return 5;
  }
  if (dlclose(hsa) != 0) {
    std::fprintf(stderr, "dlclose(HSA runtime probe) failed: %s\n", dlerror());
    rj_test_set_asan_hsa_runtime_handle(nullptr);
    dlclose(hip);
    return 6;
  }
#endif

  int result = 0;
  const HipInit hip_init = resolve<HipInit>(hip, "hipInit");
  if (hip_init == nullptr) {
    std::fprintf(stderr, "dlsym(hipInit) failed: %s\n", dlerror());
    result = 7;
  } else {
    const int status = hip_init(0);
    if (status != 0) {
      std::fprintf(stderr, "hipInit failed: %d\n", status);
      result = 8;
    }
  }

  // Keep ROCr available to callbacks made by HIP's unload-time cleanup. The
  // forwarder owns a separate ROCr reference, so it can be cleared afterward.
  if (dlclose(hip) != 0 && result == 0) {
    std::fprintf(stderr, "dlclose(libamdhip64) failed: %s\n", dlerror());
    result = 9;
  }
#if defined(RJ_TEST_ASAN_HSA_RUNTIME_FORWARDER)
  if (rj_test_set_asan_hsa_runtime_handle(nullptr) == 0 && result == 0) {
    std::fprintf(stderr, "failed to clear the HSA runtime lookup handle\n");
    result = 10;
  }
#endif
  return result;
}

int check_tool_is_resident(const char *path) {
  void *tool = dlopen(path, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
  if (tool == nullptr) {
    std::fprintf(stderr, "HSA tool was unloaded with the dynamic HIP runtime: %s\n", dlerror());
    return 11;
  }
  if (dlclose(tool) != 0) {
    std::fprintf(stderr, "dlclose(HSA tool probe) failed: %s\n", dlerror());
    return 12;
  }
  return 0;
}

} // namespace

int main(int argc, char **argv) {
#if defined(RJ_TEST_ASAN_HSA_RUNTIME_FORWARDER)
  rj_test_keep_asan_hsa_runtime_forwarder();
#endif

  if (argc != 3) {
    std::fprintf(stderr, "usage: %s /path/to/libamdhip64.so /path/to/hsa-tool.so\n", argv[0]);
    return 2;
  }

  const int runtime_status = initialize_and_close_runtime(argv[1]);
  if (runtime_status != 0) {
    return runtime_status;
  }
  return check_tool_is_resident(argv[2]);
}

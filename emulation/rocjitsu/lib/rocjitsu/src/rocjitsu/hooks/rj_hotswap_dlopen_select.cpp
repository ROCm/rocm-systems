// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_hotswap_dlopen_select.cpp
/// @brief Redirect only ROCr's explicit HotSwap COMGR dlopen to rocjitsu.

#include "rocjitsu/base/rj_compiler.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <string_view>

namespace {

using DlopenFn = void *(*)(const char *, int);

[[nodiscard]] const char *basename_of(const char *path) {
  if (path == nullptr)
    return nullptr;
  const char *slash = std::strrchr(path, '/');
  return slash == nullptr ? path : slash + 1;
}

[[nodiscard]] bool is_comgr_request(const char *path) {
  const char *name = basename_of(path);
  return name != nullptr &&
         (std::strcmp(name, "libamd_comgr.so.3") == 0 ||
          std::strcmp(name, "libamd_comgr.so") == 0);
}

[[nodiscard]] bool caller_is_hsa_runtime(void *return_address) {
  Dl_info info{};
  if (return_address == nullptr || ::dladdr(return_address, &info) == 0 || info.dli_fname == nullptr)
    return false;
  const char *name = basename_of(info.dli_fname);
  return name != nullptr && std::string_view(name).starts_with("libhsa-runtime64.so");
}

[[nodiscard]] DlopenFn real_dlopen() {
  static DlopenFn function = reinterpret_cast<DlopenFn>(::dlsym(RTLD_NEXT, "dlopen"));
  return function;
}

} // namespace

extern "C" RJ_INTERPOSER_EXPORT __attribute__((noinline)) void *dlopen(const char *filename,
                                                                        int flags) {
  DlopenFn next = real_dlopen();
  if (next == nullptr)
    return nullptr;

  void *caller = __builtin_extract_return_addr(__builtin_return_address(0));
  if (is_comgr_request(filename) && caller_is_hsa_runtime(caller)) {
    const char *adapter = std::getenv("ROCJITSU_HOTSWAP_COMGR");
    // Require an absolute path so the selector cannot recurse through loader
    // search paths or accidentally select an unrelated library.
    if (adapter != nullptr && adapter[0] == '/') {
      const char *verbose = std::getenv("HSA_HOTSWAP_VERBOSE");
      const bool log = verbose != nullptr && verbose[0] != '\0' && std::strcmp(verbose, "0") != 0;
      if (log)
        std::fprintf(stderr, "[rocjitsu-selector] redirecting %s to %s\n", filename, adapter);
      void *handle = next(adapter, flags);
      if (log) {
        void *rewrite = handle == nullptr ? nullptr : ::dlsym(handle, "amd_comgr_hotswap_rewrite");
        Dl_info resolved{};
        if (rewrite != nullptr && ::dladdr(rewrite, &resolved) != 0 && resolved.dli_fname != nullptr) {
          std::fprintf(stderr,
                       "[rocjitsu-selector] HotSwap rewrite symbol on returned handle resolves to %s\n",
                       resolved.dli_fname);
        } else {
          std::fprintf(stderr,
                       "[rocjitsu-selector] redirected handle does not resolve a HotSwap rewrite symbol\n");
        }
      }
      return handle;
    }
  }
  return next(filename, flags);
}

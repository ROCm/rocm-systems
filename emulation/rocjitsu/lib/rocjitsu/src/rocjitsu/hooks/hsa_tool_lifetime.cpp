// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/hooks/hsa_tool_lifetime.h"

#include <cstdio>

#if defined(__linux__)
#include <dlfcn.h>
#endif

namespace rocjitsu::hooks {
namespace {

#if defined(__linux__)
int dso_anchor;

void *retain_calling_dso() noexcept {
  Dl_info info{};
  if (dladdr(&dso_anchor, &info) == 0 || info.dli_fname == nullptr) {
    std::fputs("[rocjitsu-hooks] failed to resolve the HSA tool DSO\n", stderr);
    return nullptr;
  }

  // Intentionally retain this handle. ROCR calls OnUnload before closing the
  // tool, so API wrappers and runtime-owned allocations are still cleaned up.
  // Keeping the code mapped avoids a nested-dlclose teardown hazard and also
  // keeps a later hsa_init()/hsa_shut_down() cycle safe in the same process.
  void *handle = dlopen(info.dli_fname, RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    std::fprintf(stderr, "[rocjitsu-hooks] failed to retain HSA tool DSO %s: %s\n", info.dli_fname,
                 dlerror());
  }
  return handle;
}
#endif

} // namespace

bool retain_hsa_tool_dso() noexcept {
#if defined(__linux__)
  static void *const retained_handle = retain_calling_dso();
  return retained_handle != nullptr;
#else
  return true;
#endif
}

} // namespace rocjitsu::hooks

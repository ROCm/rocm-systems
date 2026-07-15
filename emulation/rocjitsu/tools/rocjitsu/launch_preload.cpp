// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "launch_preload.h"

#include <cstdlib>

#ifndef __has_feature
#define __has_feature(x) 0
#endif

#if defined(__SANITIZE_ADDRESS__) || __has_feature(address_sanitizer)
#define RJ_BUILT_WITH_ASAN 1

#include <dlfcn.h>
#include <filesystem>
#include <system_error>

extern "C" void __asan_init();
#endif

namespace rocjitsu::cli {
namespace {

#if defined(RJ_BUILT_WITH_ASAN)
std::string canonical_existing_path(const std::filesystem::path &candidate) {
  std::error_code ec;
  if (!std::filesystem::exists(candidate, ec) || ec)
    return {};

  std::filesystem::path canonical = std::filesystem::canonical(candidate, ec);
  return ec ? std::string{} : canonical.string();
}
#endif

void prepend_env_path(const char *name, const std::string &value) {
  if (const char *old_value = std::getenv(name); old_value && *old_value) {
    std::string combined = value + ":" + old_value;
    setenv(name, combined.c_str(), 1);
    return;
  }
  setenv(name, value.c_str(), 1);
}

} // namespace

std::string find_loaded_asan_runtime() {
#if defined(RJ_BUILT_WITH_ASAN)
  Dl_info info{};
  if (!dladdr(reinterpret_cast<void *>(&__asan_init), &info) || !info.dli_fname)
    return {};

  const std::filesystem::path runtime_path(info.dli_fname);
  if (runtime_path.filename().string().find("asan") == std::string::npos)
    return {};
  return canonical_existing_path(runtime_path);
#else
  return {};
#endif
}

void prepend_launch_preloads(const std::string &interposer_path) {
  prepend_env_path("LD_PRELOAD", interposer_path);
  if (std::string asan_runtime = find_loaded_asan_runtime(); !asan_runtime.empty())
    prepend_env_path("LD_PRELOAD", asan_runtime);
}

} // namespace rocjitsu::cli

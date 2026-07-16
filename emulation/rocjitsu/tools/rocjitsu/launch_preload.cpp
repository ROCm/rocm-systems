// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "launch_preload.h"

#include <cstdlib>

#ifndef __has_feature
#define __has_feature(x) 0
#endif

#if defined(__SANITIZE_ADDRESS__) || __has_feature(address_sanitizer)
#define RJ_BUILT_WITH_ASAN 1
#endif

#if defined(__SANITIZE_THREAD__) || __has_feature(thread_sanitizer)
#define RJ_BUILT_WITH_TSAN 1
#endif

#if defined(RJ_BUILT_WITH_ASAN) || defined(RJ_BUILT_WITH_TSAN)
#include <dlfcn.h>
#include <filesystem>
#include <string_view>
#include <system_error>
#endif

#if defined(RJ_BUILT_WITH_ASAN)
extern "C" void __asan_init();
#endif

#if defined(RJ_BUILT_WITH_TSAN)
extern "C" void __tsan_init();
#endif

namespace rocjitsu::cli {
namespace {

#if defined(RJ_BUILT_WITH_ASAN) || defined(RJ_BUILT_WITH_TSAN)
std::string canonical_existing_path(const std::filesystem::path &candidate) {
  std::error_code ec;
  if (!std::filesystem::exists(candidate, ec) || ec)
    return {};

  std::filesystem::path canonical = std::filesystem::canonical(candidate, ec);
  return ec ? std::string{} : canonical.string();
}

std::string current_executable_path() {
  std::error_code ec;
  std::filesystem::path exe = std::filesystem::read_symlink("/proc/self/exe", ec);
  return ec ? std::string{} : canonical_existing_path(exe);
}

std::string find_loaded_sanitizer_runtime(void *init_symbol, std::string_view runtime_name) {
  Dl_info info{};
  if (!dladdr(init_symbol, &info) || !info.dli_fname)
    return {};

  const std::filesystem::path runtime_path(info.dli_fname);
  if (runtime_path.filename().string().find(runtime_name) == std::string::npos)
    return {};

  std::string runtime = canonical_existing_path(runtime_path);
  if (runtime.empty() || runtime == current_executable_path())
    return {};
  return runtime;
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
  return find_loaded_sanitizer_runtime(reinterpret_cast<void *>(&__asan_init), "asan");
#else
  return {};
#endif
}

std::string find_loaded_tsan_runtime() {
#if defined(RJ_BUILT_WITH_TSAN)
  return find_loaded_sanitizer_runtime(reinterpret_cast<void *>(&__tsan_init), "tsan");
#else
  return {};
#endif
}

void prepend_launch_preloads(const std::string &interposer_path) {
  prepend_env_path("LD_PRELOAD", interposer_path);
  if (std::string asan_runtime = find_loaded_asan_runtime(); !asan_runtime.empty())
    prepend_env_path("LD_PRELOAD", asan_runtime);
  if (std::string tsan_runtime = find_loaded_tsan_runtime(); !tsan_runtime.empty())
    prepend_env_path("LD_PRELOAD", tsan_runtime);
}

} // namespace rocjitsu::cli

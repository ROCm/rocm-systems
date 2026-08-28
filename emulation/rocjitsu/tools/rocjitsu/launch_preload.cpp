// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "launch_preload.h"

#include <algorithm>
#include <string_view>
#include <unistd.h>
#include <utility>
#include <vector>

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
#include <system_error>
#endif

#if defined(RJ_BUILT_WITH_ASAN)
extern "C" void __asan_init();
#endif

#if defined(RJ_BUILT_WITH_TSAN)
extern "C" void __tsan_init();
#endif

extern "C" char **environ;

namespace rocjitsu::cli {
namespace {

bool env_name_matches(std::string_view entry, const std::string &name) {
  return entry.size() > name.size() && entry[name.size()] == '=' &&
         entry.substr(0, name.size()) == name;
}

std::vector<std::string>::iterator find_env_entry(std::vector<std::string> &entries,
                                                  const std::string &name) {
  return std::find_if(entries.begin(), entries.end(),
                      [&](const std::string &entry) { return env_name_matches(entry, name); });
}

std::vector<std::string>::const_iterator find_env_entry(const std::vector<std::string> &entries,
                                                        const std::string &name) {
  return std::find_if(entries.begin(), entries.end(),
                      [&](const std::string &entry) { return env_name_matches(entry, name); });
}

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
} // namespace

LaunchEnvironment::LaunchEnvironment() {
  for (char **entry = environ; entry && *entry; ++entry)
    entries_.emplace_back(*entry);
}

const char *LaunchEnvironment::get(const std::string &name) const {
  auto it = find_env_entry(entries_, name);
  if (it == entries_.end())
    return nullptr;
  return it->c_str() + name.size() + 1;
}

void LaunchEnvironment::set(const std::string &name, const std::string &value) {
  std::string entry = name + "=" + value;
  auto it = find_env_entry(entries_, name);
  if (it == entries_.end()) {
    entries_.push_back(std::move(entry));
  } else {
    *it = std::move(entry);
  }
  envp_dirty_ = true;
}

void LaunchEnvironment::prepend_path(const std::string &name, const std::string &value) {
  if (const char *old_value = get(name); old_value && *old_value) {
    set(name, value + ":" + old_value);
    return;
  }
  set(name, value);
}

char *const *LaunchEnvironment::envp() {
  if (envp_dirty_) {
    envp_.clear();
    envp_.reserve(entries_.size() + 1);
    for (std::string &entry : entries_)
      envp_.push_back(entry.data());
    envp_.push_back(nullptr);
    envp_dirty_ = false;
  }
  return envp_.data();
}

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

void prepend_launch_preloads(LaunchEnvironment &environment, const std::string &interposer_path) {
  environment.prepend_path("LD_PRELOAD", interposer_path);
  if (std::string asan_runtime = find_loaded_asan_runtime(); !asan_runtime.empty())
    environment.prepend_path("LD_PRELOAD", asan_runtime);
  if (std::string tsan_runtime = find_loaded_tsan_runtime(); !tsan_runtime.empty())
    environment.prepend_path("LD_PRELOAD", tsan_runtime);
}

int execvp_with_environment(const char *file, char *const argv[], LaunchEnvironment &environment) {
  return execvpe(file, argv, environment.envp());
}

} // namespace rocjitsu::cli

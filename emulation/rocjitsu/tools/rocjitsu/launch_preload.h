// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>

namespace rocjitsu::cli {

class LaunchEnvironment {
public:
  LaunchEnvironment();
  LaunchEnvironment(const LaunchEnvironment &) = delete;
  LaunchEnvironment &operator=(const LaunchEnvironment &) = delete;

  const char *get(const std::string &name) const;
  void set(const std::string &name, const std::string &value);
  void prepend_path(const std::string &name, const std::string &value);
  char *const *envp();

private:
  std::vector<std::string> entries_;
  std::vector<char *> envp_;
  bool envp_dirty_ = true;
};

std::string find_loaded_asan_runtime();

std::string find_loaded_tsan_runtime();

void prepend_launch_preloads(LaunchEnvironment &environment, const std::string &interposer_path);

int execvp_with_environment(const char *file, char *const argv[], LaunchEnvironment &environment);

} // namespace rocjitsu::cli

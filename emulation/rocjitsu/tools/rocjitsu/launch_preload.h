// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>

namespace rocjitsu::launcher {

enum class ParentEnvironment { Inherit, Omit };

class LaunchEnvironment {
public:
  // Defaults to inheriting the parent process environment. Use
  // ParentEnvironment::Omit for a clean launch environment.
  explicit LaunchEnvironment(ParentEnvironment parent_environment = ParentEnvironment::Inherit);
  LaunchEnvironment(const LaunchEnvironment &) = delete;
  LaunchEnvironment &operator=(const LaunchEnvironment &) = delete;

  const char *get(const std::string &name) const;
  void set(const std::string &name, const std::string &value);
  void prepend_path(const std::string &name, const std::string &value);
  char **envp();

private:
  std::vector<std::string> entries_;
  std::vector<char *> envp_;
  bool envp_dirty_ = true;
};

std::string find_loaded_asan_runtime();

std::string find_loaded_tsan_runtime();

void prepend_launch_preloads(LaunchEnvironment &environment, const std::string &interposer_path);

void configure_dbt_guest_tool_environment(LaunchEnvironment &environment,
                                          const std::string &hooks_path);

int execvp_with_environment(const char *file, char *const argv[], LaunchEnvironment &environment);

} // namespace rocjitsu::launcher

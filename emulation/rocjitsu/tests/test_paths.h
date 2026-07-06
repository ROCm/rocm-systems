// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <string>

namespace rocjitsu::test {

inline std::string config_path(const char *name) {
  if (const char *dir = std::getenv("ROCJITSU_CONFIG_DIR"))
    return std::string(dir) + "/" + name;
#ifdef CONFIG_DIR
  return std::string(CONFIG_DIR) + "/" + name;
#else
  return name;
#endif
}

inline std::string kernel_path(const char *name) {
  if (const char *dir = std::getenv("ROCJITSU_KERNEL_DIR"))
    return std::string(dir) + "/" + name + ".o";
#ifdef KERNEL_DIR
  return std::string(KERNEL_DIR) + "/" + name + ".o";
#else
  return std::string(name) + ".o";
#endif
}

inline std::string kernel_hsaco_path(const char *name) {
  if (const char *dir = std::getenv("ROCJITSU_KERNEL_DIR"))
    return std::string(dir) + "/" + name + ".hsaco";
#ifdef KERNEL_DIR
  return std::string(KERNEL_DIR) + "/" + name + ".hsaco";
#else
  return std::string(name) + ".hsaco";
#endif
}

} // namespace rocjitsu::test

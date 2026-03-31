// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace test_utils {

// Get path to test assets directory
//
// Reads ROCM_KPACK_TEST_ASSETS_DIR environment variable set by CMake.
// Tests must be run via CTest for this to be available.
//
// Throws:
//   std::runtime_error if env var not set or directory doesn't exist
inline std::filesystem::path get_test_assets_dir() {
  const char* env_path = std::getenv("ROCM_KPACK_TEST_ASSETS_DIR");
  if (!env_path) {
    throw std::runtime_error(
        "ROCM_KPACK_TEST_ASSETS_DIR environment variable not set. "
        "Tests must be run via CTest.");
  }

  std::filesystem::path assets_dir(env_path);
  if (!std::filesystem::exists(assets_dir)) {
    throw std::runtime_error("Test assets directory does not exist: " +
                             assets_dir.string());
  }

  return assets_dir;
}

// Get path to a test asset by relative path
//
// Args:
//   relative_path: Path relative to test_assets/
//
// Returns:
//   Full path to the asset
//
// Throws:
//   std::runtime_error if the asset does not exist
//
// Example:
//   auto exe =
//   get_test_asset("bundled_binaries/linux/cov5/test_kernel_single.exe"); auto
//   co = get_test_asset("ccob/ccob_gfx942_sample1.co");
inline std::filesystem::path get_test_asset(const std::string& relative_path) {
  std::filesystem::path full_path = get_test_assets_dir() / relative_path;
  if (!std::filesystem::exists(full_path)) {
    throw std::runtime_error("Test asset does not exist: " +
                             full_path.string());
  }
  return full_path;
}

}  // namespace test_utils

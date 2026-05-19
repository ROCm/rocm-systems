/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: NCSA
 */

#ifndef ROCRTST_COMMON_ENV_CONFIG_H_
#define ROCRTST_COMMON_ENV_CONFIG_H_

#include <string>
#include <cstddef>

namespace rocrtst {

/// \brief Centralized manager for all rocrtst environment variables
///
/// This singleton class reads all environment variables used by rocrtst
/// at initialization time and provides accessor methods for their values.
/// This eliminates scattered getenv() calls throughout the codebase.
class EnvironmentConfig {
 public:
  /// Get the singleton instance
  static EnvironmentConfig& getInstance();

  /// Initialize the environment configuration (reads all env vars)
  /// Should be called early in main() before other components initialize
  void initialize();

  /// Check if already initialized
  bool isInitialized() const { return initialized_; }

  // Platform-related environment variables

  /// Get HSA_MODEL_TOPOLOGY value (for FFM detection)
  /// Returns empty string if not set
  const std::string& getHsaModelTopology() const {
    return hsa_model_topology_;
  }

  /// Check if HSA_MODEL_TOPOLOGY is set (FFM environment)
  bool hasHsaModelTopology() const {
    return !hsa_model_topology_.empty();
  }

  /// Get ROCRTST_PLATFORM_OVERRIDE value
  /// Returns empty string if not set
  const std::string& getPlatformOverride() const {
    return platform_override_;
  }

  /// Check if ROCRTST_PLATFORM_OVERRIDE is set
  bool hasPlatformOverride() const {
    return !platform_override_.empty();
  }

  /// Get ROCRTST_PLATFORM_CONFIG value (custom config file path)
  /// Returns empty string if not set
  const std::string& getPlatformConfigPath() const {
    return platform_config_path_;
  }

  /// Check if ROCRTST_PLATFORM_CONFIG is set
  bool hasPlatformConfigPath() const {
    return !platform_config_path_.empty();
  }

  /// Get ROCRTST_TEST_GROUPS value (comma-separated group names)
  /// Returns empty string if not set
  const std::string& getTestGroups() const {
    return test_groups_;
  }

  /// Check if ROCRTST_TEST_GROUPS is set
  bool hasTestGroups() const {
    return !test_groups_.empty();
  }

  // Memory-related environment variables

  /// Get ROCRTST_LIMIT_POOL_SIZE value
  /// Returns 0 if not set or invalid
  size_t getPoolSizeLimit() const {
    return pool_size_limit_;
  }

  /// Check if ROCRTST_LIMIT_POOL_SIZE is set
  bool hasPoolSizeLimit() const {
    return pool_size_limit_ > 0;
  }

 private:
  EnvironmentConfig();
  ~EnvironmentConfig() = default;

  // Prevent copying
  EnvironmentConfig(const EnvironmentConfig&) = delete;
  EnvironmentConfig& operator=(const EnvironmentConfig&) = delete;

  /// Read a string environment variable
  static std::string readEnvString(const char* name);

  /// Read a size_t environment variable
  static size_t readEnvSize(const char* name);

  bool initialized_;

  // Platform-related
  std::string hsa_model_topology_;
  std::string platform_override_;
  std::string platform_config_path_;
  std::string test_groups_;

  // Memory-related
  size_t pool_size_limit_;
};

}  // namespace rocrtst

#endif  // ROCRTST_COMMON_ENV_CONFIG_H_

/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: NCSA
 */

/// \file
/// Platform-aware test filtering for rocrtst

#ifndef ROCRTST_COMMON_PLATFORM_FILTER_H_
#define ROCRTST_COMMON_PLATFORM_FILTER_H_

#include <string>
#include <vector>
#include <map>
#include <set>
#include "common/common.h"

namespace rocrtst {

/// Configuration for a specific platform
struct PlatformConfig {
  std::vector<std::string> allowed_groups;
  std::vector<std::string> allowed_tests;
  std::vector<std::string> blocked_tests;
};

/// Manages test filtering based on platform type and test groups
class TestFilterManager {
public:
  /// Get singleton instance
  static TestFilterManager& getInstance();

  /// Initialize the filter manager with configuration
  /// \param configPath Path to YAML configuration file
  /// \returns true if initialization succeeded
  bool initialize(const std::string& configPath = "");

  /// Check if a test should run on the current platform
  /// \param testName Full test name
  ///   (e.g., "rocrtstFunc.MemoryAccessTests")
  /// \param skipReason Optional output for skip reason (nullptr to ignore)
  /// \returns true if the test should run
  bool shouldRunTest(const std::string& testName,
                     std::string* skipReason = nullptr);

  /// Get the detected platform type
  /// \returns Current platform type
  PlatformType getPlatform() const { return currentPlatform_; }

  /// Get the configuration file path being used
  /// \returns Path to configuration file
  std::string getConfigPath() const { return configPath_; }

  /// Get count of tests in active groups
  /// \returns Number of tests that should run based on group filtering
  size_t getActiveTestCount() const;

  /// Get list of active group names
  /// \returns Vector of active group names
  std::vector<std::string> getActiveGroups() const;

private:
  TestFilterManager();
  ~TestFilterManager() = default;

  // Prevent copying
  TestFilterManager(const TestFilterManager&) = delete;
  TestFilterManager& operator=(const TestFilterManager&) = delete;

  /// Load configuration from YAML file
  bool loadConfiguration(const std::string& path);

  /// Parse active groups from environment variable
  void parseActiveGroups();

  /// Find all groups a test belongs to
  std::vector<std::string> findTestGroups(
      const std::string& testName) const;

  /// Check if a group is in the allowed list
  bool isGroupAllowed(const std::string& groupName) const;

  PlatformType currentPlatform_;
  std::string configPath_;
  std::map<std::string, std::vector<std::string>> groups_;  // group name -> test list
  std::map<PlatformType, PlatformConfig> platformConfigs_;
  std::vector<std::string> activeGroups_;  // Empty = all groups
  bool initialized_;
};

}  // namespace rocrtst

#endif  // ROCRTST_COMMON_PLATFORM_FILTER_H_

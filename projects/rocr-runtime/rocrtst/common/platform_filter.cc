/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: NCSA
 */

/// \file
/// Implementation of platform-aware test filtering

#include "common/platform_filter.h"
#include "common/env_config.h"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

namespace rocrtst {

// Inline function to access environment configuration (safe from static init order fiasco)
static inline EnvironmentConfig& env_() {
  return EnvironmentConfig::getInstance();
}

TestFilterManager::TestFilterManager()
    : currentPlatform_(PlatformType::UNKNOWN),
      initialized_(false) {
}

TestFilterManager& TestFilterManager::getInstance() {
  static TestFilterManager instance;
  return instance;
}

bool TestFilterManager::initialize(const std::string& configPath) {
  if (initialized_) {
    return true;
  }

  // Detect platform
  currentPlatform_ = PlatformDetector::detectPlatform();

  configPath_ = configPath;

  // Load configuration
  if (!loadConfiguration(configPath_)) {
    std::cerr << "Warning: Failed to load platform config from: "
              << configPath_ << '\n';
    std::cerr << "All tests will be allowed to run.\n";
    initialized_ = true;
    return false;
  }

  // Parse active groups from environment
  parseActiveGroups();

  initialized_ = true;
  return true;
}

bool TestFilterManager::loadConfiguration(const std::string& path) {
  try {
    YAML::Node config = YAML::LoadFile(path);

    // Load test groups
    if (config["groups"]) {
      for (const auto& group : config["groups"]) {
        std::string groupName = group.first.as<std::string>();
        std::vector<std::string> tests;

        for (const auto& test : group.second) {
          tests.push_back(test.as<std::string>());
        }

        groups_[groupName] = tests;
      }
    }

    // Load platform configurations
    if (config["platforms"]) {
      for (const auto& platform : config["platforms"]) {
        std::string platformName = platform.first.as<std::string>();
        PlatformType platformType;

        // Map platform name to enum
        if (platformName == "REAL_HARDWARE") {
          platformType = PlatformType::REAL_HARDWARE;
        } else if (platformName == "EMULATOR") {
          platformType = PlatformType::EMULATOR;
        } else if (platformName == "FFM_SIMULATOR") {
          platformType = PlatformType::FFM_SIMULATOR;
        } else {
          continue;  // Skip unknown platforms
        }

        PlatformConfig pconfig;

        // Load allowed groups
        if (platform.second["allowed_groups"]) {
          for (const auto& group : platform.second["allowed_groups"]) {
            pconfig.allowed_groups.push_back(group.as<std::string>());
          }
        }

        // Load allowed tests
        if (platform.second["allowed_tests"]) {
          for (const auto& test : platform.second["allowed_tests"]) {
            pconfig.allowed_tests.push_back(test.as<std::string>());
          }
        }

        // Load blocked tests
        if (platform.second["blocked_tests"]) {
          for (const auto& test : platform.second["blocked_tests"]) {
            pconfig.blocked_tests.push_back(test.as<std::string>());
          }
        }

        platformConfigs_[platformType] = pconfig;
      }
    }

    return true;
  } catch (const YAML::Exception& e) {
    std::cerr << "YAML parsing error: " << e.what() << "\n";
    return false;
  } catch (const std::exception& e) {
    std::cerr << "Error loading configuration: " << e.what() << "\n";
    return false;
  }
}

void TestFilterManager::parseActiveGroups() {
  activeGroups_.clear();

  if (!env_().hasTestGroups()) {
    return;  // Empty means all groups
  }

  const std::string& groupsStr = env_().getTestGroups();
  std::stringstream ss(groupsStr);
  std::string group;

  while (std::getline(ss, group, ',')) {
    // Trim whitespace
    group.erase(0, group.find_first_not_of(" \t"));
    group.erase(group.find_last_not_of(" \t") + 1);

    if (!group.empty()) {
      activeGroups_.push_back(group);
    }
  }
}

std::vector<std::string> TestFilterManager::findTestGroups(
                                          const std::string& testName) const {
  std::vector<std::string> result;
  for (const auto& group : groups_) {
    const auto& tests = group.second;
    if (std::find(tests.begin(), tests.end(), testName) != tests.end()) {
      result.push_back(group.first);
    }
  }
  return result;
}

bool TestFilterManager::isGroupAllowed(const std::string& groupName) const {
  if (groupName.empty()) {
    return true;  // Tests not in any group are allowed
  }

  auto it = platformConfigs_.find(currentPlatform_);
  if (it == platformConfigs_.end()) {
    return true;  // No config for this platform, allow all
  }

  const PlatformConfig& pconfig = it->second;

  // Check if "*" is in allowed groups (means all groups)
  if (std::find(pconfig.allowed_groups.begin(),
                pconfig.allowed_groups.end(), "*") !=
                                              pconfig.allowed_groups.end()) {
    return true;
  }

  // Check if this specific group is allowed
  return std::find(pconfig.allowed_groups.begin(),
                   pconfig.allowed_groups.end(),
                   groupName) != pconfig.allowed_groups.end();
}

bool TestFilterManager::shouldRunTest(const std::string& testName,
                                      std::string* skipReason) {
  if (!initialized_) {
    initialize();
  }

  // Helper to set skip reason and return false
  auto skip = [skipReason](const std::string& reason) {
    if (skipReason != nullptr) {
      *skipReason = reason;
    }
    return false;
  };

  // If no configuration loaded, allow all tests
  if (platformConfigs_.empty()) {
    return true;
  }

  // Get platform configuration
  auto it = platformConfigs_.find(currentPlatform_);
  if (it == platformConfigs_.end()) {
    return true;  // No config for this platform, allow all
  }

  const PlatformConfig& pconfig = it->second;

  // 1. Check explicit block list (exclusion overrides inclusion)
  if (std::find(pconfig.blocked_tests.begin(),
                pconfig.blocked_tests.end(),
                testName) != pconfig.blocked_tests.end()) {
    return skip("Test in blocked list for " +
                std::string(PlatformDetector::platformName(
                    currentPlatform_)));
  }

  // 2. Find all groups this test belongs to
  std::vector<std::string> testGroups = findTestGroups(testName);

  // 3. Check active groups filter (if specified)
  if (!activeGroups_.empty()) {
    if (testGroups.empty()) {
      return skip("Test not in any group (group filter active)");
    }
    bool inActiveGroup = false;
    for (const auto& group : testGroups) {
      if (std::find(activeGroups_.begin(),
                    activeGroups_.end(),
                    group) != activeGroups_.end()) {
        inActiveGroup = true;
        break;
      }
    }
    if (!inActiveGroup) {
      return skip("Test groups not in active filter");
    }
  }

  // 4. Check group-based filtering
  if (testGroups.empty()) {
    // Ungrouped test - check if platform has specific allowed groups
    if (!pconfig.allowed_groups.empty() &&
        std::find(pconfig.allowed_groups.begin(),
                  pconfig.allowed_groups.end(), "*") ==
                  pconfig.allowed_groups.end()) {
      // Platform specifies allowed groups (not wildcard)
      // Ungrouped tests are not in any allowed group, so skip
      return skip("Test not in any group (platform allows only specific groups)");
    }
    // Platform has wildcard or no group restrictions - allow ungrouped test
  } else {
    // Grouped test - check if ANY of test's groups are allowed
    bool anyGroupAllowed = false;
    for (const auto& group : testGroups) {
      if (isGroupAllowed(group)) {
        anyGroupAllowed = true;
        break;
      }
    }
    if (!anyGroupAllowed) {
      std::string groupList;
      for (size_t i = 0; i < testGroups.size(); ++i) {
        groupList += testGroups[i];
        if (i < testGroups.size() - 1) groupList += ", ";
      }
      return skip("Test groups [" + groupList + "] not allowed on " +
                  std::string(PlatformDetector::platformName(
                      currentPlatform_)));
    }
  }

  // 5. Check explicit allow list
  if (!pconfig.allowed_tests.empty()) {
    if (std::find(pconfig.allowed_tests.begin(),
                  pconfig.allowed_tests.end(), "*") !=
                                               pconfig.allowed_tests.end()) {
      return true;  // "*" means all tests allowed
    }
    if (std::find(pconfig.allowed_tests.begin(),
                  pconfig.allowed_tests.end(),
                  testName) == pconfig.allowed_tests.end()) {
      return skip("Test not in allowed list for " +
                  std::string(PlatformDetector::platformName(
                      currentPlatform_)));
    }
  }

  // Default: allow the test
  return true;
}

size_t TestFilterManager::getActiveTestCount() const {
  size_t count = 0;

  if (activeGroups_.empty()) {
    // Count all tests in allowed groups for current platform
    auto it = platformConfigs_.find(currentPlatform_);
    if (it == platformConfigs_.end()) {
      // No config, count all tests
      for (const auto& group : groups_) {
        count += group.second.size();
      }
    } else {
      const PlatformConfig& pconfig = it->second;
      for (const auto& group : groups_) {
        if (isGroupAllowed(group.first)) {
          count += group.second.size();
        }
      }
    }
  } else {
    // Count tests in active groups
    for (const auto& groupName : activeGroups_) {
      auto it = groups_.find(groupName);
      if (it != groups_.end()) {
        count += it->second.size();
      }
    }
  }

  return count;
}

std::vector<std::string> TestFilterManager::getActiveGroups() const {
  if (!activeGroups_.empty()) {
    return activeGroups_;
  }

  // Return all groups that are allowed on current platform
  std::vector<std::string> result;
  for (const auto& group : groups_) {
    if (isGroupAllowed(group.first)) {
      result.push_back(group.first);
    }
  }

  return result;
}

}  // namespace rocrtst

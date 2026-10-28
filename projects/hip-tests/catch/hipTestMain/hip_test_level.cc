/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_level.hh>
#include <hip_test_parameters.hh>

#include <algorithm>

namespace HipTestLevel {

bool isSupportedLevel(const std::string& level) {
  for (const char* supported : kSupportedLevels) {
    if (level == supported) {
      return true;
    }
  }
  return false;
}

LevelResolution resolveLevel(const char* envLevel) {
  LevelResolution resolution;

  // An empty value is treated as unset, so CI scripts that export an empty
  // variable keep the default.
  if (envLevel == nullptr || *envLevel == '\0') {
    return resolution;
  }

  if (isSupportedLevel(envLevel)) {
    resolution.level = envLevel;
    resolution.source = LevelSource::kEnvironment;
    return resolution;
  }

  // A test category stands for the highest level it covers. Built once; the
  // generated table is fixed for the life of the process.
  static const auto categories = TestCategories::initializeTestCategories();
  const auto found = categories.find(envLevel);
  if (found != categories.end() && !found->second.empty()) {
    const std::string level =
        "level_" + std::to_string(*std::max_element(found->second.begin(), found->second.end()));
    if (isSupportedLevel(level)) {
      resolution.level = level;
      resolution.source = LevelSource::kEnvironment;
      resolution.category = envLevel;
      return resolution;
    }
  }

  // Construct error message
  std::string expected;
  for (const char* supported : kSupportedLevels) {
    if (!expected.empty()) {
      expected += ", ";
    }
    expected += supported;
  }
  std::string categoryNames;
  for (const auto& entry : categories) {
    if (!categoryNames.empty()) {
      categoryNames += ", ";
    }
    categoryNames += entry.first;
  }
  resolution.error = std::string("HIP_TEST_LEVEL='") + envLevel +
                     "' is not a supported level; expected exactly one of " + expected;
  if (!categoryNames.empty()) {
    resolution.error += ", or a test category: " + categoryNames;
  }
  return resolution;
}

std::string LevelResolution::describe() const {
  const char* sourceName = source == LevelSource::kEnvironment ? "HIP_TEST_LEVEL" : "default";
  std::string line = "[Test Level] Parameters from " + std::string(sourceName) + ": " + level;
  if (!category.empty()) {
    line += " (category " + category + ")";
  }
  return line;
}

}  // namespace HipTestLevel

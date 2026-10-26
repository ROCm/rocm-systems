/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_level.hh>

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

  if (!isSupportedLevel(envLevel)) {
    std::string expected;
    for (const char* supported : kSupportedLevels) {
      if (!expected.empty()) {
        expected += ", ";
      }
      expected += supported;
    }
    resolution.error = std::string("HIP_TEST_LEVEL='") + envLevel +
                       "' is not a supported level; expected exactly one of " + expected;
    return resolution;
  }

  resolution.level = envLevel;
  resolution.source = LevelSource::kEnvironment;
  return resolution;
}

std::string LevelResolution::describe() const {
  const char* sourceName = source == LevelSource::kEnvironment ? "HIP_TEST_LEVEL" : "default";
  return "[Test Level] Parameters from " + std::string(sourceName) + ": " + level;
}

}  // namespace HipTestLevel

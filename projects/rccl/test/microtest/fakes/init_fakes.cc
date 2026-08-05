/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the init-only fake seams. See init_fakes.h. The macro
// `getenv` is NOT active in this translation unit, so micro_getenv() can call
// the real libc getenv() as its default.

#include "init_fakes.h"

#include <cstdlib>
#include <string>
#include <unordered_map>

namespace {
// Scripted environment overrides. When a name is present, its value (which may
// be an explicit "absent" -> nullptr) is returned; otherwise fall through to
// the real getenv so unrelated reads keep working.
std::unordered_map<std::string, std::string>& microEnvMap() {
  static std::unordered_map<std::string, std::string> m;
  return m;
}
}  // namespace

const char* micro_getenv(const char* name) {
  if (name != nullptr) {
    auto& m = microEnvMap();
    auto it = m.find(name);
    if (it != m.end()) {
      return it->second.c_str();
    }
  }
  return std::getenv(name);
}

void SetMicroEnv(const char* name, const char* value) {
  if (name != nullptr && value != nullptr) {
    microEnvMap()[name] = value;
  }
}

void ClearMicroEnv() { microEnvMap().clear(); }

void ResetInitFakes() {
  ResetHipFakes();
  ResetNcclFakes();
  ClearMicroEnv();
}

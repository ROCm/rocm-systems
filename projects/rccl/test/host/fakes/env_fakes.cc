/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// See env_fakes.h. One env implementation for every microtest binary.

#include "env_fakes.h"

#include <dlfcn.h>

#include <cstdlib>
#include <optional>
#include <string>
#include <unordered_map>

#include "nccl.h"

namespace {
// A nullopt entry means "absent". Unmapped names read as unset via micro_getenv, real via the getenv interposer.
std::unordered_map<std::string, std::optional<std::string>>& microEnvMap() {
  static std::unordered_map<std::string, std::optional<std::string>> m;
  return m;
}
// Resolved past our interposing definition below so the map-miss fallback doesn't recurse into ourselves.
char* real_getenv(const char* name) {
  using Fn = char* (*)(const char*);
  static Fn next = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "getenv"));
  return next ? next(name) : nullptr;
}
}  // namespace

const char* micro_getenv(const char* name) {
  if (name == nullptr) return nullptr;
  auto& m = microEnvMap();
  auto it = m.find(name);
  return (it != m.end() && it->second) ? it->second->c_str() : nullptr;
}

// Link-level override rather than a scoped macro: production has bare std::getenv call sites, which a macro
// cannot catch. It is process-wide, so gtest/libstdc++ reads must still see the real environment -- unlike
// micro_getenv it falls back.
extern "C" char* getenv(const char* name) {
  if (name != nullptr) {
    auto& m = microEnvMap();
    auto it = m.find(name);
    if (it != m.end()) return it->second ? const_cast<char*>(it->second->c_str()) : nullptr;
  }
  return real_getenv(name);
}

void SetMicroEnv(const char* name, const char* value) {
  if (name == nullptr) return;
  if (value == nullptr) microEnvMap()[name] = std::nullopt;
  else microEnvMap()[name] = value;
}

void SetMicroEnvAbsent(const char* name) { SetMicroEnv(name, nullptr); }

void ClearMicroEnv() { microEnvMap().clear(); }

void ResetEnvFakes() { ClearMicroEnv(); }

const char* ncclGetEnv(const char* name) { return micro_getenv(name); }

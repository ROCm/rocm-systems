/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// See env_fakes.h. One env implementation for every microtest binary.

#include "env_fakes.h"

#include <dlfcn.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <unistd.h>

#include "nccl.h"
#include "param.h"

#include "signature-drift.h"

namespace {
struct MicroEnvironState {
  bool active = false;
  char** original = nullptr;
  std::vector<std::string> entries;
  std::vector<char*> pointers;
};

MicroEnvironState& microEnvironState() {
  static MicroEnvironState state;
  return state;
}

void PublishMicroEnviron() {
  auto& state = microEnvironState();
  state.pointers.clear();
  for (auto& entry : state.entries) state.pointers.push_back(entry.data());
  state.pointers.push_back(nullptr);
  environ = state.pointers.data();
}

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
  auto& state = microEnvironState();
  if (state.active) {
    const std::string prefix = std::string(name) + "=";
    std::vector<std::string> entries;
    bool replaced = false;
    for (const auto& entry : state.entries) {
      if (entry.compare(0, prefix.size(), prefix) != 0) {
        entries.push_back(entry);
      } else if (!replaced && value != nullptr) {
        entries.push_back(prefix + value);
        replaced = true;
      }
    }
    if (!replaced && value != nullptr) entries.push_back(prefix + value);
    state.entries = std::move(entries);
    PublishMicroEnviron();
  }
}

void SetMicroEnvAbsent(const char* name) { SetMicroEnv(name, nullptr); }

void SetMicroEnviron(const std::vector<std::string>& entries) {
  auto& state = microEnvironState();
  if (!state.active) state.original = environ;
  state.active = true;
  state.entries = entries;
  auto& values = microEnvMap();
  values.clear();
  for (const auto& entry : entries) {
    const auto separator = entry.find('=');
    if (separator != std::string::npos)
      values.emplace(entry.substr(0, separator), entry.substr(separator + 1));
  }
  PublishMicroEnviron();
}

void ClearMicroEnv() {
  auto& state = microEnvironState();
  if (state.active) {
    environ = state.original;
    state.active = false;
    state.original = nullptr;
    state.entries.clear();
    state.pointers.clear();
  }
  microEnvMap().clear();
}

void ResetEnvFakes() {
  ClearMicroEnv();
  g_ncclLoadParam = DefaultLoadParam;
}

const char* ncclGetEnv(const char* name) { return micro_getenv(name); }

int64_t DefaultLoadParam(char const* env, int64_t deftVal, int64_t /*uninitialized*/,
                         int64_t* cache, int8_t* noCache) {
  int64_t value = deftVal;

  const char* str = micro_getenv(env);
  if (str && std::strlen(str) > 0) {
    errno = 0;
    char* end = nullptr;
    const int64_t parsed = std::strtoll(str, &end, 0);
    // Matches production: a non-numeric value falls back to the default rather
    // than being reported as an error.
    if (errno == 0 && end != str) value = parsed;
  }

  if (cache) *cache = value;
  if (noCache) *noCache = 0;
  return value;
}

std::function<int64_t(char const*, int64_t, int64_t, int64_t*, int8_t*)> g_ncclLoadParam =
    DefaultLoadParam;
ASSERT_HOOK_MATCHES_PROD(g_ncclLoadParam, ncclLoadParam);

int64_t ncclLoadParam(char const* env, int64_t deftVal, int64_t uninitialized, int64_t* cache,
                      int8_t* noCache) {
  return g_ncclLoadParam(env, deftVal, uninitialized, cache, noCache);
}

// src/misc/param.cc:69. The real one reads /etc/nccl.conf into the environment; the microtests drive
// the environment through SetMicroEnv instead, so this is a no-op rather than a seam.
void initEnv() {}

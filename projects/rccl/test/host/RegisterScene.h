/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Helpers shared by the src/register/ test TUs.

#ifndef RCCL_TEST_HOST_REGISTER_SCENE_H_
#define RCCL_TEST_HOST_REGISTER_SCENE_H_

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

inline void* Ptr(uintptr_t addr) { return reinterpret_cast<void*>(addr); }

// Param values for the redirected NCCL_PARAM bodies (g_loadParam); unset names keep their defaults.
inline std::function<int64_t(const char*, int64_t)> ParamOverride(
  std::initializer_list<std::pair<const char*, int64_t>> kv) {
  std::vector<std::pair<std::string, int64_t>> table(kv.begin(), kv.end());
  return [table](const char* env, int64_t deft) {
    for (const auto& [name, value] : table) {
      if (name == env) return value;
    }
    return deft;
  };
}

#endif  // RCCL_TEST_HOST_REGISTER_SCENE_H_

/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <hsa/hsa.h>

#include <cstdint>
#include <functional>
#include <string>

namespace hip_impl {
inline void* address(hsa_executable_symbol_t x) {
  void* r = nullptr;
  hsa_executable_symbol_get_info(x, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS, &r);

  return r;
}

inline hsa_agent_t agent(hsa_executable_symbol_t x) {
  hsa_agent_t r = {};
  hsa_executable_symbol_get_info(x, HSA_EXECUTABLE_SYMBOL_INFO_AGENT, &r);

  return r;
}

inline std::uint32_t group_size(hsa_executable_symbol_t x) {
  std::uint32_t r = 0u;
  hsa_executable_symbol_get_info(x, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &r);

  return r;
}

inline hsa_isa_t isa(hsa_agent_t x) {
  hsa_isa_t r = {};
  hsa_agent_iterate_isas(
      x,
      [](hsa_isa_t i, void* o) {
        *static_cast<hsa_isa_t*>(o) = i;  // Pick the first.

        return HSA_STATUS_INFO_BREAK;
      },
      &r);

  return r;
}

inline std::uint64_t kernel_object(hsa_executable_symbol_t x) {
  std::uint64_t r = 0u;
  hsa_executable_symbol_get_info(x, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &r);

  return r;
}

inline std::string name(hsa_executable_symbol_t x) {
  std::uint32_t sz = 0u;
  hsa_executable_symbol_get_info(x, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &sz);

  std::string r(sz, '\0');
  hsa_executable_symbol_get_info(x, HSA_EXECUTABLE_SYMBOL_INFO_NAME, &r.front());

  return r;
}

inline std::uint32_t private_size(hsa_executable_symbol_t x) {
  std::uint32_t r = 0u;
  hsa_executable_symbol_get_info(x, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &r);

  return r;
}

inline std::uint32_t size(hsa_executable_symbol_t x) {
  std::uint32_t r = 0;
  hsa_executable_symbol_get_info(x, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE, &r);

  return r;
}

inline hsa_symbol_kind_t type(hsa_executable_symbol_t x) {
  hsa_symbol_kind_t r = {};
  hsa_executable_symbol_get_info(x, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &r);

  return r;
}
}  // namespace hip_impl
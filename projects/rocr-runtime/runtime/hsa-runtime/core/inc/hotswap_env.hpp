////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef HSA_RUNTIME_CORE_INC_HOTSWAP_ENV_HPP_
#define HSA_RUNTIME_CORE_INC_HOTSWAP_ENV_HPP_

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

namespace rocr {
namespace hotswap {

inline bool IsEnvFlagEnabled(const char* name) {
  const char* raw_value = std::getenv(name);
  if (!raw_value) return false;

  std::string value(raw_value);
  if (value.empty()) return false;

  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value != "0" && value != "off" && value != "false" &&
         value != "no" && value != "n" && value != "f";
}

}  // namespace hotswap
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_HOTSWAP_ENV_HPP_

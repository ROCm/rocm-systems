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

#include "core/util/os.h"

#include <algorithm>
#include <cctype>
#include <string>

namespace rocr {
namespace hotswap {

inline bool IsEnvFlagValueEnabled(std::string value) {
  if (value.empty()) return false;

  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value != "0" && value != "off" && value != "false" &&
         value != "no" && value != "n" && value != "f";
}

inline bool IsEnvFlagEnabled(const char* name) {
  std::string env_name(name);
  if (!os::IsEnvVarSet(env_name)) return false;

  return IsEnvFlagValueEnabled(os::GetEnvVar(env_name));
}

}  // namespace hotswap
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_HOTSWAP_ENV_HPP_

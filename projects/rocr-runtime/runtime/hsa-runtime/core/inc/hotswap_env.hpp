/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HSA_RUNTIME_CORE_INC_HOTSWAP_ENV_HPP_
#define HSA_RUNTIME_CORE_INC_HOTSWAP_ENV_HPP_

#ifdef ROCR_HOTSWAP_COMGR_ADAPTER
#include "core/inc/isa.h"
#endif
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

inline bool IsPresentationModeEnabled() {
  static const bool Enabled = [] {
    if (IsEnvFlagEnabled("HSA_HOTSWAP_DISABLE")) return false;
    return IsEnvFlagEnabled("HSA_HOTSWAP_PRESENT_ISA");
  }();
  return Enabled;
}

#ifdef ROCR_HOTSWAP_COMGR_ADAPTER
inline bool ValidateHotSwapPresentationTarget(
    const std::string& target_env, const std::string& override_env,
    const core::Isa& execution_isa, std::string& failure) {
  const std::string& configured_target =
      target_env.empty() ? override_env : target_env;
  if (configured_target.empty() || configured_target == "0" ||
      configured_target == "1") {
    failure =
        "HSA_HOTSWAP_TARGET must name the execution ISA when "
        "HSA_HOTSWAP_PRESENT_ISA is set";
    return false;
  }

  std::string configured_name(configured_target);
  if (configured_name.rfind("amdgcn-amd-amdhsa--", 0) != 0)
    configured_name = "amdgcn-amd-amdhsa--" + configured_name;
  const core::Isa* configured_isa = core::IsaRegistry::GetIsa(configured_name);
  if (!configured_isa) {
    failure = "configured HotSwap target ISA '" + configured_name +
              "' is not recognized by ROCr";
    return false;
  }

  // Require the same processor; IsCompatible applies target-ID "any"
  // semantics to omitted features and rejects explicitly mismatched features.
  if (configured_isa->GetProcessorName() != execution_isa.GetProcessorName() ||
      !core::Isa::IsCompatible(*configured_isa, execution_isa, 0)) {
    failure = "configured HotSwap target ISA '" +
              configured_isa->GetIsaName() +
              "' does not match physical execution ISA '" +
              execution_isa.GetIsaName() + "'";
    return false;
  }

  failure.clear();
  return true;
}
#endif

}  // namespace hotswap
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_HOTSWAP_ENV_HPP_

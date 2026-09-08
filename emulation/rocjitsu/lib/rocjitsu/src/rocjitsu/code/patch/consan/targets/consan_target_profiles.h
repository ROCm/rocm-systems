// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_target_profiles.h
/// @brief Concrete ConSan target-profile registry and target-family predicates.

#pragma once

#include "rocjitsu/code/patch/consan/consan_capability_contract.h"

#include <array>

namespace rocjitsu {

/// The production target-admission map and documentation iteration order.
/// Concrete architectural facts are assembled entirely within the target layer.
extern const std::array<ConSanTargetProfile, 5> kConSanTargetProfiles;

[[nodiscard]] constexpr bool consan_arch_is_cdna3_or_cdna4(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4;
}

[[nodiscard]] constexpr bool consan_arch_is_rdna4_or_cdna5(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5;
}

[[nodiscard]] constexpr bool consan_arch_is_cdna5(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_CDNA5;
}

[[nodiscard]] constexpr bool consan_arch_is_rdna3(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA3;
}

} // namespace rocjitsu

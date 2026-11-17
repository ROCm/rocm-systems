// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/fault_injection/ordinary_acquire.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace rocjitsu {

/// @brief An acquire profile with an enabled ordinary address recipe.
struct OrdinaryWrongAddressCapability {
  rj_code_target_id_t target = ROCJITSU_CODE_TARGET_INVALID;
  uint32_t wave_size = 0;
  OrdinaryAcquireProfile profile = OrdinaryAcquireProfile::Cdna5GlobalB32;
};

struct OrdinaryWrongAddressResult {
  OrdinaryAcquireSelection selection;
  std::vector<uint8_t> image;
  uint64_t logical_mutation_count = 1;
};

/// @brief Report concrete target/wave/profile rows with an enabled address recipe.
/// @details Acquire query support alone does not imply mutation support.
[[nodiscard]] std::vector<OrdinaryWrongAddressCapability> ordinary_wrong_address_capabilities();

/// @brief Change one acquire-qualified load's byte immediate on a fresh source fork.
/// @details The delta must be positive, divisible by four, and at most 0x7fffff.
/// The signed source-plus-delta sum must fit the profile's immediate and be
/// divisible by four. Cache, wait, bookkeeping, scope, and data fields retain
/// their pristine values. CDNA5 scale_offset is preserved; the immediate and
/// requested delta remain byte offsets. Each call makes an independent mutant.
[[nodiscard]] FailureOr<OrdinaryWrongAddressResult>
apply_ordinary_wrong_address(const FaultMutationContext &, const OrdinaryAcquireSelection &,
                             std::optional<uint32_t> byte_delta, FaultMutationError &);

} // namespace rocjitsu

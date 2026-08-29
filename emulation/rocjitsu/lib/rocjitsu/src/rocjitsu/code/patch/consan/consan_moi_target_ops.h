// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_target_ops.h
/// @brief Target-operation contracts shared by MOI engines.

#pragma once

#include "rocjitsu/code/patch/consan/consan_capability_contract.h"

#include <cstdint>
#include <vector>

namespace rocjitsu::consan_detail {

/// Semantic request to restore SCC from bit zero of a scalar route key.
struct MoiEncodedSccRestoreRequest {
  uint16_t encoded_sgpr = 0;
  bool normalize_encoded_sgpr = true;

  bool operator==(const MoiEncodedSccRestoreRequest &) const = default;
};

/// Append the target sequence that restores SCC from an encoded route key.
/// Only target profiles with a qualified bit-test recipe are admitted.
[[nodiscard]] bool append_restore_moi_scc_from_route_key(std::vector<uint32_t> &words,
                                                         const MoiEncodedSccRestoreRequest &request,
                                                         const ConSanTargetProfile &target);

} // namespace rocjitsu::consan_detail

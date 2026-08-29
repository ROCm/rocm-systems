// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi.h"

#include <span>
#include <string>
#include <vector>

namespace rocjitsu::consan_detail {

/// Project policy-admitted access intents into immutable lowering candidates.
/// This boundary copies normalized inventory facts and never reclassifies a
/// site or selects an engine policy.
[[nodiscard]] std::vector<ConSanMoiCandidate>
build_moi_candidates(const ProgramInventory &inventory, const ConSanObservationPlan &plan,
                     std::vector<std::string> &errors);

/// Make a non-owning traversal view without filtering admitted candidates.
[[nodiscard]] std::vector<const ConSanMoiCandidate *>
make_moi_candidate_pointer_view(std::span<const ConSanMoiCandidate> admitted);

} // namespace rocjitsu::consan_detail

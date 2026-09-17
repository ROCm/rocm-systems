// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_instrumentation.h"

#include <span>
#include <string>
#include <vector>

namespace rocjitsu::consan::detail {

/// Project policy-admitted access intents into immutable lowering candidates.
/// This boundary copies normalized inventory facts and never reclassifies a
/// site or selects a mode policy.
[[nodiscard]] std::vector<Candidate> build_candidates(const ProgramInventory &inventory,
                                                      const ObservationPlan &plan,
                                                      std::vector<std::string> &errors);

/// Make a non-owning traversal view without filtering admitted candidates.
[[nodiscard]] std::vector<const Candidate *>
make_candidate_pointer_view(std::span<const Candidate> admitted);

} // namespace rocjitsu::consan::detail

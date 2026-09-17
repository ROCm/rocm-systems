// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_contracts.h
/// @brief Pure ConSan-mode resource and scalar-state planning contracts.

#pragma once

#include "rocjitsu/code/patch/consan/consan_access_target.h"
#include "rocjitsu/code/patch/consan/consan_internal.h"

namespace rocjitsu::consan::detail {

struct PublicationStateSgprs {
  uint16_t original_exec_save_sgpr = 0;
  uint16_t selection_vcc_save_sgpr = 0;
  uint16_t publication_exec_save_sgpr = 0;
  uint16_t guest_scc_snapshot_sgpr = 0;
};

[[nodiscard]] std::optional<PublicationStateSgprs>
publication_state_sgprs(std::optional<uint16_t> exec_save_sgpr);

[[nodiscard]] bool
access_can_emit_spill_over_guest_operands(const AccessResourceFacts &resource_facts,
                                          const Candidate &candidate);

[[nodiscard]] bool access_can_plan_spill_over_guest_operands(
    const Request &request, const AccessResourceFacts &resource_facts, const Candidate &candidate);

[[nodiscard]] uint16_t direct_scratch_count(const Request &request,
                                            const AccessResourceFacts &resource_facts);

} // namespace rocjitsu::consan::detail

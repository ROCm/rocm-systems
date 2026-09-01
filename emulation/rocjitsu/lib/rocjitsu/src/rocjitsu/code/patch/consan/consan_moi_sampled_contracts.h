// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_sampled_contracts.h
/// @brief Pure Sampled-engine resource and scalar-state planning contracts.

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_access_target.h"
#include "rocjitsu/code/patch/consan/consan_moi_internal.h"

namespace rocjitsu::consan_moi_impl {

struct MoiSampledPublicationStateSgprs {
  uint16_t original_exec_save_sgpr = 0;
  uint16_t selection_vcc_save_sgpr = 0;
  uint16_t publication_exec_save_sgpr = 0;
  uint16_t guest_scc_snapshot_sgpr = 0;
};

[[nodiscard]] std::optional<MoiSampledPublicationStateSgprs>
moi_sampled_publication_state_sgprs(std::optional<uint16_t> exec_save_sgpr);

[[nodiscard]] bool
sampled_access_can_emit_spill_over_guest_operands(const MoiAccessResourceFacts &resource_facts,
                                                  const ConSanMoiCandidate &candidate);

[[nodiscard]] bool
sampled_access_can_plan_spill_over_guest_operands(const ConSanRequest &request,
                                                  const MoiAccessResourceFacts &resource_facts,
                                                  const ConSanMoiCandidate &candidate);

[[nodiscard]] uint16_t direct_sampled_scratch_count(const ConSanRequest &request,
                                                    const MoiAccessResourceFacts &resource_facts);

} // namespace rocjitsu::consan_moi_impl

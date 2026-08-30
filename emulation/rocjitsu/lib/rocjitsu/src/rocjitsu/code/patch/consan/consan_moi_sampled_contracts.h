// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_sampled_contracts.h
/// @brief Pure Sampled-engine resource and scalar-state planning contracts.

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_internal.h"

namespace rocjitsu::consan_moi_impl {

struct MoiSampledPublicationStateSgprs {
  uint16_t original_exec_save_sgpr = 0;
  uint16_t selection_vcc_save_sgpr = 0;
  uint16_t publication_exec_save_sgpr = 0;
  uint16_t guest_scc_snapshot_sgpr = 0;
};

[[nodiscard]] std::optional<MoiSampledPublicationStateSgprs>
moi_sampled_publication_state_sgprs(const ConSanRequest &request,
                                    const ConSanMoiOperatingPoint &point);

[[nodiscard]] bool sampled_access_supports_spill_backed_operand_recovery(
    const ConSanRequest &request, const ConSanMoiCandidate &candidate, rj_code_arch_t arch);

[[nodiscard]] bool
sampled_access_can_emit_spill_over_guest_operands(const ConSanMoiOperatingPoint &point,
                                                  const ConSanMoiCandidate &candidate);

[[nodiscard]] bool
sampled_access_can_plan_spill_over_guest_operands(const ConSanRequest &request,
                                                  const ConSanMoiOperatingPoint &point,
                                                  const ConSanMoiCandidate &candidate);

[[nodiscard]] uint16_t direct_sampled_scratch_count(const ConSanRequest &request,
                                                    const ConSanMoiOperatingPoint &point,
                                                    const ConSanMoiCandidate &candidate,
                                                    rj_code_arch_t arch);

[[nodiscard]] uint16_t sampled_spill_backed_scratch_count(const ConSanRequest &request,
                                                          const ConSanMoiOperatingPoint &point,
                                                          const ConSanMoiCandidate &candidate,
                                                          rj_code_arch_t arch);

} // namespace rocjitsu::consan_moi_impl

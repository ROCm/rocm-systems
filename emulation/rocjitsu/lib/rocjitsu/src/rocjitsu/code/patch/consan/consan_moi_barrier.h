// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_barrier.h
/// @brief Compiled MOI barrier lowering contract.

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_evidence_planning.h"
#include "rocjitsu/code/patch/consan/consan_moi_placement_contracts.h"

namespace rocjitsu::consan_moi_impl {

/// Exact island prefix reserved by an earlier lowering pass for barrier
/// routing. Consumers may validate the reservation against emitted patch
/// provenance, but must not reconstruct its strided layout from mode state.
struct MoiBarrierIslandReservation {
  uint64_t begin = 0u;
  uint32_t slot_words = 0u;
  uint64_t reserved_slot_count = 0u;
};

void try_apply_inline_shadow_barrier_patch(std::span<const uint8_t> bytes,
                                           const ConSanOptions &options,
                                           const ConSanMoiOperatingPoint &operating_point,
                                           rj_code_arch_t arch,
                                           MoiResourcePlanningState &resource_state,
                                           const std::optional<MoiBarrierIslandReservation>
                                               &reserved_sync_islands,
                                           const MoiObjectModeSemantics &semantics,
                                           ConSanTransformArtifacts &result);

} // namespace rocjitsu::consan_moi_impl

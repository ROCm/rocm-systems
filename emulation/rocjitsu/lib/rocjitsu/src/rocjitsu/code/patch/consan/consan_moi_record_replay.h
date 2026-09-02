// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_record_replay.h
/// @brief Record/Replay engine lowering entry points.

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_barrier.h"

namespace rocjitsu::consan_moi_impl {

// A branch-only access layout retains a two-word relay pair after every
// reserved synchronization island. The island remains available to the later
// sync pass while the dedicated pair forms a capacity-accounted relay spine.
inline constexpr uint32_t kMoiRecordReplayBarrierRelayWords = 2u;
inline constexpr uint32_t kMoiRecordReplayBarrierRelaySlotWords =
    kMoiCompactIndirectIslandWords + kMoiRecordReplayBarrierRelayWords;
inline constexpr uint32_t kMoiRecordReplayBorrowedEntryIslandWords = 20u;

/// Record/Replay access-placement facts consumed by the immediately following
/// synchronization lowering stage.
struct MoiRecordReplayAccessOutput {
  std::optional<MoiBarrierIslandReservation> reserved_sync_islands;
  std::vector<std::pair<uint64_t, uint64_t>> generated_branch_relay_ranges;
};

void try_apply_first_light_access_record_patch(
    std::span<const uint8_t> bytes, const ConSanOptions &options,
    const ConSanMoiOperatingPoint &operating_point, const ConSanTargetProfile &target,
    MoiResourcePlanningState &resource_state, MoiRecordReplayAccessOutput &access_output,
    std::span<const ConSanMoiCandidate> admitted, const MoiObjectModeSemantics &mode_semantics,
    ConSanTransformArtifacts &result);

void try_apply_atomic_record_patch(std::span<const uint8_t> bytes, const ConSanOptions &options,
                                   const ConSanMoiOperatingPoint &operating_point,
                                   const MoiObjectModeSemantics &mode_semantics,
                                   rj_code_arch_t arch, ConSanTransformArtifacts &result);

void try_apply_fence_record_patch(std::span<const uint8_t> bytes, const ConSanOptions &options,
                                  const ConSanMoiOperatingPoint &operating_point,
                                  const MoiObjectModeSemantics &mode_semantics, rj_code_arch_t arch,
                                  ConSanTransformArtifacts &result);

void try_apply_record_replay_barrier_patch(
    std::span<const uint8_t> bytes, const ConSanOptions &options,
    const ConSanMoiOperatingPoint &operating_point, rj_code_arch_t arch,
    MoiResourcePlanningState &resource_state, const MoiRecordReplayAccessOutput &access_output,
    const MoiObjectModeSemantics &semantics, ConSanTransformArtifacts &result);

} // namespace rocjitsu::consan_moi_impl

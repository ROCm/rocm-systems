// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_record_replay.h
/// @brief Record/Replay engine lowering entry points.

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_placement_contracts.h"

namespace rocjitsu::consan_moi_impl {

/// Exact prefix reserved by access placement for a later synchronization
/// pass. The consumer must not rediscover this strided layout from patch order
/// or emitted instruction bytes.
struct MoiReservedBarrierIslandLayout {
  uint64_t begin = 0u;
  uint32_t slot_words = kMoiRecordReplayIndirectIslandWords;
  uint64_t reserved_slot_count = 0u;
};

/// Record/Replay access-placement facts consumed by the immediately following
/// synchronization lowering stage.
struct MoiRecordReplayAccessOutput {
  std::optional<MoiReservedBarrierIslandLayout> reserved_sync_islands;
  std::vector<std::pair<uint64_t, uint64_t>> generated_branch_relay_ranges;
};

void try_apply_first_light_access_record_patch(
    std::span<const uint8_t> bytes, const ConSanOptions &options,
    const ConSanMoiOperatingPoint &operating_point, const ConSanTargetProfile &target,
    MoiResourcePlanningState &resource_state, MoiRecordReplayAccessOutput &access_output,
    std::span<const ConSanMoiCandidate> admitted, ConSanTransformArtifacts &result);

void try_apply_atomic_record_patch(std::span<const uint8_t> bytes, const ConSanOptions &options,
                                   const ConSanMoiOperatingPoint &operating_point,
                                   rj_code_arch_t arch, ConSanTransformArtifacts &result);

void try_apply_fence_record_patch(std::span<const uint8_t> bytes, const ConSanOptions &options,
                                  const ConSanMoiOperatingPoint &operating_point,
                                  rj_code_arch_t arch, ConSanTransformArtifacts &result);

} // namespace rocjitsu::consan_moi_impl

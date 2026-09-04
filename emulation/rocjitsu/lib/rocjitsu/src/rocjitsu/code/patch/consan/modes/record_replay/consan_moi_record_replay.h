// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_record_replay.h
/// @brief Record/Replay engine lowering entry points.

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_barrier.h"

namespace rocjitsu::consan_moi_impl {

[[nodiscard]] std::optional<std::vector<uint32_t>> build_first_light_access_record_words(
    std::span<const uint8_t> bytes, const ConSanMoiCandidate &candidate,
    const ConSanRequest &request, const BoundRuntimeResources &bound_resources,
    const ConSanMoiOperatingPoint &point, uint16_t scratch_vgpr, rj_code_arch_t arch,
    uint32_t record_index, uint32_t record_count, uint32_t logical_range_index,
    const ConSanMoiReportBufferLayout &layout, bool spill_overlaps_guest_operands,
    const VgprSpillSequence *spill, const ConSanMoiPrivateStateLayout *private_layout,
    const std::optional<consan_detail::MoiWorkitemOwnerDerivationPlan> &owner_derivation,
    std::vector<std::string> &errors, uint32_t *guest_instruction_offset = nullptr,
    uint32_t *guest_instruction_word_count = nullptr);

void try_stage_first_light_access_fragments(std::span<const uint8_t> bytes,
                                            const ConSanOptions &options,
                                            const ConSanMoiOperatingPoint &operating_point,
                                            const ConSanTargetProfile &target,
                                            std::span<const ConSanMoiCandidate> admitted,
                                            const MoiObjectModeSemantics &mode_semantics,
                                            ConSanTransformArtifacts &result);

void try_stage_atomic_record_fragments(std::span<const uint8_t> bytes, const ConSanOptions &options,
                                       const ConSanMoiOperatingPoint &operating_point,
                                       const MoiObjectModeSemantics &mode_semantics,
                                       rj_code_arch_t arch, ConSanTransformArtifacts &result);

void try_stage_fence_record_fragments(std::span<const uint8_t> bytes, const ConSanOptions &options,
                                      const ConSanMoiOperatingPoint &operating_point,
                                      const MoiObjectModeSemantics &mode_semantics,
                                      rj_code_arch_t arch, ConSanTransformArtifacts &result);

void try_stage_record_replay_barrier_fragments(std::span<const uint8_t> bytes,
                                               const ConSanOptions &options,
                                               const ConSanMoiOperatingPoint &operating_point,
                                               rj_code_arch_t arch,
                                               const MoiObjectModeSemantics &semantics,
                                               ConSanTransformArtifacts &result);

} // namespace rocjitsu::consan_moi_impl

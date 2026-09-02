// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_sampled_contracts.h"

namespace rocjitsu::consan_moi_impl {

[[nodiscard]] bool
append_sampled_window_bank_index(std::vector<uint32_t> &words, const ConSanMoiOperatingPoint &point,
                                 const BoundRuntimeResources &resources,
                                 const ConSanMoiWorkgroupSources &workgroup_sources,
                                 uint32_t bank_count, uint16_t bank_vgpr, uint16_t temporary_vgpr,
                                 uint16_t owner_vgpr, rj_code_arch_t arch);

[[nodiscard]] std::optional<std::vector<uint32_t>> build_direct_sampled_watchpoint_words(
    std::span<const uint8_t> bytes, const ConSanMoiCandidate &candidate,
    const ConSanRequest &request, const BoundRuntimeResources &bound_resources,
    const ConSanMoiOperatingPoint &point, const ConSanMoiWorkgroupSources &workgroup_sources,
    const ConSanMoiOwnerEpochVgprSources &owner_epoch_vgprs, uint16_t scratch_vgpr,
    rj_code_arch_t arch, uint32_t first_record_index,
    std::optional<uint32_t> prior_first_record_index, uint32_t prior_access_range_count,
    uint32_t window_bank_count, size_t sampled_causal_windows_offset,
    size_t sampled_watchpoints_offset, size_t sampled_pending_acquires_offset,
    uint32_t pending_acquire_owner_bank_count, bool spill_overlaps_guest_operands,
    bool spill_backed_operand_recovery, const VgprSpillSequence *spill,
    std::optional<uint32_t> private_epoch_offset,
    const std::optional<consan_detail::MoiWorkitemOwnerDerivationPlan> &owner_derivation,
    bool runtime_workgroup_gate_in_body, std::vector<std::string> &errors,
    uint32_t *guest_instruction_offset = nullptr, uint32_t *guest_instruction_word_count = nullptr);

} // namespace rocjitsu::consan_moi_impl

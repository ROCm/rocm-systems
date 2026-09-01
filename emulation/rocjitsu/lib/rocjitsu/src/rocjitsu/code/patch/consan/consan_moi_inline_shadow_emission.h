// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_access_target.h"
#include "rocjitsu/code/patch/consan/consan_moi_internal.h"

namespace rocjitsu::consan_moi_impl {

[[nodiscard]] uint16_t inline_shadow_loop_scratch_count(const ConSanMoiCandidate &candidate);
[[nodiscard]] uint16_t inline_shadow_scratch_count(const ConSanRequest &request,
                                                   const MoiAccessResourceFacts &resource_facts,
                                                   const ConSanMoiCandidate &candidate);
[[nodiscard]] uint16_t
inline_shadow_spill_backed_scratch_count(const ConSanRequest &request,
                                         const MoiAccessResourceFacts &resource_facts,
                                         const ConSanMoiCandidate &candidate);

[[nodiscard]] bool validate_inline_shadow_exec_save_sgpr(const ConSanRequest &request,
                                                         const BoundRuntimeResources &resources,
                                                         const ConSanMoiOperatingPoint &point,
                                                         const MoiObjectModeSemantics &semantics,
                                                         rj_code_arch_t arch,
                                                         std::vector<std::string> &errors);

[[nodiscard]] std::optional<uint16_t>
inline_shadow_visible_evidence_sgpr(const ConSanRequest &request,
                                    const ConSanMoiOperatingPoint &point);

[[nodiscard]] std::optional<std::vector<uint32_t>> build_inline_shadow_words(
    std::span<const uint8_t> bytes, const ConSanMoiCandidate &candidate,
    const ConSanRequest &request, const BoundRuntimeResources &bound_resources,
    const ConSanMoiOperatingPoint &point, uint16_t scratch_vgpr,
    const MoiObjectModeSemantics &semantics, rj_code_arch_t arch,
    const ConSanMoiReportBufferLayout &layout, const ConSanMoiWorkgroupSources &workgroup_sources,
    std::optional<ConSanMoiWorkgroupShadowLayout> workgroup_shadow,
    bool byte_granular_external_shadow, std::optional<uint32_t> private_epoch_offset,
    const std::optional<consan_detail::MoiWorkitemOwnerDerivationPlan> &owner_derivation,
    std::optional<uint32_t> private_workgroup_key_offset,
    std::optional<uint32_t> private_dispatch_id_offset, uint16_t workgroup_shadow_compact_token,
    uint16_t scratch_count, bool spill_overlaps_guest_operands, const VgprSpillSequence *spill,
    std::vector<std::string> &errors, uint32_t *guest_instruction_word_count = nullptr);

} // namespace rocjitsu::consan_moi_impl

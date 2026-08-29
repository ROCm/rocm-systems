// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_sync_emission.h
/// @brief Shared typed synchronization planning and native emission contracts.

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_exact_shadow_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_internal.h"
#include "rocjitsu/code/patch/consan/consan_moi_probe_contracts.h"

namespace rocjitsu::consan_moi_impl {

[[nodiscard]] bool
append_moi_atomic_lowering_commit(ConSanTransformArtifacts &result,
                                  const consan_detail::MoiAtomicEvidenceSitePlan &plan,
                                  const ConSanPatchInfo &patch, std::string_view probe_name,
                                  std::vector<ConSanCommittedLowering> &commits);

[[nodiscard]] bool
append_moi_fence_lowering_commit(ConSanTransformArtifacts &result,
                                 const consan_detail::MoiFenceEvidenceSitePlan &plan,
                                 const ConSanPatchInfo &patch, std::string_view probe_name,
                                 std::vector<ConSanCommittedLowering> &commits);

[[nodiscard]] bool
append_moi_barrier_lowering_commit(ConSanTransformArtifacts &result,
                                   const consan_detail::MoiBarrierEvidenceSitePlan &plan,
                                   const ConSanPatchInfo &patch, std::string_view probe_name,
                                   std::vector<ConSanCommittedLowering> &commits);

[[nodiscard]] bool publish_moi_sync_lowering_commits(ConSanTransformArtifacts &result,
                                                     std::vector<ConSanCommittedLowering> commits,
                                                     std::string_view probe_name);

[[nodiscard]] std::optional<consan_detail::SampledAtomicSemantics>
sampled_atomic_semantics_for_plan(const SynchronizationInventoryView &graph,
                                  const consan_detail::MoiAtomicEvidenceSitePlan &plan,
                                  std::string &unsupported_reason);

[[nodiscard]] std::vector<consan_detail::MoiFenceEvidenceSitePlan>
build_moi_fence_evidence_site_plans(const ConSanTransformArtifacts &result,
                                    std::vector<std::string> &errors);

[[nodiscard]] uint16_t fence_record_scratch_count(const ConSanAtomicLoweringForm &form);
[[nodiscard]] uint16_t atomic_record_scratch_count(const ConSanAtomicLoweringForm &form);
[[nodiscard]] uint16_t inline_shadow_atomic_scratch_count(rj_code_arch_t arch);

[[nodiscard]] std::vector<consan_detail::MoiAtomicEvidenceSitePlan>
build_moi_atomic_evidence_site_plans(const ConSanTransformArtifacts &result,
                                     ConSanProbeIntentKind evidence_kind,
                                     std::vector<std::string> &errors);

[[nodiscard]] bool neutralize_atomic_scalar_clause(
    std::vector<uint8_t> &text, const consan_detail::MoiAtomicEvidenceSitePlan &candidate,
    rj_code_arch_t arch, std::vector<ConSanPatchInfo> &patches, std::vector<std::string> &errors);

[[nodiscard]] bool validate_inline_atomic_exec_save_sgpr(const ConSanMoiOperatingPoint &point,
                                                         std::vector<std::string> &errors);

[[nodiscard]] bool append_inline_atomic_slot_address(std::vector<uint32_t> &words,
                                                     uint64_t table_base, uint32_t table_capacity,
                                                     uint16_t atomic_address_vgpr,
                                                     uint16_t scratch_vgpr, rj_code_arch_t arch);

[[nodiscard]] bool
append_inline_atomic_snapshot_address(std::vector<uint32_t> &words, uint64_t table_base,
                                      uint32_t table_capacity, uint16_t atomic_address_vgpr,
                                      uint16_t snapshot_address_vgpr, uint16_t hash_vgpr,
                                      uint16_t temporary_vgpr, rj_code_arch_t arch);

[[nodiscard]] bool
inline_atomic_scalar_spill_aliases_guest_address(const ConSanMoiAtomicAddressPlan &address_plan,
                                                 uint16_t scalar_base, uint16_t scalar_count);

[[nodiscard]] std::optional<std::vector<uint32_t>> build_inline_atomic_ordering_cave_words(
    std::span<const uint8_t> bytes, const consan_detail::MoiAtomicEvidenceSitePlan &candidate,
    const ConSanMoiAtomicAddressPlan &address_plan, const ConSanRequest &request,
    const BoundRuntimeResources &bound_resources, const ConSanMoiOperatingPoint &input_point,
    uint16_t scratch_vgpr, const VgprSpillSequence *spill, const SgprSpillSequence *scalar_spill,
    const MoiPrivateEpochLayout *private_layout, rj_code_arch_t arch,
    size_t inline_atomic_release_slots_offset, uint32_t inline_atomic_release_capacity,
    size_t inline_causal_snapshots_offset, uint32_t inline_causal_snapshot_capacity,
    size_t inline_acquired_token_slots_offset, uint32_t inline_acquired_token_capacity,
    uint64_t cave_text_offset, uint64_t return_text_offset, uint32_t &guest_instruction_offset,
    std::vector<std::string> &errors, std::span<const uint32_t> trailing_guest_words = {});

} // namespace rocjitsu::consan_moi_impl

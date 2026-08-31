// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_dynamic_record_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_internal.h"
#include "rocjitsu/code/patch/consan/consan_moi_relocation.h"
#include "rocjitsu/code/patch/consan/consan_moi_runtime_workgroup_gate.h"

#include <array>
#include <optional>
#include <span>
#include <vector>

namespace rocjitsu::consan_moi_impl {

/// Resolved lowering input shared by Record/Replay atomic, fence, and barrier
/// emitters. Owner-local placement resolves every optional source before this
/// immutable value crosses into native emission.
struct MoiRecordEventEmissionPlan {
  std::optional<uint16_t> scratch_vgpr;
  std::optional<uint16_t> moi_exec_save_sgpr;
  ConSanMoiOwnerEpochRegisterState moi_owner_epoch_vgprs;
  std::optional<uint16_t> moi_workgroup_key_vgpr;
  std::optional<uint16_t> moi_dispatch_id_vgpr;
  ConSanMoiPersistentWorkgroupRegisters moi_record_replay_workgroup_vgprs;
  ConSanMoiPersistentSgprState moi_persistent_sgprs;
  std::optional<uint64_t> moi_report_buffer_address;
  bool automatic_moi_record_replay_sgpr_spill = false;
  std::optional<uint16_t> router_dispatch_key_sgpr;
  std::optional<uint16_t> router_call_return_sgpr;
  ConSanMoiWorkgroupSources workgroup_sources;
  consan_detail::MoiSpecialStateSgprs special_state;
  consan_moi_detail::ConSanMoiReportDispatchIdSource dispatch_id_sources;
  std::optional<MoiRuntimeWorkgroupGatePlan> runtime_workgroup_gate;
  std::optional<ConSanMoiIndirectJumpSgprs> indirect_jump;
  uint16_t required_sgpr_count = 0;
};

[[nodiscard]] inline std::optional<consan_detail::MoiSpecialStateSgprs>
moi_special_state_sgprs(const MoiRecordEventEmissionPlan &plan) {
  return plan.special_state;
}

[[nodiscard]] inline std::optional<ConSanMoiIndirectJumpSgprs>
moi_indirect_jump_sgprs(const MoiRecordEventEmissionPlan &plan) {
  return plan.indirect_jump;
}

[[nodiscard]] inline bool append_dynamic_record_store_moi_report_dispatch_id_pair(
    std::vector<uint32_t> &words, const consan_moi_detail::DynamicRecordLayout &layout,
    uint64_t low_field_address, const MoiRecordEventEmissionPlan &plan, uint16_t slot_vgpr,
    uint16_t scratch_vgpr, rj_code_arch_t arch) {
  return consan_moi_detail::append_dynamic_record_store_moi_report_dispatch_id_pair(
      words, layout, low_field_address, plan.dispatch_id_sources, slot_vgpr, scratch_vgpr, arch);
}

[[nodiscard]] inline bool
append_moi_direct_or_indirect_return(std::vector<uint32_t> &words, uint64_t cave_text_offset,
                                     uint64_t return_text_offset,
                                     const MoiRecordEventEmissionPlan &plan, rj_code_arch_t arch) {
  return append_moi_direct_or_indirect_return(words, cave_text_offset, return_text_offset,
                                              plan.indirect_jump, arch);
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_barrier_record_cave_words(
    std::span<const uint8_t> bytes, const consan_detail::MoiBarrierEvidenceSitePlan &candidate,
    const MoiRecordEventEmissionPlan &options, const VgprSpillSequence *spill,
    const SgprSpillSequence *scalar_spill, rj_code_arch_t arch, uint32_t barrier_record_capacity,
    size_t barrier_records_offset, uint32_t original_barrier_word, uint64_t cave_text_offset,
    uint64_t return_text_offset,
    const std::optional<consan_detail::MoiWorkitemOwnerDerivationPlan> &derived_owner,
    std::vector<std::string> &errors);

[[nodiscard]] std::optional<std::vector<uint32_t>> build_atomic_record_cave_words(
    std::span<const uint8_t> bytes, const consan_detail::MoiAtomicEvidenceSitePlan &candidate,
    const ConSanMoiAtomicAddressPlan &address_plan, const MoiRecordEventEmissionPlan &options,
    const VgprSpillSequence *spill, const SgprSpillSequence *scalar_spill, rj_code_arch_t arch,
    uint32_t record_index, uint32_t atomic_record_capacity, size_t atomic_records_offset,
    uint64_t cave_text_offset, uint64_t return_text_offset, bool already_runtime_workgroup_gated,
    const std::optional<consan_detail::MoiWorkitemOwnerDerivationPlan> &derived_owner,
    uint32_t &guest_instruction_offset, std::vector<std::string> &errors);

[[nodiscard]] std::optional<std::vector<uint32_t>> build_fence_record_cave_words(
    std::span<const uint8_t> bytes, const consan_detail::MoiFenceEvidenceSitePlan &candidate,
    const ConSanMoiAtomicAddressPlan &address_plan, const MoiRecordEventEmissionPlan &options,
    const VgprSpillSequence *spill, const SgprSpillSequence *scalar_spill, rj_code_arch_t arch,
    uint32_t record_index, uint32_t record_capacity_or_count, size_t fence_records_offset,
    uint64_t cave_text_offset, uint64_t return_text_offset,
    std::optional<uint16_t> call_return_sgpr, std::span<const uint32_t> displaced_tail_words,
    const std::optional<consan_detail::MoiWorkitemOwnerDerivationPlan> &derived_owner,
    std::vector<std::string> &errors);

} // namespace rocjitsu::consan_moi_impl

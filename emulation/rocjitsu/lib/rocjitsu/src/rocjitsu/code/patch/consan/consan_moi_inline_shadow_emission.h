// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_access_target.h"
#include "rocjitsu/code/patch/consan/consan_moi_exact_shadow_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_inline_shadow.h"
#include "rocjitsu/code/patch/consan/consan_moi_internal.h"
#include "rocjitsu/code/patch/consan/consan_moi_mode_planning.h"
#include "rocjitsu/code/patch/consan/consan_moi_report_emission.h"

namespace rocjitsu::consan_moi_impl {

/// Complete exact state consumed by one InlineShadow access body. Candidate
/// planning retains this product after owner-local assignment; provisional,
/// appended, and inline emission cannot reconstruct broad request, resource,
/// or operating-point state independently.
struct MoiInlineShadowEmissionPlan {
  uint64_t report_buffer_address = 0;
  uint64_t report_generation = 0;
  uint64_t report_dispatch_id = 0;
  bool track_atomics = false;
  bool byte_granular_external_shadow = false;
  bool spill_overlaps_guest_operands = false;
  uint16_t required_exec_save_sgpr_count = 0;
  uint16_t scratch_vgpr = 0;
  uint16_t scratch_count = 0;
  uint16_t workgroup_shadow_compact_token = 0;
  MoiInlineShadowScalarState scalar_state;
  MoiScalarAbiPlan scalar_abi;
  consan_moi_detail::ConSanMoiReportDispatchIdSource dispatch_id;
  MoiInlineShadowOwnerFieldPlan owner_field;
  MoiInlineShadowEpochFieldPlan epoch_field;
  consan_detail::MoiWorkgroupKeyRegisterPlan workgroup_key_registers;
  MoiAccessResourceFacts access_resource_facts;
  ConSanMoiWorkgroupSources workgroup_sources;
  std::optional<ConSanMoiWorkgroupShadowLayout> workgroup_shadow;
  std::optional<uint32_t> private_epoch_offset;
  std::optional<consan_detail::MoiWorkitemOwnerDerivationPlan> owner_derivation;
  std::optional<uint32_t> private_workgroup_key_offset;
  std::optional<uint32_t> private_dispatch_id_offset;
};

/// Exact private-state materialization selected for one InlineShadow atomic
/// body. The alternatives are resolved while the owner-local operating point
/// is still available; native emission cannot rediscover owner policy or
/// descriptor geometry.
struct MoiInlineAtomicPrivateStatePlan {
  uint32_t epoch_offset = 0;
  uint32_t workgroup_key_offset = 0;
  std::optional<consan_detail::MoiWorkitemOwnerDerivationPlan> workitem_owner;
  std::optional<consan_detail::MoiResidentWaveOwnerRequest> resident_wave_owner;
  bool resident_wave_owner_is_borrowed = false;
  std::optional<uint32_t> dispatch_id_offset;
  std::optional<uint16_t> dispatch_id_sgpr;
};

/// Complete exact state consumed by one InlineShadow atomic-ordering body.
/// Candidate or dense-group planning publishes this immutable product after
/// owner-local assignment. The body emitter sees neither the request, bound
/// resources, mutable operating point, nor object-wide mode semantics.
struct MoiInlineAtomicEmissionPlan {
  uint64_t report_buffer_address = 0;
  ConSanMoiReportBufferLayout report_layout;
  bool inline_access_present = false;
  uint16_t scratch_vgpr = 0;
  uint16_t scratch_vgpr_count = 0;
  bool requires_aligned_flat_compare_swap_data_pair = false;
  uint16_t exec_save_sgpr = 0;
  ConSanMoiOwnerEpochRegisters owner_epoch_vgprs;
  ConSanMoiPersistentSgprState persistent_sgprs;
  consan_detail::MoiSpecialStateSgprs special_state;
  consan_detail::MoiWorkgroupKeyRegisterPlan workgroup_key_registers;
  ConSanMoiWorkgroupSources workgroup_sources;
  consan_moi_detail::ConSanMoiReportDispatchIdSource dispatch_id;
  std::optional<ConSanMoiIndirectJumpSgprs> indirect_jump;
  std::optional<MoiInlineAtomicPrivateStatePlan> private_state;
  ConSanPatchAbiEffects patch_abi;
};

[[nodiscard]] std::optional<std::vector<uint32_t>>
build_inline_shadow_words(std::span<const uint8_t> bytes, const ConSanMoiCandidate &candidate,
                          const MoiInlineShadowEmissionPlan &plan, rj_code_arch_t arch,
                          const ConSanMoiReportBufferLayout &layout, const VgprSpillSequence *spill,
                          std::vector<std::string> &errors,
                          uint32_t *guest_instruction_word_count = nullptr);

} // namespace rocjitsu::consan_moi_impl

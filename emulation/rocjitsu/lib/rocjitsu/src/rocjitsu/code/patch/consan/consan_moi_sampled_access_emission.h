// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_report_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_sampled_contracts.h"

namespace rocjitsu::consan_moi_impl {

/// Complete site-local facts selected before a Sampled access body is sized.
///
/// Provisional sizing and every final placement consume this same value. The
/// emitter therefore cannot rediscover mode policy, bound report resources,
/// persistent state, or scratch requirements from broad mutable inputs after
/// placement has committed to the body's size.
struct MoiSampledAccessEmissionPlan {
  bool sampled_check = false;
  ConSanDelayMode delay_mode = ConSanDelayMode::Nop;
  uint32_t delay_count = 0;
  uint32_t delay_variable_source = 0;
  uint64_t report_buffer_address = 0;
  uint64_t report_generation = 0;
  uint32_t runtime_sample_stride = 1;
  uint32_t runtime_sample_offset = 0;
  std::optional<uint16_t> exec_save_sgpr;
  bool automatic_private_epoch = false;
  ConSanMoiPersistentSgprState persistent_sgprs;
  consan_moi_detail::ConSanMoiReportDispatchIdSource dispatch_id;
  ConSanMoiWorkgroupSources workgroup_sources;
  ConSanMoiOwnerEpochVgprSources owner_epoch_vgprs;
  uint16_t scratch_vgpr = 0;
  uint16_t base_scratch_vgpr_count = 0;
  uint16_t scratch_vgpr_count = 0;
  bool supports_native_lds_spill_recovery = false;
  uint32_t first_record_index = 0;
  std::optional<uint32_t> prior_first_record_index;
  uint32_t prior_access_range_count = 0;
  uint32_t window_bank_count = 1;
  size_t sampled_causal_windows_offset = 0;
  size_t sampled_watchpoints_offset = 0;
  size_t sampled_pending_acquires_offset = 0;
  uint32_t pending_acquire_owner_bank_count = 0;
  bool spill_overlaps_guest_operands = false;
  bool spill_backed_operand_recovery = false;
  std::optional<uint32_t> private_epoch_offset;
  std::optional<consan_detail::MoiWorkitemOwnerDerivationPlan> owner_derivation;
  bool runtime_workgroup_gate_in_body = false;
};

[[nodiscard]] bool
append_sampled_window_bank_index(std::vector<uint32_t> &words,
                                 const consan_moi_detail::ConSanMoiReportDispatchIdSource &dispatch,
                                 const ConSanMoiWorkgroupSources &workgroup_sources,
                                 uint32_t bank_count, uint16_t bank_vgpr, uint16_t temporary_vgpr,
                                 uint16_t owner_vgpr, rj_code_arch_t arch);

[[nodiscard]] std::optional<std::vector<uint32_t>> build_direct_sampled_watchpoint_words(
    std::span<const uint8_t> bytes, const ConSanMoiCandidate &candidate,
    const MoiSampledAccessEmissionPlan &plan, const VgprSpillSequence *spill, rj_code_arch_t arch,
    std::vector<std::string> &errors, uint32_t *guest_instruction_offset = nullptr,
    uint32_t *guest_instruction_word_count = nullptr);

} // namespace rocjitsu::consan_moi_impl

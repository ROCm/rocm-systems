// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_dynamic_record_emission.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/consan/consan_capability_contract.h"
#include "rocjitsu/code/patch/consan/consan_moi_report_emission.h"
#include "rocjitsu/code/patch/instruction_sequence.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"

namespace rocjitsu::consan_moi_detail {
namespace {

[[nodiscard]] bool append_moi_global_atomic_wait(std::vector<uint32_t> &words,
                                                 rj_code_arch_t arch) {
  const ConSanTargetProfile *target = consan_target_profile(arch);
  return target != nullptr && consan_detail::append_moi_global_atomic_completion(words, *target);
}

} // namespace

[[nodiscard]] bool append_atomic_fetch_add_one_u32(std::vector<uint32_t> &words,
                                                   uint64_t counter_address, uint16_t result_vgpr,
                                                   uint16_t scratch_vgpr, rj_code_arch_t arch) {
  const ConSanTargetProfile *target = consan_target_profile(arch);
  return target != nullptr &&
         consan_detail::append_moi_atomic_counter_increment(words,
                                                            {
                                                                .counter_address = counter_address,
                                                                .result_vgpr = result_vgpr,
                                                                .address_vgpr = scratch_vgpr,
                                                            },
                                                            *target);
}

[[nodiscard]] bool append_atomic_or_u32_literal(std::vector<uint32_t> &words, uint64_t address,
                                                uint32_t value, uint16_t scratch_vgpr,
                                                rj_code_arch_t arch) {
  const auto mov_address_lo =
      instrumentation::build_v_mov_b32_literal(scratch_vgpr, static_cast<uint32_t>(address), arch);
  const auto mov_address_hi = instrumentation::build_v_mov_b32_literal(
      static_cast<uint16_t>(scratch_vgpr + 1u), static_cast<uint32_t>(address >> 32u), arch);
  const auto mov_value = instrumentation::build_v_mov_b32_literal(
      static_cast<uint16_t>(scratch_vgpr + 2u), value, arch);
  const auto atomic_or = instrumentation::build_flat_atomic_or_u32(
      scratch_vgpr, static_cast<uint16_t>(scratch_vgpr + 2u),
      static_cast<uint16_t>(scratch_vgpr + 2u), /*return_old_value=*/true, kAmdGpuScopeDevice,
      arch);
  InstructionSequence sequence(words);
  return sequence.emit_all(mov_address_lo, mov_address_hi, mov_value, atomic_or) &&
         append_moi_global_atomic_wait(words, arch);
}

[[nodiscard]] bool append_atomic_load_u32(std::vector<uint32_t> &words, uint16_t address_vgpr,
                                          uint16_t result_vgpr, rj_code_arch_t arch) {
  const auto atomic_load = instrumentation::build_flat_atomic_add_u32(
      address_vgpr, result_vgpr, result_vgpr, /*return_old_value=*/true, kAmdGpuScopeDevice, arch);
  const uint32_t zero = build_v_mov_b32_e32(result_vgpr, scalar_positive_inline_u32(0), arch);
  InstructionSequence sequence(words);
  return sequence.emit_all(zero, atomic_load) && append_moi_global_atomic_wait(words, arch);
}

bool append_select_first_active_lane(std::vector<uint32_t> &words, uint16_t lane_rank_vgpr,
                                     uint16_t active_exec_sgpr, uint16_t saved_exec_sgpr,
                                     rj_code_arch_t arch) {
  InstructionSequence sequence(words);
  sequence.append(
      instrumentation::build_s_mov_b64(active_exec_sgpr, kAmdGpuExecLo, arch),
      instrumentation::build_salu_to_valu_dependency_wait(arch),
      instrumentation::build_v_mbcnt_lo_u32_b32(lane_rank_vgpr, active_exec_sgpr,
                                                scalar_positive_inline_u32(0), arch),
      instrumentation::build_v_mbcnt_hi_u32_b32(lane_rank_vgpr,
                                                static_cast<uint16_t>(active_exec_sgpr + 1u),
                                                vector_source_vgpr(lane_rank_vgpr), arch),
      instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), lane_rank_vgpr, arch),
      instrumentation::build_s_and_saveexec_b64(saved_exec_sgpr, kAmdGpuVccLo, arch));
  return sequence.finish();
}

// Publish only enough aggregate evidence to prove that the workgroup-local
// Inline detector executed. Exact per-access accounting is not part of this
// counter's host contract, and issuing a device-scope atomic for every exact
// LDS cell creates a severe contention cliff in wide, hot kernels. Once any
// publisher has made the counter nonzero, later publishers take the cached
// load-only path. A bounded startup race may increment it more than once; the
// value remains an honest nonzero publication count without serializing the
// workload on one global atomic.
[[nodiscard]] static bool
append_publish_visible_evidence_if_zero(std::vector<uint32_t> &words, uint64_t counter_address,
                                        uint16_t result_vgpr, uint16_t scratch_vgpr,
                                        uint16_t exec_save_sgpr, rj_code_arch_t arch) {
  const auto mov_address_lo = instrumentation::build_v_mov_b32_literal(
      scratch_vgpr, static_cast<uint32_t>(counter_address), arch);
  const auto mov_address_hi =
      instrumentation::build_v_mov_b32_literal(static_cast<uint16_t>(scratch_vgpr + 1u),
                                               static_cast<uint32_t>(counter_address >> 32u), arch);
  const auto load = instrumentation::build_flat_load_b32(scratch_vgpr, result_vgpr, arch);
  const auto is_zero =
      instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), result_vgpr, arch);
  const auto select_unpublished =
      instrumentation::build_s_and_saveexec_b64(exec_save_sgpr, kAmdGpuVccLo, arch);
  const auto wait = instrumentation::build_s_wait_global_load0(arch);
  InstructionSequence sequence(words);
  if (!sequence.emit_all(mov_address_lo, mov_address_hi, load, wait, is_zero, select_unpublished))
    return false;
  return append_atomic_fetch_add_one_u32(words, counter_address, result_vgpr, scratch_vgpr, arch);
}

[[nodiscard]] MoiVisibleEvidencePublicationResult
append_publish_first_active_lane_visible_evidence_if_zero(
    std::vector<uint32_t> &words, uint64_t counter_address, uint16_t result_vgpr,
    uint16_t address_vgpr, uint16_t active_exec_sgpr, uint16_t temporary_exec_sgpr,
    rj_code_arch_t arch) {
  if (!append_select_first_active_lane(words, result_vgpr, active_exec_sgpr, temporary_exec_sgpr,
                                       arch)) {
    return MoiVisibleEvidencePublicationResult::ElectionUnsupported;
  }
  if (!append_publish_visible_evidence_if_zero(words, counter_address, result_vgpr, address_vgpr,
                                               temporary_exec_sgpr, arch)) {
    return MoiVisibleEvidencePublicationResult::PublicationUnsupported;
  }
  return MoiVisibleEvidencePublicationResult::Appended;
}

[[nodiscard]] bool append_dynamic_record_address(std::vector<uint32_t> &words,
                                                 const DynamicRecordLayout &layout,
                                                 uint64_t field_address, uint16_t slot_vgpr,
                                                 uint16_t scratch_vgpr, rj_code_arch_t arch) {
  const ConSanTargetProfile *target = consan_target_profile(arch);
  return target != nullptr &&
         consan_detail::append_moi_indexed_address(words,
                                                   {
                                                       .table_address = field_address,
                                                       .stride_bytes = layout.stride_bytes,
                                                       .address_vgpr = scratch_vgpr,
                                                       .index_vgpr = slot_vgpr,
                                                   },
                                                   *target);
}

DynamicRecordEmitter::DynamicRecordEmitter(std::vector<uint32_t> &words,
                                           const DynamicRecordLayout &layout, uint64_t record_base,
                                           uint16_t slot_vgpr, uint16_t scratch_vgpr,
                                           rj_code_arch_t arch)
    : words_(words), layout_(layout), record_base_(record_base), slot_vgpr_(slot_vgpr),
      scratch_vgpr_(scratch_vgpr), arch_(arch), initial_size_(words.size()) {}

DynamicRecordEmitter::~DynamicRecordEmitter() {
  if (failed_)
    words_.resize(initial_size_);
}

void DynamicRecordEmitter::require(bool success) {
  if (failed_ || success)
    return;
  words_.resize(initial_size_);
  failed_ = true;
}

DynamicRecordEmitter &DynamicRecordEmitter::vgpr(size_t field_offset, uint16_t value_vgpr) {
  if (failed_)
    return *this;
  if (value_vgpr == scratch_vgpr_ || value_vgpr == static_cast<uint16_t>(scratch_vgpr_ + 1u)) {
    require(false);
    return *this;
  }
  const auto store = instrumentation::build_flat_store_b32(scratch_vgpr_, value_vgpr, arch_);
  require(store && append_dynamic_record_address(words_, layout_, record_base_ + field_offset,
                                                 slot_vgpr_, scratch_vgpr_, arch_));
  if (!failed_)
    words_.insert(words_.end(), store->begin(), store->end());
  return *this;
}

DynamicRecordEmitter &DynamicRecordEmitter::scalar(size_t field_offset, uint16_t scalar_src) {
  if (failed_)
    return *this;
  const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr_ + layout_.value_vgpr_offset);
  require(value_vgpr != slot_vgpr_ &&
          append_dynamic_record_address(words_, layout_, record_base_ + field_offset, slot_vgpr_,
                                        scratch_vgpr_, arch_));
  if (failed_)
    return *this;
  words_.push_back(build_v_mov_b32_e32(value_vgpr, scalar_src, arch_));
  const auto store = instrumentation::build_flat_store_b32(scratch_vgpr_, value_vgpr, arch_);
  require(store.has_value());
  if (!failed_)
    words_.insert(words_.end(), store->begin(), store->end());
  return *this;
}

DynamicRecordEmitter &DynamicRecordEmitter::literal(size_t field_offset, uint32_t value) {
  if (failed_)
    return *this;
  const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr_ + layout_.value_vgpr_offset);
  require(value_vgpr != slot_vgpr_ &&
          append_dynamic_record_address(words_, layout_, record_base_ + field_offset, slot_vgpr_,
                                        scratch_vgpr_, arch_));
  if (failed_)
    return *this;
  const auto mov_value = instrumentation::build_v_mov_b32_literal(value_vgpr, value, arch_);
  require(mov_value.has_value());
  if (failed_)
    return *this;
  words_.insert(words_.end(), mov_value->begin(), mov_value->end());
  const auto store = instrumentation::build_flat_store_b32(scratch_vgpr_, value_vgpr, arch_);
  require(store.has_value());
  if (!failed_)
    words_.insert(words_.end(), store->begin(), store->end());
  return *this;
}

DynamicRecordEmitter &DynamicRecordEmitter::private_value(size_t field_offset,
                                                          uint32_t private_offset) {
  if (failed_)
    return *this;
  const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr_ + layout_.value_vgpr_offset);
  const auto load = instrumentation::build_private_load_b32(value_vgpr, private_offset, arch_);
  const auto wait = instrumentation::build_s_wait_private_load0(arch_);
  require(load.has_value() && wait.has_value());
  if (failed_)
    return *this;
  words_.insert(words_.end(), load->begin(), load->end());
  words_.push_back(*wait);
  return vgpr(field_offset, value_vgpr);
}

DynamicRecordEmitter &
DynamicRecordEmitter::dispatch_id(size_t field_offset,
                                  const ConSanMoiReportDispatchIdSource &source) {
  if (failed_)
    return *this;
  if (!source.is_well_formed()) {
    require(false);
    return *this;
  }
  for (const bool high_word : {false, true}) {
    const size_t word_offset = field_offset + (high_word ? sizeof(uint32_t) : 0u);
    const uint16_t register_word = high_word ? 1u : 0u;
    if (source.sgpr)
      scalar(word_offset, static_cast<uint16_t>(*source.sgpr + register_word));
    else if (source.vgpr)
      vgpr(word_offset, static_cast<uint16_t>(*source.vgpr + register_word));
    else if (source.private_offset)
      private_value(word_offset,
                    *source.private_offset + (high_word ? SpillManager::kSlotBytes : 0u));
    else
      literal(word_offset, static_cast<uint32_t>(*source.literal >> (high_word ? 32u : 0u)));
  }
  return *this;
}

DynamicRecordEmitter &DynamicRecordEmitter::workgroup(size_t field_offset,
                                                      const ConSanMoiWorkgroupSource &source) {
  if (failed_)
    return *this;
  if (!source.is_well_formed()) {
    require(false);
    return *this;
  }
  if (!source.has_value())
    return *this;
  const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr_ + layout_.value_vgpr_offset);
  require(consan_detail::append_workgroup_source_value(words_, source, value_vgpr, arch_));
  if (!failed_)
    vgpr(field_offset, value_vgpr);
  return *this;
}

DynamicRecordEmitter &DynamicRecordEmitter::event_index(size_t field_offset,
                                                        uint64_t counter_address) {
  if (failed_)
    return *this;
  const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr_ + layout_.value_vgpr_offset);
  require(
      append_atomic_fetch_add_one_u32(words_, counter_address, value_vgpr, scratch_vgpr_, arch_));
  if (!failed_)
    vgpr(field_offset, value_vgpr);
  return *this;
}

[[nodiscard]] bool append_dynamic_diagnostic_record_address(std::vector<uint32_t> &words,
                                                            uint64_t address, uint16_t slot,
                                                            uint16_t address_vgpr,
                                                            rj_code_arch_t arch) {
  return append_dynamic_record_address(words, kDiagnosticRecordLayout, address, slot, address_vgpr,
                                       arch);
}

} // namespace rocjitsu::consan_moi_detail

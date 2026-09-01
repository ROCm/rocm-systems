// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_dynamic_record_emission.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/consan/consan_moi_report_emission.h"
#include "rocjitsu/code/patch/instruction_sequence.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"

namespace rocjitsu::consan_moi_detail {
namespace {

inline constexpr uint16_t kRdna4VccLo = 106;
inline constexpr uint8_t kRdna4ScopeDevice = 2;

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
      static_cast<uint16_t>(scratch_vgpr + 2u), /*return_old_value=*/true, kRdna4ScopeDevice, arch);
  InstructionSequence sequence(words);
  return sequence.emit_all(mov_address_lo, mov_address_hi, mov_value, atomic_or) &&
         append_moi_global_atomic_wait(words, arch);
}

[[nodiscard]] bool append_atomic_load_u32(std::vector<uint32_t> &words, uint16_t address_vgpr,
                                          uint16_t result_vgpr, rj_code_arch_t arch) {
  const auto atomic_load = instrumentation::build_flat_atomic_add_u32(
      address_vgpr, result_vgpr, result_vgpr, /*return_old_value=*/true, kRdna4ScopeDevice, arch);
  const uint32_t zero = build_v_mov_b32_e32(result_vgpr, scalar_positive_inline_u32(0), arch);
  InstructionSequence sequence(words);
  return sequence.emit_all(zero, atomic_load) && append_moi_global_atomic_wait(words, arch);
}

// Publish only enough aggregate evidence to prove that the workgroup-local
// Inline detector executed. Exact per-access accounting is not part of this
// counter's host contract, and issuing a device-scope atomic for every exact
// LDS cell creates a severe contention cliff in wide, hot kernels. Once any
// publisher has made the counter nonzero, later publishers take the cached
// load-only path. A bounded startup race may increment it more than once; the
// value remains an honest nonzero publication count without serializing the
// workload on one global atomic.
[[nodiscard]] bool
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
      instrumentation::build_s_and_saveexec_b64(exec_save_sgpr, kRdna4VccLo, arch);
  const auto wait = instrumentation::build_s_wait_global_load0(arch);
  InstructionSequence sequence(words);
  if (!sequence.emit_all(mov_address_lo, mov_address_hi, load, wait, is_zero, select_unpublished))
    return false;
  return append_atomic_fetch_add_one_u32(words, counter_address, result_vgpr, scratch_vgpr, arch);
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

[[nodiscard]] bool append_dynamic_record_store_u32_vgpr(std::vector<uint32_t> &words,
                                                        const DynamicRecordLayout &layout,
                                                        uint64_t field_address, uint16_t value_vgpr,
                                                        uint16_t slot_vgpr, uint16_t scratch_vgpr,
                                                        rj_code_arch_t arch) {
  if (value_vgpr == scratch_vgpr || value_vgpr == static_cast<uint16_t>(scratch_vgpr + 1u)) {
    return false;
  }
  const auto store = instrumentation::build_flat_store_b32(scratch_vgpr, value_vgpr, arch);
  if (!store ||
      !append_dynamic_record_address(words, layout, field_address, slot_vgpr, scratch_vgpr, arch))
    return false;
  words.insert(words.end(), store->begin(), store->end());
  return true;
}

[[nodiscard]] bool append_dynamic_record_store_u32_literal(
    std::vector<uint32_t> &words, const DynamicRecordLayout &layout, uint64_t field_address,
    uint32_t value, uint16_t slot_vgpr, uint16_t scratch_vgpr, rj_code_arch_t arch) {
  const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr + layout.value_vgpr_offset);
  if (value_vgpr == slot_vgpr ||
      !append_dynamic_record_address(words, layout, field_address, slot_vgpr, scratch_vgpr, arch))
    return false;
  const auto mov_value = instrumentation::build_v_mov_b32_literal(value_vgpr, value, arch);
  if (!mov_value)
    return false;
  words.insert(words.end(), mov_value->begin(), mov_value->end());
  const auto store = instrumentation::build_flat_store_b32(scratch_vgpr, value_vgpr, arch);
  if (!store)
    return false;
  words.insert(words.end(), store->begin(), store->end());
  return true;
}

[[nodiscard]] bool append_dynamic_record_store_u32_scalar_src(
    std::vector<uint32_t> &words, const DynamicRecordLayout &layout, uint64_t field_address,
    uint16_t scalar_src, uint16_t slot_vgpr, uint16_t scratch_vgpr, rj_code_arch_t arch) {
  const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr + layout.value_vgpr_offset);
  if (value_vgpr == slot_vgpr ||
      !append_dynamic_record_address(words, layout, field_address, slot_vgpr, scratch_vgpr, arch))
    return false;
  words.push_back(build_v_mov_b32_e32(value_vgpr, scalar_src, arch));
  const auto store = instrumentation::build_flat_store_b32(scratch_vgpr, value_vgpr, arch);
  if (!store)
    return false;
  words.insert(words.end(), store->begin(), store->end());
  return true;
}

[[nodiscard]] bool append_dynamic_record_store_moi_report_dispatch_id_pair(
    std::vector<uint32_t> &words, const DynamicRecordLayout &layout, uint64_t low_field_address,
    const ConSanMoiReportDispatchIdSource &source, uint16_t slot_vgpr, uint16_t scratch_vgpr,
    rj_code_arch_t arch) {
  if (!source.is_well_formed())
    return false;
  for (const bool high_word : {false, true}) {
    const uint64_t field_address = low_field_address + (high_word ? sizeof(uint32_t) : 0u);
    const uint16_t register_word = high_word ? 1u : 0u;
    const bool stored =
        source.sgpr
            ? append_dynamic_record_store_u32_scalar_src(
                  words, layout, field_address, static_cast<uint16_t>(*source.sgpr + register_word),
                  slot_vgpr, scratch_vgpr, arch)
        : source.vgpr
            ? append_dynamic_record_store_u32_vgpr(
                  words, layout, field_address, static_cast<uint16_t>(*source.vgpr + register_word),
                  slot_vgpr, scratch_vgpr, arch)
            : append_dynamic_record_store_u32_literal(
                  words, layout, field_address,
                  static_cast<uint32_t>(*source.literal >> (high_word ? 32u : 0u)), slot_vgpr,
                  scratch_vgpr, arch);
    if (!stored)
      return false;
  }
  return true;
}

[[nodiscard]] bool append_dynamic_record_store_workgroup_source(
    std::vector<uint32_t> &words, const DynamicRecordLayout &layout, uint64_t field_address,
    const ConSanMoiWorkgroupSource &source, uint16_t slot_vgpr, uint16_t scratch_vgpr,
    rj_code_arch_t arch) {
  if (!source.is_well_formed())
    return false;
  if (!source.has_value())
    return true;
  const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr + layout.value_vgpr_offset);
  if (!consan_detail::append_workgroup_source_value(words, source, value_vgpr, arch))
    return false;
  return append_dynamic_record_store_u32_vgpr(words, layout, field_address, value_vgpr, slot_vgpr,
                                              scratch_vgpr, arch);
}

[[nodiscard]] bool append_dynamic_record_event_index_store(
    std::vector<uint32_t> &words, const DynamicRecordLayout &layout, uint64_t counter_address,
    uint64_t field_address, uint16_t slot_vgpr, uint16_t scratch_vgpr, rj_code_arch_t arch) {
  const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr + layout.value_vgpr_offset);
  return append_atomic_fetch_add_one_u32(words, counter_address, value_vgpr, scratch_vgpr, arch) &&
         append_dynamic_record_store_u32_vgpr(words, layout, field_address, value_vgpr, slot_vgpr,
                                              scratch_vgpr, arch);
}

[[nodiscard]] bool append_dynamic_diagnostic_record_address(std::vector<uint32_t> &words,
                                                            uint64_t address, uint16_t slot,
                                                            uint16_t address_vgpr,
                                                            rj_code_arch_t arch) {
  return append_dynamic_record_address(words, kDiagnosticRecordLayout, address, slot, address_vgpr,
                                       arch);
}

} // namespace rocjitsu::consan_moi_detail

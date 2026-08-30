// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_record_event_emission.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/consan/consan_capability_contract.h"
#include "rocjitsu/code/patch/consan/consan_moi_dynamic_record_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_memory_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_native_abi.h"
#include "rocjitsu/code/patch/consan/consan_moi_relocation.h"
#include "rocjitsu/code/patch/consan/consan_moi_report_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_runtime_workgroup_gate.h"
#include "rocjitsu/code/patch/consan/consan_vgpr_bank_state.h"
#include "rocjitsu/code/patch/instruction_sequence.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <ranges>

namespace rocjitsu::consan_moi_impl {

using consan_detail::MoiAtomicEvidenceSitePlan;
using consan_detail::MoiBarrierEvidenceSitePlan;
using consan_detail::MoiFenceEvidenceSitePlan;
using consan_detail::MoiSpecialStateSgprs;
using consan_detail::MoiWorkitemOwnerDerivationPlan;
using consan_detail::reject_atomic_candidate_scratch_overlap;
using consan_detail::reject_optional_scratch_range_overlap;
using consan_moi_detail::append_atomic_fetch_add_one_u32;
using consan_moi_detail::append_dynamic_record_event_index_store;
using consan_moi_detail::append_dynamic_record_store_moi_report_dispatch_id_pair;
using consan_moi_detail::append_dynamic_record_store_u32_literal;
using consan_moi_detail::append_dynamic_record_store_u32_scalar_src;
using consan_moi_detail::append_dynamic_record_store_u32_vgpr;
using consan_moi_detail::append_dynamic_record_store_workgroup_source;
using consan_moi_detail::ConSanMoiLiteralDispatchIdPolicy;
using consan_moi_detail::kAtomicRecordLayout;
using consan_moi_detail::kBarrierRecordLayout;
using consan_moi_detail::kFenceRecordLayout;

[[nodiscard]] std::optional<std::vector<uint32_t>> build_barrier_record_cave_words(
    std::span<const uint8_t> bytes, const MoiBarrierEvidenceSitePlan &candidate,
    const MoiRecordEventEmissionPlan &options, const VgprSpillSequence *spill,
    const SgprSpillSequence *scalar_spill, rj_code_arch_t arch, uint32_t barrier_record_capacity,
    size_t barrier_records_offset, uint32_t original_barrier_word, uint64_t cave_text_offset,
    uint64_t return_text_offset, const std::optional<MoiWorkitemOwnerDerivationPlan> &derived_owner,
    std::vector<std::string> &errors) {
  if (!options.scratch_vgpr) {
    errors.emplace_back("ConSan MOI barrier record patch requires RJ_CONSAN_TMP_VGPR");
    return std::nullopt;
  }
  if (static_cast<uint32_t>(*options.scratch_vgpr) + 6u > kMaxVgprs) {
    errors.emplace_back("ConSan MOI barrier record patch needs six scratch VGPRs");
    return std::nullopt;
  }
  if (!options.moi_exec_save_sgpr) {
    errors.emplace_back("ConSan MOI barrier record patch requires RJ_CONSAN_MOI_EXEC_SAVE_SGPR");
    return std::nullopt;
  }
  if (*options.moi_exec_save_sgpr > 100u || *options.moi_exec_save_sgpr % 2u != 0u) {
    errors.emplace_back(
        "ConSan MOI barrier record patch requires an even RJ_CONSAN_MOI_EXEC_SAVE_SGPR in 0..100");
    return std::nullopt;
  }
  if (reject_optional_scratch_range_overlap(options.moi_owner_vgpr, *options.scratch_vgpr, 6,
                                            "MOI owner", errors))
    return std::nullopt;

  std::optional<uint16_t> derived_owner_vgpr;
  std::vector<uint32_t> derived_owner_words;
  if (!options.moi_owner_vgpr && !options.moi_persistent_sgprs.owner) {
    if (!derived_owner) {
      errors.emplace_back("ConSan MOI barrier record patch requires a planned owner derivation");
      return std::nullopt;
    }
    const uint16_t value_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 5u);
    const auto owner = build_moi_workitem_owner_derivation(*derived_owner, value_vgpr, arch,
                                                           "barrier record patch", errors);
    if (!owner)
      return std::nullopt;
    derived_owner_vgpr = owner->vgpr;
    derived_owner_words = owner->words;
  }
  const ConSanMoiWorkgroupSources &workgroup_sources = options.workgroup_sources;

  std::vector<uint32_t> words;
  const uint64_t base = *options.moi_report_buffer_address;
  const uint64_t barrier_record_base = base + barrier_records_offset;
  const uint16_t slot_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 2u);
  const uint16_t lane_rank_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 5u);
  constexpr uint16_t kScalarInlineMinusOne = 0xC1;

  const ConSanTargetProfile *target = consan_target_profile(arch);
  const std::optional<uint16_t> vgpr_msb_mode =
      target != nullptr && target->has_selectable_vgpr_bank
          ? consan_selectable_vgpr_bank_mode_at(arch, bytes, candidate.text_file_offset,
                                                candidate.container_entry_text_offset,
                                                candidate.site.file_offset)
          : std::nullopt;
  const bool select_low_vgpr_bank = vgpr_msb_mode.value_or(0u) != 0u;
  words.reserve(
      186 + (spill ? spill->save_words.size() + spill->restore_words.size() : 0u) +
      (scalar_spill ? scalar_spill->save_words.size() + scalar_spill->restore_words.size() : 0u));
  const bool runtime_workgroup_gate = options.runtime_workgroup_gate.has_value();
  if (runtime_workgroup_gate) {
    auto gate = build_moi_runtime_workgroup_gate_call_words(
        bytes.subspan(candidate.site.file_offset, candidate.site.size),
        *options.runtime_workgroup_gate, workgroup_sources, cave_text_offset, return_text_offset,
        moi_runtime_workgroup_gate_reserved_words(
            candidate.site.size, workgroup_sources.cluster_workgroup_id.has_value()),
        arch);
    if (!gate) {
      errors.emplace_back(
          "ConSan MOI barrier record patch could not encode its runtime workgroup gate");
      return std::nullopt;
    }
    words = std::move(*gate);
  }
  if (select_low_vgpr_bank)
    words.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        static_cast<uint8_t>(*vgpr_msb_mode), 0u, arch));
  if (spill)
    words.insert(words.end(), spill->save_words.begin(), spill->save_words.end());
  if (scalar_spill)
    words.insert(words.end(), scalar_spill->save_words.begin(), scalar_spill->save_words.end());
  if (!append_save_moi_special_state(words, moi_special_state_sgprs(options), arch)) {
    errors.emplace_back("ConSan MOI barrier record patch could not save VCC/SCC");
    return std::nullopt;
  }
  const auto mbcnt_lo = instrumentation::build_v_mbcnt_lo_u32_b32(
      lane_rank_vgpr, kScalarInlineMinusOne, scalar_positive_inline_u32(0), arch);
  const auto mbcnt_hi = instrumentation::build_v_mbcnt_hi_u32_b32(
      lane_rank_vgpr, kScalarInlineMinusOne, vector_source_vgpr(lane_rank_vgpr), arch);
  const auto first_active_lane =
      instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), lane_rank_vgpr, arch);
  const auto save_exec =
      instrumentation::build_s_and_saveexec_b64(*options.moi_exec_save_sgpr, kRdna4VccLo, arch);
  if (!mbcnt_lo || !mbcnt_hi || !first_active_lane || !save_exec) {
    errors.emplace_back("ConSan MOI barrier record patch could not encode EXEC narrowing");
    return std::nullopt;
  }
  words.insert(words.end(), mbcnt_lo->begin(), mbcnt_lo->end());
  words.insert(words.end(), mbcnt_hi->begin(), mbcnt_hi->end());
  words.push_back(*first_active_lane);
  words.push_back(*save_exec);

  if (!append_atomic_fetch_add_one_u32(words,
                                       base + offsetof(ConSanMoiReportHeader, barrier_record_count),
                                       slot_vgpr, *options.scratch_vgpr, arch)) {
    errors.emplace_back("ConSan MOI barrier record patch could not encode dynamic slot reserve");
    return std::nullopt;
  }

  const uint16_t value_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 5u);
  const auto mov_capacity =
      instrumentation::build_v_mov_b32_literal(value_vgpr, barrier_record_capacity, arch);
  const auto slot_in_capacity =
      instrumentation::build_v_cmp_gt_u32_vcc(vector_source_vgpr(value_vgpr), slot_vgpr, arch);
  if (!mov_capacity || !slot_in_capacity) {
    errors.emplace_back("ConSan MOI barrier record patch could not encode capacity guard");
    return std::nullopt;
  }
  words.insert(words.end(), mov_capacity->begin(), mov_capacity->end());
  words.push_back(*slot_in_capacity);

  std::vector<uint32_t> record_words;
  record_words.reserve(128);
  record_words.insert(record_words.end(), derived_owner_words.begin(), derived_owner_words.end());
  if ((derived_owner_vgpr && !append_dynamic_record_store_u32_vgpr(
                                 record_words, kBarrierRecordLayout,
                                 barrier_record_base + offsetof(ConSanMoiBarrierRecord, wave_id),
                                 *derived_owner_vgpr, slot_vgpr, *options.scratch_vgpr, arch)) ||
      !append_dynamic_record_store_moi_report_dispatch_id_pair(
          record_words, kBarrierRecordLayout,
          barrier_record_base + offsetof(ConSanMoiBarrierRecord, generation), options, slot_vgpr,
          *options.scratch_vgpr, arch, ConSanMoiLiteralDispatchIdPolicy::ExternalBindingAllowed) ||
      !append_dynamic_record_event_index_store(
          record_words, kBarrierRecordLayout, base + offsetof(ConSanMoiReportHeader, event_counter),
          barrier_record_base + offsetof(ConSanMoiBarrierRecord, event_index), slot_vgpr,
          *options.scratch_vgpr, arch) ||
      !append_dynamic_record_store_workgroup_source(
          record_words, kBarrierRecordLayout,
          barrier_record_base + offsetof(ConSanMoiBarrierRecord, workgroup_x), workgroup_sources.x,
          slot_vgpr, *options.scratch_vgpr, arch) ||
      !append_dynamic_record_store_workgroup_source(
          record_words, kBarrierRecordLayout,
          barrier_record_base + offsetof(ConSanMoiBarrierRecord, workgroup_y), workgroup_sources.y,
          slot_vgpr, *options.scratch_vgpr, arch) ||
      !append_dynamic_record_store_workgroup_source(
          record_words, kBarrierRecordLayout,
          barrier_record_base + offsetof(ConSanMoiBarrierRecord, workgroup_z), workgroup_sources.z,
          slot_vgpr, *options.scratch_vgpr, arch) ||
      !append_dynamic_record_store_u32_scalar_src(
          record_words, kBarrierRecordLayout,
          barrier_record_base + offsetof(ConSanMoiBarrierRecord, lane_mask),
          *options.moi_exec_save_sgpr, slot_vgpr, *options.scratch_vgpr, arch) ||
      !append_dynamic_record_store_u32_scalar_src(
          record_words, kBarrierRecordLayout,
          barrier_record_base + offsetof(ConSanMoiBarrierRecord, lane_mask) + sizeof(uint32_t),
          static_cast<uint16_t>(*options.moi_exec_save_sgpr + 1u), slot_vgpr, *options.scratch_vgpr,
          arch) ||
      (options.moi_owner_vgpr &&
       !append_dynamic_record_store_u32_vgpr(
           record_words, kBarrierRecordLayout,
           barrier_record_base + offsetof(ConSanMoiBarrierRecord, wave_id), *options.moi_owner_vgpr,
           slot_vgpr, *options.scratch_vgpr, arch)) ||
      (options.moi_persistent_sgprs.owner &&
       !append_dynamic_record_store_u32_scalar_src(
           record_words, kBarrierRecordLayout,
           barrier_record_base + offsetof(ConSanMoiBarrierRecord, wave_id),
           *options.moi_persistent_sgprs.owner, slot_vgpr, *options.scratch_vgpr, arch)) ||
      !append_dynamic_record_store_u32_literal(
          record_words, kBarrierRecordLayout,
          barrier_record_base + offsetof(ConSanMoiBarrierRecord, instruction_offset),
          static_cast<uint32_t>(candidate.site.text_offset), slot_vgpr, *options.scratch_vgpr,
          arch)) {
    errors.emplace_back("ConSan MOI barrier record patch could not encode record stores");
    return std::nullopt;
  }

  if (record_words.size() > static_cast<size_t>(std::numeric_limits<int16_t>::max())) {
    errors.emplace_back("ConSan MOI barrier record overflow branch is out of range");
    return std::nullopt;
  }
  const auto skip_record =
      instrumentation::build_s_cbranch_vccz(static_cast<int16_t>(record_words.size()), arch);
  const auto restore_exec =
      instrumentation::build_s_mov_b64(kRdna4ExecLo, *options.moi_exec_save_sgpr, arch);
  if (!skip_record || !restore_exec) {
    errors.emplace_back("ConSan MOI barrier record patch could not encode EXEC restore");
    return std::nullopt;
  }
  words.push_back(*skip_record);
  words.insert(words.end(), record_words.begin(), record_words.end());
  words.push_back(*restore_exec);
  if (!append_restore_moi_special_state(words, moi_special_state_sgprs(options), arch)) {
    errors.emplace_back("ConSan MOI barrier record patch could not restore VCC/SCC");
    return std::nullopt;
  }

  if (scalar_spill) {
    words.insert(words.end(), scalar_spill->restore_words.begin(),
                 scalar_spill->restore_words.end());
  }
  if (spill)
    words.insert(words.end(), spill->restore_words.begin(), spill->restore_words.end());
  if (select_low_vgpr_bank)
    words.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        0u, static_cast<uint8_t>(*vgpr_msb_mode), arch));
  words.push_back(original_barrier_word);

  if (!append_moi_direct_or_indirect_return(words, cave_text_offset, return_text_offset, options,
                                            arch)) {
    errors.emplace_back("ConSan MOI barrier record could not encode its return");
    return std::nullopt;
  }
  return words;
}
[[nodiscard]] std::optional<std::vector<uint32_t>> build_atomic_record_cave_words(
    std::span<const uint8_t> bytes, const MoiAtomicEvidenceSitePlan &candidate,
    const ConSanMoiAtomicAddressPlan &address_plan, const MoiRecordEventEmissionPlan &options,
    const VgprSpillSequence *spill, const SgprSpillSequence *scalar_spill, rj_code_arch_t arch,
    uint32_t record_index, uint32_t atomic_record_capacity, size_t atomic_records_offset,
    uint64_t cave_text_offset, uint64_t return_text_offset, bool already_runtime_workgroup_gated,
    const std::optional<MoiWorkitemOwnerDerivationPlan> &derived_owner,
    uint32_t &guest_instruction_offset, std::vector<std::string> &errors) {
  if (!options.scratch_vgpr) {
    errors.emplace_back("ConSan MOI atomic record patch requires RJ_CONSAN_TMP_VGPR");
    return std::nullopt;
  }
  if (!options.moi_exec_save_sgpr) {
    errors.emplace_back("ConSan MOI atomic record patch requires scalar EXEC state");
    return std::nullopt;
  }
  if (static_cast<uint32_t>(*options.scratch_vgpr) + address_plan.scratch_vgpr_count > 256u) {
    errors.emplace_back("ConSan MOI atomic record patch has an invalid scratch VGPR window");
    return std::nullopt;
  }
  if (reject_atomic_candidate_scratch_overlap(candidate.lowering_form, *options.scratch_vgpr,
                                              address_plan.scratch_vgpr_count, errors) ||
      reject_optional_scratch_range_overlap(options.moi_owner_vgpr, *options.scratch_vgpr,
                                            address_plan.scratch_vgpr_count, "MOI owner", errors) ||
      reject_optional_scratch_range_overlap(options.moi_epoch_vgpr, *options.scratch_vgpr,
                                            address_plan.scratch_vgpr_count, "MOI epoch", errors))
    return std::nullopt;

  if (!address_plan.supported()) {
    errors.emplace_back("ConSan MOI atomic record patch requires a supported address plan");
    return std::nullopt;
  }

  (void)record_index;
  std::optional<uint16_t> derived_owner_vgpr;
  std::vector<uint32_t> derived_owner_words;
  if (!options.moi_owner_vgpr && !options.moi_persistent_sgprs.owner) {
    if (!derived_owner) {
      errors.emplace_back("ConSan MOI atomic record patch requires a planned owner derivation");
      return std::nullopt;
    }
    const uint16_t value_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 4u);
    const auto owner = build_moi_workitem_owner_derivation(*derived_owner, value_vgpr, arch,
                                                           "atomic record patch", errors);
    if (!owner)
      return std::nullopt;
    derived_owner_vgpr = owner->vgpr;
    derived_owner_words = owner->words;
  }

  const ConSanMoiWorkgroupSources &workgroup_sources = options.workgroup_sources;

  if (candidate.site.file_offset > bytes.size() ||
      candidate.site.size > bytes.size() - candidate.site.file_offset) {
    errors.emplace_back("ConSan MOI atomic record patch site exceeds ELF bytes");
    return std::nullopt;
  }

  std::vector<uint32_t> words;
  const bool runtime_workgroup_gate =
      !already_runtime_workgroup_gated && options.runtime_workgroup_gate.has_value();
  if (runtime_workgroup_gate) {
    auto gate = build_moi_runtime_workgroup_gate_call_words(
        bytes.subspan(candidate.site.file_offset, candidate.site.size),
        *options.runtime_workgroup_gate, workgroup_sources, cave_text_offset, return_text_offset,
        moi_runtime_workgroup_gate_reserved_words(
            candidate.site.size, workgroup_sources.cluster_workgroup_id.has_value()),
        arch);
    if (!gate) {
      errors.emplace_back(
          "ConSan MOI atomic record patch could not encode its runtime workgroup gate");
      return std::nullopt;
    }
    words = std::move(*gate);
  }
  words.reserve(words.size() + candidate.site.size / sizeof(uint32_t) + 96u +
                (spill ? spill->save_words.size() + spill->restore_words.size() : 0u));
  if (spill)
    words.insert(words.end(), spill->save_words.begin(), spill->save_words.end());
  if (scalar_spill)
    words.insert(words.end(), scalar_spill->save_words.begin(), scalar_spill->save_words.end());
  if (address_plan.requires_materialization()) {
    const auto special_state = moi_special_state_sgprs(options);
    if (!special_state) {
      errors.emplace_back(
          "ConSan MOI atomic record address materialization requires special-state SGPRs");
      return std::nullopt;
    }
    const auto address_words = build_consan_moi_atomic_address_materialization(
        address_plan, special_state->vcc_save_sgpr, special_state->scc_save_sgpr, arch);
    if (!address_words) {
      errors.emplace_back("ConSan MOI atomic record patch could not materialize its address");
      return std::nullopt;
    }
    words.insert(words.end(), address_words->begin(), address_words->end());
  }
  const bool is_compare_exchange = consan_atomic_is_compare_exchange(candidate.site);
  std::vector<uint32_t> guest_atomic_words;
  guest_atomic_words.reserve(candidate.site.size / sizeof(uint32_t));
  for (uint64_t offset = 0; offset < candidate.site.size; offset += sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, bytes.data() + candidate.site.file_offset + offset, sizeof(word));
    guest_atomic_words.push_back(word);
  }

  const uint64_t base = *options.moi_report_buffer_address;
  const uint16_t recorded_address_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 5u);
  words.push_back(build_v_mov_b32_e32(recorded_address_vgpr,
                                      vector_source_vgpr(address_plan.result_address_vgpr), arch));
  words.push_back(build_v_mov_b32_e32(
      static_cast<uint16_t>(recorded_address_vgpr + 1u),
      vector_source_vgpr(static_cast<uint16_t>(address_plan.result_address_vgpr + 1u)), arch));

  const bool reserve_event_before_guest =
      is_compare_exchange && (candidate.event_kind == ConSanMoiAtomicEventKind::Release ||
                              candidate.event_kind == ConSanMoiAtomicEventKind::AcquireRelease);
  const uint16_t reserved_event_index_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 7u);
  if (reserve_event_before_guest) {
    if (!append_save_moi_special_state(words, moi_special_state_sgprs(options), arch)) {
      errors.emplace_back(
          "ConSan MOI release CAS could not preserve VCC/SCC before event reservation");
      return std::nullopt;
    }
    const uint16_t lane_rank_vgpr = *options.scratch_vgpr;
    const auto save_active_exec =
        instrumentation::build_s_mov_b64(*options.moi_exec_save_sgpr, kRdna4ExecLo, arch);
    const auto mbcnt_lo = instrumentation::build_v_mbcnt_lo_u32_b32(
        lane_rank_vgpr, *options.moi_exec_save_sgpr, scalar_positive_inline_u32(0), arch);
    const auto mbcnt_hi = instrumentation::build_v_mbcnt_hi_u32_b32(
        lane_rank_vgpr, static_cast<uint16_t>(*options.moi_exec_save_sgpr + 1u),
        vector_source_vgpr(lane_rank_vgpr), arch);
    const auto first_active_lane = instrumentation::build_v_cmp_eq_u32_vcc(
        scalar_positive_inline_u32(0), lane_rank_vgpr, arch);
    const auto narrow_exec =
        instrumentation::build_s_and_saveexec_b64(*options.moi_exec_save_sgpr, kRdna4VccLo, arch);
    const auto saved_exec_wait = instrumentation::build_salu_to_valu_dependency_wait(arch);
    const auto restore_exec =
        instrumentation::build_s_mov_b64(kRdna4ExecLo, *options.moi_exec_save_sgpr, arch);
    if (!save_active_exec || !mbcnt_lo || !mbcnt_hi || !first_active_lane || !narrow_exec ||
        !saved_exec_wait || !restore_exec) {
      errors.emplace_back("ConSan MOI release CAS could not select its event publisher");
      return std::nullopt;
    }
    words.push_back(*save_active_exec);
    words.push_back(*saved_exec_wait);
    words.insert(words.end(), mbcnt_lo->begin(), mbcnt_lo->end());
    words.insert(words.end(), mbcnt_hi->begin(), mbcnt_hi->end());
    words.push_back(*first_active_lane);
    words.push_back(*narrow_exec);
    if (!append_atomic_fetch_add_one_u32(words,
                                         base + offsetof(ConSanMoiReportHeader, event_counter),
                                         reserved_event_index_vgpr, *options.scratch_vgpr, arch)) {
      errors.emplace_back("ConSan MOI release CAS could not reserve its causal event index");
      return std::nullopt;
    }
    words.push_back(*restore_exec);
    if (!append_restore_moi_special_state(words, moi_special_state_sgprs(options), arch)) {
      errors.emplace_back(
          "ConSan MOI release CAS could not restore VCC/SCC after event reservation");
      return std::nullopt;
    }
  }

  const bool record_before_guest_release =
      candidate.event_kind == ConSanMoiAtomicEventKind::Release && !is_compare_exchange;
  if (!record_before_guest_release) {
    guest_instruction_offset = static_cast<uint32_t>(words.size() * sizeof(uint32_t));
    words.insert(words.end(), guest_atomic_words.begin(), guest_atomic_words.end());
    if (!append_moi_flat_load_wait(words, arch))
      return std::nullopt;
  }

  if (!append_save_moi_special_state(words, moi_special_state_sgprs(options), arch)) {
    errors.emplace_back("ConSan MOI atomic record patch could not save VCC/SCC");
    return std::nullopt;
  }

  const uint64_t atomic_record_base = base + atomic_records_offset;
  const uint16_t slot_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 2u);
  const uint16_t lane_rank_vgpr = *options.scratch_vgpr;
  if (is_compare_exchange) {
    if (!candidate.site.data_vgpr || !candidate.site.dst_vgpr) {
      errors.emplace_back("ConSan MOI atomic record CAS lacks compare/result operands");
      return std::nullopt;
    }
    const uint16_t value_word_count = static_cast<uint16_t>(candidate.site.width_bits / 32u);
    const uint16_t compare_vgpr =
        static_cast<uint16_t>(*candidate.site.data_vgpr + value_word_count);
    const auto compare_low = instrumentation::build_v_cmp_eq_u32_vcc(
        vector_source_vgpr(compare_vgpr), *candidate.site.dst_vgpr, arch);
    if (!compare_low) {
      errors.emplace_back("ConSan MOI atomic record CAS could not capture its outcome mask");
      return std::nullopt;
    }
    const uint16_t success_lo = static_cast<uint16_t>(*options.scratch_vgpr + 3u);
    const uint16_t success_hi = static_cast<uint16_t>(*options.scratch_vgpr + 4u);
    words.push_back(*compare_low);
    words.push_back(build_v_mov_b32_e32(success_lo, kRdna4VccLo, arch));
    words.push_back(build_v_mov_b32_e32(success_hi, kRdna4VccHi, arch));
    if (candidate.site.width_bits == 64u) {
      const auto compare_high = instrumentation::build_v_cmp_eq_u32_vcc(
          vector_source_vgpr(static_cast<uint16_t>(compare_vgpr + 1u)),
          static_cast<uint16_t>(*candidate.site.dst_vgpr + 1u), arch);
      const auto intersect_lo =
          instrumentation::build_v_and_b32(success_lo, kRdna4VccLo, success_lo, arch);
      const auto intersect_hi =
          instrumentation::build_v_and_b32(success_hi, kRdna4VccHi, success_hi, arch);
      if (!compare_high || !intersect_lo || !intersect_hi) {
        errors.emplace_back(
            "ConSan MOI atomic record 64-bit CAS could not combine its outcome masks");
        return std::nullopt;
      }
      words.push_back(*compare_high);
      words.push_back(*intersect_lo);
      words.push_back(*intersect_hi);
    }
    // VCC bits outside EXEC are architecturally unspecified after a vector
    // compare. Simulators commonly leave them clear, but physical gfx1201 can
    // retain arbitrary values there. Only lanes that executed the guest CAS
    // have an outcome, so canonicalize both halves before publishing them.
    const auto active_success_lo =
        instrumentation::build_v_and_b32(success_lo, kRdna4ExecLo, success_lo, arch);
    const auto active_success_hi =
        instrumentation::build_v_and_b32(success_hi, kRdna4ExecHi, success_hi, arch);
    if (!active_success_lo || !active_success_hi) {
      errors.emplace_back("ConSan MOI atomic record CAS could not mask inactive outcome lanes");
      return std::nullopt;
    }
    words.push_back(*active_success_lo);
    words.push_back(*active_success_hi);
  }
  const auto save_active_exec =
      instrumentation::build_s_mov_b64(*options.moi_exec_save_sgpr, kRdna4ExecLo, arch);
  const auto mbcnt_lo = instrumentation::build_v_mbcnt_lo_u32_b32(
      lane_rank_vgpr, *options.moi_exec_save_sgpr, scalar_positive_inline_u32(0), arch);
  const auto mbcnt_hi = instrumentation::build_v_mbcnt_hi_u32_b32(
      lane_rank_vgpr, static_cast<uint16_t>(*options.moi_exec_save_sgpr + 1u),
      vector_source_vgpr(lane_rank_vgpr), arch);
  const auto first_active_lane =
      instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), lane_rank_vgpr, arch);
  const auto save_exec =
      instrumentation::build_s_and_saveexec_b64(*options.moi_exec_save_sgpr, kRdna4VccLo, arch);
  const auto saved_exec_wait = instrumentation::build_salu_to_valu_dependency_wait(arch);
  if (!save_active_exec || !saved_exec_wait || !mbcnt_lo || !mbcnt_hi || !first_active_lane ||
      !save_exec) {
    errors.emplace_back("ConSan MOI atomic record patch could not select a wave publisher");
    return std::nullopt;
  }
  words.push_back(*save_active_exec);
  words.push_back(*saved_exec_wait);
  words.insert(words.end(), mbcnt_lo->begin(), mbcnt_lo->end());
  words.insert(words.end(), mbcnt_hi->begin(), mbcnt_hi->end());
  words.push_back(*first_active_lane);
  words.push_back(*save_exec);
  if (!options.moi_exec_save_sgpr ||
      !append_atomic_fetch_add_one_u32(words,
                                       base + offsetof(ConSanMoiReportHeader, atomic_record_count),
                                       slot_vgpr, *options.scratch_vgpr, arch)) {
    errors.emplace_back("ConSan MOI atomic record patch could not reserve a dynamic record slot");
    return std::nullopt;
  }
  const auto mov_capacity =
      instrumentation::build_v_mov_b32_literal(lane_rank_vgpr, atomic_record_capacity, arch);
  const auto slot_in_capacity =
      instrumentation::build_v_cmp_gt_u32_vcc(vector_source_vgpr(lane_rank_vgpr), slot_vgpr, arch);
  if (!mov_capacity || !slot_in_capacity) {
    errors.emplace_back("ConSan MOI atomic record patch could not guard dynamic record capacity");
    return std::nullopt;
  }
  words.insert(words.end(), mov_capacity->begin(), mov_capacity->end());
  words.push_back(*slot_in_capacity);
  const ConSanMoiAtomicOperation atomic_operation = is_compare_exchange
                                                        ? ConSanMoiAtomicOperation::CompareExchange
                                                        : ConSanMoiAtomicOperation::Rmw;
  const ConSanMoiAtomicOutcome atomic_outcome = is_compare_exchange
                                                    ? ConSanMoiAtomicOutcome::Unavailable
                                                    : ConSanMoiAtomicOutcome::NotApplicable;
  std::vector<uint32_t> record_words;
  record_words.reserve(128);
  if (is_compare_exchange) {
    if (!append_dynamic_record_store_u32_vgpr(
            record_words, kAtomicRecordLayout,
            atomic_record_base + offsetof(ConSanMoiAtomicRecord, success_lane_mask),
            static_cast<uint16_t>(*options.scratch_vgpr + 3u), slot_vgpr, *options.scratch_vgpr,
            arch) ||
        !append_dynamic_record_store_u32_vgpr(
            record_words, kAtomicRecordLayout,
            atomic_record_base + offsetof(ConSanMoiAtomicRecord, success_lane_mask) +
                sizeof(uint32_t),
            static_cast<uint16_t>(*options.scratch_vgpr + 4u), slot_vgpr, *options.scratch_vgpr,
            arch)) {
      errors.emplace_back("ConSan MOI atomic record patch could not preserve CAS outcome");
      return std::nullopt;
    }
  }
  record_words.insert(record_words.end(), derived_owner_words.begin(), derived_owner_words.end());
  const auto append_record_event_index = [&]() {
    if (reserve_event_before_guest) {
      return append_dynamic_record_store_u32_vgpr(
          record_words, kAtomicRecordLayout,
          atomic_record_base + offsetof(ConSanMoiAtomicRecord, event_index),
          reserved_event_index_vgpr, slot_vgpr, *options.scratch_vgpr, arch);
    }
    return append_dynamic_record_event_index_store(
        record_words, kAtomicRecordLayout, base + offsetof(ConSanMoiReportHeader, event_counter),
        atomic_record_base + offsetof(ConSanMoiAtomicRecord, event_index), slot_vgpr,
        *options.scratch_vgpr, arch);
  };
  // Probe-local owner derivation uses the record-emitter temporary. Preserve
  // it before reserving the event index, which reuses that VGPR.
  if ((derived_owner_vgpr && !append_dynamic_record_store_u32_vgpr(
                                 record_words, kAtomicRecordLayout,
                                 atomic_record_base + offsetof(ConSanMoiAtomicRecord, owner_id),
                                 *derived_owner_vgpr, slot_vgpr, *options.scratch_vgpr, arch)) ||
      (options.moi_persistent_sgprs.owner &&
       !append_dynamic_record_store_u32_scalar_src(
           record_words, kAtomicRecordLayout,
           atomic_record_base + offsetof(ConSanMoiAtomicRecord, owner_id),
           *options.moi_persistent_sgprs.owner, slot_vgpr, *options.scratch_vgpr, arch)) ||
      !append_dynamic_record_store_moi_report_dispatch_id_pair(
          record_words, kAtomicRecordLayout,
          atomic_record_base + offsetof(ConSanMoiAtomicRecord, generation), options, slot_vgpr,
          *options.scratch_vgpr, arch, ConSanMoiLiteralDispatchIdPolicy::ExternalBindingAllowed) ||
      !append_record_event_index() ||
      !append_dynamic_record_store_workgroup_source(
          record_words, kAtomicRecordLayout,
          atomic_record_base + offsetof(ConSanMoiAtomicRecord, workgroup_x), workgroup_sources.x,
          slot_vgpr, *options.scratch_vgpr, arch) ||
      !append_dynamic_record_store_workgroup_source(
          record_words, kAtomicRecordLayout,
          atomic_record_base + offsetof(ConSanMoiAtomicRecord, workgroup_y), workgroup_sources.y,
          slot_vgpr, *options.scratch_vgpr, arch) ||
      !append_dynamic_record_store_workgroup_source(
          record_words, kAtomicRecordLayout,
          atomic_record_base + offsetof(ConSanMoiAtomicRecord, workgroup_z), workgroup_sources.z,
          slot_vgpr, *options.scratch_vgpr, arch) ||
      (options.moi_owner_vgpr &&
       !append_dynamic_record_store_u32_vgpr(
           record_words, kAtomicRecordLayout,
           atomic_record_base + offsetof(ConSanMoiAtomicRecord, owner_id), *options.moi_owner_vgpr,
           slot_vgpr, *options.scratch_vgpr, arch)) ||
      (options.moi_epoch_vgpr &&
       !append_dynamic_record_store_u32_vgpr(
           record_words, kAtomicRecordLayout,
           atomic_record_base + offsetof(ConSanMoiAtomicRecord, epoch), *options.moi_epoch_vgpr,
           slot_vgpr, *options.scratch_vgpr, arch)) ||
      (options.moi_persistent_sgprs.epoch &&
       !append_dynamic_record_store_u32_scalar_src(
           record_words, kAtomicRecordLayout,
           atomic_record_base + offsetof(ConSanMoiAtomicRecord, epoch),
           *options.moi_persistent_sgprs.epoch, slot_vgpr, *options.scratch_vgpr, arch)) ||
      !append_dynamic_record_store_u32_vgpr(
          record_words, kAtomicRecordLayout,
          atomic_record_base + offsetof(ConSanMoiAtomicRecord, atomic_address),
          recorded_address_vgpr, slot_vgpr, *options.scratch_vgpr, arch) ||
      !append_dynamic_record_store_u32_vgpr(
          record_words, kAtomicRecordLayout,
          atomic_record_base + offsetof(ConSanMoiAtomicRecord, atomic_address) + sizeof(uint32_t),
          static_cast<uint16_t>(recorded_address_vgpr + 1u), slot_vgpr, *options.scratch_vgpr,
          arch) ||
      !append_dynamic_record_store_u32_literal(
          record_words, kAtomicRecordLayout,
          atomic_record_base + offsetof(ConSanMoiAtomicRecord, instruction_offset),
          static_cast<uint32_t>(candidate.site.text_offset), slot_vgpr, *options.scratch_vgpr,
          arch) ||
      !append_dynamic_record_store_u32_literal(
          record_words, kAtomicRecordLayout,
          atomic_record_base + offsetof(ConSanMoiAtomicRecord, kind),
          static_cast<uint32_t>(candidate.event_kind), slot_vgpr, *options.scratch_vgpr, arch) ||
      !append_dynamic_record_store_u32_literal(
          record_words, kAtomicRecordLayout,
          atomic_record_base + offsetof(ConSanMoiAtomicRecord, scope), *candidate.site.raw_scope,
          slot_vgpr, *options.scratch_vgpr, arch) ||
      !append_dynamic_record_store_u32_literal(
          record_words, kAtomicRecordLayout,
          atomic_record_base + offsetof(ConSanMoiAtomicRecord, semantics),
          candidate.site.raw_th.value_or(0u), slot_vgpr, *options.scratch_vgpr, arch) ||
      !append_dynamic_record_store_u32_literal(
          record_words, kAtomicRecordLayout,
          atomic_record_base + offsetof(ConSanMoiAtomicRecord, operation),
          static_cast<uint32_t>(atomic_operation), slot_vgpr, *options.scratch_vgpr, arch) ||
      !append_dynamic_record_store_u32_literal(
          record_words, kAtomicRecordLayout,
          atomic_record_base + offsetof(ConSanMoiAtomicRecord, outcome),
          static_cast<uint32_t>(atomic_outcome), slot_vgpr, *options.scratch_vgpr, arch) ||
      (is_compare_exchange
           ? (!append_dynamic_record_store_u32_scalar_src(
                  record_words, kAtomicRecordLayout,
                  atomic_record_base + offsetof(ConSanMoiAtomicRecord, lane_mask),
                  *options.moi_exec_save_sgpr, slot_vgpr, *options.scratch_vgpr, arch) ||
              !append_dynamic_record_store_u32_scalar_src(
                  record_words, kAtomicRecordLayout,
                  atomic_record_base + offsetof(ConSanMoiAtomicRecord, lane_mask) +
                      sizeof(uint32_t),
                  static_cast<uint16_t>(*options.moi_exec_save_sgpr + 1u), slot_vgpr,
                  *options.scratch_vgpr, arch))
           : false)) {
    errors.emplace_back("ConSan MOI atomic record patch could not encode record stores");
    return std::nullopt;
  }

  const auto wait_store = instrumentation::build_s_wait_global_store0(arch);
  const auto restore_exec =
      instrumentation::build_s_mov_b64(kRdna4ExecLo, *options.moi_exec_save_sgpr, arch);
  if (!wait_store || !restore_exec) {
    errors.emplace_back("ConSan MOI atomic record patch could not complete dynamic publication");
    return std::nullopt;
  }
  record_words.push_back(*wait_store);
  if (record_words.size() > static_cast<size_t>(std::numeric_limits<int16_t>::max())) {
    errors.emplace_back("ConSan MOI atomic record overflow branch is out of range");
    return std::nullopt;
  }
  const auto skip_record =
      instrumentation::build_s_cbranch_vccz(static_cast<int16_t>(record_words.size()), arch);
  if (!skip_record) {
    errors.emplace_back("ConSan MOI atomic record patch could not encode its capacity branch");
    return std::nullopt;
  }
  words.push_back(*skip_record);
  words.insert(words.end(), record_words.begin(), record_words.end());
  words.push_back(*restore_exec);

  if (!append_restore_moi_special_state(words, moi_special_state_sgprs(options), arch)) {
    errors.emplace_back("ConSan MOI atomic record patch could not restore VCC/SCC");
    return std::nullopt;
  }

  if (scalar_spill)
    words.insert(words.end(), scalar_spill->restore_words.begin(),
                 scalar_spill->restore_words.end());
  if (spill)
    words.insert(words.end(), spill->restore_words.begin(), spill->restore_words.end());
  if (record_before_guest_release) {
    // An ordinary release RMW has no result needed by the record.  Keep its
    // linearization point at the end of the trampoline: placing the large
    // reporting sequence after the guest atomic opens an artificial window in
    // which another wave can observe the release while the releasing wave has
    // not yet resumed its guest bookkeeping.  An acquire that observes this
    // release cannot reserve its replay event until after this pre-release
    // record, so the replay order remains causal.  Restore spilled operands and
    // guest VCC/SCC before executing the atomic, then return immediately.
    guest_instruction_offset = static_cast<uint32_t>(words.size() * sizeof(uint32_t));
    words.insert(words.end(), guest_atomic_words.begin(), guest_atomic_words.end());
  }
  if (!append_moi_direct_or_indirect_return(words, cave_text_offset, return_text_offset, options,
                                            arch)) {
    errors.emplace_back("ConSan MOI atomic record could not encode its return");
    return std::nullopt;
  }
  return words;
}
[[nodiscard]] std::optional<std::vector<uint32_t>> build_fence_record_cave_words(
    std::span<const uint8_t> bytes, const MoiFenceEvidenceSitePlan &candidate,
    const ConSanMoiAtomicAddressPlan &address_plan, const MoiRecordEventEmissionPlan &options,
    const VgprSpillSequence *spill, const SgprSpillSequence *scalar_spill, rj_code_arch_t arch,
    uint32_t record_index, uint32_t record_capacity_or_count, size_t fence_records_offset,
    uint64_t cave_text_offset, uint64_t return_text_offset,
    std::optional<uint16_t> call_return_sgpr, std::span<const uint32_t> displaced_tail_words,
    const std::optional<MoiWorkitemOwnerDerivationPlan> &derived_owner,
    std::vector<std::string> &errors) {
  (void)record_index;
  if (!options.scratch_vgpr) {
    errors.emplace_back("ConSan MOI fence record patch requires RJ_CONSAN_TMP_VGPR");
    return std::nullopt;
  }
  const ConSanTargetProfile *target = consan_target_profile(arch);
  if (target == nullptr) {
    errors.emplace_back("ConSan MOI fence record patch has no supported target profile");
    return std::nullopt;
  }
  if (!address_plan.supported() || !candidate.communication_site.raw_scope ||
      (target->requires_raw_memory_order_qualifier && !candidate.communication_site.raw_th) ||
      !candidate.is_well_formed()) {
    errors.emplace_back("ConSan MOI fence record patch requires qualified address semantics");
    return std::nullopt;
  }
  if (static_cast<uint32_t>(*options.scratch_vgpr) + address_plan.scratch_vgpr_count > 256u ||
      reject_atomic_candidate_scratch_overlap(candidate.communication_lowering_form,
                                              *options.scratch_vgpr,
                                              address_plan.scratch_vgpr_count, errors) ||
      reject_optional_scratch_range_overlap(options.moi_owner_vgpr, *options.scratch_vgpr,
                                            address_plan.scratch_vgpr_count, "MOI owner", errors) ||
      reject_optional_scratch_range_overlap(options.moi_epoch_vgpr, *options.scratch_vgpr,
                                            address_plan.scratch_vgpr_count, "MOI epoch", errors))
    return std::nullopt;
  if (candidate.patch_file_offset > bytes.size() ||
      candidate.patch_size > bytes.size() - candidate.patch_file_offset) {
    errors.emplace_back("ConSan MOI fence record patch site exceeds ELF bytes");
    return std::nullopt;
  }

  std::optional<uint16_t> derived_owner_vgpr;
  std::vector<uint32_t> derived_owner_words;
  if (!options.moi_owner_vgpr && !options.moi_persistent_sgprs.owner) {
    if (!derived_owner) {
      errors.emplace_back("ConSan MOI fence record patch requires a planned owner derivation");
      return std::nullopt;
    }
    const uint16_t value_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 5u);
    const auto owner = build_moi_workitem_owner_derivation(*derived_owner, value_vgpr, arch,
                                                           "fence record patch", errors);
    if (!owner)
      return std::nullopt;
    derived_owner_vgpr = owner->vgpr;
    derived_owner_words = owner->words;
  }

  const ConSanMoiWorkgroupSources &workgroup_sources = options.workgroup_sources;

  std::vector<uint32_t> words;
  const uint64_t displaced_tail_bytes =
      static_cast<uint64_t>(displaced_tail_words.size()) * sizeof(uint32_t);
  if (displaced_tail_bytes > std::numeric_limits<uint32_t>::max() - candidate.patch_size ||
      candidate.patch_file_offset > bytes.size() ||
      candidate.patch_size + displaced_tail_bytes > bytes.size() - candidate.patch_file_offset) {
    errors.emplace_back("ConSan MOI fence record runtime guest exceeds ELF bytes");
    return std::nullopt;
  }
  const bool runtime_workgroup_gate =
      !candidate.capture_address_before_guest && options.runtime_workgroup_gate.has_value();
  if (runtime_workgroup_gate) {
    if (candidate.capture_address_before_guest) {
      errors.emplace_back(
          "ConSan MOI pre-acquire fence capture does not support a runtime workgroup gate");
      return std::nullopt;
    }
    const uint32_t guest_byte_count =
        static_cast<uint32_t>(candidate.patch_size + displaced_tail_bytes);
    auto gate = build_moi_runtime_workgroup_gate_call_words(
        bytes.subspan(candidate.patch_file_offset, guest_byte_count),
        *options.runtime_workgroup_gate, workgroup_sources, cave_text_offset, return_text_offset,
        moi_runtime_workgroup_gate_reserved_words(
            guest_byte_count, workgroup_sources.cluster_workgroup_id.has_value()),
        arch);
    if (!gate) {
      errors.emplace_back(
          "ConSan MOI fence record patch could not encode its runtime workgroup gate");
      return std::nullopt;
    }
    words = std::move(*gate);
  }
  const std::optional<uint16_t> vgpr_msb_mode =
      target->has_selectable_vgpr_bank
          ? consan_selectable_vgpr_bank_mode_at(arch, bytes, candidate.text_file_offset,
                                                candidate.container_entry_text_offset,
                                                candidate.patch_file_offset)
          : std::nullopt;
  const bool select_low_vgpr_bank = vgpr_msb_mode.value_or(0u) != 0u;
  words.reserve(words.size() + candidate.patch_size / sizeof(uint32_t) + 80u +
                (spill ? spill->save_words.size() + spill->restore_words.size() : 0u) +
                (select_low_vgpr_bank ? 2u : 0u));
  if (select_low_vgpr_bank)
    words.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        static_cast<uint8_t>(*vgpr_msb_mode), 0u, arch));
  if (spill)
    words.insert(words.end(), spill->save_words.begin(), spill->save_words.end());
  if (scalar_spill)
    words.insert(words.end(), scalar_spill->save_words.begin(), scalar_spill->save_words.end());
  const uint16_t recorded_address_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 6u);
  const auto append_address_capture = [&]() {
    if (address_plan.requires_materialization()) {
      const auto special_state = moi_special_state_sgprs(options);
      if (!special_state) {
        errors.emplace_back(
            "ConSan MOI fence record address materialization requires special-state SGPRs");
        return false;
      }
      const auto address_words = build_consan_moi_atomic_address_materialization(
          address_plan, special_state->vcc_save_sgpr, special_state->scc_save_sgpr, arch);
      if (!address_words) {
        errors.emplace_back("ConSan MOI fence record patch could not materialize its address");
        return false;
      }
      words.insert(words.end(), address_words->begin(), address_words->end());
    }
    const uint16_t address_vgpr = address_plan.result_address_vgpr;
    words.push_back(
        build_v_mov_b32_e32(recorded_address_vgpr, vector_source_vgpr(address_vgpr), arch));
    words.push_back(
        build_v_mov_b32_e32(static_cast<uint16_t>(recorded_address_vgpr + 1u),
                            vector_source_vgpr(static_cast<uint16_t>(address_vgpr + 1u)), arch));
    return true;
  };
  if (candidate.capture_address_before_guest && !append_address_capture())
    return std::nullopt;
  for (uint64_t offset = 0; offset < candidate.patch_size; offset += sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, bytes.data() + candidate.patch_file_offset + offset, sizeof(word));
    words.push_back(word);
  }
  if (!candidate.capture_address_before_guest && !append_address_capture())
    return std::nullopt;
  if (!append_save_moi_special_state(words, moi_special_state_sgprs(options), arch)) {
    errors.emplace_back("ConSan MOI fence record patch could not save VCC/SCC");
    return std::nullopt;
  }

  const uint64_t base = *options.moi_report_buffer_address;
  const uint64_t record_base = base + fence_records_offset;
  const ConSanMoiFenceEventKind kind = candidate.memory_role == ConSanSyncMemoryRole::Release
                                           ? ConSanMoiFenceEventKind::Release
                                           : ConSanMoiFenceEventKind::Acquire;
  if (!options.moi_exec_save_sgpr) {
    errors.emplace_back("ConSan MOI dynamic fence record requires scalar EXEC state");
    return std::nullopt;
  }
  {
    const uint16_t lane_rank_vgpr = *options.scratch_vgpr;
    const uint16_t slot_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 2u);
    const uint16_t value_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 5u);
    const auto save_active_exec =
        instrumentation::build_s_mov_b64(*options.moi_exec_save_sgpr, kRdna4ExecLo, arch);
    const auto mbcnt_lo = instrumentation::build_v_mbcnt_lo_u32_b32(
        lane_rank_vgpr, *options.moi_exec_save_sgpr, scalar_positive_inline_u32(0), arch);
    const auto mbcnt_hi = instrumentation::build_v_mbcnt_hi_u32_b32(
        lane_rank_vgpr, static_cast<uint16_t>(*options.moi_exec_save_sgpr + 1u),
        vector_source_vgpr(lane_rank_vgpr), arch);
    const auto first_active_lane = instrumentation::build_v_cmp_eq_u32_vcc(
        scalar_positive_inline_u32(0), lane_rank_vgpr, arch);
    const auto narrow_exec =
        instrumentation::build_s_and_saveexec_b64(*options.moi_exec_save_sgpr, kRdna4VccLo, arch);
    const auto saved_exec_wait = instrumentation::build_salu_to_valu_dependency_wait(arch);
    if (!save_active_exec || !saved_exec_wait || !mbcnt_lo || !mbcnt_hi || !first_active_lane ||
        !narrow_exec) {
      errors.emplace_back("ConSan MOI dynamic fence record could not select a wave publisher");
      return std::nullopt;
    }
    words.push_back(*save_active_exec);
    words.push_back(*saved_exec_wait);
    words.insert(words.end(), mbcnt_lo->begin(), mbcnt_lo->end());
    words.insert(words.end(), mbcnt_hi->begin(), mbcnt_hi->end());
    words.push_back(*first_active_lane);
    words.push_back(*narrow_exec);
    if (!append_atomic_fetch_add_one_u32(words,
                                         base + offsetof(ConSanMoiReportHeader, fence_record_count),
                                         slot_vgpr, *options.scratch_vgpr, arch)) {
      errors.emplace_back("ConSan MOI dynamic fence record could not reserve a wave slot");
      return std::nullopt;
    }
    const auto mov_capacity =
        instrumentation::build_v_mov_b32_literal(value_vgpr, record_capacity_or_count, arch);
    const auto slot_in_capacity =
        instrumentation::build_v_cmp_gt_u32_vcc(vector_source_vgpr(value_vgpr), slot_vgpr, arch);
    if (!mov_capacity || !slot_in_capacity) {
      errors.emplace_back("ConSan MOI dynamic fence record could not guard capacity");
      return std::nullopt;
    }
    words.insert(words.end(), mov_capacity->begin(), mov_capacity->end());
    words.push_back(*slot_in_capacity);
    std::vector<uint32_t> record_words;
    record_words.insert(record_words.end(), derived_owner_words.begin(), derived_owner_words.end());
    if ((derived_owner_vgpr && !append_dynamic_record_store_u32_vgpr(
                                   record_words, kFenceRecordLayout,
                                   record_base + offsetof(ConSanMoiFenceRecord, owner_id),
                                   *derived_owner_vgpr, slot_vgpr, *options.scratch_vgpr, arch)) ||
        !append_dynamic_record_event_index_store(
            record_words, kFenceRecordLayout, base + offsetof(ConSanMoiReportHeader, event_counter),
            record_base + offsetof(ConSanMoiFenceRecord, event_index), slot_vgpr,
            *options.scratch_vgpr, arch) ||
        !append_dynamic_record_store_moi_report_dispatch_id_pair(
            record_words, kFenceRecordLayout,
            record_base + offsetof(ConSanMoiFenceRecord, generation), options, slot_vgpr,
            *options.scratch_vgpr, arch,
            ConSanMoiLiteralDispatchIdPolicy::ExternalBindingAllowed) ||
        !append_dynamic_record_store_workgroup_source(
            record_words, kFenceRecordLayout,
            record_base + offsetof(ConSanMoiFenceRecord, workgroup_x), workgroup_sources.x,
            slot_vgpr, *options.scratch_vgpr, arch) ||
        !append_dynamic_record_store_workgroup_source(
            record_words, kFenceRecordLayout,
            record_base + offsetof(ConSanMoiFenceRecord, workgroup_y), workgroup_sources.y,
            slot_vgpr, *options.scratch_vgpr, arch) ||
        !append_dynamic_record_store_workgroup_source(
            record_words, kFenceRecordLayout,
            record_base + offsetof(ConSanMoiFenceRecord, workgroup_z), workgroup_sources.z,
            slot_vgpr, *options.scratch_vgpr, arch) ||
        (options.moi_owner_vgpr &&
         !append_dynamic_record_store_u32_vgpr(
             record_words, kFenceRecordLayout,
             record_base + offsetof(ConSanMoiFenceRecord, owner_id), *options.moi_owner_vgpr,
             slot_vgpr, *options.scratch_vgpr, arch)) ||
        (options.moi_persistent_sgprs.owner &&
         !append_dynamic_record_store_u32_scalar_src(
             record_words, kFenceRecordLayout,
             record_base + offsetof(ConSanMoiFenceRecord, owner_id),
             *options.moi_persistent_sgprs.owner, slot_vgpr, *options.scratch_vgpr, arch)) ||
        (options.moi_epoch_vgpr &&
         !append_dynamic_record_store_u32_vgpr(
             record_words, kFenceRecordLayout, record_base + offsetof(ConSanMoiFenceRecord, epoch),
             *options.moi_epoch_vgpr, slot_vgpr, *options.scratch_vgpr, arch)) ||
        (options.moi_persistent_sgprs.epoch &&
         !append_dynamic_record_store_u32_scalar_src(
             record_words, kFenceRecordLayout, record_base + offsetof(ConSanMoiFenceRecord, epoch),
             *options.moi_persistent_sgprs.epoch, slot_vgpr, *options.scratch_vgpr, arch)) ||
        !append_dynamic_record_store_u32_literal(
            record_words, kFenceRecordLayout,
            record_base + offsetof(ConSanMoiFenceRecord, instruction_offset),
            static_cast<uint32_t>(candidate.semantic_site.physical.original_text_offset), slot_vgpr,
            *options.scratch_vgpr, arch) ||
        !append_dynamic_record_store_u32_literal(
            record_words, kFenceRecordLayout, record_base + offsetof(ConSanMoiFenceRecord, kind),
            static_cast<uint32_t>(kind), slot_vgpr, *options.scratch_vgpr, arch) ||
        !append_dynamic_record_store_u32_literal(
            record_words, kFenceRecordLayout, record_base + offsetof(ConSanMoiFenceRecord, scope),
            *candidate.communication_site.raw_scope, slot_vgpr, *options.scratch_vgpr, arch) ||
        !append_dynamic_record_store_u32_literal(record_words, kFenceRecordLayout,
                                                 record_base +
                                                     offsetof(ConSanMoiFenceRecord, semantics),
                                                 candidate.communication_site.raw_th.value_or(0u),
                                                 slot_vgpr, *options.scratch_vgpr, arch) ||
        !append_dynamic_record_store_u32_vgpr(
            record_words, kFenceRecordLayout,
            record_base + offsetof(ConSanMoiFenceRecord, communication_token),
            recorded_address_vgpr, slot_vgpr, *options.scratch_vgpr, arch) ||
        !append_dynamic_record_store_u32_vgpr(
            record_words, kFenceRecordLayout,
            record_base + offsetof(ConSanMoiFenceRecord, communication_token) + sizeof(uint32_t),
            static_cast<uint16_t>(recorded_address_vgpr + 1u), slot_vgpr, *options.scratch_vgpr,
            arch)) {
      errors.emplace_back("ConSan MOI dynamic fence record could not encode record stores");
      return std::nullopt;
    }
    const auto wait_store = instrumentation::build_s_wait_global_store0(arch);
    const auto restore_exec =
        instrumentation::build_s_mov_b64(kRdna4ExecLo, *options.moi_exec_save_sgpr, arch);
    if (!wait_store || !restore_exec ||
        record_words.size() > static_cast<size_t>(std::numeric_limits<int16_t>::max()))
      return std::nullopt;
    record_words.push_back(*wait_store);
    const auto skip_record =
        instrumentation::build_s_cbranch_vccz(static_cast<int16_t>(record_words.size()), arch);
    if (!skip_record)
      return std::nullopt;
    words.push_back(*skip_record);
    words.insert(words.end(), record_words.begin(), record_words.end());
    words.push_back(*restore_exec);
  }
  if (!append_restore_moi_special_state(words, moi_special_state_sgprs(options), arch)) {
    errors.emplace_back("ConSan MOI fence record patch could not restore VCC/SCC");
    return std::nullopt;
  }
  if (scalar_spill)
    words.insert(words.end(), scalar_spill->restore_words.begin(),
                 scalar_spill->restore_words.end());
  if (spill)
    words.insert(words.end(), spill->restore_words.begin(), spill->restore_words.end());
  if (select_low_vgpr_bank)
    words.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        0u, static_cast<uint8_t>(*vgpr_msb_mode), arch));
  words.insert(words.end(), displaced_tail_words.begin(), displaced_tail_words.end());
  if (call_return_sgpr) {
    words.push_back(build_s_setpc_b64(*call_return_sgpr, arch));
  } else if (!append_moi_direct_or_indirect_return(words, cave_text_offset, return_text_offset,
                                                   options, arch)) {
    errors.emplace_back("ConSan MOI fence record could not encode its return");
    return std::nullopt;
  }
  return words;
}

} // namespace rocjitsu::consan_moi_impl

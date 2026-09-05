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
#include "rocjitsu/code/patch/consan/targets/consan_vgpr_bank_state.h"
#include "rocjitsu/code/patch/instruction_sequence.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <ranges>

namespace rocjitsu::consan_moi_impl {

using consan_detail::MoiSpecialStateSgprs;
using consan_detail::MoiWorkitemOwnerDerivationPlan;
using consan_detail::reject_atomic_candidate_scratch_overlap;
using consan_detail::reject_optional_scratch_range_overlap;
using consan_moi_detail::append_atomic_fetch_add_one_u32;
using consan_moi_detail::append_guarded_dynamic_record;
using consan_moi_detail::append_reserve_bounded_dynamic_record_slot;
using consan_moi_detail::append_select_first_active_lane;
using consan_moi_detail::DynamicRecordEmitter;
using consan_moi_detail::kAtomicRecordLayout;
using consan_moi_detail::kBarrierRecordLayout;
using consan_moi_detail::kFenceRecordLayout;

[[nodiscard]] std::optional<std::vector<uint32_t>> build_barrier_record_cave_words(
    std::span<const uint8_t> bytes, const ConSanBarrierSite &site,
    const ConSanProgramContainer &container, const MoiRecordEventEmissionPlan &options,
    const VgprSpillSequence *spill, const SgprSpillSequence *scalar_spill, rj_code_arch_t arch,
    uint32_t barrier_record_capacity, size_t barrier_records_offset, uint32_t original_barrier_word,
    std::vector<std::string> &errors, uint32_t *guest_instruction_offset) {
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
  if (reject_optional_scratch_range_overlap(options.moi_owner_epoch_vgprs.owner(),
                                            *options.scratch_vgpr, 6, "MOI owner", errors))
    return std::nullopt;

  std::optional<uint16_t> derived_owner_vgpr;
  std::vector<uint32_t> derived_owner_words;
  if (!options.moi_owner_epoch_vgprs.owner() && !options.moi_persistent_sgprs.owner()) {
    if (!options.derived_owner) {
      errors.emplace_back("ConSan MOI barrier record patch requires a planned owner derivation");
      return std::nullopt;
    }
    const uint16_t value_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 5u);
    const auto owner = build_moi_workitem_owner_derivation(*options.derived_owner, value_vgpr, arch,
                                                           "barrier record patch", errors);
    if (!owner)
      return std::nullopt;
    derived_owner_vgpr = owner->vgpr;
    derived_owner_words = owner->words;
  }
  const ConSanMoiWorkgroupSources &workgroup_sources = options.workgroup_sources;

  std::vector<uint32_t> words;
  InstructionSequence sequence(words);
  MoiEmissionRequirement require_emission(sequence, errors);
  const uint64_t base = *options.moi_report_buffer_address;
  const uint64_t barrier_record_base = base + barrier_records_offset;
  const uint16_t slot_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 2u);
  const uint16_t lane_rank_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 5u);
  constexpr uint16_t kScalarInlineMinusOne = 0xC1;

  const ConSanTargetProfile *target = consan_target_profile(arch);
  const std::optional<uint16_t> vgpr_msb_mode =
      target != nullptr && target->has_selectable_vgpr_bank
          ? consan_selectable_vgpr_bank_mode_at(arch, bytes, container.text_file_offset,
                                                container.entry_text_offset, site.file_offset)
          : std::nullopt;
  const bool select_low_vgpr_bank = vgpr_msb_mode.value_or(0u) != 0u;
  words.reserve(
      186 + (spill ? spill->save_words.size() + spill->restore_words.size() : 0u) +
      (scalar_spill ? scalar_spill->save_words.size() + scalar_spill->restore_words.size() : 0u));
  if (select_low_vgpr_bank)
    words.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        static_cast<uint8_t>(*vgpr_msb_mode), 0u, arch));
  if (spill)
    words.insert(words.end(), spill->save_words.begin(), spill->save_words.end());
  if (scalar_spill)
    words.insert(words.end(), scalar_spill->save_words.begin(), scalar_spill->save_words.end());
  require_emission(append_save_moi_special_state(words, options.special_state, arch),
                   "ConSan MOI barrier record patch could not save VCC/SCC");
  require_emission(
      sequence.emit_all(
          instrumentation::build_v_mbcnt_lo_u32_b32(lane_rank_vgpr, kScalarInlineMinusOne,
                                                    scalar_positive_inline_u32(0), arch),
          instrumentation::build_v_mbcnt_hi_u32_b32(lane_rank_vgpr, kScalarInlineMinusOne,
                                                    vector_source_vgpr(lane_rank_vgpr), arch),
          instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), lane_rank_vgpr,
                                                  arch),
          instrumentation::build_s_and_saveexec_b64(*options.moi_exec_save_sgpr, kAmdGpuVccLo,
                                                    arch)),
      "ConSan MOI barrier record patch could not encode EXEC narrowing");

  const uint16_t value_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 5u);
  require_emission(
      append_reserve_bounded_dynamic_record_slot(
          words, base + offsetof(ConSanMoiReportHeader, barrier_record_count),
          barrier_record_capacity, slot_vgpr, value_vgpr, *options.scratch_vgpr, arch),
      "ConSan MOI barrier record patch could not reserve a bounded record slot");

  std::vector<uint32_t> record_words;
  record_words.reserve(128);
  DynamicRecordEmitter record(record_words, kBarrierRecordLayout, barrier_record_base, slot_vgpr,
                              *options.scratch_vgpr, arch);
  record_words.insert(record_words.end(), derived_owner_words.begin(), derived_owner_words.end());
  if (derived_owner_vgpr)
    record.vgpr(offsetof(ConSanMoiBarrierRecord, wave_id), *derived_owner_vgpr);
  record.dispatch_id(offsetof(ConSanMoiBarrierRecord, generation), options.dispatch_id_sources)
      .event_index(offsetof(ConSanMoiBarrierRecord, event_index),
                   base + offsetof(ConSanMoiReportHeader, event_counter))
      .workgroup(offsetof(ConSanMoiBarrierRecord, workgroup_x), workgroup_sources.x)
      .workgroup(offsetof(ConSanMoiBarrierRecord, workgroup_y), workgroup_sources.y)
      .workgroup(offsetof(ConSanMoiBarrierRecord, workgroup_z), workgroup_sources.z)
      .scalar(offsetof(ConSanMoiBarrierRecord, lane_mask), *options.moi_exec_save_sgpr)
      .scalar(offsetof(ConSanMoiBarrierRecord, lane_mask) + sizeof(uint32_t),
              static_cast<uint16_t>(*options.moi_exec_save_sgpr + 1u));
  if (options.moi_owner_epoch_vgprs.owner())
    record.vgpr(offsetof(ConSanMoiBarrierRecord, wave_id), options.moi_owner_epoch_vgprs->owner);
  if (options.moi_persistent_sgprs.owner())
    record.scalar(offsetof(ConSanMoiBarrierRecord, wave_id), *options.moi_persistent_sgprs.owner());
  record.literal(offsetof(ConSanMoiBarrierRecord, instruction_offset),
                 static_cast<uint32_t>(site.text_offset));
  require_emission(record.finish(),
                   "ConSan MOI barrier record patch could not encode record stores");
  require_emission(append_guarded_dynamic_record(words, record_words,
                                                 *options.moi_exec_save_sgpr,
                                                 /*wait_for_global_stores=*/false, arch),
                   "ConSan MOI barrier record patch could not append its guarded record");
  require_emission(append_restore_moi_special_state(words, options.special_state, arch),
                   "ConSan MOI barrier record patch could not restore VCC/SCC");

  if (scalar_spill) {
    words.insert(words.end(), scalar_spill->restore_words.begin(),
                 scalar_spill->restore_words.end());
  }
  if (spill)
    words.insert(words.end(), spill->restore_words.begin(), spill->restore_words.end());
  if (select_low_vgpr_bank)
    words.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        0u, static_cast<uint8_t>(*vgpr_msb_mode), arch));
  const uint32_t staged_guest_instruction_offset =
      static_cast<uint32_t>(words.size() * sizeof(uint32_t));
  words.push_back(original_barrier_word);

  if (!sequence.finish())
    return std::nullopt;
  if (guest_instruction_offset)
    *guest_instruction_offset = staged_guest_instruction_offset;
  return words;
}
[[nodiscard]] std::optional<std::vector<uint32_t>> build_atomic_record_cave_words(
    std::span<const uint8_t> bytes, const MoiAtomicEvidenceSourceView &source,
    const ConSanAtomicLoweringForm &lowering_form,
    const ConSanMoiAtomicAddressPlan &address_plan,
    const MoiRecordEventEmissionPlan &options,
    const VgprSpillSequence *spill, const SgprSpillSequence *scalar_spill, rj_code_arch_t arch,
    uint32_t record_index, uint32_t atomic_record_capacity, size_t atomic_records_offset,
    uint32_t &guest_instruction_offset, std::vector<std::string> &errors) {
  const ConSanAtomicSite &site = source.site;
  const std::optional<ConSanMoiAtomicEventKind> event_kind =
      moi_atomic_event_kind(source.sequence->memory_role);
  if (!event_kind) {
    errors.emplace_back("ConSan MOI atomic record patch lost its ordering role");
    return std::nullopt;
  }
  if (!options.scratch_vgpr) {
    errors.emplace_back("ConSan MOI atomic record patch requires RJ_CONSAN_TMP_VGPR");
    return std::nullopt;
  }
  const auto report_scope =
      site.scope ? consan_moi_record_replay_scope(*site.scope) : std::nullopt;
  if (!report_scope) {
    errors.emplace_back("ConSan MOI atomic record patch requires a recordable causal scope");
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
  if (reject_atomic_candidate_scratch_overlap(lowering_form, *options.scratch_vgpr,
                                              address_plan.scratch_vgpr_count, errors) ||
      reject_optional_scratch_range_overlap(options.moi_owner_epoch_vgprs.owner(),
                                            *options.scratch_vgpr, address_plan.scratch_vgpr_count,
                                            "MOI owner", errors) ||
      reject_optional_scratch_range_overlap(options.moi_owner_epoch_vgprs.epoch(),
                                            *options.scratch_vgpr, address_plan.scratch_vgpr_count,
                                            "MOI epoch", errors))
    return std::nullopt;

  if (!address_plan.supported()) {
    errors.emplace_back("ConSan MOI atomic record patch requires a supported address plan");
    return std::nullopt;
  }

  (void)record_index;
  std::optional<uint16_t> derived_owner_vgpr;
  std::vector<uint32_t> derived_owner_words;
  if (!options.moi_owner_epoch_vgprs.owner() && !options.moi_persistent_sgprs.owner()) {
    if (!options.derived_owner) {
      errors.emplace_back("ConSan MOI atomic record patch requires a planned owner derivation");
      return std::nullopt;
    }
    const uint16_t value_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 4u);
    const auto owner = build_moi_workitem_owner_derivation(*options.derived_owner, value_vgpr, arch,
                                                           "atomic record patch", errors);
    if (!owner)
      return std::nullopt;
    derived_owner_vgpr = owner->vgpr;
    derived_owner_words = owner->words;
  }

  const ConSanMoiWorkgroupSources &workgroup_sources = options.workgroup_sources;

  if (site.file_offset > bytes.size() || site.size > bytes.size() - site.file_offset) {
    errors.emplace_back("ConSan MOI atomic record patch site exceeds ELF bytes");
    return std::nullopt;
  }

  std::vector<uint32_t> words;
  InstructionSequence sequence(words);
  MoiEmissionRequirement require_emission(sequence, errors);
  words.reserve(words.size() + site.size / sizeof(uint32_t) + 96u +
                (spill ? spill->save_words.size() + spill->restore_words.size() : 0u));
  if (spill)
    words.insert(words.end(), spill->save_words.begin(), spill->save_words.end());
  if (scalar_spill)
    words.insert(words.end(), scalar_spill->save_words.begin(), scalar_spill->save_words.end());
  if (address_plan.requires_materialization()) {
    const auto address_words = build_consan_moi_atomic_address_materialization(
        address_plan, options.special_state.vcc_save_sgpr, options.special_state.scc_save_sgpr,
        arch);
    if (!address_words) {
      errors.emplace_back("ConSan MOI atomic record patch could not materialize its address");
      return std::nullopt;
    }
    words.insert(words.end(), address_words->begin(), address_words->end());
  }
  const bool is_compare_exchange = consan_atomic_is_compare_exchange(site);
  std::vector<uint32_t> guest_atomic_words;
  guest_atomic_words.reserve(site.size / sizeof(uint32_t));
  for (uint64_t offset = 0; offset < site.size; offset += sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, bytes.data() + site.file_offset + offset, sizeof(word));
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
      is_compare_exchange && (*event_kind == ConSanMoiAtomicEventKind::Release ||
                              *event_kind == ConSanMoiAtomicEventKind::AcquireRelease);
  const uint16_t reserved_event_index_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 7u);
  if (reserve_event_before_guest) {
    require_emission(
        append_save_moi_special_state(words, options.special_state, arch),
        "ConSan MOI release CAS could not preserve VCC/SCC before event reservation");
    const uint16_t lane_rank_vgpr = *options.scratch_vgpr;
    require_emission(append_select_first_active_lane(words, lane_rank_vgpr,
                                                     *options.moi_exec_save_sgpr,
                                                     *options.moi_exec_save_sgpr, arch),
                     "ConSan MOI release CAS could not select its event publisher");
    require_emission(
        append_atomic_fetch_add_one_u32(words,
                                        base + offsetof(ConSanMoiReportHeader, event_counter),
                                        reserved_event_index_vgpr, *options.scratch_vgpr, arch),
        "ConSan MOI release CAS could not reserve its causal event index");
    require_emission(
        sequence.emit(
            instrumentation::build_s_mov_b64(kAmdGpuExecLo, *options.moi_exec_save_sgpr, arch)),
        "ConSan MOI release CAS could not restore EXEC after event reservation");
    require_emission(
        append_restore_moi_special_state(words, options.special_state, arch),
        "ConSan MOI release CAS could not restore VCC/SCC after event reservation");
  }

  const bool record_before_guest_release =
      *event_kind == ConSanMoiAtomicEventKind::Release && !is_compare_exchange;
  if (!record_before_guest_release) {
    guest_instruction_offset = static_cast<uint32_t>(words.size() * sizeof(uint32_t));
    words.insert(words.end(), guest_atomic_words.begin(), guest_atomic_words.end());
    require_emission(append_moi_flat_load_wait(words, arch));
  }

  require_emission(append_save_moi_special_state(words, options.special_state, arch),
                   "ConSan MOI atomic record patch could not save VCC/SCC");

  const uint64_t atomic_record_base = base + atomic_records_offset;
  const uint16_t slot_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 2u);
  const uint16_t lane_rank_vgpr = *options.scratch_vgpr;
  if (is_compare_exchange) {
    if (!site.data_vgpr || !site.destination_vgpr) {
      errors.emplace_back("ConSan MOI atomic record CAS lacks compare/result operands");
      return std::nullopt;
    }
    const uint16_t value_word_count = static_cast<uint16_t>(site.width_bits / 32u);
    const uint16_t compare_vgpr =
        static_cast<uint16_t>(*site.data_vgpr + value_word_count);
    require_emission(sequence.emit(instrumentation::build_v_cmp_eq_u32_vcc(
                         vector_source_vgpr(compare_vgpr), *site.destination_vgpr, arch)),
                     "ConSan MOI atomic record CAS could not capture its outcome mask");
    const uint16_t success_lo = static_cast<uint16_t>(*options.scratch_vgpr + 3u);
    const uint16_t success_hi = static_cast<uint16_t>(*options.scratch_vgpr + 4u);
    words.push_back(build_v_mov_b32_e32(success_lo, kAmdGpuVccLo, arch));
    words.push_back(build_v_mov_b32_e32(success_hi, kAmdGpuVccHi, arch));
    if (site.width_bits == 64u) {
      require_emission(
          sequence.emit_all(
              instrumentation::build_v_cmp_eq_u32_vcc(
                  vector_source_vgpr(static_cast<uint16_t>(compare_vgpr + 1u)),
                  static_cast<uint16_t>(*site.destination_vgpr + 1u), arch),
              instrumentation::build_v_and_b32(success_lo, kAmdGpuVccLo, success_lo, arch),
              instrumentation::build_v_and_b32(success_hi, kAmdGpuVccHi, success_hi, arch)),
          "ConSan MOI atomic record 64-bit CAS could not combine its outcome masks");
    }
    // VCC bits outside EXEC are architecturally unspecified after a vector
    // compare. Simulators commonly leave them clear, but physical gfx1201 can
    // retain arbitrary values there. Only lanes that executed the guest CAS
    // have an outcome, so canonicalize both halves before publishing them.
    require_emission(
        sequence.emit_all(
            instrumentation::build_v_and_b32(success_lo, kAmdGpuExecLo, success_lo, arch),
            instrumentation::build_v_and_b32(success_hi, kAmdGpuExecHi, success_hi, arch)),
        "ConSan MOI atomic record CAS could not mask inactive outcome lanes");
  }
  require_emission(append_select_first_active_lane(words, lane_rank_vgpr,
                                                   *options.moi_exec_save_sgpr,
                                                   *options.moi_exec_save_sgpr, arch),
                   "ConSan MOI atomic record patch could not select a wave publisher");
  require_emission(
      append_reserve_bounded_dynamic_record_slot(
          words, base + offsetof(ConSanMoiReportHeader, atomic_record_count),
          atomic_record_capacity, slot_vgpr, lane_rank_vgpr, *options.scratch_vgpr, arch),
      "ConSan MOI atomic record patch could not reserve a bounded record slot");
  const ConSanMoiAtomicOperation atomic_operation = is_compare_exchange
                                                        ? ConSanMoiAtomicOperation::CompareExchange
                                                        : ConSanMoiAtomicOperation::Rmw;
  const ConSanMoiAtomicOutcome atomic_outcome = is_compare_exchange
                                                    ? ConSanMoiAtomicOutcome::Unavailable
                                                    : ConSanMoiAtomicOutcome::NotApplicable;
  std::vector<uint32_t> record_words;
  record_words.reserve(128);
  DynamicRecordEmitter record(record_words, kAtomicRecordLayout, atomic_record_base, slot_vgpr,
                              *options.scratch_vgpr, arch);
  if (is_compare_exchange) {
    record
        .vgpr(offsetof(ConSanMoiAtomicRecord, success_lane_mask),
              static_cast<uint16_t>(*options.scratch_vgpr + 3u))
        .vgpr(offsetof(ConSanMoiAtomicRecord, success_lane_mask) + sizeof(uint32_t),
              static_cast<uint16_t>(*options.scratch_vgpr + 4u));
    require_emission(static_cast<bool>(record),
                     "ConSan MOI atomic record patch could not preserve CAS outcome");
  }
  record_words.insert(record_words.end(), derived_owner_words.begin(), derived_owner_words.end());
  const auto emit_record_event_index = [&]() {
    if (reserve_event_before_guest) {
      record.vgpr(offsetof(ConSanMoiAtomicRecord, event_index), reserved_event_index_vgpr);
    } else {
      record.event_index(offsetof(ConSanMoiAtomicRecord, event_index),
                         base + offsetof(ConSanMoiReportHeader, event_counter));
    }
  };
  // Probe-local owner derivation uses the record-emitter temporary. Preserve
  // it before reserving the event index, which reuses that VGPR.
  if (derived_owner_vgpr)
    record.vgpr(offsetof(ConSanMoiAtomicRecord, owner_id), *derived_owner_vgpr);
  if (options.moi_persistent_sgprs.owner())
    record.scalar(offsetof(ConSanMoiAtomicRecord, owner_id), *options.moi_persistent_sgprs.owner());
  record.dispatch_id(offsetof(ConSanMoiAtomicRecord, generation), options.dispatch_id_sources);
  emit_record_event_index();
  record.workgroup(offsetof(ConSanMoiAtomicRecord, workgroup_x), workgroup_sources.x)
      .workgroup(offsetof(ConSanMoiAtomicRecord, workgroup_y), workgroup_sources.y)
      .workgroup(offsetof(ConSanMoiAtomicRecord, workgroup_z), workgroup_sources.z);
  if (options.moi_owner_epoch_vgprs.owner())
    record.vgpr(offsetof(ConSanMoiAtomicRecord, owner_id), options.moi_owner_epoch_vgprs->owner);
  if (options.moi_owner_epoch_vgprs.epoch())
    record.vgpr(offsetof(ConSanMoiAtomicRecord, epoch), options.moi_owner_epoch_vgprs->epoch);
  if (options.moi_persistent_sgprs.epoch())
    record.scalar(offsetof(ConSanMoiAtomicRecord, epoch), *options.moi_persistent_sgprs.epoch());
  record.vgpr(offsetof(ConSanMoiAtomicRecord, atomic_address), recorded_address_vgpr)
      .vgpr(offsetof(ConSanMoiAtomicRecord, atomic_address) + sizeof(uint32_t),
            static_cast<uint16_t>(recorded_address_vgpr + 1u))
      .literal(offsetof(ConSanMoiAtomicRecord, instruction_offset),
               static_cast<uint32_t>(site.text_offset))
      .literal(offsetof(ConSanMoiAtomicRecord, kind), static_cast<uint32_t>(*event_kind))
      .literal(offsetof(ConSanMoiAtomicRecord, scope), *report_scope)
      .literal(offsetof(ConSanMoiAtomicRecord, semantics), site.raw_th.value_or(0u))
      .literal(offsetof(ConSanMoiAtomicRecord, operation), static_cast<uint32_t>(atomic_operation))
      .literal(offsetof(ConSanMoiAtomicRecord, outcome), static_cast<uint32_t>(atomic_outcome));
  if (is_compare_exchange) {
    record.scalar(offsetof(ConSanMoiAtomicRecord, lane_mask), *options.moi_exec_save_sgpr)
        .scalar(offsetof(ConSanMoiAtomicRecord, lane_mask) + sizeof(uint32_t),
                static_cast<uint16_t>(*options.moi_exec_save_sgpr + 1u));
  }
  require_emission(record.finish(),
                   "ConSan MOI atomic record patch could not encode record stores");
  require_emission(append_guarded_dynamic_record(words, record_words,
                                                 *options.moi_exec_save_sgpr,
                                                 /*wait_for_global_stores=*/true, arch),
                   "ConSan MOI atomic record patch could not append its guarded record");
  require_emission(append_restore_moi_special_state(words, options.special_state, arch),
                   "ConSan MOI atomic record patch could not restore VCC/SCC");

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
  if (!sequence.finish())
    return std::nullopt;
  return words;
}
[[nodiscard]] std::optional<std::vector<uint32_t>> build_fence_record_cave_words(
    std::span<const uint8_t> bytes, const MoiFenceEvidenceSourceView &source,
    const ConSanAtomicLoweringForm &communication_lowering_form,
    const ConSanProgramContainer &container,
    const ConSanMoiAtomicAddressPlan &address_plan, const MoiRecordEventEmissionPlan &options,
    const VgprSpillSequence *spill, const SgprSpillSequence *scalar_spill, rj_code_arch_t arch,
    uint32_t record_index, uint32_t record_capacity_or_count, size_t fence_records_offset,
    std::span<const uint32_t> displaced_tail_words, std::vector<std::string> &errors,
    uint32_t *guest_instruction_offset) {
  (void)record_index;
  if (!source.is_resolved() || !options.scratch_vgpr) {
    errors.emplace_back("ConSan MOI fence record patch requires RJ_CONSAN_TMP_VGPR");
    return std::nullopt;
  }
  const ConSanAtomicSite &communication_site = source.communication_site;
  const uint64_t fence_text_offset = source.fence_event->text_offset();
  const uint64_t patch_file_offset = source.patch_file_offset();
  const uint32_t patch_size = source.patch_size();
  const ConSanTargetProfile *target = consan_target_profile(arch);
  if (target == nullptr) {
    errors.emplace_back("ConSan MOI fence record patch has no supported target profile");
    return std::nullopt;
  }
  if (!address_plan.supported() || !communication_site.scope ||
      (target->requires_raw_memory_order_qualifier && !communication_site.raw_th)) {
    errors.emplace_back("ConSan MOI fence record patch requires qualified address semantics");
    return std::nullopt;
  }
  const auto report_scope = consan_moi_record_replay_scope(*communication_site.scope);
  if (!report_scope) {
    errors.emplace_back("ConSan MOI fence record patch requires a recordable causal scope");
    return std::nullopt;
  }
  if (static_cast<uint32_t>(*options.scratch_vgpr) + address_plan.scratch_vgpr_count > 256u ||
      reject_atomic_candidate_scratch_overlap(communication_lowering_form,
                                              *options.scratch_vgpr,
                                              address_plan.scratch_vgpr_count, errors) ||
      reject_optional_scratch_range_overlap(options.moi_owner_epoch_vgprs.owner(),
                                            *options.scratch_vgpr, address_plan.scratch_vgpr_count,
                                            "MOI owner", errors) ||
      reject_optional_scratch_range_overlap(options.moi_owner_epoch_vgprs.epoch(),
                                            *options.scratch_vgpr, address_plan.scratch_vgpr_count,
                                            "MOI epoch", errors))
    return std::nullopt;
  if (patch_file_offset > bytes.size() || patch_size > bytes.size() - patch_file_offset) {
    errors.emplace_back("ConSan MOI fence record patch site exceeds ELF bytes");
    return std::nullopt;
  }

  std::optional<uint16_t> derived_owner_vgpr;
  std::vector<uint32_t> derived_owner_words;
  if (!options.moi_owner_epoch_vgprs.owner() && !options.moi_persistent_sgprs.owner()) {
    if (!options.derived_owner) {
      errors.emplace_back("ConSan MOI fence record patch requires a planned owner derivation");
      return std::nullopt;
    }
    const uint16_t value_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 5u);
    const auto owner = build_moi_workitem_owner_derivation(*options.derived_owner, value_vgpr, arch,
                                                           "fence record patch", errors);
    if (!owner)
      return std::nullopt;
    derived_owner_vgpr = owner->vgpr;
    derived_owner_words = owner->words;
  }

  const ConSanMoiWorkgroupSources &workgroup_sources = options.workgroup_sources;

  std::vector<uint32_t> words;
  InstructionSequence sequence(words);
  MoiEmissionRequirement require_emission(sequence, errors);
  const uint64_t displaced_tail_bytes =
      static_cast<uint64_t>(displaced_tail_words.size()) * sizeof(uint32_t);
  if (displaced_tail_bytes > std::numeric_limits<uint32_t>::max() - patch_size ||
      patch_file_offset > bytes.size() ||
      patch_size + displaced_tail_bytes > bytes.size() - patch_file_offset) {
    errors.emplace_back("ConSan MOI fence record runtime guest exceeds ELF bytes");
    return std::nullopt;
  }
  const std::optional<uint16_t> vgpr_msb_mode =
      target->has_selectable_vgpr_bank
          ? consan_selectable_vgpr_bank_mode_at(arch, bytes, container.text_file_offset,
                                                container.entry_text_offset,
                                                patch_file_offset)
          : std::nullopt;
  const bool select_low_vgpr_bank = vgpr_msb_mode.value_or(0u) != 0u;
  words.reserve(words.size() + patch_size / sizeof(uint32_t) + 80u +
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
      const auto address_words = build_consan_moi_atomic_address_materialization(
          address_plan, options.special_state.vcc_save_sgpr, options.special_state.scc_save_sgpr,
          arch);
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
  if (source.captures_address_before_guest() && !append_address_capture())
    return std::nullopt;
  if (guest_instruction_offset)
    *guest_instruction_offset = static_cast<uint32_t>(words.size() * sizeof(uint32_t));
  for (uint64_t offset = 0; offset < patch_size; offset += sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, bytes.data() + patch_file_offset + offset, sizeof(word));
    words.push_back(word);
  }
  if (!source.captures_address_before_guest() && !append_address_capture())
    return std::nullopt;
  require_emission(append_save_moi_special_state(words, options.special_state, arch),
                   "ConSan MOI fence record patch could not save VCC/SCC");

  const uint64_t base = *options.moi_report_buffer_address;
  const uint64_t record_base = base + fence_records_offset;
  const ConSanMoiFenceEventKind kind =
      source.association->memory_role == ConSanSyncMemoryRole::Release
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
    require_emission(append_select_first_active_lane(words, lane_rank_vgpr,
                                                     *options.moi_exec_save_sgpr,
                                                     *options.moi_exec_save_sgpr, arch),
                     "ConSan MOI dynamic fence record could not select a wave publisher");
    require_emission(
        append_reserve_bounded_dynamic_record_slot(
            words, base + offsetof(ConSanMoiReportHeader, fence_record_count),
            record_capacity_or_count, slot_vgpr, value_vgpr, *options.scratch_vgpr, arch),
        "ConSan MOI dynamic fence record could not reserve a bounded record slot");
    std::vector<uint32_t> record_words;
    DynamicRecordEmitter record(record_words, kFenceRecordLayout, record_base, slot_vgpr,
                                *options.scratch_vgpr, arch);
    record_words.insert(record_words.end(), derived_owner_words.begin(), derived_owner_words.end());
    if (derived_owner_vgpr)
      record.vgpr(offsetof(ConSanMoiFenceRecord, owner_id), *derived_owner_vgpr);
    record
        .event_index(offsetof(ConSanMoiFenceRecord, event_index),
                     base + offsetof(ConSanMoiReportHeader, event_counter))
        .dispatch_id(offsetof(ConSanMoiFenceRecord, generation), options.dispatch_id_sources)
        .workgroup(offsetof(ConSanMoiFenceRecord, workgroup_x), workgroup_sources.x)
        .workgroup(offsetof(ConSanMoiFenceRecord, workgroup_y), workgroup_sources.y)
        .workgroup(offsetof(ConSanMoiFenceRecord, workgroup_z), workgroup_sources.z);
    if (options.moi_owner_epoch_vgprs.owner())
      record.vgpr(offsetof(ConSanMoiFenceRecord, owner_id), options.moi_owner_epoch_vgprs->owner);
    if (options.moi_persistent_sgprs.owner())
      record.scalar(offsetof(ConSanMoiFenceRecord, owner_id),
                    *options.moi_persistent_sgprs.owner());
    if (options.moi_owner_epoch_vgprs.epoch())
      record.vgpr(offsetof(ConSanMoiFenceRecord, epoch), options.moi_owner_epoch_vgprs->epoch);
    if (options.moi_persistent_sgprs.epoch())
      record.scalar(offsetof(ConSanMoiFenceRecord, epoch), *options.moi_persistent_sgprs.epoch());
    record
        .literal(offsetof(ConSanMoiFenceRecord, instruction_offset),
                 static_cast<uint32_t>(fence_text_offset))
        .literal(offsetof(ConSanMoiFenceRecord, kind), static_cast<uint32_t>(kind))
        .literal(offsetof(ConSanMoiFenceRecord, scope), *report_scope)
        .literal(offsetof(ConSanMoiFenceRecord, semantics),
                 communication_site.raw_th.value_or(0u))
        .vgpr(offsetof(ConSanMoiFenceRecord, communication_token), recorded_address_vgpr)
        .vgpr(offsetof(ConSanMoiFenceRecord, communication_token) + sizeof(uint32_t),
              static_cast<uint16_t>(recorded_address_vgpr + 1u));
    require_emission(record.finish(),
                     "ConSan MOI dynamic fence record could not encode record stores");
    require_emission(append_guarded_dynamic_record(words, record_words,
                                                   *options.moi_exec_save_sgpr,
                                                   /*wait_for_global_stores=*/true, arch));
  }
  require_emission(append_restore_moi_special_state(words, options.special_state, arch),
                   "ConSan MOI fence record patch could not restore VCC/SCC");
  if (scalar_spill)
    words.insert(words.end(), scalar_spill->restore_words.begin(),
                 scalar_spill->restore_words.end());
  if (spill)
    words.insert(words.end(), spill->restore_words.begin(), spill->restore_words.end());
  if (select_low_vgpr_bank)
    words.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        0u, static_cast<uint8_t>(*vgpr_msb_mode), arch));
  words.insert(words.end(), displaced_tail_words.begin(), displaced_tail_words.end());
  if (!sequence.finish())
    return std::nullopt;
  return words;
}

} // namespace rocjitsu::consan_moi_impl

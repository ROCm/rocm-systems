// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/modes/sampled/consan_moi_sampled_atomic_emission.h"

#include "rocjitsu/code/patch/consan/modes/sampled/consan_moi_sampled.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/consan/consan_atomic_classifier.h"
#include "rocjitsu/code/patch/consan/consan_capability_contract.h"
#include "rocjitsu/code/patch/consan/consan_moi_dynamic_record_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_memory_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_native_abi.h"
#include "rocjitsu/code/patch/consan/consan_moi_relocation.h"
#include "rocjitsu/code/patch/consan/consan_moi_report_emission.h"
#include "rocjitsu/code/patch/consan/modes/sampled/consan_moi_sampled_window_emission.h"
#include "rocjitsu/code/patch/instruction_sequence.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstring>
#include <limits>

namespace rocjitsu::consan_moi_impl {

using consan_detail::range_overlaps;
using consan_moi_detail::append_atomic_fetch_add_one_u32;
using consan_moi_detail::append_load_u32_vgpr_at_offset;
using consan_moi_detail::append_select_first_lane_in_exec_mask;
using consan_moi_detail::append_store_u32_vgpr_at_offset;
using consan_moi_detail::ConSanMoiRecordEmitter;

std::optional<ConSanMoiSampledSyncRole> sampled_atomic_role(ConSanMoiAtomicEventKind kind,
                                                            bool is_rmw) {
  switch (kind) {
  case ConSanMoiAtomicEventKind::Release:
    return is_rmw ? ConSanMoiSampledSyncRole::RmwRelease : ConSanMoiSampledSyncRole::Release;
  case ConSanMoiAtomicEventKind::Acquire:
    return is_rmw ? ConSanMoiSampledSyncRole::RmwAcquire : ConSanMoiSampledSyncRole::Acquire;
  case ConSanMoiAtomicEventKind::AcquireRelease:
    return is_rmw ? std::optional(ConSanMoiSampledSyncRole::RmwAcquireRelease) : std::nullopt;
  }
  return std::nullopt;
}

[[nodiscard]] bool sampled_atomic_guest_preserves_address(const ConSanAtomicLoweringForm &form) {
  if (!form.destination_vgpr || form.destination_register_count == 0u)
    return true;
  return !range_overlaps(form.address_vgpr, form.address_vgpr_count, *form.destination_vgpr,
                         form.destination_register_count);
}

[[nodiscard]] bool
sampled_atomic_spill_overlaps_guest_operands(const VgprSpillSequence &spill,
                                             const ConSanAtomicLoweringForm &form) {
  const auto overlaps = [&](std::optional<uint16_t> base, uint16_t count) {
    return base && count != 0u && range_overlaps(spill.vgpr_base, spill.vgpr_count, *base, count);
  };
  return range_overlaps(spill.vgpr_base, spill.vgpr_count, form.address_vgpr,
                        form.address_vgpr_count) ||
         overlaps(form.data_vgpr, form.data_register_count) ||
         overlaps(form.destination_vgpr, form.destination_register_count);
}

[[nodiscard]] bool append_sampled_atomic_guest(
    std::vector<uint32_t> &words, std::span<const uint8_t> bytes, const ConSanAtomicSite &site,
    bool is_rmw, rj_code_arch_t arch, uint32_t *guest_instruction_offset,
    std::span<const uint32_t> leading_guest_words, std::span<const uint32_t> trailing_guest_words,
    uint32_t *emitted_guest_size = nullptr) {
  const size_t guest_begin = words.size();
  if (guest_instruction_offset)
    *guest_instruction_offset = static_cast<uint32_t>(words.size() * sizeof(uint32_t));
  words.insert(words.end(), leading_guest_words.begin(), leading_guest_words.end());
  for (uint64_t offset = 0; offset < site.size; offset += sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, bytes.data() + site.file_offset + offset, sizeof(word));
    words.push_back(word);
  }
  // A non-RMW candidate carries its compiler-emitted wait/cache suffix. Do
  // not splice a generic atomic wait into that target-native sequence.
  if (is_rmw && !append_moi_flat_load_wait(words, arch))
    return false;
  words.insert(words.end(), trailing_guest_words.begin(), trailing_guest_words.end());
  if (emitted_guest_size)
    *emitted_guest_size = static_cast<uint32_t>((words.size() - guest_begin) * sizeof(uint32_t));
  return true;
}

struct SampledAtomicPreludeState {
  std::optional<uint16_t> cas_compare_vgpr;
  std::optional<uint16_t> cas_result_vgpr;
};

// Spill builders always produce complete metadata. Keep this check defensive
// for internal callers that construct a sequence directly.
[[nodiscard]] bool validate_sampled_atomic_spill_metadata(const VgprSpillSequence &spill,
                                                          std::vector<std::string> &errors) {
  if (spill.has_complete_slot_metadata())
    return true;
  errors.emplace_back("ConSan MOI sampled atomic spill has incomplete slot metadata");
  return false;
}

/// @pre @p spill has complete slot metadata.
[[nodiscard]] bool append_sampled_atomic_snapshot_word(std::vector<uint32_t> &words,
                                                       const VgprSpillSequence &spill,
                                                       uint16_t destination, uint16_t source,
                                                       rj_code_arch_t arch,
                                                       bool &loaded_private_state) {
  const uint32_t spill_end = static_cast<uint32_t>(spill.vgpr_base) + spill.vgpr_count;
  if (source >= spill.vgpr_base && source < spill_end) {
    const uint16_t slot = static_cast<uint16_t>(source - spill.vgpr_base);
    assert(slot < spill.slot_offsets.size());
    // Spill-range sources may have been overwritten by address or evidence
    // scratch, so reload their authoritative post-guest value from private.
    if (!InstructionSequence(words).emit(
            instrumentation::build_private_load_b32(destination, spill.slot_offsets[slot], arch)))
      return false;
    loaded_private_state = true;
    return true;
  }
  words.push_back(build_v_mov_b32_e32(destination, vector_source_vgpr(source), arch));
  return true;
}

[[nodiscard]] bool append_sampled_atomic_snapshot_wait(std::vector<uint32_t> &words,
                                                       bool loaded_private_state,
                                                       rj_code_arch_t arch) {
  if (!loaded_private_state)
    return true;
  return InstructionSequence(words).emit(instrumentation::build_s_wait_private_load0(arch));
}

[[nodiscard]] bool append_sampled_atomic_cas_snapshot(std::vector<uint32_t> &words,
                                                      const VgprSpillSequence &spill,
                                                      uint16_t compare_vgpr, uint16_t result_vgpr,
                                                      uint16_t saved_compare, uint16_t saved_result,
                                                      rj_code_arch_t arch,
                                                      std::vector<std::string> &errors) {
  if (!validate_sampled_atomic_spill_metadata(spill, errors))
    return false;
  bool loaded_private_state = false;
  if (!append_sampled_atomic_snapshot_word(words, spill, saved_compare, compare_vgpr, arch,
                                           loaded_private_state) ||
      !append_sampled_atomic_snapshot_word(words, spill, saved_result, result_vgpr, arch,
                                           loaded_private_state)) {
    errors.emplace_back("ConSan MOI sampled atomic could not preserve CAS outcome evidence");
    return false;
  }
  if (!append_sampled_atomic_snapshot_wait(words, loaded_private_state, arch)) {
    errors.emplace_back("ConSan MOI sampled atomic could not wait for CAS outcome evidence");
    return false;
  }
  return true;
}

[[nodiscard]] bool
append_sampled_atomic_address_snapshot(std::vector<uint32_t> &words, const VgprSpillSequence &spill,
                                       uint16_t source_address, uint16_t saved_address,
                                       rj_code_arch_t arch, std::vector<std::string> &errors) {
  if (!validate_sampled_atomic_spill_metadata(spill, errors))
    return false;
  bool loaded_private_state = false;
  if (!append_sampled_atomic_snapshot_word(words, spill, saved_address, source_address, arch,
                                           loaded_private_state) ||
      !append_sampled_atomic_snapshot_word(words, spill, static_cast<uint16_t>(saved_address + 1u),
                                           static_cast<uint16_t>(source_address + 1u), arch,
                                           loaded_private_state)) {
    errors.emplace_back("ConSan MOI sampled atomic could not preserve its guest address");
    return false;
  }
  if (!append_sampled_atomic_snapshot_wait(words, loaded_private_state, arch)) {
    errors.emplace_back("ConSan MOI sampled atomic could not wait for its guest address");
    return false;
  }
  return true;
}

[[nodiscard]] bool append_sampled_atomic_prelude(
    std::vector<uint32_t> &words, std::span<const uint8_t> bytes,
    const MoiAtomicEvidenceSourceView &source, const ConSanAtomicLoweringForm &lowering_form,
    uint64_t owner_descriptor_file_offset, const ConSanMoiAtomicAddressPlan &address_plan,
    const MoiSampledSyncEmissionPlan &plan, const VgprSpillSequence *spill,
    const SgprSpillSequence *scalar_spill, const ConSanMoiPrivateStateLayout *private_layout,
    rj_code_arch_t arch, uint16_t saved_address, bool defer_guest, SampledAtomicPreludeState &state,
    std::vector<std::string> &errors, uint32_t *guest_instruction_offset,
    std::span<const uint32_t> leading_guest_words, std::span<const uint32_t> trailing_guest_words,
    uint32_t *emitted_guest_size) {
  const ConSanAtomicSite &site = source.site;
  const bool is_rmw = source.is_rmw();
  const bool is_cas = consan_atomic_is_compare_exchange(site);
  if (is_cas) {
    assert(site.data_vgpr && site.destination_vgpr);
    state.cas_compare_vgpr = static_cast<uint16_t>(*site.data_vgpr + 1u);
    state.cas_result_vgpr = *site.destination_vgpr;
  }
  const bool guest_first =
      spill && sampled_atomic_spill_overlaps_guest_operands(*spill, lowering_form);
  assert(!defer_guest || !guest_first);
  const bool guest_preserves_address = sampled_atomic_guest_preserves_address(lowering_form);
  assert(!guest_first || guest_preserves_address);
  if (guest_first && !guest_preserves_address) {
    errors.emplace_back("ConSan MOI sampled atomic overlap spill cannot preserve guest address");
    return false;
  }
  if (guest_first &&
      !append_sampled_atomic_guest(words, bytes, site, is_rmw, arch, guest_instruction_offset,
                                   leading_guest_words, trailing_guest_words, emitted_guest_size))
    return false;
  if (spill)
    words.insert(words.end(), spill->save_words.begin(), spill->save_words.end());
  if (scalar_spill)
    words.insert(words.end(), scalar_spill->save_words.begin(), scalar_spill->save_words.end());
  if (address_plan.requires_materialization()) {
    const auto special = plan.scalar_abi.special_state;
    if (!special)
      return false;
    const auto materialize = build_consan_moi_atomic_address_materialization(
        address_plan, special->vcc_save_sgpr, special->scc_save_sgpr, arch);
    if (!materialize)
      return false;
    words.insert(words.end(), materialize->begin(), materialize->end());
  }
  if (guest_first) {
    if (address_plan.requires_materialization()) {
      assert(address_plan.result_address_vgpr == saved_address);
      if (address_plan.result_address_vgpr != saved_address) {
        errors.emplace_back(
            "ConSan MOI sampled atomic materialized address does not use the scratch tail");
        return false;
      }
    } else if (!append_sampled_atomic_address_snapshot(
                   words, *spill, address_plan.result_address_vgpr, saved_address, arch, errors)) {
      return false;
    }
  }
  if (guest_first && is_cas) {
    // Address materialization/snapshot must precede CAS evidence because the
    // evidence destinations may overlap the live guest address pair.
    const uint16_t saved_compare =
        static_cast<uint16_t>(plan.scratch_vgpr + SampledAtomicScratchLayout::kCasCompare);
    const uint16_t saved_result =
        static_cast<uint16_t>(plan.scratch_vgpr + SampledAtomicScratchLayout::kCasResult);
    if (!append_sampled_atomic_cas_snapshot(words, *spill, *state.cas_compare_vgpr,
                                            *state.cas_result_vgpr, saved_compare, saved_result,
                                            arch, errors))
      return false;
    state.cas_compare_vgpr = saved_compare;
    state.cas_result_vgpr = saved_result;
  }
  if (private_layout &&
      !append_sampled_private_owner_epoch_load(words, bytes, owner_descriptor_file_offset,
                                               plan.automatic_private_epoch, plan.owner_epoch_vgprs,
                                               *private_layout, arch, errors))
    return false;
  if (plan.persistent_sgprs.complete()) {
    if (!consan_detail::validate_scalar_state_temporaries(
            plan.persistent_sgprs, plan.owner_epoch_vgprs, "sampled atomic prelude", errors))
      return false;
    words.push_back(
        build_v_mov_b32_e32(*plan.owner_epoch_vgprs.owner, *plan.persistent_sgprs.owner(), arch));
    words.push_back(
        build_v_mov_b32_e32(*plan.owner_epoch_vgprs.epoch, *plan.persistent_sgprs.epoch(), arch));
  }
  if (!guest_first && !guest_preserves_address) {
    // Compiler-emitted ordinary acquire loads may reuse the low address VGPR
    // as their destination. Preserve the synchronization address before that
    // load replaces it with the acquired value. Atomic RMWs and disjoint
    // ordinary loads retain the post-guest path below.
    words.push_back(build_v_mov_b32_e32(
        saved_address, vector_source_vgpr(address_plan.result_address_vgpr), arch));
    words.push_back(build_v_mov_b32_e32(
        static_cast<uint16_t>(saved_address + 1u),
        vector_source_vgpr(static_cast<uint16_t>(address_plan.result_address_vgpr + 1u)), arch));
  }
  if (!guest_first) {
    if (!defer_guest) {
      if (!append_sampled_atomic_guest(words, bytes, site, is_rmw, arch, guest_instruction_offset,
                                       leading_guest_words, trailing_guest_words,
                                       emitted_guest_size))
        return false;
    }
    if (guest_preserves_address) {
      words.push_back(build_v_mov_b32_e32(
          saved_address, vector_source_vgpr(address_plan.result_address_vgpr), arch));
      words.push_back(build_v_mov_b32_e32(
          static_cast<uint16_t>(saved_address + 1u),
          vector_source_vgpr(static_cast<uint16_t>(address_plan.result_address_vgpr + 1u)), arch));
    }
  }
  return true;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_sampled_pending_acquire_cave_words(
    std::span<const uint8_t> bytes, const MoiAtomicEvidenceSourceView &source,
    const ConSanAtomicLoweringForm &lowering_form, uint64_t owner_descriptor_file_offset,
    const ConSanMoiAtomicAddressPlan &address_plan, const MoiSampledSyncEmissionPlan &plan,
    const VgprSpillSequence *spill, const SgprSpillSequence *scalar_spill,
    const ConSanMoiPrivateStateLayout *private_layout, rj_code_arch_t arch, uint32_t selected_slot,
    uint32_t bank_count, std::optional<uint32_t> release_selected_slot,
    const ConSanMoiReportBufferLayout &layout, std::vector<std::string> &errors,
    uint32_t *guest_instruction_offset, std::span<const uint32_t> leading_guest_words,
    std::span<const uint32_t> trailing_guest_words, uint32_t *emitted_guest_size) {
  const ConSanAtomicSite &site = source.site;
  const bool is_rmw = source.is_rmw();
  const auto event_kind = moi_atomic_event_kind(source.sequence->memory_role);
  const uint32_t pending_owner_bank_count = consan_moi_sampled_pending_acquire_owner_bank_count(
      layout.sampled_pending_acquire_capacity, layout.sampled_causal_window_capacity);
  if (!plan.owner_epoch_vgprs.owner || !plan.owner_epoch_vgprs.epoch || !plan.exec_save_sgpr ||
      static_cast<uint32_t>(plan.scratch_vgpr) + sampled_atomic_scratch_count() > kMaxVgprs ||
      bank_count == 0u || !std::has_single_bit(bank_count) || pending_owner_bank_count == 0u ||
      selected_slot > layout.sampled_causal_window_capacity ||
      bank_count > layout.sampled_causal_window_capacity - selected_slot ||
      !address_plan.supported() || !site.scope || site.width_bits != 32u)
    return std::nullopt;
  const ConSanTargetProfile *target = consan_target_profile(arch);
  if (target == nullptr)
    return std::nullopt;
  const auto role = event_kind ? sampled_atomic_role(*event_kind, is_rmw) : std::nullopt;
  const auto scope = consan_moi_sampled_sync_scope(*site.scope);
  if (!role || !scope ||
      (static_cast<uint32_t>(*role) & static_cast<uint32_t>(ConSanMoiSampledSyncRole::Acquire)) ==
          0u)
    return std::nullopt;
  if (site.file_offset > bytes.size() || site.size > bytes.size() - site.file_offset)
    return std::nullopt;

  const bool is_cas = is_rmw && consan_atomic_is_compare_exchange(site);
  if (is_cas &&
      (!site.data_vgpr || !site.destination_vgpr || !site.returns_old_value.value_or(false)))
    return std::nullopt;
  const auto descriptor_for = [&](ConSanMoiSampledSyncOutcome outcome) -> std::optional<uint32_t> {
    const auto encoded = encode_consan_moi_sampled_sync_metadata({
        .address = 1,
        .byte_count = 4,
        .kind = ConSanMoiSampledSyncKind::Atomic,
        .role = *role,
        .scope = *scope,
        .outcome = outcome,
    });
    if (encoded.classification != ConSanMoiSampledSyncClassification::Valid)
      return std::nullopt;
    return encoded.packed.descriptor;
  };
  const auto success_descriptor = descriptor_for(
      !is_rmw                                  ? ConSanMoiSampledSyncOutcome::NotApplicable
      : is_cas                                 ? ConSanMoiSampledSyncOutcome::CasSuccess
      : site.returns_old_value.value_or(false) ? ConSanMoiSampledSyncOutcome::RmwReturnsOld
                                               : ConSanMoiSampledSyncOutcome::RmwNoReturn);
  const auto failure_descriptor =
      is_cas ? descriptor_for(ConSanMoiSampledSyncOutcome::CasFailure) : success_descriptor;
  if (!success_descriptor || !failure_descriptor)
    return std::nullopt;

  const uint16_t base = plan.scratch_vgpr;
  const uint16_t value = static_cast<uint16_t>(base + SampledAtomicScratchLayout::kValue);
  const uint16_t expected = static_cast<uint16_t>(base + SampledAtomicScratchLayout::kExpected);
  const uint16_t saved_address =
      static_cast<uint16_t>(base + SampledAtomicScratchLayout::kSavedAddress);
  const uint16_t bank = static_cast<uint16_t>(base + SampledAtomicScratchLayout::kBank);
  const uint16_t original_exec = static_cast<uint16_t>(*plan.exec_save_sgpr + 6u);
  const uint16_t temporary_exec = *plan.exec_save_sgpr;
  const uint64_t report_base = plan.report_buffer_address;
  const uint64_t first_pending_address = report_base + layout.sampled_pending_acquires_offset;

  std::vector<uint32_t> words;
  InstructionSequence sequence(words);
  SampledAtomicPreludeState prelude;
  if (!append_sampled_atomic_prelude(
          words, bytes, source, lowering_form, owner_descriptor_file_offset, address_plan, plan,
          spill, scalar_spill, private_layout, arch, saved_address, /*defer_guest=*/false, prelude,
          errors, guest_instruction_offset, leading_guest_words, trailing_guest_words,
          emitted_guest_size))
    return std::nullopt;
  if (!append_sampled_window_bank_index(words, plan.dispatch_id, plan.workgroup_sources, bank_count,
                                        bank, expected, *plan.owner_epoch_vgprs.owner, arch)) {
    errors.emplace_back("ConSan MOI sampled pending acquire failed at bank selection");
    return std::nullopt;
  }
  if (selected_slot != 0u) {
    if (!sequence.emit(
            instrumentation::build_v_add_u32_literal(bank, expected, selected_slot, bank, arch)))
      return std::nullopt;
  }
  ConSanMoiRecordEmitter record(words, base, value, arch);
  const auto contention_label = sequence.make_label();
  const auto collision_label = sequence.make_label();
  const auto restore_label = sequence.make_label();
  if (!append_save_moi_special_state(words, plan.scalar_abi.special_state, arch)) {
    errors.emplace_back("ConSan MOI sampled barrier failed at special-state save");
    return std::nullopt;
  }
  sequence
      .append(instrumentation::build_s_mov_b64(original_exec, kAmdGpuExecLo, arch),
              instrumentation::build_s_mov_b64(temporary_exec, kAmdGpuExecLo, arch))
      .require(append_select_first_lane_in_exec_mask(words, value, temporary_exec, temporary_exec,
                                                     arch));

  // BANK still names the causal-window slot stored in the pending payload.
  // Address the pending table through a separate (window, owner) index so all
  // waves at one static last-arriver RMW can retain their exact acquire.
  const auto append_pending_address = [&]() {
    if (pending_owner_bank_count > 1u) {
      if (!sequence.emit_all(
              instrumentation::build_v_mul_lo_u32_literal(value, expected, pending_owner_bank_count,
                                                          bank, arch),
              instrumentation::build_v_and_b32_literal(expected, pending_owner_bank_count - 1u,
                                                       *plan.owner_epoch_vgprs.owner, arch),
              instrumentation::build_v_add_u32(value, vector_source_vgpr(value), expected, arch)))
        return false;
    } else {
      words.push_back(build_v_mov_b32_e32(value, vector_source_vgpr(bank), arch));
    }
    return consan_detail::append_moi_indexed_address(
        words,
        {.table_address = first_pending_address,
         .stride_bytes = sizeof(ConSanMoiSampledPendingAcquireSlot),
         .address_vgpr = base,
         .index_vgpr = value},
        *target);
  };
  sequence.require(append_pending_address())
      .append(instrumentation::build_v_mov_b32_literal(value, 1u, arch),
              instrumentation::build_v_mov_b32_literal(expected, 0u, arch),
              instrumentation::build_flat_atomic_cmpswap_b32(
                  base, value, value, /*return_old_value=*/true, kAmdGpuScopeDevice, arch))
      .require(append_moi_global_atomic_wait(words, arch))
      .append(instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), value, arch),
              instrumentation::build_s_and_saveexec_b64(temporary_exec, kAmdGpuVccLo, arch))
      .branch(contention_label, InstructionSequence::BranchKind::SccZero);

  if (release_selected_slot) {
    // RESERVED carries release_slot + 1 (zero means no release attachment).
    // BANK is the absolute acquire slot, so the wrapped unsigned delta also
    // handles a release window whose static base precedes the acquire window.
    const uint32_t release_delta = *release_selected_slot - selected_slot + 1u;
    sequence.append(
        instrumentation::build_v_add_u32_literal(expected, value, release_delta, bank, arch));
  }

  sequence
      .require(record.store_vgpr(offsetof(ConSanMoiSampledPendingAcquireSlot, selected_slot), bank))
      .require(record.store_literal(offsetof(ConSanMoiSampledPendingAcquireSlot, generation),
                                    static_cast<uint32_t>(plan.report_generation)))
      .require(record.store_literal(offsetof(ConSanMoiSampledPendingAcquireSlot, generation) + 4u,
                                    static_cast<uint32_t>(plan.report_generation >> 32u)))
      .require(append_store_moi_report_dispatch_id_pair(
          record, plan.dispatch_id, offsetof(ConSanMoiSampledPendingAcquireSlot, dispatch_id)))
      .require(record.store_workgroup(offsetof(ConSanMoiSampledPendingAcquireSlot, workgroup_x),
                                      plan.workgroup_sources.x))
      .require(record.store_workgroup(offsetof(ConSanMoiSampledPendingAcquireSlot, workgroup_y),
                                      plan.workgroup_sources.y))
      .require(record.store_workgroup(offsetof(ConSanMoiSampledPendingAcquireSlot, workgroup_z),
                                      plan.workgroup_sources.z))
      .require(record.store_vgpr(offsetof(ConSanMoiSampledPendingAcquireSlot, owner_id),
                                 *plan.owner_epoch_vgprs.owner))
      .require(record.store_vgpr(offsetof(ConSanMoiSampledPendingAcquireSlot, source_epoch),
                                 *plan.owner_epoch_vgprs.epoch))
      .require(
          release_selected_slot
              ? record.store_vgpr(offsetof(ConSanMoiSampledPendingAcquireSlot, reserved), expected)
              : record.store_literal(offsetof(ConSanMoiSampledPendingAcquireSlot, reserved), 0u))
      .require(record.store_vgpr(offsetof(ConSanMoiSampledPendingAcquireSlot, metadata) +
                                     offsetof(ConSanMoiSampledSyncMetadataPacked, address),
                                 saved_address))
      .require(record.store_vgpr(offsetof(ConSanMoiSampledPendingAcquireSlot, metadata) +
                                     offsetof(ConSanMoiSampledSyncMetadataPacked, address) + 4u,
                                 static_cast<uint16_t>(saved_address + 1u)))
      .require(record.store_literal(offsetof(ConSanMoiSampledPendingAcquireSlot, metadata) +
                                        offsetof(ConSanMoiSampledSyncMetadataPacked, byte_count),
                                    4u))
      .require(record.store_vgpr(offsetof(ConSanMoiSampledPendingAcquireSlot, metadata) +
                                     offsetof(ConSanMoiSampledSyncMetadataPacked, epoch_before),
                                 *plan.owner_epoch_vgprs.epoch))
      .require(record.store_vgpr(offsetof(ConSanMoiSampledPendingAcquireSlot, metadata) +
                                     offsetof(ConSanMoiSampledSyncMetadataPacked, epoch_after),
                                 *plan.owner_epoch_vgprs.epoch));

  if (is_cas) {
    assert(prelude.cas_compare_vgpr && prelude.cas_result_vgpr);
    const auto cas_failure_label = sequence.make_label();
    const auto descriptor_done_label = sequence.make_label();
    sequence
        .append(instrumentation::build_v_cmp_eq_u32_vcc(
            vector_source_vgpr(*prelude.cas_compare_vgpr), *prelude.cas_result_vgpr, arch))
        .branch(cas_failure_label, InstructionSequence::BranchKind::VccZero)
        .require(record.store_literal(offsetof(ConSanMoiSampledPendingAcquireSlot, metadata) +
                                          offsetof(ConSanMoiSampledSyncMetadataPacked, descriptor),
                                      *success_descriptor))
        .branch(descriptor_done_label, InstructionSequence::BranchKind::Unconditional)
        .bind_label(cas_failure_label)
        .require(record.store_literal(offsetof(ConSanMoiSampledPendingAcquireSlot, metadata) +
                                          offsetof(ConSanMoiSampledSyncMetadataPacked, descriptor),
                                      *failure_descriptor))
        .bind_label(descriptor_done_label);
  } else {
    sequence.require(
        record.store_literal(offsetof(ConSanMoiSampledPendingAcquireSlot, metadata) +
                                 offsetof(ConSanMoiSampledSyncMetadataPacked, descriptor),
                             *success_descriptor));
  }

  sequence.append(instrumentation::build_s_wait_global_store0(arch))
      .require(append_pending_address())
      .append(instrumentation::build_v_mov_b32_literal(value, 2u, arch),
              instrumentation::build_v_mov_b32_literal(expected, 1u, arch),
              instrumentation::build_flat_atomic_cmpswap_b32(
                  base, value, value, /*return_old_value=*/true, kAmdGpuScopeDevice, arch))
      .require(append_moi_global_atomic_wait(words, arch))
      .append(instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(expected), value, arch),
              instrumentation::build_s_and_saveexec_b64(temporary_exec, kAmdGpuVccLo, arch))
      .branch(collision_label, InstructionSequence::BranchKind::SccZero)
      .require(append_atomic_fetch_add_one_u32(
          words, report_base + offsetof(ConSanMoiReportHeader, sampled_pending_acquire_count),
          value, base, arch))
      .branch(restore_label, InstructionSequence::BranchKind::Unconditional);

  const auto append_counter_path = [&](size_t counter_offset) {
    sequence
        .append(instrumentation::build_s_mov_b64(kAmdGpuExecLo, original_exec, arch),
                instrumentation::build_s_mov_b64(temporary_exec, kAmdGpuExecLo, arch))
        .require(append_select_first_lane_in_exec_mask(words, value, temporary_exec, temporary_exec,
                                                       arch))
        .require(
            append_atomic_fetch_add_one_u32(words, report_base + counter_offset, value, base, arch))
        .branch(restore_label, InstructionSequence::BranchKind::Unconditional);
  };
  // The sole successful claim chooses one sampled wave. Losing claims are
  // ordinary bounded contention and never write the pending record. The later
  // read requires the exact winner, so winner-without-read remains fail closed.
  sequence.bind_label(contention_label);
  append_counter_path(offsetof(ConSanMoiReportHeader, sampled_pending_acquire_contention_count));
  sequence.bind_label(collision_label);
  append_counter_path(offsetof(ConSanMoiReportHeader, sampled_pending_acquire_collision_count));

  sequence.bind_label(restore_label)
      .append(instrumentation::build_s_mov_b64(kAmdGpuExecLo, original_exec, arch))
      .require(append_restore_moi_special_state(words, plan.scalar_abi.special_state, arch));
  if (scalar_spill)
    words.insert(words.end(), scalar_spill->restore_words.begin(),
                 scalar_spill->restore_words.end());
  if (spill)
    words.insert(words.end(), spill->restore_words.begin(), spill->restore_words.end());
  if (!sequence.finish(arch))
    return std::nullopt;
  return words;
}
[[nodiscard]] std::optional<std::vector<uint32_t>> build_sampled_atomic_sync_cave_words(
    std::span<const uint8_t> bytes, const MoiAtomicEvidenceSourceView &source,
    const ConSanAtomicLoweringForm &lowering_form, uint64_t owner_descriptor_file_offset,
    const ConSanMoiAtomicAddressPlan &address_plan, const MoiSampledSyncEmissionPlan &plan,
    const VgprSpillSequence *spill, const SgprSpillSequence *scalar_spill,
    const ConSanMoiPrivateStateLayout *private_layout, rj_code_arch_t arch, uint32_t selected_slot,
    uint32_t bank_count, const ConSanMoiReportBufferLayout &layout,
    std::vector<std::string> &errors, uint32_t *guest_instruction_offset,
    std::span<const uint32_t> leading_guest_words, std::span<const uint32_t> trailing_guest_words,
    uint32_t *emitted_guest_size) {
  const ConSanAtomicSite &site = source.site;
  const bool is_rmw = source.is_rmw();
  const auto event_kind = moi_atomic_event_kind(source.sequence->memory_role);
  if (!plan.owner_epoch_vgprs.owner || !plan.owner_epoch_vgprs.epoch || !plan.exec_save_sgpr ||
      static_cast<uint32_t>(plan.scratch_vgpr) + sampled_atomic_scratch_count() > kMaxVgprs ||
      bank_count == 0u || !std::has_single_bit(bank_count) ||
      selected_slot > layout.sampled_sync_metadata_capacity ||
      bank_count > layout.sampled_sync_metadata_capacity - selected_slot ||
      selected_slot > layout.sampled_causal_window_capacity ||
      bank_count > layout.sampled_causal_window_capacity - selected_slot ||
      selected_slot > layout.sampled_watchpoint_capacity ||
      bank_count > layout.sampled_watchpoint_capacity - selected_slot)
    return std::nullopt;
  const ConSanTargetProfile *target = consan_target_profile(arch);
  if (target == nullptr)
    return std::nullopt;
  if (!address_plan.supported() || !site.scope || site.width_bits != 32u)
    return std::nullopt;
  const auto role = event_kind ? sampled_atomic_role(*event_kind, is_rmw) : std::nullopt;
  const auto scope = consan_moi_sampled_sync_scope(*site.scope);
  if (!role || !scope)
    return std::nullopt;
  if (site.file_offset > bytes.size() || site.size > bytes.size() - site.file_offset) {
    errors.emplace_back("ConSan MOI sampled atomic metadata site exceeds ELF bytes");
    return std::nullopt;
  }

  const bool is_cas = is_rmw && consan_atomic_is_compare_exchange(site);
  if (is_cas &&
      (!site.data_vgpr || !site.destination_vgpr || !site.returns_old_value.value_or(false))) {
    errors.emplace_back("ConSan MOI sampled atomic metadata requires an exact CAS outcome");
    return std::nullopt;
  }

  const auto descriptor_for = [&](ConSanMoiSampledSyncOutcome outcome) -> std::optional<uint32_t> {
    const ConSanMoiSampledSyncEncodeResult encoded = encode_consan_moi_sampled_sync_metadata({
        .address = 1,
        .byte_count = 4,
        .kind = ConSanMoiSampledSyncKind::Atomic,
        .role = *role,
        .scope = *scope,
        .outcome = outcome,
        .epoch_before = 0,
        .epoch_after = 0,
    });
    if (encoded.classification != ConSanMoiSampledSyncClassification::Valid)
      return std::nullopt;
    return encoded.packed.descriptor;
  };
  const auto success_descriptor = descriptor_for(
      !is_rmw                                  ? ConSanMoiSampledSyncOutcome::NotApplicable
      : is_cas                                 ? ConSanMoiSampledSyncOutcome::CasSuccess
      : site.returns_old_value.value_or(false) ? ConSanMoiSampledSyncOutcome::RmwReturnsOld
                                               : ConSanMoiSampledSyncOutcome::RmwNoReturn);
  const auto failure_descriptor =
      is_cas ? descriptor_for(ConSanMoiSampledSyncOutcome::CasFailure) : success_descriptor;
  if (!success_descriptor || !failure_descriptor)
    return std::nullopt;

  const uint16_t base = plan.scratch_vgpr;
  const uint16_t value = static_cast<uint16_t>(base + SampledAtomicScratchLayout::kValue);
  const uint16_t expected = static_cast<uint16_t>(base + SampledAtomicScratchLayout::kExpected);
  const uint16_t saved_address =
      static_cast<uint16_t>(base + SampledAtomicScratchLayout::kSavedAddress);
  const uint16_t bank = static_cast<uint16_t>(base + SampledAtomicScratchLayout::kBank);
  const uint16_t original_exec = static_cast<uint16_t>(*plan.exec_save_sgpr + 6u);
  const uint16_t temporary_exec = *plan.exec_save_sgpr;
  const uint64_t report_base = plan.report_buffer_address;
  const uint64_t first_window_address = report_base + layout.sampled_causal_windows_offset;
  const uint64_t first_watchpoint_address = report_base + layout.sampled_watchpoints_offset;
  const uint64_t first_metadata_address = report_base + layout.sampled_sync_metadata_offset;

  // A release-only RMW that does not return a value has no outcome evidence
  // needed by the probe. Publish its shadow metadata before executing the
  // guest release so the release operation orders that metadata and the
  // original code can resume immediately after the externally visible RMW.
  // This is especially important for code that publishes its own side
  // metadata immediately after the atomic. Keep guest-first ordering when a
  // spill overlaps guest operands, and for acquire/CAS operations whose
  // outcome is needed by the probe.
  const bool release_only =
      (static_cast<uint32_t>(*role) & static_cast<uint32_t>(ConSanMoiSampledSyncRole::Release)) !=
          0u &&
      (static_cast<uint32_t>(*role) & static_cast<uint32_t>(ConSanMoiSampledSyncRole::Acquire)) ==
          0u;
  const bool defer_guest =
      release_only && !is_cas &&
      !(spill && sampled_atomic_spill_overlaps_guest_operands(*spill, lowering_form));

  std::vector<uint32_t> words;
  InstructionSequence sequence(words);
  SampledAtomicPreludeState prelude;
  sequence.require(append_sampled_atomic_prelude(
      words, bytes, source, lowering_form, owner_descriptor_file_offset, address_plan, plan, spill,
      scalar_spill, private_layout, arch, saved_address, defer_guest, prelude, errors,
      guest_instruction_offset, leading_guest_words, trailing_guest_words, emitted_guest_size));
  if (sequence &&
      !append_sampled_window_bank_index(words, plan.dispatch_id, plan.workgroup_sources, bank_count,
                                        bank, expected, *plan.owner_epoch_vgprs.owner, arch)) {
    errors.emplace_back("ConSan MOI sampled atomic metadata failed at bank selection");
    sequence.require(false);
  }
  if (selected_slot != 0u)
    sequence.append(
        instrumentation::build_v_add_u32_literal(bank, expected, selected_slot, bank, arch));
  ConSanMoiRecordEmitter record(words, base, value, arch);
  const auto collision_label = sequence.make_label();
  const auto restore_label = sequence.make_label();
  sequence.require(append_save_moi_special_state(words, plan.scalar_abi.special_state, arch))
      .append(instrumentation::build_s_mov_b64(original_exec, kAmdGpuExecLo, arch))
      .require(consan_detail::append_moi_indexed_address(
          words,
          {.table_address = first_window_address,
           .stride_bytes = sizeof(ConSanMoiSampledCausalWindow),
           .address_vgpr = base,
           .index_vgpr = bank},
          *target))
      .require(append_sampled_causal_window_validation(words, sequence,
                                                       {.generation = plan.report_generation,
                                                        .dispatch_id = plan.dispatch_id,
                                                        .workgroup_sources = plan.workgroup_sources,
                                                        .address_vgpr = base,
                                                        .value_vgpr = value,
                                                        .expected_vgpr = expected,
                                                        .epoch_vgpr = *plan.owner_epoch_vgprs.epoch,
                                                        .first_entry_vgpr = bank,
                                                        .temporary_exec_sgpr = temporary_exec,
                                                        .mismatch_label = restore_label,
                                                        .arch = arch}))
      .require(consan_detail::append_moi_indexed_address(words,
                                                         {.table_address = first_watchpoint_address,
                                                          .stride_bytes = sizeof(uint64_t),
                                                          .address_vgpr = base,
                                                          .index_vgpr = bank},
                                                         *target));
  const auto narrow_current_vcc = [&]() {
    sequence.append(instrumentation::build_s_and_saveexec_b64(temporary_exec, kAmdGpuVccLo, arch))
        .branch(restore_label, InstructionSequence::BranchKind::SccZero);
  };
  const auto narrow_masked_low = [&](uint32_t mask, uint32_t wanted) {
    sequence.require(append_load_u32_vgpr_at_offset(words, base, 0u, value, arch))
        .append(instrumentation::build_v_and_b32_literal(value, mask, value, arch),
                instrumentation::build_v_mov_b32_literal(expected, wanted, arch),
                instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(expected), value, arch));
    narrow_current_vcc();
  };
  narrow_masked_low(static_cast<uint32_t>(consan_moi_sampled_watchpoint::valid_mask),
                    static_cast<uint32_t>(consan_moi_sampled_watchpoint::valid_mask));
  narrow_masked_low(static_cast<uint32_t>(consan_moi_sampled_watchpoint::consumed_mask), 0u);
  sequence.require(append_load_u32_vgpr_at_offset(words, base, 0u, value, arch))
      .append(instrumentation::build_v_lshrrev_b32(
                  value,
                  scalar_positive_inline_u32(consan_moi_sampled_watchpoint::access_kind_shift),
                  value, arch),
              instrumentation::build_v_and_b32_literal(
                  value, (1u << consan_moi_sampled_watchpoint::access_kind_bits) - 1u, value, arch),
              instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0u), value, arch));
  narrow_current_vcc();
  sequence.append(
      instrumentation::build_v_cmp_gt_u32_vcc(scalar_positive_inline_u32(3u), value, arch));
  narrow_current_vcc();
  sequence.require(append_load_u32_vgpr_at_offset(words, base, 0u, value, arch))
      .append(instrumentation::build_v_lshrrev_b32(
                  value, scalar_positive_inline_u32(consan_moi_sampled_watchpoint::owner_shift),
                  value, arch),
              instrumentation::build_v_and_b32_literal(
                  value, consan_moi_sampled_watchpoint::max_owner, value, arch),
              instrumentation::build_v_cmp_eq_u32_vcc(
                  vector_source_vgpr(*plan.owner_epoch_vgprs.owner), value, arch),
              instrumentation::build_s_and_saveexec_b64(temporary_exec, kAmdGpuVccLo, arch))
      .branch(restore_label, InstructionSequence::BranchKind::SccZero);

  sequence.require(append_load_u32_vgpr_at_offset(words, base, 0u, value, arch))
      .append(instrumentation::build_v_lshrrev_b32(
                  value, scalar_positive_inline_u32(consan_moi_sampled_watchpoint::epoch_shift),
                  value, arch),
              instrumentation::build_v_and_b32_literal(
                  value, consan_moi_sampled_watchpoint::max_epoch, value, arch),
              instrumentation::build_v_cmp_eq_u32_vcc(
                  vector_source_vgpr(*plan.owner_epoch_vgprs.epoch), value, arch));
  narrow_current_vcc();
  sequence.require(append_load_u32_vgpr_at_offset(words, base, 0u, value, arch))
      .require(append_load_u32_vgpr_at_offset(words, base, sizeof(uint32_t), expected, arch))
      .append(instrumentation::build_v_lshrrev_b32(
                  value,
                  scalar_positive_inline_u32(consan_moi_sampled_watchpoint::generation_shift),
                  value, arch),
              instrumentation::build_v_lshlrev_b32(
                  expected,
                  scalar_positive_inline_u32(32u - consan_moi_sampled_watchpoint::generation_shift),
                  expected, arch),
              instrumentation::build_v_add_u32(value, vector_source_vgpr(expected), value, arch),
              instrumentation::build_v_and_b32_literal(
                  value, consan_moi_sampled_watchpoint::max_generation, value, arch),
              instrumentation::build_v_mov_b32_literal(
                  expected,
                  static_cast<uint32_t>(plan.report_generation) &
                      consan_moi_sampled_watchpoint::max_generation,
                  arch),
              instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(expected), value, arch));
  narrow_current_vcc();

  sequence.append(instrumentation::build_s_mov_b64(temporary_exec, kAmdGpuExecLo, arch))
      .require(append_select_first_lane_in_exec_mask(words, value, temporary_exec, temporary_exec,
                                                     arch));

  const auto append_publication = [&](uint32_t descriptor) {
    sequence
        .require(consan_detail::append_moi_indexed_address(
            words,
            {.table_address =
                 first_metadata_address + offsetof(ConSanMoiSampledSyncMetadataPacked, descriptor),
             .stride_bytes = sizeof(ConSanMoiSampledSyncMetadataPacked),
             .address_vgpr = base,
             .index_vgpr = bank},
            *target))
        .append(instrumentation::build_v_mov_b32_literal(
                    value, kConSanMoiSampledSyncPublishingDescriptor, arch),
                instrumentation::build_v_mov_b32_literal(expected, 0, arch),
                instrumentation::build_flat_atomic_cmpswap_b32(base, value, value, true,
                                                               kAmdGpuScopeDevice, arch))
        .require(append_moi_global_atomic_wait(words, arch))
        .append(instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), value, arch),
                instrumentation::build_s_and_saveexec_b64(temporary_exec, kAmdGpuVccLo, arch))
        .branch(collision_label, InstructionSequence::BranchKind::SccZero)
        .require(consan_detail::append_moi_indexed_address(
            words,
            {.table_address = first_metadata_address,
             .stride_bytes = sizeof(ConSanMoiSampledSyncMetadataPacked),
             .address_vgpr = base,
             .index_vgpr = bank},
            *target))
        .require(
            record.store_vgpr(offsetof(ConSanMoiSampledSyncMetadataPacked, address), saved_address))
        .require(record.store_vgpr(offsetof(ConSanMoiSampledSyncMetadataPacked, address) + 4u,
                                   static_cast<uint16_t>(saved_address + 1u)))
        .require(record.store_literal(offsetof(ConSanMoiSampledSyncMetadataPacked, byte_count), 4u))
        .require(record.store_vgpr(offsetof(ConSanMoiSampledSyncMetadataPacked, epoch_before),
                                   *plan.owner_epoch_vgprs.epoch))
        .require(record.store_vgpr(offsetof(ConSanMoiSampledSyncMetadataPacked, epoch_after),
                                   *plan.owner_epoch_vgprs.epoch))
        .append(instrumentation::build_s_wait_global_store0(arch))
        .require(consan_detail::append_moi_indexed_address(
            words,
            {.table_address =
                 first_metadata_address + offsetof(ConSanMoiSampledSyncMetadataPacked, descriptor),
             .stride_bytes = sizeof(ConSanMoiSampledSyncMetadataPacked),
             .address_vgpr = base,
             .index_vgpr = bank},
            *target))
        .append(instrumentation::build_v_mov_b32_literal(value, descriptor, arch),
                instrumentation::build_v_mov_b32_literal(
                    expected, kConSanMoiSampledSyncPublishingDescriptor, arch),
                instrumentation::build_flat_atomic_cmpswap_b32(base, value, value, true,
                                                               kAmdGpuScopeDevice, arch))
        .require(append_moi_global_atomic_wait(words, arch))
        .append(instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(expected), value, arch),
                instrumentation::build_s_and_saveexec_b64(temporary_exec, kAmdGpuVccLo, arch))
        .branch(collision_label, InstructionSequence::BranchKind::SccZero)
        .require(append_atomic_fetch_add_one_u32(
            words, report_base + offsetof(ConSanMoiReportHeader, sampled_sync_metadata_count),
            value, base, arch))
        .branch(restore_label, InstructionSequence::BranchKind::Unconditional);
  };

  if (is_cas) {
    assert(prelude.cas_compare_vgpr && prelude.cas_result_vgpr);
    const auto cas_failure_label = sequence.make_label();
    sequence
        .append(instrumentation::build_v_cmp_eq_u32_vcc(
            vector_source_vgpr(*prelude.cas_compare_vgpr), *prelude.cas_result_vgpr, arch))
        .branch(cas_failure_label, InstructionSequence::BranchKind::VccZero);
    append_publication(*success_descriptor);
    sequence.bind_label(cas_failure_label);
  }
  append_publication(is_cas ? *failure_descriptor : *success_descriptor);

  // A failed claim/commit reaches here after its cumulative EXEC intersection
  // became empty. Reconstitute the guest mask and select exactly one active
  // lane so the collision counter is neither lost nor multiplied by wave size.
  sequence.bind_label(collision_label)
      .append(instrumentation::build_s_mov_b64(kAmdGpuExecLo, original_exec, arch),
              instrumentation::build_s_mov_b64(temporary_exec, original_exec, arch))
      .require(
          append_select_first_lane_in_exec_mask(words, value, temporary_exec, temporary_exec, arch))
      .require(append_atomic_fetch_add_one_u32(
          words, report_base + offsetof(ConSanMoiReportHeader, sampled_dropped_window_count), value,
          base, arch))
      .bind_label(restore_label)
      .append(instrumentation::build_s_mov_b64(kAmdGpuExecLo, original_exec, arch))
      .require(append_restore_moi_special_state(words, plan.scalar_abi.special_state, arch));
  if (scalar_spill)
    words.insert(words.end(), scalar_spill->restore_words.begin(),
                 scalar_spill->restore_words.end());
  if (spill)
    words.insert(words.end(), spill->restore_words.begin(), spill->restore_words.end());
  if (defer_guest) {
    std::vector<uint32_t> deferred_guest_words;
    const bool guest_ok =
        append_sampled_atomic_guest(deferred_guest_words, bytes, site, is_rmw, arch,
                                    /*guest_instruction_offset=*/nullptr, leading_guest_words,
                                    trailing_guest_words, emitted_guest_size);
    sequence.require(guest_ok);
    if (guest_ok) {
      if (guest_instruction_offset)
        *guest_instruction_offset = static_cast<uint32_t>(words.size() * sizeof(uint32_t));
      words.insert(words.end(), deferred_guest_words.begin(), deferred_guest_words.end());
    }
  }

  if (!sequence.finish(arch))
    return std::nullopt;
  return words;
}

} // namespace rocjitsu::consan_moi_impl

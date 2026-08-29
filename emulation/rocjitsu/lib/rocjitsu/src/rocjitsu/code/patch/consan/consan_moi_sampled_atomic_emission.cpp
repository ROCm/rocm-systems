// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_sampled_atomic_emission.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/consan/consan_atomic_classifier.h"
#include "rocjitsu/code/patch/consan/consan_capability_contract.h"
#include "rocjitsu/code/patch/consan/consan_moi_dynamic_record_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_memory_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_native_abi.h"
#include "rocjitsu/code/patch/consan/consan_moi_relocation.h"
#include "rocjitsu/code/patch/consan/consan_moi_report_emission.h"
#include "rocjitsu/code/patch/instruction_sequence.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstring>
#include <limits>

namespace rocjitsu::consan_moi_impl {

using consan_detail::append_reload_moi_spilled_vgpr;
using consan_detail::MoiAtomicEvidenceSitePlan;
using consan_detail::MoiSpecialStateSgprs;
using consan_detail::MoiSpilledVgprReloadResult;
using consan_detail::range_overlaps;
using consan_moi_detail::append_atomic_fetch_add_one_u32;
using consan_moi_detail::append_load_u32_vgpr_at_offset;
using consan_moi_detail::append_moi_prepare_scc_preserving_indirect_jump;
using consan_moi_detail::append_store_u32_literal;
using consan_moi_detail::append_store_u32_vgpr;
using consan_moi_detail::append_store_u32_vgpr_at_offset;
using consan_moi_detail::ConSanMoiLiteralDispatchIdPolicy;
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

[[nodiscard]] std::optional<ConSanMoiSampledSyncScope> sampled_atomic_scope(uint32_t raw_scope) {
  if (raw_scope == 0u)
    return ConSanMoiSampledSyncScope::Wavefront;
  if (raw_scope == 1u)
    return ConSanMoiSampledSyncScope::Workgroup;
  if (raw_scope == 2u)
    return ConSanMoiSampledSyncScope::Agent;
  if (raw_scope == 3u)
    return ConSanMoiSampledSyncScope::System;
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

[[nodiscard]] bool append_sampled_atomic_guest(std::vector<uint32_t> &words,
                                               std::span<const uint8_t> bytes,
                                               const MoiAtomicEvidenceSitePlan &candidate,
                                               rj_code_arch_t arch,
                                               uint32_t *guest_instruction_offset) {
  if (guest_instruction_offset)
    *guest_instruction_offset = static_cast<uint32_t>(words.size() * sizeof(uint32_t));
  for (uint64_t offset = 0; offset < candidate.site.size; offset += sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, bytes.data() + candidate.site.file_offset + offset, sizeof(word));
    words.push_back(word);
  }
  // A non-RMW candidate carries its compiler-emitted wait/cache suffix. Do
  // not splice a generic atomic wait into that target-native sequence.
  return !candidate.is_rmw || append_moi_flat_load_wait(words, arch);
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
    const auto load =
        instrumentation::build_private_load_b32(destination, spill.slot_offsets[slot], arch);
    if (!load)
      return false;
    words.insert(words.end(), load->begin(), load->end());
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
  const auto wait = instrumentation::build_s_wait_private_load0(arch);
  if (!wait)
    return false;
  words.push_back(*wait);
  return true;
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
    const MoiAtomicEvidenceSitePlan &candidate, const ConSanMoiAtomicAddressPlan &address_plan,
    const ConSanRequest &request, const ConSanMoiOperatingPoint &point, uint16_t scratch_vgpr,
    const VgprSpillSequence *spill, const SgprSpillSequence *scalar_spill,
    const MoiPrivateEpochLayout *private_layout, rj_code_arch_t arch, uint16_t saved_address,
    bool defer_guest, SampledAtomicPreludeState &state, std::vector<std::string> &errors,
    uint32_t *guest_instruction_offset, std::span<const uint32_t> trailing_guest_words) {
  const bool is_cas = consan_atomic_is_compare_exchange(candidate.site);
  if (is_cas) {
    assert(candidate.site.data_vgpr && candidate.site.dst_vgpr);
    state.cas_compare_vgpr = static_cast<uint16_t>(*candidate.site.data_vgpr + 1u);
    state.cas_result_vgpr = *candidate.site.dst_vgpr;
  }
  const bool guest_first =
      spill && sampled_atomic_spill_overlaps_guest_operands(*spill, candidate.lowering_form);
  assert(!defer_guest || !guest_first);
  const bool guest_preserves_address =
      sampled_atomic_guest_preserves_address(candidate.lowering_form);
  assert(!guest_first || guest_preserves_address);
  if (guest_first && !guest_preserves_address) {
    errors.emplace_back("ConSan MOI sampled atomic overlap spill cannot preserve guest address");
    return false;
  }
  if (guest_first &&
      !append_sampled_atomic_guest(words, bytes, candidate, arch, guest_instruction_offset))
    return false;
  if (guest_first)
    words.insert(words.end(), trailing_guest_words.begin(), trailing_guest_words.end());
  if (spill)
    words.insert(words.end(), spill->save_words.begin(), spill->save_words.end());
  if (scalar_spill)
    words.insert(words.end(), scalar_spill->save_words.begin(), scalar_spill->save_words.end());
  if (address_plan.requires_materialization()) {
    const auto special = moi_special_state_sgprs(request, point);
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
        static_cast<uint16_t>(scratch_vgpr + SampledAtomicScratchLayout::kCasCompare);
    const uint16_t saved_result =
        static_cast<uint16_t>(scratch_vgpr + SampledAtomicScratchLayout::kCasResult);
    if (!append_sampled_atomic_cas_snapshot(words, *spill, *state.cas_compare_vgpr,
                                            *state.cas_result_vgpr, saved_compare, saved_result,
                                            arch, errors))
      return false;
    state.cas_compare_vgpr = saved_compare;
    state.cas_result_vgpr = saved_result;
  }
  if (private_layout && !append_sampled_private_owner_epoch_load(
                            words, bytes, *candidate.kernel_descriptor_file_offset, point,
                            *private_layout, arch, errors))
    return false;
  if (point.moi_persistent_sgprs.complete()) {
    if (!consan_detail::validate_scalar_state_temporaries(point, "sampled atomic prelude", errors))
      return false;
    words.push_back(
        build_v_mov_b32_e32(*point.moi_owner_vgpr, *point.moi_persistent_sgprs.owner, arch));
    words.push_back(
        build_v_mov_b32_e32(*point.moi_epoch_vgpr, *point.moi_persistent_sgprs.epoch, arch));
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
      if (!append_sampled_atomic_guest(words, bytes, candidate, arch, guest_instruction_offset))
        return false;
      words.insert(words.end(), trailing_guest_words.begin(), trailing_guest_words.end());
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
    std::span<const uint8_t> bytes, const MoiAtomicEvidenceSitePlan &candidate,
    const ConSanMoiAtomicAddressPlan &address_plan, const ConSanRequest &request,
    const BoundRuntimeResources &bound_resources, const ConSanMoiOperatingPoint &point,
    uint16_t scratch_vgpr, const VgprSpillSequence *spill, const SgprSpillSequence *scalar_spill,
    const MoiPrivateEpochLayout *private_layout, rj_code_arch_t arch, uint32_t selected_slot,
    uint32_t bank_count, std::optional<uint32_t> release_selected_slot,
    const ConSanMoiReportBufferLayout &layout, uint64_t cave_text_offset,
    uint64_t return_text_offset, std::vector<std::string> &errors,
    uint32_t *guest_instruction_offset, std::span<const uint32_t> trailing_guest_words) {
  const uint32_t pending_owner_bank_count = consan_moi_sampled_pending_acquire_owner_bank_count(
      layout.sampled_pending_acquire_capacity, layout.sampled_causal_window_capacity);
  if (!bound_resources.moi_report_buffer_address || !point.moi_owner_vgpr ||
      !point.moi_epoch_vgpr || !point.moi_exec_save_sgpr ||
      static_cast<uint32_t>(scratch_vgpr) + sampled_atomic_scratch_count() > kMaxVgprs ||
      bank_count == 0u || !std::has_single_bit(bank_count) || pending_owner_bank_count == 0u ||
      selected_slot > layout.sampled_causal_window_capacity ||
      bank_count > layout.sampled_causal_window_capacity - selected_slot ||
      !address_plan.supported() || !candidate.site.raw_scope || candidate.site.width_bits != 32u ||
      !candidate.kernel_descriptor_file_offset)
    return std::nullopt;
  const auto role = sampled_atomic_role(candidate.event_kind, candidate.is_rmw);
  const auto scope = sampled_atomic_scope(*candidate.site.raw_scope);
  if (!role || !scope ||
      (static_cast<uint32_t>(*role) & static_cast<uint32_t>(ConSanMoiSampledSyncRole::Acquire)) ==
          0u)
    return std::nullopt;
  const auto workgroup_sources = moi_persistent_or_descriptor_workgroup_sources(
      bytes, *candidate.kernel_descriptor_file_offset, request.moi_engine, point, arch, errors,
      candidate.uses_cluster_workgroup_id);
  if (!workgroup_sources || candidate.site.file_offset > bytes.size() ||
      candidate.site.size > bytes.size() - candidate.site.file_offset)
    return std::nullopt;

  const bool is_cas = candidate.is_rmw && consan_atomic_is_compare_exchange(candidate.site);
  if (is_cas && (!candidate.site.data_vgpr || !candidate.site.dst_vgpr ||
                 !candidate.site.returns_old_value.value_or(false)))
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
  const auto success_descriptor =
      descriptor_for(!candidate.is_rmw ? ConSanMoiSampledSyncOutcome::NotApplicable
                     : is_cas          ? ConSanMoiSampledSyncOutcome::CasSuccess
                     : candidate.site.returns_old_value.value_or(false)
                         ? ConSanMoiSampledSyncOutcome::RmwReturnsOld
                         : ConSanMoiSampledSyncOutcome::RmwNoReturn);
  const auto failure_descriptor =
      is_cas ? descriptor_for(ConSanMoiSampledSyncOutcome::CasFailure) : success_descriptor;
  if (!success_descriptor || !failure_descriptor)
    return std::nullopt;

  const uint16_t base = scratch_vgpr;
  const uint16_t value = static_cast<uint16_t>(base + SampledAtomicScratchLayout::kValue);
  const uint16_t expected = static_cast<uint16_t>(base + SampledAtomicScratchLayout::kExpected);
  const uint16_t saved_address =
      static_cast<uint16_t>(base + SampledAtomicScratchLayout::kSavedAddress);
  const uint16_t bank = static_cast<uint16_t>(base + SampledAtomicScratchLayout::kBank);
  const uint16_t original_exec = static_cast<uint16_t>(*point.moi_exec_save_sgpr + 6u);
  const uint16_t temporary_exec = *point.moi_exec_save_sgpr;
  const uint64_t report_base = *bound_resources.moi_report_buffer_address;
  const uint64_t first_pending_address = report_base + layout.sampled_pending_acquires_offset;

  std::vector<uint32_t> words;
  SampledAtomicPreludeState prelude;
  if (!append_sampled_atomic_prelude(words, bytes, candidate, address_plan, request, point, base,
                                     spill, scalar_spill, private_layout, arch, saved_address,
                                     /*defer_guest=*/false, prelude, errors,
                                     guest_instruction_offset, trailing_guest_words))
    return std::nullopt;
  if (!append_sampled_window_bank_index(words, point, bound_resources, *workgroup_sources,
                                        bank_count, bank, expected, *point.moi_owner_vgpr, arch)) {
    errors.emplace_back("ConSan MOI sampled pending acquire failed at bank selection");
    return std::nullopt;
  }
  if (selected_slot != 0u) {
    const auto absolute_slot =
        instrumentation::build_v_add_u32_literal(bank, expected, selected_slot, bank, arch);
    if (!absolute_slot)
      return std::nullopt;
    words.insert(words.end(), absolute_slot->begin(), absolute_slot->end());
  }
  InstructionSequence sequence(words);
  ConSanMoiRecordEmitter record(words, base, value, arch);
  const auto contention_label = sequence.make_label();
  const auto collision_label = sequence.make_label();
  const auto restore_label = sequence.make_label();
  if (!append_save_moi_special_state(words, moi_special_state_sgprs(request, point), arch)) {
    errors.emplace_back("ConSan MOI sampled barrier failed at special-state save");
    return std::nullopt;
  }
  const auto save_exec = instrumentation::build_s_mov_b64(original_exec, kRdna4ExecLo, arch);
  const auto active_mask = instrumentation::build_s_mov_b64(temporary_exec, kRdna4ExecLo, arch);
  const auto mbcnt_lo = instrumentation::build_v_mbcnt_lo_u32_b32(
      value, temporary_exec, scalar_positive_inline_u32(0), arch);
  const auto mbcnt_hi = instrumentation::build_v_mbcnt_hi_u32_b32(
      value, static_cast<uint16_t>(temporary_exec + 1u), vector_source_vgpr(value), arch);
  const auto first =
      instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), value, arch);
  const auto narrow_first =
      instrumentation::build_s_and_saveexec_b64(temporary_exec, kRdna4VccLo, arch);
  if (!sequence.emit_all(save_exec, active_mask, mbcnt_lo, mbcnt_hi, first, narrow_first))
    return std::nullopt;

  const auto publishing = instrumentation::build_v_mov_b32_literal(value, 1u, arch);
  const auto empty = instrumentation::build_v_mov_b32_literal(expected, 0u, arch);
  const auto claim = instrumentation::build_flat_atomic_cmpswap_b32(
      base, value, value, /*return_old_value=*/true, kRdna4ScopeDevice, arch);
  // BANK still names the causal-window slot stored in the pending payload.
  // Address the pending table through a separate (window, owner) index so all
  // waves at one static last-arriver RMW can retain their exact acquire.
  const auto append_pending_address = [&]() {
    if (pending_owner_bank_count > 1u) {
      const auto scale_window = instrumentation::build_v_mul_lo_u32_literal(
          value, expected, pending_owner_bank_count, bank, arch);
      const auto owner_bank = instrumentation::build_v_and_b32_literal(
          expected, pending_owner_bank_count - 1u, *point.moi_owner_vgpr, arch);
      const auto combine =
          instrumentation::build_v_add_u32(value, vector_source_vgpr(value), expected, arch);
      if (!scale_window || !owner_bank || !combine)
        return false;
      words.insert(words.end(), scale_window->begin(), scale_window->end());
      words.insert(words.end(), owner_bank->begin(), owner_bank->end());
      words.insert(words.end(), combine->begin(), combine->end());
    } else {
      words.push_back(build_v_mov_b32_e32(value, vector_source_vgpr(bank), arch));
    }
    return append_sampled_indexed_address(words, first_pending_address,
                                          sizeof(ConSanMoiSampledPendingAcquireSlot), value, base,
                                          expected, arch);
  };
  if (!append_pending_address() || !sequence.emit_all(publishing, empty, claim) ||
      !append_moi_global_atomic_wait(words, arch)) {
    return std::nullopt;
  }
  const auto claimed =
      instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), value, arch);
  const auto narrow_claimed =
      instrumentation::build_s_and_saveexec_b64(temporary_exec, kRdna4VccLo, arch);
  if (!sequence.emit_all(claimed, narrow_claimed))
    return std::nullopt;
  if (!sequence.emit_branch(contention_label, InstructionSequence::BranchKind::SccZero))
    return std::nullopt;

  if (release_selected_slot) {
    // RESERVED carries release_slot + 1 (zero means no release attachment).
    // BANK is the absolute acquire slot, so the wrapped unsigned delta also
    // handles a release window whose static base precedes the acquire window.
    const uint32_t release_delta = *release_selected_slot - selected_slot + 1u;
    const auto release_route =
        instrumentation::build_v_add_u32_literal(expected, value, release_delta, bank, arch);
    if (!release_route)
      return std::nullopt;
    words.insert(words.end(), release_route->begin(), release_route->end());
  }

  if (!record.store_vgpr(offsetof(ConSanMoiSampledPendingAcquireSlot, selected_slot), bank) ||
      !record.store_literal(offsetof(ConSanMoiSampledPendingAcquireSlot, generation),
                            static_cast<uint32_t>(bound_resources.moi_report_generation)) ||
      !record.store_literal(offsetof(ConSanMoiSampledPendingAcquireSlot, generation) + 4u,
                            static_cast<uint32_t>(bound_resources.moi_report_generation >> 32u)) ||
      !append_store_moi_report_dispatch_id_pair(
          record, point, bound_resources, offsetof(ConSanMoiSampledPendingAcquireSlot, dispatch_id),
          arch, ConSanMoiLiteralDispatchIdPolicy::AnyArchitecture) ||
      !record.store_workgroup(offsetof(ConSanMoiSampledPendingAcquireSlot, workgroup_x),
                              workgroup_sources->x) ||
      !record.store_workgroup(offsetof(ConSanMoiSampledPendingAcquireSlot, workgroup_y),
                              workgroup_sources->y) ||
      !record.store_workgroup(offsetof(ConSanMoiSampledPendingAcquireSlot, workgroup_z),
                              workgroup_sources->z) ||
      !record.store_vgpr(offsetof(ConSanMoiSampledPendingAcquireSlot, owner_id),
                         *point.moi_owner_vgpr) ||
      !record.store_vgpr(offsetof(ConSanMoiSampledPendingAcquireSlot, source_epoch),
                         *point.moi_epoch_vgpr) ||
      !(release_selected_slot
            ? record.store_vgpr(offsetof(ConSanMoiSampledPendingAcquireSlot, reserved), expected)
            : record.store_literal(offsetof(ConSanMoiSampledPendingAcquireSlot, reserved), 0u)) ||
      !record.store_vgpr(offsetof(ConSanMoiSampledPendingAcquireSlot, metadata) +
                             offsetof(ConSanMoiSampledSyncMetadataPacked, address),
                         saved_address) ||
      !record.store_vgpr(offsetof(ConSanMoiSampledPendingAcquireSlot, metadata) +
                             offsetof(ConSanMoiSampledSyncMetadataPacked, address) + 4u,
                         static_cast<uint16_t>(saved_address + 1u)) ||
      !record.store_literal(offsetof(ConSanMoiSampledPendingAcquireSlot, metadata) +
                                offsetof(ConSanMoiSampledSyncMetadataPacked, byte_count),
                            4u) ||
      !record.store_vgpr(offsetof(ConSanMoiSampledPendingAcquireSlot, metadata) +
                             offsetof(ConSanMoiSampledSyncMetadataPacked, epoch_before),
                         *point.moi_epoch_vgpr) ||
      !record.store_vgpr(offsetof(ConSanMoiSampledPendingAcquireSlot, metadata) +
                             offsetof(ConSanMoiSampledSyncMetadataPacked, epoch_after),
                         *point.moi_epoch_vgpr))
    return std::nullopt;

  if (is_cas) {
    assert(prelude.cas_compare_vgpr && prelude.cas_result_vgpr);
    const auto cas_failure_label = sequence.make_label();
    const auto descriptor_done_label = sequence.make_label();
    const auto success = instrumentation::build_v_cmp_eq_u32_vcc(
        vector_source_vgpr(*prelude.cas_compare_vgpr), *prelude.cas_result_vgpr, arch);
    if (!success)
      return std::nullopt;
    words.push_back(*success);
    if (!sequence.emit_branch(cas_failure_label, InstructionSequence::BranchKind::VccZero))
      return std::nullopt;
    if (!record.store_literal(offsetof(ConSanMoiSampledPendingAcquireSlot, metadata) +
                                  offsetof(ConSanMoiSampledSyncMetadataPacked, descriptor),
                              *success_descriptor))
      return std::nullopt;
    if (!sequence.emit_branch(descriptor_done_label,
                              InstructionSequence::BranchKind::Unconditional) ||
        !sequence.bind(cas_failure_label))
      return std::nullopt;
    if (!record.store_literal(offsetof(ConSanMoiSampledPendingAcquireSlot, metadata) +
                                  offsetof(ConSanMoiSampledSyncMetadataPacked, descriptor),
                              *failure_descriptor))
      return std::nullopt;
    if (!sequence.bind(descriptor_done_label))
      return std::nullopt;
  } else if (!record.store_literal(offsetof(ConSanMoiSampledPendingAcquireSlot, metadata) +
                                       offsetof(ConSanMoiSampledSyncMetadataPacked, descriptor),
                                   *success_descriptor)) {
    return std::nullopt;
  }

  const auto wait_store = instrumentation::build_s_wait_global_store0(arch);
  const auto ready = instrumentation::build_v_mov_b32_literal(value, 2u, arch);
  const auto expected_publishing = instrumentation::build_v_mov_b32_literal(expected, 1u, arch);
  const auto commit = instrumentation::build_flat_atomic_cmpswap_b32(
      base, value, value, /*return_old_value=*/true, kRdna4ScopeDevice, arch);
  if (!sequence.emit(wait_store) || !append_pending_address() ||
      !sequence.emit_all(ready, expected_publishing, commit) ||
      !append_moi_global_atomic_wait(words, arch)) {
    return std::nullopt;
  }
  const auto committed =
      instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(expected), value, arch);
  const auto narrow_committed =
      instrumentation::build_s_and_saveexec_b64(temporary_exec, kRdna4VccLo, arch);
  if (!sequence.emit_all(committed, narrow_committed))
    return std::nullopt;
  if (!sequence.emit_branch(collision_label, InstructionSequence::BranchKind::SccZero))
    return std::nullopt;
  if (!append_atomic_fetch_add_one_u32(
          words, report_base + offsetof(ConSanMoiReportHeader, sampled_pending_acquire_count),
          value, base, arch))
    return std::nullopt;
  if (!sequence.emit_branch(restore_label, InstructionSequence::BranchKind::Unconditional))
    return std::nullopt;

  const auto append_counter_path = [&](size_t counter_offset) -> bool {
    const auto restore = instrumentation::build_s_mov_b64(kRdna4ExecLo, original_exec, arch);
    if (!sequence.emit_all(restore, active_mask, mbcnt_lo, mbcnt_hi, first, narrow_first))
      return false;
    if (!append_atomic_fetch_add_one_u32(words, report_base + counter_offset, value, base, arch))
      return false;
    return sequence.emit_branch(restore_label, InstructionSequence::BranchKind::Unconditional);
  };
  // The sole successful claim chooses one sampled wave. Losing claims are
  // ordinary bounded contention and never write the pending record. The later
  // read requires the exact winner, so winner-without-read remains fail closed.
  if (!sequence.bind(contention_label) ||
      !append_counter_path(
          offsetof(ConSanMoiReportHeader, sampled_pending_acquire_contention_count)) ||
      !sequence.bind(collision_label) ||
      !append_counter_path(
          offsetof(ConSanMoiReportHeader, sampled_pending_acquire_collision_count)))
    return std::nullopt;

  const auto restore_exec = instrumentation::build_s_mov_b64(kRdna4ExecLo, original_exec, arch);
  if (!sequence.bind(restore_label) || !sequence.emit(restore_exec) ||
      !append_restore_moi_special_state(words, moi_special_state_sgprs(request, point), arch))
    return std::nullopt;
  if (scalar_spill)
    words.insert(words.end(), scalar_spill->restore_words.begin(),
                 scalar_spill->restore_words.end());
  if (spill)
    words.insert(words.end(), spill->restore_words.begin(), spill->restore_words.end());
  if (!append_moi_direct_or_indirect_return(words, cave_text_offset, return_text_offset, request,
                                            point, arch))
    return std::nullopt;

  if (!sequence.resolve_branches(arch))
    return std::nullopt;
  return words;
}
[[nodiscard]] std::optional<std::vector<uint32_t>> build_sampled_atomic_sync_cave_words(
    std::span<const uint8_t> bytes, const MoiAtomicEvidenceSitePlan &candidate,
    const ConSanMoiAtomicAddressPlan &address_plan, const ConSanRequest &request,
    const BoundRuntimeResources &bound_resources, const ConSanMoiOperatingPoint &point,
    uint16_t scratch_vgpr, const VgprSpillSequence *spill, const SgprSpillSequence *scalar_spill,
    const MoiPrivateEpochLayout *private_layout, rj_code_arch_t arch, uint32_t selected_slot,
    uint32_t bank_count, const ConSanMoiReportBufferLayout &layout, uint64_t cave_text_offset,
    uint64_t return_text_offset, std::vector<std::string> &errors,
    uint32_t *guest_instruction_offset, std::span<const uint32_t> trailing_guest_words,
    bool preserve_guest_at_anchor, std::span<const uint32_t> leading_guest_words) {
  if (!bound_resources.moi_report_buffer_address || !point.moi_owner_vgpr ||
      !point.moi_epoch_vgpr || !point.moi_exec_save_sgpr ||
      static_cast<uint32_t>(scratch_vgpr) + sampled_atomic_scratch_count() > kMaxVgprs ||
      bank_count == 0u || !std::has_single_bit(bank_count) ||
      selected_slot > layout.sampled_sync_metadata_capacity ||
      bank_count > layout.sampled_sync_metadata_capacity - selected_slot ||
      selected_slot > layout.sampled_causal_window_capacity ||
      bank_count > layout.sampled_causal_window_capacity - selected_slot ||
      selected_slot > layout.sampled_watchpoint_capacity ||
      bank_count > layout.sampled_watchpoint_capacity - selected_slot)
    return std::nullopt;
  if (!address_plan.supported() || !candidate.site.raw_scope || candidate.site.width_bits != 32u)
    return std::nullopt;
  const auto role = sampled_atomic_role(candidate.event_kind, candidate.is_rmw);
  const auto scope = sampled_atomic_scope(*candidate.site.raw_scope);
  if (!role || !scope)
    return std::nullopt;
  if (!candidate.kernel_descriptor_file_offset) {
    errors.emplace_back("ConSan MOI sampled atomic metadata lacks an owning kernel descriptor");
    return std::nullopt;
  }
  const auto workgroup_sources = moi_persistent_or_descriptor_workgroup_sources(
      bytes, *candidate.kernel_descriptor_file_offset, request.moi_engine, point, arch, errors,
      candidate.uses_cluster_workgroup_id);
  if (!workgroup_sources)
    return std::nullopt;
  if (candidate.site.file_offset > bytes.size() ||
      candidate.site.size > bytes.size() - candidate.site.file_offset) {
    errors.emplace_back("ConSan MOI sampled atomic metadata site exceeds ELF bytes");
    return std::nullopt;
  }

  const bool is_cas = candidate.is_rmw && consan_atomic_is_compare_exchange(candidate.site);
  if (is_cas && (!candidate.site.data_vgpr || !candidate.site.dst_vgpr ||
                 !candidate.site.returns_old_value.value_or(false))) {
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
  const auto success_descriptor =
      descriptor_for(!candidate.is_rmw ? ConSanMoiSampledSyncOutcome::NotApplicable
                     : is_cas          ? ConSanMoiSampledSyncOutcome::CasSuccess
                     : candidate.site.returns_old_value.value_or(false)
                         ? ConSanMoiSampledSyncOutcome::RmwReturnsOld
                         : ConSanMoiSampledSyncOutcome::RmwNoReturn);
  const auto failure_descriptor =
      is_cas ? descriptor_for(ConSanMoiSampledSyncOutcome::CasFailure) : success_descriptor;
  if (!success_descriptor || !failure_descriptor)
    return std::nullopt;

  const uint16_t base = scratch_vgpr;
  const uint16_t value = static_cast<uint16_t>(base + SampledAtomicScratchLayout::kValue);
  const uint16_t expected = static_cast<uint16_t>(base + SampledAtomicScratchLayout::kExpected);
  const uint16_t saved_address =
      static_cast<uint16_t>(base + SampledAtomicScratchLayout::kSavedAddress);
  const uint16_t bank = static_cast<uint16_t>(base + SampledAtomicScratchLayout::kBank);
  const uint16_t original_exec = static_cast<uint16_t>(*point.moi_exec_save_sgpr + 6u);
  const uint16_t temporary_exec = *point.moi_exec_save_sgpr;
  const uint64_t report_base = *bound_resources.moi_report_buffer_address;
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
      (release_only || preserve_guest_at_anchor) && !is_cas &&
      !(spill && sampled_atomic_spill_overlaps_guest_operands(*spill, candidate.lowering_form));
  if (preserve_guest_at_anchor && !defer_guest)
    return std::nullopt;

  std::vector<uint32_t> words(leading_guest_words.begin(), leading_guest_words.end());
  SampledAtomicPreludeState prelude;
  if (!append_sampled_atomic_prelude(words, bytes, candidate, address_plan, request, point, base,
                                     spill, scalar_spill, private_layout, arch, saved_address,
                                     defer_guest, prelude, errors, guest_instruction_offset,
                                     trailing_guest_words))
    return std::nullopt;
  if (!append_sampled_window_bank_index(words, point, bound_resources, *workgroup_sources,
                                        bank_count, bank, expected, *point.moi_owner_vgpr, arch)) {
    errors.emplace_back("ConSan MOI sampled atomic metadata failed at bank selection");
    return std::nullopt;
  }
  if (selected_slot != 0u) {
    const auto absolute_slot =
        instrumentation::build_v_add_u32_literal(bank, expected, selected_slot, bank, arch);
    if (!absolute_slot)
      return std::nullopt;
    words.insert(words.end(), absolute_slot->begin(), absolute_slot->end());
  }
  InstructionSequence sequence(words);
  ConSanMoiRecordEmitter record(words, base, value, arch);
  const auto collision_label = sequence.make_label();
  const auto restore_label = sequence.make_label();
  if (!append_save_moi_special_state(words, moi_special_state_sgprs(request, point), arch))
    return std::nullopt;
  const auto save_exec = instrumentation::build_s_mov_b64(original_exec, kRdna4ExecLo, arch);
  if (!save_exec)
    return std::nullopt;
  words.push_back(*save_exec);

  if (!append_sampled_indexed_address(words, first_window_address,
                                      sizeof(ConSanMoiSampledCausalWindow), bank, base, expected,
                                      arch))
    return std::nullopt;

  const auto narrow_equal_literal = [&](uint32_t offset, uint32_t literal) -> bool {
    if (!append_load_u32_vgpr_at_offset(words, base, offset, value, arch))
      return false;
    const auto mov = instrumentation::build_v_mov_b32_literal(expected, literal, arch);
    const auto compare =
        instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(expected), value, arch);
    const auto narrow =
        instrumentation::build_s_and_saveexec_b64(temporary_exec, kRdna4VccLo, arch);
    if (!sequence.emit_all(mov, compare, narrow))
      return false;
    return sequence.emit_branch(restore_label, InstructionSequence::BranchKind::SccZero);
  };
  const auto narrow_equal_vgpr = [&](uint32_t offset, uint16_t expected_vgpr) -> bool {
    if (!append_load_u32_vgpr_at_offset(words, base, offset, value, arch))
      return false;
    const auto compare =
        instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(expected_vgpr), value, arch);
    const auto narrow =
        instrumentation::build_s_and_saveexec_b64(temporary_exec, kRdna4VccLo, arch);
    if (!sequence.emit_all(compare, narrow))
      return false;
    return sequence.emit_branch(restore_label, InstructionSequence::BranchKind::SccZero);
  };
  const auto narrow_equal_dispatch_id = [&](uint32_t offset, bool high_word) -> bool {
    if (!append_load_u32_vgpr_at_offset(words, base, offset, value, arch) ||
        !append_compare_moi_report_dispatch_id_word(
            words, point, bound_resources, value, expected, high_word, arch,
            ConSanMoiLiteralDispatchIdPolicy::AnyArchitecture)) {
      return false;
    }
    const auto narrow =
        instrumentation::build_s_and_saveexec_b64(temporary_exec, kRdna4VccLo, arch);
    if (!sequence.emit(narrow))
      return false;
    return sequence.emit_branch(restore_label, InstructionSequence::BranchKind::SccZero);
  };
  const auto narrow_equal_workgroup = [&](uint32_t offset,
                                          const ConSanMoiWorkgroupSource &source) -> bool {
    if (!source.has_value())
      return narrow_equal_literal(offset, 0u);
    if (!consan_detail::append_workgroup_source_value(words, source, expected, arch))
      return false;
    return narrow_equal_vgpr(offset, expected);
  };
  if (!narrow_equal_literal(offsetof(ConSanMoiSampledCausalWindow, generation),
                            static_cast<uint32_t>(bound_resources.moi_report_generation)) ||
      !narrow_equal_literal(offsetof(ConSanMoiSampledCausalWindow, generation) + 4u,
                            static_cast<uint32_t>(bound_resources.moi_report_generation >> 32u)) ||
      !narrow_equal_dispatch_id(offsetof(ConSanMoiSampledCausalWindow, dispatch_id),
                                /*high_word=*/false) ||
      !narrow_equal_dispatch_id(offsetof(ConSanMoiSampledCausalWindow, dispatch_id) + 4u,
                                /*high_word=*/true) ||
      !narrow_equal_workgroup(offsetof(ConSanMoiSampledCausalWindow, workgroup_x),
                              workgroup_sources->x) ||
      !narrow_equal_workgroup(offsetof(ConSanMoiSampledCausalWindow, workgroup_y),
                              workgroup_sources->y) ||
      !narrow_equal_workgroup(offsetof(ConSanMoiSampledCausalWindow, workgroup_z),
                              workgroup_sources->z) ||
      !narrow_equal_vgpr(offsetof(ConSanMoiSampledCausalWindow, epoch), *point.moi_epoch_vgpr) ||
      !narrow_equal_vgpr(offsetof(ConSanMoiSampledCausalWindow, first_entry), bank) ||
      !narrow_equal_literal(offsetof(ConSanMoiSampledCausalWindow, entry_count), 1u) ||
      !narrow_equal_literal(offsetof(ConSanMoiSampledCausalWindow, publication_state),
                            static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Ready)) ||
      !narrow_equal_workgroup(offsetof(ConSanMoiSampledCausalWindow, cluster_workgroup_id),
                              workgroup_sources->cluster_workgroup_id))
    return std::nullopt;

  if (!append_sampled_indexed_address(words, first_watchpoint_address, sizeof(uint64_t), bank, base,
                                      expected, arch))
    return std::nullopt;
  const auto narrow_current_vcc = [&]() -> bool {
    const auto narrow =
        instrumentation::build_s_and_saveexec_b64(temporary_exec, kRdna4VccLo, arch);
    if (!sequence.emit(narrow))
      return false;
    return sequence.emit_branch(restore_label, InstructionSequence::BranchKind::SccZero);
  };
  const auto narrow_masked_low = [&](uint32_t mask, uint32_t wanted) -> bool {
    if (!append_load_u32_vgpr_at_offset(words, base, 0u, value, arch))
      return false;
    const auto masked = instrumentation::build_v_and_b32_literal(value, mask, value, arch);
    const auto expected_value = instrumentation::build_v_mov_b32_literal(expected, wanted, arch);
    if (!sequence.emit_all(masked, expected_value))
      return false;
    const auto equal =
        instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(expected), value, arch);
    if (!sequence.emit(equal))
      return false;
    return narrow_current_vcc();
  };
  if (!narrow_masked_low(static_cast<uint32_t>(consan_moi_sampled_watchpoint::valid_mask),
                         static_cast<uint32_t>(consan_moi_sampled_watchpoint::valid_mask)) ||
      !narrow_masked_low(static_cast<uint32_t>(consan_moi_sampled_watchpoint::consumed_mask), 0u) ||
      !append_load_u32_vgpr_at_offset(words, base, 0u, value, arch))
    return std::nullopt;
  const auto kind_shift = instrumentation::build_v_lshrrev_b32(
      value, scalar_positive_inline_u32(consan_moi_sampled_watchpoint::access_kind_shift), value,
      arch);
  const auto kind_mask = instrumentation::build_v_and_b32_literal(
      value, (1u << consan_moi_sampled_watchpoint::access_kind_bits) - 1u, value, arch);
  if (!sequence.emit_all(kind_shift, kind_mask))
    return std::nullopt;
  const auto kind_nonempty =
      instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0u), value, arch);
  if (!sequence.emit(kind_nonempty))
    return std::nullopt;
  if (!narrow_current_vcc())
    return std::nullopt;
  const auto kind_supported =
      instrumentation::build_v_cmp_gt_u32_vcc(scalar_positive_inline_u32(3u), value, arch);
  if (!sequence.emit(kind_supported))
    return std::nullopt;
  if (!narrow_current_vcc() || !append_load_u32_vgpr_at_offset(words, base, 0u, value, arch))
    return std::nullopt;
  const auto owner_shift = instrumentation::build_v_lshrrev_b32(
      value, scalar_positive_inline_u32(consan_moi_sampled_watchpoint::owner_shift), value, arch);
  const auto owner_mask = instrumentation::build_v_and_b32_literal(
      value, consan_moi_sampled_watchpoint::max_owner, value, arch);
  const auto owner_equal = instrumentation::build_v_cmp_eq_u32_vcc(
      vector_source_vgpr(*point.moi_owner_vgpr), value, arch);
  const auto narrow_owner =
      instrumentation::build_s_and_saveexec_b64(temporary_exec, kRdna4VccLo, arch);
  if (!sequence.emit_all(owner_shift, owner_mask, owner_equal, narrow_owner))
    return std::nullopt;
  if (!sequence.emit_branch(restore_label, InstructionSequence::BranchKind::SccZero))
    return std::nullopt;

  if (!append_load_u32_vgpr_at_offset(words, base, 0u, value, arch))
    return std::nullopt;
  const auto epoch_shift = instrumentation::build_v_lshrrev_b32(
      value, scalar_positive_inline_u32(consan_moi_sampled_watchpoint::epoch_shift), value, arch);
  const auto epoch_mask = instrumentation::build_v_and_b32_literal(
      value, consan_moi_sampled_watchpoint::max_epoch, value, arch);
  const auto epoch_equal = instrumentation::build_v_cmp_eq_u32_vcc(
      vector_source_vgpr(*point.moi_epoch_vgpr), value, arch);
  if (!sequence.emit_all(epoch_shift, epoch_mask, epoch_equal))
    return std::nullopt;
  if (!narrow_current_vcc() || !append_load_u32_vgpr_at_offset(words, base, 0u, value, arch) ||
      !append_load_u32_vgpr_at_offset(words, base, sizeof(uint32_t), expected, arch))
    return std::nullopt;
  const auto generation_low = instrumentation::build_v_lshrrev_b32(
      value, scalar_positive_inline_u32(consan_moi_sampled_watchpoint::generation_shift), value,
      arch);
  const auto generation_high = instrumentation::build_v_lshlrev_b32(
      expected, scalar_positive_inline_u32(32u - consan_moi_sampled_watchpoint::generation_shift),
      expected, arch);
  const auto combine_generation =
      instrumentation::build_v_add_u32(value, vector_source_vgpr(expected), value, arch);
  const auto mask_generation = instrumentation::build_v_and_b32_literal(
      value, consan_moi_sampled_watchpoint::max_generation, value, arch);
  const auto expected_generation = instrumentation::build_v_mov_b32_literal(
      expected,
      static_cast<uint32_t>(bound_resources.moi_report_generation) &
          consan_moi_sampled_watchpoint::max_generation,
      arch);
  if (!sequence.emit_all(generation_low, generation_high, combine_generation, mask_generation,
                         expected_generation))
    return std::nullopt;
  const auto generation_equal =
      instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(expected), value, arch);
  if (!sequence.emit(generation_equal))
    return std::nullopt;
  if (!narrow_current_vcc())
    return std::nullopt;

  const auto matching_exec = instrumentation::build_s_mov_b64(temporary_exec, kRdna4ExecLo, arch);
  const auto mbcnt_lo = instrumentation::build_v_mbcnt_lo_u32_b32(
      value, temporary_exec, scalar_positive_inline_u32(0), arch);
  const auto mbcnt_hi = instrumentation::build_v_mbcnt_hi_u32_b32(
      value, static_cast<uint16_t>(temporary_exec + 1u), vector_source_vgpr(value), arch);
  const auto first =
      instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), value, arch);
  const auto narrow_first =
      instrumentation::build_s_and_saveexec_b64(temporary_exec, kRdna4VccLo, arch);
  if (!sequence.emit_all(matching_exec, mbcnt_lo, mbcnt_hi, first, narrow_first))
    return std::nullopt;

  const auto append_publication = [&](uint32_t descriptor) -> bool {
    const auto sentinel = instrumentation::build_v_mov_b32_literal(
        value, kConSanMoiSampledSyncPublishingDescriptor, arch);
    const auto zero = instrumentation::build_v_mov_b32_literal(expected, 0, arch);
    const auto claim = instrumentation::build_flat_atomic_cmpswap_b32(base, value, value, true,
                                                                      kRdna4ScopeDevice, arch);
    if (!append_sampled_indexed_address(
            words,
            first_metadata_address + offsetof(ConSanMoiSampledSyncMetadataPacked, descriptor),
            sizeof(ConSanMoiSampledSyncMetadataPacked), bank, base, expected, arch) ||
        !sequence.emit_all(sentinel, zero, claim) || !append_moi_global_atomic_wait(words, arch)) {
      return false;
    }
    const auto claimed =
        instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), value, arch);
    const auto narrow_claimed =
        instrumentation::build_s_and_saveexec_b64(temporary_exec, kRdna4VccLo, arch);
    if (!sequence.emit_all(claimed, narrow_claimed))
      return false;
    if (!sequence.emit_branch(collision_label, InstructionSequence::BranchKind::SccZero))
      return false;
    if (!append_sampled_indexed_address(words, first_metadata_address,
                                        sizeof(ConSanMoiSampledSyncMetadataPacked), bank, base,
                                        expected, arch) ||
        !record.store_vgpr(offsetof(ConSanMoiSampledSyncMetadataPacked, address), saved_address) ||
        !record.store_vgpr(offsetof(ConSanMoiSampledSyncMetadataPacked, address) + 4u,
                           static_cast<uint16_t>(saved_address + 1u)) ||
        !record.store_literal(offsetof(ConSanMoiSampledSyncMetadataPacked, byte_count), 4u) ||
        !record.store_vgpr(offsetof(ConSanMoiSampledSyncMetadataPacked, epoch_before),
                           *point.moi_epoch_vgpr) ||
        !record.store_vgpr(offsetof(ConSanMoiSampledSyncMetadataPacked, epoch_after),
                           *point.moi_epoch_vgpr))
      return false;
    const auto wait = instrumentation::build_s_wait_global_store0(arch);
    const auto final_value = instrumentation::build_v_mov_b32_literal(value, descriptor, arch);
    const auto final_expected = instrumentation::build_v_mov_b32_literal(
        expected, kConSanMoiSampledSyncPublishingDescriptor, arch);
    const auto commit = instrumentation::build_flat_atomic_cmpswap_b32(base, value, value, true,
                                                                       kRdna4ScopeDevice, arch);
    if (!sequence.emit(wait) ||
        !append_sampled_indexed_address(
            words,
            first_metadata_address + offsetof(ConSanMoiSampledSyncMetadataPacked, descriptor),
            sizeof(ConSanMoiSampledSyncMetadataPacked), bank, base, expected, arch) ||
        !sequence.emit_all(final_value, final_expected, commit) ||
        !append_moi_global_atomic_wait(words, arch)) {
      return false;
    }
    const auto committed =
        instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(expected), value, arch);
    const auto narrow_committed =
        instrumentation::build_s_and_saveexec_b64(temporary_exec, kRdna4VccLo, arch);
    if (!sequence.emit_all(committed, narrow_committed))
      return false;
    if (!sequence.emit_branch(collision_label, InstructionSequence::BranchKind::SccZero))
      return false;
    if (!append_atomic_fetch_add_one_u32(
            words, report_base + offsetof(ConSanMoiReportHeader, sampled_sync_metadata_count),
            value, base, arch))
      return false;
    return sequence.emit_branch(restore_label, InstructionSequence::BranchKind::Unconditional);
  };

  if (is_cas) {
    assert(prelude.cas_compare_vgpr && prelude.cas_result_vgpr);
    const auto cas_failure_label = sequence.make_label();
    const auto success = instrumentation::build_v_cmp_eq_u32_vcc(
        vector_source_vgpr(*prelude.cas_compare_vgpr), *prelude.cas_result_vgpr, arch);
    if (!success)
      return std::nullopt;
    words.push_back(*success);
    if (!sequence.emit_branch(cas_failure_label, InstructionSequence::BranchKind::VccZero) ||
        !append_publication(*success_descriptor) || !sequence.bind(cas_failure_label))
      return std::nullopt;
  }
  if (!append_publication(is_cas ? *failure_descriptor : *success_descriptor))
    return std::nullopt;

  // A failed claim/commit reaches here after its cumulative EXEC intersection
  // became empty. Reconstitute the guest mask and select exactly one active
  // lane so the collision counter is neither lost nor multiplied by wave size.
  const auto restore_collision_exec =
      instrumentation::build_s_mov_b64(kRdna4ExecLo, original_exec, arch);
  const auto collision_mask = instrumentation::build_s_mov_b64(temporary_exec, original_exec, arch);
  const auto collision_mbcnt_lo = instrumentation::build_v_mbcnt_lo_u32_b32(
      value, temporary_exec, scalar_positive_inline_u32(0), arch);
  const auto collision_mbcnt_hi = instrumentation::build_v_mbcnt_hi_u32_b32(
      value, static_cast<uint16_t>(temporary_exec + 1u), vector_source_vgpr(value), arch);
  const auto collision_first =
      instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), value, arch);
  const auto narrow_collision =
      instrumentation::build_s_and_saveexec_b64(temporary_exec, kRdna4VccLo, arch);
  if (!sequence.bind(collision_label) ||
      !sequence.emit_all(restore_collision_exec, collision_mask, collision_mbcnt_lo,
                         collision_mbcnt_hi, collision_first, narrow_collision))
    return std::nullopt;
  if (!append_atomic_fetch_add_one_u32(
          words, report_base + offsetof(ConSanMoiReportHeader, sampled_dropped_window_count), value,
          base, arch))
    return std::nullopt;
  const auto restore_exec = instrumentation::build_s_mov_b64(kRdna4ExecLo, original_exec, arch);
  if (!sequence.bind(restore_label) || !sequence.emit(restore_exec) ||
      !append_restore_moi_special_state(words, moi_special_state_sgprs(request, point), arch))
    return std::nullopt;
  if (scalar_spill)
    words.insert(words.end(), scalar_spill->restore_words.begin(),
                 scalar_spill->restore_words.end());
  if (spill)
    words.insert(words.end(), spill->restore_words.begin(), spill->restore_words.end());
  if (preserve_guest_at_anchor) {
    if (!append_moi_direct_or_indirect_return(words, cave_text_offset, return_text_offset, request,
                                              point, arch)) {
      return std::nullopt;
    }
  } else if (defer_guest) {
    std::vector<uint32_t> deferred_guest_words;
    if (!append_sampled_atomic_guest(deferred_guest_words, bytes, candidate, arch,
                                     /*guest_instruction_offset=*/nullptr)) {
      return std::nullopt;
    }
    deferred_guest_words.insert(deferred_guest_words.end(), trailing_guest_words.begin(),
                                trailing_guest_words.end());
    const uint64_t branch_text_offset =
        cave_text_offset + (words.size() + deferred_guest_words.size()) * sizeof(uint32_t);
    if (const auto direct = compute_sopp_branch_simm16(branch_text_offset, return_text_offset)) {
      if (guest_instruction_offset)
        *guest_instruction_offset = static_cast<uint32_t>(words.size() * sizeof(uint32_t));
      words.insert(words.end(), deferred_guest_words.begin(), deferred_guest_words.end());
      words.push_back(build_s_branch(*direct, arch));
    } else {
      const auto jump_sgprs = moi_indirect_jump_sgprs(request, point);
      if (!jump_sgprs || !append_moi_prepare_scc_preserving_indirect_jump(
                             words, cave_text_offset, return_text_offset, jump_sgprs->pc_sgpr,
                             jump_sgprs->scc_save_sgpr, arch)) {
        return std::nullopt;
      }
      if (guest_instruction_offset)
        *guest_instruction_offset = static_cast<uint32_t>(words.size() * sizeof(uint32_t));
      words.insert(words.end(), deferred_guest_words.begin(), deferred_guest_words.end());
      words.push_back(build_s_setpc_b64(jump_sgprs->pc_sgpr, arch));
    }
  } else if (!append_moi_direct_or_indirect_return(words, cave_text_offset, return_text_offset,
                                                   request, point, arch)) {
    return std::nullopt;
  }

  if (!sequence.resolve_branches(arch))
    return std::nullopt;
  return words;
}

} // namespace rocjitsu::consan_moi_impl

// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_atomic_emission.h"

#include "rocjitsu/code/patch/consan/consan_probe_lowering.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/consan/consan_atomic_classifier.h"
#include "rocjitsu/code/patch/consan/consan_capability_contract.h"
#include "rocjitsu/code/patch/consan/consan_device_primitives.h"
#include "rocjitsu/code/patch/consan/consan_memory_emission.h"
#include "rocjitsu/code/patch/consan/consan_native_abi.h"
#include "rocjitsu/code/patch/consan/consan_relocation.h"
#include "rocjitsu/code/patch/consan/consan_report_emission.h"
#include "rocjitsu/code/patch/consan/consan_window_emission.h"
#include "rocjitsu/code/patch/instruction_sequence.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"

#include "rocjitsu/code/patch/consan/targets/rdna4/consan_atomic_observation.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstring>
#include <limits>

namespace rocjitsu::consan::detail {

using detail::append_atomic_fetch_add_one_u32;
using detail::append_load_u32_vgpr_at_offset;
using detail::append_select_first_lane_in_exec_mask;
using detail::append_store_u32_vgpr_at_offset;
using detail::range_overlaps;
using detail::RecordEmitter;

AtomicSemanticsResult atomic_semantics_for_source(const AtomicEvidenceSourceView &source) {
  using Reason = AtomicSemanticsReason;
  const auto reject = [](Reason reason) {
    return AtomicSemanticsResult{.semantics = std::nullopt, .reason = reason};
  };
  const SyncSequence &sequence = *source.sequence;
  const AtomicSite &site = source.site;
  if (!sync_confidence_meets(sequence.confidence, SemanticConfidence::Conservative) ||
      !sync_confidence_meets(sequence.memory_role_confidence, SemanticConfidence::Conservative)) {
    return reject(Reason::UnqualifiedSharedSyncSequence);
  }

  detail::AtomicSemantics semantics;
  switch (sequence.memory_role) {
  case SyncMemoryRole::Release:
    semantics.role = SyncRole::RmwRelease;
    break;
  case SyncMemoryRole::Acquire:
    semantics.role = SyncRole::RmwAcquire;
    break;
  case SyncMemoryRole::AcquireRelease:
    semantics.role = SyncRole::RmwAcquireRelease;
    break;
  case SyncMemoryRole::Unknown:
  case SyncMemoryRole::None:
  case SyncMemoryRole::SequentiallyConsistent:
    return reject(Reason::UnsupportedQualifiedMemoryRole);
  }
  if (!sequence.scope)
    return reject(Reason::MissingQualifiedScope);
  const auto scope = sync_scope(*sequence.scope);
  if (!scope)
    return reject(Reason::UnsupportedQualifiedScope);
  semantics.scope = *scope;
  if (site.width_bits == 0 || site.width_bits % 8u != 0 ||
      site.width_bits / 8u > std::numeric_limits<uint32_t>::max()) {
    return reject(Reason::UnsupportedQualifiedByteRange);
  }
  semantics.byte_count = site.width_bits / 8u;
  switch (sequence.rmw_outcome) {
  case SyncRmwOutcome::NoReturn:
    semantics.outcome = SyncOutcome::RmwNoReturn;
    break;
  case SyncRmwOutcome::ReturnsOldValue:
    semantics.outcome = SyncOutcome::RmwReturnsOld;
    break;
  case SyncRmwOutcome::CompareExchange:
    if (!site.returns_old_value.value_or(false) || !site.data_vgpr || !site.destination_vgpr)
      return reject(Reason::CompareExchangeDynamicOutcomeUnavailable);
    semantics.outcome = SyncOutcome::CasSuccess;
    break;
  case SyncRmwOutcome::NotApplicable:
  case SyncRmwOutcome::Unknown:
    return reject(Reason::UnsupportedQualifiedRmwOutcome);
  }

  const SyncEncodeResult encoded = encode_sync_metadata({
      .address = 1,
      .byte_count = semantics.byte_count,
      .kind = SyncMetadataKind::Atomic,
      .role = semantics.role,
      .scope = semantics.scope,
      .outcome = semantics.outcome,
  });
  if (encoded.classification != SyncClassification::Valid)
    return reject(Reason::SyncAbiRejectedQualifiedSequence);
  semantics.descriptor = encoded.packed.descriptor;

  if (sequence.rmw_outcome == SyncRmwOutcome::CompareExchange) {
    const SyncEncodeResult failure = encode_sync_metadata({
        .address = 1,
        .byte_count = semantics.byte_count,
        .kind = SyncMetadataKind::Atomic,
        .role = semantics.role,
        .scope = semantics.scope,
        .outcome = SyncOutcome::CasFailure,
    });
    if (failure.classification != SyncClassification::Valid)
      return reject(Reason::SyncAbiRejectedCasFailure);
    semantics.cas_failure_descriptor = failure.packed.descriptor;
  }
  return {.semantics = semantics, .reason = Reason::None};
}

std::string_view atomic_semantics_reason_name(AtomicSemanticsReason reason) {
  using Reason = AtomicSemanticsReason;
  constexpr auto vocabulary = make_enum_vocabulary(
      "invalid-sampled-atomic-semantics-reason", enum_entry(Reason::None, ""),
      enum_entry(Reason::UnqualifiedSharedSyncSequence, "unqualified-shared-sync-sequence"),
      enum_entry(Reason::UnsupportedQualifiedMemoryRole, "unsupported-qualified-memory-role"),
      enum_entry(Reason::MissingQualifiedScope, "missing-qualified-scope"),
      enum_entry(Reason::UnsupportedQualifiedScope, "unsupported-qualified-scope"),
      enum_entry(Reason::UnsupportedQualifiedByteRange, "unsupported-qualified-byte-range"),
      enum_entry(Reason::CompareExchangeDynamicOutcomeUnavailable,
                 "compare-exchange-dynamic-outcome-unavailable"),
      enum_entry(Reason::UnsupportedQualifiedRmwOutcome, "unsupported-qualified-rmw-outcome"),
      enum_entry(Reason::SyncAbiRejectedQualifiedSequence,
                 "sampled-sync-abi-rejected-qualified-sequence"),
      enum_entry(Reason::SyncAbiRejectedCasFailure, "sampled-sync-abi-rejected-cas-failure"));
  return vocabulary.name(reason);
}

std::optional<SyncRole> atomic_role(AtomicEventKind kind, bool is_rmw) {
  switch (kind) {
  case AtomicEventKind::Release:
    return is_rmw ? SyncRole::RmwRelease : SyncRole::Release;
  case AtomicEventKind::Acquire:
    return is_rmw ? SyncRole::RmwAcquire : SyncRole::Acquire;
  case AtomicEventKind::AcquireRelease:
    return is_rmw ? std::optional(SyncRole::RmwAcquireRelease) : std::nullopt;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<uint32_t> atomic_byte_count(const AtomicEvidenceSourceView &source) {
  const uint32_t width_bits = source.site.width_bits;
  if (width_bits == 0u || width_bits % 8u != 0u)
    return std::nullopt;
  // The CAS outcome path snapshots one compare and one result dword. Other
  // synchronization sites only need their exact address range and may be wide.
  if (atomic_is_compare_exchange(source.site) && width_bits != 32u)
    return std::nullopt;
  return width_bits / 8u;
}

[[nodiscard]] bool atomic_guest_preserves_address(const AtomicLoweringForm &form) {
  if (!form.destination_vgpr || form.destination_register_count == 0u)
    return true;
  return !range_overlaps(form.address_vgpr, form.address_vgpr_count, *form.destination_vgpr,
                         form.destination_register_count);
}

[[nodiscard]] bool atomic_spill_overlaps_guest_operands(const VgprSpillSequence &spill,
                                                        const AtomicLoweringForm &form) {
  const auto overlaps = [&](std::optional<uint16_t> base, uint16_t count) {
    return base && count != 0u && range_overlaps(spill.vgpr_base, spill.vgpr_count, *base, count);
  };
  return range_overlaps(spill.vgpr_base, spill.vgpr_count, form.address_vgpr,
                        form.address_vgpr_count) ||
         overlaps(form.data_vgpr, form.data_register_count) ||
         overlaps(form.destination_vgpr, form.destination_register_count);
}

[[nodiscard]] bool append_atomic_guest(std::vector<uint32_t> &words, std::span<const uint8_t> bytes,
                                       const AtomicSite &site, bool is_rmw, rj_code_arch_t arch,
                                       uint32_t *guest_instruction_offset,
                                       std::span<const uint32_t> leading_guest_words,
                                       std::span<const uint32_t> trailing_guest_words,
                                       uint32_t *emitted_guest_size = nullptr,
                                       std::optional<uint16_t> observation_vgpr = std::nullopt) {
  const size_t guest_begin = words.size();
  if (guest_instruction_offset)
    *guest_instruction_offset = static_cast<uint32_t>(words.size() * sizeof(uint32_t));
  words.insert(words.end(), leading_guest_words.begin(), leading_guest_words.end());
  if (observation_vgpr) {
    if (arch != ROCJITSU_CODE_ARCH_RDNA4)
      return false;
    const auto rewritten = build_rdna4_atomic_observation(
        bytes.subspan(site.file_offset, site.size), *observation_vgpr);
    if (!rewritten)
      return false;
    words.insert(words.end(), rewritten->begin(), rewritten->end());
  } else {
    for (uint64_t offset = 0; offset < site.size; offset += sizeof(uint32_t)) {
      uint32_t word = 0;
      std::memcpy(&word, bytes.data() + site.file_offset + offset, sizeof(word));
      words.push_back(word);
    }
  }
  // A non-RMW candidate carries its compiler-emitted wait/cache suffix. Do
  // not splice a generic atomic wait into that target-native sequence.
  if (is_rmw && !append_flat_load_wait(words, arch))
    return false;
  words.insert(words.end(), trailing_guest_words.begin(), trailing_guest_words.end());
  if (emitted_guest_size)
    *emitted_guest_size = static_cast<uint32_t>((words.size() - guest_begin) * sizeof(uint32_t));
  return true;
}

struct AtomicPreludeState {
  std::optional<uint16_t> cas_compare_vgpr;
  std::optional<uint16_t> cas_result_vgpr;
};

// Spill builders always produce complete metadata. Keep this check defensive
// for internal callers that construct a sequence directly.
[[nodiscard]] bool validate_atomic_spill_metadata(const VgprSpillSequence &spill,
                                                  std::vector<std::string> &errors) {
  if (spill.has_complete_slot_metadata())
    return true;
  errors.emplace_back("ConSan atomic spill has incomplete slot metadata");
  return false;
}

/// @pre @p spill has complete slot metadata.
[[nodiscard]] bool append_atomic_snapshot_word(std::vector<uint32_t> &words,
                                               const VgprSpillSequence &spill, uint16_t destination,
                                               uint16_t source, rj_code_arch_t arch,
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

[[nodiscard]] bool append_atomic_snapshot_wait(std::vector<uint32_t> &words,
                                               bool loaded_private_state, rj_code_arch_t arch) {
  if (!loaded_private_state)
    return true;
  return InstructionSequence(words).emit(instrumentation::build_s_wait_private_load0(arch));
}

[[nodiscard]] bool append_atomic_cas_snapshot(std::vector<uint32_t> &words,
                                              const VgprSpillSequence &spill, uint16_t compare_vgpr,
                                              uint16_t result_vgpr, uint16_t saved_compare,
                                              uint16_t saved_result, rj_code_arch_t arch,
                                              std::vector<std::string> &errors) {
  if (!validate_atomic_spill_metadata(spill, errors))
    return false;
  bool loaded_private_state = false;
  if (!append_atomic_snapshot_word(words, spill, saved_compare, compare_vgpr, arch,
                                   loaded_private_state) ||
      !append_atomic_snapshot_word(words, spill, saved_result, result_vgpr, arch,
                                   loaded_private_state)) {
    errors.emplace_back("ConSan atomic could not preserve CAS outcome evidence");
    return false;
  }
  if (!append_atomic_snapshot_wait(words, loaded_private_state, arch)) {
    errors.emplace_back("ConSan atomic could not wait for CAS outcome evidence");
    return false;
  }
  return true;
}

[[nodiscard]] bool append_atomic_address_snapshot(std::vector<uint32_t> &words,
                                                  const VgprSpillSequence &spill,
                                                  uint16_t source_address, uint16_t saved_address,
                                                  rj_code_arch_t arch,
                                                  std::vector<std::string> &errors) {
  if (!validate_atomic_spill_metadata(spill, errors))
    return false;
  bool loaded_private_state = false;
  if (!append_atomic_snapshot_word(words, spill, saved_address, source_address, arch,
                                   loaded_private_state) ||
      !append_atomic_snapshot_word(words, spill, static_cast<uint16_t>(saved_address + 1u),
                                   static_cast<uint16_t>(source_address + 1u), arch,
                                   loaded_private_state)) {
    errors.emplace_back("ConSan atomic could not preserve its guest address");
    return false;
  }
  if (!append_atomic_snapshot_wait(words, loaded_private_state, arch)) {
    errors.emplace_back("ConSan atomic could not wait for its guest address");
    return false;
  }
  return true;
}

[[nodiscard]] bool append_atomic_prelude(
    std::vector<uint32_t> &words, std::span<const uint8_t> bytes,
    const AtomicEvidenceSourceView &source, const AtomicLoweringForm &lowering_form,
    uint64_t owner_descriptor_file_offset, const AtomicAddressPlan &address_plan,
    const SyncEmissionPlan &plan, const VgprSpillSequence *spill,
    const SgprSpillSequence *scalar_spill, const PrivateStateLayout *private_layout,
    rj_code_arch_t arch, uint16_t saved_address, bool defer_guest, AtomicPreludeState &state,
    std::vector<std::string> &errors, uint32_t *guest_instruction_offset,
    std::span<const uint32_t> leading_guest_words, std::span<const uint32_t> trailing_guest_words,
    uint32_t *emitted_guest_size, std::optional<uint16_t> observation_vgpr = std::nullopt) {
  const AtomicSite &site = source.site;
  const bool is_rmw = source.is_rmw();
  const bool is_cas = atomic_is_compare_exchange(site);
  if (is_cas) {
    assert(site.data_vgpr && site.destination_vgpr);
    state.cas_compare_vgpr = static_cast<uint16_t>(*site.data_vgpr + 1u);
    state.cas_result_vgpr = *site.destination_vgpr;
  }
  const bool guest_first = spill && atomic_spill_overlaps_guest_operands(*spill, lowering_form);
  if (observation_vgpr && guest_first) {
    errors.emplace_back("ConSan non-return observation needs pre-guest scratch preservation");
    return false;
  }
  assert(!defer_guest || !guest_first);
  const bool guest_preserves_address = atomic_guest_preserves_address(lowering_form);
  assert(!guest_first || guest_preserves_address);
  if (guest_first && !guest_preserves_address) {
    errors.emplace_back("ConSan atomic overlap spill cannot preserve guest address");
    return false;
  }
  const bool wraps_banked_polling_loop =
      source.relocates_polling_loop() && plan.polling_loop_entry_vgpr_bank_mode.has_value();
  const uint8_t polling_loop_entry_mode =
      static_cast<uint8_t>(plan.polling_loop_entry_vgpr_bank_mode.value_or(0u));
  const auto append_guest = [&]() {
    if (wraps_banked_polling_loop && !guest_first && polling_loop_entry_mode != 0u) {
      const auto restore_entry = instrumentation::build_s_set_vgpr_msb_transition(
          /*previous_mode=*/0u, polling_loop_entry_mode, arch);
      if (!restore_entry)
        return false;
      words.push_back(*restore_entry);
    }
    return append_atomic_guest(words, bytes, site, is_rmw, arch, guest_instruction_offset,
                               leading_guest_words, trailing_guest_words, emitted_guest_size,
                               observation_vgpr);
  };
  if (guest_first && !append_guest())
    return false;
  // A guest-first polling loop establishes bank zero itself before any spill
  // or instrumentation. Otherwise, select the scratch bank before preserving
  // registers and restore the original entry mode only around the guest span.
  if (wraps_banked_polling_loop && !guest_first && polling_loop_entry_mode != 0u) {
    const auto select_scratch = instrumentation::build_s_set_vgpr_msb_transition(
        polling_loop_entry_mode, /*new_mode=*/0u, arch);
    if (!select_scratch)
      return false;
    words.push_back(*select_scratch);
  }
  if (spill)
    words.insert(words.end(), spill->save_words.begin(), spill->save_words.end());
  if (scalar_spill)
    words.insert(words.end(), scalar_spill->save_words.begin(), scalar_spill->save_words.end());
  if (address_plan.requires_materialization()) {
    const auto special = plan.special_state;
    if (!special)
      return false;
    const auto materialize = build_atomic_address_materialization(
        address_plan, special->vcc_save_sgpr, special->scc_save_sgpr, arch);
    if (!materialize)
      return false;
    words.insert(words.end(), materialize->begin(), materialize->end());
  }
  if (guest_first) {
    if (address_plan.requires_materialization()) {
      assert(address_plan.result_address_vgpr == saved_address);
      if (address_plan.result_address_vgpr != saved_address) {
        errors.emplace_back("ConSan atomic materialized address does not use the scratch tail");
        return false;
      }
    } else if (!append_atomic_address_snapshot(words, *spill, address_plan.result_address_vgpr,
                                               saved_address, arch, errors)) {
      return false;
    }
  }
  if (guest_first && is_cas) {
    // Address materialization/snapshot must precede CAS evidence because the
    // evidence destinations may overlap the live guest address pair.
    const uint16_t saved_compare =
        static_cast<uint16_t>(plan.scratch_vgpr + AtomicScratchLayout::kCasCompare);
    const uint16_t saved_result =
        static_cast<uint16_t>(plan.scratch_vgpr + AtomicScratchLayout::kCasResult);
    if (!append_atomic_cas_snapshot(words, *spill, *state.cas_compare_vgpr, *state.cas_result_vgpr,
                                    saved_compare, saved_result, arch, errors))
      return false;
    state.cas_compare_vgpr = saved_compare;
    state.cas_result_vgpr = saved_result;
  }
  if (private_layout &&
      !append_private_owner_epoch_load(words, bytes, owner_descriptor_file_offset,
                                       plan.automatic_private_epoch, plan.owner_epoch_vgprs,
                                       *private_layout, arch, errors))
    return false;
  if (plan.persistent_sgprs.complete()) {
    if (!detail::validate_scalar_state_temporaries(plan.persistent_sgprs, plan.owner_epoch_vgprs,
                                                   "sampled atomic prelude", errors))
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
      if (!append_guest())
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

bool publication_observation_supported(const AtomicEvidenceSourceView &source) {
  const auto &site = source.site;
  const bool operation =
      site.mnemonic == "global_atomic_add_u32" || site.mnemonic == "flat_atomic_add_u32" ||
      site.mnemonic == "global_atomic_or_b32" || site.mnemonic == "flat_atomic_or_b32";
  return source.sequence && source.is_rmw() && !source.relocates_polling_loop() && operation &&
         site.width_bits == 32 && site.data_vgpr && site.scope && site.returns_old_value &&
         (site.returns_old_value.value()
              ? site.destination_vgpr && *site.data_vgpr != *site.destination_vgpr
              : site.raw_th == 0u);
}

std::optional<std::vector<uint32_t>> build_publication_cave_words(
    std::span<const uint8_t> bytes, const AtomicEvidenceSourceView &source,
    const AtomicLoweringForm &lowering_form, uint64_t owner_descriptor_file_offset,
    const AtomicAddressPlan &address_plan, const SyncEmissionPlan &plan,
    const VgprSpillSequence *spill, const SgprSpillSequence *scalar_spill,
    const PrivateStateLayout *private_layout, rj_code_arch_t arch, const ReportBufferLayout &layout,
    std::vector<std::string> &errors, uint32_t *guest_instruction_offset,
    uint32_t *emitted_guest_size, PublicationCapture capture) {
  const bool opaque = capture == PublicationCapture::OpaqueModification;
  const bool supported = opaque ? source.is_rmw() && source.site.width_bits != 0u &&
                                      source.site.width_bits <= 1024u &&
                                      source.site.width_bits % 8u == 0u
                                : publication_observation_supported(source);
  const auto *target = target_profile(arch);
  if (!supported || !target || !plan.exec_save_sgpr || !plan.owner_epoch_vgprs.owner ||
      !plan.owner_epoch_vgprs.epoch || !layout.publication_event_capacity ||
      layout.publication_event_capacity >
          std::numeric_limits<uint32_t>::max() / sizeof(PublicationRecord) ||
      static_cast<uint32_t>(plan.scratch_vgpr) + atomic_scratch_count() > kMaxVgprs ||
      !address_plan.supported() || source.site.file_offset > bytes.size() ||
      source.site.size > bytes.size() - source.site.file_offset)
    return std::nullopt;
  const auto scope = opaque ? std::optional{SyncScope::None} : sync_scope(*source.site.scope);
  if (!scope)
    return std::nullopt;
  const uint16_t base = plan.scratch_vgpr;
  const uint16_t ticket = base + AtomicScratchLayout::kValue;
  const uint16_t observed = base + AtomicScratchLayout::kCasCompare;
  const uint16_t written = base + AtomicScratchLayout::kCasResult;
  const uint16_t index = base + AtomicScratchLayout::kBank;
  const uint16_t address = base + AtomicScratchLayout::kSavedAddress;
  const uint16_t saved_exec = *plan.exec_save_sgpr + 6u;
  const uint16_t temporary_exec = *plan.exec_save_sgpr;
  const uint64_t report = plan.supercollider_report_buffer_address;
  std::vector<uint32_t> words;
  InstructionSequence sequence(words);
  AtomicPreludeState prelude;
  if (!append_atomic_prelude(words, bytes, source, lowering_form, owner_descriptor_file_offset,
                             address_plan, plan, spill, scalar_spill, private_layout, arch, address,
                             false, prelude, errors, guest_instruction_offset, {}, {},
                             emitted_guest_size,
                             opaque || source.site.returns_old_value.value_or(false)
                                 ? std::nullopt
                                 : std::optional<uint16_t>{observed}))
    return std::nullopt;
  // The guest has completed, and its operands are still available either in
  // registers or in the post-guest spill. Capture both before scratch reuse.
  bool loaded_private = false;
  const auto snapshot = [&](uint16_t destination, uint16_t guest) {
    if (spill)
      return append_atomic_snapshot_word(words, *spill, destination, guest, arch, loaded_private);
    words.push_back(build_v_mov_b32_e32(destination, vector_source_vgpr(guest), arch));
    return true;
  };
  sequence
      .require(opaque || !source.site.returns_old_value.value_or(false) ||
               snapshot(observed, *source.site.destination_vgpr))
      .require(opaque || snapshot(written, *source.site.data_vgpr))
      .require(append_atomic_snapshot_wait(words, loaded_private, arch))
      .require(append_save_special_state(words, plan.special_state, arch))
      .append(instrumentation::build_s_mov_b64(saved_exec, kAmdGpuExecLo, arch));
  if (!opaque && source.site.mnemonic.find("_add_") != std::string::npos)
    sequence.append(
        instrumentation::build_v_add_u32(written, vector_source_vgpr(observed), written, arch));
  else if (!opaque) {
    // a | b = ~(~a & ~b), using the shared target-normalized VALU builders.
    sequence.append(
        instrumentation::build_v_xor_b32(ticket, kScalarInlineNegativeOneOperand, observed, arch),
        instrumentation::build_v_xor_b32(written, kScalarInlineNegativeOneOperand, written, arch),
        instrumentation::build_v_and_b32(written, vector_source_vgpr(ticket), written, arch),
        instrumentation::build_v_xor_b32(written, kScalarInlineNegativeOneOperand, written, arch));
  }
  // Every active guest lane gets its own immutable record. Slot allocation is
  // not modification order; old/new observations reconstruct that on the host.
  sequence
      .require(append_atomic_fetch_add_one_u32(
          words, report + offsetof(ReportHeader, publication_event_count), index, base, arch))
      .append(instrumentation::build_v_cmp_gt_u32_literal_vcc(layout.publication_event_capacity,
                                                              index, arch),
              instrumentation::build_s_and_saveexec_b64(temporary_exec, kAmdGpuVccLo, arch));
  const auto overflow = sequence.make_label();
  const auto finish = sequence.make_label();
  sequence.branch(overflow, InstructionSequence::BranchKind::ExecZero)
      .require(append_publication_ticket(words, report + offsetof(ReportHeader, publication_clock),
                                         ticket, base, arch))
      .require(append_indexed_address(words,
                                      {.table_address = report + layout.publication_events_offset,
                                       .stride_bytes = sizeof(PublicationRecord),
                                       .address_vgpr = base,
                                       .index_vgpr = index},
                                      *target));
  RecordEmitter record(words, base, ticket, arch);
  sequence.require(record.store_vgpr(offsetof(PublicationRecord, sequence), ticket))
      .require(record.store_vgpr(offsetof(PublicationRecord, sequence) + 4u, ticket + 1u))
      .require(opaque ? record.store_literal(offsetof(PublicationRecord, observed), 0u)
                      : record.store_vgpr(offsetof(PublicationRecord, observed), observed))
      .require(opaque ? record.store_literal(offsetof(PublicationRecord, written), 0u)
                      : record.store_vgpr(offsetof(PublicationRecord, written), written))
      .require(record.store_vgpr(offsetof(PublicationRecord, address), address))
      .require(record.store_vgpr(offsetof(PublicationRecord, address) + 4u, address + 1u))
      .require(record.store_literal(offsetof(PublicationRecord, observed) + 4u, 0u))
      .require(record.store_literal(offsetof(PublicationRecord, written) + 4u, 0u))
      .require(record.store_literal(offsetof(PublicationRecord, generation),
                                    static_cast<uint32_t>(plan.report_generation)))
      .require(record.store_literal(offsetof(PublicationRecord, generation) + 4u,
                                    static_cast<uint32_t>(plan.report_generation >> 32)))
      .require(append_store_report_dispatch_id_pair(record, plan.dispatch_id,
                                                    offsetof(PublicationRecord, dispatch_id)))
      .require(record.store_workgroup(offsetof(PublicationRecord, workgroup_x),
                                      plan.workgroup_sources.x,
                                      RecordEmitter::MissingWorkgroupSource::StoreZero))
      .require(record.store_workgroup(offsetof(PublicationRecord, workgroup_y),
                                      plan.workgroup_sources.y,
                                      RecordEmitter::MissingWorkgroupSource::StoreZero))
      .require(record.store_workgroup(offsetof(PublicationRecord, workgroup_z),
                                      plan.workgroup_sources.z,
                                      RecordEmitter::MissingWorkgroupSource::StoreZero))
      .require(record.store_workgroup(offsetof(PublicationRecord, cluster_workgroup_id),
                                      plan.workgroup_sources.cluster_workgroup_id,
                                      RecordEmitter::MissingWorkgroupSource::StoreZero))
      .require(
          record.store_vgpr(offsetof(PublicationRecord, owner_id), *plan.owner_epoch_vgprs.owner))
      .require(record.store_vgpr(offsetof(PublicationRecord, epoch), *plan.owner_epoch_vgprs.epoch))
      .require(record.store_literal(offsetof(PublicationRecord, byte_count),
                                    source.site.width_bits / 8u));
  const auto memory_role = source.sequence->memory_role;
  const bool release = source.sequence->lds_release_wait_text_offset.has_value();
  const bool acquire =
      memory_role == SyncMemoryRole::Acquire || memory_role == SyncMemoryRole::AcquireRelease;
  sequence
      .require(record.store_literal(offsetof(PublicationRecord, roles),
                                    opaque ? 0u
                                           : kPublicationObserved |
                                                 (release ? kPublicationRelease : 0u) |
                                                 (acquire ? kPublicationAcquire : 0u)))
      .require(
          record.store_literal(offsetof(PublicationRecord, scope), static_cast<uint32_t>(*scope)))
      .require(record.store_literal(
          offsetof(PublicationRecord, operation),
          static_cast<uint32_t>(opaque ? PublicationRecordOperation::OpaqueModification
                                       : PublicationRecordOperation::Rmw)))
      .append(instrumentation::build_v_mbcnt_lo_u32_b32(ticket, 0xc1u,
                                                        scalar_positive_inline_u32(0u), arch),
              instrumentation::build_v_mbcnt_hi_u32_b32(ticket, 0xc1u, vector_source_vgpr(ticket),
                                                        arch))
      .require(record.store_vgpr(offsetof(PublicationRecord, lane_id), ticket))
      .append(instrumentation::build_s_wait_flat_store0(arch))
      .require(record.store_literal(offsetof(PublicationRecord, state), kPublicationReady));
  // Overflow is sticky even if an extremely long execution wraps the slot
  // counter. No out-of-range lane may address the bounded record allocation.
  sequence.bind_label(overflow)
      .append(instrumentation::build_s_mov_b64(kAmdGpuExecLo, saved_exec, arch),
              instrumentation::build_v_cmp_gt_u32_literal_vcc(layout.publication_event_capacity,
                                                              index, arch),
              instrumentation::build_s_andn2_b64(kAmdGpuExecLo, saved_exec, kAmdGpuVccLo, arch))
      .branch(finish, InstructionSequence::BranchKind::ExecZero)
      .require(
          record.materialize_address(report + offsetof(ReportHeader, publication_dropped_count)))
      .append(instrumentation::build_v_mov_b32_literal(ticket, 1u, arch),
              instrumentation::build_flat_atomic_or_u32(base, ticket, ticket, false,
                                                        kAmdGpuScopeDevice, arch))
      .require(append_global_atomic_wait(words, arch));
  sequence.bind_label(finish)
      .append(instrumentation::build_s_mov_b64(kAmdGpuExecLo, saved_exec, arch))
      .require(record.materialize_address(report + offsetof(ReportHeader, publication_flags)))
      .append(instrumentation::build_v_mov_b32_literal(ticket, kPublicationTraceEnabled, arch),
              instrumentation::build_flat_atomic_or_u32(base, ticket, ticket, false,
                                                        kAmdGpuScopeDevice, arch))
      .require(append_global_atomic_wait(words, arch))
      .require(append_restore_special_state(words, plan.special_state, arch));
  if (scalar_spill)
    words.insert(words.end(), scalar_spill->restore_words.begin(),
                 scalar_spill->restore_words.end());
  if (spill)
    words.insert(words.end(), spill->restore_words.begin(), spill->restore_words.end());
  if (!sequence.finish(arch))
    return std::nullopt;
  return words;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_pending_acquire_cave_words(
    std::span<const uint8_t> bytes, const AtomicEvidenceSourceView &source,
    const AtomicLoweringForm &lowering_form, uint64_t owner_descriptor_file_offset,
    const AtomicAddressPlan &address_plan, const SyncEmissionPlan &plan,
    const VgprSpillSequence *spill, const SgprSpillSequence *scalar_spill,
    const PrivateStateLayout *private_layout, rj_code_arch_t arch, uint32_t selected_slot,
    uint32_t bank_count, std::optional<uint32_t> release_selected_slot,
    const ReportBufferLayout &layout, std::vector<std::string> &errors,
    uint32_t *guest_instruction_offset, std::span<const uint32_t> leading_guest_words,
    std::span<const uint32_t> trailing_guest_words, uint32_t *emitted_guest_size) {
  const AtomicSite &site = source.site;
  const bool is_rmw = source.is_rmw();
  const bool is_cas = is_rmw && atomic_is_compare_exchange(site);
  const std::optional<uint32_t> byte_count = atomic_byte_count(source);
  const auto event_kind = atomic_event_kind(source.sequence->memory_role);
  const uint32_t pending_owner_bank_count = pending_acquire_owner_bank_count(
      layout.pending_acquire_capacity, layout.causal_window_capacity);
  if (!plan.owner_epoch_vgprs.owner || !plan.owner_epoch_vgprs.epoch || !plan.exec_save_sgpr ||
      static_cast<uint32_t>(plan.scratch_vgpr) + atomic_scratch_count() > kMaxVgprs ||
      bank_count == 0u || !std::has_single_bit(bank_count) || pending_owner_bank_count == 0u ||
      selected_slot > layout.causal_window_capacity ||
      bank_count > layout.causal_window_capacity - selected_slot || !address_plan.supported() ||
      !site.scope || !byte_count)
    return std::nullopt;
  const TargetProfile *target = target_profile(arch);
  if (target == nullptr)
    return std::nullopt;
  const auto role = event_kind ? atomic_role(*event_kind, is_rmw) : std::nullopt;
  const auto scope = sync_scope(*site.scope);
  if (!role || !scope ||
      (static_cast<uint32_t>(*role) & static_cast<uint32_t>(SyncRole::Acquire)) == 0u)
    return std::nullopt;
  if (site.file_offset > bytes.size() || site.size > bytes.size() - site.file_offset)
    return std::nullopt;

  if (is_cas &&
      (!site.data_vgpr || !site.destination_vgpr || !site.returns_old_value.value_or(false)))
    return std::nullopt;
  const auto descriptor_for = [&](SyncOutcome outcome) -> std::optional<uint32_t> {
    const auto encoded = encode_sync_metadata({
        .address = 1,
        .byte_count = *byte_count,
        .kind = SyncMetadataKind::Atomic,
        .role = *role,
        .scope = *scope,
        .outcome = outcome,
    });
    if (encoded.classification != SyncClassification::Valid)
      return std::nullopt;
    return encoded.packed.descriptor;
  };
  const auto success_descriptor =
      descriptor_for(!is_rmw                                  ? SyncOutcome::NotApplicable
                     : is_cas                                 ? SyncOutcome::CasSuccess
                     : site.returns_old_value.value_or(false) ? SyncOutcome::RmwReturnsOld
                                                              : SyncOutcome::RmwNoReturn);
  const auto failure_descriptor =
      is_cas ? descriptor_for(SyncOutcome::CasFailure) : success_descriptor;
  if (!success_descriptor || !failure_descriptor)
    return std::nullopt;

  const uint16_t base = plan.scratch_vgpr;
  const uint16_t value = static_cast<uint16_t>(base + AtomicScratchLayout::kValue);
  const uint16_t expected = static_cast<uint16_t>(base + AtomicScratchLayout::kExpected);
  const uint16_t saved_address = static_cast<uint16_t>(base + AtomicScratchLayout::kSavedAddress);
  const uint16_t bank = static_cast<uint16_t>(base + AtomicScratchLayout::kBank);
  const uint16_t original_exec = static_cast<uint16_t>(*plan.exec_save_sgpr + 6u);
  const uint16_t temporary_exec = *plan.exec_save_sgpr;
  const uint64_t report_base = plan.supercollider_report_buffer_address;
  const uint64_t first_pending_address = report_base + layout.pending_acquires_offset;

  std::vector<uint32_t> words;
  InstructionSequence sequence(words);
  AtomicPreludeState prelude;
  if (!append_atomic_prelude(words, bytes, source, lowering_form, owner_descriptor_file_offset,
                             address_plan, plan, spill, scalar_spill, private_layout, arch,
                             saved_address, /*defer_guest=*/false, prelude, errors,
                             guest_instruction_offset, leading_guest_words, trailing_guest_words,
                             emitted_guest_size))
    return std::nullopt;
  if (!append_window_bank_index(words, plan.dispatch_id, plan.workgroup_sources, bank_count, bank,
                                expected, *plan.owner_epoch_vgprs.owner, arch)) {
    errors.emplace_back("ConSan pending acquire failed at bank selection");
    return std::nullopt;
  }
  if (selected_slot != 0u) {
    if (!sequence.emit(
            instrumentation::build_v_add_u32_literal(bank, expected, selected_slot, bank, arch)))
      return std::nullopt;
  }
  RecordEmitter record(words, base, value, arch);
  const auto contention_label = sequence.make_label();
  const auto collision_label = sequence.make_label();
  const auto restore_label = sequence.make_label();
  if (!append_save_special_state(words, plan.special_state, arch)) {
    errors.emplace_back("ConSan barrier failed at special-state save");
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
    return detail::append_indexed_address(words,
                                          {.table_address = first_pending_address,
                                           .stride_bytes = sizeof(PendingAcquireSlot),
                                           .address_vgpr = base,
                                           .index_vgpr = value},
                                          *target);
  };
  sequence.require(append_pending_address())
      .append(instrumentation::build_v_mov_b32_literal(value, 1u, arch),
              instrumentation::build_v_mov_b32_literal(expected, 0u, arch),
              instrumentation::build_flat_atomic_cmpswap_b32(
                  base, value, value, /*return_old_value=*/true, kAmdGpuScopeDevice, arch))
      .require(append_global_atomic_wait(words, arch))
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

  sequence.require(record.store_vgpr(offsetof(PendingAcquireSlot, selected_slot), bank))
      .require(record.store_literal(offsetof(PendingAcquireSlot, generation),
                                    static_cast<uint32_t>(plan.report_generation)))
      .require(record.store_literal(offsetof(PendingAcquireSlot, generation) + 4u,
                                    static_cast<uint32_t>(plan.report_generation >> 32u)))
      .require(append_store_report_dispatch_id_pair(record, plan.dispatch_id,
                                                    offsetof(PendingAcquireSlot, dispatch_id)))
      .require(record.store_workgroup(offsetof(PendingAcquireSlot, workgroup_x),
                                      plan.workgroup_sources.x))
      .require(record.store_workgroup(offsetof(PendingAcquireSlot, workgroup_y),
                                      plan.workgroup_sources.y))
      .require(record.store_workgroup(offsetof(PendingAcquireSlot, workgroup_z),
                                      plan.workgroup_sources.z))
      .require(
          record.store_vgpr(offsetof(PendingAcquireSlot, owner_id), *plan.owner_epoch_vgprs.owner))
      .require(record.store_vgpr(offsetof(PendingAcquireSlot, source_epoch),
                                 *plan.owner_epoch_vgprs.epoch))
      .require(release_selected_slot
                   ? record.store_vgpr(offsetof(PendingAcquireSlot, reserved), expected)
                   : record.store_literal(offsetof(PendingAcquireSlot, reserved), 0u))
      .require(record.store_vgpr(offsetof(PendingAcquireSlot, metadata) +
                                     offsetof(SyncMetadataPacked, address),
                                 saved_address))
      .require(record.store_vgpr(offsetof(PendingAcquireSlot, metadata) +
                                     offsetof(SyncMetadataPacked, address) + 4u,
                                 static_cast<uint16_t>(saved_address + 1u)))
      .require(record.store_literal(offsetof(PendingAcquireSlot, metadata) +
                                        offsetof(SyncMetadataPacked, byte_count),
                                    *byte_count))
      .require(record.store_vgpr(offsetof(PendingAcquireSlot, metadata) +
                                     offsetof(SyncMetadataPacked, epoch_before),
                                 *plan.owner_epoch_vgprs.epoch))
      .require(record.store_vgpr(offsetof(PendingAcquireSlot, metadata) +
                                     offsetof(SyncMetadataPacked, epoch_after),
                                 *plan.owner_epoch_vgprs.epoch));

  if (is_cas) {
    assert(prelude.cas_compare_vgpr && prelude.cas_result_vgpr);
    const auto cas_failure_label = sequence.make_label();
    const auto descriptor_done_label = sequence.make_label();
    sequence
        .append(instrumentation::build_v_cmp_eq_u32_vcc(
            vector_source_vgpr(*prelude.cas_compare_vgpr), *prelude.cas_result_vgpr, arch))
        .branch(cas_failure_label, InstructionSequence::BranchKind::VccZero)
        .require(record.store_literal(offsetof(PendingAcquireSlot, metadata) +
                                          offsetof(SyncMetadataPacked, descriptor),
                                      *success_descriptor))
        .branch(descriptor_done_label, InstructionSequence::BranchKind::Unconditional)
        .bind_label(cas_failure_label)
        .require(record.store_literal(offsetof(PendingAcquireSlot, metadata) +
                                          offsetof(SyncMetadataPacked, descriptor),
                                      *failure_descriptor))
        .bind_label(descriptor_done_label);
  } else {
    sequence.require(record.store_literal(offsetof(PendingAcquireSlot, metadata) +
                                              offsetof(SyncMetadataPacked, descriptor),
                                          *success_descriptor));
  }

  sequence.append(instrumentation::build_s_wait_global_store0(arch))
      .require(append_pending_address())
      .append(instrumentation::build_v_mov_b32_literal(value, 2u, arch),
              instrumentation::build_v_mov_b32_literal(expected, 1u, arch),
              instrumentation::build_flat_atomic_cmpswap_b32(
                  base, value, value, /*return_old_value=*/true, kAmdGpuScopeDevice, arch))
      .require(append_global_atomic_wait(words, arch))
      .append(instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(expected), value, arch),
              instrumentation::build_s_and_saveexec_b64(temporary_exec, kAmdGpuVccLo, arch))
      .branch(collision_label, InstructionSequence::BranchKind::SccZero)
      .require(append_atomic_fetch_add_one_u32(
          words, report_base + offsetof(ReportHeader, pending_acquire_count), value, base, arch))
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
  append_counter_path(offsetof(ReportHeader, pending_acquire_contention_count));
  sequence.bind_label(collision_label);
  append_counter_path(offsetof(ReportHeader, pending_acquire_collision_count));

  sequence.bind_label(restore_label)
      .append(instrumentation::build_s_mov_b64(kAmdGpuExecLo, original_exec, arch))
      .require(append_restore_special_state(words, plan.special_state, arch));
  if (scalar_spill)
    words.insert(words.end(), scalar_spill->restore_words.begin(),
                 scalar_spill->restore_words.end());
  if (spill)
    words.insert(words.end(), spill->restore_words.begin(), spill->restore_words.end());
  if (!sequence.finish(arch))
    return std::nullopt;
  return words;
}
[[nodiscard]] std::optional<std::vector<uint32_t>> build_atomic_sync_cave_words(
    std::span<const uint8_t> bytes, const AtomicEvidenceSourceView &source,
    const AtomicLoweringForm &lowering_form, uint64_t owner_descriptor_file_offset,
    const AtomicAddressPlan &address_plan, const SyncEmissionPlan &plan,
    const VgprSpillSequence *spill, const SgprSpillSequence *scalar_spill,
    const PrivateStateLayout *private_layout, rj_code_arch_t arch, uint32_t selected_slot,
    uint32_t bank_count, const ReportBufferLayout &layout, std::vector<std::string> &errors,
    uint32_t *guest_instruction_offset, std::span<const uint32_t> leading_guest_words,
    std::span<const uint32_t> trailing_guest_words, uint32_t *emitted_guest_size) {
  const AtomicSite &site = source.site;
  const bool is_rmw = source.is_rmw();
  const bool is_cas = is_rmw && atomic_is_compare_exchange(site);
  const std::optional<uint32_t> byte_count = atomic_byte_count(source);
  const auto event_kind = atomic_event_kind(source.sequence->memory_role);
  if (!plan.owner_epoch_vgprs.owner || !plan.owner_epoch_vgprs.epoch || !plan.exec_save_sgpr ||
      static_cast<uint32_t>(plan.scratch_vgpr) + atomic_scratch_count() > kMaxVgprs ||
      bank_count == 0u || !std::has_single_bit(bank_count) ||
      selected_slot > layout.sync_metadata_capacity ||
      bank_count > layout.sync_metadata_capacity - selected_slot ||
      selected_slot > layout.causal_window_capacity ||
      bank_count > layout.causal_window_capacity - selected_slot ||
      selected_slot > layout.watchpoint_capacity ||
      bank_count > layout.watchpoint_capacity - selected_slot)
    return std::nullopt;
  const TargetProfile *target = target_profile(arch);
  if (target == nullptr)
    return std::nullopt;
  if (!address_plan.supported() || !site.scope || !byte_count)
    return std::nullopt;
  const auto role = event_kind ? atomic_role(*event_kind, is_rmw) : std::nullopt;
  const auto scope = sync_scope(*site.scope);
  if (!role || !scope)
    return std::nullopt;
  if (site.file_offset > bytes.size() || site.size > bytes.size() - site.file_offset) {
    errors.emplace_back("ConSan atomic metadata site exceeds ELF bytes");
    return std::nullopt;
  }

  if (is_cas &&
      (!site.data_vgpr || !site.destination_vgpr || !site.returns_old_value.value_or(false))) {
    errors.emplace_back("ConSan atomic metadata requires an exact CAS outcome");
    return std::nullopt;
  }

  const auto descriptor_for = [&](SyncOutcome outcome) -> std::optional<uint32_t> {
    const SyncEncodeResult encoded = encode_sync_metadata({
        .address = 1,
        .byte_count = *byte_count,
        .kind = SyncMetadataKind::Atomic,
        .role = *role,
        .scope = *scope,
        .outcome = outcome,
        .epoch_before = 0,
        .epoch_after = 0,
    });
    if (encoded.classification != SyncClassification::Valid)
      return std::nullopt;
    return encoded.packed.descriptor;
  };
  const auto success_descriptor =
      descriptor_for(!is_rmw                                  ? SyncOutcome::NotApplicable
                     : is_cas                                 ? SyncOutcome::CasSuccess
                     : site.returns_old_value.value_or(false) ? SyncOutcome::RmwReturnsOld
                                                              : SyncOutcome::RmwNoReturn);
  const auto failure_descriptor =
      is_cas ? descriptor_for(SyncOutcome::CasFailure) : success_descriptor;
  if (!success_descriptor || !failure_descriptor)
    return std::nullopt;

  const uint16_t base = plan.scratch_vgpr;
  const uint16_t value = static_cast<uint16_t>(base + AtomicScratchLayout::kValue);
  const uint16_t expected = static_cast<uint16_t>(base + AtomicScratchLayout::kExpected);
  const uint16_t saved_address = static_cast<uint16_t>(base + AtomicScratchLayout::kSavedAddress);
  const uint16_t bank = static_cast<uint16_t>(base + AtomicScratchLayout::kBank);
  const uint16_t original_exec = static_cast<uint16_t>(*plan.exec_save_sgpr + 6u);
  const uint16_t temporary_exec = *plan.exec_save_sgpr;
  const uint64_t report_base = plan.supercollider_report_buffer_address;
  const uint64_t first_window_address = report_base + layout.causal_windows_offset;
  const uint64_t first_watchpoint_address = report_base + layout.watchpoints_offset;
  const uint64_t first_metadata_address = report_base + layout.sync_metadata_offset;

  // A release-only RMW that does not return a value has no outcome evidence
  // needed by the probe. Publish its shadow metadata before executing the
  // guest release so the release operation orders that metadata and the
  // original code can resume immediately after the externally visible RMW.
  // This is especially important for code that publishes its own side
  // metadata immediately after the atomic. Keep guest-first ordering when a
  // spill overlaps guest operands, and for acquire/CAS operations whose
  // outcome is needed by the probe.
  const bool release_only =
      (static_cast<uint32_t>(*role) & static_cast<uint32_t>(SyncRole::Release)) != 0u &&
      (static_cast<uint32_t>(*role) & static_cast<uint32_t>(SyncRole::Acquire)) == 0u;
  const bool defer_guest = release_only && !is_cas &&
                           !(spill && atomic_spill_overlaps_guest_operands(*spill, lowering_form));

  std::vector<uint32_t> words;
  InstructionSequence sequence(words);
  AtomicPreludeState prelude;
  sequence.require(append_atomic_prelude(
      words, bytes, source, lowering_form, owner_descriptor_file_offset, address_plan, plan, spill,
      scalar_spill, private_layout, arch, saved_address, defer_guest, prelude, errors,
      guest_instruction_offset, leading_guest_words, trailing_guest_words, emitted_guest_size));
  if (sequence &&
      !append_window_bank_index(words, plan.dispatch_id, plan.workgroup_sources, bank_count, bank,
                                expected, *plan.owner_epoch_vgprs.owner, arch)) {
    errors.emplace_back("ConSan atomic metadata failed at bank selection");
    sequence.require(false);
  }
  if (selected_slot != 0u)
    sequence.append(
        instrumentation::build_v_add_u32_literal(bank, expected, selected_slot, bank, arch));
  RecordEmitter record(words, base, value, arch);
  const auto collision_label = sequence.make_label();
  const auto restore_label = sequence.make_label();
  sequence.require(append_save_special_state(words, plan.special_state, arch))
      .append(instrumentation::build_s_mov_b64(original_exec, kAmdGpuExecLo, arch))
      .require(detail::append_indexed_address(words,
                                              {.table_address = first_window_address,
                                               .stride_bytes = sizeof(CausalWindow),
                                               .address_vgpr = base,
                                               .index_vgpr = bank},
                                              *target))
      .require(append_causal_window_validation(words, sequence,
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
      .require(detail::append_indexed_address(words,
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
  narrow_masked_low(static_cast<uint32_t>(watchpoint::valid_mask),
                    static_cast<uint32_t>(watchpoint::valid_mask));
  narrow_masked_low(static_cast<uint32_t>(watchpoint::consumed_mask), 0u);
  sequence.require(append_load_u32_vgpr_at_offset(words, base, 0u, value, arch))
      .append(instrumentation::build_v_lshrrev_b32(
                  value, scalar_positive_inline_u32(watchpoint::access_kind_shift), value, arch),
              instrumentation::build_v_and_b32_literal(
                  value, (1u << watchpoint::access_kind_bits) - 1u, value, arch),
              instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0u), value, arch));
  narrow_current_vcc();
  sequence.append(
      instrumentation::build_v_cmp_gt_u32_vcc(scalar_positive_inline_u32(3u), value, arch));
  narrow_current_vcc();
  sequence.require(append_load_u32_vgpr_at_offset(words, base, 0u, value, arch))
      .append(instrumentation::build_v_lshrrev_b32(
                  value, scalar_positive_inline_u32(watchpoint::owner_shift), value, arch),
              instrumentation::build_v_and_b32_literal(value, watchpoint::max_owner, value, arch),
              instrumentation::build_v_cmp_eq_u32_vcc(
                  vector_source_vgpr(*plan.owner_epoch_vgprs.owner), value, arch),
              instrumentation::build_s_and_saveexec_b64(temporary_exec, kAmdGpuVccLo, arch))
      .branch(restore_label, InstructionSequence::BranchKind::SccZero);

  sequence.require(append_load_u32_vgpr_at_offset(words, base, 0u, value, arch))
      .append(instrumentation::build_v_lshrrev_b32(
                  value, scalar_positive_inline_u32(watchpoint::epoch_shift), value, arch),
              instrumentation::build_v_and_b32_literal(value, watchpoint::max_epoch, value, arch),
              instrumentation::build_v_cmp_eq_u32_vcc(
                  vector_source_vgpr(*plan.owner_epoch_vgprs.epoch), value, arch));
  narrow_current_vcc();
  sequence.require(append_load_u32_vgpr_at_offset(words, base, 0u, value, arch))
      .require(append_load_u32_vgpr_at_offset(words, base, sizeof(uint32_t), expected, arch))
      .append(
          instrumentation::build_v_lshrrev_b32(
              value, scalar_positive_inline_u32(watchpoint::generation_shift), value, arch),
          instrumentation::build_v_lshlrev_b32(
              expected, scalar_positive_inline_u32(32u - watchpoint::generation_shift), expected,
              arch),
          instrumentation::build_v_add_u32(value, vector_source_vgpr(expected), value, arch),
          instrumentation::build_v_and_b32_literal(value, watchpoint::max_generation, value, arch),
          instrumentation::build_v_mov_b32_literal(
              expected, static_cast<uint32_t>(plan.report_generation) & watchpoint::max_generation,
              arch),
          instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(expected), value, arch));
  narrow_current_vcc();

  sequence.append(instrumentation::build_s_mov_b64(temporary_exec, kAmdGpuExecLo, arch))
      .require(append_select_first_lane_in_exec_mask(words, value, temporary_exec, temporary_exec,
                                                     arch));

  const auto append_publication = [&](uint32_t descriptor) {
    sequence
        .require(detail::append_indexed_address(
            words,
            {.table_address = first_metadata_address + offsetof(SyncMetadataPacked, descriptor),
             .stride_bytes = sizeof(SyncMetadataPacked),
             .address_vgpr = base,
             .index_vgpr = bank},
            *target))
        .append(instrumentation::build_v_mov_b32_literal(value, kSyncPublishingDescriptor, arch),
                instrumentation::build_v_mov_b32_literal(expected, 0, arch),
                instrumentation::build_flat_atomic_cmpswap_b32(base, value, value, true,
                                                               kAmdGpuScopeDevice, arch))
        .require(append_global_atomic_wait(words, arch))
        .append(instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), value, arch),
                instrumentation::build_s_and_saveexec_b64(temporary_exec, kAmdGpuVccLo, arch))
        .branch(collision_label, InstructionSequence::BranchKind::SccZero)
        .require(detail::append_indexed_address(words,
                                                {.table_address = first_metadata_address,
                                                 .stride_bytes = sizeof(SyncMetadataPacked),
                                                 .address_vgpr = base,
                                                 .index_vgpr = bank},
                                                *target))
        .require(record.store_vgpr(offsetof(SyncMetadataPacked, address), saved_address))
        .require(record.store_vgpr(offsetof(SyncMetadataPacked, address) + 4u,
                                   static_cast<uint16_t>(saved_address + 1u)))
        .require(record.store_literal(offsetof(SyncMetadataPacked, byte_count), *byte_count))
        .require(record.store_vgpr(offsetof(SyncMetadataPacked, epoch_before),
                                   *plan.owner_epoch_vgprs.epoch))
        .require(record.store_vgpr(offsetof(SyncMetadataPacked, epoch_after),
                                   *plan.owner_epoch_vgprs.epoch))
        .append(instrumentation::build_s_wait_global_store0(arch))
        .require(detail::append_indexed_address(
            words,
            {.table_address = first_metadata_address + offsetof(SyncMetadataPacked, descriptor),
             .stride_bytes = sizeof(SyncMetadataPacked),
             .address_vgpr = base,
             .index_vgpr = bank},
            *target))
        .append(instrumentation::build_v_mov_b32_literal(value, descriptor, arch),
                instrumentation::build_v_mov_b32_literal(expected, kSyncPublishingDescriptor, arch),
                instrumentation::build_flat_atomic_cmpswap_b32(base, value, value, true,
                                                               kAmdGpuScopeDevice, arch))
        .require(append_global_atomic_wait(words, arch))
        .append(instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(expected), value, arch),
                instrumentation::build_s_and_saveexec_b64(temporary_exec, kAmdGpuVccLo, arch))
        .branch(collision_label, InstructionSequence::BranchKind::SccZero)
        .require(append_atomic_fetch_add_one_u32(
            words, report_base + offsetof(ReportHeader, sync_metadata_count), value, base, arch))
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
          words, report_base + offsetof(ReportHeader, dropped_window_count), value, base, arch))
      .bind_label(restore_label)
      .append(instrumentation::build_s_mov_b64(kAmdGpuExecLo, original_exec, arch))
      .require(append_restore_special_state(words, plan.special_state, arch));
  if (scalar_spill)
    words.insert(words.end(), scalar_spill->restore_words.begin(),
                 scalar_spill->restore_words.end());
  if (spill)
    words.insert(words.end(), spill->restore_words.begin(), spill->restore_words.end());
  if (defer_guest) {
    std::vector<uint32_t> deferred_guest_words;
    const bool guest_ok =
        append_atomic_guest(deferred_guest_words, bytes, site, is_rmw, arch,
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

} // namespace rocjitsu::consan::detail

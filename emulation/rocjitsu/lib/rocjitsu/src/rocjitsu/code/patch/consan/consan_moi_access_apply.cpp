// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_access_apply.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/consan/consan_capability_contract.h"
#include "rocjitsu/code/patch/consan/consan_growth_policy.h"
#include "rocjitsu/code/patch/consan/consan_moi_report_emission.h"
#include "rocjitsu/code/patch/consan/consan_placement.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"
#include "rocjitsu/code/patch/spill_manager.h"

#include <algorithm>
#include <ranges>

namespace rocjitsu::consan_moi_impl {

using consan_moi_detail::append_word_bytes;

[[nodiscard]] std::optional<ConSanStaticAccessAttribution> make_moi_access_attribution(
    const ConSanObservationPlan &observation, const ConSanMoiCandidate &candidate,
    const ConSanPatchLoweringProduct &patch, ConSanProbeIntentKind expected_intent) {
  if (patch.original_size == 0u)
    return std::nullopt;
  ConSanStaticAccessAttribution access{
      .intent_ids = candidate.intent_ids,
      .original_site = candidate.physical_id,
      .original_semantic_sites = {},
      .execution_owner_descriptor_file_offsets = patch.owner_descriptor_file_offsets,
      .owner_provenance_complete = !patch.owner_descriptor_file_offsets.empty(),
  };
  std::ranges::sort(access.execution_owner_descriptor_file_offsets);
  access.execution_owner_descriptor_file_offsets.erase(
      std::ranges::unique(access.execution_owner_descriptor_file_offsets).begin(),
      access.execution_owner_descriptor_file_offsets.end());
  for (ConSanProbeIntentId id : candidate.intent_ids) {
    const ConSanProbeIntent *intent = observation.intent(id);
    if (intent == nullptr || intent->kind != expected_intent)
      return std::nullopt;
    for (const SemanticSiteId &site : intent->covered_semantic_sites) {
      if (std::ranges::find(access.original_semantic_sites, site) ==
          access.original_semantic_sites.end()) {
        access.original_semantic_sites.push_back(site);
      }
    }
  }
  if (candidate.intent_ids.empty())
    return std::nullopt;
  return access;
}

/// Apply the descriptor growth shared by every appended MOI access engine.
///
/// VGPR, SGPR, and private-memory requirements come from the common access
/// plan. InlineShadow may additionally supply an LDS requirement for its
/// workgroup-local shadow; replay-style engines pass no LDS requirement.
[[nodiscard]] bool apply_moi_appended_access_descriptor_requirements(
    CodeObjectPatcher &patcher, const AmdGpuCodeObject &code_object,
    const RuntimeCapabilities &capabilities,
    const MoiDescriptorVgprRequirements &descriptor_requirements,
    const MoiDescriptorSgprRequirements &scalar_requirements,
    const MoiDescriptorPrivateRequirements &private_requirements,
    const MoiDescriptorLdsRequirements *lds_requirements, rj_code_arch_t arch,
    const ProgramInventory &program_inventory, std::vector<std::string> &errors) {
  return apply_moi_descriptor_requirements(
      patcher, code_object, program_inventory, descriptor_requirements, scalar_requirements,
      private_requirements, lds_requirements, &capabilities, arch, "ConSan MOI access", errors);
}

/// Initialize one appended MOI access image and materialize its shared direct
/// relay reservoirs.
///
/// This is byte-placement mechanics only: the caller has already selected the
/// engine's evidence and patch plans. The body prefix is filled with target
/// NOPs so every subsequently planned offset remains stable.
[[nodiscard]] bool initialize_moi_appended_access_text(
    CodeObjectPatcher &patcher, bool use_indirect_appended, uint64_t indirect_body_begin,
    const BranchOnlyDirectRelayReservoirSet &direct_reservoirs, std::string_view probe_name,
    rj_code_arch_t arch, std::vector<uint8_t> &new_text, std::vector<ConSanPatchInfo> &patches,
    std::vector<std::string> &errors) {
  const std::span<const uint8_t> old_text = patcher.text_bytes();
  if (old_text.empty()) {
    errors.emplace_back("ConSan MOI " + std::string(probe_name) + " found no .text section");
    return false;
  }
  new_text.assign(old_text.begin(), old_text.end());
  if (use_indirect_appended) {
    while (new_text.size() < indirect_body_begin)
      append_word_bytes(new_text, build_s_nop(0, arch));
  }
  for (const BranchOnlyDirectRelayReservoir &reservoir : direct_reservoirs.reservoirs) {
    std::string relay_error;
    if (!BranchOnlyRelayRouter::emit_direct_reservoir(new_text, reservoir, arch, patches,
                                                      &relay_error)) {
      errors.emplace_back("ConSan MOI " + std::string(probe_name) + " " + relay_error);
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool write_moi_appended_anchor(std::vector<uint8_t> &text,
                                             const MoiAppendedAnchorPlan &plan,
                                             uint64_t entry_target, rj_code_arch_t arch,
                                             std::string_view probe_name,
                                             std::vector<std::string> &errors) {
  const auto forward = compute_sopp_branch_simm16(plan.placement_anchor, entry_target);
  if (!plan.entry_island_at_anchor && !forward) {
    errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                        " appended cave entry is out of range");
    return false;
  }
  std::vector<uint32_t> anchor_words(plan.original_size / sizeof(uint32_t), build_s_nop(0, arch));
  if (anchor_words.empty()) {
    errors.emplace_back("ConSan MOI " + std::string(probe_name) + " has an empty anchor");
    return false;
  }
  if (!plan.entry_island_at_anchor)
    anchor_words.front() = build_s_branch(*forward, arch);
  const uint64_t anchor_bytes = anchor_words.size() * sizeof(uint32_t);
  if (plan.candidate_anchor > text.size() || anchor_bytes > text.size() - plan.candidate_anchor) {
    errors.emplace_back("ConSan MOI " + std::string(probe_name) + " anchor exceeds .text");
    return false;
  }
  std::memcpy(text.data() + plan.candidate_anchor, anchor_words.data(),
              static_cast<size_t>(anchor_bytes));
  return true;
}

[[nodiscard]] bool write_moi_entry_island(std::vector<uint8_t> &text,
                                          const MoiEntryIslandPlan &plan,
                                          std::span<const uint32_t> island_words,
                                          std::vector<ConSanPatchInfo> &patches,
                                          std::string_view probe_name,
                                          std::vector<std::string> &errors) {
  if (!plan.island_offset)
    return true;
  const uint64_t island_bytes = island_words.size() * sizeof(uint32_t);
  const uint64_t island_offset = *plan.island_offset;
  if (island_offset > text.size() || island_bytes > text.size() - island_offset) {
    errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                        " entry island exceeds reserved text");
    return false;
  }
  std::memcpy(text.data() + island_offset, island_words.data(), static_cast<size_t>(island_bytes));
  if (!plan.entry_island_at_anchor) {
    ConSanPatchInfo island_info;
    island_info.kind = ConSanPatchKind::TrampolineMoiIndirectBranchIsland;
    island_info.anchor_offset = plan.candidate_anchor;
    island_info.trampoline_offset = island_offset;
    island_info.trampoline_size = static_cast<uint32_t>(island_bytes);
    island_info.owner_descriptor_file_offsets.assign(plan.owner_descriptor_file_offsets.begin(),
                                                     plan.owner_descriptor_file_offsets.end());
    patches.push_back(std::move(island_info));
  }
  return true;
}

[[nodiscard]] std::optional<std::vector<uint32_t>>
assemble_moi_appended_body(const MoiAppendedBodyPatchPlan &plan,
                           std::span<const uint32_t> probe_words, std::string_view probe_name,
                           std::vector<std::string> &errors,
                           const MoiAppendedBodyOptions &options) {
  std::vector<uint32_t> body;
  if (options.trailing_guest_word_count > probe_words.size()) {
    errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                        " VGPR-bank guest split exceeds the probe body");
    return std::nullopt;
  }
  // These checks are defensive backstops for plans assembled by ConSan's
  // internal callers. Public options cannot directly construct either invalid
  // combination, but release builds must not silently emit a corrupt body if a
  // future planner violates the contract.
  if (!moi_appended_body_vgpr_bank_mode_is_valid(options.arch, options.incoming_vgpr_bank_mode)) {
    errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                        " has a selectable-VGPR-bank mode on an unsupported target body");
    return std::nullopt;
  }
  if (options.high_bank_address_source.has_value() !=
      options.high_bank_address_destination.has_value()) {
    errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                        " has an incomplete high-bank address capture");
    return std::nullopt;
  }
  const bool select_low_vgpr_bank =
      moi_appended_body_uses_selectable_vgpr_bank(options.arch, options.incoming_vgpr_bank_mode);
  const bool capture_high_bank_address = options.high_bank_address_source.has_value();
  const uint8_t incoming_vgpr_msb_mode =
      static_cast<uint8_t>(options.incoming_vgpr_bank_mode.value_or(0u));
  const uint8_t address_capture_mode = static_cast<uint8_t>(incoming_vgpr_msb_mode & 0x3u);
  if (capture_high_bank_address &&
      (!select_low_vgpr_bank || address_capture_mode == 0u ||
       *options.high_bank_address_source > 255u || *options.high_bank_address_destination > 255u)) {
    errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                        " has an invalid high-bank address capture");
    return std::nullopt;
  }
  const size_t probe_prefix_words = probe_words.size() - options.trailing_guest_word_count;
  body.insert(body.end(), options.entry_prefix_words.begin(), options.entry_prefix_words.end());
  uint8_t current_vgpr_msb_mode = incoming_vgpr_msb_mode;
  const bool save_spill_before_high_bank_capture = capture_high_bank_address && plan.spill;
  if (save_spill_before_high_bank_capture) {
    body.push_back(
        *instrumentation::build_s_set_vgpr_msb_transition(current_vgpr_msb_mode, 0u, options.arch));
    body.insert(body.end(), plan.spill->save_words.begin(), plan.spill->save_words.end());
    current_vgpr_msb_mode = 0u;
  }
  if (capture_high_bank_address) {
    if (current_vgpr_msb_mode != address_capture_mode) {
      body.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
          current_vgpr_msb_mode, address_capture_mode, options.arch));
      current_vgpr_msb_mode = address_capture_mode;
    }
    body.push_back(build_v_mov_b32_e32(*options.high_bank_address_destination,
                                       vector_source_vgpr(*options.high_bank_address_source),
                                       options.arch));
  }
  if (select_low_vgpr_bank && current_vgpr_msb_mode != 0u)
    body.push_back(
        *instrumentation::build_s_set_vgpr_msb_transition(current_vgpr_msb_mode, 0u, options.arch));
  if (plan.spill && !save_spill_before_high_bank_capture)
    body.insert(body.end(), plan.spill->save_words.begin(), plan.spill->save_words.end());
  body.insert(body.end(), options.scalar_save_words.begin(), options.scalar_save_words.end());
  const size_t probe_body_begin = body.size();
  std::optional<size_t> guest_word;
  size_t guest_word_count = 0u;
  if (options.probe_guest_instruction_offset) {
    if (*options.probe_guest_instruction_offset % sizeof(uint32_t) != 0u) {
      errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                          " has an unaligned relocated guest offset");
      return std::nullopt;
    }
    guest_word = *options.probe_guest_instruction_offset / sizeof(uint32_t);
    guest_word_count = options.guest_instruction_word_count.value_or(plan.guest_instruction_size /
                                                                     sizeof(uint32_t));
    if (*guest_word >= probe_words.size() || guest_word_count == 0u ||
        guest_word_count > probe_words.size() - *guest_word) {
      errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                          " has an invalid relocated guest range");
      return std::nullopt;
    }
  }
  const bool valid_embedded_guest_plan = moi_embedded_guest_vgpr_bank_plan_is_valid(
      options.wrap_embedded_guest_vgpr_bank, select_low_vgpr_bank, guest_word.has_value(),
      options.trailing_guest_word_count);
  if (!valid_embedded_guest_plan) {
    errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                        " has an invalid embedded-guest VGPR-bank plan");
    return std::nullopt;
  }
  const bool wrap_embedded_guest = options.wrap_embedded_guest_vgpr_bank;
  if (wrap_embedded_guest) {
    body.insert(body.end(), probe_words.begin(), probe_words.begin() + *guest_word);
    body.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        0u, static_cast<uint8_t>(*options.incoming_vgpr_bank_mode), options.arch));
    if (options.body_guest_instruction_offset)
      *options.body_guest_instruction_offset =
          static_cast<uint32_t>(body.size() * sizeof(uint32_t));
    body.insert(body.end(), probe_words.begin() + *guest_word,
                probe_words.begin() + *guest_word + guest_word_count);
    body.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        static_cast<uint8_t>(*options.incoming_vgpr_bank_mode), 0u, options.arch));
    body.insert(body.end(), probe_words.begin() + *guest_word + guest_word_count,
                probe_words.begin() + probe_prefix_words);
  } else {
    body.insert(body.end(), probe_words.begin(), probe_words.begin() + probe_prefix_words);
  }
  body.insert(body.end(), options.scalar_restore_words.begin(), options.scalar_restore_words.end());
  if (options.restore_vgpr_spill_before_trailing_guest && plan.spill)
    body.insert(body.end(), plan.spill->restore_words.begin(), plan.spill->restore_words.end());
  if (select_low_vgpr_bank) {
    body.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        0u, static_cast<uint8_t>(*options.incoming_vgpr_bank_mode), options.arch));
  }
  const size_t trailing_probe_body_begin = body.size();
  body.insert(body.end(), probe_words.begin() + probe_prefix_words, probe_words.end());
  if (guest_word && options.body_guest_instruction_offset && !wrap_embedded_guest) {
    const size_t body_word = *guest_word < probe_prefix_words
                                 ? probe_body_begin + *guest_word
                                 : trailing_probe_body_begin + *guest_word - probe_prefix_words;
    *options.body_guest_instruction_offset = static_cast<uint32_t>(body_word * sizeof(uint32_t));
  }
  if (select_low_vgpr_bank && plan.spill)
    body.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        static_cast<uint8_t>(*options.incoming_vgpr_bank_mode), 0u, options.arch));
  if (plan.spill && !options.restore_vgpr_spill_before_trailing_guest)
    body.insert(body.end(), plan.spill->restore_words.begin(), plan.spill->restore_words.end());
  if (select_low_vgpr_bank && plan.spill) {
    body.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        0u, static_cast<uint8_t>(*options.incoming_vgpr_bank_mode), options.arch));
  }
  body.insert(body.end(), plan.displaced_tail_words.begin(), plan.displaced_tail_words.end());
  if ((body.size() + options.deferred_guest_word_count) * sizeof(uint32_t) != plan.body_size) {
    errors.emplace_back(
        "ConSan MOI " + std::string(probe_name) + " body size changed after placement: emitted=" +
        std::to_string((body.size() + options.deferred_guest_word_count) * sizeof(uint32_t)) +
        " planned=" + std::to_string(plan.body_size));
    return std::nullopt;
  }
  return body;
}

[[nodiscard]] std::optional<MoiFinalizedAccessBody> finalize_moi_access_body(
    std::vector<uint32_t> body, uint64_t body_text_offset, uint64_t planned_body_size,
    const MoiAccessContinuationPlan &continuation, rj_code_arch_t arch,
    std::string_view probe_name, std::vector<std::string> &errors) {
  if (continuation.displaced_tail_word_count > body.size()) {
    errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                        " lost its displaced continuation");
    return std::nullopt;
  }
  if (continuation.transfer == MoiAccessTransferKind::IndirectJump &&
      (!continuation.pc_sgpr || !continuation.scc_save_sgpr)) {
    errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                        " has incomplete indirect-return scalar state");
    return std::nullopt;
  }
  if (continuation.transfer == MoiAccessTransferKind::SetPc && !continuation.setpc_sgpr) {
    errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                        " has no call-return scalar state");
    return std::nullopt;
  }
  if (continuation.guest_before_terminal_transfer &&
      continuation.transfer != MoiAccessTransferKind::IndirectJump) {
    errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                        " requested a non-indirect terminal guest boundary");
    return std::nullopt;
  }
  if (continuation.empty_exec_resumes_at_transfer &&
      continuation.empty_exec_resumes_at_displaced_tail) {
    errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                        " has ambiguous empty-wave continuation state");
    return std::nullopt;
  }

  std::vector<uint32_t> displaced_tail;
  if (!continuation.deferred_guest_words.empty() && continuation.displaced_tail_word_count != 0u) {
    auto tail_begin = body.end() - continuation.displaced_tail_word_count;
    displaced_tail.assign(tail_begin, body.end());
    body.erase(tail_begin, body.end());
  }
  const size_t continuation_begin = body.size();
  std::optional<size_t> deferred_guest_word;
  size_t transfer_word = continuation_begin;

  const auto insert_guest_and_tail = [&](std::vector<uint32_t>::iterator position) {
    const size_t position_index = static_cast<size_t>(position - body.begin());
    if (!continuation.deferred_guest_words.empty()) {
      deferred_guest_word = position_index;
      position = body.insert(position, continuation.deferred_guest_words.begin(),
                             continuation.deferred_guest_words.end());
      position += continuation.deferred_guest_words.size();
    }
    body.insert(position, displaced_tail.begin(), displaced_tail.end());
  };

  if (!continuation.guest_before_terminal_transfer)
    insert_guest_and_tail(body.end());

  transfer_word = body.size();
  switch (continuation.transfer) {
  case MoiAccessTransferKind::DirectBranch: {
    const auto branch = compute_sopp_branch_simm16(
        body_text_offset + body.size() * sizeof(uint32_t), continuation.return_target);
    if (!branch) {
      errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                          " direct return is out of range");
      return std::nullopt;
    }
    body.push_back(build_s_branch(*branch, arch));
    break;
  }
  case MoiAccessTransferKind::IndirectJump: {
    const size_t return_begin = body.size();
    if (!append_moi_scc_preserving_indirect_jump(
            body, body_text_offset, continuation.return_target, *continuation.pc_sgpr,
            *continuation.scc_save_sgpr, continuation.capture_scc, arch)) {
      errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                          " could not encode its indirect return");
      return std::nullopt;
    }
    while (body.size() < return_begin + continuation.reserved_transfer_words)
      body.push_back(build_s_nop(0u, arch));
    if (continuation.dependency_wait_before_transfer) {
      const auto wait_pc = instrumentation::build_s_wait_indirect_pc0(arch);
      if (!wait_pc || body.size() <= return_begin) {
        errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                            " could not encode its return dependency wait");
        return std::nullopt;
      }
      body.insert(body.end() - 1, *wait_pc);
    }
    if (continuation.guest_before_terminal_transfer) {
      insert_guest_and_tail(body.end() - 1);
      transfer_word = body.size() - 1u;
    }
    break;
  }
  case MoiAccessTransferKind::SetPc:
    body.push_back(build_s_setpc_b64(*continuation.setpc_sgpr, arch));
    break;
  }

  const size_t continuation_payload_words = continuation.deferred_guest_words.size() +
                                            displaced_tail.size();
  const size_t transfer_and_suffix_words = body.size() -
                                           (continuation_begin + continuation_payload_words);
  if ((body.size() - transfer_and_suffix_words) * sizeof(uint32_t) != planned_body_size) {
    errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                        " body size changed after placing its guest continuation");
    return std::nullopt;
  }

  if (continuation.empty_exec_guard_word) {
    size_t target = continuation.empty_exec_resumes_at_transfer ? transfer_word
                                                                : continuation_begin;
    if (continuation.empty_exec_resumes_at_displaced_tail) {
      if (continuation.displaced_tail_word_count > continuation_begin) {
        errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                            " has no displaced continuation for its empty-wave guard");
        return std::nullopt;
      }
      target = continuation_begin - continuation.displaced_tail_word_count;
    }
    const size_t guard = *continuation.empty_exec_guard_word;
    if (guard >= body.size() || target <= guard || target - guard - 1u >
                                                   static_cast<size_t>(
                                                       std::numeric_limits<int16_t>::max())) {
      errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                          " empty-wave guard cannot reach its continuation");
      return std::nullopt;
    }
    const auto skip_empty_wave = instrumentation::build_s_cbranch_execz(
        static_cast<int16_t>(target - guard - 1u), arch);
    if (!skip_empty_wave) {
      errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                          " could not encode its empty-wave guard");
      return std::nullopt;
    }
    body[guard] = *skip_empty_wave;
  }
  return MoiFinalizedAccessBody{
      .words = std::move(body),
      .deferred_guest_offset = deferred_guest_word
                                   ? std::optional<uint32_t>(static_cast<uint32_t>(
                                         *deferred_guest_word * sizeof(uint32_t)))
                                   : std::nullopt,
  };
}

[[nodiscard]] bool emit_moi_access_programs(
    std::vector<uint8_t> &text, std::vector<MoiAccessPatchProgram> programs,
    rj_code_arch_t arch, std::string_view probe_name, std::vector<ConSanPatchInfo> &patches,
    std::vector<ConSanCommittedLowering> &commits, std::vector<std::string> &errors) {
  commits.reserve(commits.size() + programs.size());
  patches.reserve(patches.size() + programs.size());
  for (MoiAccessPatchProgram &program : programs) {
    for (const MoiAccessReservedRegion &region : program.reserved_regions) {
      if (text.size() != region.offset) {
        errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                            " emitted a stale reserved-region mapping");
        return false;
      }
      for (uint32_t word = 0; word < region.word_count; ++word)
        append_word_bytes(text, build_s_nop(0u, arch));
    }

    if (program.publication == MoiAccessBodyPublication::Append) {
      if (program.pad_to_body_offset) {
        while (text.size() < program.body_offset)
          append_word_bytes(text, build_s_nop(0u, arch));
      }
      if (text.size() != program.body_offset) {
        errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                            " emitted a stale appended-body mapping");
        return false;
      }
      if (program.anchor &&
          !write_moi_appended_anchor(text, *program.anchor, program.entry_target, arch, probe_name,
                                     errors)) {
        return false;
      }
      if (program.entry_island &&
          !write_moi_entry_island(text, *program.entry_island, program.entry_island_words, patches,
                                  probe_name, errors)) {
        return false;
      }
      append_words_bytes(text, program.body_words);
      for (uint32_t word = 0; word < program.trailing_reserved_word_count; ++word)
        append_word_bytes(text, build_s_nop(0u, arch));
    } else {
      const uint64_t body_bytes = program.body_words.size() * sizeof(uint32_t);
      if (body_bytes != program.patch.original_size) {
        errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                            " final inline body size changed");
        return false;
      }
      if (program.body_offset > text.size() || body_bytes > text.size() - program.body_offset) {
        errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                            " inline body exceeds .text");
        return false;
      }
      std::memcpy(text.data() + program.body_offset, program.body_words.data(),
                  static_cast<size_t>(body_bytes));
    }

    patches.push_back(std::move(program.patch));
    commits.push_back(std::move(program.commit));
    if (program.branch_only_route) {
      std::string relay_error;
      const ConSanPatchInfo &patch = patches.back();
      if (!BranchOnlyRelayRouter::emit_and_record(
              text, *program.branch_only_route, patch.trampoline_offset, program.return_target,
              arch, patches, &relay_error)) {
        errors.emplace_back("ConSan MOI " + std::string(probe_name) + " " + relay_error);
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] bool publish_moi_appended_access_text(
    CodeObjectPatcher patcher, std::vector<uint8_t> text,
    const ConSanPatchedImageGrowthLimit &growth_limit,
    std::string_view probe_name, ConSanTransformArtifacts &result,
    std::vector<ConSanCommittedLowering> commits, std::vector<ConSanPatchInfo> patches) {
  if (!replace_consan_text(patcher, text, growth_limit, "MOI " + std::string(probe_name),
                           result.program_inventory.code_object_id(), result.errors,
                           &result.transform_failure_cause)) {
    return false;
  }
  return result.publish_access_lowering(std::move(patcher).emit(),
                                        "ConSan MOI " + std::string(probe_name),
                                        std::move(commits), std::move(patches));
}

[[nodiscard]] bool
moi_scalar_spill_requires_dynamic_vgpr_frame(const ProgramInventory &inventory,
                                             const ResolvedMoiScratchPlan &resources,
                                             const ConSanMoiOperatingPoint &point) {
  if (!point.has_moi_scalar_spill()) {
    return false;
  }
  return std::ranges::any_of(resources.owner_descriptor_file_offsets, [&](uint64_t offset) {
    const ConSanKernelInfo *kernel = inventory.find_kernel_by_descriptor(offset);
    return kernel != nullptr && kernel->uses_dynamic_stack.value_or(false);
  });
}

/// Build the common scalar-preservation transaction selected by resource
/// planning. Dynamic-stack owners borrow the already-established VGPR spill
/// frame; fixed-stack owners use the planned lane-backed reservoir. Engine
/// policy decides whether scalar spilling is enabled, but no engine privately
/// interprets the owner stack model or reservoir geometry.
[[nodiscard]] std::optional<SgprSpillSequence> build_moi_sgpr_spill_sequence(
    const ProgramInventory &inventory, const ResolvedMoiScratchPlan &resources,
    const ConSanRequest &request, const BoundRuntimeResources &bound_resources,
    const ConSanMoiOperatingPoint &point, const MoiObjectModeSemantics &mode_semantics,
    MoiSpillManagers &managers, rj_code_arch_t arch, std::vector<std::string> &warnings,
    std::optional<uint32_t> private_layout_base, const VgprSpillSequence *vgpr_spill) {
  if ((!point.has_moi_scalar_spill()) || !point.moi_exec_save_sgpr)
    return std::nullopt;
  if (!consan_is_capability_arch(arch) || resources.owner_descriptor_file_offsets.empty()) {
    warnings.emplace_back(
        "ConSan MOI spill-backed scalar state requires at least one supported kernel owner");
    return std::nullopt;
  }
  const auto private_limit = consan_address_free_private_limit(arch);
  if (!private_limit) {
    warnings.emplace_back("ConSan MOI spill-backed Inline scalar state has no private capacity");
    return std::nullopt;
  }
  const uint16_t count = moi_exec_save_sgpr_count(
      resolve_moi_exec_save_requirement(request, bound_resources, point, mode_semantics), arch);
  if (moi_scalar_spill_requires_dynamic_vgpr_frame(inventory, resources, point)) {
    if (vgpr_spill == nullptr || !vgpr_spill->uses_dynamic_stack_frame) {
      warnings.emplace_back(
          "ConSan MOI dynamic-stack scalar spill requires an established VGPR spill frame");
      return std::nullopt;
    }
    auto sequence = build_dynamic_stack_sgpr_spill_sequence(
        *point.moi_exec_save_sgpr, count, resources.base, *vgpr_spill,
        static_cast<uint32_t>(resources.count) * SpillManager::kSlotBytes, arch);
    if (!sequence)
      warnings.emplace_back("ConSan MOI could not encode the dynamic-stack scalar spill sequence");
    return sequence;
  }
  (void)managers;
  (void)private_layout_base;
  const uint32_t reservoir_end = static_cast<uint32_t>(resources.base) + resources.count + 1u;
  if (reservoir_end > REGISTER_SET_MAX_VGPRS || resources.required_vgpr_count < reservoir_end ||
      (vgpr_spill && (vgpr_spill->vgpr_base != resources.base ||
                      vgpr_spill->vgpr_count != resources.count + 1u))) {
    warnings.emplace_back(
        "ConSan MOI fixed-stack scalar spill has no planned lane-backed VGPR reservoir");
    return std::nullopt;
  }
  const uint16_t reservoir_vgpr = static_cast<uint16_t>(reservoir_end - 1u);
  const uint32_t total_private_bytes =
      vgpr_spill ? vgpr_spill->total_private_bytes : resources.original_private_segment_size;
  auto sequence = build_lane_sgpr_spill_sequence(*point.moi_exec_save_sgpr, count, reservoir_vgpr,
                                                 total_private_bytes, arch);
  if (!sequence) {
    warnings.emplace_back("ConSan MOI could not encode the lane-backed scalar spill sequence");
    return std::nullopt;
  }
  return sequence;
}

} // namespace rocjitsu::consan_moi_impl

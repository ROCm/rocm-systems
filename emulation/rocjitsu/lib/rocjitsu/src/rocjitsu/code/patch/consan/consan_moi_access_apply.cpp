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
  if (plan.body_size != 0u &&
      (body.size() + options.deferred_guest_word_count) * sizeof(uint32_t) != plan.body_size) {
    errors.emplace_back(
        "ConSan MOI " + std::string(probe_name) + " body size changed after placement: emitted=" +
        std::to_string((body.size() + options.deferred_guest_word_count) * sizeof(uint32_t)) +
        " planned=" + std::to_string(plan.body_size));
    return std::nullopt;
  }
  return body;
}

[[nodiscard]] bool
moi_scalar_spill_requires_dynamic_vgpr_frame(const ProgramInventory &inventory,
                                             const ResolvedMoiScratchPlan &resources,
                                             const ConSanMoiOperatingPoint &point) {
  if (!point.has_moi_scalar_spill()) {
    return false;
  }
  return std::ranges::any_of(resources.owner_descriptor_file_offsets, [&](uint64_t offset) {
    const ConSanProgramContainer *kernel = inventory.find_kernel_by_descriptor(offset);
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

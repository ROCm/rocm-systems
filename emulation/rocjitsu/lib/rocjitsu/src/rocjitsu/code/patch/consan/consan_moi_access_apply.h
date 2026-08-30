// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_access_apply.h
/// @brief Shared byte-application transaction for MOI access engines.

#pragma once

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/consan/consan_branch_only_relay_router.h"
#include "rocjitsu/code/patch/consan/consan_descriptor.h"
#include "rocjitsu/code/patch/consan/consan_moi_common_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_internal.h"
#include "rocjitsu/code/patch/consan/consan_moi_native_abi.h"
#include "rocjitsu/code/patch/consan/consan_moi_placement_contracts.h"
#include "rocjitsu/code/patch/consan/consan_moi_relocation.h"
#include "rocjitsu/code/patch/consan/consan_moi_shared_lowering.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocjitsu::consan_moi_impl {

using consan_moi_detail::append_moi_scc_preserving_indirect_jump;
using consan_moi_detail::append_word_bytes;
using consan_moi_detail::append_words_bytes;

[[nodiscard]] std::optional<ConSanCommittedLowering>
make_moi_access_lowering_commit(const ConSanTransformArtifacts &result,
                                const ConSanMoiCandidate &candidate,
                                const ConSanPatchLoweringProduct &patch);

template <typename PlannedPatch, typename BuildWords, typename MakePatchInfo,
          typename ApplyExtraRequirements>
[[nodiscard]] bool apply_inline_moi_access_patches(
    std::span<const uint8_t> bytes, const std::vector<PlannedPatch> &planned_patches,
    const MoiDescriptorVgprRequirements &descriptor_requirements,
    const MoiDescriptorSgprRequirements &scalar_requirements, std::string_view probe_name,
    rj_code_arch_t arch, BuildWords build_words, MakePatchInfo make_patch_info,
    ApplyExtraRequirements apply_extra_requirements, ConSanTransformArtifacts &result) {
  result.replacement.assign(bytes.begin(), bytes.end());
  for (const PlannedPatch &planned_patch : planned_patches) {
    const ConSanMoiCandidate &candidate = *planned_patch.candidate;
    auto words = build_words(planned_patch);
    if (!words) {
      result.replacement.clear();
      return false;
    }
    const uint64_t patch_bytes = static_cast<uint64_t>(words->size() * sizeof(uint32_t));
    if (patch_bytes != planned_patch.placement.body_size) {
      result.errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                                 " final patch size changed");
      result.replacement.clear();
      return false;
    }
    if (candidate.file_offset > result.replacement.size() ||
        patch_bytes > result.replacement.size() - candidate.file_offset) {
      result.errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                                 " inline patch exceeds the code object");
      result.replacement.clear();
      return false;
    }
    std::memcpy(result.replacement.data() + candidate.file_offset, words->data(),
                static_cast<size_t>(patch_bytes));
  }

  if (!apply_descriptor_requirements(result.replacement, result, descriptor_requirements, arch,
                                     result.errors) ||
      !apply_sgpr_descriptor_requirements(result.replacement, result, scalar_requirements,
                                          result.errors) ||
      !apply_extra_requirements(result.replacement, result)) {
    result.replacement.clear();
    return false;
  }

  std::vector<ConSanPatchInfo> patch_infos;
  std::vector<ConSanCommittedLowering> lowering_commits;
  patch_infos.reserve(planned_patches.size());
  lowering_commits.reserve(planned_patches.size());
  for (const PlannedPatch &planned_patch : planned_patches) {
    ConSanPatchInfo patch = make_patch_info(planned_patch);
    auto commit = make_moi_access_lowering_commit(result, *planned_patch.candidate, patch);
    if (!commit) {
      result.errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                                 " produced an invalid intent-bound lowering");
      result.replacement.clear();
      return false;
    }
    patch_infos.push_back(std::move(patch));
    lowering_commits.push_back(std::move(*commit));
  }
  if (!result.publish_lowering_commits(std::move(lowering_commits))) {
    result.errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                               " could not commit its semantic lowerings");
    result.replacement.clear();
    return false;
  }
  result.patches.insert(result.patches.end(), std::make_move_iterator(patch_infos.begin()),
                        std::make_move_iterator(patch_infos.end()));
  result.mark_modified();
  return true;
}

[[nodiscard]] bool apply_moi_appended_access_descriptor_requirements(
    CodeObjectPatcher &patcher, const AmdGpuCodeObject &code_object, std::span<const uint8_t> bytes,
    const RuntimeCapabilities &capabilities,
    const MoiDescriptorVgprRequirements &descriptor_requirements,
    const MoiDescriptorSgprRequirements &scalar_requirements,
    const MoiDescriptorPrivateRequirements &private_requirements,
    const MoiDescriptorLdsRequirements *lds_requirements, rj_code_arch_t arch,
    ConSanTransformArtifacts &result);

[[nodiscard]] bool initialize_moi_appended_access_text(
    CodeObjectPatcher &patcher, bool use_indirect_appended, uint64_t indirect_body_begin,
    const BranchOnlyDirectRelayReservoirSet &direct_reservoirs, std::string_view probe_name,
    rj_code_arch_t arch, std::vector<uint8_t> &new_text, std::vector<ConSanPatchInfo> &patches,
    ConSanTransformArtifacts &result);

/// Group planned access patches by the dense dispatcher selected during
/// placement. The map preserves dispatcher order so generated bodies retain
/// the offsets proved by planning.
template <typename PlannedPatch>
[[nodiscard]] std::map<uint64_t, std::vector<const PlannedPatch *>>
group_moi_dense_access_patches(const std::vector<PlannedPatch> &planned_patches) {
  std::map<uint64_t, std::vector<const PlannedPatch *>> groups;
  for (const PlannedPatch &planned_patch : planned_patches) {
    if (planned_patch.dense_call_anchor)
      groups[*planned_patch.dense_dispatcher_offset].push_back(&planned_patch);
  }
  return groups;
}

/// Emit the shared dense access-dispatcher mechanics for one MOI engine.
///
/// `eligible` retains the few engine-specific ABI preconditions that are not
/// implied by the common patch plan. It observes the representative patch and
/// its owner-bound operating point but does not perform placement or native
/// emission.
template <typename PlannedPatch, typename Eligible>
[[nodiscard]] bool emit_moi_dense_access_groups(
    const std::map<uint64_t, std::vector<const PlannedPatch *>> &groups,
    const std::map<uint64_t, MoiDenseEntryHost> &dense_entry_hosts, const ConSanRequest &request,
    const ConSanMoiOperatingPoint &operating_point, uint64_t access_slot_words,
    bool collapse_spill_router_for_explicit_key, std::string_view probe_name, rj_code_arch_t arch,
    std::vector<uint8_t> &new_text, std::vector<ConSanPatchInfo> &patches,
    std::vector<std::string> &errors, Eligible eligible) {
  for (const auto &[dispatcher_offset, group] : groups) {
    if (group.empty() || !group.front()->entry_island_offset)
      continue;
    ConSanMoiOperatingPoint group_point = operating_point;
    if (!apply_moi_transient_sgpr_assignment(
            request, group_point, operating_point,
            group.front()->resources.owner_descriptor_file_offsets) ||
        !eligible(*group.front(), group_point)) {
      continue;
    }
    std::vector<const MoiPlannedAccessPatch *> common_group(group.begin(), group.end());
    if (!emit_moi_dense_access_group(common_group, dispatcher_offset, dense_entry_hosts, request,
                                     group_point, arch, access_slot_words,
                                     collapse_spill_router_for_explicit_key, probe_name, new_text,
                                     patches, errors)) {
      return false;
    }
  }
  return true;
}

/// Emit the relocated entry-host bodies shared by dense MOI access routers.
///
/// Every engine restores the displaced host prefix and returns through the
/// same owner-bound indirect-jump ABI. `eligible` retains any stronger flavor
/// precondition, while `finalize_host_words` adds only a genuinely different
/// suffix or fixed-size check before the common byte publication. InlineShadow
/// keeps SCC in the indirect-jump assignment itself; the other engines use the
/// shared special-state assignment. `use_indirect_jump_scc_save` states that
/// remaining semantic distinction without duplicating the emission loop.
template <typename PlannedPatch, typename Eligible, typename FinalizeHostWords>
[[nodiscard]] bool emit_moi_dense_entry_hosts(
    const std::map<uint64_t, MoiDenseEntryHost> &dense_entry_hosts,
    const std::map<uint64_t, std::vector<const PlannedPatch *>> &groups,
    const ConSanRequest &request, const ConSanMoiOperatingPoint &operating_point,
    uint32_t indirect_island_words, bool pad_to_planned_offset, bool use_indirect_jump_scc_save,
    std::string_view probe_name, std::string_view lost_assignment_error,
    std::string_view return_error, rj_code_arch_t arch, std::vector<uint8_t> &new_text,
    std::vector<std::string> &errors, Eligible eligible, FinalizeHostWords finalize_host_words) {
  for (const auto &[dispatcher, host] : dense_entry_hosts) {
    if (pad_to_planned_offset) {
      while (new_text.size() < host.body_offset)
        append_word_bytes(new_text, build_s_nop(0u, arch));
    }
    if (new_text.size() != host.body_offset) {
      errors.emplace_back("ConSan MOI " + std::string(probe_name) +
                          " emitted a stale dense entry-host mapping");
      return false;
    }
    const auto group = groups.find(dispatcher);
    if (group == groups.end()) {
      const size_t reserved_words = host.displaced_words.size() + indirect_island_words;
      for (size_t word = 0; word < reserved_words; ++word)
        append_word_bytes(new_text, build_s_nop(0, arch));
      continue;
    }
    const PlannedPatch &representative = *group->second.front();
    ConSanMoiOperatingPoint group_point = operating_point;
    if (!apply_moi_transient_sgpr_assignment(
            request, group_point, operating_point,
            representative.resources.owner_descriptor_file_offsets) ||
        !eligible(representative, group_point)) {
      errors.emplace_back(std::string(lost_assignment_error));
      return false;
    }
    const auto jump_sgprs = moi_indirect_jump_sgprs(request, group_point);
    const auto special_state = moi_special_state_sgprs(request, group_point);
    std::vector<uint32_t> host_words = host.displaced_words;
    if (!jump_sgprs || (!use_indirect_jump_scc_save && !special_state) ||
        !append_moi_scc_preserving_indirect_jump(
            host_words, host.body_offset,
            host.host_offset + host.displaced_words.size() * sizeof(uint32_t), jump_sgprs->pc_sgpr,
            use_indirect_jump_scc_save ? jump_sgprs->scc_save_sgpr : special_state->scc_save_sgpr,
            /*capture_scc=*/true, arch) ||
        !finalize_host_words(host, host_words)) {
      errors.emplace_back(std::string(return_error));
      return false;
    }
    append_words_bytes(new_text, host_words);
  }
  return true;
}

struct MoiAppendedAnchorPlan {
  uint64_t candidate_anchor = 0;
  uint64_t placement_anchor = 0;
  uint32_t original_size = 0;
  bool entry_island_at_anchor = false;
};

[[nodiscard]] bool write_moi_appended_anchor(std::vector<uint8_t> &text,
                                             const MoiAppendedAnchorPlan &plan,
                                             uint64_t entry_target, rj_code_arch_t arch,
                                             std::string_view probe_name,
                                             std::vector<std::string> &errors);

template <typename PlannedPatch>
[[nodiscard]] bool
write_moi_appended_anchor(std::vector<uint8_t> &text, const PlannedPatch &planned_patch,
                          uint64_t entry_target, rj_code_arch_t arch, std::string_view probe_name,
                          std::vector<std::string> &errors) {
  return write_moi_appended_anchor(
      text,
      MoiAppendedAnchorPlan{
          .candidate_anchor = planned_patch.candidate->anchor(),
          .placement_anchor = planned_patch.placement.anchor_offset,
          .original_size = planned_patch.placement.original_size,
          .entry_island_at_anchor = planned_patch.entry_island_at_anchor,
      },
      entry_target, arch, probe_name, errors);
}

struct MoiEntryIslandPlan {
  uint64_t candidate_anchor = 0;
  std::optional<uint64_t> island_offset;
  bool entry_island_at_anchor = false;
  std::span<const uint64_t> owner_descriptor_file_offsets;
};

[[nodiscard]] bool write_moi_entry_island(std::vector<uint8_t> &text,
                                          const MoiEntryIslandPlan &plan,
                                          std::span<const uint32_t> island_words,
                                          std::vector<ConSanPatchInfo> &patches,
                                          std::string_view probe_name,
                                          std::vector<std::string> &errors);

template <typename PlannedPatch>
[[nodiscard]] bool
write_moi_entry_island(std::vector<uint8_t> &text, const PlannedPatch &planned_patch,
                       std::span<const uint32_t> island_words,
                       std::vector<ConSanPatchInfo> &patches, std::string_view probe_name,
                       std::vector<std::string> &errors) {
  return write_moi_entry_island(
      text,
      MoiEntryIslandPlan{
          .candidate_anchor = planned_patch.candidate->anchor(),
          .island_offset = planned_patch.entry_island_offset,
          .entry_island_at_anchor = planned_patch.entry_island_at_anchor,
          .owner_descriptor_file_offsets = planned_patch.resources.owner_descriptor_file_offsets,
      },
      island_words, patches, probe_name, errors);
}

/// Complete target-state and composition contract for one relocated MOI
/// access body.
///
/// The access planners decide guest placement, preservation, and target-state
/// requirements before calling `assemble_moi_appended_body`. This value then
/// gives the common assembler everything needed to establish the probe's
/// required register-bank state, preserve scalar/vector scratch, place the
/// relocated guest instruction, and restore the incoming state. It contains no
/// engine evidence policy and owns no storage; every span and output pointer is
/// valid only for the duration of the assembly call.
struct MoiAppendedBodyOptions {
  explicit MoiAppendedBodyOptions(rj_code_arch_t arch) : arch(arch) {}

  /// Target whose native instruction builders assemble this body.
  rj_code_arch_t arch;
  /// Guest words emitted after the assembled body by its caller.
  size_t deferred_guest_word_count = 0u;
  /// Scalar preservation sequences surrounding the probe body.
  std::span<const uint32_t> scalar_save_words = {};
  std::span<const uint32_t> scalar_restore_words = {};
  /// Selectable-VGPR-bank state observed on entry. Absence and zero both need
  /// no transition; a nonzero value is valid only on a target whose profile
  /// advertises selectable banks.
  std::optional<uint16_t> incoming_vgpr_bank_mode = std::nullopt;
  /// Guest address operand to copy before selecting the low instrumentation
  /// bank, and the low-bank scratch register that receives the copy. They are
  /// either both present or both absent.
  std::optional<uint16_t> high_bank_address_source = std::nullopt;
  std::optional<uint16_t> high_bank_address_destination = std::nullopt;
  /// Probe suffix that must remain after vector-spill restoration.
  size_t trailing_guest_word_count = 0u;
  /// Already-planned entry gate or preservation words prepended to the body.
  std::span<const uint32_t> entry_prefix_words = {};
  /// Guest location within `probe_words`, if the probe embeds it.
  std::optional<uint32_t> probe_guest_instruction_offset = std::nullopt;
  std::optional<size_t> guest_instruction_word_count = std::nullopt;
  /// Receives the final byte offset of the relocated guest within the body.
  uint32_t *body_guest_instruction_offset = nullptr;
  /// Whether overlapping guest operands require restoring vector spill state
  /// before the trailing probe suffix rather than at the ordinary exit.
  bool restore_vgpr_spill_before_trailing_guest = false;
  /// Whether the guest instruction itself must run in the incoming selectable
  /// bank while the surrounding instrumentation runs in the low bank.
  bool wrap_embedded_guest_vgpr_bank = false;
};

[[nodiscard]] constexpr bool
moi_appended_body_uses_selectable_vgpr_bank(rj_code_arch_t arch,
                                            std::optional<uint16_t> incoming_vgpr_bank_mode) {
  return consan_arch_has_selectable_vgpr_bank(arch) && incoming_vgpr_bank_mode.value_or(0u) != 0u;
}

[[nodiscard]] constexpr bool
moi_appended_body_vgpr_bank_mode_is_valid(rj_code_arch_t arch,
                                          std::optional<uint16_t> incoming_vgpr_bank_mode) {
  return incoming_vgpr_bank_mode.value_or(0u) == 0u || consan_arch_has_selectable_vgpr_bank(arch);
}

[[nodiscard]] constexpr size_t
moi_appended_body_vgpr_bank_transition_word_count(rj_code_arch_t arch,
                                                  std::optional<uint16_t> incoming_vgpr_bank_mode,
                                                  bool has_vgpr_spill, bool wraps_embedded_guest) {
  if (!moi_appended_body_uses_selectable_vgpr_bank(arch, incoming_vgpr_bank_mode))
    return 0u;
  return (has_vgpr_spill ? 4u : 2u) + (wraps_embedded_guest ? 2u : 0u);
}

[[nodiscard]] constexpr bool
moi_embedded_guest_vgpr_bank_plan_is_valid(bool wraps_embedded_guest, bool selects_low_vgpr_bank,
                                           bool has_guest_instruction,
                                           size_t trailing_guest_word_count) {
  return !wraps_embedded_guest ||
         (selects_low_vgpr_bank && has_guest_instruction && trailing_guest_word_count == 0u);
}

struct MoiAppendedBodyPatchPlan {
  const VgprSpillSequence *spill = nullptr;
  std::span<const uint32_t> displaced_tail_words;
  uint64_t body_size = 0;
  uint32_t guest_instruction_size = 0;
};

[[nodiscard]] std::optional<std::vector<uint32_t>>
assemble_moi_appended_body(const MoiAppendedBodyPatchPlan &plan,
                           std::span<const uint32_t> probe_words, std::string_view probe_name,
                           std::vector<std::string> &errors, const MoiAppendedBodyOptions &options);

template <typename PlannedPatch>
[[nodiscard]] std::optional<std::vector<uint32_t>>
assemble_moi_appended_body(const PlannedPatch &planned_patch, std::span<const uint32_t> probe_words,
                           std::string_view probe_name, std::vector<std::string> &errors,
                           const MoiAppendedBodyOptions &options) {
  return assemble_moi_appended_body(
      MoiAppendedBodyPatchPlan{
          .spill = planned_patch.spill ? &*planned_patch.spill : nullptr,
          .displaced_tail_words = planned_patch.displaced_tail_words,
          .body_size = planned_patch.placement.body_size,
          .guest_instruction_size = planned_patch.candidate->size(),
      },
      probe_words, probe_name, errors, options);
}

[[nodiscard]] bool
moi_scalar_spill_requires_dynamic_vgpr_frame(const ProgramInventory &inventory,
                                             const ResolvedMoiScratchPlan &resources,
                                             const ConSanMoiOperatingPoint &point);

[[nodiscard]] std::optional<SgprSpillSequence> build_moi_sgpr_spill_sequence(
    const ProgramInventory &inventory, const ResolvedMoiScratchPlan &resources,
    const ConSanRequest &request, const BoundRuntimeResources &bound_resources,
    const ConSanMoiOperatingPoint &point, MoiSpillManagers &managers, rj_code_arch_t arch,
    std::vector<std::string> &warnings, std::optional<uint32_t> private_layout_base,
    const VgprSpillSequence *vgpr_spill = nullptr);

} // namespace rocjitsu::consan_moi_impl

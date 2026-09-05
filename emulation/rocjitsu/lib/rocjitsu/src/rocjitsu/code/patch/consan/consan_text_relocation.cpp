// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_text_relocation.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/dbt/binary_translator.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/consan/consan_capability_contract.h"
#include "rocjitsu/code/patch/consan/consan_descriptor_growth.h"
#include "rocjitsu/code/patch/consan/consan_growth_policy.h"
#include "rocjitsu/code/patch/consan/consan_placement.h"
#include "rocjitsu/code/patch/trampoline_builder.h"

#include <algorithm>
#include <cstring>
#include <iterator>
#include <limits>
#include <string>

namespace rocjitsu {

namespace {

struct ConSanRelocatedText {
  std::vector<uint8_t> image;
  std::vector<TranslatedTextPlacement> placements;
  std::vector<ClientTextMarkerPlacement> marker_placements;
  uint64_t text_size = 0;
  std::optional<ConSanTextRelocationProof> relocation;
};

enum class FragmentMarkerRole : uint64_t {
  Begin = 1u,
  Guest = 2u,
  End = 3u,
};

[[nodiscard]] constexpr uint64_t fragment_marker_id(uint64_t fragment_id, FragmentMarkerRole role) {
  return (fragment_id << 2u) | static_cast<uint64_t>(role);
}

[[nodiscard]] bool fragment_applies_to_owner(const ConSanTextFragment &fragment,
                                             const ProgramInventory &inventory,
                                             uint64_t owner_source_entry) {
  if (fragment.patch.owner_descriptor_file_offsets.empty())
    return true;
  return std::ranges::any_of(fragment.patch.owner_descriptor_file_offsets,
                             [&](uint64_t descriptor_file_offset) {
                               const ConSanProgramContainer *owner =
                                   inventory.find_kernel_by_descriptor(descriptor_file_offset);
                               return owner != nullptr && owner->has_text_range &&
                                      owner->entry_text_offset == owner_source_entry;
                             });
}

[[nodiscard]] std::optional<InstructionRewrite> compose_consan_text_fragments(
    std::span<const ConSanTextFragment> fragments, const InstructionRewriteContext &context,
    const ProgramInventory &inventory, std::span<const uint8_t> source_text, rj_code_arch_t arch,
    std::vector<std::string> &errors) {
  std::vector<const ConSanTextFragment *> applicable;
  for (const ConSanTextFragment &fragment : fragments) {
    if (fragment.kind != ConSanTextFragmentKind::EntryPrefix &&
        fragment.patch.anchor_offset == context.source_offset &&
        fragment_applies_to_owner(fragment, inventory, context.owner_entry_source_offset)) {
      applicable.push_back(&fragment);
    }
  }
  if (applicable.empty())
    return std::nullopt;

  const uint32_t source_size = applicable.front()->patch.original_size;
  if (source_size == 0u || source_size % sizeof(uint32_t) != 0u ||
      context.source_offset > source_text.size() ||
      source_size > source_text.size() - context.source_offset ||
      std::ranges::any_of(applicable, [&](const ConSanTextFragment *fragment) {
        return fragment->id == 0u || fragment->patch.original_size != source_size;
      })) {
    errors.emplace_back("ConSan text fragments disagree on their bounded source span");
    return std::nullopt;
  }

  const ConSanTextFragment *replacement = nullptr;
  for (const ConSanTextFragment *fragment : applicable) {
    const bool invalid_shape =
        (fragment->kind == ConSanTextFragmentKind::Prefix &&
         (!fragment->replacement_words.empty() || !fragment->after_words.empty())) ||
        (fragment->kind == ConSanTextFragmentKind::Replacement &&
         (!fragment->before_words.empty() || !fragment->after_words.empty() ||
          fragment->replacement_words.empty()));
    const bool supplies_replacement =
        fragment->kind == ConSanTextFragmentKind::Replacement ||
        (fragment->kind == ConSanTextFragmentKind::Around && !fragment->replacement_words.empty());
    if (invalid_shape || (supplies_replacement && replacement != nullptr)) {
      errors.emplace_back("ConSan text fragments have an invalid composition shape");
      return std::nullopt;
    }
    if (supplies_replacement)
      replacement = fragment;
  }

  InstructionRewrite rewrite;
  rewrite.source_size = source_size;
  std::vector<uint32_t> composed;
  const auto add_marker = [&](const ConSanTextFragment &fragment, FragmentMarkerRole role,
                              uint64_t byte_offset) {
    rewrite.markers.push_back({.id = fragment_marker_id(fragment.id, role),
                               .byte_offset = static_cast<uint32_t>(byte_offset)});
  };

  for (const ConSanTextFragment *fragment : applicable) {
    if (fragment->kind != ConSanTextFragmentKind::Prefix)
      continue;
    add_marker(*fragment, FragmentMarkerRole::Begin, composed.size() * sizeof(uint32_t));
    composed.insert(composed.end(), fragment->before_words.begin(), fragment->before_words.end());
    add_marker(*fragment, FragmentMarkerRole::End, composed.size() * sizeof(uint32_t));
  }

  std::vector<const ConSanTextFragment *> around;
  for (const ConSanTextFragment *fragment : applicable) {
    if (fragment->kind != ConSanTextFragmentKind::Around)
      continue;
    const uint64_t begin = composed.size() * sizeof(uint32_t);
    add_marker(*fragment, FragmentMarkerRole::Begin, begin);
    composed.insert(composed.end(), fragment->before_words.begin(), fragment->before_words.end());
    around.push_back(fragment);
  }

  const uint64_t replacement_begin = composed.size() * sizeof(uint32_t);
  uint64_t guest_offset = replacement_begin;
  if (replacement != nullptr) {
    if (replacement->kind == ConSanTextFragmentKind::Replacement)
      add_marker(*replacement, FragmentMarkerRole::Begin, replacement_begin);
    if (replacement->kind == ConSanTextFragmentKind::Replacement &&
        replacement->patch.relocated_guest_instruction_offset) {
      if (*replacement->patch.relocated_guest_instruction_offset >
          replacement->replacement_words.size() * sizeof(uint32_t)) {
        errors.emplace_back("ConSan replacement fragment has an invalid guest marker");
        return std::nullopt;
      }
      guest_offset += *replacement->patch.relocated_guest_instruction_offset;
    }
    composed.insert(composed.end(), replacement->replacement_words.begin(),
                    replacement->replacement_words.end());
    if (replacement->kind == ConSanTextFragmentKind::Replacement) {
      add_marker(*replacement, FragmentMarkerRole::Guest, guest_offset);
      add_marker(*replacement, FragmentMarkerRole::End, composed.size() * sizeof(uint32_t));
    }
  } else {
    const size_t begin = static_cast<size_t>(context.source_offset);
    const size_t end = begin + source_size;
    const size_t old_size = composed.size();
    composed.resize(old_size + source_size / sizeof(uint32_t));
    std::memcpy(composed.data() + old_size, source_text.data() + begin, end - begin);
  }

  for (const ConSanTextFragment *fragment : around)
    add_marker(*fragment, FragmentMarkerRole::Guest, guest_offset);
  for (auto it = around.rbegin(); it != around.rend(); ++it) {
    composed.insert(composed.end(), (*it)->after_words.begin(), (*it)->after_words.end());
    add_marker(**it, FragmentMarkerRole::End, composed.size() * sizeof(uint32_t));
  }
  const bool needs_bypass = std::ranges::any_of(applicable, [](const ConSanTextFragment *fragment) {
    return std::ranges::any_of(fragment->branch_fixups, [](const auto &fixup) {
      return fixup.target == ConSanTextFragmentBranchTarget::Bypass;
    });
  });
  std::optional<uint64_t> bypass_offset;
  std::optional<uint64_t> normal_exit_branch_offset;
  if (needs_bypass) {
    normal_exit_branch_offset = composed.size() * sizeof(uint32_t);
    composed.push_back(0u);
    bypass_offset = composed.size() * sizeof(uint32_t);
    if (replacement != nullptr) {
      composed.insert(composed.end(), replacement->replacement_words.begin(),
                      replacement->replacement_words.end());
    } else {
      const size_t begin = static_cast<size_t>(context.source_offset);
      const size_t end = begin + source_size;
      const size_t old_size = composed.size();
      composed.resize(old_size + source_size / sizeof(uint32_t));
      std::memcpy(composed.data() + old_size, source_text.data() + begin, end - begin);
    }
  }
  const auto relative_marker = [&](const ConSanTextFragment &fragment, FragmentMarkerRole role) {
    return std::ranges::find_if(rewrite.markers, [&](const InstructionRewriteMarker &marker) {
      return marker.id == fragment_marker_id(fragment.id, role);
    });
  };
  for (const ConSanTextFragment *fragment : applicable) {
    const auto begin = relative_marker(*fragment, FragmentMarkerRole::Begin);
    for (const ConSanTextFragmentBranchFixup &fixup : fragment->branch_fixups) {
      const auto marker_target = fixup.target == ConSanTextFragmentBranchTarget::Guest
                                     ? relative_marker(*fragment, FragmentMarkerRole::Guest)
                                     : relative_marker(*fragment, FragmentMarkerRole::End);
      const std::optional<uint64_t> target_offset =
          fixup.target == ConSanTextFragmentBranchTarget::Bypass
              ? bypass_offset
              : (marker_target == rewrite.markers.end()
                     ? std::nullopt
                     : std::optional<uint64_t>{marker_target->byte_offset});
      if (begin == rewrite.markers.end() || !target_offset ||
          fixup.word_index >= fragment->before_words.size()) {
        errors.emplace_back("ConSan text fragment has an invalid symbolic branch");
        return std::nullopt;
      }
      const uint64_t source_offset = begin->byte_offset + fixup.word_index * sizeof(uint32_t);
      const auto branch = compute_sopp_branch_simm16(source_offset, *target_offset);
      if (!branch) {
        errors.emplace_back("ConSan text fragment symbolic branch is out of range");
        return std::nullopt;
      }
      uint32_t &encoded = composed[source_offset / sizeof(uint32_t)];
      encoded = encoded == 0u ? build_s_branch(*branch, arch)
                              : (encoded & 0xffff0000u) | static_cast<uint16_t>(*branch);
    }
  }
  if (normal_exit_branch_offset) {
    const auto branch =
        compute_sopp_branch_simm16(*normal_exit_branch_offset, composed.size() * sizeof(uint32_t));
    if (!branch) {
      errors.emplace_back("ConSan text fragment bypass exit is out of range");
      return std::nullopt;
    }
    composed[*normal_exit_branch_offset / sizeof(uint32_t)] = build_s_branch(*branch, arch);
  }
  rewrite.replacement_words = std::move(composed);
  return rewrite;
}

[[nodiscard]] bool supports_local_text_transaction(std::span<const uint8_t> image,
                                                   const AmdGpuCodeObject &source,
                                                   rj_code_arch_t arch,
                                                   std::span<const ConSanTextFragment> fragments,
                                                   bool allow_instrumented_replacements = false) {
  if (source.text_sections().size() != 1u)
    return false;
  const Section &text = *source.text_sections().front();
  return !fragments.empty() &&
         std::ranges::all_of(fragments, [&](const ConSanTextFragment &fragment) {
           const uint64_t replacement_size = fragment.replacement_words.size() * sizeof(uint32_t);
           const bool direct =
               fragment.kind == ConSanTextFragmentKind::Replacement &&
               fragment.before_words.empty() && fragment.after_words.empty() &&
               fragment.patch.original_size != 0u &&
               fragment.patch.original_size % sizeof(uint32_t) == 0u &&
               replacement_size >= fragment.patch.original_size &&
               fragment.patch.anchor_offset <= text.size() &&
               replacement_size <= text.size() - fragment.patch.anchor_offset &&
               count_nop_padding(
                   image,
                   text.sectionOffset() + fragment.patch.anchor_offset +
                       fragment.patch.original_size,
                   arch,
                   static_cast<uint32_t>((replacement_size - fragment.patch.original_size) /
                                         sizeof(uint32_t))) ==
                   (replacement_size - fragment.patch.original_size) / sizeof(uint32_t);
           const bool around = fragment.kind == ConSanTextFragmentKind::Around &&
                               fragment.replacement_words.empty() &&
                               (!fragment.before_words.empty() || !fragment.after_words.empty());
           return fragment.branch_fixups.empty() &&
                  ((direct && (fragment.intent_ids.empty() || allow_instrumented_replacements)) ||
                   (around && fragment.intent_ids.empty() &&
                    fragment.patch.owner_descriptor_file_offsets.size() <= 1u &&
                    fragment.patch.original_size != 0u));
         });
}

/// Realize small validation and scheduling edits without relocating the
/// containing functions. The fragment vocabulary and publication path remain
/// shared with whole-text instrumentation; this backend preserves the bounded
/// direct-branch geometry required by fault and perturbation validation.
[[nodiscard]] std::optional<ConSanRelocatedText>
rewrite_consan_text_locally(const AmdGpuCodeObject &source, rj_code_arch_t arch,
                            ConSanPatchedImageGrowthLimit growth_limit, std::string_view operation,
                            std::span<const ConSanTextFragment> fragments,
                            ConSanTransformArtifacts &result) {
  const Section &text = *source.text_sections().front();
  const std::span<const uint8_t> source_text(reinterpret_cast<const uint8_t *>(text.data()),
                                             text.size());
  DbiPatchPlacementPlanner planner(arch, text.size());
  for (const ConSanPatchInfo &patch : result.patches) {
    if ((patch.original_size != 0u &&
         !planner.reserve_existing_range(patch.anchor_offset, patch.original_size)) ||
        (patch.trampoline_size != 0u &&
         !planner.reserve_existing_range(patch.trampoline_offset, patch.trampoline_size))) {
      result.errors.emplace_back("ConSan local text transaction overlaps an existing patch");
      return std::nullopt;
    }
  }

  struct PlannedFragment {
    const ConSanTextFragment *fragment = nullptr;
    std::optional<DbiPatchPlacement> placement;
  };
  std::vector<PlannedFragment> planned;
  const std::vector<LocalNopCave> caves =
      find_uncovered_nop_caves(source, result.program_inventory, arch);
  for (const ConSanTextFragment &fragment : fragments) {
    if (fragment.kind == ConSanTextFragmentKind::Replacement) {
      if (!planner.reserve_existing_range(fragment.patch.anchor_offset,
                                          fragment.replacement_words.size() * sizeof(uint32_t))) {
        result.errors.emplace_back("ConSan local text replacements overlap");
        return std::nullopt;
      }
      planned.push_back({.fragment = &fragment, .placement = std::nullopt});
      continue;
    }
    DbiPatchPlacementRequest request{
        .anchor_offset = fragment.patch.anchor_offset,
        .original_size = fragment.patch.original_size,
        .body_size =
            (fragment.before_words.size() + fragment.patch.original_size / sizeof(uint32_t) +
             fragment.after_words.size()) *
            sizeof(uint32_t),
        .inline_capacity = 0u,
        .local_cave = std::nullopt,
        .allow_appended_cave = true,
    };
    std::optional<DbiPatchPlacement> placement;
    for (const LocalNopCave &cave : caves) {
      const uint64_t cave_end =
          cave.text_offset + static_cast<uint64_t>(cave.word_count) * sizeof(uint32_t);
      for (uint64_t offset = cave.text_offset;
           offset <= cave_end && request.body_size + sizeof(uint32_t) <= cave_end - offset;
           offset += sizeof(uint32_t)) {
        request.local_cave = DbiPatchLocalCave{offset, cave_end - offset};
        placement = planner.plan(request);
        if (placement)
          break;
      }
      if (placement)
        break;
    }
    if (!placement) {
      request.local_cave.reset();
      std::string placement_error;
      placement = planner.plan(request, &placement_error);
      if (!placement) {
        result.outcome = ConSanTransformOutcome::Unsupported;
        result.warnings.emplace_back(
            "ConSan " + std::string(operation) +
            " has no reachable local or appended cave: " + placement_error);
        return std::nullopt;
      }
    }
    planned.push_back({.fragment = &fragment, .placement = placement});
  }

  std::vector<uint8_t> new_text(source_text.begin(), source_text.end());
  ConSanRelocatedText rewritten;
  for (const PlannedFragment &item : planned) {
    const ConSanTextFragment &fragment = *item.fragment;
    const uint64_t owner = fragment.patch.owner_descriptor_file_offsets.empty()
                               ? 0u
                               : fragment.patch.owner_descriptor_file_offsets.front();
    if (!item.placement) {
      const uint64_t replacement_size = fragment.replacement_words.size() * sizeof(uint32_t);
      std::memcpy(new_text.data() + fragment.patch.anchor_offset, fragment.replacement_words.data(),
                  replacement_size);
      rewritten.placements.push_back({.source_offset = fragment.patch.anchor_offset,
                                      .target_offset = fragment.patch.anchor_offset,
                                      .owner_descriptor_file_offset = owner,
                                      .client_rewrite = true,
                                      .client_rewrite_source_size = fragment.patch.original_size});
      continue;
    }
    const DbiPatchPlacement &placement = *item.placement;
    TrampolinePlan trampoline;
    trampoline.arch = arch;
    trampoline.anchor_offset = placement.anchor_offset;
    trampoline.original_size = fragment.patch.original_size;
    trampoline.trampoline_offset = placement.body_offset;
    trampoline.return_target = placement.return_target;
    trampoline.original_words.resize(fragment.patch.original_size / sizeof(uint32_t));
    std::memcpy(trampoline.original_words.data(), source_text.data() + fragment.patch.anchor_offset,
                fragment.patch.original_size);
    trampoline.before_items = {InlineAsmItem{fragment.before_words}};
    trampoline.after_items = {InlineAsmItem{fragment.after_words}};
    std::string trampoline_error;
    auto emitted = TrampolineBuilder::build(trampoline, &trampoline_error);
    if (!emitted) {
      result.errors.emplace_back("ConSan local text transaction failed: " + trampoline_error);
      return std::nullopt;
    }
    std::memcpy(new_text.data() + placement.anchor_offset, emitted->patched_anchor_bytes.data(),
                emitted->patched_anchor_bytes.size());
    if (placement.kind == DbiPatchPlacementKind::AppendedCave) {
      if (new_text.size() != placement.body_offset) {
        result.errors.emplace_back("ConSan local text transaction lost its appended cursor");
        return std::nullopt;
      }
      append_consan_patch_words(new_text, emitted->trampoline_words);
    } else {
      std::memcpy(new_text.data() + placement.body_offset, emitted->trampoline_words.data(),
                  emitted->trampoline_words.size() * sizeof(uint32_t));
    }
    rewritten.placements.push_back({.source_offset = fragment.patch.anchor_offset,
                                    .target_offset = placement.body_offset,
                                    .owner_descriptor_file_offset = owner,
                                    .client_rewrite = true,
                                    .client_rewrite_source_size = fragment.patch.original_size});
    rewritten.marker_placements.push_back(
        {.id = fragment_marker_id(fragment.id, FragmentMarkerRole::Begin),
         .source_offset = fragment.patch.anchor_offset,
         .target_offset = placement.body_offset,
         .owner_descriptor_file_offset = owner});
    rewritten.marker_placements.push_back(
        {.id = fragment_marker_id(fragment.id, FragmentMarkerRole::Guest),
         .source_offset = fragment.patch.anchor_offset,
         .target_offset = placement.body_offset + fragment.before_words.size() * sizeof(uint32_t),
         .owner_descriptor_file_offset = owner});
    rewritten.marker_placements.push_back(
        {.id = fragment_marker_id(fragment.id, FragmentMarkerRole::End),
         .source_offset = fragment.patch.anchor_offset,
         .target_offset =
             placement.body_offset + emitted->trampoline_words.size() * sizeof(uint32_t),
         .owner_descriptor_file_offset = owner});
  }
  CodeObjectPatcher patcher(source);
  if (!replace_consan_text(patcher, new_text, growth_limit, operation,
                           result.program_inventory.code_object_id(), result.errors,
                           &result.transform_failure_cause))
    return std::nullopt;
  rewritten.image = std::move(patcher).emit();
  AmdGpuCodeObject output(rewritten.image.data(), rewritten.image.size());
  if (!output.is_valid() || output.text_sections().size() != 1u) {
    result.errors.emplace_back("ConSan local text transaction produced an invalid image");
    return std::nullopt;
  }
  rewritten.text_size = output.text_sections().front()->size();
  return rewritten;
}

} // namespace

std::optional<ConSanTextFragment> make_consan_around_text_fragment(
    std::vector<uint32_t> words, uint32_t guest_offset, uint32_t guest_size,
    std::span<const ConSanProbeIntentId> intent_ids, ConSanRuntimeStaticMapping runtime_mapping,
    ConSanPatchInfo patch, std::vector<std::string> &errors, std::string_view subject,
    std::optional<uint32_t> emitted_guest_size) {
  const uint64_t byte_size = words.size() * sizeof(uint32_t);
  const uint32_t emitted_size = emitted_guest_size.value_or(guest_size);
  if (guest_size == 0u || guest_size % sizeof(uint32_t) != 0u || emitted_size == 0u ||
      emitted_size % sizeof(uint32_t) != 0u || guest_offset % sizeof(uint32_t) != 0u ||
      guest_offset > byte_size || emitted_size > byte_size - guest_offset ||
      patch.original_size != guest_size) {
    errors.emplace_back("ConSan " + std::string(subject) +
                        " has an invalid source-fragment guest boundary");
    return std::nullopt;
  }
  const size_t guest_word = guest_offset / sizeof(uint32_t);
  const size_t guest_words = emitted_size / sizeof(uint32_t);
  patch.relocated_guest_instruction_offset = 0u;
  return ConSanTextFragment{
      .id = 0u,
      .kind = ConSanTextFragmentKind::Around,
      .intent_ids = std::vector<ConSanProbeIntentId>(intent_ids.begin(), intent_ids.end()),
      .runtime_mapping = std::move(runtime_mapping),
      .before_words = std::vector<uint32_t>(words.begin(), words.begin() + guest_word),
      .replacement_words = emitted_guest_size
                               ? std::vector<uint32_t>(words.begin() + guest_word,
                                                       words.begin() + guest_word + guest_words)
                               : std::vector<uint32_t>{},
      .after_words = std::vector<uint32_t>(words.begin() + guest_word + guest_words, words.end()),
      .branch_fixups = {},
      .sync_relocations = {},
      .patch = std::move(patch),
  };
}

bool stage_consan_text_rewrites(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                const ConSanDescriptorMutationBatch &descriptor_mutations,
                                const ConSanDescriptorMutationPolicy &descriptor_policy,
                                std::string_view subject, std::vector<ConSanTextFragment> fragments,
                                ConSanTransformArtifacts &result) {
  const auto source =
      result.replacement.empty()
          ? std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(code_object.image_data()),
                                     code_object.image_size())
          : std::span<const uint8_t>(result.replacement);
  std::vector<uint8_t> descriptor_image(source.begin(), source.end());
  if (!apply_consan_descriptor_mutations_to_bytes(descriptor_image, result.program_inventory,
                                                  descriptor_mutations, descriptor_policy, arch,
                                                  subject, result.errors))
    return false;

  result.replacement = std::move(descriptor_image);
  return stage_consan_text_fragments(std::move(fragments), result);
}

bool stage_consan_text_fragments(std::vector<ConSanTextFragment> fragments,
                                 ConSanTransformArtifacts &result) {
  append_consan_text_fragments(std::move(fragments), result.staged_text_fragments);
  result.mark_modified();
  return true;
}

void append_consan_text_fragments(std::vector<ConSanTextFragment> fragments,
                                  std::vector<ConSanTextFragment> &transaction) {
  uint64_t next_fragment_id = transaction.size() + 1u;
  for (ConSanTextFragment &fragment : fragments)
    fragment.id = next_fragment_id++;
  transaction.insert(transaction.end(), std::make_move_iterator(fragments.begin()),
                     std::make_move_iterator(fragments.end()));
}

namespace {

std::optional<ConSanRelocatedText>
relocate_consan_text(std::span<const uint8_t> descriptor_patched_image, rj_code_arch_t arch,
                     ConSanPatchedImageGrowthLimit growth_limit, std::string_view operation,
                     ConSanTransformArtifacts &result) {
  const auto &fragments = result.staged_text_fragments;
  const ConSanCodeObjectId &input_id = result.program_inventory.code_object_id();
  std::vector<std::string> &errors = result.errors;
  const std::string error_prefix = "ConSan " + std::string(operation);
  AmdGpuCodeObject source(descriptor_patched_image.data(), descriptor_patched_image.size());
  if (!source.is_valid() || source.text_sections().size() != 1u || !input_id.valid()) {
    errors.emplace_back(error_prefix +
                        " cannot prepare one executable text relocation transaction");
    return std::nullopt;
  }
  const uint64_t source_text_size = source.text_sections().front()->size();
  const std::span<const uint8_t> source_text(
      reinterpret_cast<const uint8_t *>(source.text_sections().front()->data()), source_text_size);
  if (supports_local_text_transaction(descriptor_patched_image, source, arch, fragments))
    return rewrite_consan_text_locally(source, arch, growth_limit, operation, fragments, result);
  BinaryTranslatorOptions translator_options;
  translator_options.preserve_source_text_prefix = true;
  translator_options.preserve_source_descriptor_resources = true;
  const auto &preapplied_code_ranges = result.program_inventory.preapplied_mutation().code_ranges;
  translator_options.source_text_code_ranges.reserve(
      source.functions().size() + preapplied_code_ranges.size() + result.patches.size());
  for (const AmdGpuFunctionInfo &function : source.functions()) {
    if (function.code_size != 0u) {
      translator_options.source_text_code_ranges.push_back(
          {.start_offset = function.entry_text_offset, .size = function.code_size});
    }
  }
  const bool has_preexisting_code =
      !preapplied_code_ranges.empty() ||
      std::ranges::any_of(result.patches,
                          [](const ConSanPatchInfo &patch) { return patch.trampoline_size != 0u; });
  for (const ConSanPreappliedCodeRange &range : preapplied_code_ranges) {
    translator_options.source_text_code_ranges.push_back(
        {.start_offset = range.text_offset, .size = range.size});
  }
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.trampoline_size == 0u)
      continue;
    translator_options.source_text_code_ranges.push_back(
        {.start_offset = patch.trampoline_offset, .size = patch.trampoline_size});
  }
  if (const ConSanTargetProfile *target = consan_target_profile(arch);
      target != nullptr &&
      target->identity_translation_revision != ProcessorRevision::Unspecified) {
    translator_options.input_revision = target->identity_translation_revision;
    translator_options.output_revision = target->identity_translation_revision;
  }
  if (descriptor_patched_image.size() < sizeof(Elf64_Ehdr)) {
    errors.emplace_back(error_prefix + " has no complete ELF header");
    return std::nullopt;
  }
  const auto *header = reinterpret_cast<const Elf64_Ehdr *>(descriptor_patched_image.data());
  BinaryTranslator translator(arch, arch, header->e_flags & EF_AMDGPU_MACH, translator_options);
  translator.set_kernel_entry_rewrite_callback([&](const KernelEntryRewriteContext &context)
                                                   -> std::optional<KernelEntryRewrite> {
    KernelEntryRewrite rewrite;
    for (const ConSanTextFragment &fragment : fragments) {
      if (fragment.kind != ConSanTextFragmentKind::EntryPrefix ||
          fragment.patch.anchor_offset != context.source_entry_offset ||
          !fragment_applies_to_owner(fragment, result.program_inventory,
                                     context.source_entry_offset)) {
        continue;
      }
      const uint32_t begin = static_cast<uint32_t>(rewrite.prefix_words.size() * sizeof(uint32_t));
      rewrite.markers.push_back(
          {.id = fragment_marker_id(fragment.id, FragmentMarkerRole::Begin), .byte_offset = begin});
      rewrite.prefix_words.insert(rewrite.prefix_words.end(), fragment.before_words.begin(),
                                  fragment.before_words.end());
      rewrite.markers.push_back(
          {.id = fragment_marker_id(fragment.id, FragmentMarkerRole::End),
           .byte_offset = static_cast<uint32_t>(rewrite.prefix_words.size() * sizeof(uint32_t))});
    }
    return rewrite.prefix_words.empty() ? std::nullopt
                                        : std::optional<KernelEntryRewrite>{std::move(rewrite)};
  });
  translator.set_instruction_rewrite_callback(
      [&](const InstructionRewriteContext &context) -> std::optional<InstructionRewrite> {
        return compose_consan_text_fragments(fragments, context, result.program_inventory,
                                             source_text, arch, errors);
      });
  TranslatedCodeObject translated = translator.translate(source);
  if (!translated.dispatchable() || !errors.empty()) {
    errors.emplace_back(error_prefix + " could not relocate executable text");
    for (const TranslationDiagnostic &diagnostic : translated.diagnostics) {
      if (diagnostic.severity == DiagnosticSeverity::Error)
        errors.emplace_back("ConSan text relocation: " + diagnostic.message);
    }
    return std::nullopt;
  }

  const size_t input_image_bytes = static_cast<size_t>(input_id.byte_size);
  const auto limit = consan_patched_image_growth_limit_bytes(growth_limit, input_image_bytes);
  const std::string policy =
      consan_patched_image_growth_policy_description(growth_limit, input_image_bytes);
  if (!limit) {
    errors.emplace_back(error_prefix + " has an invalid patched-image growth policy (" + policy +
                        ")");
    return std::nullopt;
  }
  const size_t required_growth = translated.elf_bytes.size() > input_image_bytes
                                     ? translated.elf_bytes.size() - input_image_bytes
                                     : 0u;
  if (required_growth > *limit) {
    if (!has_preexisting_code &&
        supports_local_text_transaction(descriptor_patched_image, source, arch, fragments,
                                        /*allow_instrumented_replacements=*/true)) {
      auto in_place =
          rewrite_consan_text_locally(source, arch, growth_limit, operation, fragments, result);
      const size_t in_place_growth = in_place && in_place->image.size() > input_image_bytes
                                         ? in_place->image.size() - input_image_bytes
                                         : 0u;
      if (in_place && in_place_growth <= *limit)
        return in_place;
    }
    result.transform_failure_cause = ConSanTransformFailureCause::PatchedImageGrowthLimit;
    errors.emplace_back(error_prefix + " rejected patched-image file growth: required total " +
                        std::to_string(required_growth) + " bytes, limit " +
                        std::to_string(*limit) + " bytes (policy " + policy + ")");
    return std::nullopt;
  }
  AmdGpuCodeObject output(translated.elf_bytes.data(), translated.elf_bytes.size());
  if (!output.is_valid() || output.text_sections().size() != 1u ||
      output.text_sections().front()->size() < source_text_size) {
    errors.emplace_back(error_prefix + " produced an invalid relocated executable image");
    return std::nullopt;
  }
  // Whole-text placement changes only executable entry coordinates. ConSan
  // has already applied its exact ABI/resource transaction to the input
  // descriptors, so retain those bytes rather than accepting DBT's general
  // cross-target descriptor normalization as a second owner of the ABI.
  for (const AmdGpuKernelInfo &source_kernel : source.kernels()) {
    const auto output_kernel =
        std::ranges::find(output.kernels(), source_kernel.name, &AmdGpuKernelInfo::name);
    if (output_kernel == output.kernels().end()) {
      errors.emplace_back(error_prefix + " lost a kernel while relocating executable text");
      return std::nullopt;
    }
    auto source_descriptor =
        read_kernel_descriptor(descriptor_patched_image, source_kernel.descriptor_file_offset);
    const auto output_descriptor = read_kernel_descriptor(
        std::span<const uint8_t>(translated.elf_bytes), output_kernel->descriptor_file_offset);
    if (!source_descriptor || !output_descriptor) {
      errors.emplace_back(error_prefix + " could not retain a relocated kernel descriptor");
      return std::nullopt;
    }
    source_descriptor->kernel_code_entry_byte_offset =
        output_descriptor->kernel_code_entry_byte_offset;
    if (!write_kernel_descriptor(std::span<uint8_t>(translated.elf_bytes),
                                 output_kernel->descriptor_file_offset, *source_descriptor)) {
      errors.emplace_back(error_prefix + " could not publish a relocated kernel descriptor");
      return std::nullopt;
    }
  }
  const auto canonical_descriptor_offset = [&](uint64_t active_descriptor_offset) {
    const auto active_kernel = std::ranges::find(source.kernels(), active_descriptor_offset,
                                                 &AmdGpuKernelInfo::descriptor_file_offset);
    if (active_kernel == source.kernels().end())
      return active_descriptor_offset;
    const ConSanProgramContainer *canonical =
        result.program_inventory.find_kernel_by_name(active_kernel->name);
    return canonical != nullptr ? canonical->descriptor_file_offset : active_descriptor_offset;
  };
  for (TranslatedTextPlacement &placement : translated.text_placements) {
    placement.owner_descriptor_file_offset =
        canonical_descriptor_offset(placement.owner_descriptor_file_offset);
  }
  for (ClientTextMarkerPlacement &placement : translated.client_marker_placements) {
    placement.owner_descriptor_file_offset =
        canonical_descriptor_offset(placement.owner_descriptor_file_offset);
  }
  return ConSanRelocatedText{
      .image = std::move(translated.elf_bytes),
      .placements = std::move(translated.text_placements),
      .marker_placements = std::move(translated.client_marker_placements),
      .text_size = output.text_sections().front()->size(),
      .relocation = ConSanTextRelocationProof{source_text_size},
  };
}

} // namespace

bool finalize_consan_text_rewrites(std::span<const uint8_t> descriptor_image, rj_code_arch_t arch,
                                   ConSanPatchedImageGrowthLimit growth_limit,
                                   std::string_view subject, ConSanTransformArtifacts &result) {
  if (result.staged_text_fragments.empty())
    return true;

  auto relocated = relocate_consan_text(descriptor_image, arch, growth_limit, subject, result);
  if (!relocated)
    return false;
  std::vector<ConSanPatchInfo> placed_patches;
  std::vector<ConSanCommittedLowering> commits;
  commits.reserve(result.staged_text_fragments.size());
  const auto marker_for = [&](const ConSanTextFragment &fragment, FragmentMarkerRole role,
                              uint64_t owner, uint64_t source_offset) {
    return std::ranges::find_if(relocated->marker_placements,
                                [&](const ClientTextMarkerPlacement &placement) {
                                  return placement.id == fragment_marker_id(fragment.id, role) &&
                                         placement.source_offset == source_offset &&
                                         placement.owner_descriptor_file_offset == owner;
                                });
  };
  for (ConSanTextFragment &staged : result.staged_text_fragments) {
    const bool in_place =
        !relocated->relocation && staged.kind == ConSanTextFragmentKind::Replacement;
    std::optional<ConSanPatchInfo> primary;
    std::vector<ConSanPatchInfo> fragment_placements;
    if (staged.kind == ConSanTextFragmentKind::EntryPrefix) {
      for (uint64_t owner : staged.patch.owner_descriptor_file_offsets) {
        const auto begin =
            marker_for(staged, FragmentMarkerRole::Begin, owner, staged.patch.anchor_offset);
        const auto end =
            marker_for(staged, FragmentMarkerRole::End, owner, staged.patch.anchor_offset);
        if (begin == relocated->marker_placements.end() ||
            end == relocated->marker_placements.end() ||
            end->target_offset < begin->target_offset) {
          continue;
        }
        ConSanPatchInfo placed = staged.patch;
        // Entry prefixes insert bytes before the source entry; they do not
        // consume or overwrite an original instruction range.
        placed.original_size = 0u;
        placed.trampoline_offset = begin->target_offset;
        placed.trampoline_size = static_cast<uint32_t>(end->target_offset - begin->target_offset);
        placed.dispatch_id_primary_prologue_offset = begin->target_offset;
        const auto secondary = std::ranges::find_if(
            relocated->marker_placements, [&](const ClientTextMarkerPlacement &placement) {
              return placement.id == fragment_marker_id(staged.id, FragmentMarkerRole::Begin) &&
                     placement.owner_descriptor_file_offset == owner &&
                     placement.source_offset != staged.patch.anchor_offset;
            });
        if (secondary != relocated->marker_placements.end()) {
          placed.dispatch_id_secondary_prologue_offset = secondary->target_offset;
          const auto secondary_end =
              marker_for(staged, FragmentMarkerRole::End, owner, secondary->source_offset);
          if (secondary_end == relocated->marker_placements.end() ||
              secondary_end->target_offset < secondary->target_offset ||
              secondary_end->target_offset < placed.trampoline_offset ||
              secondary_end->target_offset - placed.trampoline_offset >
                  std::numeric_limits<uint32_t>::max()) {
            result.errors.emplace_back("ConSan " + std::string(subject) +
                                       " lost a paired entry-prefix boundary");
            return false;
          }
          placed.trampoline_size =
              static_cast<uint32_t>(secondary_end->target_offset - placed.trampoline_offset);
        }
        if (placed.trampoline_offset > relocated->text_size ||
            placed.trampoline_size > relocated->text_size - placed.trampoline_offset) {
          result.errors.emplace_back("ConSan " + std::string(subject) +
                                     " entry-prefix placement exceeds relocated text");
          return false;
        }
        if (!primary)
          primary = placed;
        fragment_placements.push_back(placed);
        placed_patches.push_back(std::move(placed));
      }
    }
    for (const TranslatedTextPlacement &placement : relocated->placements) {
      if (staged.kind == ConSanTextFragmentKind::EntryPrefix)
        break;
      if (placement.source_offset != staged.patch.anchor_offset || !placement.client_rewrite)
        continue;
      ConSanPatchInfo placed = staged.patch;
      if (in_place) {
        placed.trampoline_offset = placement.target_offset;
        placed.original_size =
            static_cast<uint32_t>(staged.replacement_words.size() * sizeof(uint32_t));
        placed.trampoline_size = 0u;
        if (placed.relocated_guest_instruction_offset)
          *placed.relocated_guest_instruction_offset += placement.target_offset;
      } else {
        const auto begin =
            marker_for(staged, FragmentMarkerRole::Begin, placement.owner_descriptor_file_offset,
                       staged.patch.anchor_offset);
        const auto end =
            marker_for(staged, FragmentMarkerRole::End, placement.owner_descriptor_file_offset,
                       staged.patch.anchor_offset);
        if (begin == relocated->marker_placements.end() ||
            end == relocated->marker_placements.end())
          continue;
        if (end->target_offset < begin->target_offset) {
          result.errors.emplace_back("ConSan " + std::string(subject) +
                                     " lost a relocated fragment boundary");
          return false;
        }
        placed.trampoline_offset = begin->target_offset;
        placed.trampoline_size = static_cast<uint32_t>(end->target_offset - begin->target_offset);
        if (placed.relocated_guest_instruction_offset) {
          const auto guest =
              marker_for(staged, FragmentMarkerRole::Guest, placement.owner_descriptor_file_offset,
                         staged.patch.anchor_offset);
          if (guest == relocated->marker_placements.end()) {
            result.errors.emplace_back("ConSan " + std::string(subject) +
                                       " lost a relocated guest marker");
            return false;
          }
          placed.relocated_guest_instruction_offset = guest->target_offset;
        }
      }
      const uint64_t emitted_offset = in_place ? placed.anchor_offset : placed.trampoline_offset;
      const uint64_t emitted_size = in_place ? placed.original_size : placed.trampoline_size;
      if (emitted_offset > relocated->text_size ||
          emitted_size > relocated->text_size - emitted_offset) {
        result.errors.emplace_back(
            "ConSan " + std::string(subject) + " placement exceeds relocated text: source=" +
            std::to_string(staged.patch.anchor_offset) +
            " target=" + std::to_string(emitted_offset) + " size=" + std::to_string(emitted_size) +
            " text=" + std::to_string(relocated->text_size));
        return false;
      }
      if (!primary)
        primary = placed;
      fragment_placements.push_back(placed);
      placed_patches.push_back(std::move(placed));
    }
    if (!primary) {
      result.errors.emplace_back(
          "ConSan " + std::string(subject) + " lost relocated instruction placement for fragment " +
          std::to_string(staged.id) + " at source offset " +
          std::to_string(staged.patch.anchor_offset) + " with " +
          std::to_string(staged.patch.owner_descriptor_file_offsets.size()) + " owner(s)");
      return false;
    }
    if (!staged.sync_relocations.empty()) {
      if (fragment_placements.size() != 1u) {
        result.errors.emplace_back("ConSan " + std::string(subject) +
                                   " cloned one semantic synchronization relocation");
        return false;
      }
      for (const ConSanTextFragmentSyncRelocation &semantic : staged.sync_relocations) {
        if (semantic.fragment_relative_offset > primary->trampoline_size) {
          result.errors.emplace_back("ConSan " + std::string(subject) +
                                     " has an out-of-range synchronization relocation");
          return false;
        }
        result.synchronization_mutation.events.push_back(
            {.source_event = semantic.source_event,
             .kind = ConSanSyncEventMutationKind::Relocated,
             .relocated_text_offset =
                 primary->trampoline_offset + semantic.fragment_relative_offset});
      }
    }
    if (!staged.intent_ids.empty()) {
      ConSanPatchInfo primary_location = *primary;
      // A relocated fragment replaces its source instruction; unlike the old
      // trampoline pipeline, no instrumented bytes remain at the source
      // coordinate. Keep original_size in the public patch description, but
      // publish only the emitted fragment as semantic lowering geometry.
      if (!in_place)
        primary_location.original_size = 0u;
      // Keep the staged mapping intact until every owner clone has been
      // validated below. Shared helper fragments can produce several emitted
      // locations for one intent set, and moving here would leave later clone
      // validation without the access attribution required by that intent.
      ConSanRuntimeStaticMapping runtime_mapping = staged.runtime_mapping;
      if (auto *sampled = runtime_mapping.sampled()) {
        for (ConSanSampledStaticAccessMapping &mapping : *sampled) {
          mapping.emitted_probe_text_offset = primary_location.trampoline_offset;
          mapping.relocated_guest_text_offset = primary_location.relocated_guest_instruction_offset;
        }
      }
      auto commit =
          make_consan_instrumented_patch_lowering(result.observation_plan(), staged.intent_ids,
                                                  primary_location, std::move(runtime_mapping));
      if (!commit) {
        result.errors.emplace_back("ConSan " + std::string(subject) +
                                   " produced an invalid intent-bound lowering");
        result.discard_candidate_modification();
        return false;
      }
      for (size_t index = 1u; index < fragment_placements.size(); ++index) {
        ConSanPatchInfo clone_location = fragment_placements[index];
        if (!in_place)
          clone_location.original_size = 0u;
        auto clone = make_consan_instrumented_patch_lowering(
            result.observation_plan(), staged.intent_ids, clone_location, staged.runtime_mapping);
        if (!clone) {
          result.errors.emplace_back("ConSan " + std::string(subject) +
                                     " produced invalid cloned lowering geometry");
          result.discard_candidate_modification();
          return false;
        }
        for (ConSanCommittedLoweringLocation &location : clone->locations) {
          if (std::ranges::find(commit->locations, location) == commit->locations.end())
            commit->locations.push_back(std::move(location));
        }
      }
      commits.push_back(std::move(*commit));
    }
  }

  if (relocated->relocation)
    result.text_relocation = relocated->relocation;
  result.staged_text_fragments.clear();
  // One source instruction may implement both an access intent and an atomic
  // or fence intent.  The relocation transaction commits their nested
  // fragments together, so merge overlapping semantic commits just as the
  // former sequential synchronization pass did after publishing access
  // instrumentation.
  if (!result.coverage_ledger.publish_replacing_instrumented_commits(std::move(commits))) {
    result.errors.emplace_back("ConSan " + std::string(subject) +
                               " could not commit its composed semantic lowerings");
    result.discard_candidate_modification();
    return false;
  }
  result.replacement = std::move(relocated->image);
  result.patches.insert(result.patches.end(), std::make_move_iterator(placed_patches.begin()),
                        std::make_move_iterator(placed_patches.end()));
  result.mark_modified();
  return true;
}

} // namespace rocjitsu

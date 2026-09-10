// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_final_validation.h"
#include "rocjitsu/code/patch/consan/consan_final_validation_internal.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/analysis/def_use_chain.h"
#include "rocjitsu/code/analysis/kernel_scope.h"
#include "rocjitsu/code/analysis/liveness.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/major_image_ownership.h"
#include "rocjitsu/code/patch/consan/consan_barrier_move_proof.h"
#include "rocjitsu/code/patch/consan/consan_cfg.h"
#include "rocjitsu/code/patch/consan/consan_descriptor.h"
#include "rocjitsu/code/patch/consan/consan_fault_selection.h"
#include "rocjitsu/code/patch/consan/consan_input_layout.h"
#include "rocjitsu/code/patch/consan/consan_instruction_semantics.h"
#include "rocjitsu/code/patch/consan/consan_inventory_diagnostics.h"
#include "rocjitsu/code/patch/consan/consan_lowering.h"
#include "rocjitsu/code/patch/consan/consan_moi.h"
#include "rocjitsu/code/patch/consan/consan_moi_internal.h"
#include "rocjitsu/code/patch/consan/consan_perturbation_policy.h"
#include "rocjitsu/code/patch/consan/consan_physical_site_alias.h"
#include "rocjitsu/code/patch/consan/consan_resource.h"
#include "rocjitsu/code/patch/consan/consan_runtime_kernel.h"
#include "rocjitsu/code/patch/consan/consan_sync_event_index.h"
#include "rocjitsu/code/patch/consan/consan_validation_inventory.h"
#include "rocjitsu/code/patch/consan/targets/consan_validation_target_ops.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"
#include "rocjitsu/code/patch/instrumentor.h"
#include "rocjitsu/code/patch/spill_manager.h"
#include "rocjitsu/code/patch/trampoline_builder.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "util/bit.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace rocjitsu {

namespace kd = rocr::llvm::amdhsa;

namespace {

using consan_validation_detail::ConSanFinalValidationInput;
using consan_validation_detail::FinalValidationEnvironment;
using consan_validation_detail::has_consan_patch_phase;

struct ValidationByteRange {
  uint64_t begin = 0;
  uint64_t end = 0;
};

[[nodiscard]] bool validation_ranges_overlap(ValidationByteRange lhs, ValidationByteRange rhs) {
  return lhs.begin < rhs.end && rhs.begin < lhs.end;
}

[[nodiscard]] bool has_blocked_kernel(const ProgramInventory &inventory) {
  return std::ranges::any_of(inventory.kernels(), [](const ConSanProgramContainer &kernel) {
    return kernel.preflight_action == ConSanPreflightAction::Blocked;
  });
}

template <typename Entity>
[[nodiscard]] std::vector<std::string> sorted_entity_names(std::span<const Entity> entities) {
  std::vector<std::string> names;
  names.reserve(entities.size());
  for (const Entity &entity : entities)
    names.push_back(entity.name);
  std::ranges::sort(names);
  return names;
}

struct TextSectionIdentity {
  std::string name;
  uint64_t virtual_address = 0;
  uint64_t size = 0;
};

[[nodiscard]] std::vector<TextSectionIdentity>
text_section_identities(const AmdGpuCodeObject &code_object) {
  std::vector<TextSectionIdentity> identities;
  identities.reserve(code_object.text_sections().size());
  for (const Section *section : code_object.text_sections())
    identities.push_back({section->name(), section->vaddr(), section->size()});
  std::ranges::sort(identities, [](const auto &lhs, const auto &rhs) {
    if (lhs.name != rhs.name)
      return lhs.name < rhs.name;
    return lhs.virtual_address < rhs.virtual_address;
  });
  return identities;
}

struct InventoryRange {
  ValidationByteRange bytes;
  ConSanPatchPhase phase = ConSanPatchPhase::Instrumentation;
  ConSanPatchKind kind = ConSanPatchKind::LdsLoadCheckTrap;
  bool must_be_original = false;
  bool is_trampoline = false;
  size_t patch_index = 0;
  uint64_t source_anchor = 0;
  uint64_t source_trampoline = 0;
};

void validate_patch_byte_accounting(const FinalValidationEnvironment &environment,
                                    const ConSanFinalValidationInput &result,
                                    std::vector<std::string> &errors,
                                    std::optional<ConSanTransformFailureCause> *failure_cause) {
  const size_t initial_error_count = errors.size();
  if (!environment.original_text.available || !environment.replacement_text.available) {
    errors.emplace_back(
        "ConSan final validation cannot map patch inventory across multiple text sections");
    return;
  }
  const std::span<const uint8_t> original_bytes = environment.original_text.bytes;
  const std::span<const uint8_t> replacement_bytes = environment.replacement_text.bytes;
  std::vector<InventoryRange> ranges;
  std::vector<bool> accounted(replacement_bytes.size(), false);

  if (result.text_relocation) {
    const ConSanTextRelocationProof &relocation = *result.text_relocation;
    if (relocation.source_text_size < original_bytes.size() ||
        relocation.source_text_size > replacement_bytes.size()) {
      errors.emplace_back("ConSan final validation found invalid whole-text relocation geometry");
      return;
    }
    std::fill(accounted.begin() + static_cast<ptrdiff_t>(relocation.source_text_size),
              accounted.end(), true);
  }

  const auto add_range = [&](uint64_t begin, uint32_t size, ConSanPatchPhase phase,
                             ConSanPatchKind kind, bool must_be_original, bool is_trampoline,
                             size_t patch_index, uint64_t source_anchor,
                             uint64_t source_trampoline) {
    if (errors.size() != initial_error_count)
      return;
    if (size == 0)
      return;
    if (begin % sizeof(uint32_t) != 0 || size % sizeof(uint32_t) != 0 ||
        begin > replacement_bytes.size() || size > replacement_bytes.size() - begin) {
      errors.emplace_back("ConSan final validation found a stale or unaligned patch range");
      return;
    }
    ranges.push_back({{begin, begin + size},
                      phase,
                      kind,
                      must_be_original,
                      is_trampoline,
                      patch_index,
                      source_anchor,
                      source_trampoline});
  };

  for (size_t patch_index = 0; patch_index < result.patches.size(); ++patch_index) {
    const ConSanPatchInfo &patch = result.patches[patch_index];
    add_range(patch.anchor_offset, patch.original_size, patch.phase, patch.kind,
              /*must_be_original=*/true, /*is_trampoline=*/false, patch_index, patch.anchor_offset,
              patch.trampoline_offset);
    add_range(patch.trampoline_offset, patch.trampoline_size, patch.phase, patch.kind,
              /*must_be_original=*/false, /*is_trampoline=*/true, patch_index, patch.anchor_offset,
              patch.trampoline_offset);
  }
  if (errors.size() != initial_error_count)
    return;

  std::ranges::sort(ranges, [](const InventoryRange &lhs, const InventoryRange &rhs) {
    if (lhs.bytes.begin != rhs.bytes.begin)
      return lhs.bytes.begin < rhs.bytes.begin;
    if (lhs.bytes.end != rhs.bytes.end)
      return lhs.bytes.end > rhs.bytes.end;
    return lhs.phase == ConSanPatchPhase::Mutation && rhs.phase != ConSanPatchPhase::Mutation;
  });
  std::vector<const InventoryRange *> active;
  for (const InventoryRange &next : ranges) {
    std::erase_if(active, [&](const InventoryRange *existing) {
      return existing->bytes.end <= next.bytes.begin;
    });
    const bool nested_in_mutation = std::ranges::any_of(active, [&](const auto *existing) {
      return existing->phase == ConSanPatchPhase::Mutation &&
             existing->bytes.begin <= next.bytes.begin && next.bytes.end <= existing->bytes.end;
    });
    const uint64_t size = next.bytes.end - next.bytes.begin;
    if (next.must_be_original &&
        (next.bytes.begin > original_bytes.size() ||
         size > original_bytes.size() - next.bytes.begin) &&
        !nested_in_mutation) {
      errors.emplace_back("ConSan final validation found a stale or unaligned patch range");
      return;
    }
    for (const InventoryRange *existing : active) {
      if (!validation_ranges_overlap(existing->bytes, next.bytes))
        continue;
      const bool exact_alias =
          existing->bytes.begin == next.bytes.begin && existing->bytes.end == next.bytes.end;
      const bool nested =
          (existing->bytes.begin <= next.bytes.begin && next.bytes.end <= existing->bytes.end) ||
          (next.bytes.begin <= existing->bytes.begin && existing->bytes.end <= next.bytes.end);
      const bool nested_cross_phase = existing->phase != next.phase && nested;
      const bool nested_composed_access_atomic =
          existing->phase == ConSanPatchPhase::Instrumentation &&
          next.phase == ConSanPatchPhase::Instrumentation && nested &&
          (((existing->kind == ConSanPatchKind::InlineMoiAccessRecordStore ||
             existing->kind == ConSanPatchKind::TrampolineMoiAccessRecordStore) &&
            next.kind == ConSanPatchKind::TrampolineMoiAtomicRecord) ||
           ((next.kind == ConSanPatchKind::InlineMoiAccessRecordStore ||
             next.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore) &&
            existing->kind == ConSanPatchKind::TrampolineMoiAtomicRecord));
      const bool nested_composed_sampled_access_atomic =
          existing->phase == ConSanPatchPhase::Instrumentation &&
          next.phase == ConSanPatchPhase::Instrumentation && nested &&
          existing->source_anchor == next.source_anchor &&
          (((existing->kind == ConSanPatchKind::InlineMoiSampledWatchpointStore ||
             existing->kind == ConSanPatchKind::TrampolineMoiSampledWatchpointStore) &&
            next.kind == ConSanPatchKind::TrampolineMoiSampledSyncMetadata) ||
           ((next.kind == ConSanPatchKind::InlineMoiSampledWatchpointStore ||
             next.kind == ConSanPatchKind::TrampolineMoiSampledWatchpointStore) &&
            existing->kind == ConSanPatchKind::TrampolineMoiSampledSyncMetadata));
      if (!exact_alias && !nested_cross_phase && !nested_composed_access_atomic &&
          !nested_composed_sampled_access_atomic) {
        if (failure_cause)
          *failure_cause = ConSanTransformFailureCause::OverlappingPatchRanges;
        errors.emplace_back(
            "ConSan final validation found partially overlapping patch ranges: existing=" +
            std::to_string(existing->bytes.begin) + "-" + std::to_string(existing->bytes.end) +
            " kind=" + std::to_string(static_cast<unsigned>(existing->kind)) +
            " role=" + (existing->is_trampoline ? "trampoline" : "anchor") +
            " patch=" + std::to_string(existing->patch_index) +
            " source=" + std::to_string(existing->source_anchor) + "->" +
            std::to_string(existing->source_trampoline) +
            " next=" + std::to_string(next.bytes.begin) + "-" + std::to_string(next.bytes.end) +
            " kind=" + std::to_string(static_cast<unsigned>(next.kind)) +
            " role=" + (next.is_trampoline ? "trampoline" : "anchor") + " patch=" +
            std::to_string(next.patch_index) + " source=" + std::to_string(next.source_anchor) +
            "->" + std::to_string(next.source_trampoline));
        return;
      }
    }
    active.push_back(&next);
    std::fill(accounted.begin() + static_cast<ptrdiff_t>(next.bytes.begin),
              accounted.begin() + static_cast<ptrdiff_t>(next.bytes.end), true);
  }

  const size_t common_size = std::min(original_bytes.size(), replacement_bytes.size());
  for (size_t i = 0; i < common_size; ++i) {
    if (original_bytes[i] != replacement_bytes[i] && !accounted[i]) {
      uint64_t previous_end = 0;
      uint64_t next_begin = replacement_bytes.size();
      for (const InventoryRange &range : ranges) {
        if (range.bytes.end <= i)
          previous_end = std::max(previous_end, range.bytes.end);
        if (range.bytes.begin > i)
          next_begin = std::min(next_begin, range.bytes.begin);
      }
      errors.emplace_back(
          "ConSan final validation found an unaccounted executable byte change at " +
          std::to_string(i) + " from " + std::to_string(original_bytes[i]) + " to " +
          std::to_string(replacement_bytes[i]) + " between ranges ending at " +
          std::to_string(previous_end) + " and beginning at " + std::to_string(next_begin));
      return;
    }
  }

  const uint32_t padding_nop = build_s_nop(0, environment.arch());
  size_t offset = original_bytes.size();
  while (offset < replacement_bytes.size()) {
    if (accounted[offset]) {
      ++offset;
      continue;
    }
    if (offset % sizeof(uint32_t) != 0 || sizeof(uint32_t) > replacement_bytes.size() - offset) {
      errors.emplace_back("ConSan final validation found unaccounted appended text bytes");
      return;
    }
    uint32_t word = 0;
    std::memcpy(&word, replacement_bytes.data() + offset, sizeof(word));
    if (word != padding_nop) {
      uint64_t previous_end = original_bytes.size();
      uint64_t next_begin = replacement_bytes.size();
      std::optional<InventoryRange> previous_range;
      std::optional<InventoryRange> next_range;
      for (const InventoryRange &range : ranges) {
        if (range.bytes.end <= offset && range.bytes.end >= previous_end) {
          previous_end = std::max(previous_end, range.bytes.end);
          previous_range = range;
        }
        if (range.bytes.begin > offset && range.bytes.begin <= next_begin) {
          next_begin = std::min(next_begin, range.bytes.begin);
          next_range = range;
        }
      }
      std::string samples;
      size_t sample_count = 0;
      for (size_t candidate = original_bytes.size();
           candidate + sizeof(uint32_t) <= replacement_bytes.size() && sample_count < 8u;
           candidate += sizeof(uint32_t)) {
        uint32_t candidate_word = 0;
        std::memcpy(&candidate_word, replacement_bytes.data() + candidate, sizeof(candidate_word));
        if (!accounted[candidate] && candidate_word != padding_nop) {
          samples += (samples.empty() ? "" : ",") + std::to_string(candidate) + ":" +
                     std::to_string(candidate_word);
          ++sample_count;
        }
      }
      errors.emplace_back(
          "ConSan final validation found unaccounted appended executable code at " +
          std::to_string(offset) + " word " + std::to_string(word) + " between ranges ending at " +
          std::to_string(previous_end) +
          (previous_range ? " kind " + std::to_string(static_cast<unsigned>(previous_range->kind))
                          : "") +
          " and beginning at " + std::to_string(next_begin) +
          (next_range ? " kind " + std::to_string(static_cast<unsigned>(next_range->kind)) : "") +
          " samples " + samples);
      return;
    }
    offset += sizeof(uint32_t);
  }
  return;
}

[[nodiscard]] bool decode_inventory_range(std::span<const uint8_t> text, uint64_t begin,
                                          uint32_t size, Decoder &decoder, std::string &error) {
  if (size == 0)
    return true;
  if (begin > text.size() || size > text.size() - begin) {
    error = "range is outside replacement text";
    return false;
  }
  const std::span<const uint8_t> range = text.subspan(begin, size);
  size_t offset = 0;
  while (offset < range.size()) {
    const size_t remaining = range.size() - offset;
    if (remaining < sizeof(uint32_t)) {
      error = "range ends in a partial instruction word";
      return false;
    }
    std::array<uint32_t, 4> words{};
    std::memcpy(words.data(), range.data() + offset,
                std::min(remaining, words.size() * sizeof(uint32_t)));
    std::unique_ptr<Instruction> instruction = decode_bounded_instruction(
        decoder, std::span<const uint32_t>(words).first(std::min(words.size(), remaining / 4u)),
        begin + offset);
    if (!instruction) {
      error = "decoder rejected instruction at text byte " + std::to_string(begin + offset) +
              " (encoding=0x" + consan_fixed_hex(words.front(), 8u) + ")";
      const size_t context_begin =
          offset > 4u * sizeof(uint32_t) ? offset - 4u * sizeof(uint32_t) : 0u;
      const size_t context_end =
          std::min(static_cast<size_t>(size), offset + 5u * sizeof(uint32_t));
      error += " context=";
      for (size_t context_offset = context_begin; context_offset < context_end;
           context_offset += sizeof(uint32_t)) {
        uint32_t encoding = 0u;
        std::memcpy(&encoding, text.data() + begin + context_offset, sizeof(encoding));
        if (context_offset != context_begin)
          error += ",";
        error += "0x" + consan_fixed_hex(encoding, 8u);
      }
      return false;
    }
    const int instruction_size = instruction->size();
    if (instruction_size <= 0 || instruction_size % static_cast<int>(sizeof(uint32_t)) != 0 ||
        static_cast<size_t>(instruction_size) > remaining) {
      error = "decoded instruction does not fit its inventoried range";
      return false;
    }
    offset += static_cast<size_t>(instruction_size);
    // Proof rewrites intentionally terminate before all bytes in the replaced
    // source encoding. Bytes after an unconditional terminal are unreachable
    // and do not need to form another instruction.
    if (instruction->mnemonic() == std::string_view("s_endpgm") ||
        instruction->mnemonic() == std::string_view("s_trap"))
      return true;
  }
  return true;
}

void validate_patch_decoding(const FinalValidationEnvironment &environment,
                             const ConSanFinalValidationInput &result,
                             std::vector<std::string> &errors) {
  if (!environment.replacement_text.available)
    return;
  if (!environment.require_target(
          errors, "ConSan final validation found no target profile while decoding patches",
          "ConSan final validation could not create a replacement decoder"))
    return;
  const std::span<const uint8_t> text = environment.replacement_text.bytes;
  for (const ConSanPatchInfo &patch : result.patches) {
    std::string error;
    if (!decode_inventory_range(text, patch.anchor_offset, patch.original_size,
                                *environment.decoder, error)) {
      errors.emplace_back("ConSan final validation could not re-decode patch anchor: " + error);
      return;
    }
    if (!decode_inventory_range(text, patch.trampoline_offset, patch.trampoline_size,
                                *environment.decoder, error)) {
      errors.emplace_back("ConSan final validation could not re-decode generated body at " +
                          std::to_string(patch.trampoline_offset) + " (size " +
                          std::to_string(patch.trampoline_size) + ", anchor " +
                          std::to_string(patch.anchor_offset) + "): " + error);
      return;
    }
  }
  return;
}

[[nodiscard]] bool decodes_as_barrier(std::span<const uint8_t> text, uint64_t offset,
                                      Decoder &decoder) {
  if (offset > text.size() || sizeof(uint32_t) > text.size() - offset)
    return false;
  std::array<uint32_t, 4> words{};
  std::memcpy(words.data(), text.data() + offset,
              std::min(words.size() * sizeof(uint32_t), text.size() - offset));
  std::unique_ptr<Instruction> instruction =
      decode_bounded_instruction(decoder,
                                 std::span<const uint32_t>(words).first(std::min(
                                     words.size(), (text.size() - offset) / sizeof(uint32_t))),
                                 offset);
  return instruction != nullptr && instruction->size() == static_cast<int>(sizeof(uint32_t)) &&
         is_barrier_instruction(*instruction);
}

void validate_mutation_semantics(const FinalValidationEnvironment &environment,
                                 const ConSanFinalValidationInput &result,
                                 const ConSanPristineValidationInventory *pristine_inventory,
                                 std::vector<std::string> &errors) {
  // Instrumentation-only transforms have no mutation semantics to rederive.
  // Avoid rebuilding the entire pristine synchronization/CFG inventory here:
  // final validation calls this function once internally and callers may
  // independently validate the same result again. On large code objects that
  // otherwise turned one transform into three identical whole-image analyses.
  if (!has_consan_patch_phase(result, ConSanPatchPhase::Mutation)) {
    return;
  }
  const bool staged_composition = has_consan_patch_phase(result, ConSanPatchPhase::Instrumentation);
  if (!environment.original_text.available || !environment.replacement_text.available)
    return;
  const std::span<const uint8_t> original_text = environment.original_text.bytes;
  const std::span<const uint8_t> replacement_text = environment.replacement_text.bytes;
  if (!environment.require_target(errors, "ConSan mutation proof found no target profile",
                                  "ConSan mutation proof could not create a replacement decoder"))
    return;
  const rj_code_arch_t arch = environment.arch();
  const uint32_t nop = build_s_nop(0, arch);
  if (pristine_inventory == nullptr) {
    errors.emplace_back("ConSan mutation proof found no pristine validation inventory");
    return;
  }
  const ConSanFaultSelectionView pristine_faults = pristine_inventory->fault_selection();
  std::vector<const ConSanPatchInfo *> markerless_sources;
  const ConSanPatchInfo *markerless_target = nullptr;
  std::vector<ConSanBarrierSite> barrier_id_scope_originals;
  std::vector<ConSanBarrierSite> barrier_id_scope_rewrites;
  std::vector<const ConSanPatchInfo *> barrier_id_scope_patches;
  std::vector<const ConSanPatchInfo *> exact_barrier_drop_patches;
  bool barrier_id_scope_reinstrumented = false;
  const auto composed_atomic_bytes =
      [&](const ConSanPatchInfo &mutation) -> std::optional<std::span<const uint8_t>> {
    if (!staged_composition)
      return std::nullopt;
    const ConSanPatchInfo *match = nullptr;
    for (const ConSanPatchInfo &candidate : result.patches) {
      if (candidate.phase != ConSanPatchPhase::Instrumentation ||
          candidate.anchor_offset != mutation.anchor_offset ||
          candidate.original_size != mutation.original_size)
        continue;
      const bool sc_relocation = candidate.kind == ConSanPatchKind::TrampolineScPerturbation &&
                                 candidate.perturbation_composite_atomic_overlap &&
                                 candidate.perturbation_edge;
      const bool moi_atomic_relocation =
          (candidate.kind == ConSanPatchKind::TrampolineMoiAtomicRecord ||
           candidate.kind == ConSanPatchKind::TrampolineMoiSampledSyncMetadata) &&
          candidate.relocated_guest_instruction_offset;
      const bool inline_shadow_relocation =
          candidate.kind == ConSanPatchKind::TrampolineMoiInlineAtomicOrdering &&
          candidate.relocated_guest_instruction_offset;
      if (!sc_relocation && !moi_atomic_relocation && !inline_shadow_relocation)
        continue;
      if (match != nullptr)
        return std::nullopt;
      match = &candidate;
    }
    if (match == nullptr)
      return std::nullopt;
    uint64_t body_offset = 0;
    if (match->kind == ConSanPatchKind::TrampolineMoiAtomicRecord ||
        match->kind == ConSanPatchKind::TrampolineMoiSampledSyncMetadata ||
        match->kind == ConSanPatchKind::TrampolineMoiInlineAtomicOrdering) {
      body_offset = *match->relocated_guest_instruction_offset;
      if (match->trampoline_offset > replacement_text.size() ||
          match->trampoline_size > replacement_text.size() - match->trampoline_offset)
        return std::nullopt;
      const uint64_t trampoline_end = match->trampoline_offset + match->trampoline_size;
      if (body_offset < match->trampoline_offset || body_offset > trampoline_end ||
          mutation.original_size > trampoline_end - body_offset)
        return std::nullopt;
    } else {
      body_offset =
          match->trampoline_offset +
          (*match->perturbation_edge == ConSanPerturbationEdge::Release ? sizeof(uint32_t) : 0u);
    }
    if (body_offset > replacement_text.size() ||
        mutation.original_size > replacement_text.size() - body_offset)
      return std::nullopt;
    return replacement_text.subspan(body_offset, mutation.original_size);
  };
  const auto composed_lds_bytes =
      [&](const ConSanPatchInfo &mutation) -> std::optional<std::span<const uint8_t>> {
    if (!staged_composition)
      return std::nullopt;
    const ConSanPatchInfo *match = nullptr;
    for (const ConSanPatchInfo &candidate : result.patches) {
      if (candidate.phase != ConSanPatchPhase::Instrumentation ||
          candidate.anchor_offset != mutation.anchor_offset ||
          !candidate.relocated_guest_instruction_offset)
        continue;
      if (match != nullptr)
        return std::nullopt;
      match = &candidate;
    }
    if (match == nullptr)
      return std::nullopt;
    const uint64_t relocated = *match->relocated_guest_instruction_offset;
    const uint64_t region_begin =
        match->trampoline_size == 0u ? match->anchor_offset : match->trampoline_offset;
    const uint64_t region_size =
        match->trampoline_size == 0u ? match->original_size : match->trampoline_size;
    if (region_begin > replacement_text.size() ||
        region_size > replacement_text.size() - region_begin || relocated < region_begin ||
        relocated > region_begin + region_size ||
        mutation.original_size > region_begin + region_size - relocated)
      return std::nullopt;
    return replacement_text.subspan(relocated, mutation.original_size);
  };
  const auto composed_inline_barrier = [&](const ConSanPatchInfo &mutation) {
    const ConSanPatchInfo *match = nullptr;
    for (const ConSanPatchInfo &candidate : result.patches) {
      if (candidate.phase != ConSanPatchPhase::Instrumentation ||
          candidate.kind != ConSanPatchKind::TrampolineMoiInlineEpochBarrier ||
          candidate.anchor_offset != mutation.anchor_offset ||
          candidate.original_size != mutation.original_size ||
          !candidate.relocated_guest_instruction_offset)
        continue;
      if (match != nullptr)
        return false;
      match = &candidate;
    }
    if (match == nullptr || match->trampoline_offset > replacement_text.size() ||
        match->trampoline_size > replacement_text.size() - match->trampoline_offset)
      return false;
    const uint64_t trampoline_end = match->trampoline_offset + match->trampoline_size;
    const uint64_t relocated = *match->relocated_guest_instruction_offset;
    return relocated >= match->trampoline_offset && relocated < trampoline_end &&
           decodes_as_barrier(replacement_text, relocated, *environment.decoder);
  };

  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.phase != ConSanPatchPhase::Mutation)
      continue;
    if (patch.anchor_offset > original_text.size() ||
        patch.original_size > original_text.size() - patch.anchor_offset ||
        patch.anchor_offset > replacement_text.size() ||
        patch.original_size > replacement_text.size() - patch.anchor_offset) {
      errors.emplace_back("ConSan mutation proof found an out-of-range mutation anchor");
      continue;
    }

    if (patch.kind == ConSanPatchKind::InlineBarrierNopRewrite ||
        patch.kind == ConSanPatchKind::InlineBarrierMoveSourceRewrite) {
      if (patch.kind == ConSanPatchKind::InlineBarrierNopRewrite &&
          !patch.fault_sequence_identity.empty())
        exact_barrier_drop_patches.push_back(&patch);
      if (patch.kind == ConSanPatchKind::InlineBarrierMoveSourceRewrite &&
          patch.barrier_move_direction)
        markerless_sources.push_back(&patch);
      uint32_t replacement_word = 0;
      std::memcpy(&replacement_word, replacement_text.data() + patch.anchor_offset,
                  sizeof(replacement_word));
      if (!decodes_as_barrier(original_text, patch.anchor_offset, *environment.decoder) ||
          replacement_word != nop) {
        errors.emplace_back(
            "ConSan mutation proof did not replace the selected barrier with s_nop 0 at text "
            "offset " +
            std::to_string(patch.anchor_offset) +
            " (replacement=" + std::to_string(replacement_word) + ")");
      }
      continue;
    }
    if (patch.kind == ConSanPatchKind::InlineBarrierMoveTargetRewrite) {
      if (patch.trampoline_size != 0) {
        if (markerless_target != nullptr) {
          errors.emplace_back(
              "ConSan mutation proof found multiple markerless barrier relocation targets");
        } else {
          markerless_target = &patch;
        }
        continue;
      }
      if (!decodes_as_barrier(replacement_text, patch.anchor_offset, *environment.decoder) &&
          !composed_inline_barrier(patch)) {
        errors.emplace_back(
            "ConSan mutation proof did not place a barrier at the relocation target");
      }
      continue;
    }
    if (patch.kind == ConSanPatchKind::InlineOrdinaryAddressRewrite ||
        patch.kind == ConSanPatchKind::InlineOrdinaryOrderRewrite ||
        patch.kind == ConSanPatchKind::InlineOrdinaryScopeRewrite) {
      const ConSanFaultSelection selection{
          .primary_site_identity = patch.fault_primary_identity,
          .primary_sequence_identity = {},
          .companion_site_identity = {},
          .companion_sequence_identity = {},
          .kernel_name_filter = {},
          .ordinal = 0,
      };
      const auto target = select_ordinary_acquire_mutation_target(pristine_faults, selection);
      const ConSanProgramSite *target_source =
          target ? pristine_faults.source(*target->site) : nullptr;
      const std::span<const ConSanExecutionOwner> target_owners =
          target ? pristine_faults.execution_owners(*target->site)
                 : std::span<const ConSanExecutionOwner>{};
      const std::vector<uint64_t> target_owner_descriptors =
          pristine_faults.program_inventory.execution_owner_descriptors(target_owners);
      const bool identities_match =
          target && target_source != nullptr && !patch.fault_primary_identity.empty() &&
          patch.fault_sequence_identity == target->sequence->identity &&
          result.mutation.applied_fault_logical_identity == patch.fault_sequence_identity &&
          patch.owner_descriptor_file_offsets == target_owner_descriptors;
      if (!identities_match) {
        errors.emplace_back(
            "ConSan mutation proof could not rederive the exact ordinary acquire identity and "
            "owners");
        continue;
      }
      if (patch.kind == ConSanPatchKind::InlineOrdinaryAddressRewrite) {
        if (!patch.fault_companion_identity.empty() ||
            patch.anchor_offset != target_source->text_offset()) {
          errors.emplace_back(
              "ConSan mutation proof found stale ordinary acquire address metadata");
          continue;
        }
        const auto validation = validate_consan_encoded_mutation(
            arch, ConSanEncodedMutationKind::OrdinaryGlobalAddress,
            original_text.subspan(patch.anchor_offset, patch.original_size),
            replacement_text.subspan(patch.anchor_offset, patch.original_size));
        if (validation == ConSanEncodedMutationValidation::UnexpectedInstructionSize ||
            validation == ConSanEncodedMutationValidation::UnsupportedInstructionEncoding) {
          errors.emplace_back(
              "ConSan mutation proof found stale ordinary acquire address metadata");
        } else if (validation != ConSanEncodedMutationValidation::Valid) {
          errors.emplace_back(
              "ConSan mutation proof changed fields other than the ordinary load ioffset");
        }
        continue;
      }
      if (patch.kind == ConSanPatchKind::InlineOrdinaryOrderRewrite) {
        const ConSanFenceSite *cache_source =
            pristine_faults.program_inventory.program_site<ConSanFenceSite>(
                target->cache->source_site);
        if (patch.fault_companion_identity != target->cache->identity ||
            patch.anchor_offset != target->cache->text_offset() || cache_source == nullptr ||
            patch.original_size != cache_source->size) {
          errors.emplace_back(
              "ConSan mutation proof did not target the exact ordinary acquire cache boundary");
          continue;
        }
        bool all_nops = true;
        for (uint32_t offset = 0; offset < patch.original_size; offset += sizeof(uint32_t)) {
          uint32_t word = 0;
          std::memcpy(&word, replacement_text.data() + patch.anchor_offset + offset, sizeof(word));
          all_nops &= word == nop;
        }
        if (!all_nops) {
          errors.emplace_back(
              "ConSan mutation proof did not remove only the ordinary acquire global_inv");
        }
        continue;
      }

      if (!patch.fault_companion_identity.empty() ||
          patch.anchor_offset != target_source->text_offset()) {
        errors.emplace_back("ConSan mutation proof found stale ordinary acquire scope metadata");
        continue;
      }
      const auto validation = validate_consan_encoded_mutation(
          arch, ConSanEncodedMutationKind::OrdinaryGlobalScope,
          original_text.subspan(patch.anchor_offset, patch.original_size),
          replacement_text.subspan(patch.anchor_offset, patch.original_size));
      if (validation == ConSanEncodedMutationValidation::UnexpectedInstructionSize ||
          validation == ConSanEncodedMutationValidation::UnsupportedInstructionEncoding) {
        errors.emplace_back("ConSan mutation proof found stale ordinary acquire scope metadata");
      } else if (validation != ConSanEncodedMutationValidation::Valid) {
        errors.emplace_back(
            "ConSan mutation proof changed fields other than the exact ordinary load scope");
      }
      continue;
    }
    if (patch.kind == ConSanPatchKind::InlineLdsAddressRewrite) {
      const ConSanFaultSelection selection{
          .primary_site_identity = patch.fault_primary_identity,
          .primary_sequence_identity = {},
          .companion_site_identity = {},
          .companion_sequence_identity = {},
          .kernel_name_filter = {},
          .ordinal = 0,
      };
      const ConSanFaultSite *target =
          select_fault_site_for_plan(pristine_faults, selection, ConSanFaultSiteKind::LdsAccess);
      const ConSanProgramSite *target_source =
          target == nullptr ? nullptr : pristine_faults.source(*target);
      const std::span<const ConSanExecutionOwner> target_owners =
          target == nullptr ? std::span<const ConSanExecutionOwner>{}
                            : pristine_faults.execution_owners(*target);
      const std::vector<uint64_t> target_owner_descriptors =
          pristine_faults.program_inventory.execution_owner_descriptors(target_owners);
      const std::optional<uint16_t> target_address_vgpr =
          target_source == nullptr ? std::nullopt : target_source->operands.address_vgpr;
      const bool identities_match =
          target != nullptr && target_source != nullptr && !patch.fault_primary_identity.empty() &&
          patch.fault_primary_identity == target->identity &&
          result.mutation.applied_fault_logical_identity == patch.fault_primary_identity &&
          patch.owner_descriptor_file_offsets == target_owner_descriptors;
      if (!identities_match) {
        errors.emplace_back(
            "ConSan mutation proof could not rederive the exact LDS access identity and owners");
        continue;
      }
      if (!patch.fault_companion_identity.empty() || !patch.fault_sequence_identity.empty() ||
          patch.anchor_offset != target_source->text_offset() ||
          patch.original_size != target_source->size() ||
          patch.original_size != 2u * sizeof(uint32_t) || !target_address_vgpr ||
          patch.fault_original_address_vgpr != target_address_vgpr ||
          !patch.fault_target_address_vgpr ||
          *patch.fault_target_address_vgpr == *target_address_vgpr) {
        errors.emplace_back("ConSan mutation proof found stale LDS address metadata");
        continue;
      }
      std::array<uint32_t, 2> before{};
      std::array<uint32_t, 2> after{};
      std::memcpy(before.data(), original_text.data() + patch.anchor_offset, patch.original_size);
      std::span<const uint8_t> after_bytes;
      if (staged_composition) {
        const auto staged = composed_lds_bytes(patch);
        if (!staged) {
          errors.emplace_back(
              "ConSan mutation proof could not locate one exact relocated LDS access");
          continue;
        }
        after_bytes = *staged;
      } else {
        after_bytes = replacement_text.subspan(patch.anchor_offset, patch.original_size);
      }
      std::memcpy(after.data(), after_bytes.data(), patch.original_size);
      const bool exact_operand_rewrite = (before[1] & 0xffu) == *target_address_vgpr &&
                                         before[0] == after[0] &&
                                         (before[1] & ~0xffu) == (after[1] & ~0xffu) &&
                                         (after[1] & 0xffu) == *patch.fault_target_address_vgpr;
      if (!exact_operand_rewrite) {
        errors.emplace_back(
            "ConSan mutation proof changed fields other than the selected LDS address VGPR");
      }
      continue;
    }
    if (patch.kind == ConSanPatchKind::InlineBarrierParticipantCountRewrite) {
      if (patch.original_size != 2u * sizeof(uint32_t) || !patch.original_participant_count ||
          !patch.target_participant_count ||
          *patch.original_participant_count == *patch.target_participant_count ||
          *patch.original_participant_count == 0u || *patch.original_participant_count > 63u ||
          *patch.target_participant_count == 0u || *patch.target_participant_count > 63u) {
        errors.emplace_back("ConSan mutation proof found invalid participant-count patch metadata");
        continue;
      }
      uint32_t before_setup = 0;
      uint32_t after_setup = 0;
      uint32_t before_literal = 0;
      uint32_t after_literal = 0;
      std::memcpy(&before_setup, original_text.data() + patch.anchor_offset, sizeof(before_setup));
      std::memcpy(&after_setup, replacement_text.data() + patch.anchor_offset, sizeof(after_setup));
      std::memcpy(&before_literal, original_text.data() + patch.anchor_offset + sizeof(uint32_t),
                  sizeof(before_literal));
      std::memcpy(&after_literal, replacement_text.data() + patch.anchor_offset + sizeof(uint32_t),
                  sizeof(after_literal));
      constexpr uint32_t kBarrierIdMask = 0x3fu;
      constexpr uint32_t kParticipantCountMask = 0x3fu << 16u;
      constexpr uint32_t kAllowedMask = kBarrierIdMask | kParticipantCountMask;
      const uint32_t barrier_id = before_literal & kBarrierIdMask;
      const bool exact_setup =
          before_setup == after_setup && barrier_id >= 1u && barrier_id <= 16u &&
          (before_literal & ~kAllowedMask) == 0u && (after_literal & ~kAllowedMask) == 0u &&
          (after_literal & kBarrierIdMask) == barrier_id &&
          ((before_literal >> 16u) & 0x3fu) == *patch.original_participant_count &&
          ((after_literal >> 16u) & 0x3fu) == *patch.target_participant_count &&
          (before_literal & ~kParticipantCountMask) == (after_literal & ~kParticipantCountMask);
      const uint64_t init_offset = patch.anchor_offset + patch.original_size;
      std::unique_ptr<Instruction> setup_instruction;
      std::unique_ptr<Instruction> init_instruction;
      std::array<uint32_t, 2> setup_words{before_setup, before_literal};
      setup_instruction =
          decode_bounded_instruction(*environment.decoder, setup_words, patch.anchor_offset);
      if (init_offset + sizeof(uint32_t) <= original_text.size()) {
        uint32_t init_word = 0;
        std::memcpy(&init_word, original_text.data() + init_offset, sizeof(init_word));
        init_instruction = decode_bounded_instruction(
            *environment.decoder, std::span<const uint32_t>(&init_word, 1), init_offset);
      }
      if (!setup_instruction || !init_instruction) {
        errors.emplace_back("ConSan mutation proof could not decode participant-count setup");
        continue;
      }
      if (!exact_setup || !setup_instruction || setup_instruction->mnemonic() != "s_mov_b32" ||
          setup_instruction->size() != static_cast<int>(patch.original_size) || !init_instruction ||
          init_instruction->mnemonic() != "s_barrier_init") {
        errors.emplace_back(
            "ConSan mutation proof changed fields outside a proven literal M0 participant count");
      }
      continue;
    }
    if (patch.kind == ConSanPatchKind::InlineBarrierIdScopeRewrite) {
      std::array<uint32_t, 4> before_words{};
      std::array<uint32_t, 4> after_words{};
      if (patch.original_size == 0 || patch.original_size > sizeof(before_words) ||
          patch.original_size % sizeof(uint32_t) != 0) {
        errors.emplace_back("ConSan mutation proof found an invalid barrier ID/scope member size");
        continue;
      }
      std::memcpy(before_words.data(), original_text.data() + patch.anchor_offset,
                  patch.original_size);
      std::memcpy(after_words.data(), replacement_text.data() + patch.anchor_offset,
                  patch.original_size);
      std::unique_ptr<Instruction> before_instruction = decode_bounded_instruction(
          *environment.decoder,
          std::span<const uint32_t>(before_words).first(patch.original_size / sizeof(uint32_t)),
          patch.anchor_offset);
      std::unique_ptr<Instruction> after_instruction;
      if (!before_instruction) {
        errors.emplace_back("ConSan mutation proof could not decode a barrier ID/scope rewrite");
        continue;
      }
      const bool reinstrumented =
          staged_composition &&
          std::ranges::any_of(result.patches, [&](const ConSanPatchInfo &candidate) {
            return candidate.phase == ConSanPatchPhase::Instrumentation &&
                   candidate.anchor_offset <= patch.anchor_offset &&
                   patch.anchor_offset + patch.original_size <=
                       candidate.anchor_offset + candidate.original_size;
          });
      if (reinstrumented) {
        if (!before_instruction ||
            before_instruction->size() != static_cast<int>(patch.original_size)) {
          errors.emplace_back(
              "ConSan mutation proof changed the pristine shape of a composed barrier member");
          continue;
        }
        ConSanBarrierSite before_site;
        before_site.mnemonic = std::string(before_instruction->mnemonic());
        decode_barrier_operand(
            *before_instruction,
            original_text.subspan(patch.anchor_offset, static_cast<size_t>(patch.original_size)),
            before_site);
        if (!before_site.barrier_id || before_site.scope == ConSanBarrierSite::Scope::Unknown) {
          errors.emplace_back(
              "ConSan mutation proof found an unsupported pristine composed barrier member");
          continue;
        }
        barrier_id_scope_originals.push_back(std::move(before_site));
        barrier_id_scope_patches.push_back(&patch);
        barrier_id_scope_reinstrumented = true;
        continue;
      }
      after_instruction = decode_bounded_instruction(
          *environment.decoder,
          std::span<const uint32_t>(after_words).first(patch.original_size / sizeof(uint32_t)),
          patch.anchor_offset);
      if (!after_instruction) {
        errors.emplace_back("ConSan mutation proof could not decode a barrier ID/scope rewrite");
        continue;
      }
      if (!before_instruction || !after_instruction ||
          before_instruction->mnemonic() != after_instruction->mnemonic() ||
          before_instruction->size() != static_cast<int>(patch.original_size) ||
          after_instruction->size() != static_cast<int>(patch.original_size)) {
        errors.emplace_back("ConSan mutation proof changed the shape of a barrier ID/scope member");
        continue;
      }
      ConSanBarrierSite before_site;
      ConSanBarrierSite after_site;
      before_site.mnemonic = std::string(before_instruction->mnemonic());
      after_site.mnemonic = std::string(after_instruction->mnemonic());
      decode_barrier_operand(
          *before_instruction,
          original_text.subspan(patch.anchor_offset, static_cast<size_t>(patch.original_size)),
          before_site);
      decode_barrier_operand(
          *after_instruction,
          replacement_text.subspan(patch.anchor_offset, static_cast<size_t>(patch.original_size)),
          after_site);
      bool preserved = before_site.barrier_id && after_site.barrier_id &&
                       before_site.barrier_id != after_site.barrier_id &&
                       before_site.operand_source == after_site.operand_source &&
                       after_site.scope != ConSanBarrierSite::Scope::Unknown;
      if (before_site.mnemonic == "s_barrier_init" || before_site.mnemonic == "s_barrier_join" ||
          before_site.mnemonic == "s_barrier_signal" ||
          before_site.mnemonic == "s_barrier_signal_isfirst") {
        if (before_site.operand_source == ConSanBarrierSite::OperandSource::Immediate) {
          preserved &= (before_words[0] & ~0xffu) == (after_words[0] & ~0xffu);
        } else if (before_site.operand_source == ConSanBarrierSite::OperandSource::Literal32) {
          preserved &= before_words[0] == after_words[0];
        } else {
          preserved = false;
        }
      } else if (before_site.mnemonic == "s_barrier_wait") {
        preserved &= patch.original_size == sizeof(uint32_t) &&
                     (before_words[0] & 0xffff0000u) == (after_words[0] & 0xffff0000u);
      } else {
        preserved = false;
      }
      if (!preserved) {
        errors.emplace_back(
            "ConSan mutation proof changed fields outside a barrier ID/scope operand");
        continue;
      }
      barrier_id_scope_originals.push_back(std::move(before_site));
      barrier_id_scope_rewrites.push_back(std::move(after_site));
      barrier_id_scope_patches.push_back(&patch);
      continue;
    }
    if (patch.kind != ConSanPatchKind::InlineAtomicAddressRewrite &&
        patch.kind != ConSanPatchKind::InlineAtomicOrderRewrite &&
        patch.kind != ConSanPatchKind::InlineAtomicScopeRewrite) {
      continue;
    }
    if (patch.kind == ConSanPatchKind::InlineAtomicOrderRewrite) {
      if (patch.original_size == 0 || patch.original_size % sizeof(uint32_t) != 0) {
        errors.emplace_back(
            "ConSan mutation proof found an unexpected atomic cache-operation size");
        continue;
      }
      bool all_nops = true;
      const auto staged = composed_atomic_bytes(patch);
      const std::span<const uint8_t> after_bytes =
          staged.value_or(replacement_text.subspan(patch.anchor_offset, patch.original_size));
      for (uint32_t offset = 0; offset < patch.original_size; offset += sizeof(uint32_t)) {
        uint32_t after_word = 0;
        std::memcpy(&after_word, after_bytes.data() + offset, sizeof(after_word));
        all_nops &= after_word == nop;
      }
      if (!all_nops) {
        errors.emplace_back(
            "ConSan mutation proof did not remove the selected atomic cache operation");
      }
      continue;
    }

    const auto staged = composed_atomic_bytes(patch);
    const std::span<const uint8_t> after_bytes =
        staged.value_or(replacement_text.subspan(patch.anchor_offset, patch.original_size));
    const std::span<const uint8_t> before_bytes =
        original_text.subspan(patch.anchor_offset, patch.original_size);
    if (patch.kind == ConSanPatchKind::InlineAtomicAddressRewrite) {
      const auto validation = validate_consan_encoded_mutation(
          arch, ConSanEncodedMutationKind::AtomicAddress, before_bytes, after_bytes);
      if (validation == ConSanEncodedMutationValidation::UnexpectedInstructionSize ||
          validation == ConSanEncodedMutationValidation::UnsupportedInstructionEncoding) {
        errors.emplace_back("ConSan mutation proof found an unexpected atomic instruction size");
      } else if (validation != ConSanEncodedMutationValidation::Valid) {
        errors.emplace_back("ConSan mutation proof found the wrong atomic address displacement");
      }
    } else {
      const auto validation = validate_consan_encoded_mutation(
          arch, ConSanEncodedMutationKind::AtomicScope, before_bytes, after_bytes);
      if (validation == ConSanEncodedMutationValidation::UnexpectedInstructionSize) {
        errors.emplace_back("ConSan mutation proof found an unexpected atomic instruction size");
        continue;
      }
      if (validation == ConSanEncodedMutationValidation::UnsupportedInstructionEncoding) {
        errors.emplace_back(
            "ConSan mutation proof found scope mutation on an unsupported atomic encoding");
        continue;
      }
      if (validation != ConSanEncodedMutationValidation::Valid) {
        errors.emplace_back(
            "ConSan mutation proof changed fields other than the selected atomic scope");
      }
    }
  }

  if (!barrier_id_scope_originals.empty()) {
    const bool consistent_original_id_and_scope =
        std::ranges::all_of(barrier_id_scope_originals, [&](const ConSanBarrierSite &site) {
          return barrier_id_scope_originals.front().barrier_id && site.barrier_id &&
                 site.barrier_id == barrier_id_scope_originals.front().barrier_id &&
                 site.scope == barrier_id_scope_originals.front().scope;
        });
    const bool consistent_target_id_and_scope =
        std::ranges::all_of(barrier_id_scope_rewrites, [&](const ConSanBarrierSite &site) {
          return barrier_id_scope_rewrites.front().barrier_id && site.barrier_id &&
                 site.barrier_id == barrier_id_scope_rewrites.front().barrier_id &&
                 site.scope == barrier_id_scope_rewrites.front().scope &&
                 site.scope == barrier_scope_for_id(*site.barrier_id);
        });
    const bool pair_shape =
        barrier_id_scope_originals.size() == 2u &&
        barrier_id_scope_originals[0].mnemonic.find("signal") != std::string::npos &&
        barrier_id_scope_originals[1].mnemonic == "s_barrier_wait";
    bool lifecycle_shape =
        barrier_id_scope_originals.size() >= 4u &&
        barrier_id_scope_originals.front().mnemonic == "s_barrier_init" &&
        barrier_id_scope_originals[barrier_id_scope_originals.size() - 2u].mnemonic.find(
            "signal") != std::string::npos &&
        barrier_id_scope_originals.back().mnemonic == "s_barrier_wait";
    for (size_t i = 1; lifecycle_shape && i + 2u < barrier_id_scope_originals.size(); ++i)
      lifecycle_shape &= barrier_id_scope_originals[i].mnemonic == "s_barrier_join";
    if ((!pair_shape && !lifecycle_shape) || !consistent_original_id_and_scope ||
        (!barrier_id_scope_reinstrumented && !consistent_target_id_and_scope)) {
      errors.emplace_back(
          "ConSan mutation proof found an incomplete or inconsistent barrier ID/scope group");
    } else {
      // Re-derive the exact selected group from pristine executable bytes and
      // patch anchors. Do not use result.program_inventory.sync().sync_sequences or
      // lifecycle groups: they are mutable transform output, not validation evidence.
      for (size_t i = 1; i < barrier_id_scope_patches.size(); ++i) {
        const ConSanPatchInfo &previous = *barrier_id_scope_patches[i - 1u];
        const ConSanPatchInfo &current = *barrier_id_scope_patches[i];
        const uint64_t gap_begin = previous.anchor_offset + previous.original_size;
        if (gap_begin > current.anchor_offset) {
          errors.emplace_back(
              "ConSan mutation proof found overlapping or out-of-order barrier group members");
          break;
        }
        const uint64_t gap_size = current.anchor_offset - gap_begin;
        if (lifecycle_shape && gap_size > 0u) {
          errors.emplace_back(
              "ConSan mutation proof found an omitted or noncontiguous barrier lifecycle member");
          break;
        }
        if (pair_shape && gap_size > 0u &&
            !std::equal(original_text.begin() + gap_begin,
                        original_text.begin() + current.anchor_offset,
                        replacement_text.begin() + gap_begin)) {
          errors.emplace_back(
              "ConSan mutation proof found changed bytes between barrier group members");
          break;
        }
      }
      if (lifecycle_shape) {
        const ConSanPatchInfo &wait_patch = *barrier_id_scope_patches.back();
        const uint64_t leave_offset = wait_patch.anchor_offset + wait_patch.original_size;
        if (leave_offset > original_text.size() ||
            sizeof(uint32_t) > original_text.size() - leave_offset ||
            leave_offset > replacement_text.size() ||
            sizeof(uint32_t) > replacement_text.size() - leave_offset) {
          errors.emplace_back(
              "ConSan mutation proof could not derive the lifecycle leave from pristine text");
        } else {
          std::array<uint32_t, 4> leave_words{};
          std::memcpy(leave_words.data(), original_text.data() + leave_offset, sizeof(uint32_t));
          std::unique_ptr<Instruction> leave_instruction = decode_bounded_instruction(
              *environment.decoder, std::span<const uint32_t>(leave_words).first(1), leave_offset);
          uint32_t original_leave = 0;
          uint32_t replacement_leave = 0;
          std::memcpy(&original_leave, original_text.data() + leave_offset, sizeof(original_leave));
          std::memcpy(&replacement_leave, replacement_text.data() + leave_offset,
                      sizeof(replacement_leave));
          const bool leave_reinstrumented =
              staged_composition &&
              std::ranges::any_of(result.patches, [&](const ConSanPatchInfo &candidate) {
                return candidate.phase == ConSanPatchPhase::Instrumentation &&
                       candidate.anchor_offset <= leave_offset &&
                       leave_offset + sizeof(uint32_t) <=
                           candidate.anchor_offset + candidate.original_size &&
                       candidate.relocated_guest_instruction_offset &&
                       decodes_as_barrier(replacement_text,
                                          *candidate.relocated_guest_instruction_offset,
                                          *environment.decoder);
              });
          if (!leave_instruction || leave_instruction->mnemonic() != "s_barrier_leave" ||
              leave_instruction->size() != static_cast<int>(sizeof(uint32_t)) ||
              static_cast<uint16_t>(original_leave) != 0u ||
              (!leave_reinstrumented && replacement_leave != original_leave)) {
            errors.emplace_back(
                "ConSan mutation proof did not preserve the derived fixed-zero lifecycle leave");
          }
          if (!barrier_id_scope_reinstrumented &&
              (*barrier_id_scope_rewrites.front().barrier_id < 1 ||
               *barrier_id_scope_rewrites.front().barrier_id > 16 ||
               barrier_id_scope_rewrites.front().scope != ConSanBarrierSite::Scope::Workgroup)) {
            errors.emplace_back(
                "ConSan mutation proof found invalid lifecycle target ID/scope metadata");
          }
        }
      }
    }
    if (!result.mutation.applied_fault_logical_identity)
      errors.emplace_back("ConSan mutation proof lost the retained logical fault identity");
  }

  if (!exact_barrier_drop_patches.empty()) {
    std::vector<std::string> sequence_identities;
    for (const ConSanPatchInfo *patch : exact_barrier_drop_patches) {
      if (std::ranges::find(sequence_identities, patch->fault_sequence_identity) ==
          sequence_identities.end())
        sequence_identities.push_back(patch->fault_sequence_identity);
    }
    std::ranges::sort(sequence_identities, [&](const std::string &left, const std::string &right) {
      const auto first_anchor = [&](const std::string &identity) {
        uint64_t value = std::numeric_limits<uint64_t>::max();
        for (const ConSanPatchInfo *patch : exact_barrier_drop_patches)
          if (patch->fault_sequence_identity == identity)
            value = std::min(value, patch->anchor_offset);
        return value;
      };
      return first_anchor(left) < first_anchor(right);
    });
    std::vector<ExactBarrierDropPair> pairs;
    bool metadata_matches = sequence_identities.size() == 1u || sequence_identities.size() == 2u;
    for (const std::string &sequence_identity : sequence_identities) {
      const auto member = std::ranges::find_if(exact_barrier_drop_patches, [&](const auto *patch) {
        return patch->fault_sequence_identity == sequence_identity;
      });
      const ConSanFaultSelection selection{
          .primary_site_identity = (*member)->fault_primary_identity,
          .primary_sequence_identity = sequence_identity,
          .companion_site_identity = {},
          .companion_sequence_identity = {},
          .kernel_name_filter = {},
          .ordinal = 0,
      };
      const auto pair_resolution = resolve_exact_barrier_drop_pair(pristine_faults, selection);
      const auto &pair = pair_resolution.pair;
      const size_t member_count = static_cast<size_t>(
          std::ranges::count_if(exact_barrier_drop_patches, [&](const auto *patch) {
            return patch->fault_sequence_identity == sequence_identity;
          }));
      metadata_matches &= pair.has_value() && member_count == 2u;
      if (!pair)
        continue;
      const ConSanProgramSite *primary_source = pristine_faults.source(*pair->primary);
      const ConSanProgramSite *companion_source = pristine_faults.source(*pair->companion);
      metadata_matches &= primary_source != nullptr && companion_source != nullptr;
      if (primary_source == nullptr || companion_source == nullptr)
        continue;
      bool found_primary = false;
      bool found_companion = false;
      for (const ConSanPatchInfo *patch : exact_barrier_drop_patches) {
        if (patch->fault_sequence_identity != sequence_identity)
          continue;
        metadata_matches &= patch->fault_primary_identity == pair->primary->identity &&
                            patch->fault_companion_identity == pair->companion->identity &&
                            patch->original_size == sizeof(uint32_t);
        found_primary |= patch->anchor_offset == primary_source->text_offset();
        found_companion |= patch->anchor_offset == companion_source->text_offset();
      }
      metadata_matches &= found_primary && found_companion;
      pairs.push_back(*pair);
    }
    if (pairs.size() == 1u) {
      metadata_matches &=
          exact_barrier_drop_patches.size() == 2u &&
          result.mutation.applied_fault_logical_identity == pairs.front().sequence->identity;
    } else if (pairs.size() == 2u) {
      const ConSanFaultSelection selection{
          .primary_site_identity = pairs[0].primary->identity,
          .primary_sequence_identity = pairs[0].sequence->identity,
          .companion_site_identity = pairs[1].primary->identity,
          .companion_sequence_identity = pairs[1].sequence->identity,
          .kernel_name_filter = {},
          .ordinal = 0,
      };
      const auto group_resolution = resolve_exact_barrier_drop_group(pristine_faults, selection);
      const auto &group = group_resolution.group;
      metadata_matches &= exact_barrier_drop_patches.size() == 4u && group &&
                          result.mutation.applied_fault_logical_identity ==
                              exact_barrier_drop_group_identity(*group);
    } else {
      metadata_matches = false;
    }
    if (!metadata_matches) {
      errors.emplace_back(
          "ConSan mutation proof could not rederive the complete exact whole-barrier drop group");
    }
  }

  if (markerless_target != nullptr || !markerless_sources.empty()) {
    if (markerless_target == nullptr || markerless_sources.size() != 2u ||
        !markerless_target->barrier_move_direction ||
        (*markerless_target->barrier_move_direction != ConSanBarrierMoveDirection::Earlier &&
         *markerless_target->barrier_move_direction != ConSanBarrierMoveDirection::Later)) {
      errors.emplace_back(
          "ConSan mutation proof found an incomplete markerless barrier relocation record");
      return;
    }
    std::ranges::sort(markerless_sources, {}, &ConSanPatchInfo::anchor_offset);
    const ConSanPatchInfo &first = *markerless_sources[0];
    const ConSanPatchInfo &second = *markerless_sources[1];
    const ConSanPatchInfo &target = *markerless_target;
    if (first.original_size != sizeof(uint32_t) || second.original_size != sizeof(uint32_t) ||
        first.anchor_offset + sizeof(uint32_t) != second.anchor_offset) {
      errors.emplace_back(
          "ConSan mutation proof did not identify an adjacent ordered barrier pair");
      return;
    }
    if (target.original_size == 0 || target.original_size % sizeof(uint32_t) != 0) {
      errors.emplace_back("ConSan mutation proof found an unsupported displaced destination size");
      return;
    }
    const uint32_t expected_trampoline_size =
        2u * sizeof(uint32_t) + target.original_size + sizeof(uint32_t);
    if (target.trampoline_size != expected_trampoline_size ||
        target.trampoline_offset > replacement_text.size() ||
        target.trampoline_size > replacement_text.size() - target.trampoline_offset) {
      errors.emplace_back(
          "ConSan mutation proof found incorrect whole-pair trampoline byte accounting");
      return;
    }

    const ConSanBarrierMoveDirection direction = *target.barrier_move_direction;
    if ((direction == ConSanBarrierMoveDirection::Earlier &&
         target.anchor_offset + target.original_size > first.anchor_offset) ||
        (direction == ConSanBarrierMoveDirection::Later &&
         target.anchor_offset < second.anchor_offset + second.original_size)) {
      errors.emplace_back(
          "ConSan mutation proof found the relocation target on the wrong side of the pair");
      return;
    }
    const bool has_structured_proof =
        target.barrier_move_cfg_contract != ConSanBarrierMoveCfgContract::SameBlock ||
        target.structured_guard_block_index.has_value() ||
        target.structured_destination_block_index.has_value() ||
        target.structured_source_block_index.has_value() ||
        target.structured_guard_offset.has_value() ||
        target.structured_destination_offset.has_value() ||
        target.structured_source_offset.has_value();
    auto original_blocks =
        build_original_proof_basic_blocks(environment.original, *environment.decoder, arch);
    const auto containing_block_index = [&](uint64_t offset) -> std::optional<uint32_t> {
      for (size_t i = 0; i < original_blocks.size(); ++i) {
        const BasicBlock *block = original_blocks[i].get();
        if (block != nullptr && offset >= block->start_offset() && offset < block->end_offset())
          return static_cast<uint32_t>(i);
      }
      return std::nullopt;
    };
    const auto original_source_block = containing_block_index(first.anchor_offset);
    const auto original_destination_block = containing_block_index(target.anchor_offset);
    if (!original_source_block || !original_destination_block) {
      errors.emplace_back("ConSan mutation proof could not locate barrier relocation CFG blocks");
      return;
    }
    if (*original_source_block != *original_destination_block && !has_structured_proof) {
      errors.emplace_back(
          "ConSan mutation proof found a cross-block move without a structured CFG contract and "
          "metadata");
      return;
    }
    if (has_structured_proof) {
      if (!target.structured_guard_block_index || !target.structured_destination_block_index ||
          !target.structured_source_block_index || !target.structured_guard_offset ||
          !target.structured_destination_offset || !target.structured_source_offset ||
          direction != ConSanBarrierMoveDirection::Earlier ||
          *target.structured_destination_offset != target.anchor_offset ||
          *target.structured_source_block_index == *target.structured_destination_block_index) {
        errors.emplace_back("ConSan mutation proof found incomplete structured CFG metadata");
        return;
      }
      if (*original_source_block != *target.structured_source_block_index ||
          *original_destination_block != *target.structured_destination_block_index) {
        errors.emplace_back(
            "ConSan mutation proof found structured CFG block metadata that does not "
            "contain the selected relocation instructions");
        return;
      }
      std::optional<StructuredExecDiamondProof> rederived;
      if (target.barrier_move_cfg_contract ==
          ConSanBarrierMoveCfgContract::CompletingStructuredDiamond) {
        rederived = prove_completing_structured_diamond(
            original_blocks, *target.structured_source_block_index,
            *target.structured_destination_block_index, first.anchor_offset, target.anchor_offset);
      } else if (target.barrier_move_cfg_contract ==
                 ConSanBarrierMoveCfgContract::DestructiveStructuredExecDiamond) {
        rederived = prove_structured_exec_diamond(
            original_blocks, *target.structured_source_block_index,
            *target.structured_destination_block_index, first.anchor_offset);
      }
      if (!rederived || rederived->guard_block_index != *target.structured_guard_block_index ||
          rederived->source_block_index != *target.structured_source_block_index ||
          rederived->destination_block_index != *target.structured_destination_block_index ||
          rederived->guard_offset != *target.structured_guard_offset ||
          rederived->source_offset != *target.structured_source_offset) {
        errors.emplace_back("ConSan mutation proof could not rederive its structured CFG contract");
        return;
      }
    }
    const auto forward = compute_sopp_branch_simm16(target.anchor_offset, target.trampoline_offset);
    uint32_t replacement_anchor = 0;
    std::memcpy(&replacement_anchor, replacement_text.data() + target.anchor_offset,
                sizeof(replacement_anchor));
    if (!forward || replacement_anchor != build_s_branch(*forward, arch)) {
      errors.emplace_back(
          "ConSan mutation proof found an invalid branch to the whole-pair trampoline");
      return;
    }
    for (uint32_t offset = sizeof(uint32_t); offset < target.original_size;
         offset += sizeof(uint32_t)) {
      uint32_t anchor_tail = 0;
      std::memcpy(&anchor_tail, replacement_text.data() + target.anchor_offset + offset,
                  sizeof(anchor_tail));
      if (anchor_tail != nop) {
        errors.emplace_back(
            "ConSan mutation proof did not fully replace the displaced destination anchor");
        return;
      }
    }

    std::array<uint32_t, 2> original_barriers{};
    std::memcpy(&original_barriers[0], original_text.data() + first.anchor_offset,
                sizeof(uint32_t));
    std::memcpy(&original_barriers[1], original_text.data() + second.anchor_offset,
                sizeof(uint32_t));
    const ValidationByteRange mutation_body{target.trampoline_offset,
                                            target.trampoline_offset + target.trampoline_size};
    const bool body_reinstrumented =
        std::ranges::any_of(result.patches, [&](const ConSanPatchInfo &patch) {
          if (patch.phase != ConSanPatchPhase::Instrumentation || patch.original_size == 0)
            return false;
          const ValidationByteRange anchor{patch.anchor_offset,
                                           patch.anchor_offset + patch.original_size};
          return validation_ranges_overlap(mutation_body, anchor);
        });
    // A staged composition validates pristine -> mutation before lowering the
    // instrumentation stage, then validates the complete pristine -> final
    // transaction. If instrumentation replaced an instruction inside this
    // body, the final image cannot also retain the intermediate raw bytes.
    // Cross-phase nesting is checked by byte accounting; retain the source-NOP
    // and entry-branch checks above and rely on final instrumentation proof for
    // the nested rewrite.
    if (body_reinstrumented && staged_composition)
      return;
    const uint64_t barrier_body_offset =
        direction == ConSanBarrierMoveDirection::Earlier ? 0u : target.original_size;
    const uint64_t destination_body_offset =
        direction == ConSanBarrierMoveDirection::Earlier ? 2u * sizeof(uint32_t) : 0u;
    const uint8_t *body = replacement_text.data() + target.trampoline_offset;
    if (std::memcmp(body + barrier_body_offset, original_barriers.data(),
                    sizeof(original_barriers)) != 0) {
      errors.emplace_back(
          "ConSan mutation proof did not preserve an adjacent byte-identical ordered barrier "
          "pair");
      return;
    }
    if (std::memcmp(body + destination_body_offset, original_text.data() + target.anchor_offset,
                    target.original_size) != 0) {
      errors.emplace_back(
          "ConSan mutation proof did not preserve the displaced destination semantics");
      return;
    }

    // The body is exactly pair + destination + return, and the two
    // non-overlapping structural slots above exhaust its payload. That proves
    // exact-once execution without rejecting incidental byte-pattern matches.
    const uint64_t payload_size = target.trampoline_size - sizeof(uint32_t);
    const uint64_t return_pc = target.trampoline_offset + payload_size;
    const auto ret =
        compute_sopp_branch_simm16(return_pc, target.anchor_offset + target.original_size);
    uint32_t return_word = 0;
    std::memcpy(&return_word, replacement_text.data() + return_pc, sizeof(return_word));
    if (!ret || return_word != build_s_branch(*ret, arch)) {
      errors.emplace_back("ConSan mutation proof found an invalid whole-pair trampoline return");
    }
  }
  return;
}

struct DescriptorRequirement {
  bool allow_resource_delta = false;
  bool allow_entry_delta = false;
  bool require_full_workgroup_id = false;
  std::optional<ConSanMoiDispatchIdPreloadPlan> dispatch_id_preload;
  uint16_t required_vgpr_count = 0;
  uint16_t required_sgpr_count = 0;
  uint32_t required_private_bytes = 0;
  uint32_t required_group_bytes = 0;
};

void validate_entry_scalar_backup_semantics(const FinalValidationEnvironment &environment,
                                            const ConSanFinalValidationInput &result,
                                            std::vector<std::string> &errors) {
  if (!environment.original_text.available || !environment.replacement_text.available)
    return;
  const std::span<const uint8_t> replacement_text = environment.replacement_text.bytes;
  const rj_code_arch_t arch = environment.arch();

  for (const ConSanPatchInfo &patch : result.patches) {
    if (!patch.entry_scalar_backup)
      continue;
    const ConSanMoiEntryScalarBackup &backup = *patch.entry_scalar_backup;
    if ((patch.kind != ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue &&
         patch.kind != ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue) ||
        !backup.is_well_formed(kConSanOrdinaryVgprLimit, REGISTER_SET_ALLOCATABLE_SGPRS)) {
      errors.emplace_back("ConSan final validation found invalid entry scalar backup resources");
      continue;
    }

    const auto owner =
        std::ranges::find_if(environment.original.kernels(), [&](const AmdGpuKernelInfo &kernel) {
          return kernel.entry_text_offset == patch.anchor_offset;
        });
    if (owner == environment.original.kernels().end()) {
      errors.emplace_back("ConSan final validation could not resolve an entry scalar backup owner");
      continue;
    }
    std::vector<uint32_t> expected_save;
    std::vector<uint32_t> expected_restore;
    for (uint16_t index = 0; index < backup.sgpr_count; ++index) {
      const uint16_t sgpr = static_cast<uint16_t>(backup.sgpr_base + index);
      const auto save = instrumentation::build_v_writelane_b32(backup.vgpr, sgpr, index, arch);
      const auto restore = instrumentation::build_v_readlane_b32(sgpr, backup.vgpr, index, arch);
      if (!save || !restore) {
        expected_save.clear();
        expected_restore.clear();
        break;
      }
      expected_save.insert(expected_save.end(), save->begin(), save->end());
      expected_restore.insert(expected_restore.end(), restore->begin(), restore->end());
    }
    const auto restore_wait = instrumentation::build_valu_to_salu_dependency_wait(arch);
    if (restore_wait)
      expected_restore.push_back(*restore_wait);
    else
      expected_restore.clear();
    if (expected_save.empty() || expected_restore.empty()) {
      errors.emplace_back(
          "ConSan final validation could not encode entry scalar backup expectations");
      continue;
    }

    std::vector<uint64_t> body_offsets = {
        patch.dispatch_id_primary_prologue_offset.value_or(patch.trampoline_offset)};
    if (patch.dispatch_id_secondary_prologue_offset) {
      constexpr uint64_t kPairedEntryAlignment = 256u;
      body_offsets.push_back(patch.dispatch_id_secondary_prologue_offset.value_or(
          patch.trampoline_offset + kPairedEntryAlignment));
    }
    if (!std::ranges::is_sorted(body_offsets)) {
      errors.emplace_back("ConSan final validation found unordered entry scalar backup bodies");
      continue;
    }
    const uint64_t patch_end = patch.trampoline_offset + patch.trampoline_size;
    for (size_t body_index = 0; body_index < body_offsets.size(); ++body_index) {
      const uint64_t body_offset = body_offsets[body_index];
      const uint64_t body_end =
          body_index + 1u < body_offsets.size() ? body_offsets[body_index + 1u] : patch_end;
      const uint64_t save_bytes = expected_save.size() * sizeof(uint32_t);
      if (body_offset < patch.trampoline_offset || body_offset > body_end || body_end > patch_end ||
          patch_end > replacement_text.size() || save_bytes > body_end - body_offset) {
        errors.emplace_back("ConSan final validation found an invalid entry scalar backup save");
        continue;
      }
      const auto body_words = std::span<const uint32_t>(
          reinterpret_cast<const uint32_t *>(replacement_text.data() + body_offset),
          static_cast<size_t>((body_end - body_offset) / sizeof(uint32_t)));
      const auto save = std::search(body_words.begin(), body_words.end(), expected_save.begin(),
                                    expected_save.end());
      if (save == body_words.end()) {
        errors.emplace_back("ConSan final validation found an invalid entry scalar backup save");
        continue;
      }
      // If descriptor surgery moved guest inputs, the backup must capture the
      // repaired guest ABI, not the replacement preload layout.  The dispatch
      // validator independently proves that this exact restore is attached to
      // the launch-identity capture; here we prove it precedes the save.
      if (patch.dispatch_id_prologue &&
          patch.dispatch_id_prologue->preload.guest_restore_count != 0u) {
        const auto &preload = patch.dispatch_id_prologue->preload;
        const auto dependency_delay = instrumentation::build_salu_dependency_delay(arch);
        std::vector<uint32_t> guest_restore;
        if (dependency_delay) {
          for (uint16_t index = 0; index < preload.guest_restore_count; ++index) {
            guest_restore.push_back(build_s_mov_b32(preload.guest_restore_destinations[index],
                                                    preload.guest_restore_sources[index], arch));
            guest_restore.push_back(*dependency_delay);
          }
        }
        const auto repair =
            guest_restore.empty()
                ? body_words.end()
                : std::search(body_words.begin(), save, guest_restore.begin(), guest_restore.end());
        if (repair == save) {
          errors.emplace_back(
              "ConSan final validation found an entry scalar backup before guest ABI repair");
          continue;
        }
      }
      if (std::search(save + static_cast<ptrdiff_t>(expected_save.size()), body_words.end(),
                      expected_restore.begin(), expected_restore.end()) == body_words.end()) {
        errors.emplace_back("ConSan final validation found an invalid entry scalar backup restore");
      }
    }
  }
  return;
}

void validate_dispatch_id_prologue_semantics(const FinalValidationEnvironment &environment,
                                             const ConSanFinalValidationInput &result,
                                             std::vector<std::string> &errors) {
  if (!environment.original_text.available || !environment.replacement_text.available)
    return;
  const std::span<const uint8_t> replacement_text = environment.replacement_text.bytes;
  const rj_code_arch_t arch = environment.arch();
  const auto validate_entry = [&](const ConSanPatchInfo &patch, uint64_t entry_offset,
                                  std::string_view kernel_name,
                                  bool has_semantic_system_sgpr_restore) {
    if (!patch.dispatch_id_prologue) {
      errors.emplace_back("ConSan final validation found missing dispatch-ID prologue proof");
      return;
    }
    const ConSanMoiDispatchIdPrologueEffect &dispatch = *patch.dispatch_id_prologue;
    const ConSanMoiDispatchIdPreloadPlan &preload = dispatch.preload;
    const std::optional<uint16_t> capture_sgpr = dispatch.capture.sgpr();
    const std::optional<uint16_t> capture_vgpr = dispatch.capture.vgpr();
    const bool captures_sgpr = capture_sgpr.has_value();
    const bool captures_vgpr = capture_vgpr.has_value();
    if (captures_sgpr == captures_vgpr) {
      errors.emplace_back(
          "ConSan final validation found an ambiguous dispatch-ID capture for kernel '" +
          std::string(kernel_name) + "'");
      return;
    }
    bool invalid_restore = preload.guest_restore_count > preload.guest_restore_sources.size();
    for (uint16_t index = 0u; !invalid_restore && index < preload.guest_restore_count; ++index) {
      const uint16_t destination = preload.guest_restore_destinations[index];
      const uint16_t source = preload.guest_restore_sources[index];
      invalid_restore =
          destination >= preload.original_user_sgpr_count ||
          source >= preload.expanded_user_sgpr_count || destination >= source ||
          (index != 0u &&
           (preload.guest_restore_destinations[index - 1u] >= destination ||
            preload.guest_restore_sources[index - 1u] >= source));
    }
    if ((!preload.descriptor_change_required() && preload.guest_restore_count != 0u) ||
        (preload.descriptor_change_required() && invalid_restore)) {
      errors.emplace_back(
          "ConSan final validation found invalid dispatch-ID restore bounds for kernel '" +
          std::string(kernel_name) + "'");
      return;
    }
    const uint32_t restore_count = preload.guest_restore_count;
    const uint32_t system_restore_count =
        has_semantic_system_sgpr_restore ? 0u : preload.shifted_system_sgpr_count;
    const uint64_t reload_words =
        preload.kernarg_reload_count != 0 ? 2u * preload.kernarg_reload_count + 1u : 0u;
    std::vector<uint32_t> capture_prefix;
    const auto dependency_delay = instrumentation::build_salu_dependency_delay(arch);
    if (!dependency_delay) {
      errors.emplace_back("ConSan final validation cannot encode the target SALU dependency delay");
      return;
    }
    if (captures_sgpr) {
      const uint16_t persistent = *capture_sgpr;
      const auto append_salu = [&](uint32_t word) {
        capture_prefix.push_back(word);
        capture_prefix.push_back(*dependency_delay);
      };
      append_salu(build_s_mov_b32(persistent, preload.dispatch_id_sgpr, arch));
      append_salu(build_s_mov_b32(static_cast<uint16_t>(persistent + 1u),
                                  static_cast<uint16_t>(preload.dispatch_id_sgpr + 1u), arch));
      if (preload.identity_salt_sgpr) {
        const auto mix = instrumentation::build_s_xor_b64(
            persistent, persistent, *preload.identity_salt_sgpr, arch);
        if (!mix) {
          errors.emplace_back(
              "ConSan final validation cannot encode the dispatch-ID identity mix for kernel '" +
              std::string(kernel_name) + "'");
          return;
        }
        append_salu(*mix);
      }
      append_salu(build_s_add_u32(persistent, persistent, scalar_positive_inline_u32(1), arch));
      append_salu(build_s_addc_u32(static_cast<uint16_t>(persistent + 1u),
                                   static_cast<uint16_t>(persistent + 1u),
                                   scalar_positive_inline_u32(0), arch));
    } else {
      const uint16_t persistent = *capture_vgpr;
      capture_prefix.push_back(build_v_mov_b32_e32(persistent, preload.dispatch_id_sgpr, arch));
      capture_prefix.push_back(
          build_v_mov_b32_e32(static_cast<uint16_t>(persistent + 1u),
                              static_cast<uint16_t>(preload.dispatch_id_sgpr + 1u), arch));
      if (preload.identity_salt_sgpr) {
        const auto mix_low = instrumentation::build_v_xor_b32(
            persistent, *preload.identity_salt_sgpr, persistent, arch);
        const auto mix_high = instrumentation::build_v_xor_b32(
            static_cast<uint16_t>(persistent + 1u),
            static_cast<uint16_t>(*preload.identity_salt_sgpr + 1u),
            static_cast<uint16_t>(persistent + 1u), arch);
        if (!mix_low || !mix_high) {
          errors.emplace_back(
              "ConSan final validation cannot encode the dispatch-ID identity mix for kernel '" +
              std::string(kernel_name) + "'");
          return;
        }
        capture_prefix.push_back(*mix_low);
        capture_prefix.push_back(*mix_high);
      }
      const auto successor = instrumentation::build_v_add_u64_literal(persistent, 1u, arch);
      if (!successor) {
        errors.emplace_back(
            "ConSan final validation cannot encode the dispatch-ID VGPR successor for kernel '" +
            std::string(kernel_name) + "'");
        return;
      }
      capture_prefix.insert(capture_prefix.end(), successor->begin(), successor->end());
    }
    const uint64_t patch_end = patch.trampoline_offset + patch.trampoline_size;
    if (entry_offset > patch_end || patch_end > replacement_text.size() ||
        (patch_end - entry_offset) % sizeof(uint32_t) != 0) {
      errors.emplace_back(
          "ConSan final validation found a truncated dispatch-ID entry prologue for kernel '" +
          std::string(kernel_name) + "' (entry=" + std::to_string(entry_offset) + ", patch=" +
          std::to_string(patch.trampoline_offset) + "+" + std::to_string(patch.trampoline_size) +
          ", text=" + std::to_string(replacement_text.size()) + ")");
      return;
    }
    const auto entry_words = std::span<const uint32_t>(
        reinterpret_cast<const uint32_t *>(replacement_text.data() + entry_offset),
        static_cast<size_t>((patch_end - entry_offset) / sizeof(uint32_t)));
    std::vector<uint32_t> restore_words;
    restore_words.reserve(2u * (restore_count + system_restore_count));
    const auto append_restore = [&](uint16_t destination, uint16_t source) {
      restore_words.push_back(build_s_mov_b32(destination, source, arch));
      restore_words.push_back(*dependency_delay);
    };
    for (uint32_t i = 0; i < restore_count; ++i) {
      append_restore(preload.guest_restore_destinations[i], preload.guest_restore_sources[i]);
    }
    for (uint16_t i = 0; i < system_restore_count; ++i) {
      const uint16_t destination = static_cast<uint16_t>(preload.original_user_sgpr_count + i);
      append_restore(destination, static_cast<uint16_t>(destination + preload.system_sgpr_shift));
    }
    // Exact CDNA workgroup identity can be captured before dispatch preload
    // repair, and an entry scalar backup follows the repair when descriptor
    // surgery moved guest inputs. Match capture and
    // the immediately following ABI repair as one sequence: matching either
    // half independently is ambiguous in a large generated prologue because
    // unrelated instrumentation can encode the same individual moves.
    std::vector<uint32_t> dispatch_repair = std::move(capture_prefix);
    dispatch_repair.insert(dispatch_repair.end(), restore_words.begin(), restore_words.end());
    const auto repair =
        std::search(entry_words.begin(), entry_words.end(), dispatch_repair.begin(),
                    dispatch_repair.end());
    if (repair == entry_words.end()) {
      errors.emplace_back(
          "ConSan final validation found no complete dispatch-ID capture and restore in kernel '" +
          std::string(kernel_name) + "'");
      return;
    }
    const uint64_t required_suffix_words = dispatch_repair.size() + reload_words;
    if (required_suffix_words > static_cast<uint64_t>(entry_words.end() - repair)) {
      errors.emplace_back(
          "ConSan final validation found a truncated dispatch-ID entry prologue for kernel '" +
          std::string(kernel_name) + "'");
      return;
    }
    const auto read_word = [&](uint64_t word_index) { return entry_words[word_index]; };
    uint64_t cursor = static_cast<uint64_t>(repair - entry_words.begin()) + dispatch_repair.size();
    if (preload.kernarg_reload_count != 0) {
      for (uint16_t i = 0; i < preload.kernarg_reload_count; ++i) {
        const auto load = instrumentation::build_s_load_dword(
            static_cast<uint16_t>(preload.kernarg_reload_sgpr + i),
            preload.kernarg_reload_base_sgpr,
            static_cast<uint32_t>(preload.kernarg_reload_offset_dwords + i) * sizeof(uint32_t),
            arch);
        if (!load || read_word(cursor++) != (*load)[0] || read_word(cursor++) != (*load)[1]) {
          errors.emplace_back("ConSan final validation found an invalid dispatch-ID kernarg "
                              "reload in kernel '" +
                              std::string(kernel_name) + "'");
        }
      }
      const auto wait = instrumentation::build_s_wait_scalar_load0(arch);
      if (!wait || read_word(cursor++) != *wait) {
        errors.emplace_back("ConSan final validation found an invalid dispatch-ID kernarg wait "
                            "in kernel '" +
                            std::string(kernel_name) + "'");
      }
    }
  };

  for (const ConSanPatchInfo &patch : result.patches) {
    if (!patch.dispatch_id_prologue)
      continue;
    const auto owner =
        std::ranges::find_if(environment.original.kernels(), [&](const AmdGpuKernelInfo &kernel) {
          return kernel.entry_text_offset == patch.anchor_offset;
        });
    if (owner == environment.original.kernels().end()) {
      errors.emplace_back("ConSan final validation could not resolve a dispatch-ID prologue owner");
      continue;
    }
    const bool has_semantic_system_sgpr_restore =
        consan_detail::patch_requires_full_workgroup_id_payload(result.observation_plan().engine,
                                                                arch, patch);
    validate_entry(patch,
                   patch.dispatch_id_primary_prologue_offset.value_or(patch.trampoline_offset),
                   owner->name, has_semantic_system_sgpr_restore);
    if (patch.dispatch_id_secondary_prologue_offset) {
      validate_entry(
          patch,
          patch.dispatch_id_secondary_prologue_offset.value_or(patch.trampoline_offset + 256u),
          owner->name, has_semantic_system_sgpr_restore);
    }
  }
  return;
}

void validate_resource_and_metadata_deltas(const FinalValidationEnvironment &environment,
                                           const ConSanFinalValidationInput &result,
                                           std::vector<std::string> &errors) {
  using KernelDescriptor = rocr::llvm::amdhsa::kernel_descriptor_t;
  const size_t initial_error_count = errors.size();
  if (!environment.original_text.available || !environment.replacement_text.available)
    return;
  const AmdGpuCodeObject &original = environment.original;
  const AmdGpuCodeObject &replacement = environment.replacement;
  if (!environment.require_target(errors, "ConSan resource validation found no target profile"))
    return;
  const rj_code_arch_t arch = environment.arch();
  std::unordered_map<uint64_t, std::string> owner_names;
  for (const AmdGpuKernelInfo &kernel : original.kernels())
    owner_names.emplace(kernel.descriptor_file_offset, kernel.name);
  for (const AmdGpuKernelInfo &kernel : replacement.kernels())
    owner_names.emplace(kernel.descriptor_file_offset, kernel.name);
  std::vector<uint64_t> referenced_owner_offsets;
  for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
    const auto descriptors = result.program_inventory.kernel_descriptors(plan.owner_kernel_ids);
    if (!descriptors) {
      errors.emplace_back("ConSan final validation found a stale resource-plan kernel owner");
      continue;
    }
    referenced_owner_offsets.insert(referenced_owner_offsets.end(), descriptors->begin(),
                                    descriptors->end());
  }
  for (const ConSanPatchInfo &patch : result.patches)
    referenced_owner_offsets.insert(referenced_owner_offsets.end(),
                                    patch.owner_descriptor_file_offsets.begin(),
                                    patch.owner_descriptor_file_offsets.end());
  const uint64_t text_growth =
      replacement.text_sections().front()->size() >= original.text_sections().front()->size()
          ? replacement.text_sections().front()->size() - original.text_sections().front()->size()
          : 0;
  std::unordered_map<uint64_t, size_t> relocation_frequency;
  for (uint64_t owner : referenced_owner_offsets) {
    if (owner_names.contains(owner))
      continue;
    for (const AmdGpuKernelInfo &kernel : original.kernels()) {
      if (owner >= kernel.descriptor_file_offset &&
          owner - kernel.descriptor_file_offset <= text_growth) {
        ++relocation_frequency[owner - kernel.descriptor_file_offset];
      }
    }
  }
  for (uint64_t owner : referenced_owner_offsets) {
    if (owner_names.contains(owner))
      continue;
    const AmdGpuKernelInfo *best = nullptr;
    size_t best_frequency = 0;
    bool ambiguous = false;
    for (const AmdGpuKernelInfo &kernel : original.kernels()) {
      if (owner < kernel.descriptor_file_offset ||
          owner - kernel.descriptor_file_offset > text_growth)
        continue;
      const size_t frequency = relocation_frequency[owner - kernel.descriptor_file_offset];
      if (frequency > best_frequency) {
        best = &kernel;
        best_frequency = frequency;
        ambiguous = false;
      } else if (frequency == best_frequency && best && best->name != kernel.name) {
        ambiguous = true;
      }
    }
    if (best && !ambiguous)
      owner_names.emplace(owner, best->name);
  }
  std::unordered_map<std::string, DescriptorRequirement> requirements;
  const auto apply_to_owner = [&](uint64_t descriptor_offset, const auto &apply) {
    const auto owner = owner_names.find(descriptor_offset);
    if (owner == owner_names.end()) {
      bool saw_relocation_candidate = false;
      for (const AmdGpuKernelInfo &kernel : original.kernels()) {
        if (descriptor_offset >= kernel.descriptor_file_offset &&
            descriptor_offset - kernel.descriptor_file_offset <= text_growth) {
          saw_relocation_candidate = true;
        }
      }
      if (!saw_relocation_candidate) {
        errors.emplace_back("ConSan final validation found unknown descriptor owner offset " +
                            std::to_string(descriptor_offset));
      }
      return;
    }
    apply(requirements[owner->second]);
  };
  const auto plan_has_emitted_patch = [&](const ConSanCandidateResourcePlan &plan) {
    return std::ranges::any_of(result.patches, [&](const ConSanPatchInfo &patch) {
      return patch.phase == ConSanPatchPhase::Instrumentation &&
             patch.anchor_offset == plan.text_offset && patch.scratch_vgpr.has_value();
    });
  };
  for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
    if (!plan_has_emitted_patch(plan))
      continue;
    const auto descriptors = result.program_inventory.kernel_descriptors(plan.owner_kernel_ids);
    if (!descriptors) {
      errors.emplace_back("ConSan final validation found a stale emitted resource-plan owner");
      continue;
    }
    for (uint64_t owner : *descriptors) {
      apply_to_owner(owner, [&](DescriptorRequirement &requirement) {
        requirement.allow_resource_delta = true;
        requirement.required_vgpr_count =
            std::max(requirement.required_vgpr_count, plan.required_vgpr_count);
      });
    }
  }
  for (const ConSanPatchInfo &patch : result.patches) {
    const bool requires_full_workgroup_id = consan_detail::patch_requires_full_workgroup_id_payload(
        result.observation_plan().engine, arch, patch);
    if (patch.private_state_layout) {
      if (!patch.private_state_layout->is_well_formed()) {
        errors.emplace_back("ConSan final validation found an invalid private-state layout");
      }
      if (patch.required_private_segment_size < patch.private_state_layout->ephemeral_base) {
        errors.emplace_back(
            "ConSan final validation found private state beyond the patch's private-segment "
            "requirement");
      }
    }
    if (patch.moi_vgpr_state && !patch.moi_vgpr_state->is_well_formed()) {
      errors.emplace_back("ConSan final validation found an invalid MOI VGPR-state effect");
    }
    if (patch.scalar_vcc_spill && !patch.scalar_vcc_spill->is_well_formed()) {
      errors.emplace_back(
          "ConSan final validation found an invalid SuperCollider scalar-VCC spill effect");
    }
    if (patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue && !patch.moi_vgpr_state) {
      errors.emplace_back("ConSan final validation found an owner/epoch prologue without its "
                          "VGPR-state effect");
    }
    if (patch.dynamic_private_segment_addend != 0) {
      if (patch.required_private_segment_size < patch.dynamic_private_segment_addend) {
        errors.emplace_back(
            "ConSan final validation found a dynamic private addend larger than its absolute "
            "private requirement");
      }
      if (patch.owner_descriptor_file_offsets.empty()) {
        errors.emplace_back(
            "ConSan final validation found an unowned dynamic private-segment requirement");
      }
      for (uint64_t owner : patch.owner_descriptor_file_offsets) {
        const auto owner_name = owner_names.find(owner);
        const auto owner_kernel = owner_name == owner_names.end()
                                      ? original.kernels().end()
                                      : std::ranges::find(original.kernels(), owner_name->second,
                                                          &AmdGpuKernelInfo::name);
        if (owner_kernel == original.kernels().end() ||
            !owner_kernel->uses_dynamic_stack.value_or(false)) {
          errors.emplace_back(
              "ConSan final validation found a dynamic private-segment requirement on a "
              "non-dynamic owner");
        }
      }
    }
    for (uint64_t owner : patch.owner_descriptor_file_offsets) {
      apply_to_owner(owner, [&](DescriptorRequirement &requirement) {
        requirement.allow_resource_delta = true;
        requirement.require_full_workgroup_id |= requires_full_workgroup_id;
        requirement.required_private_bytes =
            std::max(requirement.required_private_bytes, patch.required_private_segment_size);
        requirement.required_group_bytes =
            std::max(requirement.required_group_bytes, patch.required_group_segment_size());
        requirement.required_sgpr_count =
            std::max(requirement.required_sgpr_count, patch.required_sgpr_count);
        requirement.allow_entry_delta |=
            patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue ||
            patch.kind == ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue;
        if (patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue) {
          if (patch.moi_vgpr_state && patch.moi_vgpr_state->is_well_formed())
            requirement.required_vgpr_count = std::max<uint16_t>(
                requirement.required_vgpr_count,
                static_cast<uint16_t>(patch.moi_vgpr_state->required_vgpr_count()));
        }
        // Both register-backed and private-state entry prologues may preserve
        // borrowed guest SGPRs in a lane-backed VGPR.  The proof is also a
        // descriptor resource requirement, regardless of prologue kind.
        if (patch.entry_scalar_backup) {
          requirement.required_vgpr_count =
              std::max<uint16_t>(requirement.required_vgpr_count,
                                 static_cast<uint16_t>(patch.entry_scalar_backup->vgpr + 1u));
        }
        if (patch.dispatch_id_prologue) {
          const ConSanMoiDispatchIdPrologueEffect &dispatch = *patch.dispatch_id_prologue;
          requirement.dispatch_id_preload = dispatch.preload;
          requirement.required_sgpr_count =
              std::max(requirement.required_sgpr_count, dispatch.required_sgpr_count());
          if (const std::optional<uint16_t> capture_vgpr = dispatch.capture.vgpr()) {
            requirement.required_vgpr_count = std::max<uint16_t>(
                requirement.required_vgpr_count, static_cast<uint16_t>(*capture_vgpr + 2u));
          }
        }
      });
    }
  }
  if (errors.size() != initial_error_count)
    return;

  uint16_t unscoped_required_vgprs = 0;
  uint32_t unscoped_required_private_bytes = 0;
  for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
    if (plan.owner_kernel_ids.empty() && plan_has_emitted_patch(plan))
      unscoped_required_vgprs = std::max(unscoped_required_vgprs, plan.required_vgpr_count);
  }
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.owner_descriptor_file_offsets.empty()) {
      unscoped_required_private_bytes =
          std::max(unscoped_required_private_bytes, patch.required_private_segment_size);
    }
  }

  // A whole-text transaction may repack translated kernels entirely inside
  // existing linker padding.  In that case the section size is unchanged even
  // though every affected descriptor receives a new entry coordinate.  The
  // typed relocation proof, rather than a size delta, is the authority for
  // permitting that otherwise-forbidden descriptor change.
  const bool text_relocated = result.text_relocation.has_value();
  std::unordered_map<std::string_view, const AmdGpuKernelInfo *> replacement_kernels_by_name;
  replacement_kernels_by_name.reserve(replacement.kernels().size());
  for (const AmdGpuKernelInfo &kernel : replacement.kernels())
    replacement_kernels_by_name.emplace(kernel.name, &kernel);
  // This predicate describes the whole transform, not an individual kernel.
  // Computing it inside the kernel loop made final validation quadratic in a
  // large library's kernel and patch counts.
  const bool has_unscoped_resource_growth =
      std::ranges::any_of(result.resource_plans,
                          [&](const ConSanCandidateResourcePlan &plan) {
                            return plan.owner_kernel_ids.empty() && plan_has_emitted_patch(plan) &&
                                   (plan.source ==
                                        ConSanRegisterAllocationSource::DescriptorGrowth ||
                                    plan.source == ConSanRegisterAllocationSource::SpillRequired);
                          }) ||
      std::ranges::any_of(result.patches, [](const ConSanPatchInfo &patch) {
        return patch.owner_descriptor_file_offsets.empty() && patch.scratch_vgpr.has_value();
      });
  for (const AmdGpuKernelInfo &original_kernel : original.kernels()) {
    const auto replacement_it = replacement_kernels_by_name.find(original_kernel.name);
    if (replacement_it == replacement_kernels_by_name.end())
      continue;
    const AmdGpuKernelInfo &replacement_kernel = *replacement_it->second;
    const auto original_descriptor_value = read_kernel_descriptor(
        std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(original.image_data()),
                                 original.image_size()),
        original_kernel.descriptor_file_offset);
    const auto replacement_descriptor_value = read_kernel_descriptor(
        std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(replacement.image_data()),
                                 replacement.image_size()),
        replacement_kernel.descriptor_file_offset);
    if (!original_descriptor_value || !replacement_descriptor_value) {
      errors.emplace_back("ConSan final validation could not read descriptor pair for kernel '" +
                          original_kernel.name + "'");
      continue;
    }
    const KernelDescriptor &original_descriptor = *original_descriptor_value;
    const KernelDescriptor &replacement_descriptor = *replacement_descriptor_value;
    DescriptorRequirement requirement = requirements[original_kernel.name];
    const bool descriptor_resources_changed =
        replacement_descriptor.compute_pgm_rsrc1 != original_descriptor.compute_pgm_rsrc1 ||
        replacement_descriptor.compute_pgm_rsrc2 != original_descriptor.compute_pgm_rsrc2 ||
        replacement_descriptor.private_segment_fixed_size !=
            original_descriptor.private_segment_fixed_size ||
        replacement_descriptor.group_segment_fixed_size !=
            original_descriptor.group_segment_fixed_size;
    if (descriptor_resources_changed) {
      requirement.required_vgpr_count =
          std::max(requirement.required_vgpr_count, unscoped_required_vgprs);
      requirement.required_private_bytes =
          std::max(requirement.required_private_bytes, unscoped_required_private_bytes);
    }
    KernelDescriptor normalized = replacement_descriptor;
    const bool requires_full_workgroup_id = requirement.require_full_workgroup_id;
    const auto normalize_workgroup_id_payload = [&](uint32_t &rsrc2) {
      AMDHSA_BITS_SET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X,
                      AMDHSA_BITS_GET(original_descriptor.compute_pgm_rsrc2,
                                      kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X));
      AMDHSA_BITS_SET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y,
                      AMDHSA_BITS_GET(original_descriptor.compute_pgm_rsrc2,
                                      kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y));
      AMDHSA_BITS_SET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z,
                      AMDHSA_BITS_GET(original_descriptor.compute_pgm_rsrc2,
                                      kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z));
    };
    if (requires_full_workgroup_id) {
      const bool full_workgroup_id =
          AMDHSA_BITS_GET(replacement_descriptor.compute_pgm_rsrc2,
                          kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X) != 0 &&
          AMDHSA_BITS_GET(replacement_descriptor.compute_pgm_rsrc2,
                          kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y) != 0 &&
          AMDHSA_BITS_GET(replacement_descriptor.compute_pgm_rsrc2,
                          kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z) != 0;
      if (!full_workgroup_id) {
        errors.emplace_back(
            "ConSan final validation found an incomplete workgroup-ID launch payload for "
            "kernel '" +
            original_kernel.name + "'");
      }
      normalize_workgroup_id_payload(normalized.compute_pgm_rsrc2);
    }
    if (requirement.dispatch_id_preload) {
      const ConSanMoiDispatchIdPreloadPlan &preload = *requirement.dispatch_id_preload;
      const uint32_t original_user_count = AMDHSA_BITS_GET(original_descriptor.compute_pgm_rsrc2,
                                                           kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT);
      const uint32_t replacement_user_count = AMDHSA_BITS_GET(
          replacement_descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT);
      const bool original_dispatch =
          AMDHSA_BITS_GET(original_descriptor.kernel_code_properties,
                          kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID) != 0;
      const bool replacement_dispatch =
          AMDHSA_BITS_GET(replacement_descriptor.kernel_code_properties,
                          kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID) != 0;
      const bool original_queue =
          AMDHSA_BITS_GET(original_descriptor.kernel_code_properties,
                          kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR) != 0;
      const bool replacement_queue =
          AMDHSA_BITS_GET(replacement_descriptor.kernel_code_properties,
                          kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR) != 0;
      if (original_user_count != preload.original_user_sgpr_count ||
          replacement_user_count != preload.expanded_user_sgpr_count ||
          original_dispatch != preload.dispatch_id_was_enabled || !replacement_dispatch ||
          original_queue != preload.queue_ptr_was_enabled || !replacement_queue) {
        errors.emplace_back(
            "ConSan final validation found an invalid dispatch-ID preload descriptor delta for "
            "kernel '" +
            original_kernel.name + "'");
      }
      const uint32_t original_preload_length =
          AMDHSA_BITS_GET(original_descriptor.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH);
      const uint32_t replacement_preload_length =
          AMDHSA_BITS_GET(replacement_descriptor.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH);
      if (original_preload_length != preload.original_kernarg_preload_length ||
          replacement_preload_length != preload.replacement_kernarg_preload_length) {
        errors.emplace_back(
            "ConSan final validation found an invalid dispatch-ID kernarg-preload delta for "
            "kernel '" +
            original_kernel.name + "'");
      }

      uint32_t normalized_rsrc2 = replacement_descriptor.compute_pgm_rsrc2;
      AMDHSA_BITS_SET(normalized_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, original_user_count);
      if (requires_full_workgroup_id)
        normalize_workgroup_id_payload(normalized_rsrc2);
      if (requirement.required_private_bytes != 0) {
        AMDHSA_BITS_SET(normalized_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT,
                        AMDHSA_BITS_GET(original_descriptor.compute_pgm_rsrc2,
                                        kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT));
      }
      if (normalized_rsrc2 != original_descriptor.compute_pgm_rsrc2) {
        errors.emplace_back(
            "ConSan final validation found an extra RSRC2 delta beside dispatch-ID insertion "
            "for kernel '" +
            original_kernel.name + "'");
      }

      uint32_t normalized_rsrc1 = replacement_descriptor.compute_pgm_rsrc1;
      AMDHSA_BITS_SET(normalized_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                      AMDHSA_BITS_GET(original_descriptor.compute_pgm_rsrc1,
                                      kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT));
      AMDHSA_BITS_SET(normalized_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                      AMDHSA_BITS_GET(original_descriptor.compute_pgm_rsrc1,
                                      kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT));
      if (normalized_rsrc1 != original_descriptor.compute_pgm_rsrc1) {
        errors.emplace_back(
            "ConSan final validation found an extra RSRC1 delta beside register allocation for "
            "kernel '" +
            original_kernel.name + "'");
      }

      AMDHSA_BITS_SET(normalized.kernel_code_properties,
                      kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID,
                      AMDHSA_BITS_GET(original_descriptor.kernel_code_properties,
                                      kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID));
      AMDHSA_BITS_SET(normalized.kernel_code_properties,
                      kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR,
                      AMDHSA_BITS_GET(original_descriptor.kernel_code_properties,
                                      kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR));
      if (preload.original_kernarg_preload_length != 0) {
        AMDHSA_BITS_SET(normalized.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH,
                        preload.original_kernarg_preload_length);
      }
    }
    if (text_relocated || requirement.allow_entry_delta)
      normalized.kernel_code_entry_byte_offset = original_descriptor.kernel_code_entry_byte_offset;
    const ConSanDescriptorResourceDeltaValidation target_resource_validation =
        validate_consan_descriptor_resource_delta(
            arch, {
                      .original_rsrc1 = original_descriptor.compute_pgm_rsrc1,
                      .replacement_rsrc1 = replacement_descriptor.compute_pgm_rsrc1,
                      .original_rsrc3 = original_descriptor.compute_pgm_rsrc3,
                      .replacement_rsrc3 = replacement_descriptor.compute_pgm_rsrc3,
                      .required_vgpr_count = requirement.required_vgpr_count,
                      .original_accumulator_bank_proven_empty = original_kernel.agpr_count == 0u,
                      .allow_resource_delta = requirement.allow_resource_delta,
                      .normalize_resource_delta =
                          requirement.allow_resource_delta || has_unscoped_resource_growth,
                  });
    if (!target_resource_validation.valid) {
      errors.emplace_back(
          "ConSan final validation found an invalid accumulator-boundary move for kernel '" +
          original_kernel.name + "'");
    }
    if (requirement.allow_resource_delta || has_unscoped_resource_growth) {
      normalized.compute_pgm_rsrc1 = original_descriptor.compute_pgm_rsrc1;
      normalized.compute_pgm_rsrc2 = original_descriptor.compute_pgm_rsrc2;
      normalized.compute_pgm_rsrc3 = target_resource_validation.normalized_rsrc3;
      normalized.private_segment_fixed_size = original_descriptor.private_segment_fixed_size;
      normalized.group_segment_fixed_size = original_descriptor.group_segment_fixed_size;
    }
    if (text_relocated && !requirement.allow_resource_delta &&
        !has_unscoped_resource_growth) {
      const auto &descriptor_deltas = result.text_relocation->descriptor_rsrc1_deltas;
      const auto delta = std::ranges::find(
          descriptor_deltas, original_kernel.descriptor_file_offset,
          &ConSanTextRelocationDescriptorDelta::descriptor_file_offset);
      if (delta != descriptor_deltas.end()) {
        uint32_t normalized_rsrc1 = delta->replacement_compute_pgm_rsrc1;
        AMDHSA_BITS_SET(normalized_rsrc1,
                        kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                        AMDHSA_BITS_GET(delta->original_compute_pgm_rsrc1,
                                        kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT));
        if (delta->original_compute_pgm_rsrc1 != original_descriptor.compute_pgm_rsrc1 ||
            delta->replacement_compute_pgm_rsrc1 != replacement_descriptor.compute_pgm_rsrc1 ||
            normalized_rsrc1 != original_descriptor.compute_pgm_rsrc1) {
          errors.emplace_back(
              "ConSan final validation found an invalid relocation-owned descriptor delta for "
              "kernel '" +
              original_kernel.name + "'");
        } else {
          normalized.compute_pgm_rsrc1 = original_descriptor.compute_pgm_rsrc1;
        }
      }
    }
    if (std::memcmp(&normalized, &original_descriptor, sizeof(normalized)) != 0) {
      const auto *normalized_bytes = reinterpret_cast<const uint8_t *>(&normalized);
      const auto *original_descriptor_bytes =
          reinterpret_cast<const uint8_t *>(&original_descriptor);
      const size_t first_delta = static_cast<size_t>(
          std::mismatch(normalized_bytes, normalized_bytes + sizeof(normalized),
                        original_descriptor_bytes)
              .first -
          normalized_bytes);
      errors.emplace_back(
          "ConSan final validation found an unplanned descriptor delta for kernel '" +
          original_kernel.name + "' (original offset " +
          std::to_string(original_kernel.descriptor_file_offset) + ", replacement offset " +
          std::to_string(replacement_kernel.descriptor_file_offset) + ", first byte " +
          std::to_string(first_delta) + ": original=" +
          std::to_string(original_descriptor_bytes[first_delta]) + ", replacement=" +
          std::to_string(normalized_bytes[first_delta]) + ")");
      continue;
    }
    if (requirement.required_vgpr_count != 0) {
      const std::optional<uint16_t> actual_vgprs = read_descriptor_vgpr_allocation_count(
          result.replacement, replacement_kernel.descriptor_file_offset, arch);
      if (!actual_vgprs || *actual_vgprs < requirement.required_vgpr_count) {
        errors.emplace_back(
            "ConSan final validation found insufficient descriptor VGPRs for kernel '" +
            original_kernel.name + "'");
      }
    }
    const uint16_t replacement_sgpr_count =
        descriptor_sgpr_allocation_count(replacement_descriptor, arch);
    if (requirement.required_sgpr_count != 0 &&
        replacement_sgpr_count < requirement.required_sgpr_count) {
      errors.emplace_back(
          "ConSan final validation found insufficient descriptor SGPRs for kernel '" +
          original_kernel.name + "': required=" + std::to_string(requirement.required_sgpr_count) +
          " allocated=" + std::to_string(replacement_sgpr_count));
    }
    if (requirement.required_group_bytes != 0 &&
        replacement_descriptor.group_segment_fixed_size != requirement.required_group_bytes) {
      errors.emplace_back(
          "ConSan final validation found inconsistent fixed-LDS reservation for kernel '" +
          original_kernel.name + "'");
    }
    if (requirement.required_private_bytes == 0)
      continue;
    // Probe only the descriptor under validation. Copying the complete patched
    // image once per spill-owning kernel made this check scale as image bytes
    // times owner count for large generated libraries.
    std::vector<uint8_t> descriptor_probe(sizeof(replacement_descriptor));
    const major_image_ownership::ScopedOwner descriptor_probe_owner(
        major_image_ownership::OwnerKind::DescriptorProbe, descriptor_probe);
    if (!write_kernel_descriptor(descriptor_probe, 0u, replacement_descriptor)) {
      errors.emplace_back("ConSan final validation could not construct its descriptor probe");
      continue;
    }
    const SpillDescriptorUpdate descriptor_update = update_kernel_descriptor_for_spills(
        descriptor_probe, /*descriptor_file_offset=*/0, requirement.required_private_bytes);
    if (descriptor_update != SpillDescriptorUpdate::Unchanged) {
      errors.emplace_back(
          "ConSan final validation found insufficient spill descriptor state for kernel '" +
          original_kernel.name + "' (required " +
          std::to_string(requirement.required_private_bytes) + ", found " +
          std::to_string(replacement_descriptor.private_segment_fixed_size) + ")");
    }
  }
  return;
}

[[nodiscard]] std::vector<std::string>
validate_final_consan_elf(std::span<const uint8_t> original_bytes,
                          const ConSanFinalValidationInput &result,
                          uint64_t expected_moi_report_dispatch_id,
                          std::optional<ConSanTransformFailureCause> *failure_cause = nullptr) {
  std::vector<std::string> errors;
  const major_image_ownership::ScopedPhase validation_phase(
      major_image_ownership::Phase::FinalValidation, /*only_if_none=*/true);
  AmdGpuCodeObject original(original_bytes.data(), original_bytes.size());
  AmdGpuCodeObject replacement(result.replacement.data(), result.replacement.size());
  if (!original.is_valid() || !replacement.is_valid()) {
    errors.emplace_back("ConSan final validation could not parse original and replacement ELFs");
    return errors;
  }

  const bool redirects_entry =
      std::ranges::any_of(result.patches, [](const ConSanPatchInfo &patch) {
        return patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue ||
               patch.kind == ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue;
      });
  errors = validate_consan_input_layout(replacement, redirects_entry || result.text_relocation);
  for (std::string &error : errors)
    error = "ConSan final validation: " + error;

  if (replacement.target_id() != original.target_id()) {
    errors.emplace_back("ConSan final validation changed the AMDGPU target identity");
    // Every remaining proof decodes or synthesizes instructions for the
    // replacement target. A corrupted target ID is already a conclusive
    // validation failure and must not escape as an instruction-builder
    // exception while validating an untrusted replacement image.
    return errors;
  }
  if (sorted_entity_names<AmdGpuKernelInfo>(replacement.kernels()) !=
      sorted_entity_names<AmdGpuKernelInfo>(original.kernels())) {
    errors.emplace_back("ConSan final validation changed the kernel symbol identity set");
  }
  if (sorted_entity_names<AmdGpuFunctionInfo>(replacement.functions()) !=
      sorted_entity_names<AmdGpuFunctionInfo>(original.functions())) {
    errors.emplace_back("ConSan final validation changed the function symbol identity set");
  }

  const std::vector<TextSectionIdentity> original_text = text_section_identities(original);
  const std::vector<TextSectionIdentity> replacement_text = text_section_identities(replacement);
  if (original_text.size() != replacement_text.size()) {
    errors.emplace_back("ConSan final validation changed the executable section count");
  } else {
    for (size_t i = 0; i < original_text.size(); ++i) {
      if (original_text[i].name != replacement_text[i].name ||
          original_text[i].virtual_address != replacement_text[i].virtual_address) {
        errors.emplace_back("ConSan final validation changed executable section identity");
        break;
      }
      if (replacement_text[i].size < original_text[i].size) {
        errors.emplace_back("ConSan final validation unexpectedly shrank executable text");
        break;
      }
    }
  }
  if (result.replacement.size() == original_bytes.size() &&
      std::ranges::equal(result.replacement, original_bytes)) {
    errors.emplace_back("ConSan final validation found no byte change in a modified result");
  }
  if (result.patches.empty())
    errors.emplace_back("ConSan final validation found no patch inventory for a modified result");
  FinalValidationEnvironment environment(original, replacement);
  const bool validates_mutation = has_consan_patch_phase(result, ConSanPatchPhase::Mutation);
  const bool validates_perturbation =
      std::ranges::any_of(result.patches, [](const ConSanPatchInfo &patch) {
        return patch.kind == ConSanPatchKind::TrampolineScPerturbation;
      });
  std::optional<ConSanPristineValidationInventory> pristine_inventory;
  if (validates_mutation || validates_perturbation) {
    pristine_inventory.emplace(
        rederive_consan_pristine_validation_inventory(original_bytes, validates_mutation));
  }
  validate_patch_byte_accounting(environment, result, errors, failure_cause);
  validate_patch_decoding(environment, result, errors);
  validate_mutation_semantics(environment, result,
                              pristine_inventory ? &*pristine_inventory : nullptr, errors);
  consan_validation_detail::validate_supercollider_final_semantics(
      environment, result, pristine_inventory ? &*pristine_inventory : nullptr, errors);
  validate_resource_and_metadata_deltas(environment, result, errors);
  validate_entry_scalar_backup_semantics(environment, result, errors);
  validate_dispatch_id_prologue_semantics(environment, result, errors);
  consan_validation_detail::validate_inline_shadow_final_semantics(
      environment, result, expected_moi_report_dispatch_id, errors);
  return errors;
}

[[nodiscard]] ConSanTransformArtifacts
finalize_consan_result_impl(ConSanTransformArtifacts result,
                            std::span<const uint8_t> original_bytes,
                            uint64_t expected_moi_report_dispatch_id) {
  const major_image_ownership::ScopedOwner result_owner(
      major_image_ownership::OwnerKind::ResultImage, result.replacement);
  if (result.program_inventory.empty()) {
    ProgramInventoryBuilder inventory_builder(original_bytes);
    result.program_inventory = inventory_builder.view();
  }
  if (has_blocked_kernel(result.program_inventory)) {
    result.outcome = ConSanTransformOutcome::Unsupported;
    result.discard_candidate_modification();
    result.mutation.applied_fault_logical_identity.reset();
    return result;
  }
  if (!result.errors.empty()) {
    result.outcome = ConSanTransformOutcome::Invalid;
    result.discard_candidate_modification();
    result.mutation.applied_fault_logical_identity.reset();
    return result;
  }
  // A required late transform (notably the owner/epoch entry prologue) may
  // discover an architectural resource boundary after earlier probes have
  // already produced a replacement image. Never allow those partial probes
  // to turn an Unsupported result back into ModifiedValid.
  if (result.outcome == ConSanTransformOutcome::Unsupported) {
    result.discard_candidate_modification();
    result.mutation.applied_fault_logical_identity.reset();
    return result;
  }
  if (result.modified()) {
    if (result.replacement.empty()) {
      result.errors.emplace_back("ConSan modified result has no replacement ELF bytes");
      result.outcome = ConSanTransformOutcome::Invalid;
      result.mutation.applied_fault_logical_identity.reset();
    } else {
      std::vector<std::string> validation_errors = validate_final_consan_elf(
          original_bytes, ConSanFinalValidationInput(result), expected_moi_report_dispatch_id,
          &result.transform_failure_cause);
      result.errors.insert(result.errors.end(), std::make_move_iterator(validation_errors.begin()),
                           std::make_move_iterator(validation_errors.end()));
      if (result.errors.empty()) {
        result.outcome = ConSanTransformOutcome::ModifiedValid;
      } else {
        result.outcome = ConSanTransformOutcome::Invalid;
        result.mutation.applied_fault_logical_identity.reset();
        result.discard_candidate_modification();
      }
    }
    return result;
  }
  if (result.outcome == ConSanTransformOutcome::Unsupported ||
      summarize_consan_resource_plans(result.resource_plans).unsupported_plans != 0) {
    result.outcome = ConSanTransformOutcome::Unsupported;
    result.mutation.applied_fault_logical_identity.reset();
  } else {
    result.outcome = ConSanTransformOutcome::Unchanged;
  }
  return result;
}

} // namespace

ConSanTransformArtifacts finalize_consan_result(ConSanTransformArtifacts result,
                                                std::span<const uint8_t> original_bytes,
                                                uint64_t expected_moi_report_dispatch_id) {
  return finalize_consan_result_impl(std::move(result), original_bytes,
                                     expected_moi_report_dispatch_id);
}

std::vector<std::string>
validate_consan_modified_elf(std::span<const uint8_t> original_bytes,
                             const ConSanTransformArtifacts &modified_result,
                             uint64_t expected_moi_report_dispatch_id) {
  return validate_final_consan_elf(original_bytes, ConSanFinalValidationInput(modified_result),
                                   expected_moi_report_dispatch_id);
}

} // namespace rocjitsu

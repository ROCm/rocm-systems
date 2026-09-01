// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This retained `.inc` path is compiled directly as a C++ translation unit.
// It is never textually included; the filename remains only because repository
// files may not be deleted or moved during this refactoring.

#include "rocjitsu/code/patch/consan/consan_placement.h"

#include "rocjitsu/analysis/def_use_chain.h"
#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/patch/consan/consan_instruction_semantics.h"
#include "rocjitsu/code/patch/consan/consan_runtime_kernel.h"
#include "rocjitsu/code/patch/consan/consan_semantic_classifiers.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"
#include "rocjitsu/code/patch/instrumentor.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <ranges>

namespace rocjitsu {

[[nodiscard]] std::optional<uint64_t>
text_offset_to_file_offset(const AmdGpuCodeObject &code_object, uint64_t text_offset,
                           uint64_t byte_count) {
  for (const Section *section : code_object.text_sections()) {
    if (text_offset > section->size())
      continue;
    if (byte_count > section->size() - text_offset)
      continue;
    return section->sectionOffset() + text_offset;
  }
  return std::nullopt;
}

[[nodiscard]] bool ranges_overlap(ByteRange lhs, ByteRange rhs) {
  return lhs.begin < rhs.end && rhs.begin < lhs.end;
}

[[nodiscard]] bool overlaps_preapplied_reserved_range(const ProgramInventory &inventory,
                                                      ByteRange range) {
  return std::ranges::any_of(
      inventory.preapplied_mutation().reserved_ranges,
      [&](const ConSanPreappliedReservedRange &reserved) {
        if (reserved.size == 0 ||
            reserved.text_offset > std::numeric_limits<uint64_t>::max() - reserved.size)
          return false;
        return ranges_overlap(range, {reserved.text_offset, reserved.text_offset + reserved.size});
      });
}

[[nodiscard]] bool overlaps_reserved_range(std::span<const ByteRange> ranges, ByteRange range) {
  for (ByteRange reserved : ranges) {
    if (ranges_overlap(reserved, range))
      return true;
  }
  return false;
}

[[nodiscard]] bool reserve_byte_range(std::vector<ByteRange> &ranges, ByteRange range) {
  if (range.begin >= range.end || overlaps_reserved_range(ranges, range))
    return false;
  ranges.push_back(range);
  return true;
}

[[nodiscard]] std::optional<std::vector<ByteRange>>
reserved_ranges_for_existing_patches(const AmdGpuCodeObject &code_object,
                                     std::span<const ConSanPatchInfo> patches) {
  std::vector<ByteRange> ranges;
  for (const ConSanCommittedPatchGeometry &patch : patches) {
    // Some structural records describe appended dispatch machinery without
    // replacing pristine text.  Their zero-byte anchors are identities, not
    // reservations, and must not make a later composed patch fail closed.
    if (patch.original_size != 0) {
      const auto anchor_file_offset =
          text_offset_to_file_offset(code_object, patch.anchor_offset, patch.original_size);
      if (!anchor_file_offset)
        return std::nullopt;
      if (!reserve_byte_range(ranges,
                              {*anchor_file_offset, *anchor_file_offset + patch.original_size}))
        return std::nullopt;
    }

    if (patch.trampoline_size == 0)
      continue;
    const auto trampoline_file_offset =
        text_offset_to_file_offset(code_object, patch.trampoline_offset, patch.trampoline_size);
    // Appended bodies are outside the pristine image and cannot overlap the
    // inline/local-only flat selector. They are reserved later against the
    // current composed text image.
    if (!trampoline_file_offset)
      continue;
    if (!reserve_byte_range(
            ranges, {*trampoline_file_offset, *trampoline_file_offset + patch.trampoline_size}))
      return std::nullopt;
  }
  return ranges;
}

struct TextRange {
  uint64_t begin = 0;
  uint64_t end = 0;
};

struct LocalCaveOwnerRange {
  uint64_t entry = 0;
  uint64_t code_end = 0;
  uint64_t ownership_end = 0;
};

void add_covered_text_range(std::vector<TextRange> &ranges, uint64_t begin, uint64_t size,
                            uint64_t text_size) {
  if (size == 0 || begin >= text_size)
    return;
  const uint64_t end = std::min<uint64_t>(text_size, begin + std::min(size, text_size - begin));
  if (begin < end)
    ranges.push_back({begin, end});
}

[[nodiscard]] std::vector<TextRange> covered_text_ranges(const ProgramInventory &inventory,
                                                         uint64_t text_size) {
  std::vector<TextRange> ranges;
  ranges.reserve(inventory.kernels().size() + inventory.functions().size());
  for (const ConSanKernelInfo &kernel : inventory.kernels()) {
    if (kernel.has_text_range)
      add_covered_text_range(ranges, kernel.entry_text_offset, kernel.code_size, text_size);
  }
  for (const ConSanFunctionInfo &function : inventory.functions())
    add_covered_text_range(ranges, function.entry_text_offset, function.code_size, text_size);

  std::ranges::sort(ranges, [](const TextRange &lhs, const TextRange &rhs) {
    if (lhs.begin != rhs.begin)
      return lhs.begin < rhs.begin;
    return lhs.end < rhs.end;
  });

  std::vector<TextRange> merged;
  for (const TextRange &range : ranges) {
    if (merged.empty() || range.begin > merged.back().end) {
      merged.push_back(range);
    } else {
      merged.back().end = std::max(merged.back().end, range.end);
    }
  }
  return merged;
}

[[nodiscard]] bool text_offset_is_covered(uint64_t offset, std::span<const TextRange> ranges) {
  for (const TextRange &range : ranges) {
    if (offset < range.begin)
      return false;
    if (offset < range.end)
      return true;
  }
  return false;
}

[[nodiscard]] std::vector<LocalCaveOwnerRange>
local_cave_owner_ranges(const ProgramInventory &inventory, uint64_t text_size) {
  std::vector<LocalCaveOwnerRange> owners;
  owners.reserve(inventory.kernels().size() + inventory.functions().size());
  auto add_owner = [&](uint64_t entry, uint64_t code_size) {
    if (code_size == 0 || entry >= text_size)
      return;
    const uint64_t code_end = entry + std::min(code_size, text_size - entry);
    owners.push_back({entry, code_end, text_size});
  };
  for (const ConSanKernelInfo &kernel : inventory.kernels()) {
    if (kernel.has_text_range)
      add_owner(kernel.entry_text_offset, kernel.code_size);
  }
  for (const ConSanFunctionInfo &function : inventory.functions())
    add_owner(function.entry_text_offset, function.code_size);

  std::ranges::sort(owners, [](const LocalCaveOwnerRange &lhs, const LocalCaveOwnerRange &rhs) {
    if (lhs.entry != rhs.entry)
      return lhs.entry < rhs.entry;
    return lhs.code_end < rhs.code_end;
  });
  std::vector<LocalCaveOwnerRange> unique_owners;
  for (const LocalCaveOwnerRange &owner : owners) {
    if (!unique_owners.empty() && unique_owners.back().entry == owner.entry) {
      unique_owners.back().code_end = std::max(unique_owners.back().code_end, owner.code_end);
      continue;
    }
    unique_owners.push_back(owner);
  }
  for (size_t i = 0; i + 1 < unique_owners.size(); ++i)
    unique_owners[i].ownership_end = unique_owners[i + 1].entry;
  return unique_owners;
}

[[nodiscard]] const LocalCaveOwnerRange *
owner_of_uncovered_text_range(uint64_t begin, uint64_t end,
                              std::span<const LocalCaveOwnerRange> owners) {
  for (const LocalCaveOwnerRange &owner : owners) {
    // Uncovered alignment/padding after a symbol belongs only to that symbol,
    // up to (but never across) the next kernel/function entry.
    if (begin >= owner.code_end && end <= owner.ownership_end)
      return &owner;
  }
  return nullptr;
}

[[nodiscard]] std::vector<LocalNopCave>
find_uncovered_nop_caves(const AmdGpuCodeObject &code_object, const ProgramInventory &inventory,
                         rj_code_arch_t arch) {
  if (code_object.text_sections().size() != 1)
    return {};

  const Section *text = code_object.text_sections().front();
  const auto text_bytes =
      std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(text->data()), text->size());
  const std::vector<TextRange> ranges = covered_text_ranges(inventory, text->size());
  const std::vector<LocalCaveOwnerRange> owners = local_cave_owner_ranges(inventory, text->size());
  const uint32_t nop = build_s_nop(0, arch);

  std::vector<LocalNopCave> caves;
  for (uint64_t text_offset = 0; text_offset + sizeof(uint32_t) <= text_bytes.size();) {
    if (text_offset_is_covered(text_offset, ranges)) {
      text_offset += sizeof(uint32_t);
      continue;
    }

    uint32_t word = 0;
    std::memcpy(&word, text_bytes.data() + text_offset, sizeof(word));
    if (word != nop) {
      text_offset += sizeof(uint32_t);
      continue;
    }

    const uint64_t run_begin = text_offset;
    uint32_t run_words = 0;
    while (text_offset + sizeof(uint32_t) <= text_bytes.size() &&
           !text_offset_is_covered(text_offset, ranges)) {
      std::memcpy(&word, text_bytes.data() + text_offset, sizeof(word));
      if (word != nop)
        break;
      ++run_words;
      text_offset += sizeof(uint32_t);
    }
    const uint64_t run_end = run_begin + static_cast<uint64_t>(run_words) * sizeof(uint32_t);
    const LocalCaveOwnerRange *owner = owner_of_uncovered_text_range(run_begin, run_end, owners);
    if (owner != nullptr) {
      caves.push_back({run_begin, text->sectionOffset() + run_begin, run_words, owner->entry});
    }
  }

  return caves;
}

[[nodiscard]] std::optional<InPlaceNopSite> find_existing_nop_site(const AmdGpuCodeObject &obj,
                                                                   rj_code_arch_t arch) {
  if (obj.text_sections().size() != 1)
    return std::nullopt;

  std::unique_ptr<Decoder> decoder = Decoder::create(arch);
  if (!decoder)
    return std::nullopt;

  const Section *text = obj.text_sections().front();
  auto blocks = BasicBlock::build(obj, *decoder, arch);
  for (const auto &block : blocks) {
    uint64_t offset = block->start_offset();
    for (const Instruction &inst : block->instructions()) {
      if (inst.size() == static_cast<int>(sizeof(uint32_t)) && inst.mnemonic() == "s_nop") {
        InPlaceNopSite site;
        site.text_offset = offset;
        site.file_offset = text->sectionOffset() + offset;
        return site;
      }
      offset += static_cast<uint64_t>(inst.size());
    }
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<InPlaceInstructionSite>
find_preferred_in_place_instruction_site(const AmdGpuCodeObject &obj, rj_code_arch_t arch) {
  if (obj.text_sections().size() != 1)
    return std::nullopt;

  std::unique_ptr<Decoder> decoder = Decoder::create(arch);
  if (!decoder)
    return std::nullopt;

  const Section *text = obj.text_sections().front();
  auto blocks = BasicBlock::build(obj, *decoder, arch);
  for (const auto &block : blocks) {
    uint64_t offset = block->start_offset();
    for (const Instruction &inst : block->instructions()) {
      if (inst.size() == static_cast<int>(sizeof(uint32_t)) && is_preferred_proof_anchor(inst)) {
        InPlaceInstructionSite site;
        site.text_offset = offset;
        site.file_offset = text->sectionOffset() + offset;
        site.size = sizeof(uint32_t);
        return site;
      }
      offset += static_cast<uint64_t>(inst.size());
    }
  }
  return std::nullopt;
}

[[nodiscard]] bool rewrite_word_in_place(const AmdGpuCodeObject &code_object, uint64_t file_offset,
                                         uint32_t replacement, ConSanTransformArtifacts &result) {
  result.replacement.assign(reinterpret_cast<const uint8_t *>(code_object.image_data()),
                            reinterpret_cast<const uint8_t *>(code_object.image_data()) +
                                code_object.image_size());
  if (file_offset > result.replacement.size() ||
      sizeof(replacement) > result.replacement.size() - file_offset) {
    result.errors.emplace_back("ConSan proof rewrite offset is outside ELF bytes");
    result.replacement.clear();
    return false;
  }
  std::memcpy(result.replacement.data() + file_offset, &replacement, sizeof(replacement));
  result.mark_modified();
  return true;
}

[[nodiscard]] std::optional<uint64_t> find_first_relocatable_anchor(const AmdGpuCodeObject &obj,
                                                                    rj_code_arch_t arch,
                                                                    std::string *error_out) {
  if (obj.text_sections().empty()) {
    *error_out = "code object has no .text section";
    return std::nullopt;
  }
  if (obj.text_sections().size() > 1) {
    *error_out = "code object has multiple .text sections";
    return std::nullopt;
  }

  std::unique_ptr<Decoder> decoder = Decoder::create(arch);
  if (!decoder) {
    *error_out = "no decoder available for proof NOP patch";
    return std::nullopt;
  }

  const Section *text = obj.text_sections().front();
  const std::span<const uint8_t> text_bytes(reinterpret_cast<const uint8_t *>(text->data()),
                                            text->size());
  auto blocks = BasicBlock::build(obj, *decoder, arch);
  std::optional<uint64_t> fallback_anchor;
  for (const auto &block : blocks) {
    uint64_t offset = block->start_offset();
    uint32_t clause_remaining = 0;
    for (const Instruction &inst : block->instructions()) {
      const bool blocked_by_clause = clause_remaining > 0;
      if (clause_remaining > 0)
        --clause_remaining;
      if (is_s_clause(inst))
        clause_remaining = s_clause_following_instruction_count(inst);
      if (!blocked_by_clause && inst.size() == static_cast<int>(sizeof(uint32_t)) &&
          !has_unsafe_proof_trampoline_flags(inst) &&
          is_relocatable_anchor(inst, offset, text_bytes, arch)) {
        if (!fallback_anchor)
          fallback_anchor = offset;
        if (is_preferred_proof_anchor(inst))
          return offset;
      }
      offset += static_cast<uint64_t>(inst.size());
    }
  }

  if (fallback_anchor)
    return fallback_anchor;

  *error_out = "no relocatable instruction anchor found for proof NOP patch";
  return std::nullopt;
}

bool is_relocatable_consan_barrier_destination(const Instruction &instruction, uint64_t offset,
                                               std::span<const uint8_t> text, rj_code_arch_t arch,
                                               std::string *error_out) {
  const int size = instruction.size();
  if (size == 4 || size == 8)
    return is_relocatable_anchor(instruction, offset, text, arch, error_out);
  if (offset % sizeof(uint32_t) != 0 || size <= 0 || size % sizeof(uint32_t) != 0) {
    *error_out = "wide destination must have a positive dword-aligned size and offset";
    return false;
  }
  const uint64_t size_u = static_cast<uint64_t>(size);
  if (offset > text.size() || size_u > text.size() - offset ||
      instruction.raw_encoding() == nullptr) {
    *error_out = "wide destination encoding is unavailable or exceeds .text";
    return false;
  }
  constexpr uint64_t kControlFlowFlags =
      BRANCH | COND_BRANCH | INDIRECT_BRANCH | INDIRECT_CALL | PROGRAM_TERMINATOR;
  if ((instruction.flags() & kControlFlowFlags) || instruction.branch_offset_bytes() ||
      is_s_clause(instruction)) {
    *error_out = "wide destination has control-flow or clause semantics";
    return false;
  }
  return true;
}

[[nodiscard]] std::optional<uint16_t> lds_dword_count(const ConSanAccessInventorySite &access) {
  if (!access.lowering.form || access.lowering.form->data_register_count == 0u)
    return std::nullopt;
  return access.lowering.form->data_register_count;
}

[[nodiscard]] bool vgpr_ranges_overlap(uint16_t lhs_base, uint16_t lhs_count, uint16_t rhs_base,
                                       uint16_t rhs_count) {
  const uint32_t lhs_end = static_cast<uint32_t>(lhs_base) + lhs_count;
  const uint32_t rhs_end = static_cast<uint32_t>(rhs_base) + rhs_count;
  return static_cast<uint32_t>(lhs_base) < rhs_end && static_cast<uint32_t>(rhs_base) < lhs_end;
}

/// Apply the classifier-owned alignment of the duplicated data tuple.
/// Architecture and exact instruction spelling have already been normalized
/// into the form, so scratch planning cannot rediscover a target exception.
[[nodiscard]] bool lds_scratch_tuple_base_is_valid(const ConSanAccessInventorySite &access,
                                                   uint16_t candidate) {
  return !access.lowering.form || access.lowering.form->data_register_alignment <= 1u ||
         candidate % access.lowering.form->data_register_alignment == 0u;
}

[[nodiscard]] bool lds_load_clobbers_address(const ConSanAccessInventorySite &access) {
  if (!access.lowering.form)
    return false;
  const ConSanAccessLoweringForm &form = *access.lowering.form;
  if (form.access_kind != ConSanLdsAccessKind::Read || !form.address_vgpr || !form.destination_vgpr)
    return false;
  return vgpr_ranges_overlap(*form.address_vgpr, form.address_vgpr_count, *form.destination_vgpr,
                             form.destination_register_count);
}

[[nodiscard]] bool is_forbidden_scratch_vgpr_run(const ConSanAccessInventorySite &access,
                                                 uint16_t candidate, uint16_t required_vgprs) {
  if (!access.lowering.form)
    return true;
  const ConSanAccessLoweringForm &form = *access.lowering.form;
  if (form.address_vgpr &&
      vgpr_ranges_overlap(candidate, required_vgprs, *form.address_vgpr, form.address_vgpr_count))
    return true;

  if (form.destination_vgpr &&
      vgpr_ranges_overlap(candidate, required_vgprs, *form.destination_vgpr,
                          form.destination_register_count))
    return true;
  const uint16_t data_vgprs =
      form.second_data_vgpr ? form.element_register_count : form.data_register_count;
  if (form.data_vgpr && vgpr_ranges_overlap(candidate, required_vgprs, *form.data_vgpr, data_vgprs))
    return true;
  if (form.second_data_vgpr &&
      vgpr_ranges_overlap(candidate, required_vgprs, *form.second_data_vgpr, data_vgprs))
    return true;
  return false;
}

[[nodiscard]] uint16_t lds_scratch_search_start(const ConSanAccessInventorySite &access) {
  if (!access.lowering.form)
    return 0u;
  const ConSanAccessLoweringForm &form = *access.lowering.form;
  std::optional<uint32_t> first_after_operands;
  auto note_range = [&first_after_operands](std::optional<uint16_t> base, uint16_t count) {
    if (!base)
      return;
    const uint32_t end = static_cast<uint32_t>(*base) + count;
    if (!first_after_operands || end > *first_after_operands)
      first_after_operands = end;
  };

  note_range(form.address_vgpr, form.address_vgpr_count);
  note_range(form.destination_vgpr, form.destination_register_count);
  const uint16_t data_vgprs =
      form.second_data_vgpr ? form.element_register_count : form.data_register_count;
  note_range(form.data_vgpr, data_vgprs);
  note_range(form.second_data_vgpr, data_vgprs);
  if (!first_after_operands)
    return 0;
  return *first_after_operands <= 255 ? static_cast<uint16_t>(*first_after_operands) : 256;
}

[[nodiscard]] bool
needs_scratch_headroom_at_descriptor_edge(const ConSanAccessInventorySite &access) {
  return access.lowering.form && access.lowering.form->destination_allocation_headroom != 0u;
}

[[nodiscard]] std::optional<uint16_t>
required_descriptor_vgpr_allocation_for_scratch(const ConSanAccessInventorySite &access,
                                                uint16_t scratch_vgpr, uint16_t required_vgprs) {
  uint32_t required_count = static_cast<uint32_t>(scratch_vgpr) + required_vgprs;
  if (needs_scratch_headroom_at_descriptor_edge(access))
    ++required_count;
  if (required_count > REGISTER_SET_MAX_VGPRS)
    return std::nullopt;
  return static_cast<uint16_t>(required_count);
}

[[nodiscard]] std::optional<uint16_t>
find_liveness_scratch_vgpr(const ConSanAccessInventorySite &access, const Instruction *inst,
                           const LivenessAnalysis *liveness,
                           std::optional<uint16_t> min_auto_scratch_vgpr,
                           std::optional<uint16_t> max_auto_scratch_vgpr, uint16_t required_vgprs) {
  if (inst == nullptr || liveness == nullptr)
    return std::nullopt;

  uint16_t search_start =
      std::max(lds_scratch_search_start(access), min_auto_scratch_vgpr.value_or(0));
  if (max_auto_scratch_vgpr && search_start >= *max_auto_scratch_vgpr)
    return std::nullopt;
  while (search_start < REGISTER_SET_MAX_VGPRS) {
    auto candidate = liveness->find_free_run(inst, required_vgprs, search_start);
    if (!candidate)
      return std::nullopt;
    const uint32_t candidate_end = static_cast<uint32_t>(*candidate) + required_vgprs;
    if (max_auto_scratch_vgpr && candidate_end > *max_auto_scratch_vgpr)
      return std::nullopt;
    if (max_auto_scratch_vgpr && candidate_end == *max_auto_scratch_vgpr &&
        needs_scratch_headroom_at_descriptor_edge(access))
      return std::nullopt;
    if (candidate_end <= 256 && lds_scratch_tuple_base_is_valid(access, *candidate) &&
        !is_forbidden_scratch_vgpr_run(access, *candidate, required_vgprs))
      return candidate;
    if (*candidate == UINT16_MAX)
      return std::nullopt;
    search_start = static_cast<uint16_t>(*candidate + 1u);
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<uint16_t>
choose_scratch_vgpr(const ConSanAccessInventorySite &access, const ConSanOptions &options,
                    const Instruction *inst, const LivenessAnalysis *liveness,
                    std::optional<uint16_t> min_auto_scratch_vgpr,
                    std::optional<uint16_t> max_auto_scratch_vgpr, uint16_t required_vgprs) {
  if (options.scratch_vgpr) {
    if (static_cast<uint32_t>(*options.scratch_vgpr) + required_vgprs <= 256 &&
        lds_scratch_tuple_base_is_valid(access, *options.scratch_vgpr) &&
        !is_forbidden_scratch_vgpr_run(access, *options.scratch_vgpr, required_vgprs))
      return *options.scratch_vgpr;
    return std::nullopt;
  }

  if (auto scratch = find_liveness_scratch_vgpr(access, inst, liveness, min_auto_scratch_vgpr,
                                                max_auto_scratch_vgpr, required_vgprs))
    return scratch;

  return std::nullopt;
}

[[nodiscard]] std::optional<uint16_t>
choose_spill_scratch_vgpr(const ConSanAccessInventorySite &access, uint16_t allocation_count,
                          uint16_t required_vgprs) {
  for (uint32_t candidate = 0;
       candidate + required_vgprs <= allocation_count && candidate + required_vgprs <= 256u;
       ++candidate) {
    if (lds_scratch_tuple_base_is_valid(access, static_cast<uint16_t>(candidate)) &&
        !is_forbidden_scratch_vgpr_run(access, static_cast<uint16_t>(candidate), required_vgprs))
      return static_cast<uint16_t>(candidate);
  }
  return std::nullopt;
}

[[nodiscard]] std::vector<BasicBlock *>
block_ptrs_for(const std::vector<std::unique_ptr<BasicBlock>> &blocks) {
  std::vector<BasicBlock *> ptrs;
  ptrs.reserve(blocks.size());
  for (const auto &block : blocks) {
    if (block != nullptr)
      ptrs.push_back(block.get());
  }
  return ptrs;
}

[[nodiscard]] const Instruction *
find_instruction_at_text_offset(std::span<BasicBlock *const> blocks, uint64_t text_offset) {
  for (BasicBlock *block : blocks) {
    if (block == nullptr || text_offset < block->start_offset() ||
        text_offset >= block->end_offset())
      continue;
    for (const Instruction &inst : block->instructions()) {
      if (inst.src_loc() == text_offset)
        return &inst;
    }
  }
  return nullptr;
}

[[nodiscard]] bool text_offset_is_inside_s_clause(std::span<BasicBlock *const> blocks,
                                                  uint64_t text_offset) {
  for (BasicBlock *block : blocks) {
    if (block == nullptr || text_offset < block->start_offset() ||
        text_offset >= block->end_offset())
      continue;
    uint32_t clause_remaining = 0;
    for (const Instruction &inst : block->instructions()) {
      const bool blocked_by_clause = clause_remaining > 0;
      if (clause_remaining > 0)
        --clause_remaining;
      if (is_s_clause(inst))
        clause_remaining = s_clause_following_instruction_count(inst);
      if (inst.src_loc() == text_offset)
        return blocked_by_clause;
    }
  }
  return false;
}

void update_max_vgpr_ref(const RegisterSet &set, std::optional<uint16_t> &max_vgpr) {
  set.for_each([&](RegisterRef ref) {
    if (ref.cls != RegClass::VGPR)
      return;
    if (!max_vgpr || ref.index > *max_vgpr)
      max_vgpr = ref.index;
  });
}

void update_max_sgpr_ref(const RegisterSet &set, std::optional<uint16_t> &max_sgpr) {
  set.for_each([&](RegisterRef ref) {
    if (ref.cls != RegClass::SGPR)
      return;
    if (!max_sgpr || ref.index > *max_sgpr)
      max_sgpr = ref.index;
  });
}

[[nodiscard]] KernelMaxRegisterRefs
max_register_refs_in_kernel(const ConSanKernelInfo &kernel, std::span<BasicBlock *const> blocks) {
  KernelMaxRegisterRefs max_refs;
  if (!kernel.has_text_range || kernel.code_size == 0)
    return max_refs;

  const uint64_t begin = kernel.entry_text_offset;
  const uint64_t end = begin + kernel.code_size;
  for (BasicBlock *block : blocks) {
    if (block == nullptr || block->end_offset() <= begin || block->start_offset() >= end)
      continue;
    for (const Instruction &inst : block->instructions()) {
      if (inst.src_loc() < begin || inst.src_loc() >= end)
        continue;
      InstDefUse du(inst);
      update_max_sgpr_ref(du.uses, max_refs.sgpr);
      update_max_sgpr_ref(du.defs, max_refs.sgpr);
      update_max_vgpr_ref(du.uses, max_refs.vgpr);
      update_max_vgpr_ref(du.defs, max_refs.vgpr);
    }
  }
  return max_refs;
}

[[nodiscard]] bool read_words_at(std::span<const uint8_t> bytes, uint64_t offset,
                                 std::span<uint32_t> words) {
  const uint64_t size = static_cast<uint64_t>(words.size() * sizeof(uint32_t));
  if (offset > bytes.size() || size > bytes.size() - offset)
    return false;
  std::memcpy(words.data(), bytes.data() + offset, static_cast<size_t>(size));
  return true;
}

void append_consan_patch_words(std::vector<uint8_t> &bytes, std::span<const uint32_t> words) {
  const size_t old_size = bytes.size();
  const size_t added_size = words.size() * sizeof(uint32_t);
  bytes.resize(old_size + added_size);
  std::memcpy(bytes.data() + old_size, words.data(), added_size);
}

[[nodiscard]] uint32_t count_nop_padding(std::span<const uint8_t> bytes, uint64_t offset,
                                         uint32_t max_word_count, rj_code_arch_t arch) {
  const uint32_t nop = build_s_nop(0, arch);
  uint32_t count = 0;
  for (; count < max_word_count; ++count) {
    uint32_t word = 0;
    if (!read_words_at(bytes, offset + count * sizeof(uint32_t), std::span<uint32_t>(&word, 1)) ||
        word != nop)
      break;
  }
  return count;
}

[[nodiscard]] bool has_only_rocclr_runtime_kernels(const ProgramInventory &program_inventory) {
  if (program_inventory.kernels().empty())
    return false;
  for (const ConSanKernelInfo &kernel : program_inventory.kernels()) {
    if (!is_rocclr_runtime_kernel_name(kernel.name))
      return false;
  }
  return true;
}

std::optional<ConSanCommittedLowering> make_consan_instrumented_patch_lowering(
    const ConSanObservationPlan &plan, std::span<const ConSanProbeIntentId> intent_ids,
    const ConSanCommittedPatchGeometry &patch, ConSanRuntimeStaticMapping runtime_mapping) {
  std::vector<PhysicalSiteId> original_sites;
  for (ConSanProbeIntentId id : intent_ids) {
    const ConSanProbeIntent *intent = plan.intent(id);
    if (intent == nullptr)
      return std::nullopt;
    if (std::ranges::find(original_sites, intent->physical_site) == original_sites.end())
      original_sites.push_back(intent->physical_site);
  }

  std::vector<ConSanCommittedLoweringLocation> locations;
  locations.reserve(original_sites.size() * (patch.trampoline_size == 0u ? 1u : 2u));
  for (const PhysicalSiteId &site : original_sites) {
    if (patch.original_size != 0u) {
      locations.push_back({
          .original_site = site,
          .emitted_text_offset = patch.anchor_offset,
          .emitted_size = patch.original_size,
          .relocated_guest_text_offset =
              patch.trampoline_size == 0u ? patch.relocated_guest_instruction_offset : std::nullopt,
      });
    }
    if (patch.trampoline_size != 0u) {
      locations.push_back({
          .original_site = site,
          .emitted_text_offset = patch.trampoline_offset,
          .emitted_size = patch.trampoline_size,
          .relocated_guest_text_offset = patch.relocated_guest_instruction_offset,
      });
    }
  }
  return make_consan_committed_lowering(plan, intent_ids, locations,
                                        ConSanLoweringOutcomeKind::Instrumented, {},
                                        std::move(runtime_mapping));
}

} // namespace rocjitsu

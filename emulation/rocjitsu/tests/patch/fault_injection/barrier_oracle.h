// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "../replacement_test_util.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "rocjitsu/code/patch/fault_injection/drop_barrier.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace rocjitsu::test {

inline std::optional<std::vector<uint8_t>>
expected_drop_image(const FaultMutationContext &context, const DropBarrierSelection &selection) {
  const IsaTargetDescriptor *target =
      default_isa_target_registry().find(context.source().target_id());
  if (!target)
    return std::nullopt;
  const FailureOr<uint32_t> nop = build_s_nop(0, target->architecture_id);
  if (nop.failed())
    return std::nullopt;
  std::vector<test::ExpectedReplacement> edits;
  for (const StaticBarrierMember &member : selection.barrier().members) {
    std::vector<uint8_t> replacement(sizeof(uint32_t));
    const uint32_t word = nop.value();
    std::memcpy(replacement.data(), &word, sizeof(word));
    edits.push_back(
        {.offset = member.offset, .expected = member.bytes, .replacement = std::move(replacement)});
  }
  return test::expected_replacement_image(context.source(), edits, target->architecture_id);
}

inline std::vector<uint8_t> expected_drop_for_test(const FaultMutationContext &context,
                                                   const DropBarrierSelection &selection) {
  const std::optional<std::vector<uint8_t>> expected = expected_drop_image(context, selection);
  EXPECT_TRUE(expected);
  return expected ? *expected : std::vector<uint8_t>{};
}

inline bool same_barrier_members(const StaticBarrier &left, const StaticBarrier &right) {
  // Branch splices create new blocks. Source offsets, encodings and logical
  // membership remain stable; each fresh inventory proves its own block ownership.
  return left.form == right.form && left.barrier_id == right.barrier_id &&
         left.scope == right.scope && left.members == right.members;
}

inline bool branch_reaches(Decoder &decoder, const Section &text, uint64_t from, uint64_t to) {
  if (from > text.size() || sizeof(uint32_t) > text.size() - from)
    return false;
  uint32_t word = 0;
  std::memcpy(&word, reinterpret_cast<const uint8_t *>(text.data()) + from, sizeof(word));
  const DecodeResult decoded = decoder.decode_window(std::span(&word, 1), from);
  if (decoded.failed() || decoded.value()->size() != 4 || decoded.value()->mnemonic() != "s_branch")
    return false;
  const std::optional<int64_t> delta = decoded.value()->branch_offset_bytes();
  return delta && static_cast<int64_t>(from + 4) + *delta == static_cast<int64_t>(to);
}

inline bool descriptors_preserved(const FaultMutationContext &context,
                                  const FaultMutationContext &mutant) {
  const Section &before_text = *context.source().text_sections().front();
  const Section &after_text = *mutant.source().text_sections().front();
  const std::vector<KernelDescriptorInfo> before =
      scan_kernel_descriptors(context.image(), before_text.sectionOffset(), before_text.size(),
                              before_text.sectionHeaderIndex());
  const std::vector<KernelDescriptorInfo> after =
      scan_kernel_descriptors(mutant.image(), after_text.sectionOffset(), after_text.size(),
                              after_text.sectionHeaderIndex());
  if (before.size() != after.size())
    return false;
  Elf64_Ehdr header = {};
  if (mutant.image().size() < sizeof(header))
    return false;
  std::memcpy(&header, mutant.image().data(), sizeof(header));
  const std::optional<std::vector<Elf64_Phdr>> segments = read_elf_table<Elf64_Phdr>(
      mutant.image(), header.e_phoff, header.e_phnum, header.e_phentsize);
  if (!segments)
    return false;
  auto loaded_address = [&](uint64_t file_offset, uint64_t size,
                            uint32_t flags) -> std::optional<uint64_t> {
    std::optional<uint64_t> address;
    for (const Elf64_Phdr &segment : *segments) {
      if (segment.p_type != PT_LOAD || (segment.p_flags & flags) != flags ||
          file_offset < segment.p_offset || file_offset - segment.p_offset > segment.p_filesz ||
          size > segment.p_filesz - (file_offset - segment.p_offset))
        continue;
      const uint64_t delta = file_offset - segment.p_offset;
      if (address || delta > std::numeric_limits<uint64_t>::max() - segment.p_vaddr)
        return std::nullopt;
      address = segment.p_vaddr + delta;
    }
    return address;
  };
  const std::optional<uint64_t> text_address =
      loaded_address(after_text.sectionOffset(), after_text.size(), PF_X);
  if (!text_address || *text_address != after_text.vaddr())
    return false;
  for (const KernelDescriptorInfo &original : before) {
    const std::vector<KernelDescriptorInfo>::const_iterator updated =
        std::ranges::find(after, original.kernel_name, &KernelDescriptorInfo::kernel_name);
    if (updated == after.end() || updated->entry_text_offset != original.entry_text_offset)
      return false;
    const std::optional<uint64_t> descriptor_address =
        loaded_address(updated->descriptor_file_offset, sizeof(updated->descriptor), 0);
    if (!descriptor_address ||
        *descriptor_address > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        after_text.vaddr() + updated->entry_text_offset >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        updated->descriptor.kernel_code_entry_byte_offset !=
            static_cast<int64_t>(after_text.vaddr() + updated->entry_text_offset) -
                static_cast<int64_t>(*descriptor_address))
      return false;
    amdhsa::kernel_descriptor_t normalized = updated->descriptor;
    // Only the relative entry field follows the descriptor's changed address.
    normalized.kernel_code_entry_byte_offset = original.descriptor.kernel_code_entry_byte_offset;
    if (std::memcmp(&normalized, &original.descriptor, sizeof(normalized)) != 0)
      return false;
  }
  return true;
}

// Recipe oracle lives only in tests. Expected branch/cave bytes do not consume
// production plans, site records, candidate bytes or the Drop apply entry point.
inline Result matches_drop_output(const FaultMutationContext &context,
                                  const DropBarrierSelection &selection,
                                  std::span<const uint8_t> candidate, FaultMutationError &error) {
  if (!selection.belongs_to(context)) {
    error = {.code = FaultMutationErrorCode::BadIdentity,
             .offset = 0,
             .message = "test oracle: wrong source"};
    return Result::failure();
  }
  FailureOr<FaultBarrierInventory> pristine =
      discover_drop_barriers(context, selection.owner().entry, error);
  if (pristine.failed())
    return Result::failure();
  auto mismatch = [&] {
    error = {.code = FaultMutationErrorCode::ValidationFailure,
             .offset = 0,
             .message = "test oracle: incorrect complete Drop"};
    return Result::failure();
  };
  const std::span<const StaticBarrier> barriers = pristine.value().barriers();
  const std::span<const StaticBarrier>::iterator selected =
      std::ranges::find(barriers, selection.barrier());
  const std::optional<std::vector<uint8_t>> expected = expected_drop_image(context, selection);
  if (selected == barriers.end() || !expected || !std::ranges::equal(*expected, candidate))
    return mismatch();
  const FaultMutationContext mutant(candidate);
  if (!mutant.source().is_valid() || mutant.source().text_sections().size() != 1 ||
      mutant.source().target_id() != context.source().target_id() ||
      !descriptors_preserved(context, mutant))
    return mismatch();
  const Section &text = *mutant.source().text_sections().front();
  const Section &original_text = *context.source().text_sections().front();
  const auto *original_bytes = reinterpret_cast<const uint8_t *>(original_text.data());
  const auto *mutant_bytes = reinterpret_cast<const uint8_t *>(text.data());
  if (text.size() < original_text.size())
    return mismatch();
  for (uint64_t offset = 0; offset < original_text.size(); ++offset) {
    const bool selected_byte =
        std::ranges::any_of(selected->members, [&](const StaticBarrierMember &member) {
          return offset >= member.offset && offset - member.offset < member.bytes.size();
        });
    if (!selected_byte && original_bytes[offset] != mutant_bytes[offset])
      return mismatch();
  }
  std::unique_ptr<Decoder> decoder =
      Decoder::create(default_isa_target_registry(), context.source().target_id());
  if (!decoder)
    return mismatch();
  uint64_t cave = pristine.value().section().size;
  for (const StaticBarrierMember &member : selected->members) {
    if (member.bytes.size() != 4 || cave > text.size() || text.size() - cave < 8 ||
        !branch_reaches(*decoder, text, member.offset, cave) ||
        !branch_reaches(*decoder, text, cave + 4, member.offset + member.bytes.size()))
      return mismatch();
    uint32_t word = 0;
    std::memcpy(&word, reinterpret_cast<const uint8_t *>(text.data()) + cave, sizeof(word));
    const DecodeResult instruction = decoder->decode_window(std::span(&word, 1), cave);
    if (instruction.failed() || instruction.value()->size() != 4 ||
        instruction.value()->mnemonic() != "s_nop" ||
        instruction.value()->num_src_operands() != 1 ||
        instruction.value()->src_operand(0)->encoding_value() != 0)
      return mismatch();
    cave += 8;
  }
  if (cave != text.size())
    return mismatch();
  // Fresh analysis checks executable load coverage, descriptor entry resolution,
  // exclusive ownership and complete same-block membership in the changed CFG.
  FailureOr<FaultBarrierInventory> remaining =
      discover_drop_barriers(mutant, selection.owner().entry, error);
  if (remaining.failed())
    return mismatch();
  std::vector<StaticBarrier> wanted(barriers.begin(), barriers.end());
  std::erase(wanted, *selected);
  if (!std::ranges::equal(remaining.value().barriers(), wanted, same_barrier_members))
    return mismatch();
  return Result::success();
}

} // namespace rocjitsu::test

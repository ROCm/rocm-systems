// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/dbt/code_placement.h"
#include "rocjitsu/code/patch/instruction_builder.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <span>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace {

constexpr uint32_t kSCodeEnd = 0xBF9F0000u;
constexpr uint32_t kSNop0 = 0xBF800000u;

std::vector<uint8_t> words_to_bytes(std::span<const uint32_t> words) {
  std::vector<uint8_t> bytes(words.size_bytes());
  std::memcpy(bytes.data(), words.data(), bytes.size());
  return bytes;
}

std::vector<uint8_t> repeated_word_bytes(size_t word_count, uint32_t word) {
  std::vector<uint32_t> words(word_count, word);
  return words_to_bytes(words);
}

} // namespace

TEST(CodePlacement, ComputesSoppBranchOffsetsFromNextInstruction) {
  int16_t offset_dwords = 0;

  ASSERT_TRUE(compute_sopp_branch_offset(0x4, 0x14, offset_dwords));
  EXPECT_EQ(offset_dwords, 3);

  ASSERT_TRUE(compute_sopp_branch_offset(0x20, 0x14, offset_dwords));
  EXPECT_EQ(offset_dwords, -4);

  EXPECT_FALSE(compute_sopp_branch_offset(0x0, 0x6, offset_dwords));
  EXPECT_FALSE(compute_sopp_branch_offset(0x0, 0x40000, offset_dwords));
}

TEST(CodePlacement, FindsForwardLocalPaddingCave) {
  const std::vector<uint8_t> text = repeated_word_bytes(8, kSNop0);
  const LocalTextCaveRequest request{
      .source = {.start = 0, .end = sizeof(uint32_t)},
      .body_size_bytes = 2 * sizeof(uint32_t),
      .cave_size_bytes = 3 * sizeof(uint32_t),
  };

  const auto placement = find_local_text_cave(text, request, {}, {}, false);
  ASSERT_TRUE(placement.has_value());
  EXPECT_EQ(placement->offset, sizeof(uint32_t));
  EXPECT_EQ(placement->entry_branch_dwords, 0);
  EXPECT_EQ(placement->exit_branch_dwords, -3);
}

TEST(CodePlacement, LocalTextCaveSkipsReservedAndProtectedRanges) {
  const std::vector<uint8_t> text = repeated_word_bytes(16, kSCodeEnd);
  const LocalTextCaveRequest request{
      .source = {.start = 0, .end = sizeof(uint32_t)},
      .body_size_bytes = sizeof(uint32_t),
      .cave_size_bytes = 2 * sizeof(uint32_t),
  };
  const std::vector<std::pair<uint64_t, uint64_t>> reserved_ranges{{4, 12}};
  const std::vector<std::pair<uint64_t, uint64_t>> protected_ranges{{12, 20}};

  const auto placement =
      find_local_text_cave(text, request, reserved_ranges, protected_ranges, false);
  ASSERT_TRUE(placement.has_value());
  EXPECT_EQ(placement->offset, 20u);
  EXPECT_FALSE(overlaps_any_range(placement->offset, placement->offset + request.cave_size_bytes,
                                  reserved_ranges));
  EXPECT_FALSE(overlaps_any_range(placement->offset, placement->offset + request.cave_size_bytes,
                                  protected_ranges));
}

TEST(CodePlacement, LocalTextCaveRequiresPaddingUnlessUnreachableTextIsAllowed) {
  const std::vector<uint8_t> text = repeated_word_bytes(8, 0xDEADBEEFu);
  const LocalTextCaveRequest request{
      .source = {.start = 0, .end = sizeof(uint32_t)},
      .body_size_bytes = sizeof(uint32_t),
      .cave_size_bytes = 2 * sizeof(uint32_t),
  };

  EXPECT_FALSE(find_local_text_cave(text, request, {}, {}, false).has_value());

  const auto placement = find_local_text_cave(text, request, {}, {}, true);
  ASSERT_TRUE(placement.has_value());
  EXPECT_EQ(placement->offset, sizeof(uint32_t));
}

TEST(CodePlacement, UnreachableTextCaveDoesNotCrossProtectedRanges) {
  const std::vector<uint8_t> text = repeated_word_bytes(16, 0xDEADBEEFu);
  const LocalTextCaveRequest request{
      .source = {.start = 0, .end = sizeof(uint32_t)},
      .body_size_bytes = sizeof(uint32_t),
      .cave_size_bytes = 2 * sizeof(uint32_t),
  };
  const std::vector<std::pair<uint64_t, uint64_t>> protected_ranges{{8, 16}};

  const auto placement = find_local_text_cave(text, request, {}, protected_ranges, true);
  ASSERT_TRUE(placement.has_value());
  EXPECT_EQ(placement->offset, 16u);
}

TEST(CodePlacement, FindsBackwardLocalBranchIsland) {
  const std::vector<uint8_t> text = repeated_word_bytes(32, kSNop0);
  const LocalBranchIslandRequest request{
      .source = {.start = sizeof(uint32_t), .end = 2 * sizeof(uint32_t)},
      .island_target = 8 * sizeof(uint32_t),
  };

  const auto placement = find_local_branch_island(text, request, {}, {}, false);
  ASSERT_TRUE(placement.has_value());
  EXPECT_EQ(placement->offset, text.size() - sizeof(uint32_t));
  EXPECT_EQ(placement->entry_branch_dwords, 29);
  EXPECT_EQ(placement->exit_branch_dwords, -24);
}

TEST(CodePlacement, RejectsBranchIslandWhenTargetIsOutOfRange) {
  const std::vector<uint8_t> text = repeated_word_bytes(32, kSNop0);
  const LocalBranchIslandRequest request{
      .source = {.start = 0, .end = sizeof(uint32_t)},
      .island_target = 0x40000,
  };

  EXPECT_FALSE(find_local_branch_island(text, request, {}, {}, false).has_value());
}

TEST(CodePlacement, PlansExpandedTextScopeWithoutPrologue) {
  const ExpandedTextScopePlacementRequest request{
      .original_text_size_bytes = 0x1000,
      .current_tail_size_bytes = sizeof(uint32_t),
      .original_entry_offset = 0x40,
      .translated_entry_offset = 3 * sizeof(uint32_t),
  };

  const auto placement = plan_expanded_text_scope_placement(request);
  ASSERT_TRUE(placement.has_value());
  EXPECT_EQ(placement->padding_bytes, 48u);
  EXPECT_FALSE(placement->launch_stub_offset.has_value());
  EXPECT_EQ(placement->body_offset, 52u);
  EXPECT_EQ(placement->descriptor_entry_offset, 64u);
  EXPECT_FALSE(placement->prologue_branch_dwords.has_value());
}

TEST(CodePlacement, PlansExpandedTextScopeWithPrologue) {
  const ExpandedTextScopePlacementRequest request{
      .original_text_size_bytes = 0x1000,
      .current_tail_size_bytes = sizeof(uint32_t),
      .original_entry_offset = 0x40,
      .translated_entry_offset = 2 * sizeof(uint32_t),
      .prologue_size_bytes = 3 * sizeof(uint32_t),
  };

  const auto placement = plan_expanded_text_scope_placement(request);
  ASSERT_TRUE(placement.has_value());
  EXPECT_EQ(placement->padding_bytes, 60u);
  ASSERT_TRUE(placement->launch_stub_offset.has_value());
  EXPECT_EQ(*placement->launch_stub_offset, 64u);
  EXPECT_EQ(placement->body_offset, 80u);
  EXPECT_EQ(placement->descriptor_entry_offset, 64u);
  ASSERT_TRUE(placement->prologue_branch_dwords.has_value());
  EXPECT_EQ(*placement->prologue_branch_dwords, 2);
}

TEST(CodePlacement, RejectsExpandedTextScopeWhenPrologueBranchIsOutOfRange) {
  const ExpandedTextScopePlacementRequest request{
      .original_text_size_bytes = 0,
      .current_tail_size_bytes = 0,
      .original_entry_offset = 0,
      .translated_entry_offset = 0x40000,
      .prologue_size_bytes = sizeof(uint32_t),
  };

  EXPECT_FALSE(plan_expanded_text_scope_placement(request).has_value());
}

TEST(CodePlacement, RelocatesExpandedTextShortDirectBranch) {
  const std::vector<uint32_t> words{
      pack_sopp(sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4), 0),
      kSNop0,
      kSNop0,
  };
  const std::vector<ExpandedTextBranchFixup> branches{{
      .word_index = 0,
      .target_offset = 2 * sizeof(uint32_t),
      .mnemonic = "s_branch",
      .sgpr_pair = std::nullopt,
  }};
  const std::vector<std::pair<uint64_t, uint64_t>> offsets{
      {0, 0},
      {2 * sizeof(uint32_t), 2 * sizeof(uint32_t)},
  };

  const auto relocation =
      relocate_expanded_text_branches({.words = words, .branches = branches, .offset_map = offsets});
  ASSERT_TRUE(relocation.success) << relocation.message;
  ASSERT_EQ(relocation.words.size(), words.size());
  EXPECT_EQ(relocation.words[0], pack_sopp(sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4), 1));
  ASSERT_EQ(relocation.offset_map.size(), offsets.size());
  EXPECT_EQ(relocation.offset_map[1].second, 2 * sizeof(uint32_t));
}

TEST(CodePlacement, RelocatesExpandedTextLongDirectBranchAndUpdatesOffsets) {
  constexpr uint64_t kTargetWord = 0x9000;
  std::vector<uint32_t> words(kTargetWord + 1, kSNop0);
  words[0] = pack_sopp(sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4), 0);
  const std::vector<ExpandedTextBranchFixup> branches{{
      .word_index = 0,
      .target_offset = kTargetWord * sizeof(uint32_t),
      .mnemonic = "s_branch",
      .sgpr_pair = 10,
  }};
  const std::vector<std::pair<uint64_t, uint64_t>> offsets{
      {0, 0},
      {kTargetWord * sizeof(uint32_t), kTargetWord * sizeof(uint32_t)},
  };

  const auto relocation =
      relocate_expanded_text_branches({.words = words, .branches = branches, .offset_map = offsets});
  ASSERT_TRUE(relocation.success) << relocation.message;
  ASSERT_EQ(relocation.words.size(), words.size() + 5);
  EXPECT_EQ(relocation.words[0], pack_sop1(71, 10, 0));
  ASSERT_EQ(relocation.offset_map.size(), offsets.size());
  EXPECT_EQ(relocation.offset_map[1].second, (kTargetWord + 5) * sizeof(uint32_t));
}

TEST(CodePlacement, RejectsExpandedTextLongDirectBranchWithoutScratchSgpr) {
  constexpr uint64_t kTargetWord = 0x9000;
  std::vector<uint32_t> words(kTargetWord + 1, kSNop0);
  words[0] = pack_sopp(sopp_op_branch(ROCJITSU_CODE_ARCH_RDNA4), 0);
  const std::vector<ExpandedTextBranchFixup> branches{{
      .word_index = 0,
      .target_offset = kTargetWord * sizeof(uint32_t),
      .mnemonic = "s_branch",
      .sgpr_pair = std::nullopt,
  }};
  const std::vector<std::pair<uint64_t, uint64_t>> offsets{
      {0, 0},
      {kTargetWord * sizeof(uint32_t), kTargetWord * sizeof(uint32_t)},
  };

  const auto relocation =
      relocate_expanded_text_branches({.words = words, .branches = branches, .offset_map = offsets});
  EXPECT_FALSE(relocation.success);
  EXPECT_EQ(relocation.message,
            "expanded text copy needs a dead SGPR pair for long branch s_branch");
}

TEST(CodePlacement, RelocatesExpandedTextPcRelativeFixupToCopiedTarget) {
  constexpr uint16_t kPcSgpr = 20;
  constexpr uint16_t kTmpSgpr = 21;
  std::vector<uint32_t> words{
      pack_sop1(71, kPcSgpr, 0),
      pack_sop2(2, kTmpSgpr, 255, scalar_positive_inline_u32(4)),
      0,
      kSNop0,
      kSNop0,
  };
  const std::vector<std::pair<uint64_t, uint64_t>> offsets{
      {0, 0},
      {sizeof(uint32_t), sizeof(uint32_t)},
      {4 * sizeof(uint32_t), 4 * sizeof(uint32_t)},
  };
  const std::vector<ExpandedTextPcRelativeFixup> fixups{{
      .getpc_offset = 0,
      .add_tmp_offset = sizeof(uint32_t),
      .target_offset = 4 * sizeof(uint32_t),
      .kind = "setpc",
  }};

  const auto relocation = relocate_expanded_text_pc_relative_fixups(
      {.words = words,
       .offset_map = offsets,
       .fixups = fixups,
       .original_text_size_bytes = 0x1000,
       .scope_base_bytes = 0x40});
  ASSERT_TRUE(relocation.success) << relocation.message;
  EXPECT_EQ(words[2], 8u);
}

TEST(CodePlacement, RelocatesExpandedTextPcRelativeFixupToOriginalTarget) {
  constexpr uint16_t kPcSgpr = 20;
  constexpr uint16_t kTmpSgpr = 21;
  std::vector<uint32_t> words{
      pack_sop1(71, kPcSgpr, 0),
      pack_sop2(2, kTmpSgpr, 255, scalar_positive_inline_u32(4)),
      0,
  };
  const std::vector<std::pair<uint64_t, uint64_t>> offsets{
      {0, 0},
      {sizeof(uint32_t), sizeof(uint32_t)},
  };
  const std::vector<ExpandedTextPcRelativeFixup> fixups{{
      .getpc_offset = 0,
      .add_tmp_offset = sizeof(uint32_t),
      .target_offset = 0x200,
      .kind = "address",
  }};

  const auto relocation = relocate_expanded_text_pc_relative_fixups(
      {.words = words,
       .offset_map = offsets,
       .fixups = fixups,
       .original_text_size_bytes = 0x1000,
       .scope_base_bytes = 0x40});
  ASSERT_TRUE(relocation.success) << relocation.message;
  EXPECT_EQ(static_cast<int32_t>(words[2]), -3656);
}

TEST(CodePlacement, PlansExpandedTextDescriptorRedirectionsAndDeduplicatesDescriptors) {
  const std::vector<ExpandedTextDescriptorEntry> descriptors{
      {.descriptor_file_offset = 0x100, .entry_text_offset = 0x20},
      {.descriptor_file_offset = 0x100, .entry_text_offset = 0x20},
      {.descriptor_file_offset = 0x180, .entry_text_offset = 0x40},
  };
  const std::vector<std::pair<uint64_t, uint64_t>> copied_entries{
      {0x20, 0x200},
      {0x40, 0x300},
  };

  const auto plan =
      plan_expanded_text_descriptor_redirections(descriptors, copied_entries, 0x1000);
  ASSERT_TRUE(plan.success) << plan.message;
  ASSERT_EQ(plan.redirections.size(), 2u);
  EXPECT_EQ(plan.redirections[0].descriptor_file_offset, 0x100u);
  EXPECT_EQ(plan.redirections[0].original_entry_offset, 0x20u);
  EXPECT_EQ(plan.redirections[0].redirected_entry_offset, 0x1200u);
  EXPECT_EQ(plan.redirections[1].descriptor_file_offset, 0x180u);
  EXPECT_EQ(plan.redirections[1].original_entry_offset, 0x40u);
  EXPECT_EQ(plan.redirections[1].redirected_entry_offset, 0x1300u);
}

TEST(CodePlacement, RejectsExpandedTextDescriptorRedirectionForMissingCopiedEntry) {
  const std::vector<ExpandedTextDescriptorEntry> descriptors{
      {.descriptor_file_offset = 0x100, .entry_text_offset = 0x20},
  };
  const std::vector<std::pair<uint64_t, uint64_t>> copied_entries;

  const auto plan =
      plan_expanded_text_descriptor_redirections(descriptors, copied_entries, 0x1000);
  EXPECT_FALSE(plan.success);
  EXPECT_EQ(plan.message,
            "expanded text copy could not map a kernel descriptor entry; leaving code object "
            "unchanged");
}

} // namespace rocjitsu

// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/instruction_sequence.h"

#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/code/patch/rdna4_instrumentation_builder.h"

#include <gtest/gtest.h>

#include <array>
#include <optional>
#include <vector>

namespace rocjitsu {
namespace {

TEST(InstructionSequenceTest, AppendsWordsRangesAndOptionalValuesInOrder) {
  std::vector<uint32_t> words{1u};
  InstructionSequence sequence(words);
  const std::array<uint32_t, 2> range{3u, 4u};
  const std::optional<uint32_t> optional = 5u;

  EXPECT_TRUE(sequence.emit_all(2u, range, optional));
  EXPECT_EQ(words, (std::vector<uint32_t>{1u, 2u, 3u, 4u, 5u}));
}

TEST(InstructionSequenceTest, FailedBatchRollsBackEveryWordInThatBatch) {
  std::vector<uint32_t> words{1u};
  InstructionSequence sequence(words);
  const std::array<uint32_t, 2> range{2u, 3u};
  const std::optional<uint32_t> missing;

  EXPECT_FALSE(sequence.emit_all(range, missing, 4u));
  EXPECT_EQ(words, (std::vector<uint32_t>{1u}));
}

TEST(InstructionSequenceTest, ResolvesForwardAndBackwardBranches) {
  std::vector<uint32_t> words;
  InstructionSequence sequence(words);
  const auto beginning = sequence.mark_label();
  const auto end = sequence.make_label();

  ASSERT_TRUE(sequence.emit_branch(end, InstructionSequence::BranchKind::SccZero));
  ASSERT_TRUE(sequence.emit(0x12345678u));
  ASSERT_TRUE(sequence.bind(end));
  ASSERT_TRUE(sequence.emit_branch(beginning, InstructionSequence::BranchKind::Unconditional));
  ASSERT_TRUE(sequence.resolve_branches(ROCJITSU_CODE_ARCH_RDNA4));

  EXPECT_EQ(words[0], *build_s_cbranch_scc0(1, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(words[1], 0x12345678u);
  EXPECT_EQ(words[2], build_s_branch(-3, ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(InstructionSequenceTest, UnboundOrOutOfRangeBranchFailsWithoutPatching) {
  std::vector<uint32_t> words;
  InstructionSequence sequence(words);
  const auto missing = sequence.make_label();
  ASSERT_TRUE(sequence.emit_branch(missing, InstructionSequence::BranchKind::VccZero));
  EXPECT_FALSE(sequence.resolve_branches(ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(words, (std::vector<uint32_t>{0u}));

  words.resize(40000u, 0u);
  ASSERT_TRUE(sequence.bind(missing));
  EXPECT_FALSE(sequence.resolve_branches(ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(words.front(), 0u);
}

} // namespace
} // namespace rocjitsu

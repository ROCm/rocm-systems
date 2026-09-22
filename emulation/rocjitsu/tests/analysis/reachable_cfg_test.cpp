// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/code_object.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/decoder.h"
#include "util/diagnostic.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <vector>

namespace rocjitsu {
namespace {
using Blocks = std::vector<std::unique_ptr<BasicBlock>>;
using Issue = BasicBlock::SuccessorIssue;
using Range = BasicBlock::CodeRange;
constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
constexpr uint32_t kNop = build_s_nop(0, kArch);
constexpr uint32_t kEnd = build_s_endpgm(kArch);
constexpr uint32_t kInvalid = 0xffffffffu;
constexpr uint32_t branch(int16_t delta) { return build_s_branch(delta, kArch); }
constexpr uint32_t conditional(int16_t delta) {
  return pack_sopp(5, static_cast<uint16_t>(delta)); // s_cbranch_scc0.
}

class TestTextSection final : public Section {
public:
  explicit TestTextSection(std::span<const uint32_t> words, size_t trailing_bytes = 0)
      : Section(".text", std::make_unique<char[]>(words.size_bytes() + trailing_bytes)),
        size_(words.size_bytes() + trailing_bytes) {
    if (!words.empty())
      std::memcpy(data_.get(), words.data(), words.size_bytes());
  }
  size_t size() const override { return size_; }
  uint32_t sectionHeaderNameIdx() const override { return 0; }
  uint64_t sectionOffset() const override { return 0; }

private:
  size_t size_;
};

class TestCodeObject final : public CodeObject {
public:
  void add_text(std::span<const uint32_t> words, size_t trailing_bytes = 0) {
    sections_.push_back(std::make_unique<TestTextSection>(words, trailing_bytes));
    text_sections_.push_back(sections_.back().get());
  }
};

class ReachableCfg : public ::testing::Test {
protected:
  // Keep the backing object alive while the returned decoded instructions borrow it.
  FailureOr<Blocks> build_reachable(const std::vector<uint32_t> &words,
                                    const std::vector<uint64_t> &entries = {0},
                                    const std::vector<Range> &ranges = {},
                                    rj_code_arch_t arch = kArch,
                                    const std::vector<uint64_t> &decode_seeds = {},
                                    const std::vector<uint64_t> &split_points = {}) {
    error_("");
    object_ = std::make_unique<TestCodeObject>();
    object_->add_text(words);
    auto decoder = Decoder::create(arch);
    return BasicBlock::build_reachable(*object_, *decoder, arch, entries, error_.emitter(), ranges,
                                       decode_seeds, split_points);
  }

  void expect_failure(const std::vector<uint32_t> &words,
                      const std::vector<uint64_t> &entries = {0},
                      const std::vector<Range> &ranges = {}) {
    const auto result = build_reachable(words, entries, ranges);
    EXPECT_TRUE(result.failed());
    EXPECT_FALSE(error_.message().empty());
  }

  static std::vector<uint64_t> offsets(const Blocks &blocks) {
    std::vector<uint64_t> result;
    for (const auto &block : blocks)
      result.push_back(block->start_offset());
    return result;
  }

  static std::vector<uint64_t> successors(const BasicBlock &block) {
    std::vector<uint64_t> result;
    for (const auto *successor : block.successors())
      result.push_back(successor->start_offset());
    std::ranges::sort(result);
    return result;
  }

  static void expect_inverse_edges(const Blocks &blocks) {
    for (const auto &block : blocks) {
      for (const auto *successor : block->successors()) {
        EXPECT_NE(std::ranges::find(successor->predecessors(), block.get()),
                  successor->predecessors().end());
      }
      for (const auto *predecessor : block->predecessors()) {
        EXPECT_NE(std::ranges::find(predecessor->successors(), block.get()),
                  predecessor->successors().end());
      }
    }
  }

  std::unique_ptr<TestCodeObject> object_;
  util::StringDiagnostic error_;
};

TEST_F(ReachableCfg, SkipsMalformedDisconnectedTextAndPreservesOffsets) {
  const auto result = build_reachable({kInvalid, kNop, kEnd, kInvalid}, {4});
  ASSERT_TRUE(result.succeeded()) << error_.message();
  const auto &blocks = result.value();
  ASSERT_EQ(blocks.size(), 1u);
  EXPECT_EQ(blocks[0]->start_offset(), 4u);
  EXPECT_EQ(blocks[0]->end_offset(), 12u);
  EXPECT_EQ(blocks[0]->num_instructions(), 2u);
  EXPECT_EQ((*blocks[0]->instructions().begin()).src_loc(), 4u);
  EXPECT_EQ(blocks[0]->successor_issue(), Issue::None);
}

TEST_F(ReachableCfg, EmptyEntriesDoNotDecodeAnyText) {
  const auto result = build_reachable({kInvalid}, {});
  ASSERT_TRUE(result.succeeded());
  EXPECT_TRUE(result.value().empty());
}

TEST_F(ReachableCfg, MultipleDuplicateEntriesAreOrderedAndSplitBlocks) {
  const auto result = build_reachable({kNop, kEnd, kInvalid, kEnd}, {12, 4, 0, 12});
  ASSERT_TRUE(result.succeeded()) << error_.message();
  ASSERT_EQ(offsets(result.value()), (std::vector<uint64_t>{0, 4, 12}));
  EXPECT_EQ(successors(*result.value()[0]), (std::vector<uint64_t>{4}));
  expect_inverse_edges(result.value());
}

TEST_F(ReachableCfg, DirectBranchSkipsMalformedGap) {
  const auto result = build_reachable({branch(2), kInvalid, kInvalid, kEnd});
  ASSERT_TRUE(result.succeeded()) << error_.message();
  ASSERT_EQ(offsets(result.value()), (std::vector<uint64_t>{0, 12}));
  EXPECT_EQ(successors(*result.value()[0]), (std::vector<uint64_t>{12}));
  EXPECT_EQ(result.value()[0]->successor_issue(), Issue::None);
  expect_inverse_edges(result.value());
}

TEST_F(ReachableCfg, ConditionalDiamondKeepsBothArmsAndJoin) {
  const auto result = build_reachable({conditional(2), kNop, branch(1), kNop, kEnd});
  ASSERT_TRUE(result.succeeded()) << error_.message();
  const auto &blocks = result.value();
  ASSERT_EQ(offsets(blocks), (std::vector<uint64_t>{0, 4, 12, 16}));
  ASSERT_EQ(blocks.size(), 4u);
  EXPECT_EQ(successors(*blocks[0]), (std::vector<uint64_t>{4, 12}));
  EXPECT_EQ(successors(*blocks[1]), (std::vector<uint64_t>{16}));
  EXPECT_EQ(successors(*blocks[2]), (std::vector<uint64_t>{16}));
  expect_inverse_edges(blocks);
}

TEST_F(ReachableCfg, BackedgeCreatesLeaderInsideAlreadyDecodedRun) {
  const auto result = build_reachable({kNop, kNop, conditional(-2), kEnd});
  ASSERT_TRUE(result.succeeded()) << error_.message();
  ASSERT_EQ(offsets(result.value()), (std::vector<uint64_t>{0, 4, 12}));
  EXPECT_EQ(successors(*result.value()[1]), (std::vector<uint64_t>{4, 12}));
  expect_inverse_edges(result.value());
}

TEST_F(ReachableCfg, SelfLoopTerminatesDiscovery) {
  const auto result = build_reachable({branch(-1), kInvalid});
  ASSERT_TRUE(result.succeeded()) << error_.message();
  ASSERT_EQ(result.value().size(), 1u);
  EXPECT_EQ(successors(*result.value()[0]), (std::vector<uint64_t>{0}));
}

TEST_F(ReachableCfg, DirectCallIncludesCalleeAndContinuation) {
  // Callee returns with s_setpc_b64 s[4:5] (0xbe801d04).
  const auto result = build_reachable({build_s_call_b64(4, 2, kArch), kEnd, kInvalid, 0xbe801d04u});
  ASSERT_TRUE(result.succeeded()) << error_.message();
  const auto &blocks = result.value();
  ASSERT_EQ(offsets(blocks), (std::vector<uint64_t>{0, 4, 12}));
  ASSERT_EQ(blocks[0]->call_edges().size(), 1u);
  EXPECT_EQ(blocks[0]->call_edges()[0].callee->start_offset(), 12u);
  EXPECT_EQ(blocks[0]->call_edges()[0].continuation->start_offset(), 4u);
  EXPECT_EQ(blocks[0]->successor_issue(), Issue::None);
  EXPECT_EQ(blocks[2]->successor_issue(), Issue::IndirectControlFlow);
  expect_inverse_edges(blocks);
}

TEST_F(ReachableCfg, IndirectBranchIsExplicitlyIncomplete) {
  const auto result = build_reachable({0xbe801d00u, kInvalid}); // s_setpc_b64 s[0:1].
  ASSERT_TRUE(result.succeeded()) << error_.message();
  ASSERT_EQ(result.value().size(), 1u);
  EXPECT_EQ(result.value()[0]->successor_issue(), Issue::IndirectControlFlow);
  EXPECT_TRUE(result.value()[0]->successors().empty());
}

TEST_F(ReachableCfg, IndirectCallKeepsPossibleContinuation) {
  const auto result = build_reachable({0xbe841e00u, kEnd}); // s_swappc_b64 s[4:5], s[0:1].
  ASSERT_TRUE(result.succeeded()) << error_.message();
  ASSERT_EQ(result.value().size(), 2u);
  EXPECT_EQ(successors(*result.value()[0]), (std::vector<uint64_t>{4}));
  EXPECT_EQ(result.value()[0]->successor_issue(), Issue::IndirectControlFlow);
}

TEST_F(ReachableCfg, MissingBranchTargetsRemainVisible) {
  for (int16_t delta : {-2, 5}) {
    const auto result = build_reachable({branch(delta)});
    ASSERT_TRUE(result.succeeded()) << error_.message();
    ASSERT_EQ(result.value().size(), 1u);
    EXPECT_EQ(result.value()[0]->successor_issue(), Issue::MissingBranchTarget);
  }
}

TEST_F(ReachableCfg, MissingCallTargetAndContinuationAreDistinguished) {
  {
    const auto result = build_reachable({build_s_call_b64(4, 4, kArch), kEnd});
    ASSERT_TRUE(result.succeeded()) << error_.message();
    EXPECT_EQ(result.value()[0]->successor_issue(), Issue::MissingCallTarget);
  }
  const auto result = build_reachable({kEnd, build_s_call_b64(4, -2, kArch)}, {4});
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_EQ(result.value().back()->successor_issue(), Issue::MissingCallContinuation);
}

TEST_F(ReachableCfg, RangeBoundsOmitTargetsWithoutDecodingThem) {
  const auto result = build_reachable({conditional(1), kEnd, kInvalid}, {0}, {{0, 8}});
  ASSERT_TRUE(result.succeeded()) << error_.message();
  ASSERT_EQ(offsets(result.value()), (std::vector<uint64_t>{0, 4}));
  EXPECT_EQ(successors(*result.value()[0]), (std::vector<uint64_t>{4}));
  EXPECT_EQ(result.value()[0]->successor_issue(), Issue::MissingBranchTarget);
}

TEST_F(ReachableCfg, MissingFallthroughAtRangeBoundaryIsIncomplete) {
  const auto result = build_reachable({kNop, kInvalid}, {0}, {{0, 4}});
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_EQ(result.value()[0]->successor_issue(), Issue::MissingFallthrough);
}

TEST_F(ReachableCfg, OverlappingAndAdjacentRangesAreUnited) {
  const auto result = build_reachable({kNop, kNop, kEnd, kInvalid}, {0}, {{8, 4}, {0, 8}, {4, 4}});
  ASSERT_TRUE(result.succeeded()) << error_.message();
  ASSERT_EQ(result.value().size(), 1u);
  EXPECT_EQ(result.value()[0]->num_instructions(), 3u);
  EXPECT_EQ(result.value()[0]->successor_issue(), Issue::None);
}

TEST_F(ReachableCfg, DisjointRangesAllowBranchAcrossGap) {
  const auto result =
      build_reachable({branch(2), kInvalid, kInvalid, kEnd}, {0}, {{0, 4}, {12, 4}});
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_EQ(successors(*result.value()[0]), (std::vector<uint64_t>{12}));
}

TEST_F(ReachableCfg, RejectsInvalidEntries) {
  for (uint64_t entry : std::array<uint64_t, 3>{1, 4, std::numeric_limits<uint64_t>::max()})
    expect_failure({kEnd}, {entry});
  expect_failure({kEnd, kEnd}, {0}, {{4, 4}});
}

TEST_F(ReachableCfg, RejectsMalformedRanges) {
  for (Range range : {Range{1, 4}, Range{0, 3}, Range{0, 0}, Range{4, 4},
                      Range{0, std::numeric_limits<uint64_t>::max() - 3}})
    expect_failure({kEnd}, {0}, {range});
}

TEST_F(ReachableCfg, ReachableMalformedInstructionsFailWithOffset) {
  expect_failure({kNop, kInvalid});
  EXPECT_NE(error_.message().find("byte offset 4"), std::string::npos);
}

TEST_F(ReachableCfg, RangeCannotSplitMultiwordInstruction) {
  // CDNA scalar load is two words; its second word lies outside the allowed range.
  expect_failure({0xc0020100u, 0, kEnd}, {0}, {{0, 4}});
  expect_failure({0xc0020100u});
}

TEST_F(ReachableCfg, RejectsTargetsInsideInstructionsInEitherDiscoveryOrder) {
  // A literal is a valid one-word instruction in isolation, but not a CFG entry.
  const std::vector<uint32_t> words{0xbe8000ffu, kEnd, kEnd}; // s_mov_b32 s0, literal.
  expect_failure(words, {0, 4});
  expect_failure(words, {4, 0});
  expect_failure({0xc0020100u, 0, branch(-2)}); // Backedge to the load's second word.
}

TEST_F(ReachableCfg, Gfx1250FallthroughStopsAtPadding) {
  const auto result = build_reachable({build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5), 0, kInvalid}, {0},
                                      {}, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_TRUE(result.succeeded()) << error_.message();
  ASSERT_EQ(result.value().size(), 1u);
  EXPECT_TRUE(result.value()[0]->has_implicit_terminator());
  EXPECT_EQ(result.value()[0]->successor_issue(), Issue::None);
}

TEST_F(ReachableCfg, Gfx1250ExplicitPaddingTargetFails) {
  const auto result = build_reachable({build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA5), 0}, {0}, {},
                                      ROCJITSU_CODE_ARCH_CDNA5);
  EXPECT_TRUE(result.failed());
  EXPECT_NE(error_.message().find("padding"), std::string::npos);
}

TEST_F(ReachableCfg, RangeBoundaryIsNotAnImplicitPaddingTerminator) {
  const auto result = build_reachable({build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5), 0}, {0}, {{0, 4}},
                                      ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_FALSE(result.value()[0]->has_implicit_terminator());
  EXPECT_EQ(result.value()[0]->successor_issue(), Issue::MissingFallthrough);
}

TEST_F(ReachableCfg, ClippedCalleeDoesNotDiscardCallContinuation) {
  constexpr rj_code_arch_t arch = ROCJITSU_CODE_ARCH_CDNA5;
  const auto result = build_reachable(
      {build_s_call_b64(4, 2, arch), build_s_endpgm(arch), kInvalid, build_s_nop(0, arch), 0}, {0},
      {{0, 8}, {12, 4}}, arch);
  ASSERT_TRUE(result.succeeded()) << error_.message();
  ASSERT_EQ(offsets(result.value()), (std::vector<uint64_t>{0, 4, 12}));
  EXPECT_EQ(successors(*result.value()[0]), (std::vector<uint64_t>{4, 12}));
  EXPECT_EQ(result.value()[2]->successor_issue(), Issue::MissingFallthrough);
}

TEST_F(ReachableCfg, RejectsAmbiguousSectionsAndReachedPartialWords) {
  auto decoder = Decoder::create(kArch);
  TestCodeObject object;
  constexpr std::array<uint64_t, 1> entries{0};
  EXPECT_TRUE(BasicBlock::build_reachable(object, *decoder, kArch, entries).failed());
  const std::array<uint32_t, 1> words{kNop};
  object.add_text(words, 1);
  EXPECT_TRUE(BasicBlock::build_reachable(object, *decoder, kArch, entries).failed());
  object.add_text(words);
  EXPECT_TRUE(BasicBlock::build_reachable(object, *decoder, kArch, entries).failed());
}

TEST_F(ReachableCfg, FullAndReachableConstructionAgreeOnClosedGraphs) {
  for (auto arch : {ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    const uint32_t nop = build_s_nop(0, arch);
    const uint32_t end = build_s_endpgm(arch);
    std::vector<std::vector<uint32_t>> programs{
        {nop, build_s_branch(-1, arch)},
        {build_s_call_b64(4, 1, arch), end, build_s_setpc_b64(4, arch)},
        {nop, end},
    };
    if (arch == ROCJITSU_CODE_ARCH_CDNA5)
      programs.push_back({nop, 0}); // Existing implicit termination at zero padding.
    for (const auto &words : programs) {
      const auto reachable = build_reachable(words, {0}, {}, arch);
      ASSERT_TRUE(reachable.succeeded()) << error_.message();
      auto decoder = Decoder::create(arch);
      constexpr std::array<uint64_t, 1> entries{0};
      const auto full = BasicBlock::build(*object_, *decoder, arch, error_.emitter(), entries,
                                          ExternalEntryPolicy::ExplicitOnly);
      ASSERT_TRUE(full.succeeded()) << error_.message();
      ASSERT_EQ(offsets(reachable.value()), offsets(full.value()));
      for (size_t index = 0; index < full.value().size(); ++index) {
        const auto &expected = *full.value()[index];
        const auto &actual = *reachable.value()[index];
        EXPECT_EQ(actual.end_offset(), expected.end_offset());
        EXPECT_EQ(actual.num_instructions(), expected.num_instructions());
        EXPECT_EQ(successors(actual), successors(expected));
        EXPECT_EQ(actual.has_terminator(), expected.has_terminator());
        EXPECT_EQ(actual.has_implicit_terminator(), expected.has_implicit_terminator());
        EXPECT_EQ(actual.successor_issue(), expected.successor_issue());
        ASSERT_EQ(actual.call_edges().size(), expected.call_edges().size());
        for (size_t edge = 0; edge < actual.call_edges().size(); ++edge) {
          const auto &actual_call = actual.call_edges()[edge];
          const auto &expected_call = expected.call_edges()[edge];
          EXPECT_EQ(actual_call.kind, expected_call.kind);
          EXPECT_EQ(actual_call.callee->start_offset(), expected_call.callee->start_offset());
          EXPECT_EQ(actual_call.continuation->start_offset(),
                    expected_call.continuation->start_offset());
          EXPECT_EQ(actual_call.source_call_offset, expected_call.source_call_offset);
          EXPECT_EQ(actual_call.return_sreg, expected_call.return_sreg);
        }
      }
      expect_inverse_edges(full.value());
      expect_inverse_edges(reachable.value());
    }
  }
}

TEST_F(ReachableCfg, SupportsTargetSpecificDirectBranchEncodings) {
  for (auto arch :
       {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3,
        ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    const auto result =
        build_reachable({build_s_branch(1, arch), kInvalid, build_s_endpgm(arch)}, {0}, {}, arch);
    ASSERT_TRUE(result.succeeded()) << error_.message();
    ASSERT_EQ(offsets(result.value()), (std::vector<uint64_t>{0, 8}));
    EXPECT_EQ(successors(*result.value()[0]), (std::vector<uint64_t>{8}));
  }
}

// A CDNA4 getpc/add/addc/setpc chain targeting byte 24, across an undecodable gap.
static std::vector<uint32_t> indirect_branch_words() {
  return {build_s_getpc_b64(8, kArch),
          build_s_add_u32(8, 8, 255, kArch),
          20,
          build_s_addc_u32(9, 9, 128, kArch),
          build_s_setpc_b64(8, kArch),
          kInvalid,
          kEnd};
}

TEST_F(ReachableCfg, DiscoversIndirectTargetAcrossMalformedGap) {
  const auto result = build_reachable(indirect_branch_words());
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_EQ(offsets(result.value()), (std::vector<uint64_t>{0, 16, 24}));
  const BasicBlock &consumer = *result.value()[1];
  ASSERT_EQ(consumer.static_indirect_call_fixups().size(), 1u);
  EXPECT_EQ(consumer.static_indirect_call_fixups()[0].source_target_offset, 24u);
  EXPECT_EQ(successors(consumer), (std::vector<uint64_t>{24}));
  // Concrete edges remain subject to consumer-side completeness/return analysis.
  EXPECT_EQ(consumer.successor_issue(), Issue::IndirectControlFlow);
  expect_inverse_edges(result.value());
}

TEST_F(ReachableCfg, DiscoversReturningCallAndContinuation) {
  auto words = indirect_branch_words();
  words[4] = build_s_swappc_b64(30, 8, kArch);
  words[5] = kEnd;
  words[6] = build_s_setpc_b64(30, kArch);
  const auto result = build_reachable(words);
  ASSERT_TRUE(result.succeeded()) << error_.message();
  const BasicBlock &consumer = *result.value()[1];
  ASSERT_EQ(consumer.call_edges().size(), 1u);
  EXPECT_EQ(consumer.call_edges()[0].callee->start_offset(), 24u);
  EXPECT_EQ(consumer.call_edges()[0].continuation->start_offset(), 20u);
  EXPECT_EQ(consumer.call_edges()[0].return_sreg, 30u);
  expect_inverse_edges(result.value());
}

TEST_F(ReachableCfg, RepeatsDiscoveryThroughNewlyDecodedCode) {
  auto words = indirect_branch_words();
  words.resize(13, kInvalid);
  const auto second = indirect_branch_words();
  std::copy(second.begin(), second.end(), words.begin() + 6);
  const auto result = build_reachable(words);
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_EQ(offsets(result.value()), (std::vector<uint64_t>{0, 16, 24, 40, 48}));
  EXPECT_EQ(successors(*result.value()[3]), (std::vector<uint64_t>{48}));
  expect_inverse_edges(result.value());
}

TEST_F(ReachableCfg, LowerAddressCalleeKeepsCallerFacts) {
  // The helper at zero consumes s[8:9] built by the real entry at byte 8.
  // Marking the lowest decoded instruction as external would poison that fact.
  const std::vector<uint32_t> words = {build_s_setpc_b64(8, kArch),
                                       kInvalid,
                                       build_s_getpc_b64(8, kArch),
                                       build_s_add_u32(8, 8, 255, kArch),
                                       24,
                                       build_s_addc_u32(9, 9, 128, kArch),
                                       branch(-7),
                                       kInvalid,
                                       kInvalid,
                                       kEnd};
  const auto result = build_reachable(words, {8});
  ASSERT_TRUE(result.succeeded()) << error_.message();
  const BasicBlock &consumer = *result.value().front();
  ASSERT_EQ(consumer.static_indirect_call_fixups().size(), 1u);
  EXPECT_FALSE(consumer.static_indirect_call_fixups()[0].source_incomplete);
  EXPECT_EQ(consumer.static_indirect_call_fixups()[0].source_target_offset, 36u);
  EXPECT_EQ(successors(consumer), (std::vector<uint64_t>{36}));
}

TEST_F(ReachableCfg, DecodeSeedDoesNotBecomeExternalEntry) {
  const std::vector<uint32_t> words = {build_s_getpc_b64(8, kArch),
                                       build_s_add_u32(8, 8, 255, kArch),
                                       24,
                                       build_s_addc_u32(9, 9, 128, kArch),
                                       branch(0),
                                       build_s_setpc_b64(8, kArch),
                                       kInvalid,
                                       kEnd};
  const auto result = build_reachable(words, {0}, {}, kArch, {20});
  ASSERT_TRUE(result.succeeded()) << error_.message();
  const auto &fixups = result.value()[1]->static_indirect_call_fixups();
  ASSERT_EQ(fixups.size(), 1u);
  EXPECT_FALSE(fixups[0].source_incomplete);
  EXPECT_EQ(fixups[0].source_target_offset, 28u);
}

TEST_F(ReachableCfg, IndependentDecodeSeedCompletesCrossBlockIndirectClosure) {
  const std::vector<uint32_t> words = {build_s_getpc_b64(8, kArch),
                                       build_s_add_u32(8, 8, 255, kArch),
                                       24,
                                       build_s_addc_u32(9, 9, 128, kArch),
                                       branch(0),
                                       build_s_setpc_b64(8, kArch),
                                       kInvalid,
                                       kEnd};
  const auto result = build_reachable(words, {}, {}, kArch, {0});
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_EQ(offsets(result.value()), (std::vector<uint64_t>{0, 20, 28}));
  const BasicBlock &consumer = *result.value()[1];
  ASSERT_EQ(consumer.static_indirect_call_fixups().size(), 1u);
  const auto &fixup = consumer.static_indirect_call_fixups()[0];
  EXPECT_EQ(fixup.source_call_offset, 20u);
  EXPECT_EQ(fixup.source_target_offset, 28u);
  EXPECT_FALSE(fixup.source_incomplete);
  EXPECT_EQ(successors(consumer), (std::vector<uint64_t>{28}));
  expect_inverse_edges(result.value());
}

TEST_F(ReachableCfg, IndependentSeedClosurePrecedesAnOptionalLiteralSeed) {
  const std::vector<uint32_t> words = {build_s_getpc_b64(8, kArch),
                                       build_s_add_u32(8, 8, 255, kArch),
                                       24,
                                       build_s_addc_u32(9, 9, 128, kArch),
                                       branch(0),
                                       build_s_setpc_b64(8, kArch),
                                       kInvalid,
                                       build_s_add_u32(0, 0, 255, kArch),
                                       kEnd, // Literal also encodes an instruction.
                                       kEnd};
  const auto result = build_reachable(words, {}, {}, kArch, {32, 0, 20});
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_EQ(offsets(result.value()), (std::vector<uint64_t>{0, 20, 28}));
  EXPECT_EQ(result.value().back()->num_instructions(), 2u);
  ASSERT_EQ(result.value()[1]->static_indirect_call_fixups().size(), 1u);
  EXPECT_FALSE(result.value()[1]->static_indirect_call_fixups()[0].source_incomplete);
}

TEST_F(ReachableCfg, IndependentSeedPropagatesLaneStateWithoutAssumingAnEntryMode) {
  constexpr auto arch = ROCJITSU_CODE_ARCH_CDNA5;
  constexpr auto bank_zero = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 0});
  for (bool set_mode : {false, true}) {
    SCOPED_TRACE(set_mode);
    const std::vector<uint32_t> words = {
        set_mode ? bank_zero[0] : build_s_nop(0, arch), // 0: establish banking explicitly.
        0xBE804700u,                                    // 4: getpc s[0:1], PC=8.
        0xA980FE00u,
        56u,
        0u, // 8: add target delta -> 64.
        0xD761002Cu,
        0x02010000u, // 20: v44 lane 0 <- s0.
        0xD761002Cu,
        0x02010201u,             // 28: v44 lane 1 <- s1.
        build_s_branch(0, arch), // 36: branch to 40.
        0xD7600000u,
        0x0201012Cu, // 40: s0 <- v44 lane 0.
        0xD7600001u,
        0x0201032Cu,          // 48: s1 <- v44 lane 1.
        0xBE9E4900u,          // 56: swappc s[30:31], s[0:1].
        build_s_endpgm(arch), // 60: continuation.
        build_s_endpgm(arch), // 64: indirect target.
    };
    // Mark the restore block as another seed to ensure it preserves the stash
    // from its real predecessor instead of adding an empty external lane state.
    const auto result = build_reachable(words, {}, {}, arch, {0, 40});
    ASSERT_TRUE(result.succeeded()) << error_.message();
    size_t fixups = 0;
    for (const auto &block : result.value()) {
      for (const auto &fixup : block->static_indirect_call_fixups()) {
        ++fixups;
        EXPECT_EQ(fixup.source_call_offset, 56u);
        EXPECT_EQ(fixup.source_target_offset, 64u);
        EXPECT_FALSE(fixup.source_incomplete);
      }
    }
    EXPECT_EQ(fixups, set_mode ? 1u : 0u);
    const auto starts = offsets(result.value());
    EXPECT_EQ(std::ranges::find(starts, 64u) != starts.end(), set_mode);
    expect_inverse_edges(result.value());
  }
}

TEST_F(ReachableCfg, DecodeSeedsAreSeparateFromSplitPoints) {
  const auto result =
      build_reachable({kEnd, kInvalid, kNop, kEnd, kInvalid}, {0}, {}, kArch, {8}, {12, 16});
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_EQ(offsets(result.value()), (std::vector<uint64_t>{0, 8, 12}));
  EXPECT_EQ(successors(*result.value()[1]), (std::vector<uint64_t>{12}));
}

TEST_F(ReachableCfg, OptionalSeedInsideInstructionIsIgnored) {
  const auto result = build_reachable(indirect_branch_words(), {0}, {}, kArch, {8});
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_EQ(offsets(result.value()), (std::vector<uint64_t>{0, 16, 24}));
}

TEST_F(ReachableCfg, RecoveredTargetInsideInstructionFails) {
  auto words = indirect_branch_words();
  words[2] = 4; // getpc + 4 names the add's literal at byte 8.
  const auto result = build_reachable(words);
  EXPECT_TRUE(result.failed());
  EXPECT_NE(error_.message().find("overlaps an instruction"), std::string::npos);
}

TEST_F(ReachableCfg, RecoveredTargetOutsidePermittedRangeRemainsMissing) {
  const auto result = build_reachable(indirect_branch_words(), {0}, {{0, 20}});
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_EQ(offsets(result.value()), (std::vector<uint64_t>{0, 16}));
  EXPECT_EQ(result.value()[1]->successor_issue(), Issue::MissingBranchTarget);
  EXPECT_TRUE(result.value()[1]->successors().empty());
}

TEST_F(ReachableCfg, MalformedRecoveredTargetFailsWithOffset) {
  auto words = indirect_branch_words();
  words[6] = kInvalid;
  const auto result = build_reachable(words);
  EXPECT_TRUE(result.failed());
  EXPECT_NE(error_.message().find("24"), std::string::npos);
}

TEST_F(ReachableCfg, UnalignedRecoveredTargetFails) {
  auto words = indirect_branch_words();
  words[2] = 21;
  const auto result = build_reachable(words);
  EXPECT_TRUE(result.failed());
  EXPECT_NE(error_.message().find("unaligned CFG target at byte 25"), std::string::npos);
}

TEST_F(ReachableCfg, DiscoveryRoundLimitReturnsFailure) {
  std::vector<uint32_t> words;
  for (size_t index = 0; index < 65; ++index) {
    auto chain = indirect_branch_words();
    chain.pop_back();
    words.insert(words.end(), chain.begin(), chain.end());
  }
  words.push_back(kEnd);
  const auto result = build_reachable(words);
  EXPECT_TRUE(result.failed());
  EXPECT_NE(error_.message().find("round limit"), std::string::npos);
}

TEST_F(ReachableCfg, PartialIndirectTargetsRetainIncompleteStatus) {
  // The taken arm reaches setpc with an unconstrained PC; the fallthrough arm
  // contributes one concrete target. Finding that target must not hide the unknown arm.
  const std::vector<uint32_t> words = {conditional(4),
                                       build_s_getpc_b64(8, kArch),
                                       build_s_add_u32(8, 8, 255, kArch),
                                       24,
                                       build_s_addc_u32(9, 9, 128, kArch),
                                       build_s_setpc_b64(8, kArch),
                                       kInvalid,
                                       kInvalid,
                                       kEnd};
  const auto result = build_reachable(words);
  ASSERT_TRUE(result.succeeded()) << error_.message();
  const auto consumer = std::ranges::find_if(
      result.value(), [](const auto &block) { return block->start_offset() == 20; });
  ASSERT_NE(consumer, result.value().end());
  ASSERT_EQ((*consumer)->static_indirect_call_fixups().size(), 1u);
  EXPECT_TRUE((*consumer)->static_indirect_call_fixups()[0].source_incomplete);
  EXPECT_EQ((*consumer)->successor_issue(), Issue::IndirectControlFlow);
  EXPECT_EQ(successors(**consumer), (std::vector<uint64_t>{32}));
}

// Two paths produce targets at bytes 40 and 44 for the consumer at byte 32.
static std::vector<uint32_t> mixed_indirect_targets(bool call) {
  return {conditional(4),
          build_s_getpc_b64(8, kArch),
          build_s_add_u32(8, 8, 255, kArch),
          32,
          branch(3),
          build_s_getpc_b64(8, kArch),
          build_s_add_u32(8, 8, 255, kArch),
          20,
          call ? build_s_swappc_b64(30, 8, kArch) : build_s_setpc_b64(8, kArch),
          kEnd,
          build_s_setpc_b64(30, kArch),
          kEnd};
}

TEST_F(ReachableCfg, ExcludedIndirectCallTargetKeepsContinuation) {
  for (bool exclude_last : {false, true}) {
    SCOPED_TRACE(exclude_last);
    auto words = mixed_indirect_targets(true);
    if (exclude_last)
      std::swap(words[10], words[11]);
    const std::vector<Range> ranges =
        exclude_last ? std::vector<Range>{{0, 44}} : std::vector<Range>{{0, 40}, {44, 4}};
    const auto result = build_reachable(words, {0}, ranges);
    ASSERT_TRUE(result.succeeded()) << error_.message();
    const auto consumer = std::ranges::find_if(
        result.value(), [](const auto &block) { return block->start_offset() == 32; });
    ASSERT_NE(consumer, result.value().end());
    ASSERT_EQ((*consumer)->static_indirect_call_fixups().size(), 2u);
    EXPECT_EQ((*consumer)->successor_issue(), Issue::MissingCallTarget);
    EXPECT_EQ(successors(**consumer), (std::vector<uint64_t>{36, exclude_last ? 40u : 44u}));
    expect_inverse_edges(result.value());
  }
}

TEST_F(ReachableCfg, ExcludedIndirectTargetsKeepOuterCallContinuation) {
  // Exercise both a nested swappc and a tail setpc in the directly called helper.
  for (bool nested_call : {false, true}) {
    SCOPED_TRACE(nested_call);
    auto words = mixed_indirect_targets(nested_call);
    words.insert(words.begin(), {build_s_call_b64(4, 1, kArch), kEnd});
    const auto result = build_reachable(words, {0}, {{0, 48}, {52, 4}});
    ASSERT_TRUE(result.succeeded()) << error_.message();
    const auto &caller = *result.value().front();
    EXPECT_EQ(successors(caller), (std::vector<uint64_t>{4, 8}));
    EXPECT_TRUE(caller.call_edges().empty());
    expect_inverse_edges(result.value());
  }
}

TEST_F(ReachableCfg, OptionalSeedInsideIndirectlyReachedInstructionIsIgnored) {
  auto words = indirect_branch_words();
  words.back() = build_s_add_u32(0, 0, 255, kArch);
  words.push_back(kEnd); // Literal word also happens to encode a standalone endpgm.
  words.push_back(kEnd);
  const auto result = build_reachable(words, {0}, {}, kArch, {28});
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_EQ(offsets(result.value()), (std::vector<uint64_t>{0, 16, 24}));
  EXPECT_EQ(result.value().back()->num_instructions(), 2u);
}

TEST_F(ReachableCfg, OptionalSeedInsideAnotherSeedsIndirectClosureIsIgnored) {
  auto words = indirect_branch_words();
  words.back() = build_s_add_u32(0, 0, 255, kArch);
  words.push_back(kEnd);
  words.push_back(kEnd);
  const auto result = build_reachable(words, {}, {}, kArch, {0, 28});
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_EQ(offsets(result.value()), (std::vector<uint64_t>{0, 16, 24}));
  EXPECT_EQ(result.value().back()->num_instructions(), 2u);
}

TEST_F(ReachableCfg, IndependentSeedsDoNotExhaustTheIndirectClosureBudget) {
  const std::vector<uint32_t> words(80, kEnd);
  std::vector<uint64_t> seeds;
  for (size_t index = 0; index < words.size(); ++index)
    seeds.push_back(index * sizeof(uint32_t));
  const auto result = build_reachable(words, {}, {}, kArch, seeds);
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_EQ(result.value().size(), words.size());
}

TEST_F(ReachableCfg, RealTargetCannotBeHiddenByOptionalSeed) {
  auto words = indirect_branch_words();
  words[2] = 4;
  const auto result = build_reachable(words, {0}, {}, kArch, {8});
  EXPECT_TRUE(result.failed());
  EXPECT_NE(error_.message().find("overlaps an instruction at byte 8"), std::string::npos);
}

TEST_F(ReachableCfg, SeedOnlyPaddingReturnsEmptyGraph) {
  const auto result = build_reachable({0, 0}, {}, {}, ROCJITSU_CODE_ARCH_CDNA5, {0});
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_TRUE(result.value().empty());
}

TEST_F(ReachableCfg, SharedDecodePoliciesKeepDistinctCoverage) {
  TestCodeObject object;
  const std::vector<uint32_t> words = {branch(1), kInvalid, kEnd};
  object.add_text(words);
  auto decoder = Decoder::create(kArch);
  const std::array<uint64_t, 1> entries = {0};
  BasicBlock::BuildOptions options{.entries = entries};
  const auto reachable = BasicBlock::build_cfg(object, *decoder, kArch, options, error_.emitter());
  ASSERT_TRUE(reachable.succeeded()) << error_.message();
  EXPECT_EQ(offsets(reachable.value()), (std::vector<uint64_t>{0, 8}));
  options.decode_policy = BasicBlock::DecodePolicy::FullSection;
  const auto eager = BasicBlock::build_cfg(object, *decoder, kArch, options, error_.emitter());
  EXPECT_TRUE(eager.failed());
  EXPECT_NE(error_.message().find("byte offset 4"), std::string::npos);
}

TEST_F(ReachableCfg, FullSectionPolicyRetainsImplicitFirstEntry) {
  TestCodeObject object;
  const std::vector<uint32_t> words = {build_s_getpc_b64(8, kArch),
                                       build_s_add_u32(8, 8, 255, kArch),
                                       28,
                                       build_s_addc_u32(9, 9, 128, kArch),
                                       branch(0),
                                       build_s_setpc_b64(8, kArch),
                                       kEnd,
                                       kEnd,
                                       kEnd};
  object.add_text(words);
  auto decoder = Decoder::create(kArch);
  const std::array<uint64_t, 1> entries = {24};
  const auto eager = BasicBlock::build_cfg(
      object, *decoder, kArch,
      {.decode_policy = BasicBlock::DecodePolicy::FullSection, .entries = entries},
      error_.emitter());
  ASSERT_TRUE(eager.succeeded()) << error_.message();
  size_t fixup_count = 0;
  for (const auto &block : eager.value())
    fixup_count += block->static_indirect_call_fixups().size();
  EXPECT_EQ(fixup_count, 1u);
  const auto reachable = BasicBlock::build_cfg(object, *decoder, kArch, {.entries = entries});
  ASSERT_TRUE(reachable.succeeded());
  EXPECT_EQ(offsets(reachable.value()), (std::vector<uint64_t>{24}));
}

TEST_F(ReachableCfg, SharedDecodePoliciesRejectInapplicableOptions) {
  TestCodeObject object;
  const std::vector<uint32_t> words = {kEnd};
  object.add_text(words);
  auto decoder = Decoder::create(kArch);
  const std::array<uint64_t, 1> entries = {0};
  const std::array<Range, 1> ranges = {{{0, 4}}};
  BasicBlock::BuildOptions options{.decode_policy = BasicBlock::DecodePolicy::FullSection,
                                   .entries = entries,
                                   .permitted_ranges = ranges};
  EXPECT_TRUE(BasicBlock::build_cfg(object, *decoder, kArch, options, error_.emitter()).failed());
  EXPECT_NE(error_.message().find("does not accept ranges"), std::string::npos);
  options.permitted_ranges = {};
  options.decode_seeds = entries;
  EXPECT_TRUE(BasicBlock::build_cfg(object, *decoder, kArch, options, error_.emitter()).failed());
  options.decode_policy = BasicBlock::DecodePolicy::Reachable;
  options.entry_policy = ExternalEntryPolicy::InferPredecessorless;
  EXPECT_TRUE(BasicBlock::build_cfg(object, *decoder, kArch, options, error_.emitter()).failed());
  EXPECT_NE(error_.message().find("requires explicit external entries"), std::string::npos);
}

} // namespace
} // namespace rocjitsu

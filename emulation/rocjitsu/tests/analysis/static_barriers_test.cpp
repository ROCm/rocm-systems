// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/analysis/static_barriers.h"

#include "rocjitsu/code/analysis/instruction_boundaries.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/code_object.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"
#include "static_barrier_test_targets.h"
#include "util/diagnostic.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace {

constexpr uint32_t kSignal = 0xbe804ec1u;
constexpr uint32_t kWait = 0xbf94ffffu;
constexpr uint32_t kMov = 0xbe800081u;
constexpr uint32_t kEnd = 0xbfb00000u;

class BarrierTextSection final : public Section {
public:
  explicit BarrierTextSection(std::span<const uint32_t> words)
      : Section(".text", std::make_unique<char[]>(words.size_bytes())), size_(words.size_bytes()) {
    std::memcpy(data_.get(), words.data(), size_);
  }
  size_t size() const override { return size_; }
  uint32_t sectionHeaderNameIdx() const override { return 0; }
  uint64_t sectionOffset() const override { return 0; }

private:
  size_t size_;
};

class BarrierCodeObject final : public CodeObject {
public:
  explicit BarrierCodeObject(std::span<const uint32_t> words) {
    sections_.push_back(std::make_unique<BarrierTextSection>(words));
    text_sections_.push_back(sections_.back().get());
  }
};

class BarrierProgram {
public:
  ::testing::AssertionResult decode(const test::DropTargetRow &row,
                                    std::span<const uint32_t> words) {
    block_views_.clear();
    blocks_.clear();
    boundaries_ = {};
    decode_error_("");
    object_ = std::make_unique<BarrierCodeObject>(words);
    decoder_ = Decoder::create(default_isa_target_registry(), row.target);
    if (!decoder_)
      return ::testing::AssertionFailure() << "no concrete decoder for " << row.name;
    FailureOr<std::vector<std::unique_ptr<BasicBlock>>> decoded =
        BasicBlock::build(*object_, *decoder_, row.arch, decode_error_.emitter(), {},
                          ExternalEntryPolicy::InferPredecessorless, {}, row.target);
    if (decoded.failed())
      return ::testing::AssertionFailure() << decode_error_.message();
    blocks_ = std::move(decoded.value());
    for (const std::unique_ptr<BasicBlock> &block : blocks_) {
      block_views_.push_back(block.get());
      for (const Instruction &instruction : block->instructions())
        boundaries_.record(instruction, 0);
    }
    return ::testing::AssertionSuccess();
  }

  FailureOr<std::vector<StaticBarrier>> qualify(rj_code_target_id_t target,
                                                StaticBarrierError &error) const {
    return qualify_static_barriers(block_views_, target, boundaries_, error);
  }

  const std::vector<BasicBlock *> &blocks() const { return block_views_; }
  const std::string &decode_error() const { return decode_error_.message(); }

private:
  std::unique_ptr<BarrierCodeObject> object_;
  std::unique_ptr<Decoder> decoder_;
  std::vector<std::unique_ptr<BasicBlock>> blocks_;
  std::vector<BasicBlock *> block_views_;
  InstructionBoundaries boundaries_;
  util::StringDiagnostic decode_error_;
};

void expect_rejection(const test::DropTargetRow &row, const std::vector<uint32_t> &words,
                      StaticBarrierErrorCode code, uint64_t offset,
                      std::string_view diagnostic = {}) {
  BarrierProgram program;
  ASSERT_TRUE(program.decode(row, words));
  StaticBarrierError error;
  const FailureOr<std::vector<StaticBarrier>> result = program.qualify(row.target, error);
  EXPECT_TRUE(result.failed());
  EXPECT_EQ(error.code, code) << error.message;
  EXPECT_EQ(error.offset, offset) << error.message;
  EXPECT_NE(error.message.find(diagnostic), std::string::npos) << error.message;
}

class StaticBarriersTarget : public ::testing::TestWithParam<test::DropTargetRow> {};
class StaticBarriersSplit : public ::testing::TestWithParam<test::DropTargetRow> {};
class StaticBarriersRdna3 : public ::testing::TestWithParam<test::DropTargetRow> {};

TEST_P(StaticBarriersTarget, CompleteBarriersRetainFamilyAndSourceFacts) {
  const test::DropTargetRow &row = GetParam();
  const std::optional<StaticBarrierPolicy> policy = static_barrier_policy(row.target);
  ASSERT_TRUE(policy);
  EXPECT_NE(std::ranges::find(policy->wave_sizes, row.wave_size), policy->wave_sizes.end());
  const StaticBarrierForm form =
      row.split ? StaticBarrierForm::BuiltinSplitPair : StaticBarrierForm::FullSingleton;
  EXPECT_EQ(policy->form, form);

  const uint32_t full = row.arch == ROCJITSU_CODE_ARCH_RDNA3 ? 0xbfbd0000u : 0xbf8a0000u;
  const std::vector<uint32_t> words =
      row.split ? std::vector<uint32_t>{kMov, kSignal, kMov, kWait, kMov, kSignal, kMov, kWait}
                : std::vector<uint32_t>{kMov, full, kMov, kMov, full};
  const std::vector<uint32_t> pristine = words;
  BarrierProgram program;
  ASSERT_TRUE(program.decode(row, words));
  StaticBarrierError error = {
      .code = StaticBarrierErrorCode::UnsupportedForm, .offset = 100, .message = "stale"};
  const FailureOr<std::vector<StaticBarrier>> barriers = program.qualify(row.target, error);
  ASSERT_TRUE(barriers.succeeded()) << error.message;
  EXPECT_EQ(error.code, StaticBarrierErrorCode::None);
  EXPECT_EQ(error.offset, 0u);
  EXPECT_TRUE(error.message.empty());
  ASSERT_EQ(barriers.value().size(), 2u);
  const std::array<uint64_t, 2> first_offsets =
      row.split ? std::array<uint64_t, 2>{4, 20} : std::array<uint64_t, 2>{4, 16};
  for (size_t index = 0; index < barriers.value().size(); ++index) {
    const StaticBarrier &barrier = barriers.value()[index];
    EXPECT_EQ(barrier.form, form);
    EXPECT_EQ(barrier.block_offset, 0u);
    EXPECT_EQ(barrier.barrier_id, -1);
    EXPECT_EQ(barrier.scope, StaticBarrierScope::Workgroup);
    ASSERT_EQ(barrier.members.size(), row.split ? 2u : 1u);
    for (size_t member_index = 0; member_index < barrier.members.size(); ++member_index) {
      const StaticBarrierMember &member = barrier.members[member_index];
      EXPECT_EQ(member.offset, first_offsets[index] + member_index * 8);
      ASSERT_EQ(member.bytes.size(), sizeof(uint32_t));
      EXPECT_EQ(std::memcmp(member.bytes.data(),
                            reinterpret_cast<const uint8_t *>(words.data()) + member.offset,
                            member.bytes.size()),
                0);
    }
  }
  EXPECT_EQ(words, pristine);
}

TEST_P(StaticBarriersSplit, SemanticBuiltinIdsFollowFamilyPolicy) {
  const test::DropTargetRow &row = GetParam();
  for (const int32_t id : {-1, -2, -3, -4}) {
    SCOPED_TRACE(id);
    const uint32_t signal = 0xbe804e00u | static_cast<uint32_t>(192 - id);
    const uint32_t wait = 0xbf940000u | static_cast<uint16_t>(id);
    const std::vector<uint32_t> words = {signal, kMov, wait};
    BarrierProgram program;
    if (row.arch == ROCJITSU_CODE_ARCH_RDNA4 && id < -2) {
      // Those signal selectors are outside RDNA4's decoder operand domain.
      EXPECT_FALSE(program.decode(row, words));
      EXPECT_NE(program.decode_error().find("invalid scalar source selector"), std::string::npos);
      continue;
    }
    ASSERT_TRUE(program.decode(row, words));
    StaticBarrierError error;
    const FailureOr<std::vector<StaticBarrier>> barriers = program.qualify(row.target, error);
    if (id == -2 || id == -4) {
      EXPECT_TRUE(barriers.failed());
      EXPECT_EQ(error.code, StaticBarrierErrorCode::UnsupportedForm);
      EXPECT_EQ(error.offset, 0u);
      EXPECT_NE(error.message.find("trap-only"), std::string::npos);
      continue;
    }
    ASSERT_TRUE(barriers.succeeded()) << error.message;
    ASSERT_EQ(barriers.value().size(), 1u);
    const StaticBarrier &barrier = barriers.value().front();
    EXPECT_EQ(barrier.barrier_id, id);
    EXPECT_EQ(barrier.scope,
              id == -1 ? StaticBarrierScope::Workgroup : StaticBarrierScope::Cluster);
    ASSERT_EQ(barrier.members.size(), 2u);
    EXPECT_EQ(barrier.members[0].offset, 0u);
    EXPECT_EQ(barrier.members[1].offset, 8u);
  }
}

TEST_P(StaticBarriersSplit, RejectsNonImmediateAndOutOfPolicyOperands) {
  const test::DropTargetRow &row = GetParam();
  for (const uint32_t selector : {125u, 128u, 129u, 159u}) {
    SCOPED_TRACE(selector);
    expect_rejection(row, {0xbe804e00u | selector, kWait}, StaticBarrierErrorCode::UnsupportedForm,
                     0);
  }
  for (const int32_t id : {0, 1, 31, -5, -16}) {
    SCOPED_TRACE(id);
    expect_rejection(row, {kSignal, 0xbf940000u | static_cast<uint16_t>(id)},
                     StaticBarrierErrorCode::UnsupportedForm, 4);
  }
  for (const uint32_t selector : {197u, 208u, 255u}) {
    SCOPED_TRACE(selector);
    BarrierProgram program;
    const std::vector<uint32_t> words = {0xbe804e00u | selector, 0xffffffffu, kWait};
    EXPECT_FALSE(program.decode(row, words));
    const std::string_view diagnostic =
        row.arch == ROCJITSU_CODE_ARCH_CDNA5 && selector == 255
            ? "s_barrier_signal does not support 32-bit literals at .text byte offset 0"
            : "invalid scalar source selector";
    EXPECT_NE(program.decode_error().find(diagnostic), std::string::npos) << program.decode_error();
  }
}

TEST_P(StaticBarriersRdna3, RejectsNonSingletonFormsAndClauseMembers) {
  const test::DropTargetRow &row = GetParam();
  const FailureOr<uint32_t> end = build_s_endpgm(row.arch);
  ASSERT_TRUE(end.succeeded());
  expect_rejection(row, {0xd8740000u, 0, 0xbfbd0000u, end.value()},
                   StaticBarrierErrorCode::UnsupportedForm, 0);
  expect_rejection(row, {0xbf850000u, 0xbfbd0000u, end.value()},
                   StaticBarrierErrorCode::UnsupportedForm, 4);
}

TEST(StaticBarriers, RejectsPartialInterleavedDynamicAndNamedLifecycle) {
  const test::DropTargetRow &row = test::kDropRows[4];
  struct RejectedCase {
    std::vector<uint32_t> words;
    StaticBarrierErrorCode code;
    uint64_t offset;
  };
  const std::array cases = {
      RejectedCase{
          .words = {kSignal, kEnd}, .code = StaticBarrierErrorCode::IncompleteBarrier, .offset = 0},
      RejectedCase{
          .words = {kWait, kEnd}, .code = StaticBarrierErrorCode::IncompleteBarrier, .offset = 0},
      RejectedCase{.words = {kSignal, kSignal, kWait, kEnd},
                   .code = StaticBarrierErrorCode::IncompleteBarrier,
                   .offset = 4},
      RejectedCase{.words = {kSignal, 0xbf94fffeu, kEnd},
                   .code = StaticBarrierErrorCode::UnsupportedForm,
                   .offset = 4},
      RejectedCase{.words = {0xbe804e7du, kWait, kEnd},
                   .code = StaticBarrierErrorCode::UnsupportedForm,
                   .offset = 0},
      RejectedCase{.words = {0xbe804e81u, 0xbf940001u, kEnd},
                   .code = StaticBarrierErrorCode::UnsupportedForm,
                   .offset = 0},
      RejectedCase{.words = {0xbe804fc1u, kWait, kEnd},
                   .code = StaticBarrierErrorCode::UnsupportedForm,
                   .offset = 0},
      RejectedCase{.words = {kSignal, 0xbe8050c1u, kWait, kEnd},
                   .code = StaticBarrierErrorCode::UnsupportedForm,
                   .offset = 4},
  };
  for (const RejectedCase &rejected : cases)
    expect_rejection(row, rejected.words, rejected.code, rejected.offset);
}

TEST_P(StaticBarriersSplit, TrapFormsCannotPrecedeAnAdmittedUserBarrier) {
  const test::DropTargetRow &row = GetParam();
  for (const int32_t id : {-2, -4}) {
    SCOPED_TRACE(id);
    const uint32_t signal = 0xbe804e00u | static_cast<uint32_t>(192 - id);
    const uint32_t wait = 0xbf940000u | static_cast<uint16_t>(id);
    const std::vector<uint32_t> words = {signal, wait, kSignal, kWait, kEnd};
    if (row.arch == ROCJITSU_CODE_ARCH_RDNA4 && id == -4) {
      BarrierProgram program;
      EXPECT_FALSE(program.decode(row, words));
      EXPECT_NE(program.decode_error().find("invalid scalar source selector"), std::string::npos);
    } else {
      expect_rejection(row, words, StaticBarrierErrorCode::UnsupportedForm, 0, "trap-only");
    }
  }
}

TEST_P(StaticBarriersSplit, RejectsStandaloneAndMismatchedTrapWaits) {
  const test::DropTargetRow &row = GetParam();
  for (const int32_t id : {-2, -4}) {
    SCOPED_TRACE(id);
    const uint32_t wait = 0xbf940000u | static_cast<uint16_t>(id);
    for (const bool with_signal : {false, true}) {
      SCOPED_TRACE(with_signal);
      std::vector<uint32_t> words = {wait, kEnd};
      if (with_signal)
        words.insert(words.begin(), kSignal);
      expect_rejection(row, words, StaticBarrierErrorCode::UnsupportedForm, with_signal ? 4 : 0,
                       "trap-only");
    }
  }
}

TEST(StaticBarriers, RejectsMismatchedAdmittedWorkgroupAndClusterPairs) {
  const test::DropTargetRow &row = test::kDropRows[4];
  for (const int32_t signal_id : {-1, -3}) {
    SCOPED_TRACE(signal_id);
    const int32_t wait_id = signal_id == -1 ? -3 : -1;
    const uint32_t signal = 0xbe804e00u | static_cast<uint32_t>(192 - signal_id);
    const uint32_t wait = 0xbf940000u | static_cast<uint16_t>(wait_id);
    expect_rejection(row, {signal, wait, kEnd}, StaticBarrierErrorCode::IncompleteBarrier, 4);
  }
}

TEST(StaticBarriers, RejectsBarrierInsideClause) {
  expect_rejection(test::kDropRows[4], {0xbf850001u, kSignal, kWait, kEnd},
                   StaticBarrierErrorCode::UnsupportedForm, 4);
}

TEST(StaticBarriers, RejectsCrossBlockPair) {
  const test::DropTargetRow &row = test::kDropRows[4];
  const FailureOr<uint32_t> branch = build_s_branch(0, row.arch);
  ASSERT_TRUE(branch.succeeded());
  BarrierProgram program;
  const std::vector<uint32_t> words = {kSignal, branch.value(), kWait, kEnd};
  ASSERT_TRUE(program.decode(row, words));
  ASSERT_EQ(program.blocks().size(), 2u);
  StaticBarrierError error;
  EXPECT_TRUE(program.qualify(row.target, error).failed());
  EXPECT_EQ(error.code, StaticBarrierErrorCode::IncompleteBarrier);
  EXPECT_EQ(error.offset, 0u);
  EXPECT_NE(error.message.find("same-block"), std::string::npos);
}

TEST(StaticBarriers, RejectsMissingBoundaryFacts) {
  const test::DropTargetRow &row = test::kDropRows[4];
  BarrierProgram program;
  const std::vector<uint32_t> words = {kSignal, kWait, kEnd};
  ASSERT_TRUE(program.decode(row, words));
  const InstructionBoundaries missing;
  StaticBarrierError error;
  EXPECT_TRUE(qualify_static_barriers(program.blocks(), row.target, missing, error).failed());
  EXPECT_EQ(error.code, StaticBarrierErrorCode::UnsupportedForm);
  EXPECT_EQ(error.offset, 0u);
  EXPECT_NE(error.message.find("full-text instruction boundaries"), std::string::npos);
}

TEST(StaticBarriers, RejectsFamiliesWithoutPolicyAndUnknownTargets) {
  const InstructionBoundaries boundaries;
  for (const rj_code_target_id_t target :
       {ROCJITSU_CODE_TARGET_GFX90A, ROCJITSU_CODE_TARGET_NUM_TARGETS, ROCJITSU_CODE_TARGET_INVALID,
        static_cast<rj_code_target_id_t>(-1), static_cast<rj_code_target_id_t>(123456)}) {
    SCOPED_TRACE(target);
    EXPECT_FALSE(static_barrier_policy(target));
    StaticBarrierError error;
    EXPECT_TRUE(qualify_static_barriers({}, target, boundaries, error).failed());
    EXPECT_EQ(error.code, StaticBarrierErrorCode::UnsupportedForm);
    EXPECT_EQ(error.offset, 0u);
    EXPECT_NE(error.message.find("no complete-barrier mutation policy"), std::string::npos);
  }
}

INSTANTIATE_TEST_SUITE_P(TargetWaves, StaticBarriersTarget, ::testing::ValuesIn(test::kDropRows),
                         [](const ::testing::TestParamInfo<test::DropTargetRow> &info) {
                           return info.param.name;
                         });
INSTANTIATE_TEST_SUITE_P(SplitTargets, StaticBarriersSplit,
                         ::testing::ValuesIn(std::span(test::kDropRows).first(6)),
                         [](const ::testing::TestParamInfo<test::DropTargetRow> &info) {
                           return info.param.name;
                         });

INSTANTIATE_TEST_SUITE_P(Rdna3Waves, StaticBarriersRdna3, ::testing::ValuesIn(test::kRdna3DropRows),
                         [](const ::testing::TestParamInfo<test::DropTargetRow> &info) {
                           return info.param.name;
                         });

} // namespace
} // namespace rocjitsu

// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/analysis/def_use_chain.h"
#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/code_object.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/operand.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/isa/register_set.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace {

class TestOperand : public Operand {
public:
  TestOperand() = default;
  explicit TestOperand(RegisterRef ref) : Operand(ref.width * 32, ref.index), ref_(ref) {}

  std::optional<RegisterRef> to_register_ref() const override { return ref_; }

private:
  std::optional<RegisterRef> ref_;
};

class TestInstruction : public Instruction {
public:
  TestInstruction(std::string_view mnemonic, std::initializer_list<RegisterRef> defs = {},
                  std::initializer_list<RegisterRef> uses = {}, uint64_t flags = 0,
                  std::optional<int64_t> branch_delta = std::nullopt,
                  std::initializer_list<RegisterRef> implicit_uses = {})
      : Instruction(mnemonic, nullptr), implicit_uses_(implicit_uses), branch_delta_(branch_delta) {
    size_ = 4;
    flags_ = flags;

    for (RegisterRef ref : defs) {
      dst_storage_[num_dst_] = TestOperand(ref);
      dst_operands_[num_dst_] = &dst_storage_[num_dst_];
      ++num_dst_;
    }
    for (RegisterRef ref : uses) {
      src_storage_[num_src_] = TestOperand(ref);
      src_operands_[num_src_] = &src_storage_[num_src_];
      ++num_src_;
    }
  }

  std::optional<int64_t> branch_offset_bytes() const override { return branch_delta_; }

  void implicit_uses(RegisterSet &uses) const override {
    for (RegisterRef ref : implicit_uses_)
      uses.expand(ref);
  }

private:
  std::array<TestOperand, 2> dst_storage_{};
  std::array<TestOperand, 4> src_storage_{};
  std::vector<RegisterRef> implicit_uses_;
  std::optional<int64_t> branch_delta_;
};

class TestTextSection : public Section {
public:
  TestTextSection(std::unique_ptr<char[]> data, std::size_t size)
      : Section(".text", std::move(data)), size_(size) {}

  std::size_t size() const override { return size_; }
  uint32_t sectionHeaderNameIdx() const override { return 0; }
  uint64_t sectionOffset() const override { return 0; }

private:
  std::size_t size_;
};

class TestCodeObject : public CodeObject {
public:
  explicit TestCodeObject(std::vector<uint32_t> words) {
    const auto byte_size = words.size() * sizeof(uint32_t);
    image_.resize(byte_size);
    std::memcpy(image_.data(), words.data(), byte_size);

    auto data = std::make_unique<char[]>(byte_size);
    std::memcpy(data.get(), words.data(), byte_size);
    sections_.push_back(std::make_unique<TestTextSection>(std::move(data), byte_size));
    text_sections_.push_back(sections_.back().get());
  }
};

enum class TestOpcode : uint32_t {
  Nop = 0,
  End = 1,
  BranchBackToStart = 2,
  CBranchToElse = 3,
  BranchToJoin = 4,
  DefVgpr0 = 5,
  UseVgpr0 = 6,
  UseSgpr4 = 7,
  UseSgpr7 = 8,
  ReadWriteSgpr4 = 9,
  PredicatedDefSgpr4 = 10,
  ImplicitUseSgpr6Pair = 11,
  DefSgpr4 = 12,
  CBranchBackToUseSgpr4 = 13,
  CBranchToElseAfterTwo = 14,
  IndirectCall = 15,
  IndirectBranch = 16,
};

class TestDecoder : public Decoder {
public:
  Instruction *decode(const rj_code_binary_inst_t *inst) override {
    auto op = static_cast<TestOpcode>(*inst);
    switch (op) {
    case TestOpcode::Nop:
      return new TestInstruction("test_nop");
    case TestOpcode::End:
      return new TestInstruction("test_end", {}, {}, PROGRAM_TERMINATOR);
    case TestOpcode::BranchBackToStart:
      return new TestInstruction("test_branch_back", {}, {}, BRANCH, -8);
    case TestOpcode::CBranchToElse:
      return new TestInstruction("test_cbranch_else", {}, {}, COND_BRANCH, 4);
    case TestOpcode::BranchToJoin:
      return new TestInstruction("test_branch_join", {}, {}, BRANCH, 4);
    case TestOpcode::DefVgpr0:
      return new TestInstruction("test_def_v0", {{RegClass::VGPR, 0, 1}});
    case TestOpcode::UseVgpr0:
      return new TestInstruction("test_use_v0", {}, {{RegClass::VGPR, 0, 1}});
    case TestOpcode::UseSgpr4:
      return new TestInstruction("test_use_s4", {}, {{RegClass::SGPR, 4, 1}});
    case TestOpcode::UseSgpr7:
      return new TestInstruction("test_use_s7", {}, {{RegClass::SGPR, 7, 1}});
    case TestOpcode::ReadWriteSgpr4:
      return new TestInstruction("test_rw_s4", {{RegClass::SGPR, 4, 1}}, {{RegClass::SGPR, 4, 1}});
    case TestOpcode::PredicatedDefSgpr4:
      return new TestInstruction("test_pred_def_s4", {{RegClass::SGPR, 4, 1}}, {}, PREDICATED_DEF);
    case TestOpcode::ImplicitUseSgpr6Pair:
      return new TestInstruction("test_implicit_use_s6_pair", {}, {}, 0, std::nullopt,
                                 {{RegClass::SGPR, 6, 2}});
    case TestOpcode::DefSgpr4:
      return new TestInstruction("test_def_s4", {{RegClass::SGPR, 4, 1}});
    case TestOpcode::CBranchBackToUseSgpr4:
      return new TestInstruction("test_cbranch_back_to_use_s4", {}, {}, COND_BRANCH, -8);
    case TestOpcode::CBranchToElseAfterTwo:
      return new TestInstruction("test_cbranch_else_after_two", {}, {}, COND_BRANCH, 8);
    case TestOpcode::IndirectCall:
      return new TestInstruction("test_indirect_call", {}, {}, INDIRECT_CALL);
    case TestOpcode::IndirectBranch:
      return new TestInstruction("test_indirect_branch", {}, {}, INDIRECT_BRANCH);
    }
    return new TestInstruction("test_end", {}, {}, PROGRAM_TERMINATOR);
  }
};

std::vector<std::unique_ptr<BasicBlock>>
build_test_blocks(std::vector<TestOpcode> ops, std::span<const uint64_t> extra_leaders = {}) {
  std::vector<uint32_t> words;
  words.reserve(ops.size());
  for (TestOpcode op : ops)
    words.push_back(static_cast<uint32_t>(op));

  TestCodeObject co(std::move(words));
  TestDecoder decoder;
  return BasicBlock::build(co, decoder, ROCJITSU_CODE_ARCH_CDNA3, extra_leaders);
}

bool has_predecessor(const BasicBlock &block, const BasicBlock *pred) {
  return std::ranges::find(block.predecessors(), pred) != block.predecessors().end();
}

bool has_successor_start(const BasicBlock &block, uint64_t offset) {
  return std::ranges::any_of(block.successors(), [offset](const BasicBlock *succ) {
    return succ != nullptr && succ->start_offset() == offset;
  });
}

BasicBlock *block_starting_at(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                              uint64_t offset) {
  auto it = std::ranges::find_if(blocks, [offset](const auto &block) {
    return block != nullptr && block->start_offset() == offset;
  });
  return it == blocks.end() ? nullptr : it->get();
}

std::vector<BasicBlock *> block_scope(const std::vector<std::unique_ptr<BasicBlock>> &blocks) {
  std::vector<BasicBlock *> scope;
  scope.reserve(blocks.size());
  for (const auto &block : blocks)
    scope.push_back(block.get());
  return scope;
}

LivenessAnalysis analyze_scope(const std::vector<std::unique_ptr<BasicBlock>> &blocks) {
  auto scope = block_scope(blocks);
  return LivenessAnalysis(KernelBlockScope(scope));
}

uint32_t pack_sopc(uint32_t op, uint16_t ssrc0, uint16_t ssrc1) {
  constexpr uint32_t kSopcEncodingPrefix = 0x17e;
  return (kSopcEncodingPrefix << 23) | ((op & 0x7fu) << 16) | ((ssrc1 & 0xffu) << 8) |
         (ssrc0 & 0xffu);
}

uint32_t build_s_call_b64(uint16_t sdst, int16_t simm16) {
  constexpr uint32_t kSopkEncodingPrefix = 0xb;
  constexpr uint32_t kSCallB64Opcode = 0x15;
  return (kSopkEncodingPrefix << 28) | (kSCallB64Opcode << 23) | ((sdst & 0x7fu) << 16) |
         static_cast<uint16_t>(simm16);
}

TEST(RegisterSetAnalysis, KeepsRegisterClassesSeparate) {
  RegisterSet set;
  set.expand({RegClass::SGPR, 4, 1});

  EXPECT_TRUE(set.contains({RegClass::SGPR, 4, 1}));
  EXPECT_FALSE(set.contains({RegClass::VGPR, 4, 1}));
  EXPECT_FALSE(set.contains({RegClass::ACC_VGPR, 4, 1}));
}

TEST(RegisterSetAnalysis, IgnoresSpecialRegisterClasses) {
  RegisterSet set;
  set.expand({RegClass::EXEC, 0, 2});
  set.expand({RegClass::SCC, 0, 1});
  set.expand({RegClass::FLAT_SCRATCH, 0, 2});

  EXPECT_TRUE(set.none());
  EXPECT_FALSE(set.contains({RegClass::EXEC, 0, 1}));
  EXPECT_FALSE(set.contains({RegClass::SCC, 0, 1}));
  EXPECT_FALSE(set.contains({RegClass::FLAT_SCRATCH, 0, 2}));
}

TEST(RegisterSetAnalysis, GeneratedCdna4OperandsMapTrackedRegisterRefs) {
  cdna4::Operand sgpr(32, cdna4::OperandType::OPR_SRC, cdna4::OpSelSrc::OPR_SRC_SGPR_MIN + 7);
  cdna4::Operand vgpr(32, cdna4::OperandType::OPR_SRC, cdna4::OpSelSrc::OPR_SRC_VGPR_MIN + 7);
  cdna4::Operand acc(32, cdna4::OperandType::OPR_SRC_ACCVGPR,
                     cdna4::OpSelSrcAccvgpr::OPR_SRC_ACCVGPR_ACC_MIN + 7);
  cdna4::Operand imm32(32, cdna4::OperandType::OPR_SIMM32, 123);

  ASSERT_TRUE(sgpr.to_register_ref().has_value());
  EXPECT_EQ(*sgpr.to_register_ref(), (RegisterRef{RegClass::SGPR, 7, 1}));
  ASSERT_TRUE(vgpr.to_register_ref().has_value());
  EXPECT_EQ(*vgpr.to_register_ref(), (RegisterRef{RegClass::VGPR, 7, 1}));
  ASSERT_TRUE(acc.to_register_ref().has_value());
  EXPECT_EQ(*acc.to_register_ref(), (RegisterRef{RegClass::ACC_VGPR, 7, 1}));
  EXPECT_FALSE(imm32.to_register_ref().has_value());
}

TEST(CfgAnalysis, LoopBackEdgeLinksPredecessor) {
  auto blocks = build_test_blocks({TestOpcode::Nop, TestOpcode::BranchBackToStart});

  ASSERT_EQ(blocks.size(), 1u);
  ASSERT_EQ(blocks[0]->successors().size(), 1u);
  EXPECT_EQ(blocks[0]->successors()[0], blocks[0].get());
  EXPECT_TRUE(has_predecessor(*blocks[0], blocks[0].get()));
}

TEST(CfgAnalysis, IfElseSuccessorsAndPredecessorsAreInverse) {
  auto blocks = build_test_blocks(
      {TestOpcode::CBranchToElse, TestOpcode::BranchToJoin, TestOpcode::Nop, TestOpcode::End});

  ASSERT_EQ(blocks.size(), 4u);
  auto *entry = blocks[0].get();
  auto *then_block = blocks[1].get();
  auto *else_block = blocks[2].get();
  auto *join = blocks[3].get();

  ASSERT_EQ(entry->successors().size(), 2u);
  EXPECT_EQ(entry->successors()[0], else_block);
  EXPECT_EQ(entry->successors()[1], then_block);
  ASSERT_EQ(then_block->successors().size(), 1u);
  EXPECT_EQ(then_block->successors()[0], join);
  ASSERT_EQ(else_block->successors().size(), 1u);
  EXPECT_EQ(else_block->successors()[0], join);

  EXPECT_TRUE(has_predecessor(*then_block, entry));
  EXPECT_TRUE(has_predecessor(*else_block, entry));
  EXPECT_TRUE(has_predecessor(*join, then_block));
  EXPECT_TRUE(has_predecessor(*join, else_block));
}

TEST(CfgAnalysis, ExtraLeaderSplitsBlockAtKernelEntry) {
  std::array<uint64_t, 1> kernel_entries{8};
  auto blocks = build_test_blocks(
      {TestOpcode::Nop, TestOpcode::Nop, TestOpcode::UseSgpr4, TestOpcode::End}, kernel_entries);

  ASSERT_EQ(blocks.size(), 2u);
  ASSERT_EQ(blocks[0]->start_offset(), 0u);
  ASSERT_EQ(blocks[0]->end_offset(), 8u);
  ASSERT_EQ(blocks[1]->start_offset(), 8u);
  ASSERT_EQ(blocks[0]->successors().size(), 1u);
  EXPECT_EQ(blocks[0]->successors()[0], blocks[1].get());
  EXPECT_TRUE(has_predecessor(*blocks[1], blocks[0].get()));
}

TEST(CfgAnalysis, IndirectCallFallsThroughToReturnSuccessor) {
  auto blocks =
      build_test_blocks({TestOpcode::IndirectCall, TestOpcode::UseSgpr4, TestOpcode::End});

  ASSERT_EQ(blocks.size(), 2u);
  ASSERT_EQ(blocks[0]->successors().size(), 1u);
  EXPECT_EQ(blocks[0]->successors()[0], blocks[1].get());
  EXPECT_TRUE(has_predecessor(*blocks[1], blocks[0].get()));
}

TEST(CfgAnalysis, IndirectBranchHasNoStaticSuccessor) {
  auto blocks =
      build_test_blocks({TestOpcode::IndirectBranch, TestOpcode::UseSgpr4, TestOpcode::End});

  ASSERT_EQ(blocks.size(), 2u);
  EXPECT_TRUE(blocks[0]->successors().empty());
  EXPECT_TRUE(blocks[1]->predecessors().empty());
}

TEST(CfgAnalysis, RecoveredIndirectBranchEdgeStartsAtConsumerBlock) {
  constexpr uint16_t kPcSreg = 8;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;

  // The PC builder and setpc consumer are deliberately separated by an extra
  // leader. The recovered CFG edge belongs to the setpc block, because that is
  // where control flow actually leaves the straight-line path.
  std::vector<uint32_t> words = {
      pack_sop1(0x1c, kPcSreg, 0),                         // 0x00: s_getpc_b64.
      pack_sop2(0, kPcSreg, kPcSreg, kLiteralOperand),     // 0x04: s_add_u32.
      20,                                                  // 0x08: target delta.
      pack_sop2(4, kPcSreg + 1, kPcSreg + 1, kInlineInt0), // 0x0c: s_addc_u32.
      pack_sop1(0x1d, 0, kPcSreg),                         // 0x10: s_setpc_b64.
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),            // 0x14: not a successor.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),            // 0x18: recovered target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  constexpr std::array<uint64_t, 1> extra_leaders{16};
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4, extra_leaders);

  auto *builder = block_starting_at(blocks, 0);
  auto *consumer = block_starting_at(blocks, 16);
  auto *fallthrough = block_starting_at(blocks, 20);
  auto *target = block_starting_at(blocks, 24);
  ASSERT_NE(builder, nullptr);
  ASSERT_NE(consumer, nullptr);
  ASSERT_NE(fallthrough, nullptr);
  ASSERT_NE(target, nullptr);

  EXPECT_TRUE(builder->static_indirect_call_fixups().empty());
  ASSERT_EQ(builder->successors().size(), 1u);
  EXPECT_EQ(builder->successors()[0], consumer);

  ASSERT_EQ(consumer->static_indirect_call_fixups().size(), 1u);
  EXPECT_EQ(consumer->static_indirect_call_fixups()[0].source_call_offset, 16u);
  ASSERT_EQ(consumer->successors().size(), 1u);
  EXPECT_EQ(consumer->successors()[0], target);
  EXPECT_FALSE(has_predecessor(*fallthrough, consumer));
}

TEST(CfgAnalysis, DirectCallEdgeUsesTerminatorOffset) {
  constexpr uint16_t kReturnSreg = 30;

  // The call block starts at 0x00, but the s_call_b64 terminator is at 0x04.
  // CallEdge metadata is consumed later by relocation and must identify the
  // actual call instruction, not the first instruction in the containing block.
  std::vector<uint32_t> words = {
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4), // 0x00.
      build_s_call_b64(kReturnSreg, 1),         // 0x04 -> callee at 0x0c.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4), // 0x08 continuation.
      pack_sop1(0x1d, 0, kReturnSreg),          // 0x0c callee return.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

  auto *caller = block_starting_at(blocks, 0);
  auto *continuation = block_starting_at(blocks, 8);
  auto *callee = block_starting_at(blocks, 12);
  ASSERT_NE(caller, nullptr);
  ASSERT_NE(continuation, nullptr);
  ASSERT_NE(callee, nullptr);

  ASSERT_EQ(caller->call_edges().size(), 1u);
  const BasicBlock::CallEdge &edge = caller->call_edges()[0];
  EXPECT_EQ(edge.kind, BasicBlock::CallEdgeKind::DirectCall);
  EXPECT_EQ(edge.callee, callee);
  EXPECT_EQ(edge.continuation, continuation);
  EXPECT_EQ(edge.source_call_offset, 4u);
}

TEST(CfgAnalysis, DirectCallKillsCarriedPcBuilderFacts) {
  constexpr uint16_t kPcSreg = 8;
  constexpr uint16_t kReturnSreg = 30;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;
  constexpr uint32_t kOriginalGetpcDelta = 28;

  // Without a context-sensitive call/return model, a builder materialized before
  // s_call_b64 must not be reused by a continuation setpc. The callee below
  // writes the same pair before returning, so recovering the continuation setpc
  // would be a stale-value edge.
  std::vector<uint32_t> words = {
      pack_sop1(0x1c, kPcSreg, 0),                         // 0x00: s_getpc_b64.
      pack_sop2(0, kPcSreg, kPcSreg, kLiteralOperand),     // 0x04: s_add_u32.
      kOriginalGetpcDelta,                                 // 0x08: target delta.
      pack_sop2(4, kPcSreg + 1, kPcSreg + 1, kInlineInt0), // 0x0c: s_addc_u32.
      build_s_call_b64(kReturnSreg, 1),                    // 0x10 -> callee at 0x18.
      pack_sop1(0x1d, 0, kPcSreg),                         // 0x14: stale consumer.
      pack_sop2(0, kPcSreg, kPcSreg, kInlineInt0),         // 0x18: callee clobber.
      pack_sop1(0x1d, 0, kReturnSreg),                     // 0x1c: callee return.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),            // 0x20: stale target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

  auto *continuation = block_starting_at(blocks, 20);
  auto *stale_target = block_starting_at(blocks, 32);
  ASSERT_NE(continuation, nullptr);
  ASSERT_NE(stale_target, nullptr);

  EXPECT_TRUE(continuation->static_indirect_call_fixups().empty());
  EXPECT_FALSE(has_successor_start(*continuation, stale_target->start_offset()));
}

TEST(CfgAnalysis, KillPredecessorPreventsRecoveredConsumer) {
  constexpr uint16_t kPcSreg = 8;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;
  constexpr uint32_t kOriginalGetpcDelta = 32;

  // Two paths reach the same setpc consumer:
  //
  //   * the fallthrough path builds a concrete PC target in s[8:9]
  //   * the branch path writes s8 through ordinary scalar code, killing that
  //     pair for this analysis
  //
  // The concrete builder path alone is not enough to recover the consumer. A
  // real unmodeled write reaches the join, so the analysis must fail closed and
  // leave the setpc for the later DBT diagnostic.
  std::vector<uint32_t> words = {
      pack_sopp(5, 5),                                     // 0x00 -> kill path at 0x18.
      pack_sop1(0x1c, kPcSreg, 0),                         // 0x04: s_getpc_b64.
      pack_sop2(0, kPcSreg, kPcSreg, kLiteralOperand),     // 0x08: s_add_u32.
      kOriginalGetpcDelta,                                 // 0x0c: target delta.
      pack_sop2(4, kPcSreg + 1, kPcSreg + 1, kInlineInt0), // 0x10: s_addc_u32.
      build_s_branch(2, ROCJITSU_CODE_ARCH_CDNA4),         // 0x14 -> consumer at 0x20.
      pack_sop2(0, kPcSreg, kPcSreg, kInlineInt0),         // 0x18: unmodeled write.
      build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA4),         // 0x1c -> consumer at 0x20.
      pack_sop1(0x1d, 0, kPcSreg),                         // 0x20: joined consumer.
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),            // 0x24: not a target.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),            // 0x28: builder target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  constexpr std::array<uint64_t, 1> extra_leaders{40};
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4, extra_leaders);

  auto *consumer = block_starting_at(blocks, 32);
  auto *target = block_starting_at(blocks, 40);
  ASSERT_NE(consumer, nullptr);
  ASSERT_NE(target, nullptr);

  EXPECT_TRUE(consumer->static_indirect_call_fixups().empty());
  EXPECT_FALSE(has_successor_start(*consumer, target->start_offset()));
}

TEST(CfgAnalysis, RecoversSignedDeltaTemplateConsumers) {
  constexpr uint16_t kPcSreg = 8;
  constexpr uint16_t kTmpSreg = 12;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;
  constexpr uint32_t kInlineInt4 = 132;
  constexpr uint32_t kSignedDeltaLiteral = 44;

  // This is the split signed-delta template matched by static PC recovery:
  // both the subtract and add halves consume the same getpc-relative target.
  // The matcher deliberately recognizes this complete shape instead of tracking
  // arbitrary temporary SGPR values through the branch.
  std::vector<uint32_t> words = {
      pack_sop1(0x1c, kPcSreg, 0),                          // 0x00: s_getpc_b64.
      pack_sop2(2, kTmpSreg, kLiteralOperand, kInlineInt4), // 0x04: s_add_i32.
      kSignedDeltaLiteral,                                  // 0x08: literal.
      pack_sopc(3, kTmpSreg, kInlineInt0),                  // 0x0c: s_cmp_ge_i32.
      pack_sopp(5, 4),                                      // 0x10 -> add half at 0x24.
      pack_sop1(0x30, kTmpSreg, kTmpSreg),                  // 0x14: s_abs_i32.
      pack_sop2(1, kPcSreg, kPcSreg, kTmpSreg),             // 0x18: s_sub_u32.
      pack_sop2(5, kPcSreg + 1, kPcSreg + 1, kInlineInt0),  // 0x1c: s_subb_u32.
      pack_sop1(0x1d, 0, kPcSreg),                          // 0x20: subtract consumer.
      pack_sop2(0, kPcSreg, kPcSreg, kTmpSreg),             // 0x24: s_add_u32.
      pack_sop2(4, kPcSreg + 1, kPcSreg + 1, kInlineInt0),  // 0x28: s_addc_u32.
      pack_sop1(0x1d, 0, kPcSreg),                          // 0x2c: add consumer.
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),             // 0x30: not a target.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),             // 0x34: shared target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

  auto *sub_consumer = block_starting_at(blocks, 32);
  auto *add_consumer = block_starting_at(blocks, 44);
  auto *target = block_starting_at(blocks, 52);
  ASSERT_NE(sub_consumer, nullptr);
  ASSERT_NE(add_consumer, nullptr);
  ASSERT_NE(target, nullptr);

  ASSERT_EQ(sub_consumer->static_indirect_call_fixups().size(), 1u);
  EXPECT_EQ(sub_consumer->static_indirect_call_fixups()[0].source_call_offset, 32u);
  EXPECT_EQ(sub_consumer->static_indirect_call_fixups()[0].source_target_offset, 52u);
  EXPECT_TRUE(has_successor_start(*sub_consumer, target->start_offset()));

  ASSERT_EQ(add_consumer->static_indirect_call_fixups().size(), 1u);
  EXPECT_EQ(add_consumer->static_indirect_call_fixups()[0].source_call_offset, 44u);
  EXPECT_EQ(add_consumer->static_indirect_call_fixups()[0].source_target_offset, 52u);
  EXPECT_TRUE(has_successor_start(*add_consumer, target->start_offset()));
}

TEST(CfgAnalysis, ReversePostOrderStraightLine) {
  auto blocks =
      build_test_blocks({TestOpcode::DefVgpr0, TestOpcode::UseVgpr0, TestOpcode::UseSgpr4});
  auto scope = block_scope(blocks);
  auto rpo = reverse_post_order(KernelBlockScope(scope));
  ASSERT_EQ(rpo.size(), 1u);
  EXPECT_EQ(blocks[0].get(), rpo[0]);
}

TEST(CfgAnalysis, ReversePostOrderIfElseDiamond) {
  auto blocks = build_test_blocks(
      {TestOpcode::CBranchToElse, TestOpcode::BranchToJoin, TestOpcode::Nop, TestOpcode::End});
  auto scope = block_scope(blocks);
  auto rpo = reverse_post_order(KernelBlockScope(scope));
  ASSERT_EQ(rpo.size(), 4u);
  EXPECT_EQ(rpo[0], blocks[0].get());
  EXPECT_EQ(rpo[1], blocks[1].get());
  EXPECT_EQ(rpo[2], blocks[2].get());
  EXPECT_EQ(rpo[3], blocks[3].get());
}

TEST(CfgAnalysis, ReversePostOrderChangedOrder) {
  auto blocks = build_test_blocks({TestOpcode::BranchToJoin, TestOpcode::BranchToJoin,
                                   TestOpcode::BranchBackToStart, TestOpcode::End});
  auto scope = block_scope(blocks);
  auto rpo = reverse_post_order(KernelBlockScope(scope));
  ASSERT_EQ(rpo.size(), 4u);
  EXPECT_EQ(rpo[0], blocks[0].get());
  EXPECT_EQ(rpo[1], blocks[2].get());
  EXPECT_EQ(rpo[2], blocks[1].get());
  EXPECT_EQ(rpo[3], blocks[3].get());
}

TEST(CfgAnalysis, ReversePostOrderSelfLoop) {
  auto blocks = build_test_blocks({TestOpcode::Nop, TestOpcode::BranchBackToStart});
  auto scope = block_scope(blocks);
  auto rpo = reverse_post_order(KernelBlockScope(scope));
  ASSERT_EQ(rpo.size(), 1u);
  EXPECT_EQ(blocks[0].get(), rpo[0]);
}

TEST(LivenessAnalysis, ExecMaskedVgprDefDoesNotKillInactiveLaneValue) {
  auto blocks = build_test_blocks({TestOpcode::DefVgpr0, TestOpcode::UseVgpr0, TestOpcode::End});
  LivenessAnalysis liveness = analyze_scope(blocks);

  const Instruction &def = *blocks[0]->instructions().begin();
  EXPECT_TRUE(liveness.is_live_before(def, {RegClass::VGPR, 0, 1}));

  auto free_vgpr = liveness.find_free_run(&def, 1);
  ASSERT_TRUE(free_vgpr.has_value());
  EXPECT_NE(*free_vgpr, 0);
}

TEST(LivenessAnalysis, FindsDeadSgprAfterLiveSgpr) {
  auto blocks = build_test_blocks({TestOpcode::UseSgpr4, TestOpcode::End});
  LivenessAnalysis liveness = analyze_scope(blocks);

  const Instruction &use = *blocks[0]->instructions().begin();
  EXPECT_TRUE(liveness.is_live_before(use, {RegClass::SGPR, 4, 1}));
  EXPECT_EQ(liveness.find_free_sgpr(&use, 4), 5);
}

TEST(LivenessAnalysis, FindValidSgprPair) {
  auto blocks = build_test_blocks({TestOpcode::UseSgpr4, TestOpcode::End});
  LivenessAnalysis liveness = analyze_scope(blocks);

  const Instruction &use = *blocks[0]->instructions().begin();
  EXPECT_TRUE(liveness.is_live_before(use, {RegClass::SGPR, 4, 1}));
  EXPECT_EQ(liveness.find_free_sgpr_pair(&use, 4), 6);
}

TEST(LivenessAnalysis, FindSgprPairSkipsStraddle) {
  auto blocks = build_test_blocks({TestOpcode::UseSgpr4, TestOpcode::UseSgpr7, TestOpcode::End});
  LivenessAnalysis liveness = analyze_scope(blocks);

  const Instruction &use = *blocks[0]->instructions().begin();
  EXPECT_EQ(liveness.find_free_sgpr_pair(&use, 4), 8);
}

TEST(LivenessAnalysis, NoSgprPairAvailable) {
  auto blocks = build_test_blocks({TestOpcode::UseSgpr4, TestOpcode::End});
  LivenessAnalysis liveness = analyze_scope(blocks);

  const Instruction &use = *blocks[0]->instructions().begin();
  EXPECT_EQ(liveness.find_free_sgpr_pair(&use, REGISTER_SET_ALLOCATABLE_SGPRS + 10), std::nullopt);
}

TEST(LivenessAnalysis, MinFreeVgprForcesScratchAllocationAboveFloor) {
  auto blocks = build_test_blocks({TestOpcode::UseSgpr4, TestOpcode::End});
  auto scope = block_scope(blocks);

  LivenessAnalysisOptions options;
  options.min_free_vgpr = 4;

  LivenessAnalysis liveness(KernelBlockScope(scope), options);

  const Instruction &use = *blocks[0]->instructions().begin();
  EXPECT_FALSE(liveness.is_live_before(use, {RegClass::VGPR, 0, 4}));
  EXPECT_EQ(liveness.find_free_sgpr(&use, 0), 0);
  EXPECT_EQ(liveness.find_free_run(&use, 1, 0), 4);
  EXPECT_EQ(liveness.find_free_run(&use, 1, 7), 7);
}

TEST(LivenessAnalysis, ReadWriteSameRegisterIsLiveBeforeInstruction) {
  auto blocks = build_test_blocks({TestOpcode::ReadWriteSgpr4, TestOpcode::End});
  LivenessAnalysis liveness = analyze_scope(blocks);

  const Instruction &read_write = *blocks[0]->instructions().begin();
  EXPECT_TRUE(liveness.is_live_before(read_write, {RegClass::SGPR, 4, 1}));
}

TEST(LivenessAnalysis, ReadWriteRegisterStaysLiveOutWhenUsedBySuccessor) {
  std::array<uint64_t, 1> extra_leaders{4};
  auto blocks = build_test_blocks(
      {TestOpcode::ReadWriteSgpr4, TestOpcode::UseSgpr4, TestOpcode::End}, extra_leaders);
  LivenessAnalysis liveness = analyze_scope(blocks);

  ASSERT_EQ(blocks.size(), 2u);
  const Instruction &read_write = *blocks[0]->instructions().begin();
  EXPECT_TRUE(liveness.is_live_before(read_write, {RegClass::SGPR, 4, 1}));
  EXPECT_TRUE(liveness.block_liveness(*blocks[0]).live_out.contains({RegClass::SGPR, 4, 1}));
}

TEST(LivenessAnalysis, ImplicitUseIsLiveBeforeInstruction) {
  auto blocks = build_test_blocks({TestOpcode::ImplicitUseSgpr6Pair, TestOpcode::End});
  LivenessAnalysis liveness = analyze_scope(blocks);

  const Instruction &implicit_use = *blocks[0]->instructions().begin();
  EXPECT_TRUE(liveness.is_live_before(implicit_use, {RegClass::SGPR, 6, 2}));
}

TEST(LivenessAnalysis, PredicatedScalarDefDoesNotKillLiveOutValue) {
  auto blocks =
      build_test_blocks({TestOpcode::PredicatedDefSgpr4, TestOpcode::UseSgpr4, TestOpcode::End});
  LivenessAnalysis liveness = analyze_scope(blocks);

  const Instruction &pred_def = *blocks[0]->instructions().begin();
  EXPECT_TRUE(liveness.is_live_before(pred_def, {RegClass::SGPR, 4, 1}));
}

TEST(LivenessAnalysis, LoopCarriedUseRevisitsBackEdgePredecessor) {
  auto blocks = build_test_blocks({TestOpcode::DefSgpr4, TestOpcode::UseSgpr4,
                                   TestOpcode::CBranchBackToUseSgpr4, TestOpcode::End});
  LivenessAnalysis liveness = analyze_scope(blocks);

  auto *entry = block_starting_at(blocks, 0);
  auto *loop = block_starting_at(blocks, 4);
  ASSERT_NE(entry, nullptr);
  ASSERT_NE(loop, nullptr);
  EXPECT_TRUE(liveness.block_liveness(*entry).live_out.contains({RegClass::SGPR, 4, 1}));
  EXPECT_TRUE(liveness.block_liveness(*loop).live_in.contains({RegClass::SGPR, 4, 1}));
  EXPECT_TRUE(liveness.block_liveness(*loop).live_out.contains({RegClass::SGPR, 4, 1}));
}

TEST(LivenessAnalysis, BranchMeetKeepsValueLiveWhenOneSuccessorPreservesIt) {
  auto blocks = build_test_blocks({TestOpcode::CBranchToElseAfterTwo, TestOpcode::DefSgpr4,
                                   TestOpcode::BranchToJoin, TestOpcode::Nop, TestOpcode::UseSgpr4,
                                   TestOpcode::End});
  LivenessAnalysis liveness = analyze_scope(blocks);

  const Instruction &branch = *blocks[0]->instructions().begin();
  EXPECT_TRUE(liveness.is_live_before(branch, {RegClass::SGPR, 4, 1}));
  EXPECT_TRUE(liveness.block_liveness(*blocks[0]).live_out.contains({RegClass::SGPR, 4, 1}));
}

TEST(LivenessAnalysis, ExplicitBlockSubsetIgnoresOutsideSuccessors) {
  std::array<uint64_t, 1> kernel_entries{8};
  auto blocks = build_test_blocks(
      {TestOpcode::DefVgpr0, TestOpcode::Nop, TestOpcode::UseVgpr0, TestOpcode::End},
      kernel_entries);

  auto *kernel0 = block_starting_at(blocks, 0);
  ASSERT_NE(kernel0, nullptr);
  ASSERT_EQ(kernel0->successors().size(), 1u);
  ASSERT_EQ(kernel0->successors()[0]->start_offset(), 8u);

  const Instruction &def = *kernel0->instructions().begin();
  LivenessAnalysis all_decoded_liveness = analyze_scope(blocks);
  EXPECT_TRUE(all_decoded_liveness.is_live_before(def, {RegClass::VGPR, 0, 1}));

  std::vector<BasicBlock *> kernel_blocks{kernel0};
  LivenessAnalysis kernel_liveness{KernelBlockScope(kernel_blocks)};
  EXPECT_FALSE(kernel_liveness.is_live_before(def, {RegClass::VGPR, 0, 1}));
}

TEST(InstDefUse, DstOnlyVgpr) {
  const TestInstruction test_inst("test_def_v0", {{RegClass::VGPR, 0, 1}});
  InstDefUse idu(test_inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::VGPR, 0, 1}));
}

TEST(InstDefUse, SrcOnlySgpr) {
  const TestInstruction test_inst("test_use_s4", {}, {{RegClass::SGPR, 4, 1}});
  InstDefUse idu(test_inst);
  EXPECT_TRUE(idu.uses.contains({RegClass::SGPR, 4, 1}));
}

TEST(InstDefUse, RWSgpr) {
  const TestInstruction test_inst("test_rw_s4", {{RegClass::SGPR, 4, 1}}, {{RegClass::SGPR, 4, 1}});
  InstDefUse idu(test_inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::SGPR, 4, 1}));
  EXPECT_TRUE(idu.uses.contains({RegClass::SGPR, 4, 1}));
}

TEST(InstDefUse, Predicated) {
  const TestInstruction test_inst("test_pred_def_s4", {{RegClass::SGPR, 4, 1}}, {}, PREDICATED_DEF);
  InstDefUse idu(test_inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::SGPR, 4, 1}));
  EXPECT_TRUE(idu.has_predicated_def);
}

} // namespace
} // namespace rocjitsu

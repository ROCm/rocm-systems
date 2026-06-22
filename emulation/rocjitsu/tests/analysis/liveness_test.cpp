// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/analysis/def_use_chain.h"
#include "rocjitsu/analysis/exec_state.h"
#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/code_object.h"
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
  std::optional<uint64_t> literal64_value() const override { return lit_; }
  std::optional<uint64_t> const_value() const override {
    return const_.has_value() ? const_ : literal64_value();
  }

  // Turn this into a 64-bit literal source operand (no register ref). Models the
  // literal64 path: both literal64_value() and const_value() report it.
  void set_literal(uint64_t v) {
    lit_ = v;
    size_bits_ = 64;
  }

  // Turn this into an inline-constant source operand: const_value() reports it
  // but literal64_value() stays empty (mirrors how inline constants behave).
  void set_inline_const(uint64_t v) {
    const_ = v;
    size_bits_ = 64;
  }

private:
  std::optional<RegisterRef> ref_;
  std::optional<uint64_t> lit_;
  std::optional<uint64_t> const_;
};

class TestInstruction : public Instruction {
public:
  // `literal_src`, when set, appends one literal source operand carrying the
  // given value via literal64_value() — models an instruction's literal64
  // (SIMM) immediate, e.g. the all-ones source of `s_mov_b64 exec, <lit>`.
  TestInstruction(std::string_view mnemonic, std::initializer_list<RegisterRef> defs = {},
                  std::initializer_list<RegisterRef> uses = {}, uint64_t flags = 0,
                  std::optional<int64_t> branch_delta = std::nullopt,
                  std::initializer_list<RegisterRef> implicit_uses = {},
                  std::optional<uint64_t> literal_src = std::nullopt,
                  std::optional<uint64_t> inline_const_src = std::nullopt)
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
    if (literal_src) {
      src_storage_[num_src_].set_literal(*literal_src);
      src_operands_[num_src_] = &src_storage_[num_src_];
      ++num_src_;
    }
    if (inline_const_src) {
      src_storage_[num_src_].set_inline_const(*inline_const_src);
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
  WriteExecFull = 17,
  WriteExecNarrow = 18,
  WriteExecFullInline = 19,
  WriteExecOrAllOnes = 20,
  WriteExecAndSaveexec = 21,
  WriteExecLoHalf = 22,
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
    case TestOpcode::WriteExecFull:
      // s_mov exec, <all-ones literal>: COPY of a single all-ones source.
      return new TestInstruction("test_write_exec_full", {{RegClass::EXEC, 0, 2}}, {}, RESULT_COPY,
                                 std::nullopt, {}, ~0ULL);
    case TestOpcode::WriteExecNarrow:
      // exec <- sgpr (non-constant): writes EXEC but not provably all-ones.
      return new TestInstruction("test_write_exec_narrow", {{RegClass::EXEC, 0, 2}},
                                 {{RegClass::SGPR, 0, 1}}, RESULT_COPY);
    case TestOpcode::WriteExecFullInline:
      // s_mov exec, -1: COPY of an inline-constant all-ones source (reports
      // const_value() but not literal64_value()).
      return new TestInstruction("test_write_exec_full_inline", {{RegClass::EXEC, 0, 2}}, {},
                                 RESULT_COPY, std::nullopt, {}, std::nullopt, ~0ULL);
    case TestOpcode::WriteExecOrAllOnes:
      // s_or_b64 exec, exec, -1: OR of a (non-constant) source and an all-ones
      // inline constant -> all-ones regardless of the other operand.
      return new TestInstruction("test_write_exec_or_allones", {{RegClass::EXEC, 0, 2}},
                                 {{RegClass::SGPR, 0, 1}}, RESULT_OR, std::nullopt, {},
                                 std::nullopt, ~0ULL);
    case TestOpcode::WriteExecAndSaveexec:
      // s_and_saveexec exec, -1: writes EXEC (flag), single all-ones source, but
      // exec = exec & -1 = exec -> NOT all-ones. No RESULT_* -> must stay Unknown.
      return new TestInstruction("test_write_exec_and_saveexec", {{RegClass::SGPR, 0, 2}}, {},
                                 WRITES_EXEC, std::nullopt, {}, std::nullopt, ~0ULL);
    case TestOpcode::WriteExecLoHalf:
      // s_mov_b32 exec_lo, -1: a 32-bit COPY of all-ones into only the low half
      // of EXEC. Full on Wave32; a partial write on Wave64.
      return new TestInstruction("test_write_exec_lo_half", {{RegClass::EXEC, 0, 1}}, {},
                                 RESULT_COPY, std::nullopt, {}, std::nullopt, ~0ULL);
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
  return BasicBlock::build(co, decoder, extra_leaders);
}

bool has_predecessor(const BasicBlock &block, const BasicBlock *pred) {
  return std::ranges::find(block.predecessors(), pred) != block.predecessors().end();
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

// Collect a block's instructions in order for index-based assertions.
std::vector<const Instruction *> insts_of(BasicBlock &block) {
  std::vector<const Instruction *> out;
  for (const auto &inst : block.instructions())
    out.push_back(&inst);
  return out;
}

// Decode real CDNA4 instruction words into a CFG with the production decoder.
std::vector<std::unique_ptr<BasicBlock>> build_cdna4_blocks(std::vector<uint32_t> words) {
  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  return BasicBlock::build(co, *decoder);
}

// Full end-to-end on real decoded instructions: `s_mov_b64 exec, -1` makes EXEC
// provably all-ones, so the following EXEC-masked VGPR def overwrites every lane
// and becomes a real kill — the defined VGPR is dead immediately before it. This
// exercises the whole chain: decode -> to_register_ref (EXEC dest) + const_value
// (-1 -> all-ones) -> ExecMaskAnalysis (Full) -> liveness kill.
TEST(ExecFlagsRealDecode, Cdna4SMovExecAllOnesPromotesVgprDefToKill) {
  // s_mov_b64 exec, -1 ; v_mov_b32 v0, s0 ; v_mov_b32 v2, v0 ; s_endpgm
  auto blocks = build_cdna4_blocks({0xBEFE01C1u, 0x7E000200u, 0x7E040300u, 0xBF810000u});
  ASSERT_FALSE(blocks.empty());
  auto insts = insts_of(*blocks[0]);
  ASSERT_GE(insts.size(), 3u);

  // Sanity-check the decode so an encoding typo fails loudly.
  EXPECT_EQ(insts[0]->mnemonic(), "s_mov_b64");
  const Instruction &def = *insts[1];
  EXPECT_TRUE(def.mnemonic().starts_with("v_mov_b32"));
  ASSERT_NE(def.dst_operand(0), nullptr);
  auto def_ref = def.dst_operand(0)->to_register_ref();
  ASSERT_TRUE(def_ref.has_value());
  EXPECT_EQ(*def_ref, (RegisterRef{RegClass::VGPR, 0, 1}));

  // Proving EXEC=Full from `s_mov exec, -1` needs the s_mov to carry RESULT_COPY,
  // which only exists once the ISA is regenerated with the combinator metadata.
  if (!(insts[0]->flags() & RESULT_COPY))
    GTEST_SKIP() << "s_mov lacks RESULT_COPY; regenerate ISA to enable EXEC-Full tracking";

  // EXEC is provably full at the def, so its vector write is a real kill.
  auto scope = block_scope(blocks);
  ExecMaskAnalysis exec{KernelBlockScope(scope)};
  EXPECT_EQ(exec.before(def), ExecState::Full);

  LivenessAnalysis liveness = analyze_scope(blocks);
  EXPECT_FALSE(liveness.is_live_before(def, {RegClass::VGPR, 0, 1}));
}

// Contrast: without the all-ones EXEC write, EXEC stays Unknown at the def, so
// the same vector def is not promoted to a kill and the VGPR remains live.
TEST(ExecFlagsRealDecode, Cdna4VgprDefStaysLiveWithoutFullExec) {
  // v_mov_b32 v0, s0 ; v_mov_b32 v2, v0 ; s_endpgm
  auto blocks = build_cdna4_blocks({0x7E000200u, 0x7E040300u, 0xBF810000u});
  ASSERT_FALSE(blocks.empty());
  auto insts = insts_of(*blocks[0]);
  ASSERT_GE(insts.size(), 2u);
  const Instruction &def = *insts[0];
  EXPECT_TRUE(def.mnemonic().starts_with("v_mov_b32"));

  auto scope = block_scope(blocks);
  ExecMaskAnalysis exec{KernelBlockScope(scope)};
  EXPECT_EQ(exec.before(def), ExecState::Unknown);

  LivenessAnalysis liveness = analyze_scope(blocks);
  EXPECT_TRUE(liveness.is_live_before(def, {RegClass::VGPR, 0, 1}));
}

TEST(ExecMaskAnalysis, EntryIsUnknownAllOnesIsFullNarrowingIsUnknown) {
  // exec=all-ones; v0=...; exec=narrow; v0=...; end
  auto blocks =
      build_test_blocks({TestOpcode::WriteExecFull, TestOpcode::DefVgpr0,
                         TestOpcode::WriteExecNarrow, TestOpcode::DefVgpr0, TestOpcode::End});
  auto scope = block_scope(blocks);
  ExecMaskAnalysis exec{KernelBlockScope(scope)};

  auto insts = insts_of(*blocks[0]);
  ASSERT_EQ(insts.size(), 5u);
  EXPECT_EQ(exec.before(*insts[0]), ExecState::Unknown); // kernel entry
  EXPECT_EQ(exec.before(*insts[1]), ExecState::Full);    // after exec=all-ones
  EXPECT_EQ(exec.before(*insts[2]), ExecState::Full);    // narrowing not yet applied
  EXPECT_EQ(exec.before(*insts[3]), ExecState::Unknown); // after narrowing write
}

TEST(ExecMaskAnalysis, OrWithAllOnesConstantIsFull) {
  // exec = exec | -1 -> all-ones regardless of the prior EXEC (RESULT_OR with an
  // all-ones source operand).
  auto blocks =
      build_test_blocks({TestOpcode::WriteExecOrAllOnes, TestOpcode::DefVgpr0, TestOpcode::End});
  auto scope = block_scope(blocks);
  ExecMaskAnalysis exec{KernelBlockScope(scope)};

  auto insts = insts_of(*blocks[0]);
  ASSERT_GE(insts.size(), 2u);
  EXPECT_EQ(exec.before(*insts[0]), ExecState::Unknown); // entry
  EXPECT_EQ(exec.before(*insts[1]), ExecState::Full);    // after exec = exec | -1
}

TEST(ExecMaskAnalysis, AndSaveexecWithAllOnesStaysUnknown) {
  // s_and_saveexec exec, -1: exec = exec & -1 = exec, NOT all-ones. The all-ones
  // source must not be mistaken for an all-ones result (regression guard).
  auto blocks =
      build_test_blocks({TestOpcode::WriteExecAndSaveexec, TestOpcode::DefVgpr0, TestOpcode::End});
  auto scope = block_scope(blocks);
  ExecMaskAnalysis exec{KernelBlockScope(scope)};

  auto insts = insts_of(*blocks[0]);
  ASSERT_GE(insts.size(), 2u);
  EXPECT_EQ(exec.before(*insts[1]), ExecState::Unknown); // after and-saveexec
}

TEST(ExecMaskAnalysis, PartialAllOnesWritePreservesButDoesNotEstablishFull) {
  // s_mov_b32 exec_lo, -1 on Wave64 sets only the low half to all-ones.
  // Keeps an already-Full mask Full
  {
    auto blocks = build_test_blocks({TestOpcode::WriteExecFull, TestOpcode::WriteExecLoHalf,
                                     TestOpcode::DefVgpr0, TestOpcode::End});
    auto scope = block_scope(blocks);
    ExecMaskAnalysis exec{KernelBlockScope(scope)}; // default Wave64
    auto insts = insts_of(*blocks[0]);
    ASSERT_GE(insts.size(), 3u);
    EXPECT_EQ(exec.before(*insts[1]), ExecState::Full); // entering the half write
    EXPECT_EQ(exec.before(*insts[2]), ExecState::Full); // half all-ones preserved Full
  }
  // ...Cannot establish Full from Unknown (exec_hi stays unknown).
  {
    auto blocks =
        build_test_blocks({TestOpcode::WriteExecLoHalf, TestOpcode::DefVgpr0, TestOpcode::End});
    auto scope = block_scope(blocks);
    ExecMaskAnalysis exec{KernelBlockScope(scope)}; // default Wave64
    auto insts = insts_of(*blocks[0]);
    ASSERT_GE(insts.size(), 2u);
    EXPECT_EQ(exec.before(*insts[1]), ExecState::Unknown);
  }
}

TEST(ExecMaskAnalysis, Wave32ExecLoWriteCoversFullMask) {
  // On Wave32, exec_lo is the entire EXEC, so an all-ones exec_lo write is Full.
  auto blocks =
      build_test_blocks({TestOpcode::WriteExecLoHalf, TestOpcode::DefVgpr0, TestOpcode::End});
  auto scope = block_scope(blocks);
  ExecMaskAnalysis exec{KernelBlockScope(scope), /*wave_size=*/32};
  auto insts = insts_of(*blocks[0]);
  ASSERT_GE(insts.size(), 2u);
  EXPECT_EQ(exec.before(*insts[1]), ExecState::Full);
}

TEST(LivenessAnalysis, ExecFullPromotesVgprDefToKill) {
  // exec=all-ones; def v0; use v0; end. Under full EXEC the def overwrites every
  // lane, so v0 is dead immediately before the def.
  auto blocks = build_test_blocks(
      {TestOpcode::WriteExecFull, TestOpcode::DefVgpr0, TestOpcode::UseVgpr0, TestOpcode::End});
  LivenessAnalysis liveness = analyze_scope(blocks);

  const Instruction &def = *insts_of(*blocks[0])[1];
  EXPECT_FALSE(liveness.is_live_before(def, {RegClass::VGPR, 0, 1}));
  EXPECT_EQ(liveness.find_free_run(&def, 1), 0u);
}

TEST(LivenessAnalysis, ExecFullViaInlineConstantPromotesVgprDefToKill) {
  // The all-ones source here is an inline constant (`s_mov exec, -1` style):
  // it exposes const_value() but not literal64_value(), so this exercises the
  // const_value() path the analysis relies on.
  auto blocks = build_test_blocks({TestOpcode::WriteExecFullInline, TestOpcode::DefVgpr0,
                                   TestOpcode::UseVgpr0, TestOpcode::End});
  LivenessAnalysis liveness = analyze_scope(blocks);

  const Instruction &def = *insts_of(*blocks[0])[1];
  EXPECT_FALSE(liveness.is_live_before(def, {RegClass::VGPR, 0, 1}));
  EXPECT_EQ(liveness.find_free_run(&def, 1), 0u);
}

TEST(LivenessAnalysis, NarrowingExecWriteDoesNotPromoteVgprDefToKill) {
  // Same shape but EXEC is only narrowed (unknown), so the def stays a
  // non-kill and v0 remains live before it.
  auto blocks = build_test_blocks(
      {TestOpcode::WriteExecNarrow, TestOpcode::DefVgpr0, TestOpcode::UseVgpr0, TestOpcode::End});
  LivenessAnalysis liveness = analyze_scope(blocks);

  const Instruction &def = *insts_of(*blocks[0])[1];
  EXPECT_TRUE(liveness.is_live_before(def, {RegClass::VGPR, 0, 1}));

  auto free_vgpr = liveness.find_free_run(&def, 1);
  ASSERT_TRUE(free_vgpr.has_value());
  EXPECT_NE(*free_vgpr, 0u);
}

TEST(ExecMaskAnalysis, CfgJoinMeetsToUnknownUnlessAllPredecessorsFull) {
  // Branch byte layout (4 bytes/inst); target = block-end + delta:
  //   0:  WriteExecFull
  //   4:  CBranchToElse (delta +4; block ends at 8 -> target 12, skips block@8)
  //   8:  WriteExecNarrow   (fallthrough block)
  //   12: DefVgpr0          (join: reached from the Full branch path and the
  //   16: UseVgpr0           Unknown fallthrough path -> meet is Unknown, so the
  //   20: End                def must not be promoted to a kill)
  auto blocks = build_test_blocks({TestOpcode::WriteExecFull, TestOpcode::CBranchToElse,
                                   TestOpcode::WriteExecNarrow, TestOpcode::DefVgpr0,
                                   TestOpcode::UseVgpr0, TestOpcode::End});
  auto scope = block_scope(blocks);
  ExecMaskAnalysis exec{KernelBlockScope(scope)};

  BasicBlock *join = block_starting_at(blocks, 12);
  ASSERT_NE(join, nullptr);
  const Instruction &def = *join->instructions().begin();
  EXPECT_EQ(exec.before(def), ExecState::Unknown);

  LivenessAnalysis liveness = analyze_scope(blocks);
  EXPECT_TRUE(liveness.is_live_before(def, {RegClass::VGPR, 0, 1}));
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

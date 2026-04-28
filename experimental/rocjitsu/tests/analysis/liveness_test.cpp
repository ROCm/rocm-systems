// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/analysis/def_use_chain.h"
#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/analysis/register_set.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/code_object.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/operand.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/sop1.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/sopp.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace rocjitsu {
namespace {

class DefUseTestInstruction : public Instruction {
public:
  DefUseTestInstruction(Operand &dst, Operand &src) : Instruction("def_use_test", nullptr) {
    dst_operands_[0] = &dst;
    src_operands_[0] = &src;
    num_dst_ = 1;
    num_src_ = 1;
  }

  void implicit_defs(uint8_t wf_size, std::vector<RegisterRef> &defs) const override {
    (void)wf_size;
    defs.push_back({RegClass::SCC, 0, 1});
  }
};

class TestOperand : public Operand {
public:
  TestOperand() = default;
  explicit TestOperand(RegisterRef ref) : Operand(ref.width * 32, ref.index), ref_(ref) {}

  std::optional<RegisterRef> to_register_ref(uint8_t wf_size) const override {
    (void)wf_size;
    return ref_;
  }

private:
  std::optional<RegisterRef> ref_;
};

class TestInstruction : public Instruction {
public:
  TestInstruction(std::string_view mnemonic, std::initializer_list<RegisterRef> defs = {},
                  std::initializer_list<RegisterRef> uses = {}, uint64_t flags = 0,
                  std::optional<int64_t> branch_delta = std::nullopt,
                  std::initializer_list<RegisterRef> implicit_defs = {},
                  std::initializer_list<RegisterRef> implicit_uses = {})
      : Instruction(mnemonic, nullptr), branch_delta_(branch_delta), implicit_defs_(implicit_defs),
        implicit_uses_(implicit_uses) {
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

  void implicit_defs(uint8_t wf_size, std::vector<RegisterRef> &defs) const override {
    (void)wf_size;
    defs.insert(defs.end(), implicit_defs_.begin(), implicit_defs_.end());
  }

  void implicit_uses(uint8_t wf_size, std::vector<RegisterRef> &uses) const override {
    (void)wf_size;
    uses.insert(uses.end(), implicit_uses_.begin(), implicit_uses_.end());
  }

private:
  std::array<TestOperand, 2> dst_storage_{};
  std::array<TestOperand, 4> src_storage_{};
  std::optional<int64_t> branch_delta_;
  std::vector<RegisterRef> implicit_defs_;
  std::vector<RegisterRef> implicit_uses_;
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
  ReadWriteSgpr4 = 8,
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
    case TestOpcode::ReadWriteSgpr4:
      return new TestInstruction("test_rw_s4", {{RegClass::SGPR, 4, 1}}, {{RegClass::SGPR, 4, 1}});
    }
    return new TestInstruction("test_end", {}, {}, PROGRAM_TERMINATOR);
  }
};

std::vector<std::unique_ptr<BasicBlock>> build_test_blocks(std::vector<TestOpcode> ops) {
  std::vector<uint32_t> words;
  words.reserve(ops.size());
  for (TestOpcode op : ops)
    words.push_back(static_cast<uint32_t>(op));

  TestCodeObject co(std::move(words));
  TestDecoder decoder;
  return BasicBlock::build(co, decoder);
}

bool has_predecessor(const BasicBlock &block, const BasicBlock *pred) {
  return std::ranges::find(block.predecessors(), pred) != block.predecessors().end();
}

TEST(RegisterSetAnalysis, KeepsRegisterClassesSeparate) {
  RegisterSet set;
  set.expand({RegClass::SGPR, 4, 1});

  EXPECT_TRUE(set.contains({RegClass::SGPR, 4, 1}));
  EXPECT_FALSE(set.contains({RegClass::VGPR, 4, 1}));
  EXPECT_FALSE(set.contains({RegClass::ACC_VGPR, 4, 1}));
}

TEST(RegisterSetAnalysis, GeneratedCdna4OperandsMapToRegisterRefs) {
  cdna4::Operand sgpr(32, cdna4::OperandType::OPR_SRC, cdna4::OpSelSrc::OPR_SRC_SGPR_MIN + 7);
  cdna4::Operand vgpr(32, cdna4::OperandType::OPR_SRC, cdna4::OpSelSrc::OPR_SRC_VGPR_MIN + 7);
  cdna4::Operand exec64(64, cdna4::OperandType::OPR_SRC, cdna4::OpSelSrc::OPR_SRC_EXEC_LO);
  cdna4::Operand imm32(32, cdna4::OperandType::OPR_SIMM32, 123);

  ASSERT_TRUE(sgpr.to_register_ref(64).has_value());
  EXPECT_EQ(*sgpr.to_register_ref(64), (RegisterRef{RegClass::SGPR, 7, 1}));
  ASSERT_TRUE(vgpr.to_register_ref(64).has_value());
  EXPECT_EQ(*vgpr.to_register_ref(64), (RegisterRef{RegClass::VGPR, 7, 1}));
  ASSERT_TRUE(exec64.to_register_ref(64).has_value());
  EXPECT_EQ(*exec64.to_register_ref(64), (RegisterRef{RegClass::EXEC, 0, 2}));
  EXPECT_FALSE(imm32.to_register_ref(64).has_value());
}

TEST(RegisterSetAnalysis, InstDefUseIncludesExplicitAndImplicitRegisters) {
  cdna4::Operand dst(32, cdna4::OperandType::OPR_SDST, cdna4::OpSelSdst::OPR_SDST_SGPR_MIN);
  cdna4::Operand src(32, cdna4::OperandType::OPR_SSRC, cdna4::OpSelSsrc::OPR_SSRC_SGPR_MIN);
  DefUseTestInstruction inst(dst, src);

  InstDefUse du(inst, 64);
  EXPECT_TRUE(du.defs.contains({RegClass::SGPR, 0, 1}));
  EXPECT_TRUE(du.defs.contains({RegClass::SCC, 0, 1}));
  EXPECT_TRUE(du.uses.contains({RegClass::SGPR, 0, 1}));
}

TEST(RegisterSetAnalysis, SaveexecIncludesImplicitExecAndScc) {
  cdna4::Sop1MachineInst enc{};
  enc.sdst = 0;
  enc.ssrc0 = 2;
  uint32_t word = std::bit_cast<uint32_t>(enc);
  cdna4::SAndSaveexecB64Sop1 inst(reinterpret_cast<const cdna4::MachineInst *>(&word));

  InstDefUse du(inst, 64);
  EXPECT_TRUE(du.defs.contains({RegClass::EXEC, 0, 2}));
  EXPECT_TRUE(du.defs.contains({RegClass::SCC, 0, 1}));
  EXPECT_TRUE(du.uses.contains({RegClass::EXEC, 0, 2}));
}

TEST(RegisterSetAnalysis, CbranchExecUsesWaveSizedExec) {
  cdna4::SoppMachineInst enc{};
  uint32_t word = std::bit_cast<uint32_t>(enc);
  cdna4::SCbranchExeczSopp inst(reinterpret_cast<const cdna4::MachineInst *>(&word));

  InstDefUse du64(inst, 64);
  EXPECT_TRUE(du64.uses.contains({RegClass::EXEC, 0, 2}));

  InstDefUse du32(inst, 32);
  EXPECT_TRUE(du32.uses.contains({RegClass::EXEC, 0, 1}));
  EXPECT_FALSE(du32.uses.contains({RegClass::EXEC, 0, 2}));
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

TEST(LivenessAnalysis, ExecMaskedVgprDefDoesNotKillInactiveLaneValue) {
  auto blocks = build_test_blocks({TestOpcode::DefVgpr0, TestOpcode::UseVgpr0, TestOpcode::End});
  LivenessAnalysis liveness(blocks, 64);

  const Instruction &def = *blocks[0]->instructions().begin();
  EXPECT_TRUE(liveness.is_live_before(def, {RegClass::VGPR, 0, 1}));

  auto free_vgpr = liveness.find_free_run(0, 1);
  ASSERT_TRUE(free_vgpr.has_value());
  EXPECT_NE(*free_vgpr, 0);
}

TEST(LivenessAnalysis, FindsDeadSgprAfterLiveSgpr) {
  auto blocks = build_test_blocks({TestOpcode::UseSgpr4, TestOpcode::End});
  LivenessAnalysis liveness(blocks, 64);

  const Instruction &use = *blocks[0]->instructions().begin();
  EXPECT_TRUE(liveness.is_live_before(use, {RegClass::SGPR, 4, 1}));
  EXPECT_EQ(liveness.find_free_sgpr(0, 4), 5);
}

TEST(LivenessAnalysis, ReadWriteSameRegisterIsLiveBeforeInstruction) {
  auto blocks = build_test_blocks({TestOpcode::ReadWriteSgpr4, TestOpcode::End});
  LivenessAnalysis liveness(blocks, 64);

  const Instruction &read_write = *blocks[0]->instructions().begin();
  EXPECT_TRUE(liveness.is_live_before(read_write, {RegClass::SGPR, 4, 1}));
}

} // namespace
} // namespace rocjitsu

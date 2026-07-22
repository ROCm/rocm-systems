// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/analysis/def_use_chain.h"
#include "rocjitsu/analysis/indirect_branch_discovery.h"
#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/code_object.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/mubuf.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/operand.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/builders.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/vbuffer.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/mubuf.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/isa_traits.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/isa/register_set.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iterator>
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
  // Sub-register operand: same RegisterRef, but a caller-chosen bit width so partial
  // (less-than-32-bit) defs can be exercised.
  TestOperand(RegisterRef ref, int size_bits) : Operand(size_bits, ref.index), ref_(ref) {}

  std::optional<RegisterRef> to_register_ref() const override { return ref_; }

private:
  std::optional<RegisterRef> ref_;
};

class TestInstruction : public Instruction {
public:
  TestInstruction(std::string_view mnemonic, std::initializer_list<RegisterRef> defs = {},
                  std::initializer_list<RegisterRef> uses = {}, uint64_t flags = 0,
                  std::optional<int64_t> branch_delta = std::nullopt,
                  std::initializer_list<RegisterRef> implicit_uses = {}, int def_size_bits = 0)
      : Instruction(mnemonic, nullptr), implicit_uses_(implicit_uses), branch_delta_(branch_delta) {
    size_ = 4;
    flags_ = flags;

    for (RegisterRef ref : defs) {
      // def_size_bits == 0 keeps the default full-lane width; a non-zero value
      // models a partial (sub-32-bit) def of the same register.
      dst_storage_[num_dst_] =
          def_size_bits == 0 ? TestOperand(ref) : TestOperand(ref, def_size_bits);
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
    // Mirror the codegen: a sub-dword (< 32-bit) destination writes only part
    // of its register lane, so the old value survives and the register is also
    // read. Generated instructions surface these partial defs via implicit_uses.
    for (int i = 0; i < num_dst_; ++i) {
      const Operand *op = dst_operands_[i];
      if (op != nullptr && op->size_bits() > 0 && op->size_bits() < REGISTER_GRANULARITY)
        if (auto ref = op->to_register_ref())
          uses.expand(*ref);
    }
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
  PartialDefSgpr4 = 17,
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
    case TestOpcode::PartialDefSgpr4:
      // 16-bit write to s4: defines only part of the lane, so it also reads s4.
      return new TestInstruction("test_partial_def_s4", {{RegClass::SGPR, 4, 1}}, {}, 0,
                                 std::nullopt, {}, /*def_size_bits=*/16);
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

/// @brief View the code object's .text bytes for LivenessAnalysisOptions::text.
/// @details The gfx1250 VGPR_MSB analysis reads S_SETREG_IMM32_B32 literals from
/// this span (at src_loc()+4), so tests exercising immediate MODE writes must
/// supply it.
std::span<const uint8_t> text_span(const CodeObject &co) {
  const Section *text = co.text_sections().front();
  return {reinterpret_cast<const uint8_t *>(text->data()), text->size()};
}

LivenessAnalysis analyze_scope(const std::vector<std::unique_ptr<BasicBlock>> &blocks) {
  auto scope = block_scope(blocks);
  return LivenessAnalysis(KernelBlockScope(scope));
}

// CFG/liveness tests care about decoded register effects, not the physical
// field layout. Keep their compact fixture syntax while routing construction
// through the same generated CDNA3 encoders used by production translation.
uint32_t pack_sopp(uint16_t op, uint16_t simm16) {
  return cdna3::build_sopp(op, {.simm16 = simm16})[0];
}

uint32_t pack_sop1(uint16_t op, uint16_t sdst, uint16_t ssrc0) {
  return cdna3::build_sop1(
      op, {.ssrc0 = static_cast<uint8_t>(ssrc0), .sdst = static_cast<uint8_t>(sdst)})[0];
}

uint32_t pack_sop2(uint16_t op, uint16_t sdst, uint16_t ssrc0, uint16_t ssrc1) {
  return cdna3::build_sop2(op, {.ssrc0 = static_cast<uint8_t>(ssrc0),
                                .ssrc1 = static_cast<uint8_t>(ssrc1),
                                .sdst = static_cast<uint8_t>(sdst)})[0];
}

uint32_t pack_sopc(uint16_t op, uint16_t ssrc0, uint16_t ssrc1) {
  return cdna3::build_sopc(
      op, {.ssrc0 = static_cast<uint8_t>(ssrc0), .ssrc1 = static_cast<uint8_t>(ssrc1)})[0];
}

uint32_t build_s_call_b64(uint16_t sdst, int16_t simm16) {
  return cdna3::build_sopk(cdna3::kSCallB64Sopk, {.simm16 = static_cast<uint16_t>(simm16),
                                                  .sdst = static_cast<uint8_t>(sdst)})[0];
}

TEST(RegisterSetAnalysis, KeepsRegisterClassesSeparate) {
  RegisterSet set;
  set.expand({RegClass::SGPR, 4, 1});

  EXPECT_TRUE(set.contains({RegClass::SGPR, 4, 1}));
  EXPECT_FALSE(set.contains({RegClass::VGPR, 4, 1}));
  EXPECT_FALSE(set.contains({RegClass::ACC_VGPR, 4, 1}));
}

TEST(RegisterSetAnalysis, TracksGfx1250HighBankVectorRegisters) {
  RegisterSet set;
  set.expand({RegClass::VGPR, 768, 2});

  EXPECT_TRUE(set.contains({RegClass::VGPR, 768, 2}));
  EXPECT_EQ(set.size(), 2u);

  set.erase({RegClass::VGPR, 769, 1});
  EXPECT_TRUE(set.contains({RegClass::VGPR, 768, 1}));
  EXPECT_FALSE(set.contains({RegClass::VGPR, 769, 1}));
}

template <typename AtomicInst>
void expect_gfx1250_buffer_cmpswap_def_use(uint8_t return_control, uint8_t payload_width,
                                           uint8_t return_width) {
  gfx1250::VbufferMachineInst raw{};
  raw.vdata = 4;
  raw.th = return_control;
  AtomicInst inst(reinterpret_cast<const gfx1250::MachineInst *>(&raw));

  InstDefUse def_use(inst);
  EXPECT_TRUE(def_use.uses.contains({RegClass::VGPR, 4, payload_width}));
  if (return_width == 0) {
    EXPECT_EQ(def_use.defs.size(), 0u);
  } else {
    EXPECT_TRUE(def_use.defs.contains({RegClass::VGPR, 4, return_width}));
    EXPECT_FALSE(def_use.defs.contains({RegClass::VGPR, 4, payload_width}));
  }
}

TEST(GeneratedInstDefUse, Gfx1250BufferCmpswapReturnUsesElementWidth) {
  constexpr uint8_t kAtomicNoReturn = 0;
  constexpr uint8_t kAtomicReturn = 1;

  expect_gfx1250_buffer_cmpswap_def_use<gfx1250::BufferAtomicCmpswapB32Vbuffer>(kAtomicReturn, 2,
                                                                                1);
  expect_gfx1250_buffer_cmpswap_def_use<gfx1250::BufferAtomicCmpswapB32Vbuffer>(kAtomicNoReturn, 2,
                                                                                0);
  expect_gfx1250_buffer_cmpswap_def_use<gfx1250::BufferAtomicCmpswapB64Vbuffer>(kAtomicReturn, 4,
                                                                                2);
  expect_gfx1250_buffer_cmpswap_def_use<gfx1250::BufferAtomicCmpswapB64Vbuffer>(kAtomicNoReturn, 4,
                                                                                0);
}

TEST(GeneratedInstDefUse, MubufCmpswapReturnUsesElementWidthAndTargetGate) {
  cdna3::MubufMachineInst cdna_raw{};
  cdna_raw.vdata = 4;
  cdna_raw.acc = 1;
  for (uint8_t sc0 : {uint8_t{0}, uint8_t{1}}) {
    cdna_raw.sc0 = sc0;
    cdna3::BufferAtomicCmpswapMubuf inst(reinterpret_cast<const cdna3::MachineInst *>(&cdna_raw));
    InstDefUse def_use(inst);
    EXPECT_TRUE(def_use.uses.contains({RegClass::ACC_VGPR, 4, 2}));
    EXPECT_EQ(def_use.defs.contains({RegClass::ACC_VGPR, 4, 1}), sc0 != 0);
    EXPECT_FALSE(def_use.defs.contains({RegClass::ACC_VGPR, 4, 2}));
  }

  rdna3::MubufMachineInst rdna_raw{};
  rdna_raw.vdata = 8;
  for (uint8_t glc : {uint8_t{0}, uint8_t{1}}) {
    rdna_raw.glc = glc;
    rdna3::BufferAtomicCmpswapB32Mubuf inst(
        reinterpret_cast<const rdna3::MachineInst *>(&rdna_raw));
    InstDefUse def_use(inst);
    EXPECT_TRUE(def_use.uses.contains({RegClass::VGPR, 8, 2}));
    EXPECT_EQ(def_use.defs.contains({RegClass::VGPR, 8, 1}), glc != 0);
    EXPECT_FALSE(def_use.defs.contains({RegClass::VGPR, 8, 2}));
  }
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

TEST(RegisterSetAnalysis, Cdna4WritelaneDestinationIsUseAndDef) {
  constexpr std::array<uint32_t, 2> kWritelaneV141S4Lane2 = {0xd28a008du, 0x00010404u};
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);

  std::unique_ptr<Instruction> inst(decoder->decode(kWritelaneV141S4Lane2.data()));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "v_writelane_b32");

  InstDefUse du(*inst);
  EXPECT_TRUE(du.defs.contains({RegClass::VGPR, 141, 1}));
  EXPECT_TRUE(du.uses.contains({RegClass::VGPR, 141, 1}));
  EXPECT_TRUE(du.uses.contains({RegClass::SGPR, 4, 1}));
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

TEST(CfgAnalysis, IncompleteFactConsumerIsFlaggedIncomplete) {
  constexpr uint16_t kPcSreg = 8;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;

  // Two paths reach one setpc consumer:
  //   * the builder path materializes a concrete PC in s[8:9]
  //   * the bypass path does nothing to the pair, so it arrives at its
  //     unconstrained kernel-entry value
  // The joined fact is therefore INCOMPLETE with one concrete target. Recovery
  // still records that target (for relocation/liveness) but must flag it
  // incomplete, so the translator does not replace the dynamic consumer with a
  // direct window that would redirect the bypass path.
  std::vector<uint32_t> words = {
      pack_sopp(5, 5),                                 // 0x00: cbranch scc0 -> bypass at 0x18.
      pack_sop1(0x1c, kPcSreg, 0),                     // 0x04: s_getpc_b64.
      pack_sop2(0, kPcSreg, kPcSreg, kLiteralOperand), // 0x08: s_add_u32.
      28,                                              // 0x0c: target delta -> 0x08 + 28 = 0x24.
      pack_sop2(4, kPcSreg + 1, kPcSreg + 1, kInlineInt0), // 0x10: s_addc_u32.
      build_s_branch(1, ROCJITSU_CODE_ARCH_CDNA4),         // 0x14 -> consumer at 0x1c.
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4), // 0x18: bypass (leaves pair unconstrained).
      pack_sop1(0x1d, 0, kPcSreg),              // 0x1c: joined consumer setpc.
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4), // 0x20: not a target.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4), // 0x24: builder target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

  auto *consumer = block_starting_at(blocks, 28);
  ASSERT_NE(consumer, nullptr);
  ASSERT_EQ(consumer->static_indirect_call_fixups().size(), 1u);
  const auto &fixup = consumer->static_indirect_call_fixups()[0];
  EXPECT_EQ(fixup.source_target_offset, 36u);
  EXPECT_TRUE(fixup.source_incomplete)
      << "a consumer joined from an unconstrained path must be flagged incomplete";
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

TEST(CfgAnalysis, KeepsDistinctBuildersReachingSameTarget) {
  constexpr uint16_t kPcSreg = 8;
  constexpr uint32_t kLiteralOperand = 255;

  // Two DIFFERENT getpc builders on two paths both build the SAME target (0x28)
  // and reach one consumer. They are distinct lattice values (same target offset,
  // different source_getpc_offset), so the consumer must retain BOTH fixups — the
  // translator rewrites each builder to its own relocated address. Deduplicating
  // on {call,target,sreg} alone would drop one, leaving its stale pre-relocation
  // address.
  std::vector<uint32_t> words = {
      pack_sopp(5, 4),                                 // 0x00: cbranch scc0 -> builder B at 0x14.
      pack_sop1(0x1c, kPcSreg, 0),                     // 0x04: builder A getpc.
      pack_sop2(0, kPcSreg, kPcSreg, kLiteralOperand), // 0x08: s_add_u32 s8, s8, lit.
      0x20u,                                           // 0x0c: delta -> 0x08 + 0x20 = 0x28.
      build_s_branch(3, ROCJITSU_CODE_ARCH_CDNA4),     // 0x10 -> consumer at 0x20.
      pack_sop1(0x1c, kPcSreg, 0),                     // 0x14: builder B getpc.
      pack_sop2(0, kPcSreg, kPcSreg, kLiteralOperand), // 0x18: s_add_u32 s8, s8, lit.
      0x10u,                                           // 0x1c: delta -> 0x18 + 0x10 = 0x28.
      pack_sop1(0x1d, 0, kPcSreg),                     // 0x20: joined consumer setpc.
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),        // 0x24: not a target.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),        // 0x28: shared target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

  auto *consumer = block_starting_at(blocks, 32);
  ASSERT_NE(consumer, nullptr);
  const auto &fixups = consumer->static_indirect_call_fixups();
  // Both builders resolve to target 0x28 but from distinct getpc offsets (0x04,
  // 0x14); both fixups must survive.
  ASSERT_EQ(fixups.size(), 2u);
  for (const auto &fixup : fixups)
    EXPECT_EQ(fixup.source_target_offset, 40u);
  std::vector<uint64_t> getpc_offsets{fixups[0].source_getpc_offset, fixups[1].source_getpc_offset};
  std::ranges::sort(getpc_offsets);
  EXPECT_EQ(getpc_offsets, (std::vector<uint64_t>{4u, 20u}));
}

TEST(CfgAnalysis, Gfx1250RecoversSignedDeltaTemplateWithPrefetch) {
  constexpr uint16_t kPcSreg = 8;
  constexpr uint16_t kTmpSreg = 12;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;
  constexpr uint32_t kInlineInt4 = 132;
  constexpr uint32_t kSignedDeltaLiteral = 68;

  // gfx1250 sometimes emits a prefetch setup move and s_prefetch_inst_pc_rel
  // around the low/carry updates. Neither alters the PC pair, so both
  // signed paths still resolve to the same target at 0x4c.
  std::vector<uint32_t> words = {
      gfx1250::build_sop1(gfx1250::kSGetPcI64Sop1,
                          {.ssrc0 = 0, .sdst = kPcSreg})[0], // 0x00: s_get_pc_i64 s[8:9].
      gfx1250::build_sop2(gfx1250::kSAddCoI32Sop2,
                          {.ssrc0 = kLiteralOperand, .ssrc1 = kInlineInt4, .sdst = kTmpSreg})[0],
      // 0x04: s_add_co_i32.
      kSignedDeltaLiteral, // 0x08: literal.
      gfx1250::build_sopc(gfx1250::kSCmpGeI32Sopc,
                          {.ssrc0 = kTmpSreg, .ssrc1 = kInlineInt0})[0], // 0x0c: s_cmp_ge_i32.
      gfx1250::build_sopp(gfx1250::kSCbranchScc1Sopp, {.simm16 = 7})[0],
      // 0x10 -> add half at 0x30.
      gfx1250::build_sop1(gfx1250::kSMovB32Sop1,
                          {.ssrc0 = 159, .sdst = 14})[0], // 0x14: s_mov_b32 s14, 31.
      0xF404A000u,
      0x1C000000u, // 0x18: s_prefetch_inst_pc_rel.
      gfx1250::build_sop1(gfx1250::kSAbsI32Sop1,
                          {.ssrc0 = kTmpSreg, .sdst = kTmpSreg})[0], // 0x20: s_abs_i32.
      gfx1250::build_sop2(gfx1250::kSSubCoU32Sop2,
                          {.ssrc0 = kPcSreg, .ssrc1 = kTmpSreg, .sdst = kPcSreg})[0],
      // 0x24: s_sub_co_u32.
      gfx1250::build_sop2(gfx1250::kSSubCoCiU32Sop2,
                          {.ssrc0 = kPcSreg + 1, .ssrc1 = kInlineInt0, .sdst = kPcSreg + 1})[0],
      // 0x28: s_sub_co_ci_u32.
      gfx1250::build_sop1(gfx1250::kSSetPcI64Sop1,
                          {.ssrc0 = kPcSreg, .sdst = 0})[0], // 0x2c: s_set_pc_i64.
      gfx1250::build_sop2(gfx1250::kSAddCoU32Sop2,
                          {.ssrc0 = kPcSreg, .ssrc1 = kTmpSreg, .sdst = kPcSreg})[0],
      // 0x30: s_add_co_u32.
      gfx1250::build_sop1(gfx1250::kSMovB32Sop1,
                          {.ssrc0 = 159, .sdst = 14})[0], // 0x34: s_mov_b32 s14, 31.
      0xF404A000u,
      0x1C000000u, // 0x38: s_prefetch_inst_pc_rel.
      gfx1250::build_sop2(gfx1250::kSAddCoCiU32Sop2,
                          {.ssrc0 = kPcSreg + 1, .ssrc1 = kInlineInt0, .sdst = kPcSreg + 1})[0],
      // 0x40: s_add_co_ci_u32.
      gfx1250::build_sop1(gfx1250::kSSetPcI64Sop1,
                          {.ssrc0 = kPcSreg, .sdst = 0})[0], // 0x44: s_set_pc_i64.
      build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250),            // 0x48: not a target.
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250),            // 0x4c: shared target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_GFX1250);

  auto *sub_consumer = block_starting_at(blocks, 44);
  auto *add_consumer = block_starting_at(blocks, 68);
  auto *target = block_starting_at(blocks, 76);
  ASSERT_NE(sub_consumer, nullptr);
  ASSERT_NE(add_consumer, nullptr);
  ASSERT_NE(target, nullptr);

  ASSERT_EQ(sub_consumer->static_indirect_call_fixups().size(), 1u);
  EXPECT_EQ(sub_consumer->static_indirect_call_fixups()[0].source_target_offset, 76u);
  EXPECT_TRUE(has_successor_start(*sub_consumer, target->start_offset()));

  ASSERT_EQ(add_consumer->static_indirect_call_fixups().size(), 1u);
  EXPECT_EQ(add_consumer->static_indirect_call_fixups()[0].source_target_offset, 76u);
  EXPECT_TRUE(has_successor_start(*add_consumer, target->start_offset()));
}

TEST(CfgAnalysis, Gfx1250SignedDeltaRejectsMoveClobberingTemporary) {
  // Same signed-delta template as above, but the "prefetch padding" move on the
  // subtract half writes the temporary (s12) instead of an unrelated register
  // (s14). That move changes the value s_abs_i32/s_sub_co_u32 consume, so recovery
  // must NOT treat it as skippable padding and must NOT prove a static target for
  // the subtract-half setpc. Regression for the temp-clobber gap: an s_mov whose
  // destination equals tmp_sreg was previously accepted as padding.
  constexpr uint16_t kPcSreg = 8;
  constexpr uint16_t kTmpSreg = 12;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;
  constexpr uint32_t kInlineInt4 = 132;
  constexpr uint32_t kSignedDeltaLiteral = 68;

  std::vector<uint32_t> words = {
      gfx1250::build_sop1(gfx1250::kSGetPcI64Sop1,
                          {.ssrc0 = 0, .sdst = kPcSreg})[0], // 0x00: s_get_pc_i64 s[8:9].
      gfx1250::build_sop2(gfx1250::kSAddCoI32Sop2,
                          {.ssrc0 = kLiteralOperand, .ssrc1 = kInlineInt4, .sdst = kTmpSreg})[0],
      // 0x04: s_add_co_i32.
      kSignedDeltaLiteral, // 0x08: literal.
      gfx1250::build_sopc(gfx1250::kSCmpGeI32Sopc,
                          {.ssrc0 = kTmpSreg, .ssrc1 = kInlineInt0})[0], // 0x0c: s_cmp_ge_i32.
      gfx1250::build_sopp(gfx1250::kSCbranchScc1Sopp, {.simm16 = 7})[0],
      // 0x10 -> add half at 0x30.
      gfx1250::build_sop1(
          gfx1250::kSMovB32Sop1,
          {.ssrc0 = 159, .sdst = kTmpSreg})[0], // 0x14: s_mov_b32 s12, 31 (CLOBBER).
      0xF404A000u,
      0x1C000000u, // 0x18: s_prefetch_inst_pc_rel.
      gfx1250::build_sop1(gfx1250::kSAbsI32Sop1,
                          {.ssrc0 = kTmpSreg, .sdst = kTmpSreg})[0], // 0x20: s_abs_i32.
      gfx1250::build_sop2(gfx1250::kSSubCoU32Sop2,
                          {.ssrc0 = kPcSreg, .ssrc1 = kTmpSreg, .sdst = kPcSreg})[0],
      // 0x24: s_sub_co_u32.
      gfx1250::build_sop2(gfx1250::kSSubCoCiU32Sop2,
                          {.ssrc0 = kPcSreg + 1, .ssrc1 = kInlineInt0, .sdst = kPcSreg + 1})[0],
      // 0x28: s_sub_co_ci_u32.
      gfx1250::build_sop1(gfx1250::kSSetPcI64Sop1,
                          {.ssrc0 = kPcSreg, .sdst = 0})[0], // 0x2c: s_set_pc_i64.
      gfx1250::build_sop2(gfx1250::kSAddCoU32Sop2,
                          {.ssrc0 = kPcSreg, .ssrc1 = kTmpSreg, .sdst = kPcSreg})[0],
      // 0x30: s_add_co_u32.
      gfx1250::build_sop1(gfx1250::kSMovB32Sop1,
                          {.ssrc0 = 159, .sdst = 14})[0], // 0x34: s_mov_b32 s14, 31.
      0xF404A000u,
      0x1C000000u, // 0x38: s_prefetch_inst_pc_rel.
      gfx1250::build_sop2(gfx1250::kSAddCoCiU32Sop2,
                          {.ssrc0 = kPcSreg + 1, .ssrc1 = kInlineInt0, .sdst = kPcSreg + 1})[0],
      // 0x40: s_add_co_ci_u32.
      gfx1250::build_sop1(gfx1250::kSSetPcI64Sop1,
                          {.ssrc0 = kPcSreg, .sdst = 0})[0], // 0x44: s_set_pc_i64.
      build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250),            // 0x48: not a target.
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250),            // 0x4c: shared target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_GFX1250);

  // Clobbering the temporary on the subtract half breaks that half of the paired
  // signed-delta template. Because the two halves cross-validate to the same static
  // target, the whole recovery fails closed: NO block proves target 0x4c=76 — versus
  // two proven halves in Gfx1250RecoversSignedDeltaTemplateWithPrefetch.
  size_t resolved_to_target = 0;
  for (const auto &block : blocks) {
    for (const auto &fixup : block->static_indirect_call_fixups()) {
      if (fixup.source_target_offset == 76u)
        ++resolved_to_target;
    }
  }
  EXPECT_EQ(resolved_to_target, 0u);
}

TEST(CfgAnalysis, Gfx1250RecoversPcStashedInVgprLanes) {
  // s[0:1] builds target 0x38, is stashed in v44 lanes 0:1, then restored
  // through v_readlane immediately before swappc. This is the finite static
  // call idiom emitted in RCCL device functions.
  std::vector<uint32_t> words = {
      0xBE804700u, // 0x00: s_get_pc_i64 s[0:1].
      0xA980FE00u,
      52u,
      0u, // 0x04: s_add_nc_u64 ..., lit64(52).
      0xD761002Cu,
      0x02010000u, // 0x10: v_writelane_b32 v44, s0, 0.
      0xD761002Cu,
      0x02010201u, // 0x18: v_writelane_b32 v44, s1, 1.
      0xD7600000u,
      0x0201012Cu, // 0x20: v_readlane_b32 s0, v44, 0.
      0xD7600001u,
      0x0201032Cu,                                // 0x28: v_readlane_b32 s1, v44, 1.
      0xBE9E4900u,                                // 0x30: s_swap_pc_i64 s[30:31], s[0:1].
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250), // 0x34: continuation.
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250), // 0x38: target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_GFX1250);

  auto *consumer = block_starting_at(blocks, 48);
  auto *target = block_starting_at(blocks, 56);
  ASSERT_NE(consumer, nullptr);
  ASSERT_NE(target, nullptr);
  ASSERT_EQ(consumer->static_indirect_call_fixups().size(), 1u);
  EXPECT_EQ(consumer->static_indirect_call_fixups()[0].source_target_offset, 56u);
  EXPECT_TRUE(has_successor_start(*consumer, target->start_offset()));
}

TEST(CfgAnalysis, Gfx1250WideVgprWriteInvalidatesStashedLane) {
  // Same stash idiom as Gfx1250RecoversPcStashedInVgprLanes, but a width-2
  // v_mov_b64 writes v[44:45] between the writelanes and the readlanes. That wide
  // write overwrites the stashed VGPR, so the readlane no longer reconstructs the
  // original PC and recovery must fail closed. A width-one-only invalidation would
  // miss the b64 write and falsely recover a target.
  constexpr auto clobber =
      gfx1250::build_vop3(gfx1250::kVMovB64Vop3, {.vdst = 44, .src0 = 256 + 46});
  std::vector<uint32_t> words = {
      0xBE804700u, // 0x00: s_get_pc_i64 s[0:1].
      0xA980FE00u,
      52u,
      0u, // 0x04: s_add_nc_u64 ..., lit64(52).
      0xD761002Cu,
      0x02010000u, // 0x10: v_writelane_b32 v44, s0, 0.
      0xD761002Cu,
      0x02010201u, // 0x18: v_writelane_b32 v44, s1, 1.
      clobber[0],  // 0x20: v_mov_b64 v[44:45], v[46:47] (wide write over v44).
      clobber[1],
      0xD7600000u,
      0x0201012Cu, // 0x28: v_readlane_b32 s0, v44, 0.
      0xD7600001u,
      0x0201032Cu,                                // 0x30: v_readlane_b32 s1, v44, 1.
      0xBE9E4900u,                                // 0x38: s_swap_pc_i64 s[30:31], s[0:1].
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250), // 0x3c: continuation.
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250), // 0x40: would-be target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_GFX1250);

  size_t total_fixups = 0;
  for (const auto &block : blocks)
    total_fixups += block->static_indirect_call_fixups().size();
  EXPECT_EQ(total_fixups, 0u);
}

TEST(CfgAnalysis, Gfx1250DoesNotCarryLaneStashAcrossBlockBoundary) {
  // Same stash idiom, but an unconditional branch separates the writelane stashes
  // from the readlane/swappc consumer, so they land in different basic blocks. The
  // lane-slot recovery is block-local and must NOT carry the stash across the
  // terminator: a writelane reached on only one path (or jumped over) could
  // otherwise seed a consumer it never dynamically reaches. Recovery must fail
  // closed here (no proven single target), even though the same straight-line form
  // resolves in Gfx1250RecoversPcStashedInVgprLanes.
  std::vector<uint32_t> words = {
      0xBE804700u, // 0x00: s_get_pc_i64 s[0:1].
      0xA980FE00u,
      56u,
      0u, // 0x04: s_add_nc_u64 ..., lit64(56) -> 0x04+56 = 0x3c (would-be target).
      0xD761002Cu,
      0x02010000u, // 0x10: v_writelane_b32 v44, s0, 0.
      0xD761002Cu,
      0x02010201u, // 0x18: v_writelane_b32 v44, s1, 1.
      gfx1250::build_sopp(gfx1250::kSBranchSopp, {.simm16 = 0})[0], // 0x20: s_branch -> 0x24.
      0xD7600000u,
      0x0201012Cu, // 0x24: v_readlane_b32 s0, v44, 0.
      0xD7600001u,
      0x0201032Cu,                                // 0x2c: v_readlane_b32 s1, v44, 1.
      0xBE9E4900u,                                // 0x34: s_swap_pc_i64 s[30:31], s[0:1].
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250), // 0x38: continuation.
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250), // 0x3c: would-be target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_GFX1250);

  size_t total_fixups = 0;
  for (const auto &block : blocks)
    total_fixups += block->static_indirect_call_fixups().size();
  EXPECT_EQ(total_fixups, 0u);
}

TEST(CfgAnalysis, Gfx1250DoesNotRecoverLaneStashWithDifferingRoleBanks) {
  // Same straight-line stash idiom as Gfx1250RecoversPcStashedInVgprLanes, but an
  // s_set_vgpr_msb sets the DST bank to 1 while leaving the SRC0 bank at 0. The
  // v_writelane writes physical v[44+256] (DST bank 1) while the v_readlane reads
  // physical v44 (SRC0 bank 0). Because the roles resolve the same low selector to
  // different physical VGPRs, no value actually flows, and recovery must fail
  // closed rather than key both by the low selector and falsely reconstruct a PC.
  //
  // s_set_vgpr_msb immediate byte is {DST[7:6], SRC2[5:4], SRC1[3:2], SRC0[1:0]};
  // 0x40 selects DST bank 1, all other roles bank 0.
  constexpr auto set_dst_bank_one =
      gfx1250::build_sopp(gfx1250::kSSetVgprMsbSopp, {.simm16 = 0x40});
  std::vector<uint32_t> words = {
      0xBE804700u, // 0x00: s_get_pc_i64 s[0:1].
      0xA980FE00u,
      52u,
      0u,                  // 0x04: s_add_nc_u64 ..., lit64(52).
      set_dst_bank_one[0], // 0x10: s_set_vgpr_msb (DST bank 1, SRC0 bank 0).
      0xD761002Cu,
      0x02010000u, // 0x14: v_writelane_b32 v44, s0, 0 (physical v300 under DST bank 1).
      0xD761002Cu,
      0x02010201u, // 0x1c: v_writelane_b32 v44, s1, 1.
      0xD7600000u,
      0x0201012Cu, // 0x24: v_readlane_b32 s0, v44, 0 (physical v44 under SRC0 bank 0).
      0xD7600001u,
      0x0201032Cu,                                // 0x2c: v_readlane_b32 s1, v44, 1.
      0xBE9E4900u,                                // 0x34: s_swap_pc_i64 s[30:31], s[0:1].
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250), // 0x38: continuation.
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250), // 0x3c: would-be target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_GFX1250);

  size_t total_fixups = 0;
  for (const auto &block : blocks)
    total_fixups += block->static_indirect_call_fixups().size();
  EXPECT_EQ(total_fixups, 0u);
}

TEST(CfgAnalysis, Gfx1250DoesNotInheritBankFromLexicalPredecessor) {
  // An s_set_vgpr_msb in the entry block establishes a bank, then s_branch jumps to
  // a stash block that performs the full getpc/writelane/readlane/swappc idiom with
  // NO local s_set_vgpr_msb. The stash block must NOT inherit the lexically-
  // preceding block's bank (VGPR-MSB is scanned in source order, not CFG order):
  // its bank is unknown at the block boundary, so the writelane cannot record a
  // physical slot and recovery fails closed. Otherwise a false concrete target
  // could be reconstructed.
  //
  // 0x00 s_set_vgpr_msb 0 ; 0x04 s_branch -> 0x0c (skips the 0x08 filler).
  constexpr auto set_bank_zero = gfx1250::build_sopp(gfx1250::kSSetVgprMsbSopp, {.simm16 = 0});
  constexpr auto branch_to_stash = gfx1250::build_sopp(gfx1250::kSBranchSopp, {.simm16 = 1});
  std::vector<uint32_t> words = {
      set_bank_zero[0],                           // 0x00: establish bank 0 (entry block).
      branch_to_stash[0],                         // 0x04: s_branch -> stash block at 0x0c.
      build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250), // 0x08: bypassed filler.
      0xBE804700u,                                // 0x0c: s_get_pc_i64 s[0:1] (stash block).
      0xA980FE00u,
      52u,
      0u,          // 0x10: s_add_nc_u64 ..., lit64(52).
      0xD761002Cu, // 0x1c: v_writelane_b32 v44, s0, 0.
      0x02010000u,
      0xD761002Cu, // 0x24: v_writelane_b32 v44, s1, 1.
      0x02010201u,
      0xD7600000u, // 0x2c: v_readlane_b32 s0, v44, 0.
      0x0201012Cu,
      0xD7600001u, // 0x34: v_readlane_b32 s1, v44, 1.
      0x0201032Cu,
      0xBE9E4900u,                                // 0x3c: s_swap_pc_i64 s[30:31], s[0:1].
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250), // 0x40: continuation.
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250), // 0x44: would-be target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_GFX1250);

  size_t total_fixups = 0;
  for (const auto &block : blocks)
    total_fixups += block->static_indirect_call_fixups().size();
  EXPECT_EQ(total_fixups, 0u);
}

TEST(CfgAnalysis, Gfx1250DoesNotReuseStashFromSkippedFallthroughPredecessor) {
  // Diamond: a conditional branch jumps DIRECTLY into the readlane/swappc consumer
  // block, while the lexical fallthrough path holds the WHOLE getpc/writelane stash.
  //
  //   A: s_cbranch_scc1 -> C                          (fallthrough to B)
  //   B: s_set_vgpr_msb 0 ; getpc/add ; v_writelane   (fallthrough to C)
  //   C: v_readlane s0/s1, v44 ; s_swap_pc_i64        (branch target of A)
  //
  // On the A->C edge the entire stash in B never executes, so the value in v44 is
  // not proven to reach the swappc. C is a branch target — a real block leader — so
  // recovery must reset at C and fail closed, even though B lexically falls through
  // into C. The lane scan is linear: B has no terminator before C, so without a
  // reset at C's leader B's recorded slot leaks into C and falsely recovers a single
  // target. B re-establishes its VGPR-MSB bank locally (s_set_vgpr_msb 0) so the
  // writelane actually records a slot — otherwise the post-cbranch bank-unknown state
  // would mask the stash and the test could not distinguish the two behaviors.
  //
  // s_cbranch_scc1 next_pc = 0x04, target C = 0x28: delta 36 bytes = 9 dwords.
  constexpr auto cbranch_to_consumer =
      gfx1250::build_sopp(gfx1250::kSCbranchScc1Sopp, {.simm16 = 9});
  constexpr auto set_bank_zero = gfx1250::build_sopp(gfx1250::kSSetVgprMsbSopp, {.simm16 = 0});
  std::vector<uint32_t> words = {
      cbranch_to_consumer[0], // 0x00: s_cbranch_scc1 -> C at 0x28 (block A).
      set_bank_zero[0],       // 0x04: s_set_vgpr_msb 0 (block B, fallthrough).
      0xBE804700u,            // 0x08: s_get_pc_i64 s[0:1].
      0xA980FE00u,
      48u,
      0u,          // 0x0c: s_add_nc_u64 ..., lit64(48) -> target 0x3c.
      0xD761002Cu, // 0x18: v_writelane_b32 v44, s0, 0.
      0x02010000u,
      0xD761002Cu, // 0x20: v_writelane_b32 v44, s1, 1.
      0x02010201u,
      0xD7600000u, // 0x28: v_readlane_b32 s0, v44, 0 (block C, branch target).
      0x0201012Cu,
      0xD7600001u, // 0x30: v_readlane_b32 s1, v44, 1.
      0x0201032Cu,
      0xBE9E4900u,                                // 0x38: s_swap_pc_i64 s[30:31], s[0:1].
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250), // 0x3c: would-be target / continuation.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);

  // Assert on the recovery pass output directly, not the post-filtered block
  // fixups: the reviewer's concern is that the lane-stash SCAN must be block-local.
  // A downstream incomplete-predecessor filter can also hide the bogus target, so
  // check discover_indirect_branch_edges() emits NO fixup at all.
  const auto *sec = co.text_sections().front();
  const auto *inst_data = reinterpret_cast<const uint32_t *>(sec->data());
  const size_t inst_data_size = sec->size() / sizeof(uint32_t);
  std::vector<std::unique_ptr<Instruction>> owned;
  for (size_t pc = 0, byte_offset = 0; pc < inst_data_size;) {
    if (inst_data[pc] == 0) { // gfx1250 alignment padding, as in BasicBlock::build.
      ++pc;
      byte_offset += sizeof(uint32_t);
      continue;
    }
    std::unique_ptr<Instruction> inst(decoder->decode(&inst_data[pc], byte_offset));
    ASSERT_NE(inst, nullptr);
    const uint32_t inst_words = static_cast<uint32_t>(inst->size()) / sizeof(uint32_t);
    byte_offset += inst->size();
    pc += inst_words;
    owned.push_back(std::move(inst));
  }
  std::vector<const Instruction *> decoded_insts;
  decoded_insts.reserve(owned.size());
  for (const auto &inst : owned)
    decoded_insts.push_back(inst.get());
  const auto text =
      std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(sec->data()), sec->size());

  const auto fixups = discover_indirect_branch_edges(
      std::span<const Instruction *const>(decoded_insts.data(), decoded_insts.size()), text,
      ROCJITSU_CODE_ARCH_GFX1250);
  EXPECT_TRUE(fixups.empty()) << "block-local recovery must not reuse a skipped-path stash";
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

TEST(LivenessAnalysis, Gfx1250VgprMsbResolvesPhysicalRegisterBank) {
  // src0=2 and dst=2 select physical VGPR bank 2. The VOP1 source encoding
  // still contains v1, but liveness must identify the architectural register
  // as v513 rather than aliasing it with low-bank v1.
  // The upper byte records the previous state for trap recovery and must not
  // affect the active bank selected by the low byte.
  constexpr auto set_vgpr_msb = gfx1250::build_sopp(gfx1250::kSSetVgprMsbSopp, {.simm16 = 0x5a82});
  constexpr auto move = gfx1250::build_vop1(gfx1250::kVMovB32Vop1, {.src0 = 257, .vdst = 0});
  constexpr auto end = gfx1250::build_sopp(gfx1250::kSEndpgmSopp);
  TestCodeObject co({set_vgpr_msb[0], move[0], end[0]});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_EQ(blocks.size(), 1u);
  auto scope = block_scope(blocks);

  LivenessAnalysisOptions options;
  options.arch = ROCJITSU_CODE_ARCH_GFX1250;
  options.entry_block = scope.front();
  options.text = text_span(co);
  LivenessAnalysis liveness(KernelBlockScope(scope), options);

  auto instruction = blocks.front()->instructions().begin();
  ++instruction;
  ASSERT_NE(instruction, blocks.front()->instructions().end());
  EXPECT_EQ(liveness.vgpr_msb_bank_before(*instruction, amdgpu::VgprMsbRole::Src0), 2);
  EXPECT_EQ(liveness.vgpr_msb_bank_before(*instruction, amdgpu::VgprMsbRole::Dst), 2);
  EXPECT_TRUE(liveness.is_live_before(*instruction, {RegClass::VGPR, 513, 1}));
  EXPECT_FALSE(liveness.is_live_before(*instruction, {RegClass::VGPR, 1, 1}));
}

TEST(LivenessAnalysis, Gfx1250DynamicModeWriteConservativelyUsesEveryBank) {
  constexpr uint16_t kModeSrc0Hwreg = 1u | (14u << 6) | (1u << 11);
  constexpr auto setreg =
      gfx1250::build_sopk(gfx1250::kSSetregB32Sopk, {.simm16 = kModeSrc0Hwreg, .sdst = 0});
  constexpr auto move = gfx1250::build_vop1(gfx1250::kVMovB32Vop1, {.src0 = 257, .vdst = 0});
  constexpr auto end = gfx1250::build_sopp(gfx1250::kSEndpgmSopp);
  TestCodeObject co({setreg[0], move[0], end[0]});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_EQ(blocks.size(), 1u);
  auto scope = block_scope(blocks);

  LivenessAnalysisOptions options;
  options.arch = ROCJITSU_CODE_ARCH_GFX1250;
  options.entry_block = scope.front();
  options.text = text_span(co);
  LivenessAnalysis liveness(KernelBlockScope(scope), options);

  auto instruction = blocks.front()->instructions().begin();
  ++instruction;
  ASSERT_NE(instruction, blocks.front()->instructions().end());
  EXPECT_EQ(liveness.vgpr_msb_bank_before(*instruction, amdgpu::VgprMsbRole::Src0), std::nullopt);
  EXPECT_EQ(liveness.vgpr_msb_bank_before(*instruction, amdgpu::VgprMsbRole::Src1), 0);
  EXPECT_EQ(liveness.vgpr_msb_bank_before(*instruction, amdgpu::VgprMsbRole::Src2), 0);
  EXPECT_EQ(liveness.vgpr_msb_bank_before(*instruction, amdgpu::VgprMsbRole::Dst), 0);
  for (uint16_t bank = 0; bank < 4; ++bank)
    EXPECT_TRUE(liveness.is_live_before(
        *instruction, {RegClass::VGPR, static_cast<uint16_t>(1 + bank * 256), 1}));
}

TEST(LivenessAnalysis, Gfx1250FullLiteralModeWriteRecoversKnownBank) {
  constexpr uint16_t kModeSrc0Hwreg = 1u | (14u << 6) | (1u << 11);
  constexpr auto dynamic_setreg =
      gfx1250::build_sopk(gfx1250::kSSetregB32Sopk, {.simm16 = kModeSrc0Hwreg, .sdst = 0});
  constexpr auto literal_setreg =
      gfx1250::build_sopk(gfx1250::kSSetregImm32B32Sopk, {.simm16 = kModeSrc0Hwreg});
  constexpr auto move = gfx1250::build_vop1(gfx1250::kVMovB32Vop1, {.src0 = 257, .vdst = 0});
  constexpr auto end = gfx1250::build_sopp(gfx1250::kSEndpgmSopp);
  TestCodeObject co({dynamic_setreg[0], literal_setreg[0], 2u << 14, move[0], end[0]});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_EQ(blocks.size(), 1u);
  auto scope = block_scope(blocks);

  LivenessAnalysisOptions options;
  options.arch = ROCJITSU_CODE_ARCH_GFX1250;
  options.entry_block = scope.front();
  options.text = text_span(co);
  LivenessAnalysis liveness(KernelBlockScope(scope), options);

  auto instruction = blocks.front()->instructions().begin();
  std::advance(instruction, 2);
  ASSERT_NE(instruction, blocks.front()->instructions().end());
  EXPECT_EQ(instruction.operator*().mnemonic(), "v_mov_b32_e32");
  EXPECT_EQ(liveness.vgpr_msb_bank_before(*instruction, amdgpu::VgprMsbRole::Src0), 2);
  EXPECT_TRUE(liveness.is_live_before(*instruction, {RegClass::VGPR, 513, 1}));
  EXPECT_FALSE(liveness.is_live_before(*instruction, {RegClass::VGPR, 1, 1}));
}

TEST(LivenessAnalysis, Gfx1250TruncatedLiteralModeWriteMarksBanksAmbiguous) {
  // A mode-setting s_setreg_imm32_b32 whose 32-bit literal is not fully present in
  // the .text image (truncated at the end of the section) cannot have its banks
  // recovered. The analysis reads the literal from the text at src_loc()+4; when
  // that word is out of range it must mark the affected banks ambiguous (nullopt)
  // rather than read past the section. Model the truncation by handing the analysis
  // a text span that stops just after the setreg encoding word, before its literal.
  constexpr auto set_bank_two = gfx1250::build_sopp(gfx1250::kSSetVgprMsbSopp, {.simm16 = 2});
  constexpr uint16_t kModeAllBanksHwreg = 1u | (12u << 6) | (7u << 11);
  constexpr auto literal_setreg =
      gfx1250::build_sopk(gfx1250::kSSetregImm32B32Sopk, {.simm16 = kModeAllBanksHwreg});
  constexpr auto move = gfx1250::build_vop1(gfx1250::kVMovB32Vop1, {.src0 = 257, .vdst = 0});
  constexpr auto end = gfx1250::build_sopp(gfx1250::kSEndpgmSopp);
  // Full program (so decode sees a valid literal + terminator), but the analysis is
  // told the text ends right after the setreg encoding word at offset 4 (its
  // literal at offset 8 is out of range).
  TestCodeObject co({set_bank_two[0], literal_setreg[0], 0xe4u << 12, move[0], end[0]});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_EQ(blocks.size(), 1u);
  auto scope = block_scope(blocks);

  LivenessAnalysisOptions options;
  options.arch = ROCJITSU_CODE_ARCH_GFX1250;
  options.entry_block = scope.front();
  // Truncate the text span to 8 bytes: the setreg (at offset 4) has no readable
  // literal at offset 8.
  const auto full = text_span(co);
  options.text = full.subspan(0, 8);
  LivenessAnalysis liveness(KernelBlockScope(scope), options);

  auto instruction = blocks.front()->instructions().begin();
  std::advance(instruction, 2);
  ASSERT_NE(instruction, blocks.front()->instructions().end());
  EXPECT_EQ(instruction.operator*().mnemonic(), "v_mov_b32_e32");
  // Bank 2 was set before the truncated mode write; because the mode write's
  // literal is unreadable, the Src0 bank must be ambiguous, not the pre-write 2.
  EXPECT_EQ(liveness.vgpr_msb_bank_before(*instruction, amdgpu::VgprMsbRole::Src0), std::nullopt);
}

TEST(LivenessAnalysis, Gfx1250PartialLiteralModeWriteUsesUnmaskedVgprFields) {
  constexpr auto set_bank_one = gfx1250::build_sopp(gfx1250::kSSetVgprMsbSopp, {.simm16 = 1});
  constexpr uint16_t kModeSrc0HighBitHwreg = 1u | (15u << 6);
  constexpr auto literal_setreg =
      gfx1250::build_sopk(gfx1250::kSSetregImm32B32Sopk, {.simm16 = kModeSrc0HighBitHwreg});
  constexpr auto move = gfx1250::build_vop1(gfx1250::kVMovB32Vop1, {.src0 = 257, .vdst = 0});
  constexpr auto end = gfx1250::build_sopp(gfx1250::kSEndpgmSopp);
  TestCodeObject co({set_bank_one[0], literal_setreg[0], 3u << 14, move[0], end[0]});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_EQ(blocks.size(), 1u);
  auto scope = block_scope(blocks);

  LivenessAnalysisOptions options;
  options.arch = ROCJITSU_CODE_ARCH_GFX1250;
  options.entry_block = scope.front();
  options.text = text_span(co);
  LivenessAnalysis liveness(KernelBlockScope(scope), options);

  auto instruction = blocks.front()->instructions().begin();
  std::advance(instruction, 2);
  ASSERT_NE(instruction, blocks.front()->instructions().end());
  EXPECT_EQ(instruction.operator*().mnemonic(), "v_mov_b32_e32");
  EXPECT_EQ(liveness.vgpr_msb_bank_before(*instruction, amdgpu::VgprMsbRole::Src0), 3);
  EXPECT_TRUE(liveness.is_live_before(*instruction, {RegClass::VGPR, 769, 1}));
}

TEST(LivenessAnalysis, Gfx1250ImmediateModeWriteRecoversBanksOutsideRequestedSlice) {
  constexpr uint16_t kModeSrc0Hwreg = 1u | (14u << 6) | (1u << 11);
  constexpr auto dynamic_setreg =
      gfx1250::build_sopk(gfx1250::kSSetregB32Sopk, {.simm16 = kModeSrc0Hwreg, .sdst = 0});
  // Request a write to MODE bit zero. The gfx1250 erratum nevertheless updates
  // all VGPR-MSB fields from literal bits [19:12].
  constexpr uint16_t kModeBitZeroHwreg = 1u;
  constexpr auto literal_setreg =
      gfx1250::build_sopk(gfx1250::kSSetregImm32B32Sopk, {.simm16 = kModeBitZeroHwreg});
  constexpr auto move = gfx1250::build_vop1(gfx1250::kVMovB32Vop1, {.src0 = 257, .vdst = 0});
  constexpr auto end = gfx1250::build_sopp(gfx1250::kSEndpgmSopp);
  TestCodeObject co({dynamic_setreg[0], literal_setreg[0], 2u << 14, move[0], end[0]});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_EQ(blocks.size(), 1u);
  auto scope = block_scope(blocks);

  LivenessAnalysisOptions options;
  options.arch = ROCJITSU_CODE_ARCH_GFX1250;
  options.entry_block = scope.front();
  options.text = text_span(co);
  LivenessAnalysis liveness(KernelBlockScope(scope), options);

  auto instruction = blocks.front()->instructions().begin();
  std::advance(instruction, 2);
  ASSERT_NE(instruction, blocks.front()->instructions().end());
  EXPECT_EQ(liveness.vgpr_msb_bank_before(*instruction, amdgpu::VgprMsbRole::Src0), 2);
}

TEST(LivenessAnalysis, Gfx1250LiteralModeWriteTracksEveryRole) {
  constexpr uint16_t kAllVgprMsbFieldsHwreg = 1u | (12u << 6) | (7u << 11);
  constexpr auto literal_setreg =
      gfx1250::build_sopk(gfx1250::kSSetregImm32B32Sopk, {.simm16 = kAllVgprMsbFieldsHwreg});
  // MODE[19:12] is {src2=3, src1=2, src0=1, dst=0}.
  constexpr uint32_t kModeFields = 0xe4u << 12;
  constexpr auto move = gfx1250::build_vop1(gfx1250::kVMovB32Vop1, {.src0 = 257, .vdst = 0});
  constexpr auto end = gfx1250::build_sopp(gfx1250::kSEndpgmSopp);
  TestCodeObject co({literal_setreg[0], kModeFields, move[0], end[0]});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_EQ(blocks.size(), 1u);
  auto scope = block_scope(blocks);

  LivenessAnalysisOptions options;
  options.arch = ROCJITSU_CODE_ARCH_GFX1250;
  options.entry_block = scope.front();
  options.text = text_span(co);
  LivenessAnalysis liveness(KernelBlockScope(scope), options);

  auto instruction = blocks.front()->instructions().begin();
  ++instruction;
  ASSERT_NE(instruction, blocks.front()->instructions().end());
  EXPECT_EQ(liveness.vgpr_msb_bank_before(*instruction, amdgpu::VgprMsbRole::Src0), 1);
  EXPECT_EQ(liveness.vgpr_msb_bank_before(*instruction, amdgpu::VgprMsbRole::Src1), 2);
  EXPECT_EQ(liveness.vgpr_msb_bank_before(*instruction, amdgpu::VgprMsbRole::Src2), 3);
  EXPECT_EQ(liveness.vgpr_msb_bank_before(*instruction, amdgpu::VgprMsbRole::Dst), 0);
}

TEST(LivenessAnalysis, Gfx1250ImmediateNonModeWriteDoesNotChangeBanks) {
  constexpr uint16_t kNonModeHwreg = 2u;
  constexpr auto literal_setreg =
      gfx1250::build_sopk(gfx1250::kSSetregImm32B32Sopk, {.simm16 = kNonModeHwreg});
  constexpr auto move = gfx1250::build_vop1(gfx1250::kVMovB32Vop1, {.src0 = 257, .vdst = 0});
  constexpr auto end = gfx1250::build_sopp(gfx1250::kSEndpgmSopp);
  TestCodeObject co({literal_setreg[0], 0x000ff000u, move[0], end[0]});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_EQ(blocks.size(), 1u);
  auto scope = block_scope(blocks);

  LivenessAnalysisOptions options;
  options.arch = ROCJITSU_CODE_ARCH_GFX1250;
  options.entry_block = scope.front();
  options.text = text_span(co);
  LivenessAnalysis liveness(KernelBlockScope(scope), options);

  auto instruction = blocks.front()->instructions().begin();
  ++instruction;
  ASSERT_NE(instruction, blocks.front()->instructions().end());
  EXPECT_EQ(liveness.vgpr_msb_bank_before(*instruction, amdgpu::VgprMsbRole::Src0), 0);
  EXPECT_EQ(liveness.vgpr_msb_bank_before(*instruction, amdgpu::VgprMsbRole::Src1), 0);
  EXPECT_EQ(liveness.vgpr_msb_bank_before(*instruction, amdgpu::VgprMsbRole::Src2), 0);
  EXPECT_EQ(liveness.vgpr_msb_bank_before(*instruction, amdgpu::VgprMsbRole::Dst), 0);
}

TEST(LivenessAnalysis, Gfx1250VgprMsbCfgJoinRequiresPredecessorsToAgree) {
  constexpr auto branch_to_else = gfx1250::build_sopp(gfx1250::kSCbranchScc0Sopp, {.simm16 = 2});
  constexpr auto set_bank_two = gfx1250::build_sopp(gfx1250::kSSetVgprMsbSopp, {.simm16 = 0x82});
  constexpr auto branch_to_join = gfx1250::build_sopp(gfx1250::kSBranchSopp, {.simm16 = 1});
  constexpr auto set_bank_zero = gfx1250::build_sopp(gfx1250::kSSetVgprMsbSopp);
  constexpr auto move = gfx1250::build_vop1(gfx1250::kVMovB32Vop1, {.src0 = 257, .vdst = 0});
  constexpr auto end = gfx1250::build_sopp(gfx1250::kSEndpgmSopp);
  TestCodeObject co(
      {branch_to_else[0], set_bank_two[0], branch_to_join[0], set_bank_zero[0], move[0], end[0]});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_GFX1250);
  auto scope = block_scope(blocks);
  BasicBlock *join = block_starting_at(blocks, 16);
  ASSERT_NE(join, nullptr);

  LivenessAnalysisOptions options;
  options.arch = ROCJITSU_CODE_ARCH_GFX1250;
  options.entry_block = scope.front();
  options.text = text_span(co);
  LivenessAnalysis liveness(KernelBlockScope(scope), options);

  const Instruction &joined_move = *join->instructions().begin();
  EXPECT_EQ(liveness.vgpr_msb_bank_before(joined_move, amdgpu::VgprMsbRole::Src0), std::nullopt);
  for (uint16_t bank = 0; bank < 4; ++bank)
    EXPECT_TRUE(liveness.is_live_before(
        joined_move, {RegClass::VGPR, static_cast<uint16_t>(1 + bank * 256), 1}));
}

TEST(LivenessAnalysis, Gfx1250VgprMsbCfgJoinPreservesAgreeingBank) {
  constexpr auto branch_to_else = gfx1250::build_sopp(gfx1250::kSCbranchScc0Sopp, {.simm16 = 2});
  constexpr auto set_bank_two = gfx1250::build_sopp(gfx1250::kSSetVgprMsbSopp, {.simm16 = 2});
  constexpr auto branch_to_join = gfx1250::build_sopp(gfx1250::kSBranchSopp, {.simm16 = 1});
  constexpr auto move = gfx1250::build_vop1(gfx1250::kVMovB32Vop1, {.src0 = 257, .vdst = 0});
  constexpr auto end = gfx1250::build_sopp(gfx1250::kSEndpgmSopp);
  TestCodeObject co(
      {branch_to_else[0], set_bank_two[0], branch_to_join[0], set_bank_two[0], move[0], end[0]});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_GFX1250);
  auto scope = block_scope(blocks);
  BasicBlock *join = block_starting_at(blocks, 16);
  ASSERT_NE(join, nullptr);

  LivenessAnalysisOptions options;
  options.arch = ROCJITSU_CODE_ARCH_GFX1250;
  options.entry_block = scope.front();
  options.text = text_span(co);
  LivenessAnalysis liveness(KernelBlockScope(scope), options);

  const Instruction &joined_move = *join->instructions().begin();
  EXPECT_EQ(liveness.vgpr_msb_bank_before(joined_move, amdgpu::VgprMsbRole::Src0), 2);
  EXPECT_TRUE(liveness.is_live_before(joined_move, {RegClass::VGPR, 513, 1}));
  EXPECT_FALSE(liveness.is_live_before(joined_move, {RegClass::VGPR, 1, 1}));
}

TEST(LivenessAnalysis, Gfx1250VgprMsbJoinExcludesUnreachablePredecessor) {
  // The entry unconditionally branches over an unreachable block that sets bank 0,
  // landing on a block that sets bank 2 and falls through to the join. Only the
  // reachable predecessor (bank 2) may contribute to the join; the unreachable
  // bank-0 block must NOT drag the joined bank to ambiguous (nullopt). This pins
  // that the fixed point excludes unreachable predecessors rather than meeting
  // every structural in-edge.
  //
  // Layout (each op is one dword):
  //   0x00 s_branch +1        -> skips the unreachable block, targets 0x08
  //   0x04 s_set_vgpr_msb 0   (UNREACHABLE: no edge targets it)
  //   0x08 s_set_vgpr_msb 2   (reachable target; falls through to join)
  //   0x0c v_mov (join)       reads v1 under the proven bank
  //   0x10 s_endpgm
  constexpr auto branch_over = gfx1250::build_sopp(gfx1250::kSBranchSopp, {.simm16 = 1});
  constexpr auto set_bank_zero = gfx1250::build_sopp(gfx1250::kSSetVgprMsbSopp);
  constexpr auto set_bank_two = gfx1250::build_sopp(gfx1250::kSSetVgprMsbSopp, {.simm16 = 2});
  constexpr auto move = gfx1250::build_vop1(gfx1250::kVMovB32Vop1, {.src0 = 257, .vdst = 0});
  constexpr auto end = gfx1250::build_sopp(gfx1250::kSEndpgmSopp);
  TestCodeObject co({branch_over[0], set_bank_zero[0], set_bank_two[0], move[0], end[0]});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_GFX1250);
  auto scope = block_scope(blocks);

  LivenessAnalysisOptions options;
  options.arch = ROCJITSU_CODE_ARCH_GFX1250;
  options.entry_block = scope.front();
  options.text = text_span(co);
  LivenessAnalysis liveness(KernelBlockScope(scope), options);

  const Instruction *joined_move = nullptr;
  for (const auto &block : blocks) {
    for (const Instruction &inst : block->instructions()) {
      if (inst.mnemonic() == "v_mov_b32_e32")
        joined_move = &inst;
    }
  }
  ASSERT_NE(joined_move, nullptr);
  EXPECT_EQ(liveness.vgpr_msb_bank_before(*joined_move, amdgpu::VgprMsbRole::Src0), 2);
  EXPECT_TRUE(liveness.is_live_before(*joined_move, {RegClass::VGPR, 513, 1}));
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

TEST(LivenessAnalysis, FreeVgprAllocationHonorsDestinationLimit) {
  auto blocks = build_test_blocks({TestOpcode::UseSgpr4, TestOpcode::End});
  auto scope = block_scope(blocks);
  const Instruction &use = *blocks[0]->instructions().begin();

  LivenessAnalysisOptions limited_options;
  limited_options.min_free_vgpr = 256;
  LivenessAnalysis limited(KernelBlockScope(scope), limited_options);
  EXPECT_EQ(limited.find_free_run(&use, 1), std::nullopt);

  LivenessAnalysisOptions gfx1250_options;
  gfx1250_options.min_free_vgpr = 256;
  gfx1250_options.max_free_vgpr = 1024;
  LivenessAnalysis gfx1250(KernelBlockScope(scope), gfx1250_options);
  EXPECT_EQ(gfx1250.find_free_run(&use, 1), 256);
}

TEST(LivenessAnalysis, FindFreeRunHonorsBaseAlignment) {
  auto blocks = build_test_blocks({TestOpcode::UseSgpr4, TestOpcode::End});
  auto scope = block_scope(blocks);

  LivenessAnalysisOptions options;
  options.min_free_vgpr = 93;

  LivenessAnalysis liveness(KernelBlockScope(scope), options);

  const Instruction &use = *blocks[0]->instructions().begin();
  EXPECT_EQ(liveness.find_free_run(&use, 4, 0, 2), 94);
  EXPECT_EQ(liveness.find_free_run(&use, 4, 94, 4), 96);
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

TEST(LivenessAnalysis, PartialDefKeepsRegisterLiveBeforeInstruction) {
  auto blocks = build_test_blocks({TestOpcode::PartialDefSgpr4, TestOpcode::End});
  LivenessAnalysis liveness = analyze_scope(blocks);

  const Instruction &partial_def = *blocks[0]->instructions().begin();
  EXPECT_TRUE(liveness.is_live_before(partial_def, {RegClass::SGPR, 4, 1}));
}

TEST(LivenessAnalysis, PartialDefRegisterStaysLiveOutWhenUsedBySuccessor) {
  std::array<uint64_t, 1> extra_leaders{4};
  auto blocks = build_test_blocks(
      {TestOpcode::PartialDefSgpr4, TestOpcode::UseSgpr4, TestOpcode::End}, extra_leaders);
  LivenessAnalysis liveness = analyze_scope(blocks);

  ASSERT_EQ(blocks.size(), 2u);
  const Instruction &partial_def = *blocks[0]->instructions().begin();
  EXPECT_TRUE(liveness.is_live_before(partial_def, {RegClass::SGPR, 4, 1}));
  EXPECT_TRUE(liveness.block_liveness(*blocks[0]).live_out.contains({RegClass::SGPR, 4, 1}));
}

TEST(LivenessAnalysis, FullWidthDefKillsRegisterBeforeInstruction) {
  auto blocks = build_test_blocks({TestOpcode::DefSgpr4, TestOpcode::End});
  LivenessAnalysis liveness = analyze_scope(blocks);

  const Instruction &def = *blocks[0]->instructions().begin();
  EXPECT_FALSE(liveness.is_live_before(def, {RegClass::SGPR, 4, 1}));
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

TEST(InstDefUse, PartialDefIsAlsoUse) {
  const TestInstruction test_inst("test_partial_def_s4", {{RegClass::SGPR, 4, 1}}, {}, 0,
                                  std::nullopt, {}, /*def_size_bits=*/16);
  InstDefUse idu(test_inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::SGPR, 4, 1}));
  EXPECT_TRUE(idu.uses.contains({RegClass::SGPR, 4, 1}));
}

TEST(InstDefUse, FullWidthDefIsNotUse) {
  const TestInstruction test_inst("test_def_s4", {{RegClass::SGPR, 4, 1}}, {}, 0, std::nullopt, {},
                                  /*def_size_bits=*/32);
  InstDefUse idu(test_inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::SGPR, 4, 1}));
  EXPECT_FALSE(idu.uses.contains({RegClass::SGPR, 4, 1}));
}

TEST(InstDefUse, Predicated) {
  const TestInstruction test_inst("test_pred_def_s4", {{RegClass::SGPR, 4, 1}}, {}, PREDICATED_DEF);
  InstDefUse idu(test_inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::SGPR, 4, 1}));
  EXPECT_TRUE(idu.has_predicated_def);
}

// --- Generated VOP1 SDWA/DPP destination-preserve reads (real decode) ---
//
// SDWA dst_unused:PRESERVE and a partial DPP row/bank mask both keep the old
// vdst value, so the decoded instruction must report vdst as an implicit use.
// InstDefUse is the per-instruction def/use set LivenessAnalysis consumes (it
// calls Instruction::implicit_uses), so a use surfacing here is exactly what
// reaches liveness -- see ImplicitUseIsLiveBeforeInstruction for that step.
//
// CDNA4 VOP1 word0: encoding[31:25]=0x3F, vdst[24:17], op[16:9]=1 (v_mov_b32),
// src0[8:0]=marker (250=SRC_DPP, 249=SRC_SDWA).
constexpr uint32_t kVop1MovWord0Dpp = (0x3Fu << 25) | (5u << 17) | (1u << 9) | 250u;
constexpr uint32_t kVop1MovWord0Sdwa = (0x3Fu << 25) | (5u << 17) | (1u << 9) | 249u;

std::unique_ptr<Instruction> decode_cdna4(const std::array<uint32_t, 2> &words) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  return std::unique_ptr<Instruction>(decoder ? decoder->decode(words.data()) : nullptr);
}

std::unique_ptr<Instruction> decode_gfx1250(const std::array<uint32_t, 2> &words) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  return std::unique_ptr<Instruction>(decoder ? decoder->decode(words.data()) : nullptr);
}

TEST(GeneratedInstDefUse, Gfx1250Vop3CompareDefinesOneSgpr) {
  // v_cmp_eq_u32_e64 s53, 32, v4. gfx1250 is wave32-only, so the comparison
  // mask occupies s53 and must not make liveness treat the adjacent s54 as
  // clobbered.
  auto inst = decode_gfx1250({0xD44A0035u, 0x020208A0u});
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(inst->mnemonic(), "v_cmp_eq_u32");

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::SGPR, 53, 1}));
  EXPECT_FALSE(idu.defs.contains({RegClass::SGPR, 54, 1}));
}

// DPP word1 fields (CDNA4): vsrc0[7:0], dpp_ctrl[16:8], bound_ctrl[19],
// bank_mask[27:24], row_mask[31:28]. With full masks, whether vdst is
// preserved depends on bound_ctrl and whether dpp_ctrl crosses a row/wave
// edge: bound_ctrl=0 + an edge-crossing ctrl leaves OOB lanes unwritten (reads
// vdst); bound_ctrl=1 writes a zero source instead (full write); a ctrl that
// never goes OOB is a full write regardless of bound_ctrl.
constexpr uint32_t kDppFullMasks = (0xFu << 28) | (0xFu << 24);
constexpr uint32_t kDppBoundCtrl = (1u << 19);
constexpr uint32_t kDppCtrlRowShr1 = 0x111u << 8; // row_shr:1 -- crosses the row edge
constexpr uint32_t kDppCtrlRowRor1 = 0x121u << 8; // row_ror:1 -- rotates within the row

TEST(GeneratedInstDefUse, DppPartialRowMaskReadsDestination) {
  // DPP word1: row_mask[31:28]=0x7 (partial), bank_mask[27:24]=0xF, vsrc0[7:0]=2.
  auto inst = decode_cdna4({kVop1MovWord0Dpp, (0x7u << 28) | (0xFu << 24) | 2u});
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()).substr(0, 9), "v_mov_b32");

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::VGPR, 5, 1}));
  EXPECT_TRUE(idu.uses.contains({RegClass::VGPR, 5, 1}));
}

TEST(GeneratedInstDefUse, DppFullRowMaskDoesNotReadDestination) {
  // DPP word1: row_mask=0xF, bank_mask=0xF (full), dpp_ctrl=0 (quad_perm, never
  // OOB) -> every lane written, no vdst read even with bound_ctrl=0.
  auto inst = decode_cdna4({kVop1MovWord0Dpp, (0xFu << 28) | (0xFu << 24) | 2u});
  ASSERT_NE(inst, nullptr);

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::VGPR, 5, 1}));
  EXPECT_FALSE(idu.uses.contains({RegClass::VGPR, 5, 1}));
}

TEST(GeneratedInstDefUse, SdwaPreserveReadsDestination) {
  // SDWA word1: vsrc0[7:0]=2, dst_sel[10:8]=0 (BYTE_0, != DWORD),
  // dst_unused[12:11]=2 (UNUSED_PRESERVE).
  auto inst = decode_cdna4({kVop1MovWord0Sdwa, (2u << 11) | (0u << 8) | 2u});
  ASSERT_NE(inst, nullptr);

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.uses.contains({RegClass::VGPR, 5, 1}));
}

TEST(GeneratedInstDefUse, SdwaPadDoesNotReadDestination) {
  // SDWA word1: dst_sel[10:8]=0, dst_unused[12:11]=0 (UNUSED_PAD) -> no read.
  auto inst = decode_cdna4({kVop1MovWord0Sdwa, (0u << 11) | (0u << 8) | 2u});
  ASSERT_NE(inst, nullptr);

  InstDefUse idu(*inst);
  EXPECT_FALSE(idu.uses.contains({RegClass::VGPR, 5, 1}));
}

TEST(GeneratedInstDefUse, DppBoundCtrlZeroEdgeCrossingReadsDestination) {
  // Full masks, bound_ctrl=0, row_shr:1 -> row-edge lanes read OOB and are left
  // unwritten, preserving vdst.
  auto inst = decode_cdna4({kVop1MovWord0Dpp, kDppFullMasks | kDppCtrlRowShr1 | 2u});
  ASSERT_NE(inst, nullptr);

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::VGPR, 5, 1}));
  EXPECT_TRUE(idu.uses.contains({RegClass::VGPR, 5, 1}));
}

TEST(GeneratedInstDefUse, DppBoundCtrlOneEdgeCrossingDoesNotReadDestination) {
  // Full masks, row_shr:1 but bound_ctrl=1 -> OOB lanes read a zero source and
  // are still written, so every lane is defined and vdst is not read.
  auto inst =
      decode_cdna4({kVop1MovWord0Dpp, kDppFullMasks | kDppBoundCtrl | kDppCtrlRowShr1 | 2u});
  ASSERT_NE(inst, nullptr);

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::VGPR, 5, 1}));
  EXPECT_FALSE(idu.uses.contains({RegClass::VGPR, 5, 1}));
}

TEST(GeneratedInstDefUse, DppBoundCtrlZeroRotateDoesNotReadDestination) {
  // Full masks, bound_ctrl=0, row_ror:1 -> a rotate never goes OOB, so every
  // lane is written and vdst is not read despite bound_ctrl=0.
  auto inst = decode_cdna4({kVop1MovWord0Dpp, kDppFullMasks | kDppCtrlRowRor1 | 2u});
  ASSERT_NE(inst, nullptr);

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::VGPR, 5, 1}));
  EXPECT_FALSE(idu.uses.contains({RegClass::VGPR, 5, 1}));
}

TEST(GeneratedInstDefUse, Vop1DppPartialMaskReadsFullWidthDestination) {
  // v_rcp_f64_e32 writes a VGPR pair (v[6:7]). A partial DPP row mask preserves
  // the whole 64-bit destination, so the implicit use must match the width-2
  // def -- not just the low dword.
  // CDNA4 VOP1 word0: encoding[31:25]=0x3F, vdst[24:17]=6, op[16:9]=37
  // (v_rcp_f64), src0[8:0]=250 (SRC_DPP).
  constexpr uint32_t kVop1RcpF64Word0Dpp = (0x3Fu << 25) | (6u << 17) | (37u << 9) | 250u;
  auto inst = decode_cdna4({kVop1RcpF64Word0Dpp, (0x7u << 28) | (0xFu << 24) | 2u});
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()).substr(0, 9), "v_rcp_f64");

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::VGPR, 6, 2}));
  EXPECT_TRUE(idu.uses.contains({RegClass::VGPR, 6, 2}));
}

// --- Generated VOP2 SDWA/DPP destination-preserve reads (real decode) ---
//
// VOP2 shares VOP1's destination-preserve rules: SDWA dst_unused:PRESERVE and a
// partial DPP row/bank mask both keep the old vdst value, so the decoded
// instruction must report vdst as an implicit use (see Vop2::implicit_uses,
// which mirrors Vop1::implicit_uses). These cases mimic the VOP1 tests above but
// exercise the VOP2 encoding path.
//
// CDNA4 VOP2 word0: encoding[31]=0, op[30:25]=1 (v_add_f32), vdst[24:17],
// vsrc1[16:9], src0[8:0]=marker (250=SRC_DPP, 249=SRC_SDWA). The DPP/SDWA word1
// layouts are identical to VOP1, so the second-word bit fields are reused.
constexpr uint32_t kVop2AddWord0Dpp = (0u << 31) | (1u << 25) | (5u << 17) | (3u << 9) | 250u;
constexpr uint32_t kVop2AddWord0Sdwa = (0u << 31) | (1u << 25) | (5u << 17) | (3u << 9) | 249u;

TEST(GeneratedInstDefUse, Vop2DppPartialRowMaskReadsDestination) {
  // DPP word1: row_mask[31:28]=0x7 (partial), bank_mask[27:24]=0xF, vsrc0[7:0]=2.
  auto inst = decode_cdna4({kVop2AddWord0Dpp, (0x7u << 28) | (0xFu << 24) | 2u});
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()).substr(0, 9), "v_add_f32");

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::VGPR, 5, 1}));
  EXPECT_TRUE(idu.uses.contains({RegClass::VGPR, 5, 1}));
}

TEST(GeneratedInstDefUse, Vop2DppFullRowMaskDoesNotReadDestination) {
  // DPP word1: row_mask=0xF, bank_mask=0xF (full), dpp_ctrl=0 (quad_perm, never
  // OOB) -> every lane written, no vdst read even with bound_ctrl=0.
  auto inst = decode_cdna4({kVop2AddWord0Dpp, (0xFu << 28) | (0xFu << 24) | 2u});
  ASSERT_NE(inst, nullptr);

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::VGPR, 5, 1}));
  EXPECT_FALSE(idu.uses.contains({RegClass::VGPR, 5, 1}));
}

TEST(GeneratedInstDefUse, Vop2SdwaPreserveReadsDestination) {
  // SDWA word1: vsrc0[7:0]=2, dst_sel[10:8]=0 (BYTE_0, != DWORD),
  // dst_unused[12:11]=2 (UNUSED_PRESERVE).
  auto inst = decode_cdna4({kVop2AddWord0Sdwa, (2u << 11) | (0u << 8) | 2u});
  ASSERT_NE(inst, nullptr);

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.uses.contains({RegClass::VGPR, 5, 1}));
}

TEST(GeneratedInstDefUse, Vop2SdwaPadDoesNotReadDestination) {
  // SDWA word1: dst_sel[10:8]=0, dst_unused[12:11]=0 (UNUSED_PAD) -> no read.
  auto inst = decode_cdna4({kVop2AddWord0Sdwa, (0u << 11) | (0u << 8) | 2u});
  ASSERT_NE(inst, nullptr);

  InstDefUse idu(*inst);
  EXPECT_FALSE(idu.uses.contains({RegClass::VGPR, 5, 1}));
}

TEST(GeneratedInstDefUse, Vop2DppBoundCtrlZeroEdgeCrossingReadsDestination) {
  // Full masks, bound_ctrl=0, row_shr:1 -> row-edge lanes read OOB and are left
  // unwritten, preserving vdst (mirrors the VOP1 case on the VOP2 path).
  auto inst = decode_cdna4({kVop2AddWord0Dpp, kDppFullMasks | kDppCtrlRowShr1 | 2u});
  ASSERT_NE(inst, nullptr);

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::VGPR, 5, 1}));
  EXPECT_TRUE(idu.uses.contains({RegClass::VGPR, 5, 1}));
}

TEST(GeneratedInstDefUse, Vop2DppBoundCtrlOneEdgeCrossingDoesNotReadDestination) {
  // Full masks, row_shr:1 but bound_ctrl=1 -> OOB lanes read zero and are still
  // written, so every lane is defined and vdst is not read.
  auto inst =
      decode_cdna4({kVop2AddWord0Dpp, kDppFullMasks | kDppBoundCtrl | kDppCtrlRowShr1 | 2u});
  ASSERT_NE(inst, nullptr);

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::VGPR, 5, 1}));
  EXPECT_FALSE(idu.uses.contains({RegClass::VGPR, 5, 1}));
}

// --- Generated VOP3 DPP destination-preserve reads (real decode) ---
//
// VOP3 gained DPP on gfx11+ (RDNA3/RDNA4/gfx1250) and has no SDWA, so only the
// partial-DPP path applies. Unlike VOP1/VOP2 the VOP3 vdst field can name an
// SGPR: a VOP3-re-encoded compare (v_cmp_*_e64) writes its lane mask to an SGPR
// through vdst. So Vop3::implicit_uses derives the preserved ref from the
// decoded destination operand rather than assuming VGPR -- these cases exercise
// both a VGPR-dest op and an SGPR-dest compare. VOP3 is not in CDNA, so these
// decode for RDNA4.
//
// RDNA4 VOP3 word0: encoding[31:26]=53, op[25:16], clamp[15], opsel[14:11],
// abs[10:8], vdst[7:0]. word1: src0[8:0]=marker (250=SRC_DPP), src1[17:9]. The
// DPP16 word2 layout matches VOP1/VOP2, so its bit fields are reused.
constexpr uint32_t kVop3Enc = 53u << 26;
constexpr uint32_t kVop3AddF32Op = 259u << 16;  // v_add_f32_e64 (VGPR vdst)
constexpr uint32_t kVop3CmpLtF32Op = 17u << 16; // v_cmp_lt_f32_e64 (SGPR vdst)
// word1: src0=SRC_DPP, src1=VGPR3.
constexpr uint32_t kVop3DppWord1 = (3u << 9) | 250u;

std::unique_ptr<Instruction> decode_rdna4(const std::array<uint32_t, 3> &words) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  return std::unique_ptr<Instruction>(decoder ? decoder->decode(words.data()) : nullptr);
}

TEST(GeneratedInstDefUse, Vop3DppPartialRowMaskReadsVgprDestination) {
  // v_add_f32_e64 (VGPR vdst=5), DPP word2: row_mask=0x7 (partial), bank_mask=0xF.
  auto inst = decode_rdna4(
      {kVop3Enc | kVop3AddF32Op | 5u, kVop3DppWord1, (0x7u << 28) | (0xFu << 24) | 2u});
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()).substr(0, 9), "v_add_f32");

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::VGPR, 5, 1}));
  EXPECT_TRUE(idu.uses.contains({RegClass::VGPR, 5, 1}));
}

TEST(GeneratedInstDefUse, Vop3DppFullRowMaskDoesNotReadDestination) {
  // Full masks, dpp_ctrl=0 (quad_perm, never OOB) -> every lane written.
  auto inst = decode_rdna4(
      {kVop3Enc | kVop3AddF32Op | 5u, kVop3DppWord1, (0xFu << 28) | (0xFu << 24) | 2u});
  ASSERT_NE(inst, nullptr);

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::VGPR, 5, 1}));
  EXPECT_FALSE(idu.uses.contains({RegClass::VGPR, 5, 1}));
}

TEST(GeneratedInstDefUse, Vop3DppBoundCtrlZeroEdgeCrossingReadsDestination) {
  // Full masks, bound_ctrl=0, row_shr:1 -> row-edge lanes read OOB and are left
  // unwritten, preserving the VGPR vdst.
  auto inst = decode_rdna4(
      {kVop3Enc | kVop3AddF32Op | 5u, kVop3DppWord1, kDppFullMasks | kDppCtrlRowShr1 | 2u});
  ASSERT_NE(inst, nullptr);

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::VGPR, 5, 1}));
  EXPECT_TRUE(idu.uses.contains({RegClass::VGPR, 5, 1}));
}

TEST(GeneratedInstDefUse, Vop3CmpDppPartialRowMaskDoesNotReadDestination) {
  // v_cmp_lt_f32_e64 writes its lane mask to an SGPR pair via the vdst field
  // (s[8:9]). The executor's non-VOPC DPP restore only touches the VGPR file at
  // inst_.vdst -- a no-op that writes back the saved value -- and does NOT
  // preserve the SGPR mask, which is fully written. So a partial mask reads
  // neither the SGPR nor a VGPR, matching implicit_uses filtering to VGPR.
  auto inst = decode_rdna4(
      {kVop3Enc | kVop3CmpLtF32Op | 8u, kVop3DppWord1, (0x7u << 28) | (0xFu << 24) | 2u});
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()).substr(0, 9), "v_cmp_lt_");

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::SGPR, 8, 2}));
  EXPECT_FALSE(idu.uses.contains({RegClass::SGPR, 8, 2}));
  EXPECT_FALSE(idu.uses.contains({RegClass::VGPR, 8, 1}));
}

TEST(GeneratedInstDefUse, Vop3pDppPartialRowMaskReadsDestination) {
  // v_pk_add_u16 (VOP3P, VGPR vdst=6). VOP3P gained DPP on gfx11+ and has no
  // SDWA, so a partial row mask preserves the packed VGPR dst.
  // RDNA4 VOP3P word0: encoding[31:24]=204, op[22:16]=10 (v_pk_add_u16),
  // vdst[7:0]=6. word1: src0[8:0]=250 (SRC_DPP), src1[17:9]=3 (VGPR3).
  constexpr uint32_t kVop3pAddU16Word0 = (204u << 24) | (10u << 16) | 6u;
  constexpr uint32_t kVop3pDppWord1 = (3u << 9) | 250u;
  auto inst = decode_rdna4({kVop3pAddU16Word0, kVop3pDppWord1, (0x7u << 28) | (0xFu << 24) | 2u});
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()).substr(0, 12), "v_pk_add_u16");

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::VGPR, 6, 1}));
  EXPECT_TRUE(idu.uses.contains({RegClass::VGPR, 6, 1}));
}

TEST(GeneratedInstDefUse, Vop3SdstEncDppPartialRowMaskReadsOnlyVgprResult) {
  // v_add_co_ci_u32_e64 (VOP3_SDST_ENC) writes TWO destinations: a VGPR result
  // (v6) and an SGPR carry-out (s[8:9]). The executor's DPP restore preserves
  // only the VGPR result (write_vgpr); the SGPR carry is fully written, so only
  // the VGPR surfaces as a use -- implicit_uses filters to RegClass::VGPR.
  // RDNA4 VOP3_SDST_ENC word0: encoding[31:26]=53, op[25:16]=288, sdst[14:8]=8,
  // vdst[7:0]=6. word1: src0=250 (SRC_DPP), src1[17:9]=3, src2[26:18]=10 (carry).
  constexpr uint32_t kVop3SdstWord0 = (53u << 26) | (288u << 16) | (8u << 8) | 6u;
  constexpr uint32_t kVop3SdstWord1 = (10u << 18) | (3u << 9) | 250u;
  auto inst = decode_rdna4({kVop3SdstWord0, kVop3SdstWord1, (0x7u << 28) | (0xFu << 24) | 2u});
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()).substr(0, 14), "v_add_co_ci_u3");

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::VGPR, 6, 1}));
  EXPECT_TRUE(idu.defs.contains({RegClass::SGPR, 8, 2}));
  EXPECT_TRUE(idu.uses.contains({RegClass::VGPR, 6, 1}));
  EXPECT_FALSE(idu.uses.contains({RegClass::SGPR, 8, 2}));
}

} // namespace
} // namespace rocjitsu

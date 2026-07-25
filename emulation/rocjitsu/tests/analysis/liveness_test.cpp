// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/analysis/def_use_chain.h"
#include "rocjitsu/analysis/indirect_branch_discovery.h"
#include "rocjitsu/analysis/kernel_scope.h"
#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/code_object.h"
#include "rocjitsu/code/patch/cdna4_instrumentation_builder.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/mubuf.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/opcodes.h"
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

uint32_t pack_gfx1250_sop1(uint16_t op, uint16_t sdst, uint16_t ssrc0) {
  return gfx1250::build_sop1(
      op, {.ssrc0 = static_cast<uint8_t>(ssrc0), .sdst = static_cast<uint8_t>(sdst)})[0];
}

uint32_t pack_gfx1250_sop2(uint16_t op, uint16_t sdst, uint16_t ssrc0, uint16_t ssrc1) {
  return gfx1250::build_sop2(op, {.ssrc0 = static_cast<uint8_t>(ssrc0),
                                  .ssrc1 = static_cast<uint8_t>(ssrc1),
                                  .sdst = static_cast<uint8_t>(sdst)})[0];
}

uint32_t build_s_call_b64(uint16_t sdst, int16_t simm16) {
  return cdna3::build_sopk(cdna3::kSCallB64Sopk, {.simm16 = static_cast<uint16_t>(simm16),
                                                  .sdst = static_cast<uint8_t>(sdst)})[0];
}

std::array<uint32_t, 2> build_cdna4_writelane(uint16_t vgpr, uint16_t sgpr,
                                              uint16_t lane_selector) {
  return cdna4::build_vop3(
      cdna4::kVWritelaneB32Vop3,
      {.vdst = static_cast<uint8_t>(vgpr), .src0 = sgpr, .src1 = lane_selector, .src2 = 128});
}

std::array<uint32_t, 2> build_cdna4_readlane(uint16_t sgpr, uint16_t vgpr, uint16_t lane_selector) {
  return cdna4::build_vop3(cdna4::kVReadlaneB32Vop3, {.vdst = static_cast<uint8_t>(sgpr),
                                                      .src0 = static_cast<uint16_t>(256 + vgpr),
                                                      .src1 = lane_selector,
                                                      .src2 = 128});
}

std::array<uint32_t, 2> build_cdna4_accvgpr_write(uint16_t acc_vgpr, uint16_t src_vgpr) {
  return cdna4::build_vop3p(cdna4::kVAccvgprWriteVop3p,
                            {.vdst = static_cast<uint8_t>(acc_vgpr),
                             .op_sel_hi_2 = 1u,
                             .src0 = static_cast<uint16_t>(256u + src_vgpr),
                             .op_sel_hi = 3u});
}

std::array<uint32_t, 2> build_cdna4_accvgpr_read(uint16_t dst_vgpr, uint16_t acc_vgpr) {
  return cdna4::build_vop3p(cdna4::kVAccvgprReadVop3p,
                            {.vdst = static_cast<uint8_t>(dst_vgpr),
                             .op_sel_hi_2 = 1u,
                             .src0 = static_cast<uint16_t>(256u + acc_vgpr),
                             .op_sel_hi = 3u});
}

std::array<uint32_t, 2> build_gfx1250_writelane(uint16_t vgpr, uint16_t sgpr,
                                                uint16_t lane_selector) {
  return gfx1250::build_vop3(
      gfx1250::kVWritelaneB32Vop3,
      {.vdst = static_cast<uint8_t>(vgpr), .src0 = sgpr, .src1 = lane_selector});
}

std::array<uint32_t, 2> build_gfx1250_readlane(uint16_t sgpr, uint16_t vgpr,
                                               uint16_t lane_selector) {
  return gfx1250::build_vop3(gfx1250::kVReadlaneB32Vop3, {.vdst = static_cast<uint8_t>(sgpr),
                                                          .src0 = static_cast<uint16_t>(256 + vgpr),
                                                          .src1 = lane_selector});
}

std::vector<IndirectCallFixup> discover_test_indirect_fixups(const std::vector<uint32_t> &words,
                                                             rj_code_arch_t arch,
                                                             std::span<const uint64_t> leaders,
                                                             uint32_t wavefront_size) {
  auto decoder = Decoder::create(arch);
  if (!decoder)
    return {};
  std::vector<std::unique_ptr<Instruction>> decoded;
  std::vector<const Instruction *> instructions;
  for (size_t word_index = 0; word_index < words.size();) {
    const uint64_t offset = word_index * sizeof(uint32_t);
    std::unique_ptr<Instruction> instruction(decoder->decode(words.data() + word_index, offset));
    if (!instruction || instruction->size() <= 0)
      return {};
    const size_t instruction_words = static_cast<size_t>(instruction->size()) / sizeof(uint32_t);
    if (instruction_words == 0 || instruction_words > words.size() - word_index)
      return {};
    instructions.push_back(instruction.get());
    decoded.push_back(std::move(instruction));
    word_index += instruction_words;
  }
  return discover_indirect_branch_edges(
      instructions,
      std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(words.data()),
                               words.size() * sizeof(uint32_t)),
      arch, leaders, wavefront_size);
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

TEST(CfgAnalysis, RecoversRdna4CanonicalizedGetpcCall) {
  constexpr uint16_t kPcSreg = 2;
  constexpr uint16_t kReturnSreg = 30;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kTargetDelta = static_cast<uint32_t>(-20);

  // Clang's RDNA4 device-call sequence canonicalizes the high 32-bit getpc
  // half to a signed 16-bit value before adding the low/high text delta.
  std::vector<uint32_t> words = {
      pack_sop1(0x48, 0, kReturnSreg),                         // 0x00: callee return.
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),                // 0x04: padding.
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),                // 0x08: padding.
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),                // 0x0c: padding.
      pack_sop1(0x47, kPcSreg, 0),                             // 0x10: s_getpc_b64.
      pack_sop1(0x0f, kPcSreg + 1, kPcSreg + 1),               // 0x14: s_sext_i32_i16.
      pack_sop2(0, kPcSreg, kPcSreg, kLiteralOperand),         // 0x18: s_add_co_u32.
      kTargetDelta,                                            // 0x1c: target delta.
      pack_sop2(4, kPcSreg + 1, kPcSreg + 1, kLiteralOperand), // 0x20: s_add_co_ci_u32.
      UINT32_MAX,                                              // 0x24: high delta.
      pack_sop1(0x49, kReturnSreg, kPcSreg),                   // 0x28: s_swappc_b64.
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),                // 0x2c: continuation.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  constexpr std::array<uint64_t, 2> extra_leaders{0, 16};
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_RDNA4, extra_leaders);

  auto *builder = block_starting_at(blocks, 16);
  auto *caller = block_starting_at(blocks, 40);
  auto *continuation = block_starting_at(blocks, 44);
  auto *callee = block_starting_at(blocks, 0);
  ASSERT_NE(builder, nullptr);
  ASSERT_NE(caller, nullptr);
  ASSERT_NE(continuation, nullptr);
  ASSERT_NE(callee, nullptr);
  const std::array<std::string_view, 4> expected_mnemonics = {"s_getpc_b64", "s_sext_i32_i16",
                                                              "s_add_co_u32", "s_add_co_ci_u32"};
  size_t mnemonic_index = 0;
  for (const Instruction &inst : builder->instructions()) {
    ASSERT_LT(mnemonic_index, expected_mnemonics.size());
    EXPECT_EQ(inst.mnemonic(), expected_mnemonics[mnemonic_index++]);
  }
  EXPECT_EQ(mnemonic_index, expected_mnemonics.size());
  ASSERT_EQ(caller->num_instructions(), 1u);
  ASSERT_NE(caller->terminator(), nullptr);
  EXPECT_EQ(caller->terminator()->mnemonic(), "s_swappc_b64");
  ASSERT_EQ(caller->static_indirect_call_fixups().size(), 1u);
  EXPECT_EQ(caller->static_indirect_call_fixups()[0].source_target_offset, 0u);
  ASSERT_EQ(caller->call_edges().size(), 1u);
  EXPECT_EQ(caller->call_edges()[0].kind, BasicBlock::CallEdgeKind::IndirectSwapPc);
  EXPECT_EQ(caller->call_edges()[0].callee, callee);
  EXPECT_EQ(caller->call_edges()[0].continuation, continuation);
}

enum class LaneSavedTargetMutation {
  None,
  DynamicLane,
  MismatchedHalf,
  CalleeClobber,
  ReturnPairClobber,
  CallerClobber,
  SameLaneWritelaneClobber,
  UnknownInterveningCall,
  DirectPreservingCall,
  DirectClobberingCall,
  SpecialReturnSelector,
};

std::string_view lane_saved_target_mutation_name(LaneSavedTargetMutation mutation) {
  switch (mutation) {
  case LaneSavedTargetMutation::None:
    return "None";
  case LaneSavedTargetMutation::DynamicLane:
    return "DynamicLane";
  case LaneSavedTargetMutation::MismatchedHalf:
    return "MismatchedHalf";
  case LaneSavedTargetMutation::CalleeClobber:
    return "CalleeClobber";
  case LaneSavedTargetMutation::ReturnPairClobber:
    return "ReturnPairClobber";
  case LaneSavedTargetMutation::CallerClobber:
    return "CallerClobber";
  case LaneSavedTargetMutation::SameLaneWritelaneClobber:
    return "SameLaneWritelaneClobber";
  case LaneSavedTargetMutation::UnknownInterveningCall:
    return "UnknownInterveningCall";
  case LaneSavedTargetMutation::DirectPreservingCall:
    return "DirectPreservingCall";
  case LaneSavedTargetMutation::DirectClobberingCall:
    return "DirectClobberingCall";
  case LaneSavedTargetMutation::SpecialReturnSelector:
    return "SpecialReturnSelector";
  }
  return "Unknown";
}

std::vector<uint32_t> build_lane_saved_target_program(LaneSavedTargetMutation mutation) {
  constexpr uint16_t kTargetSreg = 0;
  constexpr uint16_t kUnknownTargetSreg = 4;
  constexpr uint16_t kReturnSreg = 30;
  constexpr uint16_t kCarrierVgpr = 42;
  constexpr uint16_t kLowLane = 128 + 7;
  constexpr uint16_t kHighLane = 128 + 8;
  constexpr uint32_t kLiteralOperand = 255;

  std::vector<uint32_t> words;
  const auto append = [&](const auto &encoded) {
    words.insert(words.end(), encoded.begin(), encoded.end());
  };

  // 0x00: first callee. The negative case writes the complete carrier
  // register before returning.
  words.push_back(
      mutation == LaneSavedTargetMutation::CalleeClobber ||
              mutation == LaneSavedTargetMutation::DirectClobberingCall
          ? cdna4::build_vop1(cdna4::kVMovB32Vop1, {.src0 = 128, .vdst = kCarrierVgpr})[0]
          : build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  words.push_back(pack_sop1(0x1d, 0, kReturnSreg));
  // 0x08: a second callee used between the save and restore. One negative
  // case proves that naming the expected return pair at setpc is insufficient
  // after that pair was clobbered.
  if (mutation == LaneSavedTargetMutation::ReturnPairClobber) {
    words.push_back(
        build_s_mov_b32(kReturnSreg, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_CDNA4));
    words.push_back(pack_sop1(0x1d, 0, kReturnSreg));
  } else if (mutation == LaneSavedTargetMutation::SpecialReturnSelector) {
    // Exercise the raw SOP1 selector boundary: this is an inline/special
    // selector, not an ordinary tracked SGPR pair.
    words.push_back(pack_sop1(0x1d, 0, 200));
    words.push_back(build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  } else {
    words.push_back(pack_sop1(0x1d, 0, kReturnSreg));
    words.push_back(build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  }

  // 0x10: build the first callee target in s[0:1].
  words.push_back(pack_sop1(0x1c, kTargetSreg, 0));
  words.push_back(pack_sop2(0, kTargetSreg, kTargetSreg, kLiteralOperand));
  words.push_back(static_cast<uint32_t>(-20));
  words.push_back(pack_sop2(4, kTargetSreg + 1, kTargetSreg + 1, kLiteralOperand));
  words.push_back(UINT32_MAX);
  append(build_cdna4_writelane(kCarrierVgpr, kTargetSreg,
                               mutation == LaneSavedTargetMutation::DynamicLane ? kUnknownTargetSreg
                                                                                : kLowLane));
  append(build_cdna4_writelane(kCarrierVgpr, kTargetSreg + 1, kHighLane));
  words.push_back(pack_sop1(0x1e, kReturnSreg, kTargetSreg)); // 0x34.

  // 0x38: call a different target before restoring the saved first target.
  if (mutation == LaneSavedTargetMutation::UnknownInterveningCall) {
    for (size_t i = 0; i < 5; ++i)
      words.push_back(build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  } else {
    words.push_back(pack_sop1(0x1c, kTargetSreg, 0));
    words.push_back(pack_sop2(0, kTargetSreg, kTargetSreg, kLiteralOperand));
    words.push_back(static_cast<uint32_t>(-52));
    words.push_back(pack_sop2(4, kTargetSreg + 1, kTargetSreg + 1, kLiteralOperand));
    words.push_back(UINT32_MAX);
  }
  if (mutation == LaneSavedTargetMutation::DirectPreservingCall) {
    words.push_back(build_s_call_b64(kReturnSreg, -18)); // 0x4c -> 0x08.
  } else if (mutation == LaneSavedTargetMutation::DirectClobberingCall) {
    words.push_back(build_s_call_b64(kReturnSreg, -20)); // 0x4c -> 0x00.
  } else {
    words.push_back(pack_sop1(0x1e, kReturnSreg,
                              mutation == LaneSavedTargetMutation::UnknownInterveningCall
                                  ? kUnknownTargetSreg
                                  : kTargetSreg)); // 0x4c.
  }
  if (mutation == LaneSavedTargetMutation::SameLaneWritelaneClobber) {
    append(build_cdna4_writelane(kCarrierVgpr, kUnknownTargetSreg, kLowLane));
  } else {
    words.push_back(
        mutation == LaneSavedTargetMutation::CallerClobber
            ? cdna4::build_vop1(cdna4::kVMovB32Vop1, {.src0 = 128, .vdst = kCarrierVgpr})[0]
            : build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4)); // 0x50.
  }

  append(build_cdna4_readlane(kTargetSreg, kCarrierVgpr, kLowLane));
  append(build_cdna4_readlane(kTargetSreg + 1, kCarrierVgpr,
                              mutation == LaneSavedTargetMutation::MismatchedHalf ? kHighLane + 1
                                                                                  : kHighLane));
  words.push_back(pack_sop1(0x1e, kReturnSreg, kTargetSreg)); // 0x64.
  words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4));
  return words;
}

std::vector<uint32_t> build_gfx1250_lane_saved_target_program(uint16_t low_lane = 128u + 7u,
                                                              uint16_t high_lane = 128u + 8u,
                                                              bool movrel_before_restore = false) {
  constexpr uint16_t kTargetSreg = 0;
  constexpr uint16_t kReturnSreg = 30;
  constexpr uint16_t kCarrierVgpr = 42;
  constexpr uint32_t kLiteralOperand = 255;

  std::vector<uint32_t> words;
  const auto append = [&](const auto &encoded) {
    words.insert(words.end(), encoded.begin(), encoded.end());
  };
  words.push_back(pack_gfx1250_sop1(0x48, 0, kReturnSreg));
  words.push_back(build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));
  words.push_back(pack_gfx1250_sop1(0x48, 0, kReturnSreg));
  words.push_back(build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));

  words.push_back(pack_gfx1250_sop1(0x47, kTargetSreg, 0));
  words.push_back(pack_gfx1250_sop2(0, kTargetSreg, kTargetSreg, kLiteralOperand));
  words.push_back(static_cast<uint32_t>(-20));
  words.push_back(pack_gfx1250_sop2(4, kTargetSreg + 1, kTargetSreg + 1, kLiteralOperand));
  words.push_back(UINT32_MAX);
  append(build_gfx1250_writelane(kCarrierVgpr, kTargetSreg, low_lane));
  append(build_gfx1250_writelane(kCarrierVgpr, kTargetSreg + 1, high_lane));
  words.push_back(pack_gfx1250_sop1(0x49, kReturnSreg, kTargetSreg));

  words.push_back(pack_gfx1250_sop1(0x47, kTargetSreg, 0));
  words.push_back(pack_gfx1250_sop2(0, kTargetSreg, kTargetSreg, kLiteralOperand));
  words.push_back(static_cast<uint32_t>(-52));
  words.push_back(pack_gfx1250_sop2(4, kTargetSreg + 1, kTargetSreg + 1, kLiteralOperand));
  words.push_back(UINT32_MAX);
  words.push_back(pack_gfx1250_sop1(0x49, kReturnSreg, kTargetSreg));
  words.push_back(
      movrel_before_restore
          ? gfx1250::build_vop1(gfx1250::kVMovrelsB32Vop1,
                                {.src0 = static_cast<uint16_t>(256u + kCarrierVgpr), .vdst = 0u})[0]
          : build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));

  append(build_gfx1250_readlane(kTargetSreg, kCarrierVgpr, low_lane));
  append(build_gfx1250_readlane(kTargetSreg + 1, kCarrierVgpr, high_lane));
  words.push_back(pack_gfx1250_sop1(0x49, kReturnSreg, kTargetSreg));
  words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250));
  return words;
}

enum class UnprovenCarrier {
  Scratch,
  AccVgpr,
};

struct UnprovenCarrierProgram {
  std::vector<uint32_t> words;
  uint64_t outer_callee_offset = 0;
  uint64_t caller_offset = 0;
  uint64_t initial_consumer_offset = 0;
  uint64_t final_consumer_offset = 0;
};

UnprovenCarrierProgram build_unproven_carrier_program(UnprovenCarrier carrier) {
  constexpr uint16_t kTargetSreg = 0;
  constexpr uint16_t kReturnSreg = 30;
  constexpr uint16_t kReturnCarrierVgpr = 40;
  constexpr uint16_t kReturnCarrierAccVgpr = 40;
  constexpr uint16_t kTargetCarrierVgpr = 42;
  constexpr uint16_t kReturnLowLane = 128u;
  constexpr uint16_t kReturnHighLane = 129u;
  constexpr uint16_t kTargetLowLane = 128u + 7u;
  constexpr uint16_t kTargetHighLane = 128u + 8u;
  constexpr uint16_t kScratchSaddr = 33;
  constexpr uint32_t kLiteralOperand = 255;

  UnprovenCarrierProgram program;
  const auto append = [&](const auto &encoded) {
    program.words.insert(program.words.end(), encoded.begin(), encoded.end());
  };
  const auto offset = [&] {
    return static_cast<uint64_t>(program.words.size()) * sizeof(uint32_t);
  };
  const auto append_target_builder = [&](uint64_t target) {
    const uint64_t getpc_offset = offset();
    const int64_t delta =
        static_cast<int64_t>(target) - static_cast<int64_t>(getpc_offset + sizeof(uint32_t));
    program.words.push_back(pack_sop1(0x1c, kTargetSreg, 0));
    program.words.push_back(pack_sop2(0, kTargetSreg, kTargetSreg, kLiteralOperand));
    program.words.push_back(static_cast<uint32_t>(delta));
    program.words.push_back(pack_sop2(4, kTargetSreg + 1, kTargetSreg + 1, kLiteralOperand));
    program.words.push_back(delta < 0 ? UINT32_MAX : 0u);
  };

  // Inner callee: the matching transfer round-trip looks like it preserves its
  // caller's complete v40 value, but both carrier classes are EXEC-gated and
  // scratch additionally lacks stable-frame and non-alias proofs. Recovery must
  // therefore fail closed.
  if (carrier == UnprovenCarrier::Scratch) {
    append(build_cdna4_scratch_store_b32_saddr(kReturnCarrierVgpr, kScratchSaddr, 0,
                                               ROCJITSU_CODE_ARCH_CDNA4)
               .value());
  } else {
    append(build_cdna4_accvgpr_write(kReturnCarrierAccVgpr, kReturnCarrierVgpr));
  }
  append(build_cdna4_writelane(kReturnCarrierVgpr, kReturnSreg, kReturnLowLane));
  append(build_cdna4_writelane(kReturnCarrierVgpr, kReturnSreg + 1, kReturnHighLane));
  append(build_cdna4_readlane(kReturnSreg, kReturnCarrierVgpr, kReturnLowLane));
  append(build_cdna4_readlane(kReturnSreg + 1, kReturnCarrierVgpr, kReturnHighLane));
  if (carrier == UnprovenCarrier::Scratch) {
    append(build_cdna4_scratch_load_b32_saddr(kReturnCarrierVgpr, kScratchSaddr, 0u,
                                              ROCJITSU_CODE_ARCH_CDNA4)
               .value());
    program.words.push_back(0xbf8c0f70u); // s_waitcnt vmcnt(0).
  } else {
    append(build_cdna4_accvgpr_read(kReturnCarrierVgpr, kReturnCarrierAccVgpr));
  }
  program.words.push_back(pack_sop1(0x1d, 0, kReturnSreg));

  // Outer callee: try to keep its incoming return pair in v40 across that
  // nested call. A syntactic transfer round-trip alone is not a preservation
  // proof.
  program.outer_callee_offset = offset();
  append(build_cdna4_writelane(kReturnCarrierVgpr, kReturnSreg, kReturnLowLane));
  append(build_cdna4_writelane(kReturnCarrierVgpr, kReturnSreg + 1, kReturnHighLane));
  append_target_builder(0);
  program.words.push_back(pack_sop1(0x1e, kReturnSreg, kTargetSreg));
  append(build_cdna4_readlane(kReturnSreg, kReturnCarrierVgpr, kReturnLowLane));
  append(build_cdna4_readlane(kReturnSreg + 1, kReturnCarrierVgpr, kReturnHighLane));
  program.words.push_back(pack_sop1(0x1d, 0, kReturnSreg));

  // Caller: establish one reusable static target, call it once, then attempt
  // to recover the same target from v42 after the unproven carrier chain.
  program.caller_offset = offset();
  append_target_builder(program.outer_callee_offset);
  append(build_cdna4_writelane(kTargetCarrierVgpr, kTargetSreg, kTargetLowLane));
  append(build_cdna4_writelane(kTargetCarrierVgpr, kTargetSreg + 1, kTargetHighLane));
  program.initial_consumer_offset = offset();
  program.words.push_back(pack_sop1(0x1e, kReturnSreg, kTargetSreg));
  append(build_cdna4_readlane(kTargetSreg, kTargetCarrierVgpr, kTargetLowLane));
  append(build_cdna4_readlane(kTargetSreg + 1, kTargetCarrierVgpr, kTargetHighLane));
  program.final_consumer_offset = offset();
  program.words.push_back(pack_sop1(0x1e, kReturnSreg, kTargetSreg));
  program.words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4));
  return program;
}

TEST(CfgAnalysis, RecoversLaneSavedTargetAcrossPreservingCalls) {
  TestCodeObject co(build_lane_saved_target_program(LaneSavedTargetMutation::None));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  constexpr std::array<uint64_t, 3> extra_leaders{0, 8, 16};
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4, extra_leaders);

  BasicBlock *consumer = block_starting_at(blocks, 0x64);
  BasicBlock *callee = block_starting_at(blocks, 0);
  ASSERT_NE(consumer, nullptr);
  ASSERT_NE(callee, nullptr);
  ASSERT_EQ(consumer->static_indirect_call_fixups().size(), 1u);
  EXPECT_EQ(consumer->static_indirect_call_fixups()[0].source_target_offset, 0u);
  ASSERT_EQ(consumer->call_edges().size(), 1u);
  EXPECT_EQ(consumer->call_edges()[0].callee, callee);
}

TEST(CfgAnalysis, RecoversGfx1250LaneSavedTargetOperandLayout) {
  TestCodeObject co(build_gfx1250_lane_saved_target_program());
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  constexpr std::array<uint64_t, 3> extra_leaders{0, 8, 16};
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_GFX1250, extra_leaders);

  const auto consumer_it = std::ranges::find_if(blocks, [](const auto &block) {
    return block && block->terminator() && block->terminator()->src_loc() == 0x64u;
  });
  BasicBlock *consumer = consumer_it == blocks.end() ? nullptr : consumer_it->get();
  BasicBlock *callee = block_starting_at(blocks, 0);
  ASSERT_NE(consumer, nullptr);
  ASSERT_NE(callee, nullptr);
  ASSERT_EQ(consumer->static_indirect_call_fixups().size(), 1u);
  EXPECT_EQ(consumer->static_indirect_call_fixups()[0].source_target_offset, 0u);
  ASSERT_EQ(consumer->call_edges().size(), 1u);
  EXPECT_EQ(consumer->call_edges()[0].callee, callee);
}

TEST(CfgAnalysis, RecoversLaneSavedTargetAcrossDirectPreservingCall) {
  TestCodeObject co(build_lane_saved_target_program(LaneSavedTargetMutation::DirectPreservingCall));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  constexpr std::array<uint64_t, 3> extra_leaders{0, 8, 16};
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4, extra_leaders);

  BasicBlock *consumer = block_starting_at(blocks, 0x64);
  ASSERT_NE(consumer, nullptr);
  ASSERT_EQ(consumer->static_indirect_call_fixups().size(), 1u);
  EXPECT_EQ(consumer->static_indirect_call_fixups()[0].source_target_offset, 0u);
}

TEST(CfgAnalysis, RejectsLaneSavedTargetAcrossUnprovenScratchCarrier) {
  UnprovenCarrierProgram program = build_unproven_carrier_program(UnprovenCarrier::Scratch);
  const std::array extra_leaders{uint64_t{0}, program.outer_callee_offset, program.caller_offset};
  const std::vector<IndirectCallFixup> fixups = discover_test_indirect_fixups(
      program.words, ROCJITSU_CODE_ARCH_CDNA4, extra_leaders, /*wavefront_size=*/64u);
  EXPECT_TRUE(std::ranges::any_of(fixups, [&](const IndirectCallFixup &fixup) {
    return fixup.source_call_offset == program.initial_consumer_offset &&
           fixup.source_target_offset == program.outer_callee_offset;
  }));
  EXPECT_TRUE(std::ranges::none_of(fixups, [&](const IndirectCallFixup &fixup) {
    return fixup.source_call_offset == program.final_consumer_offset;
  }));
}

TEST(CfgAnalysis, RejectsLaneSavedTargetAcrossUnprovenAccVgprCarrier) {
  UnprovenCarrierProgram program = build_unproven_carrier_program(UnprovenCarrier::AccVgpr);
  const std::array extra_leaders{uint64_t{0}, program.outer_callee_offset, program.caller_offset};
  const std::vector<IndirectCallFixup> fixups = discover_test_indirect_fixups(
      program.words, ROCJITSU_CODE_ARCH_CDNA4, extra_leaders, /*wavefront_size=*/64u);
  EXPECT_TRUE(std::ranges::any_of(fixups, [&](const IndirectCallFixup &fixup) {
    return fixup.source_call_offset == program.initial_consumer_offset &&
           fixup.source_target_offset == program.outer_callee_offset;
  }));
  EXPECT_TRUE(std::ranges::none_of(fixups, [&](const IndirectCallFixup &fixup) {
    return fixup.source_call_offset == program.final_consumer_offset;
  }));
}

TEST(CfgAnalysis, RejectsLaneSavedTargetBypassedFromExternalEntry) {
  TestCodeObject co(build_lane_saved_target_program(LaneSavedTargetMutation::None));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  constexpr std::array<uint64_t, 4> extra_leaders{0, 8, 16, 0x64};
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4, extra_leaders);

  BasicBlock *consumer = block_starting_at(blocks, 0x64);
  ASSERT_NE(consumer, nullptr);
  EXPECT_TRUE(consumer->static_indirect_call_fixups().empty());
  EXPECT_TRUE(consumer->call_edges().empty());
}

TEST(CfgAnalysis, RejectsFixedLaneOutsideKnownWavefront) {
  constexpr std::array<uint64_t, 3> extra_leaders{0, 8, 16};
  const std::vector<uint32_t> words =
      build_gfx1250_lane_saved_target_program(/*low_lane=*/128u + 40u,
                                              /*high_lane=*/128u + 41u);
  const std::vector<IndirectCallFixup> fixups = discover_test_indirect_fixups(
      words, ROCJITSU_CODE_ARCH_GFX1250, extra_leaders, /*wavefront_size=*/32u);
  EXPECT_TRUE(std::ranges::none_of(
      fixups, [](const IndirectCallFixup &fixup) { return fixup.source_call_offset == 0x64u; }));
}

TEST(CfgAnalysis, RejectsLiteralSelectedLaneEvenWhenLiteralLooksInline) {
  constexpr std::array<uint64_t, 3> extra_leaders{0, 8, 16};
  std::vector<uint32_t> words = build_gfx1250_lane_saved_target_program();
  constexpr size_t kFirstWritelaneSecondWord = 10u;
  constexpr uint32_t kVop3Src1Shift = 9u;
  constexpr uint32_t kVop3SourceMask = 0x1ffu;
  words[kFirstWritelaneSecondWord] &= ~(kVop3SourceMask << kVop3Src1Shift);
  words[kFirstWritelaneSecondWord] |= 255u << kVop3Src1Shift;
  // A decoded literal value of 135 is numerically identical to inline lane 7.
  // The raw selector still says "literal" and must not become static provenance.
  words.insert(words.begin() + static_cast<ptrdiff_t>(kFirstWritelaneSecondWord + 1u), 135u);
  const uint64_t final_consumer_offset =
      static_cast<uint64_t>(words.size() - 2u) * sizeof(uint32_t);
  const std::vector<IndirectCallFixup> fixups = discover_test_indirect_fixups(
      words, ROCJITSU_CODE_ARCH_GFX1250, extra_leaders, /*wavefront_size=*/64u);
  EXPECT_TRUE(std::ranges::none_of(fixups, [&](const IndirectCallFixup &fixup) {
    return fixup.source_call_offset == final_consumer_offset;
  }));
}

TEST(CfgAnalysis, RejectsRelativeVgprAccessAcrossLaneSavedTarget) {
  TestCodeObject co(build_gfx1250_lane_saved_target_program(
      /*low_lane=*/128u + 7u, /*high_lane=*/128u + 8u, /*movrel_before_restore=*/true));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  constexpr std::array<uint64_t, 3> extra_leaders{0, 8, 16};
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_GFX1250, extra_leaders);

  const auto consumer_it = std::ranges::find_if(blocks, [](const auto &block) {
    return block && block->terminator() && block->terminator()->src_loc() == 0x64u;
  });
  BasicBlock *consumer = consumer_it == blocks.end() ? nullptr : consumer_it->get();
  ASSERT_NE(consumer, nullptr);
  EXPECT_TRUE(consumer->static_indirect_call_fixups().empty());
  EXPECT_TRUE(consumer->call_edges().empty());
}

TEST(CfgAnalysis, RejectsUnprovenLaneSavedTargets) {
  constexpr std::array mutations{
      LaneSavedTargetMutation::DynamicLane,
      LaneSavedTargetMutation::MismatchedHalf,
      LaneSavedTargetMutation::CalleeClobber,
      LaneSavedTargetMutation::ReturnPairClobber,
      LaneSavedTargetMutation::CallerClobber,
      LaneSavedTargetMutation::SameLaneWritelaneClobber,
      LaneSavedTargetMutation::UnknownInterveningCall,
      LaneSavedTargetMutation::DirectClobberingCall,
      LaneSavedTargetMutation::SpecialReturnSelector,
  };

  for (LaneSavedTargetMutation mutation : mutations) {
    SCOPED_TRACE(lane_saved_target_mutation_name(mutation));
    std::vector<uint32_t> words = build_lane_saved_target_program(mutation);
    const uint64_t final_consumer_offset =
        static_cast<uint64_t>(words.size() - 2u) * sizeof(uint32_t);
    TestCodeObject co(std::move(words));
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    ASSERT_NE(decoder, nullptr);
    constexpr std::array<uint64_t, 3> extra_leaders{0, 8, 16};
    auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4, extra_leaders);

    const auto consumer_it = std::ranges::find_if(blocks, [&](const auto &block) {
      return block && block->terminator() &&
             block->terminator()->src_loc() == final_consumer_offset;
    });
    BasicBlock *consumer = consumer_it == blocks.end() ? nullptr : consumer_it->get();
    ASSERT_NE(consumer, nullptr);
    EXPECT_TRUE(consumer->static_indirect_call_fixups().empty());
    EXPECT_TRUE(consumer->call_edges().empty());
  }
}

TEST(CfgAnalysis, RejectsNoncanonicalRdna4GetpcSignExtensions) {
  constexpr uint16_t kPcSreg = 2;
  constexpr uint16_t kOtherSreg = 8;
  constexpr uint16_t kReturnSreg = 30;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kTargetDelta = static_cast<uint32_t>(-20);

  struct SignExtensionOperands {
    uint16_t dst;
    uint16_t src;
  };
  constexpr std::array<SignExtensionOperands, 2> cases{{
      // The low half is not a valid sign-extension destination.
      {kPcSreg, kPcSreg},
      // The high half is overwritten from an unrelated register.
      {kPcSreg + 1, kOtherSreg},
  }};

  for (const auto [sext_dst, sext_src] : cases) {
    SCOPED_TRACE(testing::Message() << "sext_src=" << sext_src << " sext_dst=" << sext_dst);
    std::vector<uint32_t> words = {
        pack_sop1(0x48, 0, kReturnSreg),
        build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
        build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
        build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
        pack_sop1(0x47, kPcSreg, 0),
        pack_sop1(0x0f, sext_dst, sext_src),
        pack_sop2(0, kPcSreg, kPcSreg, kLiteralOperand),
        kTargetDelta,
        pack_sop2(4, kPcSreg + 1, kPcSreg + 1, kLiteralOperand),
        UINT32_MAX,
        pack_sop1(0x49, kReturnSreg, kPcSreg),
        build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
    };

    TestCodeObject co(std::move(words));
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_NE(decoder, nullptr);
    constexpr std::array<uint64_t, 2> extra_leaders{0, 16};
    auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_RDNA4, extra_leaders);

    for (const auto &block : blocks) {
      EXPECT_TRUE(block->static_indirect_call_fixups().empty());
      EXPECT_TRUE(block->call_edges().empty());
    }
  }
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

TEST(KernelScopeAnalysis, SeparatesAdjacentKernelEntries) {
  std::vector<uint32_t> words = {
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  TestCodeObject co(words);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  constexpr std::array<uint64_t, 2> entries{0, 4};
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4, entries);
  const auto index = build_block_offset_index(blocks);
  const auto text = std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(words.data()),
                                             words.size() * sizeof(uint32_t));

  const auto first = build_kernel_cfg_scope(
      blocks, index, KernelScopeRequest{.entry_offset = 0, .additional_entry_offsets = {}}, entries,
      text);
  const auto second = build_kernel_cfg_scope(
      blocks, index, KernelScopeRequest{.entry_offset = 4, .additional_entry_offsets = {}}, entries,
      text);
  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(second.has_value());
  ASSERT_EQ(first->blocks.size(), 1u);
  ASSERT_EQ(second->blocks.size(), 1u);
  EXPECT_EQ(first->blocks[0]->start_offset(), 0u);
  EXPECT_EQ(second->blocks[0]->start_offset(), 4u);
  EXPECT_EQ(first->entry, block_for_offset(index, 0));
  EXPECT_EQ(second->entry, block_for_offset(index, 4));
}

TEST(KernelScopeAnalysis, SymbolRangesExcludeTextPadding) {
  std::vector<uint32_t> words = {
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
      0,
      0,
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  TestCodeObject co(words);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  constexpr std::array<uint64_t, 2> entries{0, 12};
  constexpr std::array<BasicBlock::CodeRange, 2> ranges{{{0, 4}, {12, 4}}};
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4, entries, ranges);

  ASSERT_EQ(blocks.size(), 2u);
  EXPECT_EQ(blocks[0]->start_offset(), 0u);
  EXPECT_EQ(blocks[1]->start_offset(), 12u);
  EXPECT_TRUE(blocks[0]->successors().empty());
  EXPECT_TRUE(blocks[1]->predecessors().empty());
}

TEST(CfgAnalysis, MarksUndecodedDirectTargetIncomplete) {
  std::vector<uint32_t> words = {
      build_s_branch(1, ROCJITSU_CODE_ARCH_CDNA4), // 0x00 -> 0x08.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  TestCodeObject co(words);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  constexpr std::array<BasicBlock::CodeRange, 1> ranges{{{0, 4}}};
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4, {}, ranges);

  ASSERT_EQ(blocks.size(), 1u);
  EXPECT_FALSE(blocks.front()->static_successors_complete());
  EXPECT_EQ(blocks.front()->static_successor_issue(),
            BasicBlock::StaticSuccessorIssue::MissingBranchTarget);
}

TEST(CfgAnalysis, MarksUndecodedFallthroughIncomplete) {
  std::vector<uint32_t> words = {
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  TestCodeObject co(words);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  constexpr std::array<BasicBlock::CodeRange, 1> ranges{{{0, 4}}};
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4, {}, ranges);

  ASSERT_EQ(blocks.size(), 1u);
  EXPECT_FALSE(blocks.front()->static_successors_complete());
  EXPECT_EQ(blocks.front()->static_successor_issue(),
            BasicBlock::StaticSuccessorIssue::MissingFallthrough);
}

TEST(CfgAnalysis, MarksUndecodedDirectCallTargetIncomplete) {
  constexpr uint16_t kReturnSreg = 30;
  std::vector<uint32_t> words = {
      build_s_call_b64(kReturnSreg, 1),         // 0x00 -> omitted target at 0x08.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4), // 0x04 decoded continuation.
      pack_sop1(0x1d, 0, kReturnSreg),          // 0x08 omitted target.
  };
  TestCodeObject co(words);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  constexpr std::array<BasicBlock::CodeRange, 1> ranges{{{0, 8}}};
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4, {}, ranges);

  BasicBlock *call = block_starting_at(blocks, 0);
  ASSERT_NE(call, nullptr);
  EXPECT_FALSE(call->static_successors_complete());
  EXPECT_EQ(call->static_successor_issue(), BasicBlock::StaticSuccessorIssue::MissingCallTarget);
}

TEST(CfgAnalysis, ReachableBuildMarksUndecodedCallContinuationIncomplete) {
  constexpr uint16_t kReturnSreg = 30;
  std::vector<uint32_t> words = {
      build_s_call_b64(kReturnSreg, 1),         // 0x00 -> target at 0x08.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4), // 0x04 bounded-out continuation.
      pack_sop1(0x1d, 0, kReturnSreg),          // 0x08 decoded call target.
  };
  TestCodeObject co(words);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  constexpr std::array<uint64_t, 1> entries{0};
  constexpr std::array<uint64_t, 1> entry_sizes{4};
  auto blocks = BasicBlock::build_reachable(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4, entries,
                                            entry_sizes, /*wavefront_size=*/64u);

  BasicBlock *call = block_starting_at(blocks, 0);
  ASSERT_NE(call, nullptr);
  EXPECT_FALSE(call->static_successors_complete());
  EXPECT_EQ(call->static_successor_issue(),
            BasicBlock::StaticSuccessorIssue::MissingCallContinuation);
}

TEST(KernelScopeAnalysis, SeedsAdditionalDescriptorEntryWithoutClaimingNextKernel) {
  std::vector<uint32_t> words = {
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  TestCodeObject co(words);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  constexpr std::array<uint64_t, 3> leaders{0, 4, 8};
  constexpr std::array<uint64_t, 2> kernel_entries{0, 8};
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4, leaders);
  const auto index = build_block_offset_index(blocks);
  const auto text = std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(words.data()),
                                             words.size() * sizeof(uint32_t));

  KernelScopeRequest request{.entry_offset = 0, .additional_entry_offsets = {4}};
  const auto scope = build_kernel_cfg_scope(blocks, index, request, kernel_entries, text);
  ASSERT_TRUE(scope.has_value());
  ASSERT_EQ(scope->blocks.size(), 2u);
  EXPECT_EQ(scope->blocks[0]->start_offset(), 0u);
  EXPECT_EQ(scope->blocks[1]->start_offset(), 4u);

  request.additional_entry_offsets = {12};
  EXPECT_FALSE(build_kernel_cfg_scope(blocks, index, request, kernel_entries, text).has_value());
}

TEST(KernelScopeAnalysis, SharedHelperGetsContextSpecificReturnEdges) {
  constexpr uint16_t kReturnSreg = 30;
  std::vector<uint32_t> words = {
      build_s_call_b64(kReturnSreg, 3),         // 0x00 -> helper at 0x10.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4), // 0x04 kernel 0 continuation.
      build_s_call_b64(kReturnSreg, 1),         // 0x08 -> helper at 0x10.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4), // 0x0c kernel 1 continuation.
      pack_sop1(0x1d, 0, kReturnSreg),          // 0x10 shared helper return.
  };
  TestCodeObject co(words);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  constexpr std::array<uint64_t, 2> kernel_entries{0, 8};
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4, kernel_entries);
  const auto index = build_block_offset_index(blocks);
  const auto text = std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(words.data()),
                                             words.size() * sizeof(uint32_t));

  const auto first = build_kernel_cfg_scope(
      blocks, index, KernelScopeRequest{.entry_offset = 0, .additional_entry_offsets = {}},
      kernel_entries, text);
  const auto second = build_kernel_cfg_scope(
      blocks, index, KernelScopeRequest{.entry_offset = 8, .additional_entry_offsets = {}},
      kernel_entries, text);
  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(first->blocks.size(), 3u);
  EXPECT_EQ(second->blocks.size(), 3u);
  EXPECT_TRUE(first->call_return_offsets.contains(16));
  EXPECT_TRUE(second->call_return_offsets.contains(16));
  ASSERT_EQ(first->liveness_edges.size(), 2u);
  ASSERT_EQ(second->liveness_edges.size(), 2u);

  BasicBlock *helper = block_for_offset(index, 16);
  BasicBlock *first_continuation = block_for_offset(index, 4);
  BasicBlock *second_continuation = block_for_offset(index, 12);
  ASSERT_NE(helper, nullptr);
  EXPECT_EQ(first->owner_proofs.at(helper), KernelCfgOwnerProofKind::DirectCall);
  EXPECT_EQ(second->owner_proofs.at(helper), KernelCfgOwnerProofKind::DirectCall);
  EXPECT_EQ(first->liveness_edges[1].from, helper);
  EXPECT_EQ(first->liveness_edges[1].to, first_continuation);
  EXPECT_EQ(second->liveness_edges[1].from, helper);
  EXPECT_EQ(second->liveness_edges[1].to, second_continuation);
}

TEST(KernelScopeAnalysis, MixedLocalAndCallReachabilityDoesNotExemptSetpcReturn) {
  constexpr uint16_t kReturnSreg = 30;
  std::vector<uint32_t> words = {
      build_s_call_b64(kReturnSreg, 3),            // 0x00 -> helper at 0x10.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),    // 0x04 call continuation.
      build_s_branch(1, ROCJITSU_CODE_ARCH_CDNA4), // 0x08 -> same helper.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
      pack_sop1(0x1d, 0, kReturnSreg), // 0x10 syntactic return on both paths.
  };
  TestCodeObject co(words);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  constexpr std::array<uint64_t, 2> owned_entries{0, 8};
  constexpr std::array<uint64_t, 1> kernel_entries{0};
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4, owned_entries);
  const auto index = build_block_offset_index(blocks);
  const auto text = std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(words.data()),
                                             words.size() * sizeof(uint32_t));

  const auto scope = build_kernel_cfg_scope(
      blocks, index, KernelScopeRequest{.entry_offset = 0, .additional_entry_offsets = {8}},
      kernel_entries, text);
  ASSERT_TRUE(scope.has_value());
  BasicBlock *helper = block_for_offset(index, 16);
  ASSERT_NE(helper, nullptr);
  EXPECT_EQ(scope->owner_proofs.at(helper), KernelCfgOwnerProofKind::KernelLocal);
  EXPECT_FALSE(scope->call_return_offsets.contains(16));
  EXPECT_TRUE(std::ranges::any_of(scope->liveness_edges, [&](const ScopedCfgEdge &edge) {
    return edge.from == helper && edge.to == block_for_offset(index, 4);
  }));
}

TEST(KernelScopeAnalysis, MixedReturnRegisterContextsDoNotExemptSharedSetpcReturns) {
  constexpr uint16_t kFirstReturnSreg = 30;
  constexpr uint16_t kSecondReturnSreg = 32;
  std::vector<uint32_t> words = {
      build_s_call_b64(kFirstReturnSreg, 3),    // 0x00 -> helper at 0x10.
      build_s_call_b64(kSecondReturnSreg, 2),   // 0x04 -> same helper.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4), // 0x08 final continuation.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4), // 0x0c unrelated filler.
      pack_sopp(5, 1),                          // 0x10 -> s32 return at 0x18.
      pack_sop1(0x1d, 0, kFirstReturnSreg),     // 0x14 s30 return.
      pack_sop1(0x1d, 0, kSecondReturnSreg),    // 0x18 s32 return.
  };
  TestCodeObject co(words);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  constexpr std::array<uint64_t, 1> kernel_entries{0};
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4, kernel_entries);
  const auto index = build_block_offset_index(blocks);
  const auto text = std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(words.data()),
                                             words.size() * sizeof(uint32_t));

  const auto scope = build_kernel_cfg_scope(
      blocks, index, KernelScopeRequest{.entry_offset = 0, .additional_entry_offsets = {}},
      kernel_entries, text);
  ASSERT_TRUE(scope.has_value());
  BasicBlock *first_return = block_for_offset(index, 0x14);
  BasicBlock *second_return = block_for_offset(index, 0x18);
  BasicBlock *first_continuation = block_for_offset(index, 0x04);
  BasicBlock *second_continuation = block_for_offset(index, 0x08);
  ASSERT_NE(first_return, nullptr);
  ASSERT_NE(second_return, nullptr);
  ASSERT_NE(first_continuation, nullptr);
  ASSERT_NE(second_continuation, nullptr);

  EXPECT_FALSE(scope->call_return_offsets.contains(0x14));
  EXPECT_FALSE(scope->call_return_offsets.contains(0x18));
  EXPECT_TRUE(std::ranges::any_of(scope->liveness_edges, [&](const ScopedCfgEdge &edge) {
    return edge.from == first_return && edge.to == first_continuation;
  }));
  EXPECT_TRUE(std::ranges::any_of(scope->liveness_edges, [&](const ScopedCfgEdge &edge) {
    return edge.from == second_return && edge.to == second_continuation;
  }));
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

TEST(CfgAnalysis, IncompletePredecessorMarksRecoveredTargetsNonExhaustive) {
  constexpr uint16_t kPcSreg = 8;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;
  constexpr uint32_t kOriginalGetpcDelta = 28;

  // The fallthrough path builds one concrete target, while the branch path
  // reaches the same consumer with an unconstrained incoming pair. The
  // concrete edge remains useful for relocation and liveness, but it is not an
  // exhaustive description of the runtime target set.
  const std::vector<uint32_t> words = {
      pack_sopp(5, 5),                                     // 0x00 -> open path at 0x18.
      pack_sop1(0x1c, kPcSreg, 0),                         // 0x04: s_getpc_b64.
      pack_sop2(0, kPcSreg, kPcSreg, kLiteralOperand),     // 0x08: s_add_u32.
      kOriginalGetpcDelta,                                 // 0x0c: target delta.
      pack_sop2(4, kPcSreg + 1, kPcSreg + 1, kInlineInt0), // 0x10: s_addc_u32.
      build_s_branch(1, ROCJITSU_CODE_ARCH_CDNA4),         // 0x14 -> consumer at 0x1c.
      build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA4),         // 0x18 -> consumer at 0x1c.
      pack_sop1(0x1d, 0, kPcSreg),                         // 0x1c: joined consumer.
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),            // 0x20: not a target.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),            // 0x24: concrete target.
  };

  const std::vector<IndirectCallFixup> fixups =
      discover_test_indirect_fixups(words, ROCJITSU_CODE_ARCH_CDNA4, {}, /*wavefront_size=*/64u);
  const auto recovered = std::ranges::find_if(fixups, [](const IndirectCallFixup &fixup) {
    return fixup.source_call_offset == 0x1cu && fixup.source_target_offset == 0x24u;
  });
  ASSERT_NE(recovered, fixups.end());
  EXPECT_FALSE(recovered->source_targets_exhaustive);
}

TEST(CfgAnalysis, LaterRecoveredPredecessorDowngradesTargetExhaustiveness) {
  constexpr uint16_t kEdgePcSreg = 8u;
  constexpr uint16_t kConsumerPcSreg = 12u;
  constexpr uint16_t kUnresolvedPcSreg = 20u;
  constexpr uint32_t kLiteralOperand = 255u;
  constexpr uint32_t kInlineInt0 = 128u;

  // Round 1 sees only the direct builder block as a predecessor of the final
  // consumer and initially recovers its target exhaustively. The first setpc
  // also recovers an edge to that consumer, while the unrelated unresolved
  // setpc keeps discovery iterating. Round 2 adds the new predecessor, whose
  // exit has no fact for the consumer pair, so the already-emitted target must
  // be downgraded to non-exhaustive.
  const std::vector<uint32_t> words = {
      pack_sop1(0x1c, kEdgePcSreg, 0),                             // 0x00: edge getpc.
      pack_sop2(0, kEdgePcSreg, kEdgePcSreg, kLiteralOperand),     // 0x04: edge low.
      44u,                                                         // 0x08: -> 0x30.
      pack_sop2(4, kEdgePcSreg + 1, kEdgePcSreg + 1, kInlineInt0), // 0x0c: edge high.
      pack_sop1(0x1d, 0, kEdgePcSreg),                             // 0x10: recovered edge.
      pack_sop1(0x1c, kConsumerPcSreg, 0),                         // 0x14: target getpc.
      pack_sop2(0, kConsumerPcSreg, kConsumerPcSreg, kLiteralOperand),
      32u, // 0x1c: -> 0x38.
      pack_sop2(4, kConsumerPcSreg + 1, kConsumerPcSreg + 1, kInlineInt0),
      build_s_branch(2, ROCJITSU_CODE_ARCH_CDNA4), // 0x24 -> 0x30.
      pack_sop1(0x1d, 0, kUnresolvedPcSreg),       // 0x28: keeps round 2 live.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),    // 0x2c.
      pack_sop1(0x1d, 0, kConsumerPcSreg),         // 0x30: joined consumer.
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),    // 0x34.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),    // 0x38: concrete target.
  };

  const std::vector<IndirectCallFixup> fixups =
      discover_test_indirect_fixups(words, ROCJITSU_CODE_ARCH_CDNA4, {}, /*wavefront_size=*/64u);
  const auto recovered = std::ranges::find_if(fixups, [](const IndirectCallFixup &fixup) {
    return fixup.source_call_offset == 0x30u && fixup.source_target_offset == 0x38u;
  });
  ASSERT_NE(recovered, fixups.end());
  EXPECT_FALSE(recovered->source_targets_exhaustive);
}

TEST(CfgAnalysis, ReachableRecoveryDowngradesTargetAfterDecodingNewPredecessor) {
  constexpr uint16_t kEdgePcSreg = 8u;
  constexpr uint16_t kConsumerPcSreg = 12u;
  constexpr uint32_t kLiteralOperand = 255u;
  constexpr uint32_t kInlineInt0 = 128u;

  // The first reachable pass recovers the consumer target from its only
  // decoded predecessor. Recovering the earlier setpc then decodes a second
  // predecessor with no fact for the consumer pair. The outer reachable
  // builder must retain the target for relocation while downgrading its
  // exhaustiveness.
  const std::vector<uint32_t> words = {
      pack_sop1(0x1c, kEdgePcSreg, 0),                             // 0x00: edge getpc.
      pack_sop2(0, kEdgePcSreg, kEdgePcSreg, kLiteralOperand),     // 0x04: edge low.
      52u,                                                         // 0x08: -> 0x38.
      pack_sop2(4, kEdgePcSreg + 1, kEdgePcSreg + 1, kInlineInt0), // 0x0c: edge high.
      pack_sop1(0x1d, 0, kEdgePcSreg),                             // 0x10: recovered edge.
      pack_sop1(0x1c, kConsumerPcSreg, 0),                         // 0x14: target getpc.
      pack_sop2(0, kConsumerPcSreg, kConsumerPcSreg, kLiteralOperand),
      36u, // 0x1c: -> 0x3c.
      pack_sop2(4, kConsumerPcSreg + 1, kConsumerPcSreg + 1, kInlineInt0),
      build_s_branch(2, ROCJITSU_CODE_ARCH_CDNA4),  // 0x24 -> consumer at 0x30.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),     // 0x28.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),     // 0x2c.
      pack_sop1(0x1d, 0, kConsumerPcSreg),          // 0x30: joined consumer.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),     // 0x34.
      build_s_branch(-3, ROCJITSU_CODE_ARCH_CDNA4), // 0x38 -> consumer at 0x30.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),     // 0x3c: concrete target.
  };

  TestCodeObject co(words);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  constexpr std::array<uint64_t, 2> entries{0u, 0x14u};
  auto blocks =
      BasicBlock::build_reachable(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4, entries, {}, 64u);

  BasicBlock *consumer = block_starting_at(blocks, 0x30);
  ASSERT_NE(consumer, nullptr);
  const auto &fixups = consumer->static_indirect_call_fixups();
  const auto recovered = std::ranges::find_if(fixups, [](const IndirectCallFixup &fixup) {
    return fixup.source_call_offset == 0x30u && fixup.source_target_offset == 0x3cu;
  });
  ASSERT_NE(recovered, fixups.end());
  EXPECT_FALSE(recovered->source_targets_exhaustive);
}

TEST(CfgAnalysis, LocalPcBuilderMarksRecoveredTargetExhaustive) {
  constexpr uint16_t kPcSreg = 8;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;
  const std::vector<uint32_t> words = {
      pack_sop1(0x1c, kPcSreg, 0),
      pack_sop2(0, kPcSreg, kPcSreg, kLiteralOperand),
      12u,
      pack_sop2(4, kPcSreg + 1, kPcSreg + 1, kInlineInt0),
      pack_sop1(0x1d, 0, kPcSreg),
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };

  const std::vector<IndirectCallFixup> fixups =
      discover_test_indirect_fixups(words, ROCJITSU_CODE_ARCH_CDNA4, {}, /*wavefront_size=*/64u);
  ASSERT_EQ(fixups.size(), 1u);
  EXPECT_TRUE(fixups.front().source_targets_exhaustive);
}

TEST(CfgAnalysis, RejectsLastSelectorPcPairSource) {
  constexpr uint16_t kLastSelector = 127u;
  const std::vector<uint32_t> words = {
      // The final SGPR selector cannot name a complete pair.
      pack_sop1(0x1c, kLastSelector, 0),
      pack_sop1(0x1d, 0, kLastSelector),
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };

  const std::vector<IndirectCallFixup> fixups =
      discover_test_indirect_fixups(words, ROCJITSU_CODE_ARCH_CDNA4, {}, /*wavefront_size=*/64u);
  EXPECT_TRUE(fixups.empty());
}

TEST(CfgAnalysis, RejectsLastSelectorSwappcReturnSelector) {
  constexpr uint16_t kLastSelector = 127u;
  constexpr uint16_t kPcSreg = 8u;
  constexpr uint32_t kLiteralOperand = 255u;
  constexpr uint32_t kInlineInt0 = 128u;
  const std::vector<uint32_t> words = {
      pack_sop1(0x1c, kPcSreg, 0),
      pack_sop2(0, kPcSreg, kPcSreg, kLiteralOperand),
      12u,
      pack_sop2(4, kPcSreg + 1, kPcSreg + 1, kInlineInt0),
      pack_sop1(0x1e, kLastSelector, kPcSreg),
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };

  const std::vector<IndirectCallFixup> fixups =
      discover_test_indirect_fixups(words, ROCJITSU_CODE_ARCH_CDNA4, {}, /*wavefront_size=*/64u);
  EXPECT_TRUE(fixups.empty());
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

TEST(InstDefUse, Rdna4Vop3ScalarSource) {
  constexpr std::array<uint32_t, 2> words = {
      0xD7001141u,
      0x0202821Eu,
  }; // v_add_co_u32 v65, s17, s30, v65
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> instruction(decoder->decode(words.data()));
  ASSERT_NE(instruction, nullptr);

  const InstDefUse def_use(*instruction);
  EXPECT_TRUE(def_use.uses.contains({RegClass::SGPR, 30, 1}));
}

} // namespace
} // namespace rocjitsu

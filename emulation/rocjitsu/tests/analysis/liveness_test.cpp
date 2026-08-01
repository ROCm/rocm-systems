// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/analysis/def_use_chain.h"
#include "rocjitsu/analysis/indirect_branch_discovery.h"
#include "rocjitsu/analysis/kernel_scope.h"
#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/code_object.h"
#include "rocjitsu/code/dbt/binary_translator_internal.h"
#include "rocjitsu/code/patch/cdna4_instrumentation_builder.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/mubuf.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/operand.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/builders.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/operand.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/vbuffer.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/mubuf.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/operand.h"
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
#include <stdexcept>
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
      : Section(".text"), data_(std::move(data)), size_(size) {}

  std::size_t size() const override { return size_; }
  const char *data() const override { return data_.get(); }
  uint32_t sectionHeaderNameIdx() const override { return 0; }
  uint64_t sectionOffset() const override { return 0; }

private:
  std::unique_ptr<char[]> data_;
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

TEST(RegisterSetAnalysis, GeneratedOperandsClassifyRegisterKindWithoutDisplayNames) {
  rdna4::Operand rdna_sgpr(32, rdna4::OperandType::OPR_SRC, rdna4::OpSelSrc::OPR_SRC_SGPR_MIN + 7);
  rdna4::Operand rdna_ttmp(32, rdna4::OperandType::OPR_SRC, rdna4::OpSelSrc::OPR_SRC_TTMP_MIN + 3);
  rdna4::Operand rdna_null(32, rdna4::OperandType::OPR_SRC, rdna4::OpSelSrc::OPR_SRC_NULL);
  rdna4::Operand rdna_shared_base(32, rdna4::OperandType::OPR_SRC,
                                  rdna4::OpSelSrc::OPR_SRC_SRC_SHARED_BASE);
  rdna4::Operand rdna_inline_constant(32, rdna4::OperandType::OPR_SRC,
                                      rdna4::OpSelSrc::OPR_SRC_POS_INT_MIN + 1);
  rdna4::Operand rdna_dsmem(32, rdna4::OperandType::OPR_DSMEM, rdna4::OpSelDsmem::OPR_DSMEM_DSMEM);

  EXPECT_TRUE(rdna_sgpr.is_register());
  EXPECT_TRUE(rdna_ttmp.is_register());
  EXPECT_TRUE(rdna_null.is_register());
  EXPECT_TRUE(rdna_shared_base.is_register());
  EXPECT_FALSE(rdna_inline_constant.is_register());
  EXPECT_FALSE(rdna_dsmem.is_register());
  ASSERT_TRUE(rdna_ttmp.to_register_ref());
  EXPECT_EQ(*rdna_ttmp.to_register_ref(), (RegisterRef{RegClass::TTMP, 3, 1}));
  EXPECT_FALSE(rdna_null.to_register_ref());
  EXPECT_FALSE(rdna_shared_base.to_register_ref());

  cdna4::Operand cdna_vcc(64, cdna4::OperandType::OPR_SRC, cdna4::OpSelSrc::OPR_SRC_VCC_LO);
  ASSERT_TRUE(cdna_vcc.to_register_ref());
  EXPECT_EQ(*cdna_vcc.to_register_ref(), (RegisterRef{RegClass::VCC, 0, 2}));

  gfx1250::Operand gfx1250_null(32, gfx1250::OperandType::OPR_SRC, gfx1250::OpSelSrc::OPR_SRC_NULL);
  gfx1250::Operand gfx1250_shared_base(32, gfx1250::OperandType::OPR_SRC,
                                       gfx1250::OpSelSrc::OPR_SRC_SRC_SHARED_BASE);
  gfx1250::Operand gfx1250_inline_constant(32, gfx1250::OperandType::OPR_SRC,
                                           gfx1250::OpSelSrc::OPR_SRC_POS_INT_MIN + 1);
  gfx1250::Operand gfx1250_literal64(64, gfx1250::OperandType::OPR_SRC,
                                     gfx1250::OpSelSrc::OPR_SRC_SRC_LITERAL64);

  EXPECT_TRUE(gfx1250_null.is_register());
  EXPECT_TRUE(gfx1250_shared_base.is_register());
  EXPECT_FALSE(gfx1250_inline_constant.is_register());
  EXPECT_FALSE(gfx1250_literal64.is_register());

  gfx1250::Operand gfx1250_exec(64, gfx1250::OperandType::OPR_SRC,
                                gfx1250::OpSelSrc::OPR_SRC_EXEC_LO);
  gfx1250::Operand gfx1250_ttmp(64, gfx1250::OperandType::OPR_SRC,
                                gfx1250::OpSelSrc::OPR_SRC_TTMP2);
  gfx1250::Operand gfx1250_flat_lo(32, gfx1250::OperandType::OPR_SRC,
                                   gfx1250::OpSelSrc::OPR_SRC_SRC_FLAT_SCRATCH_BASE_LO);
  ASSERT_TRUE(gfx1250_exec.to_register_ref());
  EXPECT_EQ(*gfx1250_exec.to_register_ref(), (RegisterRef{RegClass::EXEC, 0, 2}));
  ASSERT_TRUE(gfx1250_ttmp.to_register_ref());
  EXPECT_EQ(*gfx1250_ttmp.to_register_ref(), (RegisterRef{RegClass::TTMP, 2, 2}));
  ASSERT_TRUE(gfx1250_flat_lo.to_register_ref());
  EXPECT_EQ(*gfx1250_flat_lo.to_register_ref(), (RegisterRef{RegClass::FLAT_SCRATCH, 0, 1}));
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

TEST(CfgAnalysis, Gfx1250ClassifiesImplicitUnreachableStubTerminator) {
  struct Case {
    const char *name;
    std::vector<uint32_t> words;
    bool has_terminator;
    bool has_implicit_terminator;
    bool falls_through_to_undecodable_text;
  };
  const std::array cases = {
      Case{"clang unreachable stub", {0xb9800641u, 1u, 0}, true, true, false},
      Case{"clang unreachable stub with prefetch",
           {0xee174000u, 0x00040000u, 0, 0x7e000000u, 0xb9800641u, 1u, 0},
           true,
           true,
           false},
      Case{"section-final clang unreachable stub", {0xb9800641u, 1u}, true, true, false},
      Case{"ordinary fallthrough",
           {build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250), 0},
           false,
           false,
           true},
      Case{"architectural terminator",
           {build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250), 0},
           true,
           false,
           false},
  };

  for (const auto &test_case : cases) {
    SCOPED_TRACE(test_case.name);
    TestCodeObject co(test_case.words);
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
    ASSERT_NE(decoder, nullptr);
    auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_GFX1250);

    ASSERT_EQ(blocks.size(), 1u);
    EXPECT_EQ(blocks[0]->has_terminator(), test_case.has_terminator);
    EXPECT_EQ(blocks[0]->has_implicit_terminator(), test_case.has_implicit_terminator);
    EXPECT_EQ(blocks[0]->falls_through_to_undecodable_text(),
              test_case.falls_through_to_undecodable_text);
  }
}

TEST(CfgAnalysis, DirectCallToImplicitNonreturningTargetDropsFallthrough) {
  constexpr uint16_t kReturnSreg = 30;
  std::vector<uint32_t> words = {
      rocjitsu::build_s_call_b64(kReturnSreg, 1, ROCJITSU_CODE_ARCH_GFX1250),
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250), // 0x04 continuation.
      0xb9800641u,
      1u,
      0, // 0x08 clang unreachable-stub target followed by padding.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_GFX1250);

  auto *caller = block_starting_at(blocks, 0);
  auto *continuation = block_starting_at(blocks, 4);
  auto *target = block_starting_at(blocks, 8);
  ASSERT_NE(caller, nullptr);
  ASSERT_NE(continuation, nullptr);
  ASSERT_NE(target, nullptr);

  EXPECT_TRUE(caller->call_edges().empty());
  EXPECT_TRUE(has_successor_start(*caller, target->start_offset()));
  EXPECT_FALSE(has_successor_start(*caller, continuation->start_offset()));
  EXPECT_FALSE(has_predecessor(*continuation, caller));
}

TEST(CfgAnalysis, PreviousInstructionReturnsPrecedingInstructionInBlock) {
  auto blocks = build_test_blocks({TestOpcode::Nop, TestOpcode::UseSgpr4, TestOpcode::End});

  ASSERT_EQ(blocks.size(), 1u);
  auto instruction = blocks[0]->instructions().begin();
  ASSERT_NE(instruction, blocks[0]->instructions().end());
  const Instruction *first = &*instruction;
  ++instruction;
  ASSERT_NE(instruction, blocks[0]->instructions().end());
  const Instruction *second = &*instruction;

  EXPECT_EQ(first->previous_instruction(), nullptr);
  EXPECT_EQ(second->previous_instruction(), first);
}

TEST(CfgAnalysis, PreviousInstructionIsNullAtBranchTargetBlockEntry) {
  auto blocks = build_test_blocks(
      {TestOpcode::CBranchToElse, TestOpcode::Nop, TestOpcode::UseSgpr4, TestOpcode::End});

  BasicBlock *target = block_starting_at(blocks, 8);
  ASSERT_NE(target, nullptr);
  ASSERT_NE(target->instructions().begin(), target->instructions().end());
  const Instruction &entry = *target->instructions().begin();
  EXPECT_EQ(entry.previous_instruction(), nullptr);
}

TEST(CfgAnalysis, NextInstructionReturnsFollowingInstructionInBlock) {
  auto blocks = build_test_blocks({TestOpcode::Nop, TestOpcode::UseSgpr4, TestOpcode::End});

  ASSERT_EQ(blocks.size(), 1u);
  auto instruction = blocks[0]->instructions().begin();
  ASSERT_NE(instruction, blocks[0]->instructions().end());
  const Instruction *first = &*instruction;
  ++instruction;
  ASSERT_NE(instruction, blocks[0]->instructions().end());
  const Instruction *second = &*instruction;

  EXPECT_EQ(first->next_instruction(), second);
}

TEST(CfgAnalysis, NextInstructionIsNullAtBlockTerminator) {
  auto blocks = build_test_blocks(
      {TestOpcode::CBranchToElse, TestOpcode::Nop, TestOpcode::UseSgpr4, TestOpcode::End});

  ASSERT_FALSE(blocks.empty());
  const Instruction *terminator = blocks[0]->terminator();
  ASSERT_NE(terminator, nullptr);
  EXPECT_EQ(terminator->next_instruction(), nullptr);
}

TEST(CfgAnalysis, StandaloneInstructionHasNoDecodedNeighbors) {
  constexpr uint32_t kNop = 0xbf800000u;
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> instruction(decoder->decode(&kNop));

  ASSERT_NE(instruction, nullptr);
  EXPECT_EQ(instruction->previous_instruction(), nullptr);
  EXPECT_EQ(instruction->next_instruction(), nullptr);
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

TEST(CfgAnalysis, IndirectRecoveryPrefilterAdmitsSetPcConsumer) {
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

  ASSERT_EQ(consumer->static_indirect_call_fixups().size(), 1u)
      << "setpc consumer must pass the indirect-recovery prefilter";
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

TEST(CfgAnalysis, PcBuilderWithoutConsumerProducesNoRecoveredEdge) {
  constexpr uint16_t kPcSreg = 8;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;

  std::vector<uint32_t> words = {
      pack_sop1(0x1c, kPcSreg, 0),                         // s_getpc_b64.
      pack_sop2(0, kPcSreg, kPcSreg, kLiteralOperand),     // s_add_u32.
      4,                                                   // Target delta.
      pack_sop2(4, kPcSreg + 1, kPcSreg + 1, kInlineInt0), // s_addc_u32.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

  ASSERT_EQ(blocks.size(), 1u);
  EXPECT_TRUE(blocks[0]->static_indirect_call_fixups().empty());
}

TEST(CfgAnalysis, RecoversMultipleSgprPairsFromOneBlockEntry) {
  constexpr uint16_t kFirstPcSreg = 8;
  constexpr uint16_t kSecondPcSreg = 20;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;

  // Both PC builders reach both successor blocks. The two pending consumers
  // exercise lookup of distinct keys in the same sorted block-entry fact set.
  std::vector<uint32_t> words = {
      pack_sop1(0x1c, kFirstPcSreg, 0),                                // 0x00: first getpc.
      pack_sop2(0, kFirstPcSreg, kFirstPcSreg, kLiteralOperand),       // 0x04: first add.
      44,                                                              // 0x08: -> 0x30.
      pack_sop2(4, kFirstPcSreg + 1, kFirstPcSreg + 1, kInlineInt0),   // 0x0c.
      pack_sop1(0x1c, kSecondPcSreg, 0),                               // 0x10: second getpc.
      pack_sop2(0, kSecondPcSreg, kSecondPcSreg, kLiteralOperand),     // 0x14: second add.
      32,                                                              // 0x18: -> 0x34.
      pack_sop2(4, kSecondPcSreg + 1, kSecondPcSreg + 1, kInlineInt0), // 0x1c.
      pack_sopp(5, 1),                                                 // 0x20: -> 0x28.
      pack_sop1(0x1d, 0, kFirstPcSreg),                                // 0x24: first consumer.
      pack_sop1(0x1d, 0, kSecondPcSreg),                               // 0x28: second consumer.
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),                        // 0x2c.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),                        // 0x30: first target.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),                        // 0x34: second target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

  auto *first_consumer = block_starting_at(blocks, 36);
  auto *second_consumer = block_starting_at(blocks, 40);
  ASSERT_NE(first_consumer, nullptr);
  ASSERT_NE(second_consumer, nullptr);
  ASSERT_EQ(first_consumer->static_indirect_call_fixups().size(), 1u);
  EXPECT_EQ(first_consumer->static_indirect_call_fixups()[0].source_target_offset, 48u);
  ASSERT_EQ(second_consumer->static_indirect_call_fixups().size(), 1u);
  EXPECT_EQ(second_consumer->static_indirect_call_fixups()[0].source_target_offset, 52u);
}

TEST(CfgAnalysis, OutOfRangeIndirectConsumersRemainUnresolved) {
  constexpr uint16_t kOutOfRangeSelector = 106;
  const std::array<uint32_t, 2> consumers = {
      pack_sop1(0x1d, 0, kOutOfRangeSelector),  // s_setpc_b64 vcc.
      pack_sop1(0x1e, 30, kOutOfRangeSelector), // s_swappc_b64 s[30:31], vcc.
  };

  for (uint32_t consumer : consumers) {
    TestCodeObject co(std::vector<uint32_t>{consumer});
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    ASSERT_NE(decoder, nullptr);
    auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

    ASSERT_EQ(blocks.size(), 1u);
    EXPECT_TRUE(blocks[0]->static_indirect_call_fixups().empty());
  }
}

TEST(CfgAnalysis, IgnoresUnconsumedPairWhileRecoveringPendingConsumer) {
  constexpr uint16_t kUnusedPcSreg = 8;
  constexpr uint16_t kUsedPcSreg = 20;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;

  // Both builders reach the consumer block, but only s[20:21] is consumed.
  // The relevance filter must drop the s[8:9] transfer without disturbing the
  // pending consumer's fact.
  std::vector<uint32_t> words = {
      pack_sop1(0x1c, kUnusedPcSreg, 0),                           // 0x00: unused getpc.
      pack_sop1(0x1c, kUsedPcSreg, 0),                             // 0x04: used getpc.
      pack_sop2(0, kUsedPcSreg, kUsedPcSreg, kLiteralOperand),     // 0x08: used add.
      20,                                                          // 0x0c: -> 0x1c.
      pack_sop2(4, kUsedPcSreg + 1, kUsedPcSreg + 1, kInlineInt0), // 0x10.
      pack_sop1(0x1d, 0, kUsedPcSreg),                             // 0x14: consumer.
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),                    // 0x18.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),                    // 0x1c: target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  constexpr std::array<uint64_t, 1> extra_leaders{20};
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4, extra_leaders);

  auto *consumer = block_starting_at(blocks, 20);
  ASSERT_NE(consumer, nullptr);
  ASSERT_EQ(consumer->static_indirect_call_fixups().size(), 1u);
  EXPECT_EQ(consumer->static_indirect_call_fixups()[0].source_target_offset, 28u);
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

TEST(CfgAnalysis, MarksUndecodedIndirectCallContinuationIncomplete) {
  auto blocks = build_test_blocks({TestOpcode::IndirectCall});

  ASSERT_EQ(blocks.size(), 1u);
  EXPECT_FALSE(blocks.front()->static_successors_complete());
  EXPECT_EQ(blocks.front()->static_successor_issue(),
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

TEST(KernelScopeAnalysis, OutOfScopeContinuationStillContributesSharedHelperReturnRegisterContext) {
  constexpr uint16_t kFirstReturnSreg = 30;
  constexpr uint16_t kSecondReturnSreg = 32;
  std::vector<uint32_t> words = {
      build_s_call_b64(kFirstReturnSreg, 3),    // 0x00 -> helper at 0x10.
      build_s_call_b64(kSecondReturnSreg, 2),   // 0x04 -> same helper.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4), // 0x08 is another kernel entry.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4), // 0x0c unrelated filler.
      pack_sopp(5, 1),                          // 0x10 -> s32 return at 0x18.
      pack_sop1(0x1d, 0, kFirstReturnSreg),     // 0x14 s30 return.
      pack_sop1(0x1d, 0, kSecondReturnSreg),    // 0x18 s32 return.
  };
  TestCodeObject co(words);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  constexpr std::array<uint64_t, 2> kernel_entries{0, 8};
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
  EXPECT_TRUE(std::ranges::none_of(scope->liveness_edges, [&](const ScopedCfgEdge &edge) {
    return edge.from == second_return && edge.to == second_continuation;
  }));
}

TEST(CfgAnalysis, IncompleteSwappcTargetSetKeepsContinuation) {
  constexpr uint16_t kPcSreg = 8;
  constexpr uint16_t kReturnSreg = 30;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;

  std::vector<uint32_t> words = {
      pack_sopp(5, 5),                                     // 0x00 -> bypass at 0x18.
      pack_sop1(0x1c, kPcSreg, 0),                         // 0x04: s_getpc_b64.
      pack_sop2(0, kPcSreg, kPcSreg, kLiteralOperand),     // 0x08: s_add_u32.
      28,                                                  // 0x0c: target at 0x24.
      pack_sop2(4, kPcSreg + 1, kPcSreg + 1, kInlineInt0), // 0x10: s_addc_u32.
      build_s_branch(1, ROCJITSU_CODE_ARCH_CDNA4),         // 0x14 -> consumer at 0x1c.
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),            // 0x18: unconstrained bypass.
      pack_sop1(0x1e, kReturnSreg, kPcSreg),               // 0x1c: joined swappc consumer.
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),            // 0x20: continuation.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),            // 0x24: known non-returning target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

  auto *consumer = block_starting_at(blocks, 28);
  auto *continuation = block_starting_at(blocks, 32);
  auto *target = block_starting_at(blocks, 36);
  ASSERT_NE(consumer, nullptr);
  ASSERT_NE(continuation, nullptr);
  ASSERT_NE(target, nullptr);
  ASSERT_EQ(consumer->static_indirect_call_fixups().size(), 1u);
  EXPECT_TRUE(consumer->static_indirect_call_fixups()[0].source_incomplete);
  EXPECT_TRUE(consumer->call_edges().empty());
  EXPECT_TRUE(has_successor_start(*consumer, continuation->start_offset()));
  EXPECT_TRUE(has_successor_start(*consumer, target->start_offset()));
  EXPECT_TRUE(has_predecessor(*continuation, consumer));
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

TEST(CfgAnalysis, ReachableRecoveryInvalidatesOmittedTargetAfterDecodedKill) {
  constexpr uint16_t kEdgePcSreg = 8u;
  constexpr uint16_t kConsumerPcSreg = 12u;
  constexpr uint32_t kLiteralOperand = 255u;
  constexpr uint32_t kInlineInt0 = 128u;

  // The first reachable pass recovers the consumer target exhaustively. The
  // earlier recovered edge then decodes a predecessor that overwrites the
  // consumer PC pair before reaching the setpc. Rediscovery emits no fixup at
  // all for that consumer, so the retained target must lose its stale
  // exhaustive proof even though there is no replacement observation to AND.
  const std::vector<uint32_t> words = {
      pack_sop1(0x1c, kEdgePcSreg, 0),                             // 0x00: edge getpc.
      pack_sop2(0, kEdgePcSreg, kEdgePcSreg, kLiteralOperand),     // 0x04: edge low.
      52u,                                                         // 0x08: -> 0x38.
      pack_sop2(4, kEdgePcSreg + 1, kEdgePcSreg + 1, kInlineInt0), // 0x0c: edge high.
      pack_sop1(0x1d, 0, kEdgePcSreg),                             // 0x10: recovered edge.
      pack_sop1(0x1c, kConsumerPcSreg, 0),                         // 0x14: target getpc.
      pack_sop2(0, kConsumerPcSreg, kConsumerPcSreg, kLiteralOperand),
      40u, // 0x1c: -> 0x40.
      pack_sop2(4, kConsumerPcSreg + 1, kConsumerPcSreg + 1, kInlineInt0),
      build_s_branch(2, ROCJITSU_CODE_ARCH_CDNA4), // 0x24 -> consumer at 0x30.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),    // 0x28.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),    // 0x2c.
      pack_sop1(0x1d, 0, kConsumerPcSreg),         // 0x30: joined consumer.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),    // 0x34.
      build_s_mov_b32(kConsumerPcSreg, kInlineInt0,
                      ROCJITSU_CODE_ARCH_CDNA4),    // 0x38: kill tracked pair.
      build_s_branch(-4, ROCJITSU_CODE_ARCH_CDNA4), // 0x3c -> consumer at 0x30.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),     // 0x40: concrete target.
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
    return fixup.source_call_offset == 0x30u && fixup.source_target_offset == 0x40u;
  });
  ASSERT_NE(recovered, fixups.end());
  EXPECT_TRUE(recovered->source_incomplete);
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

TEST(CfgAnalysis, RecoversGfx1250SpecialPairPcBuildersAcrossDependencyWait) {
  constexpr std::array<uint16_t, 2> kSpecialPairSelectors{106u, 108u}; // VCC, TTMP0.
  constexpr uint64_t kTargetOffset = 20u;
  const auto wait = instrumentation::build_s_wait_indirect_pc0(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(wait);

  for (uint16_t pair_lo : kSpecialPairSelectors) {
    std::vector<uint32_t> words = {
        build_s_getpc_b64(pair_lo, ROCJITSU_CODE_ARCH_GFX1250),
    };
    ASSERT_TRUE(append_pc_delta_builder(words, ROCJITSU_CODE_ARCH_GFX1250, pair_lo,
                                        static_cast<int64_t>(kTargetOffset) - 4));
    words.push_back(*wait);
    words.push_back(build_s_setpc_b64(pair_lo, ROCJITSU_CODE_ARCH_GFX1250));
    ASSERT_EQ(words.size() * sizeof(uint32_t), kTargetOffset);
    words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250));

    const std::vector<IndirectCallFixup> fixups =
        discover_test_indirect_fixups(words, ROCJITSU_CODE_ARCH_GFX1250, {},
                                      /*wavefront_size=*/32u);
    ASSERT_EQ(fixups.size(), 1u) << "pair selector " << pair_lo;
    EXPECT_EQ(fixups.front().source_call_offset, 16u);
    EXPECT_EQ(fixups.front().source_target_offset, kTargetOffset);
    EXPECT_EQ(fixups.front().source_call_selector, pair_lo);
    EXPECT_TRUE(fixups.front().source_targets_exhaustive);
  }
}

TEST(CfgAnalysis, RecoversGfx1250SpecialPairLiteral64PcBuilders) {
  constexpr std::array<uint16_t, 2> kSpecialPairSelectors{106u, 108u}; // VCC, TTMP0.
  constexpr uint64_t kTargetOffset = 20u;
  constexpr uint32_t kAddNcU64Literal64Base = 0xA980FE00u;

  for (uint16_t pair_lo : kSpecialPairSelectors) {
    const std::vector<uint32_t> words = {
        build_s_getpc_b64(pair_lo, ROCJITSU_CODE_ARCH_GFX1250),
        kAddNcU64Literal64Base | (static_cast<uint32_t>(pair_lo) << 16) | pair_lo,
        static_cast<uint32_t>(kTargetOffset - sizeof(uint32_t)),
        0u,
        build_s_setpc_b64(pair_lo, ROCJITSU_CODE_ARCH_GFX1250),
        build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250),
    };

    const std::vector<IndirectCallFixup> fixups =
        discover_test_indirect_fixups(words, ROCJITSU_CODE_ARCH_GFX1250, {},
                                      /*wavefront_size=*/32u);
    ASSERT_EQ(fixups.size(), 1u) << "pair selector " << pair_lo;
    EXPECT_EQ(fixups.front().source_call_offset, 16u);
    EXPECT_EQ(fixups.front().source_target_offset, kTargetOffset);
    EXPECT_EQ(fixups.front().source_call_selector, pair_lo);
    EXPECT_TRUE(fixups.front().source_targets_exhaustive);
  }
}

TEST(CfgAnalysis, RejectsSpecialPairPcBuilderAfterUnmodeledWrite) {
  constexpr uint16_t kVccLo = 106u;
  constexpr uint16_t kExecLo = 126u;
  constexpr uint64_t kTargetOffset = 24u;
  std::vector<uint32_t> words = {
      build_s_getpc_b64(kVccLo, ROCJITSU_CODE_ARCH_GFX1250),
  };
  ASSERT_TRUE(append_pc_delta_builder(words, ROCJITSU_CODE_ARCH_GFX1250, kVccLo,
                                      static_cast<int64_t>(kTargetOffset) - 4));
  const auto clobber =
      instrumentation::build_s_mov_b64(kVccLo, kExecLo, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(clobber);
  words.push_back(*clobber);
  const auto wait = instrumentation::build_s_wait_indirect_pc0(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(wait);
  words.push_back(*wait);
  words.push_back(build_s_setpc_b64(kVccLo, ROCJITSU_CODE_ARCH_GFX1250));
  ASSERT_EQ(words.size() * sizeof(uint32_t), kTargetOffset);
  words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250));

  const std::vector<IndirectCallFixup> fixups =
      discover_test_indirect_fixups(words, ROCJITSU_CODE_ARCH_GFX1250, {},
                                    /*wavefront_size=*/32u);
  EXPECT_TRUE(fixups.empty());
}

TEST(CfgAnalysis, RecoversCdna4VccAndAlignedTtmpPcBuilders) {
  constexpr std::array<uint16_t, 2> kSupportedSelectors{106u, 108u};
  for (uint16_t selector : kSupportedSelectors) {
    std::vector<uint32_t> words = {
        build_s_getpc_b64(selector, ROCJITSU_CODE_ARCH_CDNA4),
    };
    ASSERT_TRUE(append_pc_delta_builder(words, ROCJITSU_CODE_ARCH_CDNA4, selector, /*delta=*/16));
    words.push_back(build_s_setpc_b64(selector, ROCJITSU_CODE_ARCH_CDNA4));
    words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4));

    const auto fixups =
        discover_test_indirect_fixups(words, ROCJITSU_CODE_ARCH_CDNA4, {}, /*wavefront_size=*/64u);
    ASSERT_EQ(fixups.size(), 1u) << "pair selector " << selector;
    EXPECT_EQ(fixups.front().source_call_selector, selector);
    EXPECT_EQ(fixups.front().source_target_offset, 20u);
  }
}

TEST(CfgAnalysis, RejectsUnsupportedSpecialPcCarriers) {
  // FLAT_SCRATCH and EXEC have independent architectural semantics. TTMP1 is
  // not the low half of an aligned TTMP pair.
  constexpr std::array<uint16_t, 3> kUnsupportedSelectors{102u, 109u, 126u};
  for (uint16_t selector : kUnsupportedSelectors) {
    std::vector<uint32_t> words = {
        build_s_getpc_b64(selector, ROCJITSU_CODE_ARCH_CDNA4),
    };
    ASSERT_TRUE(append_pc_delta_builder(words, ROCJITSU_CODE_ARCH_CDNA4, selector, /*delta=*/16));
    words.push_back(build_s_setpc_b64(selector, ROCJITSU_CODE_ARCH_CDNA4));
    words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4));

    const auto fixups =
        discover_test_indirect_fixups(words, ROCJITSU_CODE_ARCH_CDNA4, {}, /*wavefront_size=*/64u);
    EXPECT_TRUE(fixups.empty()) << "pair selector " << selector;
  }
}

TEST(CfgAnalysis, RejectsSpecialPcBuilderWithMismatchedConsumer) {
  constexpr uint16_t kVccLo = 106u;
  constexpr uint16_t kTtmp0 = 108u;
  std::vector<uint32_t> words = {
      build_s_getpc_b64(kVccLo, ROCJITSU_CODE_ARCH_GFX1250),
  };
  ASSERT_TRUE(append_pc_delta_builder(words, ROCJITSU_CODE_ARCH_GFX1250, kVccLo, /*delta=*/16));
  words.push_back(build_s_setpc_b64(kTtmp0, ROCJITSU_CODE_ARCH_GFX1250));
  words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250));

  const auto fixups = discover_test_indirect_fixups(words, ROCJITSU_CODE_ARCH_GFX1250, {},
                                                    /*wavefront_size=*/32u);
  EXPECT_TRUE(fixups.empty());
}

TEST(CfgAnalysis, RejectsSpecialPcBuilderSplitAcrossBlocks) {
  constexpr uint16_t kVccLo = 106u;
  std::vector<uint32_t> words = {
      build_s_getpc_b64(kVccLo, ROCJITSU_CODE_ARCH_GFX1250),
  };
  ASSERT_TRUE(append_pc_delta_builder(words, ROCJITSU_CODE_ARCH_GFX1250, kVccLo, /*delta=*/20));
  words.push_back(build_s_branch(0, ROCJITSU_CODE_ARCH_GFX1250));
  words.push_back(build_s_setpc_b64(kVccLo, ROCJITSU_CODE_ARCH_GFX1250));
  words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250));

  const auto fixups = discover_test_indirect_fixups(words, ROCJITSU_CODE_ARCH_GFX1250, {},
                                                    /*wavefront_size=*/32u);
  EXPECT_TRUE(fixups.empty());
}

TEST(CfgAnalysis, RejectsSpecialPcTempDeltaAliasingCarrier) {
  constexpr uint16_t kVccLo = 106u;
  constexpr uint32_t kLiteralOperand = 255u;
  constexpr uint32_t kInlineInt0 = 128u;
  constexpr uint32_t kInlineInt4 = 132u;
  const std::vector<uint32_t> words = {
      build_s_getpc_b64(kVccLo, ROCJITSU_CODE_ARCH_GFX1250),
      pack_sop2(2, kVccLo, kLiteralOperand, kInlineInt4),
      16u,
      pack_sop2(0, kVccLo, kVccLo, kVccLo),
      pack_sop2(4, kVccLo + 1, kVccLo + 1, kInlineInt0),
      build_s_setpc_b64(kVccLo, ROCJITSU_CODE_ARCH_GFX1250),
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250),
  };

  const auto fixups = discover_test_indirect_fixups(words, ROCJITSU_CODE_ARCH_GFX1250, {},
                                                    /*wavefront_size=*/32u);
  EXPECT_TRUE(fixups.empty());
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

TEST(CfgAnalysis, IncompleteRecoveredSetpcInCalleeKeepsOuterContinuation) {
  constexpr uint16_t kPcSreg = 8;
  constexpr uint16_t kOuterReturnSreg = 30;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;

  std::vector<uint32_t> words = {
      build_s_call_b64(kOuterReturnSreg, 1),               // 0x00 -> callee at 0x08.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),            // 0x04 live continuation.
      pack_sopp(5, 5),                                     // 0x08 -> bypass at 0x20.
      pack_sop1(0x1c, kPcSreg, 0),                         // 0x0c: s_getpc_b64.
      pack_sop2(0, kPcSreg, kPcSreg, kLiteralOperand),     // 0x10: s_add_u32.
      28,                                                  // 0x14: target at 0x2c.
      pack_sop2(4, kPcSreg + 1, kPcSreg + 1, kInlineInt0), // 0x18: s_addc_u32.
      build_s_branch(1, ROCJITSU_CODE_ARCH_CDNA4),         // 0x1c -> consumer at 0x24.
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),            // 0x20: unconstrained bypass.
      pack_sop1(0x1d, 0, kPcSreg),                         // 0x24: joined setpc.
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),            // 0x28: not a target.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),            // 0x2c: known target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

  auto *caller = block_starting_at(blocks, 0);
  auto *continuation = block_starting_at(blocks, 4);
  auto *callee = block_starting_at(blocks, 8);
  auto *consumer = block_starting_at(blocks, 36);
  ASSERT_NE(caller, nullptr);
  ASSERT_NE(continuation, nullptr);
  ASSERT_NE(callee, nullptr);
  ASSERT_NE(consumer, nullptr);
  ASSERT_EQ(consumer->static_indirect_call_fixups().size(), 1u);
  EXPECT_TRUE(consumer->static_indirect_call_fixups()[0].source_incomplete);

  EXPECT_TRUE(caller->call_edges().empty());
  EXPECT_TRUE(has_successor_start(*caller, callee->start_offset()));
  EXPECT_TRUE(has_successor_start(*caller, continuation->start_offset()));
  EXPECT_TRUE(has_predecessor(*continuation, caller));
}

TEST(CfgAnalysis, ReportsResolvedPcAddressBuilderForEveryProducer) {
  constexpr uint16_t kPcSreg = 8;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;

  // Recovered consumers are only one use of a getpc builder. DBT also needs the
  // producer itself so it can prove a whole kernel scope holds no unrelocated
  // PC-derived value, so every builder is reported with the exact byte range
  // whose delta relocation may rewrite.
  std::vector<uint32_t> words = {
      pack_sop1(0x1c, kPcSreg, 0),                         // 0x00: s_getpc_b64.
      pack_sop2(0, kPcSreg, kPcSreg, kLiteralOperand),     // 0x04: s_add_u32.
      16,                                                  // 0x08: 0x04 + 16 = 0x14.
      pack_sop2(4, kPcSreg + 1, kPcSreg + 1, kInlineInt0), // 0x0c: s_addc_u32.
      pack_sop1(0x1d, 0, kPcSreg),                         // 0x10: consumer setpc.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),            // 0x14: target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

  auto *builder_block = block_starting_at(blocks, 0);
  ASSERT_NE(builder_block, nullptr);
  ASSERT_EQ(builder_block->static_pc_address_builders().size(), 1u);
  const auto &builder = builder_block->static_pc_address_builders()[0];
  EXPECT_TRUE(builder.resolved);
  EXPECT_TRUE(builder.contiguous);
  EXPECT_EQ(builder.source_getpc_offset, 0u);
  EXPECT_EQ(builder.source_recovery_begin_offset, 4u);
  EXPECT_EQ(builder.source_recovery_end_offset, 16u);
  EXPECT_EQ(builder.source_target_offset, 20);
  EXPECT_EQ(builder.source_sreg, kPcSreg);
}

TEST(CfgAnalysis, PcAddressBuilderWithGapInstructionIsReportedNonContiguous) {
  constexpr uint16_t kPcSreg = 8;
  constexpr uint16_t kUnrelatedSreg = 20;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;

  // An unrelated s_mov_b32 sits between the low add and the high carry. The pass
  // still tracks the pair across it (the move writes s20, not the pair), so the
  // builder's recorded value is known and its recovery range spans the move. The
  // relocation patcher NOPs that whole range, so rewriting it would erase the
  // move. The producer must therefore be reported non-contiguous even though its
  // final value resolved, so the whole-scope proof declines to rewrite it.
  std::vector<uint32_t> words = {
      pack_sop1(0x1c, kPcSreg, 0),                     // 0x00: s_getpc_b64.
      pack_sop2(0, kPcSreg, kPcSreg, kLiteralOperand), // 0x04: s_add_u32.
      20,                                              // 0x08: 0x04 + 20 = 0x18.
      build_s_mov_b32(kUnrelatedSreg, 0,
                      ROCJITSU_CODE_ARCH_CDNA4),           // 0x0c: unrelated write, in range.
      pack_sop2(4, kPcSreg + 1, kPcSreg + 1, kInlineInt0), // 0x10: s_addc_u32.
      pack_sop1(0x1d, 0, kPcSreg),                         // 0x14: consumer setpc.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),            // 0x18: target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

  auto *builder_block = block_starting_at(blocks, 0);
  ASSERT_NE(builder_block, nullptr);
  ASSERT_EQ(builder_block->static_pc_address_builders().size(), 1u);
  const auto &builder = builder_block->static_pc_address_builders()[0];
  EXPECT_TRUE(builder.resolved);
  EXPECT_FALSE(builder.contiguous)
      << "a builder range spanning an unrelated instruction must be non-contiguous";
}

TEST(CfgAnalysis, UnfollowedPcAddressBuilderIsReportedUnresolved) {
  constexpr uint16_t kPcSreg = 8;
  constexpr uint16_t kAddendSreg = 12;

  // The low-half add takes a register addend the pass does not model, so the
  // pair's final value is unknown. The producer still exists and still yields a
  // PC-derived value at run time, so it must be reported as unresolved rather
  // than omitted: omitting it would let a caller conclude the scope has no
  // unrelocatable PC producer.
  std::vector<uint32_t> words = {
      pack_sop1(0x1c, kPcSreg, 0),                 // 0x00: s_getpc_b64.
      pack_sop2(0, kPcSreg, kPcSreg, kAddendSreg), // 0x04: s_add_u32 with register addend.
      pack_sop1(0x1d, 0, kPcSreg),                 // 0x08: consumer setpc.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),    // 0x0c.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

  auto *builder_block = block_starting_at(blocks, 0);
  ASSERT_NE(builder_block, nullptr);
  ASSERT_EQ(builder_block->static_pc_address_builders().size(), 1u);
  EXPECT_EQ(builder_block->static_pc_address_builders()[0].source_getpc_offset, 0u);
  EXPECT_FALSE(builder_block->static_pc_address_builders()[0].resolved)
      << "a producer the pass cannot follow must not be reported as relocatable";
  EXPECT_TRUE(builder_block->static_indirect_call_fixups().empty());
}

TEST(CfgAnalysis, DominatedPcBuilderRemainsCompleteAcrossCallLoopBackedge) {
  constexpr uint16_t kPcSreg = 8;
  constexpr uint16_t kReturnSreg = 30;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;

  // RCCL builds one helper address before a loop and reuses it for every
  // s_swappc iteration. The consumer has both the builder entry edge and a loop
  // backedge. During the first dataflow visit the backedge is still unreachable
  // (BOTTOM), not an independent path with unconstrained s[8:9]. Once the loop
  // becomes reachable it carries the same preserved builder back to the call.
  std::vector<uint32_t> words = {
      pack_sop1(0x1c, kPcSreg, 0),                         // 0x00: s_getpc_b64.
      pack_sop2(0, kPcSreg, kPcSreg, kLiteralOperand),     // 0x04: s_add_u32.
      32,                                                  // 0x08: 0x04 + 32 = target 0x24.
      pack_sop2(4, kPcSreg + 1, kPcSreg + 1, kInlineInt0), // 0x0c: s_addc_u32.
      build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA4),         // 0x10 -> call at 0x14.
      pack_sop1(0x1e, kReturnSreg, kPcSreg),               // 0x14: s_swappc_b64.
      build_s_branch(-2, ROCJITSU_CODE_ARCH_CDNA4),        // 0x18 -> call at 0x14.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),            // 0x1c: not a target.
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),            // 0x20: padding.
      pack_sop1(0x1d, 0, kReturnSreg),                     // 0x24: helper return.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

  auto *consumer = block_starting_at(blocks, 20);
  ASSERT_NE(consumer, nullptr);
  ASSERT_EQ(consumer->static_indirect_call_fixups().size(), 1u);
  const auto &fixup = consumer->static_indirect_call_fixups()[0];
  EXPECT_EQ(fixup.source_target_offset, 36u);
  EXPECT_FALSE(fixup.source_incomplete)
      << "an unreachable initial backedge must not poison a dominated PC builder";
}

TEST(CfgAnalysis, SeedsTextEntryWithLoopBackedgeForCrossBlockPcBuilder) {
  constexpr uint16_t kPcSreg = 8;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;

  // The two-block loop gives block zero a structural predecessor:
  //
  //   entry/builder -> loop latch --backedge--> entry/builder
  //                         |
  //                         +--fallthrough--> setpc consumer
  //
  // Block zero is still the external text entry and must seed reachability. Its
  // local SET transfer overwrites the unconstrained external value before the
  // cross-block consumer, so the recovered target remains complete.
  std::vector<uint32_t> words = {
      pack_sop1(0x1c, kPcSreg, 0),                         // 0x00: s_getpc_b64.
      pack_sop2(0, kPcSreg, kPcSreg, kLiteralOperand),     // 0x04: s_add_u32.
      24,                                                  // 0x08: 0x04 + 24 = 0x1c.
      pack_sop2(4, kPcSreg + 1, kPcSreg + 1, kInlineInt0), // 0x0c: s_addc_u32.
      build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA4),         // 0x10 -> latch at 0x14.
      pack_sopp(5, static_cast<uint16_t>(-6)),             // 0x14 -> entry at 0x00.
      pack_sop1(0x1d, 0, kPcSreg),                         // 0x18: cross-block consumer.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),            // 0x1c: target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

  auto *consumer = block_starting_at(blocks, 24);
  ASSERT_NE(consumer, nullptr);
  ASSERT_EQ(consumer->static_indirect_call_fixups().size(), 1u);
  EXPECT_EQ(consumer->static_indirect_call_fixups()[0].source_target_offset, 28u);
  EXPECT_FALSE(consumer->static_indirect_call_fixups()[0].source_incomplete);
}

TEST(CfgAnalysis, ExplicitKernelEntryMakesIncomingPcBuilderIncomplete) {
  constexpr uint16_t kPcSreg = 8;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;

  // Entry A builds a static target and branches into entry B. B is also a
  // separately launchable kernel, so its externally supplied s[8:9] value is
  // unconstrained and must participate in the join with A's concrete builder.
  std::vector<uint32_t> words = {
      pack_sop1(0x1c, kPcSreg, 0),                         // 0x00: s_getpc_b64.
      pack_sop2(0, kPcSreg, kPcSreg, kLiteralOperand),     // 0x04: s_add_u32.
      24,                                                  // 0x08: 0x04 + 24 = 0x1c.
      pack_sop2(4, kPcSreg + 1, kPcSreg + 1, kInlineInt0), // 0x0c: s_addc_u32.
      build_s_branch(1, ROCJITSU_CODE_ARCH_CDNA4),         // 0x10 -> entry B at 0x18.
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),            // 0x14: skipped.
      pack_sop1(0x1d, 0, kPcSreg),                         // 0x18: entry B setpc.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),            // 0x1c: A's target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  constexpr std::array<uint64_t, 1> extra_leaders{24};
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4, extra_leaders);

  auto *consumer = block_starting_at(blocks, 24);
  ASSERT_NE(consumer, nullptr);
  ASSERT_EQ(consumer->static_indirect_call_fixups().size(), 1u);
  EXPECT_EQ(consumer->static_indirect_call_fixups()[0].source_target_offset, 28u);
  EXPECT_TRUE(consumer->static_indirect_call_fixups()[0].source_incomplete)
      << "an independently launchable entry must include unconstrained external SGPR state";
}

TEST(CfgAnalysis, RocrAbortTrapStopsTemporaryPcBuilderCfg) {
  constexpr uint16_t kPcSreg = 8;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;

  // The post-trap setpc is an independent entry (for example, another kernel or
  // device function). The temporary discovery CFG must apply the same ROCr
  // non-returning trap-2 rule as final BasicBlock construction; otherwise the
  // pre-trap builder spuriously flows into this consumer.
  std::vector<uint32_t> words = {
      pack_sop1(0x1c, kPcSreg, 0),                         // 0x00: s_getpc_b64.
      pack_sop2(0, kPcSreg, kPcSreg, kLiteralOperand),     // 0x04: s_add_u32.
      20,                                                  // 0x08: would target 0x18.
      pack_sop2(4, kPcSreg + 1, kPcSreg + 1, kInlineInt0), // 0x0c: s_addc_u32.
      build_s_trap(ROCJITSU_CODE_ARCH_CDNA4, 2),           // 0x10: abort terminator.
      pack_sop1(0x1d, 0, kPcSreg),                         // 0x14: independent consumer.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),            // 0x18: would-be target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  constexpr std::array<uint64_t, 1> extra_leaders{20};
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4, extra_leaders);

  auto *consumer = block_starting_at(blocks, 20);
  ASSERT_NE(consumer, nullptr);
  EXPECT_TRUE(consumer->static_indirect_call_fixups().empty());
}

TEST(CfgAnalysis, UnreachablePostRocrAbortBlockDoesNotPoisonPcBuilder) {
  constexpr uint16_t kPcSreg = 8;
  constexpr uint16_t kReturnSreg = 30;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;

  // Model the generated complex-math kernels from the offline corpus:
  //
  //   builder --conditional--------------------------> call
  //                 |
  //                 +--> s_trap 2 -X-> dead branch --^
  //
  // ROCr's trap 2 aborts instead of returning, so the block after it has no
  // predecessor. It is not an implicit external entry: DBT supplies every
  // hardware-visible kernel entry explicitly, builds a reachable scope from
  // each one, and duplicates shared reachable blocks per kernel scope. Treating
  // the dead block as an external root would invent an unconstrained s[8:9]
  // path into the call and make this otherwise dominated builder incomplete.
  std::vector<uint32_t> words = {
      pack_sop1(0x1c, kPcSreg, 0),                         // 0x00: s_getpc_b64.
      pack_sop2(0, kPcSreg, kPcSreg, kLiteralOperand),     // 0x04: s_add_u32.
      40,                                                  // 0x08: 0x04 + 40 = helper at 0x2c.
      pack_sop2(4, kPcSreg + 1, kPcSreg + 1, kInlineInt0), // 0x0c: s_addc_u32.
      pack_sopp(5, 3),                                     // 0x10: s_cbranch_scc0 -> 0x20.
      build_s_trap(ROCJITSU_CODE_ARCH_CDNA4, 2),           // 0x14: abort terminator.
      build_s_branch(1, ROCJITSU_CODE_ARCH_CDNA4),         // 0x18: dead edge -> 0x20.
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),            // 0x1c: dead padding.
      pack_sop1(0x1e, kReturnSreg, kPcSreg),               // 0x20: s_swappc_b64.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),            // 0x24: call continuation.
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),            // 0x28: padding.
      pack_sop1(0x1d, 0, kReturnSreg),                     // 0x2c: helper return.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4, {},
                                  ExternalEntryPolicy::ExplicitOnly);

  auto *consumer = block_starting_at(blocks, 32);
  ASSERT_NE(consumer, nullptr);
  ASSERT_EQ(consumer->static_indirect_call_fixups().size(), 1u);
  const auto &fixup = consumer->static_indirect_call_fixups()[0];
  EXPECT_EQ(fixup.source_target_offset, 44u);
  EXPECT_FALSE(fixup.source_incomplete)
      << "unreachable code after a non-returning trap must remain dataflow BOTTOM";
}

TEST(CfgAnalysis, DefaultEntryPolicyRecoversPredecessorlessFunction) {
  constexpr uint16_t kPcSreg = 8;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;

  // Generic BasicBlock::build callers may not know every function entry in a
  // shared .text section. Preserve the conservative default that treats the
  // second function as an inferred external entry, allowing its cross-block
  // PC builder to reach the setpc consumer.
  std::vector<uint32_t> words = {
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),            // 0x00: first function.
      pack_sop1(0x1c, kPcSreg, 0),                         // 0x04: second entry, s_getpc_b64.
      pack_sop2(0, kPcSreg, kPcSreg, kLiteralOperand),     // 0x08: s_add_u32.
      24,                                                  // 0x0c: 0x08 + 24 = target 0x20.
      pack_sop2(4, kPcSreg + 1, kPcSreg + 1, kInlineInt0), // 0x10: s_addc_u32.
      build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA4),         // 0x14: -> consumer at 0x18.
      pack_sop1(0x1d, 0, kPcSreg),                         // 0x18: s_setpc_b64.
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),            // 0x1c: padding.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),            // 0x20: target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

  auto *consumer = block_starting_at(blocks, 24);
  ASSERT_NE(consumer, nullptr);
  ASSERT_EQ(consumer->static_indirect_call_fixups().size(), 1u);
  EXPECT_EQ(consumer->static_indirect_call_fixups()[0].source_target_offset, 32u);
  EXPECT_FALSE(consumer->static_indirect_call_fixups()[0].source_incomplete);
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
  EXPECT_TRUE(has_successor_start(*caller, continuation->start_offset()));
  EXPECT_TRUE(has_predecessor(*continuation, caller));
}

TEST(BinaryTranslatorInternal, ScopeRootsRejectRelocationTableCallee) {
  // A returning direct call makes the callee at 0x0c a call-edge target with no
  // ordinary in-scope predecessor, i.e. an external root that the whole-scope
  // stale-PC proof must classify. The caller (0x00) is the kernel entry.
  constexpr uint16_t kReturnSreg = 30;
  std::vector<uint32_t> words = {
      build_s_call_b64(kReturnSreg, 1),         // 0x00 -> callee at 0x08.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4), // 0x04 continuation.
      pack_sop1(0x1d, 0, kReturnSreg),          // 0x08 callee return.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

  auto *caller = block_starting_at(blocks, 0);
  auto *callee = block_starting_at(blocks, 8);
  ASSERT_NE(caller, nullptr);
  ASSERT_NE(callee, nullptr);
  ASSERT_EQ(caller->call_edges().size(), 1u);
  ASSERT_EQ(caller->call_edges()[0].callee, callee);

  const auto scope = block_scope(blocks);
  const std::unordered_set<uint64_t> hardware_entries{caller->start_offset()};

  // With no relocation-table roots, the callee is a getpc-recovered call target
  // and the entry is a hardware root, so the scope is accepted.
  EXPECT_TRUE(rocjitsu::internal::scope_roots_are_entry_state(scope, hardware_entries, {}));

  // Marking the callee a relocation-table dispatch target makes it an
  // unconstrained root: a dispatched callee receives arbitrary caller-supplied
  // SGPR arguments, so the gate must fail closed even though it has a CallEdge.
  const std::unordered_set<uint64_t> table_callees{callee->start_offset()};
  EXPECT_FALSE(
      rocjitsu::internal::scope_roots_are_entry_state(scope, hardware_entries, table_callees));

  // A non-hardware, non-call, non-table external root is also rejected: drop the
  // entry from the hardware set and the caller itself becomes unconstrained.
  EXPECT_FALSE(rocjitsu::internal::scope_roots_are_entry_state(scope, {}, {}));
}

TEST(CfgAnalysis, DirectCallToNonreturningTargetDropsFallthrough) {
  constexpr uint16_t kReturnSreg = 30;

  std::vector<uint32_t> words = {
      build_s_call_b64(kReturnSreg, 1),         // 0x00 -> target at 0x08.
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4), // 0x04 unreachable padding.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4), // 0x08 non-returning target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

  auto *caller = block_starting_at(blocks, 0);
  auto *continuation = block_starting_at(blocks, 4);
  auto *target = block_starting_at(blocks, 8);
  ASSERT_NE(caller, nullptr);
  ASSERT_NE(continuation, nullptr);
  ASSERT_NE(target, nullptr);

  EXPECT_TRUE(caller->call_edges().empty());
  EXPECT_TRUE(has_successor_start(*caller, target->start_offset()));
  EXPECT_FALSE(has_successor_start(*caller, continuation->start_offset()));
  EXPECT_FALSE(has_predecessor(*continuation, caller));
}

TEST(CfgAnalysis, DirectCallWithCopiedReturnPairKeepsFallthrough) {
  constexpr uint16_t kReturnSreg = 30;
  constexpr uint16_t kCopiedReturnSreg = 34;

  std::vector<uint32_t> words = {
      build_s_call_b64(kReturnSreg, 1),             // 0x00 -> callee at 0x08.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),     // 0x04 live continuation.
      pack_sop1(1, kCopiedReturnSreg, kReturnSreg), // 0x08: s_mov_b64.
      pack_sop1(0x1d, 0, kCopiedReturnSreg),        // 0x0c: copied-pair return.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

  auto *caller = block_starting_at(blocks, 0);
  auto *continuation = block_starting_at(blocks, 4);
  auto *callee = block_starting_at(blocks, 8);
  ASSERT_NE(caller, nullptr);
  ASSERT_NE(continuation, nullptr);
  ASSERT_NE(callee, nullptr);

  EXPECT_TRUE(caller->call_edges().empty());
  EXPECT_TRUE(has_successor_start(*caller, continuation->start_offset()));
  EXPECT_TRUE(has_successor_start(*caller, callee->start_offset()));
  EXPECT_TRUE(has_predecessor(*continuation, caller));
}

TEST(CfgAnalysis, DirectCallWithUnrecoveredTailExitKeepsFallthrough) {
  constexpr uint16_t kReturnSreg = 30;
  constexpr uint16_t kUnknownTargetSreg = 0;

  std::vector<uint32_t> words = {
      build_s_call_b64(kReturnSreg, 1),         // 0x00 -> callee at 0x08.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4), // 0x04 live continuation.
      pack_sop1(0x1d, 0, kUnknownTargetSreg),   // 0x08: unrecovered tail exit.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

  auto *caller = block_starting_at(blocks, 0);
  auto *continuation = block_starting_at(blocks, 4);
  auto *callee = block_starting_at(blocks, 8);
  ASSERT_NE(caller, nullptr);
  ASSERT_NE(continuation, nullptr);
  ASSERT_NE(callee, nullptr);

  EXPECT_TRUE(caller->call_edges().empty());
  EXPECT_TRUE(has_successor_start(*caller, continuation->start_offset()));
  EXPECT_TRUE(has_successor_start(*caller, callee->start_offset()));
  EXPECT_TRUE(has_predecessor(*continuation, caller));
}

TEST(CfgAnalysis, DirectCallCrossingScopeBoundaryKeepsFallthrough) {
  constexpr uint16_t kReturnSreg = 30;

  std::vector<uint32_t> words = {
      build_s_call_b64(kReturnSreg, 1),            // 0x00 -> callee at 0x08.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),    // 0x04 live continuation.
      build_s_branch(1, ROCJITSU_CODE_ARCH_CDNA4), // 0x08 -> leader at 0x10.
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),    // 0x0c skipped.
      pack_sop1(0x1d, 0, kReturnSreg),             // 0x10: return across boundary.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  constexpr std::array<uint64_t, 1> extra_leaders{16};
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4, extra_leaders);

  auto *caller = block_starting_at(blocks, 0);
  auto *continuation = block_starting_at(blocks, 4);
  auto *callee = block_starting_at(blocks, 8);
  ASSERT_NE(caller, nullptr);
  ASSERT_NE(continuation, nullptr);
  ASSERT_NE(callee, nullptr);

  EXPECT_TRUE(caller->call_edges().empty());
  EXPECT_TRUE(has_successor_start(*caller, continuation->start_offset()));
  EXPECT_TRUE(has_successor_start(*caller, callee->start_offset()));
  EXPECT_TRUE(has_predecessor(*continuation, caller));
}

TEST(CfgAnalysis, NestedReturningCallMayReturnThroughOuterPair) {
  constexpr uint16_t kOuterReturnSreg = 30;
  constexpr uint16_t kInnerReturnSreg = 28;

  std::vector<uint32_t> words = {
      build_s_call_b64(kOuterReturnSreg, 1),    // 0x00 -> outer callee at 0x08.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4), // 0x04 outer continuation.
      build_s_call_b64(kInnerReturnSreg, 1),    // 0x08 -> inner callee at 0x10.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4), // 0x0c inner continuation.
      pack_sopp(5, 1),                          // 0x10 -> outer return at 0x18.
      pack_sop1(0x1d, 0, kInnerReturnSreg),     // 0x14: normal inner return.
      pack_sop1(0x1d, 0, kOuterReturnSreg),     // 0x18: direct outer return.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

  auto *outer = block_starting_at(blocks, 0);
  auto *outer_continuation = block_starting_at(blocks, 4);
  ASSERT_NE(outer, nullptr);
  ASSERT_NE(outer_continuation, nullptr);

  ASSERT_EQ(outer->call_edges().size(), 1u);
  EXPECT_EQ(outer->call_edges()[0].continuation, outer_continuation);
  EXPECT_TRUE(has_successor_start(*outer, outer_continuation->start_offset()));
  EXPECT_TRUE(has_predecessor(*outer_continuation, outer));
}

TEST(CfgAnalysis, NestedNonreturningCallsDropBothFallthroughs) {
  constexpr uint16_t kOuterReturnSreg = 30;
  constexpr uint16_t kInnerReturnSreg = 28;

  std::vector<uint32_t> words = {
      build_s_call_b64(kOuterReturnSreg, 1),    // 0x00 -> outer callee at 0x08.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4), // 0x04 outer continuation.
      build_s_call_b64(kInnerReturnSreg, 1),    // 0x08 -> inner target at 0x10.
      pack_sop1(0x1d, 0, kOuterReturnSreg),     // 0x0c dead inner continuation.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4), // 0x10 non-returning target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

  auto *outer = block_starting_at(blocks, 0);
  auto *outer_continuation = block_starting_at(blocks, 4);
  auto *inner = block_starting_at(blocks, 8);
  auto *inner_continuation = block_starting_at(blocks, 12);
  auto *target = block_starting_at(blocks, 16);
  ASSERT_NE(outer, nullptr);
  ASSERT_NE(outer_continuation, nullptr);
  ASSERT_NE(inner, nullptr);
  ASSERT_NE(inner_continuation, nullptr);
  ASSERT_NE(target, nullptr);

  EXPECT_TRUE(outer->call_edges().empty());
  EXPECT_TRUE(inner->call_edges().empty());
  EXPECT_TRUE(has_successor_start(*outer, inner->start_offset()));
  EXPECT_FALSE(has_successor_start(*outer, outer_continuation->start_offset()));
  EXPECT_TRUE(has_successor_start(*inner, target->start_offset()));
  EXPECT_FALSE(has_successor_start(*inner, inner_continuation->start_offset()));
}

TEST(CfgAnalysis, CyclicCallGraphKeepsConservativeFallthrough) {
  constexpr uint16_t kOuterReturnSreg = 30;
  constexpr uint16_t kRecursiveReturnSreg = 28;

  std::vector<uint32_t> words = {
      build_s_call_b64(kOuterReturnSreg, 1),      // 0x00 -> callee at 0x08.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),   // 0x04 outer continuation.
      build_s_call_b64(kRecursiveReturnSreg, -1), // 0x08 -> itself.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),   // 0x0c recursive continuation.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

  auto *caller = block_starting_at(blocks, 0);
  auto *continuation = block_starting_at(blocks, 4);
  ASSERT_NE(caller, nullptr);
  ASSERT_NE(continuation, nullptr);

  EXPECT_TRUE(has_successor_start(*caller, continuation->start_offset()));
  EXPECT_TRUE(has_predecessor(*continuation, caller));
}

TEST(CfgAnalysis, CallToInfiniteLoopDropsFallthrough) {
  constexpr uint16_t kReturnSreg = 30;

  std::vector<uint32_t> words = {
      build_s_call_b64(kReturnSreg, 1),             // 0x00 -> callee at 0x08.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),     // 0x04 dead continuation.
      build_s_branch(-1, ROCJITSU_CODE_ARCH_CDNA4), // 0x08 -> itself.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

  auto *caller = block_starting_at(blocks, 0);
  auto *continuation = block_starting_at(blocks, 4);
  auto *callee = block_starting_at(blocks, 8);
  ASSERT_NE(caller, nullptr);
  ASSERT_NE(continuation, nullptr);
  ASSERT_NE(callee, nullptr);

  EXPECT_TRUE(has_successor_start(*caller, callee->start_offset()));
  EXPECT_FALSE(has_successor_start(*caller, continuation->start_offset()));
  EXPECT_FALSE(has_predecessor(*continuation, caller));
}

TEST(CfgAnalysis, ZeroDeltaCallToNonreturningTargetKeepsSingleEdge) {
  constexpr uint16_t kReturnSreg = 30;

  std::vector<uint32_t> words = {
      build_s_call_b64(kReturnSreg, 0),         // 0x00 -> target/continuation at 0x04.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4), // 0x04 non-returning target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

  auto *caller = block_starting_at(blocks, 0);
  auto *target = block_starting_at(blocks, 4);
  ASSERT_NE(caller, nullptr);
  ASSERT_NE(target, nullptr);

  ASSERT_EQ(caller->successors().size(), 1u);
  EXPECT_EQ(caller->successors()[0], target);
  ASSERT_EQ(target->predecessors().size(), 1u);
  EXPECT_EQ(target->predecessors()[0], caller);
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

TEST(CfgAnalysis, RecoveredSwappcToNonreturningTargetDropsFallthrough) {
  constexpr uint16_t kPcSreg = 8;
  constexpr uint16_t kReturnSreg = 30;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;

  std::vector<uint32_t> words = {
      pack_sop1(0x1c, kPcSreg, 0),                         // 0x00: s_getpc_b64.
      pack_sop2(0, kPcSreg, kPcSreg, kLiteralOperand),     // 0x04: s_add_u32.
      20,                                                  // 0x08: target at 0x18.
      pack_sop2(4, kPcSreg + 1, kPcSreg + 1, kInlineInt0), // 0x0c: s_addc_u32.
      pack_sop1(0x1e, kReturnSreg, kPcSreg),               // 0x10: s_swappc_b64.
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),            // 0x14: dead continuation.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),            // 0x18: terminal target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

  auto *consumer = block_starting_at(blocks, 16);
  auto *continuation = block_starting_at(blocks, 20);
  auto *target = block_starting_at(blocks, 24);
  ASSERT_NE(consumer, nullptr);
  ASSERT_NE(continuation, nullptr);
  ASSERT_NE(target, nullptr);

  ASSERT_EQ(consumer->static_indirect_call_fixups().size(), 1u);
  EXPECT_TRUE(consumer->call_edges().empty());
  EXPECT_TRUE(has_successor_start(*consumer, target->start_offset()));
  EXPECT_FALSE(has_successor_start(*consumer, continuation->start_offset()));
  EXPECT_FALSE(has_predecessor(*continuation, consumer));
}

TEST(CfgAnalysis, MixedSwappcTargetsKeepSharedContinuation) {
  constexpr uint16_t kPcSreg = 8;
  constexpr uint16_t kReturnSreg = 30;
  constexpr uint32_t kLiteralOperand = 255;

  // Two paths build different finite targets for one swappc consumer. The
  // first target returns through the saved pair; the second terminates.
  std::vector<uint32_t> words = {
      pack_sopp(5, 4),                                 // 0x00 -> builder B at 0x14.
      pack_sop1(0x1c, kPcSreg, 0),                     // 0x04: builder A getpc.
      pack_sop2(0, kPcSreg, kPcSreg, kLiteralOperand), // 0x08: s_add_u32.
      0x20u,                                           // 0x0c: target A at 0x28.
      build_s_branch(3, ROCJITSU_CODE_ARCH_CDNA4),     // 0x10 -> consumer at 0x20.
      pack_sop1(0x1c, kPcSreg, 0),                     // 0x14: builder B getpc.
      pack_sop2(0, kPcSreg, kPcSreg, kLiteralOperand), // 0x18: s_add_u32.
      0x14u,                                           // 0x1c: target B at 0x2c.
      pack_sop1(0x1e, kReturnSreg, kPcSreg),           // 0x20: s_swappc_b64.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),        // 0x24: continuation.
      pack_sop1(0x1d, 0, kReturnSreg),                 // 0x28: returning target A.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),        // 0x2c: non-returning target B.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

  auto *consumer = block_starting_at(blocks, 32);
  auto *continuation = block_starting_at(blocks, 36);
  auto *returning_target = block_starting_at(blocks, 40);
  auto *nonreturning_target = block_starting_at(blocks, 44);
  ASSERT_NE(consumer, nullptr);
  ASSERT_NE(continuation, nullptr);
  ASSERT_NE(returning_target, nullptr);
  ASSERT_NE(nonreturning_target, nullptr);

  ASSERT_EQ(consumer->static_indirect_call_fixups().size(), 2u);
  ASSERT_EQ(consumer->call_edges().size(), 1u);
  EXPECT_EQ(consumer->call_edges()[0].callee, returning_target);
  EXPECT_EQ(consumer->call_edges()[0].continuation, continuation);
  EXPECT_TRUE(has_successor_start(*consumer, continuation->start_offset()));
  EXPECT_TRUE(has_successor_start(*consumer, nonreturning_target->start_offset()));
  EXPECT_FALSE(has_successor_start(*consumer, returning_target->start_offset()));
  EXPECT_TRUE(has_predecessor(*continuation, consumer));
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

TEST(CfgAnalysis, IndirectRecoveryPrefilterAdmitsGfx1250LaneStashSwapPc) {
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
  ASSERT_EQ(consumer->static_indirect_call_fixups().size(), 1u)
      << "lane-stash swappc consumer must pass the indirect-recovery prefilter";
  EXPECT_EQ(consumer->static_indirect_call_fixups()[0].source_target_offset, 56u);
  EXPECT_TRUE(has_successor_start(*consumer, target->start_offset()));
}

TEST(CfgAnalysis, Gfx1250WideVgprWriteInvalidatesStashedLane) {
  // Same stash idiom as IndirectRecoveryPrefilterAdmitsGfx1250LaneStashSwapPc,
  // but a width-2 v_mov_b64 writes v[44:45] between the writelanes and the
  // readlanes. That wide write overwrites the stashed VGPR, so the readlane no
  // longer reconstructs the original PC and recovery must fail closed. A
  // width-one-only invalidation would miss the b64 write and falsely recover a
  // target.
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

TEST(CfgAnalysis, Gfx1250CarriesLaneStashAcrossProvenBlockBoundary) {
  // Same stash idiom, but an unconditional branch separates the writelane stashes
  // from the readlane/swappc consumer. The sole predecessor carries the identical
  // physical-v44 lane definitions, so must-reaching-definition dataflow proves
  // the target across the block boundary.
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

  auto *consumer = block_starting_at(blocks, 52);
  ASSERT_NE(consumer, nullptr);
  ASSERT_EQ(consumer->static_indirect_call_fixups().size(), 1u);
  EXPECT_EQ(consumer->static_indirect_call_fixups()[0].source_target_offset, 60u);
}

TEST(CfgAnalysis, Gfx1250UnreachablePostRocrAbortBlockDoesNotPoisonLaneStash) {
  constexpr auto live_branch = gfx1250::build_sopp(gfx1250::kSCbranchScc0Sopp, {.simm16 = 3});
  constexpr auto dead_branch = gfx1250::build_sopp(gfx1250::kSBranchSopp, {.simm16 = 1});
  constexpr auto call = gfx1250::build_sop1(gfx1250::kSSwapPcI64Sop1, {.ssrc0 = 0, .sdst = 30});

  // Mirror the scalar post-trap regression with a gfx1250 PC stashed in v44:
  //
  //   writelanes --conditional-----------------------> readlanes/call
  //                    |
  //                    +--> s_trap 2 -X-> dead edge --^
  //
  // In ExplicitOnly mode the dead post-trap block remains BOTTOM. Letting it
  // contribute an empty lane-stash state would erase the valid v44 definitions
  // at the join and lose this otherwise proven call target.
  std::vector<uint32_t> words = {
      0xBE804700u, // 0x00: s_get_pc_i64 s[0:1].
      0xA980FE00u,
      68u,
      0u, // 0x04: s_add_nc_u64 ..., lit64(68) -> target 0x48.
      0xD761002Cu,
      0x02010000u, // 0x10: v_writelane_b32 v44, s0, 0.
      0xD761002Cu,
      0x02010201u,                                 // 0x18: lane 1 <- s1.
      live_branch[0],                              // 0x20: -> join at 0x30.
      build_s_trap(ROCJITSU_CODE_ARCH_GFX1250, 2), // 0x24: abort terminator.
      dead_branch[0],                              // 0x28: dead edge -> 0x30.
      build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250),  // 0x2c: dead padding.
      0xD7600000u,
      0x0201012Cu, // 0x30: v_readlane_b32 s0, v44, 0.
      0xD7600001u,
      0x0201032Cu,                                // 0x38: lane 1 -> s1.
      call[0],                                    // 0x40: s_swap_pc_i64.
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250), // 0x44: continuation.
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250), // 0x48: target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_GFX1250, {},
                                  ExternalEntryPolicy::ExplicitOnly);

  auto *consumer = block_starting_at(blocks, 64);
  ASSERT_NE(consumer, nullptr);
  ASSERT_EQ(consumer->static_indirect_call_fixups().size(), 1u);
  EXPECT_EQ(consumer->static_indirect_call_fixups()[0].source_target_offset, 72u);
}

TEST(CfgAnalysis, Gfx1250ExplicitOnlyRecoversLaneStashInRecoveredCallee) {
  // The kernel root first makes a scalar-recoverable call to the helper at
  // 0x18. The helper is not an explicit entry and has no direct predecessor;
  // it becomes reachable only after the outer call edge is recovered. Once
  // reachable, its lane-stashed call to 0x58 must participate in the next
  // discovery iteration.
  std::vector<uint32_t> words = {
      0xBE804700u, // 0x00: s_get_pc_i64 s[0:1].
      0xA980FE00u,
      20u,
      0u,          // 0x04: s_add_nc_u64 ..., lit64(20) -> helper 0x18.
      0xBE9E4900u, // 0x10: outer s_swap_pc_i64 -> 0x18.
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250), // 0x14: outer continuation.
      0xBE804700u,                                // 0x18: helper s_get_pc_i64.
      0xA980FE00u,
      60u,
      0u, // 0x1c: s_add_nc_u64 ..., lit64(60) -> target 0x58.
      0xD761002Cu,
      0x02010000u, // 0x28: v_writelane_b32 v44, s0, 0.
      0xD761002Cu,
      0x02010201u, // 0x30: v_writelane_b32 v44, s1, 1.
      0xD7600000u,
      0x0201012Cu, // 0x38: v_readlane_b32 s0, v44, 0.
      0xD7600001u,
      0x0201032Cu,                                // 0x40: lane 1 -> s1.
      build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250), // 0x48: padding.
      build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250), // 0x4c: padding.
      0xBE9E4900u,                                // 0x50: inner call -> 0x58.
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250), // 0x54: inner continuation.
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250), // 0x58: inner target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_GFX1250, {},
                                  ExternalEntryPolicy::ExplicitOnly);

  auto *outer_consumer = block_starting_at(blocks, 16);
  auto *inner_consumer = block_starting_at(blocks, 80);
  ASSERT_NE(outer_consumer, nullptr);
  ASSERT_NE(inner_consumer, nullptr);
  ASSERT_EQ(outer_consumer->static_indirect_call_fixups().size(), 1u);
  EXPECT_EQ(outer_consumer->static_indirect_call_fixups()[0].source_target_offset, 24u);
  ASSERT_EQ(inner_consumer->static_indirect_call_fixups().size(), 1u);
  EXPECT_EQ(inner_consumer->static_indirect_call_fixups()[0].source_target_offset, 88u);
}

TEST(CfgAnalysis, Gfx1250DirectCallKillsCarriedLaneStash) {
  // The stash lives in v48, a CALLER-saved VGPR under CSR_AMDGPU_VGPRs, so a
  // call is not proven to preserve it and the continuation must not recover a
  // second call from the stale stash. (A callee-saved VGPR would survive; see
  // Gfx1250CalleeSavedLaneStashSurvivesDirectCall.)
  constexpr uint16_t kReturnSreg = 30;
  constexpr auto clobber = gfx1250::build_vop1(gfx1250::kVMovB32Vop1, {.src0 = 128, .vdst = 48});
  std::vector<uint32_t> words = {
      0xBE804700u, // 0x00: s_get_pc_i64 s[0:1].
      0xA980FE00u, 56u,
      0u, // 0x04: s_add_nc_u64 ..., lit64(56) -> stale target 0x3c.
      0xD7610030u,
      0x02010000u, // 0x10: v_writelane_b32 v48, s0, 0.
      0xD7610030u,
      0x02010201u, // 0x18: v_writelane_b32 v48, s1, 1.
      rocjitsu::build_s_call_b64(kReturnSreg, 7, ROCJITSU_CODE_ARCH_GFX1250),
      // 0x20: direct call -> callee at 0x40.
      0xD7600000u,
      0x02010130u, // 0x24: continuation reads the pre-call low half.
      0xD7600001u,
      0x02010330u,                                // 0x2c: continuation reads the high half.
      0xBE9E4900u,                                // 0x34: stale s_swap_pc_i64.
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250), // 0x38: continuation.
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250), // 0x3c: stale target.
      clobber[0],                                 // 0x40: callee clobbers v48.
      rocjitsu::build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_GFX1250),
      // 0x44: callee return.
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

TEST(CfgAnalysis, Gfx1250CalleeSavedLaneStashSurvivesDirectCall) {
  // Same shape as Gfx1250DirectCallKillsCarriedLaneStash, but the stash lives
  // in v44 (CALLEE-saved under CSR_AMDGPU_VGPRs). The synthetic callee writes
  // only caller-saved v48, so the continuation must recover the call target
  // from the surviving v44 stash.
  constexpr uint16_t kReturnSreg = 30;
  constexpr auto clobber = gfx1250::build_vop1(gfx1250::kVMovB32Vop1, {.src0 = 128, .vdst = 48});
  std::vector<uint32_t> words = {
      0xBE804700u, // 0x00: s_get_pc_i64 s[0:1].
      0xA980FE00u, 56u,
      0u, // 0x04: s_add_nc_u64 ..., lit64(56) -> stashed target 0x3c.
      0xD761002Cu,
      0x02010000u, // 0x10: v_writelane_b32 v44, s0, 0.
      0xD761002Cu,
      0x02010201u, // 0x18: v_writelane_b32 v44, s1, 1.
      rocjitsu::build_s_call_b64(kReturnSreg, 7, ROCJITSU_CODE_ARCH_GFX1250),
      // 0x20: direct call -> callee at 0x40.
      0xD7600000u,
      0x0201012Cu, // 0x24: continuation reads the surviving low half.
      0xD7600001u,
      0x0201032Cu,                                // 0x2c: reads the high half.
      0xBE9E4900u,                                // 0x34: recovered s_swap_pc_i64.
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250), // 0x38: continuation.
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250), // 0x3c: stashed target.
      clobber[0],                                 // 0x40: callee clobbers caller-saved v48.
      rocjitsu::build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_GFX1250),
      // 0x44: callee return.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_GFX1250);

  const IndirectCallFixup *continuation_fixup = nullptr;
  for (const auto &block : blocks) {
    for (const auto &fixup : block->static_indirect_call_fixups()) {
      if (fixup.source_call_offset == 52) // 0x34: the continuation swappc.
        continuation_fixup = &fixup;
    }
  }
  ASSERT_NE(continuation_fixup, nullptr);
  EXPECT_EQ(continuation_fixup->source_target_offset, 60u); // 0x3c: the stashed target.
}

TEST(CfgAnalysis, Gfx1250BankedLaneStashDoesNotSurviveDirectCall) {
  // Select bank 1 for both DST and SRC0, so the v44 operands below consistently
  // address physical v300. Although low selector v44 is callee-saved, the ABI
  // table does not prove physical VGPRs above v255 are preserved. The call must
  // therefore discard the stash rather than mask v300 down to v44.
  //
  // s_set_vgpr_msb immediate byte is {DST[7:6], SRC2[5:4], SRC1[3:2], SRC0[1:0]};
  // 0x41 selects bank 1 for DST and SRC0.
  constexpr uint16_t kReturnSreg = 30;
  constexpr auto set_dst_src0_bank_one =
      gfx1250::build_sopp(gfx1250::kSSetVgprMsbSopp, {.simm16 = 0x41});
  std::vector<uint32_t> words = {
      0xBE804700u, // 0x00: s_get_pc_i64 s[0:1].
      0xA980FE00u, 60u,
      0u,                       // 0x04: s_add_nc_u64 ..., lit64(60) -> stale target 0x40.
      set_dst_src0_bank_one[0], // 0x10: v44 DST/SRC0 operands resolve to physical v300.
      0xD761002Cu,
      0x02010000u, // 0x14: v_writelane_b32 physical v300, s0, 0.
      0xD761002Cu,
      0x02010201u, // 0x1c: v_writelane_b32 physical v300, s1, 1.
      rocjitsu::build_s_call_b64(kReturnSreg, 7, ROCJITSU_CODE_ARCH_GFX1250),
      // 0x24: direct call -> callee at 0x44.
      0xD7600000u,
      0x0201012Cu, // 0x28: continuation reads physical v300 lane 0.
      0xD7600001u,
      0x0201032Cu,                                // 0x30: reads physical v300 lane 1.
      0xBE9E4900u,                                // 0x38: must remain dynamic.
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250), // 0x3c: continuation.
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250), // 0x40: stale target.
      build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250), // 0x44: conforming callee body.
      rocjitsu::build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_GFX1250),
      // 0x48: callee return.
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

TEST(CfgAnalysis, Gfx1250IndirectCallKillsCarriedLaneStash) {
  constexpr uint16_t kCallPcSreg = 8;
  constexpr uint16_t kReturnSreg = 30;
  constexpr uint16_t kStaleReturnSreg = 28;
  constexpr auto stale_getpc = gfx1250::build_sop1(gfx1250::kSGetPcI64Sop1, {.sdst = 0});
  constexpr auto call_getpc = gfx1250::build_sop1(gfx1250::kSGetPcI64Sop1, {.sdst = kCallPcSreg});
  constexpr auto call_add = gfx1250::build_sop2(
      gfx1250::kSAddNcU64Sop2, {.ssrc0 = kCallPcSreg, .ssrc1 = 254, .sdst = kCallPcSreg});
  constexpr auto call =
      gfx1250::build_sop1(gfx1250::kSSwapPcI64Sop1, {.ssrc0 = kCallPcSreg, .sdst = kReturnSreg});
  constexpr auto stale_call =
      gfx1250::build_sop1(gfx1250::kSSwapPcI64Sop1, {.ssrc0 = 0, .sdst = kStaleReturnSreg});
  constexpr auto clobber = gfx1250::build_vop1(gfx1250::kVMovB32Vop1, {.src0 = 128, .vdst = 48});

  // Stash a target in v48 (a CALLER-saved VGPR under CSR_AMDGPU_VGPRs), then
  // issue a separately-proven indirect call whose callee clobbers v48. The
  // current call is resolved from its pre-call state, but the continuation must
  // not recover a second call from the stale stash because a caller-saved VGPR
  // is not proven to survive the call. (A callee-saved VGPR would survive; see
  // Gfx1250CalleeSavedLaneStashSurvivesIndirectCall.)
  std::vector<uint32_t> words = {
      stale_getpc[0], // 0x00: s_get_pc_i64 s[0:1].
      0xA980FE00u,
      72u,
      0u, // 0x04: stale target 0x04 + 72 = 0x4c.
      0xD7610030u,
      0x02010000u, // 0x10: v_writelane_b32 v48, s0, 0.
      0xD7610030u,
      0x02010201u, // 0x18: v_writelane_b32 v48, s1, 1.
      call_getpc[0],
      call_add[0],
      44u,
      0u,      // 0x24: call target 0x24 + 44 = 0x50.
      call[0], // 0x30: resolved indirect call.
      0xD7600000u,
      0x02010130u, // 0x34: continuation reads the pre-call low half.
      0xD7600001u,
      0x02010330u,                                // 0x3c: reads the high half.
      stale_call[0],                              // 0x44: stale indirect call.
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250), // 0x48: continuation.
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250), // 0x4c: stale target.
      clobber[0],                                 // 0x50: callee clobbers v48.
      rocjitsu::build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_GFX1250),
      // 0x54: callee return.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_GFX1250);

  size_t total_fixups = 0;
  const IndirectCallFixup *call_fixup = nullptr;
  for (const auto &block : blocks) {
    total_fixups += block->static_indirect_call_fixups().size();
    for (const auto &fixup : block->static_indirect_call_fixups()) {
      if (fixup.source_call_offset == 48)
        call_fixup = &fixup;
    }
  }
  ASSERT_EQ(total_fixups, 1u);
  ASSERT_NE(call_fixup, nullptr);
  EXPECT_EQ(call_fixup->source_target_offset, 80u);
}

TEST(CfgAnalysis, Gfx1250CalleeSavedLaneStashSurvivesIndirectCall) {
  // Same shape as Gfx1250IndirectCallKillsCarriedLaneStash, but the stash lives
  // in v44 (CALLEE-saved under CSR_AMDGPU_VGPRs). A conforming callee must
  // preserve it, so the continuation swappc IS recovered from the stash. This
  // is the RCCL ncclDevKernel pattern: a getpc code target stashed in a
  // callee-saved VGPR, carried across an intervening call, then read back and
  // called.
  constexpr uint16_t kCallPcSreg = 8;
  constexpr uint16_t kReturnSreg = 30;
  constexpr uint16_t kStaleReturnSreg = 28;
  constexpr auto stale_getpc = gfx1250::build_sop1(gfx1250::kSGetPcI64Sop1, {.sdst = 0});
  constexpr auto call_getpc = gfx1250::build_sop1(gfx1250::kSGetPcI64Sop1, {.sdst = kCallPcSreg});
  constexpr auto call_add = gfx1250::build_sop2(
      gfx1250::kSAddNcU64Sop2, {.ssrc0 = kCallPcSreg, .ssrc1 = 254, .sdst = kCallPcSreg});
  constexpr auto call =
      gfx1250::build_sop1(gfx1250::kSSwapPcI64Sop1, {.ssrc0 = kCallPcSreg, .sdst = kReturnSreg});
  constexpr auto stale_call =
      gfx1250::build_sop1(gfx1250::kSSwapPcI64Sop1, {.ssrc0 = 0, .sdst = kStaleReturnSreg});
  constexpr auto clobber = gfx1250::build_vop1(gfx1250::kVMovB32Vop1, {.src0 = 128, .vdst = 48});

  std::vector<uint32_t> words = {
      stale_getpc[0], // 0x00: s_get_pc_i64 s[0:1].
      0xA980FE00u,
      72u,
      0u, // 0x04: stashed target 0x04 + 72 = 0x4c.
      0xD761002Cu,
      0x02010000u, // 0x10: v_writelane_b32 v44, s0, 0.
      0xD761002Cu,
      0x02010201u, // 0x18: v_writelane_b32 v44, s1, 1.
      call_getpc[0],
      call_add[0],
      44u,
      0u,      // 0x24: call target 0x24 + 44 = 0x50.
      call[0], // 0x30: resolved intervening indirect call.
      0xD7600000u,
      0x0201012Cu, // 0x34: continuation reads the surviving low half.
      0xD7600001u,
      0x0201032Cu,                                // 0x3c: reads the high half.
      stale_call[0],                              // 0x44: continuation indirect call.
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250), // 0x48: continuation.
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250), // 0x4c: stashed target.
      clobber[0],                                 // 0x50: callee clobbers caller-saved v48.
      rocjitsu::build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_GFX1250),
      // 0x54: callee return.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_GFX1250);

  const IndirectCallFixup *continuation_fixup = nullptr;
  for (const auto &block : blocks) {
    for (const auto &fixup : block->static_indirect_call_fixups()) {
      if (fixup.source_call_offset == 68) // 0x44: the continuation swappc.
        continuation_fixup = &fixup;
    }
  }
  ASSERT_NE(continuation_fixup, nullptr);
  EXPECT_EQ(continuation_fixup->source_target_offset, 76u); // 0x4c: the stashed target.
}

TEST(CfgAnalysis, Gfx1250SeedsTextEntryWithLoopBackedgeForLaneStash) {
  constexpr auto getpc = gfx1250::build_sop1(gfx1250::kSGetPcI64Sop1, {.sdst = 0});
  constexpr auto branch = gfx1250::build_sopp(gfx1250::kSBranchSopp, {.simm16 = 0});
  constexpr auto backedge =
      gfx1250::build_sopp(gfx1250::kSCbranchScc0Sopp, {.simm16 = static_cast<uint16_t>(-10)});
  constexpr auto call = gfx1250::build_sop1(gfx1250::kSSwapPcI64Sop1, {.ssrc0 = 0, .sdst = 30});

  // The entry/stash block and latch form a loop, so block zero has a structural
  // predecessor. Architectural entry bank zero must still seed the vector-lane
  // dataflow; the stash written in block zero then reaches the fallthrough
  // consumer after the latch.
  std::vector<uint32_t> words = {
      getpc[0], // 0x00: s_get_pc_i64 s[0:1].
      0xA980FE00u,
      60u,
      0u, // 0x04: target 0x04 + 60 = 0x40.
      0xD761002Cu,
      0x02010000u, // 0x10: v_writelane_b32 v44, s0, 0.
      0xD761002Cu,
      0x02010201u, // 0x18: v_writelane_b32 v44, s1, 1.
      branch[0],   // 0x20: -> latch at 0x24.
      backedge[0], // 0x24: conditional backedge -> block zero.
      0xD7600000u,
      0x0201012Cu, // 0x28: v_readlane_b32 s0, v44, 0.
      0xD7600001u,
      0x0201032Cu,                                // 0x30: v_readlane_b32 s1, v44, 1.
      call[0],                                    // 0x38: cross-block consumer.
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250), // 0x3c: continuation.
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250), // 0x40: target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_GFX1250);

  auto *consumer = block_starting_at(blocks, 56);
  ASSERT_NE(consumer, nullptr);
  ASSERT_EQ(consumer->static_indirect_call_fixups().size(), 1u);
  EXPECT_EQ(consumer->static_indirect_call_fixups()[0].source_target_offset, 64u);
}

TEST(CfgAnalysis, Gfx1250ExplicitKernelEntryClearsIncomingLaneStash) {
  constexpr auto getpc = gfx1250::build_sop1(gfx1250::kSGetPcI64Sop1, {.sdst = 0});
  constexpr auto branch = gfx1250::build_sopp(gfx1250::kSBranchSopp, {.simm16 = 1});
  constexpr auto call = gfx1250::build_sop1(gfx1250::kSSwapPcI64Sop1, {.ssrc0 = 0, .sdst = 30});

  // Entry A stashes a target and branches into entry B. B is independently
  // launchable, so its external path has no proven v44 lane contents even
  // though A's internal predecessor carries a complete stash.
  std::vector<uint32_t> words = {
      getpc[0], // 0x00: s_get_pc_i64 s[0:1].
      0xA980FE00u,
      64u,
      0u, // 0x04: target 0x04 + 64 = 0x44.
      0xD761002Cu,
      0x02010000u, // 0x10: v_writelane_b32 v44, s0, 0.
      0xD761002Cu,
      0x02010201u, // 0x18: v_writelane_b32 v44, s1, 1.
      branch[0],   // 0x20: -> independently launchable entry B at 0x28.
      build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250), // 0x24: skipped.
      0xD7600000u,
      0x0201012Cu, // 0x28: entry B reads v44 lane 0.
      0xD7600001u,
      0x0201032Cu,                                // 0x30: reads v44 lane 1.
      call[0],                                    // 0x38: must remain dynamic.
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250), // 0x3c: continuation.
      build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250), // 0x40: padding.
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250), // 0x44: A's target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  constexpr std::array<uint64_t, 1> extra_leaders{40};
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_GFX1250, extra_leaders);

  size_t total_fixups = 0;
  for (const auto &block : blocks)
    total_fixups += block->static_indirect_call_fixups().size();
  EXPECT_EQ(total_fixups, 0u);
}

TEST(CfgAnalysis, Gfx1250A0UsesLowByteOfVgprMsb) {
  // The gfx1250 A0 profile stores the previous VGPR-MSB state in SIMM16[15:8].
  // Only SIMM16[7:0] updates the current operand banks. Thus 0x4400 establishes
  // bank zero (and records previous state 0x44); it must not redirect this stash
  // to physical v300 or invalidate the already-stashed physical-v44 lanes.
  constexpr auto set_bank_zero_with_previous_44 =
      gfx1250::build_sopp(gfx1250::kSSetVgprMsbSopp, {.simm16 = 0x4400});
  std::vector<uint32_t> words = {
      0xBE804700u, // 0x00: s_get_pc_i64 s[0:1].
      0xA980FE00u,
      56u,
      0u, // 0x04: s_add_nc_u64 ..., lit64(56) -> target 0x3c.
      set_bank_zero_with_previous_44[0],
      0xD761002Cu,
      0x02010000u, // 0x14: v_writelane_b32 physical v44, s0, 0.
      0xD761002Cu,
      0x02010201u, // 0x1c: v_writelane_b32 physical v44, s1, 1.
      set_bank_zero_with_previous_44[0],
      0xD7600000u,
      0x0201012Cu, // 0x28: v_readlane_b32 s0, physical v44, 0.
      0xD7600001u,
      0x0201032Cu,                                // 0x30: v_readlane_b32 s1, physical v44, 1.
      0xBE9E4900u,                                // 0x38: s_swap_pc_i64 s[30:31], s[0:1].
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250), // 0x3c: target/continuation.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_GFX1250);

  auto *consumer = block_starting_at(blocks, 56);
  ASSERT_NE(consumer, nullptr);
  ASSERT_EQ(consumer->static_indirect_call_fixups().size(), 1u);
  EXPECT_EQ(consumer->static_indirect_call_fixups()[0].source_target_offset, 60u);
}

TEST(CfgAnalysis, Gfx1250DoesNotRecoverLaneStashWithDifferingRoleBanks) {
  // Same straight-line stash idiom as
  // IndirectRecoveryPrefilterAdmitsGfx1250LaneStashSwapPc, but an
  // s_set_vgpr_msb sets the DST bank to 1 while leaving the SRC0 bank at 0. The
  // v_writelane writes physical v[44+256] (DST bank 1) while the v_readlane
  // reads physical v44 (SRC0 bank 0). Because the roles resolve the same low
  // selector to different physical VGPRs, no value actually flows, and
  // recovery must fail closed rather than key both by the low selector and
  // falsely reconstruct a PC.
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

TEST(CfgAnalysis, Gfx1250InheritsBankAlongProvenCfgEdge) {
  // An s_set_vgpr_msb in the entry block establishes a bank, then s_branch jumps to
  // a stash block that performs the full getpc/writelane/readlane/swappc idiom with
  // no local s_set_vgpr_msb. MODE is architectural state, so the sole CFG edge
  // carries bank zero into the stash block. This is CFG propagation rather than
  // accidental inheritance from lexical scan order.
  //
  // 0x00 s_set_vgpr_msb 0 ; 0x04 s_branch 0 -> next block at 0x08.
  constexpr auto set_bank_zero = gfx1250::build_sopp(gfx1250::kSSetVgprMsbSopp, {.simm16 = 0});
  constexpr auto branch_to_stash = gfx1250::build_sopp(gfx1250::kSBranchSopp, {.simm16 = 0});
  std::vector<uint32_t> words = {
      set_bank_zero[0],   // 0x00: establish bank 0 (entry block).
      branch_to_stash[0], // 0x04: s_branch -> stash block at 0x08.
      0xBE804700u,        // 0x08: s_get_pc_i64 s[0:1] (stash block).
      0xA980FE00u,
      52u,
      0u,          // 0x0c: s_add_nc_u64 ..., lit64(52).
      0xD761002Cu, // 0x18: v_writelane_b32 v44, s0, 0.
      0x02010000u,
      0xD761002Cu, // 0x20: v_writelane_b32 v44, s1, 1.
      0x02010201u,
      0xD7600000u, // 0x28: v_readlane_b32 s0, v44, 0.
      0x0201012Cu,
      0xD7600001u, // 0x30: v_readlane_b32 s1, v44, 1.
      0x0201032Cu,
      0xBE9E4900u,                                // 0x38: s_swap_pc_i64 s[30:31], s[0:1].
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250), // 0x3c: continuation.
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250), // 0x40: target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_GFX1250);

  auto *consumer = block_starting_at(blocks, 56);
  ASSERT_NE(consumer, nullptr);
  ASSERT_EQ(consumer->static_indirect_call_fixups().size(), 1u);
  EXPECT_EQ(consumer->static_indirect_call_fixups()[0].source_target_offset, 64u);
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

  // Assert on the recovery pass output directly: the must-reaching-definition
  // join must discard the stash because the A->C edge bypasses it.
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
  EXPECT_TRUE(fixups.empty()) << "must-dataflow must not reuse a skipped-path stash";
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

TEST(LivenessAnalysis, UnavailableQueriesFailClosed) {
  const TestInstruction instruction("query");
  const LivenessAnalysis liveness = LivenessAnalysis::unavailable();

  EXPECT_THROW((void)liveness.has_live_before(instruction), std::logic_error);
  EXPECT_THROW((void)liveness.live_before(instruction), std::logic_error);
  EXPECT_THROW((void)liveness.find_globally_unused_vgpr_run(&instruction, 1), std::logic_error);
}

TEST(LivenessAnalysis, ReportsWhetherLiveBeforeSnapshotWasMaterialized) {
  auto blocks = build_test_blocks({TestOpcode::UseSgpr4, TestOpcode::End});
  const Instruction &use = *blocks.front()->instructions().begin();
  const TestInstruction outside_scope("outside_scope");
  const LivenessAnalysis liveness = analyze_scope(blocks);

  EXPECT_TRUE(liveness.has_live_before(use));
  EXPECT_FALSE(liveness.has_live_before(outside_scope));
  EXPECT_TRUE(liveness.is_live_before(use, {RegClass::SGPR, 4, 1}))
      << "the materialized snapshot must contain the register used here";
  EXPECT_FALSE(liveness.is_live_before(outside_scope, {RegClass::SGPR, 4, 1}))
      << "a missing snapshot reads as nothing-live, so callers must check "
         "has_live_before first";
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
  EXPECT_EQ(liveness.find_globally_unused_vgpr_run(&*instruction, 1, 1, 1, 2), 1)
      << "a known bank-2 access must not make the raw low-bank tuple look used";
  EXPECT_TRUE(liveness.is_live_before(*instruction, {RegClass::VGPR, 513, 1}));
  EXPECT_FALSE(liveness.is_live_before(*instruction, {RegClass::VGPR, 1, 1}));
}

TEST(LivenessAnalysis, Gfx1250ImplicitVgprUseResolvesDestinationBank) {
  // v_mov_b16 is a partial (16-bit) write, so it read-modify-preserves its full
  // destination VGPR. That preserve-read is reported through
  // implicit_use_operands() with the destination's Dst VGPR-MSB role. Set DST
  // bank 2 (byte {DST[7:6],SRC2,SRC1,SRC0} = 2<<6 = 0x80) and write vdst v1: the
  // architectural register read is physical v513, so it must be live before the
  // move. If the implicit read stayed at low-bank v1, liveness would treat v513
  // as dead and a scratch borrow could clobber it.
  constexpr auto set_dst_bank_two =
      gfx1250::build_sopp(gfx1250::kSSetVgprMsbSopp, {.simm16 = 0x80});
  constexpr auto move = gfx1250::build_vop1(gfx1250::kVMovB16Vop1, {.src0 = 128, .vdst = 1});
  constexpr auto end = gfx1250::build_sopp(gfx1250::kSEndpgmSopp);
  TestCodeObject co({set_dst_bank_two[0], move[0], end[0]});
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
  EXPECT_EQ(liveness.vgpr_msb_bank_before(*instruction, amdgpu::VgprMsbRole::Dst), 2);
  EXPECT_TRUE(liveness.is_live_before(*instruction, {RegClass::VGPR, 513, 1}))
      << "implicit RMW read of the destination must resolve to the DST bank";
  EXPECT_FALSE(liveness.is_live_before(*instruction, {RegClass::VGPR, 1, 1}))
      << "the low-bank alias must not be treated as the read register";
}

TEST(LivenessAnalysis, Gfx1250ImplicitVgprUseResolvesDespiteExplicitBank0Alias) {
  // Aliasing case: v_mov_b16 v1, v1 reads v1 as an explicit SRC0 (bank 0) and
  // also preserve-reads its destination v1 in DST bank 2 (physical v513). A
  // "newly-added bits" recovery would miss v513 because raw v1 is already present
  // from the explicit source; the per-operand path must add v513 regardless.
  // 0x80 selects DST bank 2, SRC0 bank 0.
  constexpr auto set_dst_bank_two =
      gfx1250::build_sopp(gfx1250::kSSetVgprMsbSopp, {.simm16 = 0x80});
  constexpr auto move = gfx1250::build_vop1(gfx1250::kVMovB16Vop1, {.src0 = 256 + 1, .vdst = 1});
  constexpr auto end = gfx1250::build_sopp(gfx1250::kSEndpgmSopp);
  TestCodeObject co({set_dst_bank_two[0], move[0], end[0]});
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
  EXPECT_TRUE(liveness.is_live_before(*instruction, {RegClass::VGPR, 513, 1}))
      << "the DST-bank preserve-read must be added even though raw v1 is already an explicit use";
  EXPECT_TRUE(liveness.is_live_before(*instruction, {RegClass::VGPR, 1, 1}))
      << "the explicit SRC0 bank-0 read of v1 is still live";
}

TEST(LivenessAnalysis, Gfx1250SwapImplicitReadsResolvePerRole) {
  // v_swap_b16 preserve-reads BOTH operands, each in its own role: vdst in the
  // DST bank and src0 in the SRC0 bank. With SRC0 bank 1 and DST bank 2, vdst=v1
  // reads physical v513 (Dst) and src0=v2 reads physical v258 (Src0). Assigning
  // both implicit reads the DST bank would mislocate the src0 read.
  // Byte {DST[7:6],SRC2,SRC1,SRC0}: DST bank 2 (0x80) | SRC0 bank 1 (0x01) = 0x81.
  constexpr auto set_banks = gfx1250::build_sopp(gfx1250::kSSetVgprMsbSopp, {.simm16 = 0x81});
  constexpr auto swap = gfx1250::build_vop1(gfx1250::kVSwapB16Vop1, {.src0 = 256 + 2, .vdst = 1});
  constexpr auto end = gfx1250::build_sopp(gfx1250::kSEndpgmSopp);
  TestCodeObject co({set_banks[0], swap[0], end[0]});
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
  EXPECT_EQ(liveness.vgpr_msb_bank_before(*instruction, amdgpu::VgprMsbRole::Dst), 2);
  EXPECT_EQ(liveness.vgpr_msb_bank_before(*instruction, amdgpu::VgprMsbRole::Src0), 1);
  EXPECT_TRUE(liveness.is_live_before(*instruction, {RegClass::VGPR, 513, 1}))
      << "vdst preserve-read must resolve to the DST bank (v1 -> v513)";
  EXPECT_TRUE(liveness.is_live_before(*instruction, {RegClass::VGPR, 258, 1}))
      << "src0 preserve-read must resolve to the SRC0 bank (v2 -> v258)";
  // The mixed-role signature: resolving the hook-added src0 entry with the DST
  // bank puts raw v2 at 2 + 2*256 = v514. v513/v258 above stay live either way
  // (the destination supplies v513, the explicit source supplies v258), so v514
  // is the only assertion that actually fails when the roles are conflated.
  EXPECT_FALSE(liveness.is_live_before(*instruction, {RegClass::VGPR, 514, 1}))
      << "src0 must not be mislocated to the DST bank (v2 under DST bank 2 -> v514)";
}

TEST(LivenessAnalysis, Gfx1250DppPreserveReadResolvesToDstBank) {
  // Covers the ENCODING-level preserved-destination hook, the other half of the
  // implicit-operand surface: the v_mov_b16 cases above exercise the
  // per-instruction partial-def path, while a partial-DPP write reaches
  // implicit_use_operands() through the shared SDWA/DPP predicate on the VOP1
  // encoding base. Because InstDefUse strips the VGPR class from the flat
  // implicit_uses() result on gfx1250, dropping the encoding-level operand push
  // would leave this read with no live destination at all rather than a wrong
  // one -- a silent liveness hole, so it needs its own regression.
  //
  // DST bank 2 (0x80), then v_mov_b32_dpp vdst=v5 with row_mask=0x7 (partial),
  // so the unwritten rows preserve the destination: raw v5 reads physical
  // 5 + 2*256 = v517.
  constexpr auto set_banks = gfx1250::build_sopp(gfx1250::kSSetVgprMsbSopp, {.simm16 = 0x80});
  // VOP1 word0: enc[31:25]=0x3F, vdst[24:17]=5, op[15:9]=kVMovB32Vop1,
  // src0[8:0]=SRC_DPP. DPP word1: row_mask[31:28]=0x7 (partial),
  // bank_mask[27:24]=0xF, vsrc0[7:0]=2.
  constexpr uint32_t kDppMovWord0 =
      (0x3Fu << 25) | (5u << 17) | (uint32_t{gfx1250::kVMovB32Vop1} << 9) | amdgpu::SRC_DPP;
  constexpr uint32_t kDppWord1Partial = (0x7u << 28) | (0xFu << 24) | 2u;
  constexpr auto end = gfx1250::build_sopp(gfx1250::kSEndpgmSopp);
  TestCodeObject co({set_banks[0], kDppMovWord0, kDppWord1Partial, end[0]});
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
  EXPECT_EQ(liveness.vgpr_msb_bank_before(*instruction, amdgpu::VgprMsbRole::Dst), 2);
  EXPECT_TRUE(liveness.is_live_before(*instruction, {RegClass::VGPR, 517, 1}))
      << "the partial-DPP preserve-read must resolve to the DST bank (v5 -> v517)";
  EXPECT_FALSE(liveness.is_live_before(*instruction, {RegClass::VGPR, 5, 1}))
      << "the unbanked raw index must not be marked live in place of v517";
}

TEST(LivenessAnalysis, Gfx1250ImplicitVgprUseUnknownBankReadsEveryCandidate) {
  // A dynamic MODE write leaves the DST bank ambiguous. The implicit preserve-read
  // of v_mov_b16 vdst=v1 must then may-read all four candidate tuples, so v1,
  // v257, v513, and v769 are all live before the move (the sound fallback).
  constexpr uint16_t kModeAllBanksHwreg = 1u | (12u << 6) | (7u << 11);
  constexpr auto setreg =
      gfx1250::build_sopk(gfx1250::kSSetregB32Sopk, {.simm16 = kModeAllBanksHwreg, .sdst = 0});
  constexpr auto move = gfx1250::build_vop1(gfx1250::kVMovB16Vop1, {.src0 = 128, .vdst = 1});
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
  EXPECT_EQ(liveness.vgpr_msb_bank_before(*instruction, amdgpu::VgprMsbRole::Dst), std::nullopt);
  for (uint16_t bank = 0; bank < 4; ++bank)
    EXPECT_TRUE(liveness.is_live_before(*instruction,
                                        {RegClass::VGPR, static_cast<uint16_t>(1 + bank * 256), 1}))
        << "unknown-bank implicit read must may-read candidate bank " << bank;
}

TEST(LivenessAnalysis, Gfx1250UnknownBankDefMakesEveryCandidateGloballyUsed) {
  // A dynamic MODE write leaves the destination bank ambiguous. Whole-kernel
  // usage must reserve all four candidate tuples, while backward liveness must
  // not pretend the one physical write kills all four.
  constexpr uint16_t kModeAllBanksHwreg = 1u | (12u << 6) | (7u << 11);
  constexpr auto setreg =
      gfx1250::build_sopk(gfx1250::kSSetregB32Sopk, {.simm16 = kModeAllBanksHwreg, .sdst = 0});
  constexpr auto move = gfx1250::build_vop1(gfx1250::kVMovB32Vop1, {.src0 = 128, .vdst = 1});
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
  EXPECT_EQ(liveness.vgpr_msb_bank_before(*instruction, amdgpu::VgprMsbRole::Dst), std::nullopt);
  for (uint16_t bank = 0; bank < 4; ++bank) {
    const uint16_t candidate = static_cast<uint16_t>(1 + bank * 256);
    EXPECT_EQ(liveness.find_globally_unused_vgpr_run(&*instruction, 1, candidate, 1,
                                                     static_cast<uint16_t>(candidate + 1)),
              std::nullopt)
        << "unknown-bank definition must reserve candidate bank " << bank;
  }
  EXPECT_EQ(liveness.find_globally_unused_vgpr_run(&*instruction, 1, 0, 1, 4), 0);
  EXPECT_EQ(liveness.find_globally_unused_vgpr_run(&*instruction, 1, 2, 1, 4), 2);

  const BlockLiveness &state = liveness.block_liveness(*blocks.front());
  EXPECT_FALSE(state.kill.contains({RegClass::VGPR, 1, 1}));
  EXPECT_FALSE(state.kill.contains({RegClass::VGPR, 257, 1}));
}

TEST(LivenessAnalysis, Gfx1250RelativeVgprAccessDisablesGlobalUnusedQuery) {
  // M0 can redirect the encoded v0 source to any relative tuple, including v1
  // which would otherwise appear globally unused.
  constexpr auto move = gfx1250::build_vop1(gfx1250::kVMovrelsB32Vop1, {.src0 = 0, .vdst = 2});
  constexpr auto end = gfx1250::build_sopp(gfx1250::kSEndpgmSopp);
  TestCodeObject co({move[0], end[0]});
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

  const Instruction &instruction = *blocks.front()->instructions().begin();
  EXPECT_EQ(liveness.find_globally_unused_vgpr_run(&instruction, 1, 1, 1, 2), std::nullopt);
  EXPECT_FALSE(liveness.has_materialized_cfg_liveness());
}

TEST(LivenessAnalysis, Gfx1250SwaprelDisablesGlobalUnusedQuery) {
  constexpr auto swap = gfx1250::build_vop1(gfx1250::kVSwaprelB32Vop1, {.src0 = 0, .vdst = 2});
  constexpr auto end = gfx1250::build_sopp(gfx1250::kSEndpgmSopp);
  TestCodeObject co({swap[0], end[0]});
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

  const Instruction &instruction = *blocks.front()->instructions().begin();
  EXPECT_EQ(liveness.find_globally_unused_vgpr_run(&instruction, 1, 1, 1, 2), std::nullopt);
  EXPECT_FALSE(liveness.has_materialized_cfg_liveness());
}

TEST(LivenessAnalysis, Gfx1250GprIndexModeWriteDisablesGlobalUnusedQuery) {
  // A runtime MODE[27] write can enable GPR indexing, after which ordinary
  // encoded operands may access M0-offset VGPRs.
  constexpr uint16_t kModeGprIdxEnableHwreg = 1u | (27u << 6);
  constexpr auto setreg =
      gfx1250::build_sopk(gfx1250::kSSetregB32Sopk, {.simm16 = kModeGprIdxEnableHwreg, .sdst = 0});
  constexpr auto move = gfx1250::build_vop1(gfx1250::kVMovB32Vop1, {.src0 = 256, .vdst = 0});
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
  EXPECT_EQ(liveness.find_globally_unused_vgpr_run(&*instruction, 1, 1, 1, 2), std::nullopt);
  EXPECT_FALSE(liveness.has_materialized_cfg_liveness());
}

TEST(LivenessAnalysis, Gfx1250ImmediateGprIndexModeWriteUsesLiteralValue) {
  constexpr uint16_t kModeGprIdxEnableHwreg = 1u | (27u << 6);
  constexpr auto setreg =
      gfx1250::build_sopk(gfx1250::kSSetregImm32B32Sopk, {.simm16 = kModeGprIdxEnableHwreg});
  constexpr auto move = gfx1250::build_vop1(gfx1250::kVMovB32Vop1, {.src0 = 256, .vdst = 0});
  constexpr auto end = gfx1250::build_sopp(gfx1250::kSEndpgmSopp);

  for (uint32_t literal : {0u, 1u}) {
    SCOPED_TRACE(literal);
    TestCodeObject co({setreg[0], literal, move[0], end[0]});
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
    const auto unused = liveness.find_globally_unused_vgpr_run(&*instruction, 1, 1, 1, 2);
    if (literal == 0)
      EXPECT_EQ(unused, 1);
    else
      EXPECT_EQ(unused, std::nullopt);
    EXPECT_FALSE(liveness.has_materialized_cfg_liveness());
  }
}

TEST(LivenessAnalysis, Cdna4DynamicGprIndexModeWriteDisablesGlobalUnusedQuery) {
  constexpr uint16_t kModeGprIdxEnableHwreg = 1u | (27u << 6);
  constexpr auto setreg =
      cdna4::build_sopk(cdna4::kSSetregB32Sopk, {.simm16 = kModeGprIdxEnableHwreg, .sdst = 0});
  constexpr auto move = cdna4::build_vop1(cdna4::kVMovB32Vop1, {.src0 = 256, .vdst = 0});
  constexpr auto end = cdna4::build_sopp(cdna4::kSEndpgmSopp);
  TestCodeObject co({setreg[0], move[0], end[0]});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_EQ(blocks.size(), 1u);
  auto scope = block_scope(blocks);

  LivenessAnalysisOptions options;
  options.arch = ROCJITSU_CODE_ARCH_CDNA4;
  options.text = text_span(co);
  LivenessAnalysis liveness(KernelBlockScope(scope), options);

  auto instruction = blocks.front()->instructions().begin();
  ++instruction;
  ASSERT_NE(instruction, blocks.front()->instructions().end());
  EXPECT_EQ(liveness.find_globally_unused_vgpr_run(&*instruction, 1, 1, 1, 2), std::nullopt);
  EXPECT_FALSE(liveness.has_materialized_cfg_liveness());
}

TEST(LivenessAnalysis, Cdna4ImmediateGprIndexModeWriteUsesLiteralValue) {
  constexpr uint16_t kModeGprIdxEnableHwreg = 1u | (27u << 6);
  constexpr auto setreg =
      cdna4::build_sopk(cdna4::kSSetregImm32B32Sopk, {.simm16 = kModeGprIdxEnableHwreg});
  constexpr auto move = cdna4::build_vop1(cdna4::kVMovB32Vop1, {.src0 = 256, .vdst = 0});
  constexpr auto end = cdna4::build_sopp(cdna4::kSEndpgmSopp);

  for (uint32_t literal : {0u, 1u}) {
    SCOPED_TRACE(literal);
    TestCodeObject co({setreg[0], literal, move[0], end[0]});
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    ASSERT_NE(decoder, nullptr);
    auto blocks = BasicBlock::build(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);
    ASSERT_EQ(blocks.size(), 1u);
    auto scope = block_scope(blocks);

    LivenessAnalysisOptions options;
    options.arch = ROCJITSU_CODE_ARCH_CDNA4;
    options.text = text_span(co);
    LivenessAnalysis liveness(KernelBlockScope(scope), options);

    auto instruction = blocks.front()->instructions().begin();
    ++instruction;
    ASSERT_NE(instruction, blocks.front()->instructions().end());
    const auto unused = liveness.find_globally_unused_vgpr_run(&*instruction, 1, 1, 1, 2);
    if (literal == 0)
      EXPECT_EQ(unused, 1);
    else
      EXPECT_EQ(unused, std::nullopt);
    EXPECT_FALSE(liveness.has_materialized_cfg_liveness());
  }
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
  // Request a write to MODE bit zero. gfx1250 updates all VGPR-MSB fields from
  // literal bits [19:12].
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
  EXPECT_TRUE(liveness.any_live_before(use, {RegClass::SGPR, 4, 1}));
  EXPECT_EQ(liveness.find_free_sgpr(&use, 4), 5);
}

TEST(LivenessAnalysis, AnyLiveBeforeRejectsInstructionOutsideAnalysis) {
  auto analyzed_blocks = build_test_blocks({TestOpcode::UseSgpr4, TestOpcode::End});
  auto other_blocks = build_test_blocks({TestOpcode::Nop, TestOpcode::End});
  LivenessAnalysis liveness = analyze_scope(analyzed_blocks);

  const Instruction &outside = *other_blocks[0]->instructions().begin();
  EXPECT_FALSE(liveness.has_live_before(outside));
  EXPECT_FALSE(liveness.contains_block(*other_blocks[0]));
  EXPECT_TRUE(liveness.has_live_before(*analyzed_blocks[0]->instructions().begin()));
  EXPECT_TRUE(liveness.contains_block(*analyzed_blocks[0]));
  EXPECT_TRUE(liveness.any_live_before(outside, {RegClass::SGPR, 4, 2}));
  EXPECT_FALSE(liveness.all_live_before(outside, {RegClass::SGPR, 4, 2}));
  EXPECT_EQ(liveness.find_free_sgpr_pair(&outside, 4), std::nullopt);
}

TEST(LivenessAnalysis, FindValidSgprPair) {
  auto blocks = build_test_blocks({TestOpcode::UseSgpr4, TestOpcode::End});
  LivenessAnalysis liveness = analyze_scope(blocks);

  const Instruction &use = *blocks[0]->instructions().begin();
  EXPECT_TRUE(liveness.any_live_before(use, {RegClass::SGPR, 4, 1}));
  EXPECT_EQ(liveness.find_free_sgpr_pair(&use, 4), 6);
}

TEST(LivenessAnalysis, FindSgprPairSkipsStraddle) {
  auto blocks = build_test_blocks({TestOpcode::UseSgpr4, TestOpcode::UseSgpr7, TestOpcode::End});
  LivenessAnalysis liveness = analyze_scope(blocks);

  const Instruction &use = *blocks[0]->instructions().begin();
  EXPECT_TRUE(liveness.any_live_before(use, {RegClass::SGPR, 3, 2}));
  EXPECT_TRUE(liveness.any_live_before(use, {RegClass::SGPR, 4, 2}));
  EXPECT_FALSE(liveness.all_live_before(use, {RegClass::SGPR, 3, 2}));
  EXPECT_FALSE(liveness.all_live_before(use, {RegClass::SGPR, 4, 2}));
  EXPECT_TRUE(liveness.any_live_before(use, {RegClass::SGPR, 4, 0}));
  EXPECT_TRUE(liveness.all_live_before(use, {RegClass::SGPR, 4, 0}));
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
  EXPECT_FALSE(liveness.has_materialized_cfg_liveness());
  EXPECT_TRUE(liveness.contains_block(*blocks[0]));
  EXPECT_FALSE(liveness.has_materialized_cfg_liveness());
  EXPECT_FALSE(liveness.any_live_before(use, {RegClass::VGPR, 0, 4}));
  EXPECT_EQ(liveness.find_free_sgpr(&use, 0), 0);
  EXPECT_EQ(liveness.find_free_run(&use, 1, 0), 4);
  EXPECT_EQ(liveness.find_free_run(&use, 1, 7), 7);
}

TEST(LivenessAnalysis, GloballyUnusedRunHonorsMinFreeVgprFloor) {
  auto blocks = build_test_blocks({TestOpcode::UseSgpr4, TestOpcode::End});
  auto scope = block_scope(blocks);
  LivenessAnalysisOptions options;
  options.min_free_vgpr = 4;
  LivenessAnalysis liveness(KernelBlockScope(scope), options);

  const Instruction &use = *blocks[0]->instructions().begin();
  EXPECT_EQ(liveness.find_globally_unused_vgpr_run(&use, 1, 0, 1, 8), 4);
  EXPECT_EQ(liveness.find_globally_unused_vgpr_run(&use, 1, 0, 1, 4), std::nullopt);
  EXPECT_FALSE(liveness.has_materialized_cfg_liveness());
}

TEST(LivenessAnalysis, FindsGloballyUnusedRunBeforeSiteDeadFallback) {
  auto blocks = build_test_blocks({TestOpcode::UseVgpr0, TestOpcode::Nop, TestOpcode::End});
  auto scope = block_scope(blocks);
  const LivenessAnalysis liveness{KernelBlockScope(scope)};

  auto instruction = blocks.front()->instructions().begin();
  ++instruction;
  ASSERT_NE(instruction, blocks.front()->instructions().end());

  EXPECT_FALSE(liveness.has_materialized_cfg_liveness());
  EXPECT_EQ(liveness.find_globally_unused_vgpr_run(&*instruction, 1, 0, 1, 4), 1);
  EXPECT_EQ(liveness.find_globally_unused_vgpr_run(&*instruction, 2, 0, 2, 4), 2);
  EXPECT_EQ(liveness.find_globally_unused_vgpr_run(&*instruction, 2, 0, 1, 1), std::nullopt);
  EXPECT_EQ(liveness.find_globally_unused_vgpr_run(&*instruction, 1, 0, 1, 0), std::nullopt);
  EXPECT_FALSE(liveness.has_materialized_cfg_liveness());
  EXPECT_EQ(liveness.find_free_run(&*instruction, 1), 0)
      << "v0 is dead at this site but is not globally unused";
  EXPECT_TRUE(liveness.has_materialized_cfg_liveness());

  Instruction outside_scope("outside_scope", nullptr);
  EXPECT_EQ(liveness.find_globally_unused_vgpr_run(&outside_scope, 1, 0, 1, 4), std::nullopt);
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
  EXPECT_TRUE(liveness.any_live_before(read_write, {RegClass::SGPR, 4, 1}));
}

TEST(LivenessAnalysis, ReadWriteRegisterStaysLiveOutWhenUsedBySuccessor) {
  std::array<uint64_t, 1> extra_leaders{4};
  auto blocks = build_test_blocks(
      {TestOpcode::ReadWriteSgpr4, TestOpcode::UseSgpr4, TestOpcode::End}, extra_leaders);
  LivenessAnalysis liveness = analyze_scope(blocks);

  ASSERT_EQ(blocks.size(), 2u);
  const Instruction &read_write = *blocks[0]->instructions().begin();
  EXPECT_TRUE(liveness.any_live_before(read_write, {RegClass::SGPR, 4, 1}));
  EXPECT_TRUE(liveness.block_liveness(*blocks[0]).live_out.contains({RegClass::SGPR, 4, 1}));
}

TEST(LivenessAnalysis, PartialDefKeepsRegisterLiveBeforeInstruction) {
  auto blocks = build_test_blocks({TestOpcode::PartialDefSgpr4, TestOpcode::End});
  LivenessAnalysis liveness = analyze_scope(blocks);

  const Instruction &partial_def = *blocks[0]->instructions().begin();
  EXPECT_TRUE(liveness.any_live_before(partial_def, {RegClass::SGPR, 4, 1}));
}

TEST(LivenessAnalysis, PartialDefRegisterStaysLiveOutWhenUsedBySuccessor) {
  std::array<uint64_t, 1> extra_leaders{4};
  auto blocks = build_test_blocks(
      {TestOpcode::PartialDefSgpr4, TestOpcode::UseSgpr4, TestOpcode::End}, extra_leaders);
  LivenessAnalysis liveness = analyze_scope(blocks);

  ASSERT_EQ(blocks.size(), 2u);
  const Instruction &partial_def = *blocks[0]->instructions().begin();
  EXPECT_TRUE(liveness.any_live_before(partial_def, {RegClass::SGPR, 4, 1}));
  EXPECT_TRUE(liveness.block_liveness(*blocks[0]).live_out.contains({RegClass::SGPR, 4, 1}));
}

TEST(LivenessAnalysis, AnyLiveOutRejectsPartiallyLiveRegisterPair) {
  std::array<uint64_t, 1> extra_leaders{4};
  auto blocks =
      build_test_blocks({TestOpcode::Nop, TestOpcode::UseSgpr4, TestOpcode::End}, extra_leaders);
  LivenessAnalysis liveness = analyze_scope(blocks);

  ASSERT_EQ(blocks.size(), 2u);
  EXPECT_TRUE(liveness.any_live_out(*blocks[0], {RegClass::SGPR, 3, 2}));
  EXPECT_TRUE(liveness.any_live_out(*blocks[0], {RegClass::SGPR, 4, 2}));
  EXPECT_FALSE(liveness.block_liveness(*blocks[0]).live_out.contains({RegClass::SGPR, 3, 2}));
  EXPECT_FALSE(liveness.block_liveness(*blocks[0]).live_out.contains({RegClass::SGPR, 4, 2}));
  EXPECT_TRUE(liveness.any_live_out(*blocks[0], {RegClass::SGPR, 4, 0}));
}

TEST(LivenessAnalysis, FullWidthDefKillsRegisterBeforeInstruction) {
  auto blocks = build_test_blocks({TestOpcode::DefSgpr4, TestOpcode::End});
  LivenessAnalysis liveness = analyze_scope(blocks);

  const Instruction &def = *blocks[0]->instructions().begin();
  EXPECT_FALSE(liveness.any_live_before(def, {RegClass::SGPR, 4, 1}));
}

TEST(LivenessAnalysis, ImplicitUseIsLiveBeforeInstruction) {
  auto blocks = build_test_blocks({TestOpcode::ImplicitUseSgpr6Pair, TestOpcode::End});
  LivenessAnalysis liveness = analyze_scope(blocks);

  const Instruction &implicit_use = *blocks[0]->instructions().begin();
  EXPECT_TRUE(liveness.any_live_before(implicit_use, {RegClass::SGPR, 6, 2}));
}

TEST(LivenessAnalysis, PredicatedScalarDefDoesNotKillLiveOutValue) {
  auto blocks =
      build_test_blocks({TestOpcode::PredicatedDefSgpr4, TestOpcode::UseSgpr4, TestOpcode::End});
  LivenessAnalysis liveness = analyze_scope(blocks);

  const Instruction &pred_def = *blocks[0]->instructions().begin();
  EXPECT_TRUE(liveness.any_live_before(pred_def, {RegClass::SGPR, 4, 1}));
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
  EXPECT_TRUE(liveness.any_live_before(branch, {RegClass::SGPR, 4, 1}));
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
  EXPECT_TRUE(all_decoded_liveness.any_live_before(def, {RegClass::VGPR, 0, 1}));

  std::vector<BasicBlock *> kernel_blocks{kernel0};
  LivenessAnalysis kernel_liveness{KernelBlockScope(kernel_blocks)};
  EXPECT_FALSE(kernel_liveness.any_live_before(def, {RegClass::VGPR, 0, 1}));
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

TEST(GeneratedInstDefUse, Cdna3PackedF32TracksWideScalarSource) {
  constexpr std::array<uint32_t, 2> words{0xD3B04004u, 0x1C0A0810u};
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  auto inst = std::unique_ptr<Instruction>(decoder ? decoder->decode(words.data()) : nullptr);
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "v_pk_fma_f32");

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.uses.contains({RegClass::SGPR, 16, 2}));
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
  // v_cvt_f64_i32_e32 writes a VGPR pair (v[6:7]). A partial DPP row mask
  // preserves the whole 64-bit destination, so the implicit use must match the
  // width-2 def -- not just the low dword.
  // CDNA4 VOP1 word0: encoding[31:25]=0x3F, vdst[24:17]=6, op[16:9]=4
  // (v_cvt_f64_i32), src0[8:0]=250 (SRC_DPP).
  constexpr uint32_t kVop1CvtF64I32Word0Dpp = (0x3Fu << 25) | (6u << 17) | (4u << 9) | 250u;
  auto inst = decode_cdna4({kVop1CvtF64I32Word0Dpp, (0x7u << 28) | (0xFu << 24) | 2u});
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()).substr(0, 13), "v_cvt_f64_i32");

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
  // v_dot2_f32_f16 (VOP3P, VGPR vdst=6) has an explicit DPP encoding. A
  // partial row mask preserves the packed VGPR destination.
  // RDNA4 VOP3P word0: encoding[31:24]=204, op[22:16]=19 (v_dot2_f32_f16),
  // vdst[7:0]=6. word1: src0[8:0]=250 (SRC_DPP), src1[17:9]=3 (VGPR3).
  constexpr uint32_t kVop3pDot2Word0 = (204u << 24) | (19u << 16) | 6u;
  constexpr uint32_t kVop3pDppWord1 = (3u << 9) | 250u;
  auto inst = decode_rdna4({kVop3pDot2Word0, kVop3pDppWord1, (0x7u << 28) | (0xFu << 24) | 2u});
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()).substr(0, 14), "v_dot2_f32_f16");

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

// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "decode_test_util.h"
#include "rocjitsu/code/analysis/def_use_chain.h"
#include "rocjitsu/code/analysis/exec_state.h"
#include "rocjitsu/code/analysis/indirect_branch_discovery.h"
#include "rocjitsu/code/analysis/liveness.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/code_object.h"
#include "rocjitsu/code/dbt/binary_translator_internal.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/mubuf.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/operand.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/vbuffer.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/mubuf.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
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
                  std::optional<uint64_t> inline_const_src = std::nullopt, int def_size_bits = 0)
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
  WriteExecFull = 17,
  WriteExecNarrow = 18,
  WriteExecFullInline = 19,
  WriteExecOrAllOnes = 20,
  WriteExecAndSaveexec = 21,
  WriteExecLoHalf = 22,
  WriteExecHiHalf = 23,
  WriteExecHiHalfZero = 25,
  PartialDefSgpr4 = 24,
};

class TestDecoder : public Decoder {
public:
  std::size_t max_instruction_words() const override { return 1; }

  DecodeResult decode(const rj_code_binary_inst_t *inst, const DecodeErrorEmitter &) override {
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
    case TestOpcode::WriteExecHiHalf:
      // s_mov_b32 exec_hi, -1: a 32-bit COPY of all-ones into only the high half
      // of EXEC. Never covers the active lanes, so it can never establish Full.
      return new TestInstruction("test_write_exec_hi_half", {{RegClass::EXEC, 1, 1}}, {},
                                 RESULT_COPY, std::nullopt, {}, std::nullopt, ~0ULL);
    case TestOpcode::WriteExecHiHalfZero:
      // s_mov_b32 exec_hi, 0: a 32-bit COPY of zero into only the high half of EXEC.
      return new TestInstruction("test_write_exec_hi_half_zero", {{RegClass::EXEC, 1, 1}}, {},
                                 RESULT_COPY, std::nullopt, {}, std::nullopt, 0ULL);
    case TestOpcode::PartialDefSgpr4:
      // 16-bit write to s4: defines only part of the lane, so it also reads s4.
      return new TestInstruction("test_partial_def_s4", {{RegClass::SGPR, 4, 1}}, {}, 0,
                                 std::nullopt, {}, std::nullopt, std::nullopt,
                                 /*def_size_bits=*/16);
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
  return build_valid_blocks(co, decoder, ROCJITSU_CODE_ARCH_CDNA3, extra_leaders);
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
  const ExecMaskAnalysis exec(KernelBlockScope(scope), /*wave_size=*/64);
  return LivenessAnalysis(KernelBlockScope(scope), std::make_unique<ExecMaskAnalysis>(exec));
}

// Wire the same scoped call/return edges into BOTH analyses, as BinaryTranslator
// does; feeding them to only one makes the two disagree around calls and returns.
LivenessAnalysis analyze_scope_with_edges(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                                          std::span<const ScopedCfgEdge> extra_edges) {
  auto scope = block_scope(blocks);
  const ExecMaskAnalysis exec(KernelBlockScope(scope), /*wave_size=*/64, extra_edges);
  return LivenessAnalysis(KernelBlockScope(scope), std::make_unique<ExecMaskAnalysis>(exec),
                          /*options=*/{}, extra_edges);
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

TEST(RegisterSetAnalysis, IntersectsIsTheAnyOfCounterpartToContains) {
  RegisterSet set;
  set.expand({RegClass::SGPR, 4, 1});

  // contains() is all-of, intersects() is any-of: a pair straddling s4 is not
  // contained but does intersect. Register allocation depends on the latter --
  // a candidate run is unusable if even one of its lanes is taken.
  EXPECT_FALSE(set.contains({RegClass::SGPR, 3, 2}));
  EXPECT_TRUE(set.intersects({RegClass::SGPR, 3, 2}));

  // Half-open over [index, index + width).
  EXPECT_TRUE(set.intersects({RegClass::SGPR, 4, 1}));
  EXPECT_TRUE(set.intersects({RegClass::SGPR, 2, 3}));
  EXPECT_FALSE(set.intersects({RegClass::SGPR, 2, 2})) << "s4 is outside [2, 4)";
  EXPECT_FALSE(set.intersects({RegClass::SGPR, 5, 4}));
}

TEST(RegisterSetAnalysis, IntersectsKeepsRegisterClassesSeparate) {
  RegisterSet set;
  set.expand({RegClass::SGPR, 0, 1});

  EXPECT_TRUE(set.intersects({RegClass::SGPR, 0, 1}));
  EXPECT_FALSE(set.intersects({RegClass::VGPR, 0, 1}));
  EXPECT_FALSE(set.intersects({RegClass::ACC_VGPR, 0, 1}));
}

TEST(RegisterSetAnalysis, IntersectsAnswersFalseForUntrackedClasses) {
  RegisterSet set;
  set.expand({RegClass::SGPR, 0, 1});

  // Matches contains(): classes outside the dataflow model are not "present".
  // free_registers.h documents why its callers must not search one.
  EXPECT_FALSE(set.intersects({RegClass::EXEC, 0, 1}));
  EXPECT_FALSE(set.intersects({RegClass::VCC, 0, 1}));
}

TEST(RegisterSetAnalysis, IntersectsToleratesRunsPastTheTrackedRange) {
  RegisterSet set;
  set.expand({RegClass::SGPR, 0, 1});

  // A run starting in range and extending past it reports the in-range hit
  // rather than reading past the bitset.
  EXPECT_TRUE(set.intersects({RegClass::SGPR, 0, 255}));
  EXPECT_FALSE(set.intersects(
      {RegClass::SGPR, static_cast<uint16_t>(REGISTER_SET_ALLOCATABLE_SGPRS + 8), 4}));
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
  cdna5::VbufferMachineInst raw{};
  raw.vdata = 4;
  raw.th = return_control;
  AtomicInst inst(reinterpret_cast<const cdna5::MachineInst *>(&raw));

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

  expect_gfx1250_buffer_cmpswap_def_use<cdna5::BufferAtomicCmpswapB32Vbuffer>(kAtomicReturn, 2, 1);
  expect_gfx1250_buffer_cmpswap_def_use<cdna5::BufferAtomicCmpswapB32Vbuffer>(kAtomicNoReturn, 2,
                                                                              0);
  expect_gfx1250_buffer_cmpswap_def_use<cdna5::BufferAtomicCmpswapB64Vbuffer>(kAtomicReturn, 4, 2);
  expect_gfx1250_buffer_cmpswap_def_use<cdna5::BufferAtomicCmpswapB64Vbuffer>(kAtomicNoReturn, 4,
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

TEST(RegisterSetAnalysis, SpecialClassesAreHeldButCarryNoOrdinaryLanes) {
  RegisterSet set;
  set.expand({RegClass::EXEC, 0, 2});
  set.expand({RegClass::SCC, 0, 1});
  set.expand({RegClass::FLAT_SCRATCH, 0, 2});

  // Special classes are singleton members, not dropped.
  EXPECT_FALSE(set.none());
  EXPECT_TRUE(set.has_specials());
  EXPECT_TRUE(set.contains({RegClass::EXEC, 0, 1}));
  EXPECT_TRUE(set.contains({RegClass::SCC, 0, 1}));
  EXPECT_TRUE(set.contains({RegClass::FLAT_SCRATCH, 0, 2}));

  // But they contribute no ordinary lanes and vanish under ordinary_only().
  EXPECT_EQ(set.ordinary_size(), 0u);
  EXPECT_TRUE(set.ordinary_only().none());
}

TEST(RegisterSetAnalysis, GeneratedCdna4OperandsMapTrackedRegisterRefs) {
  cdna4::Operand sgpr(32, cdna4::OperandType::OPR_SRC, cdna4::OpSelSrc::OPR_SRC_SGPR_MIN + 7);
  cdna4::Operand vgpr(32, cdna4::OperandType::OPR_SRC, cdna4::OpSelSrc::OPR_SRC_VGPR_MIN + 7);
  cdna4::Operand acc(32, cdna4::OperandType::OPR_SRC_ACCVGPR,
                     cdna4::OpSelSrcAccvgpr::OPR_SRC_ACCVGPR_ACC_MIN + 7);
  cdna4::Operand ttmp(64, cdna4::OperandType::OPR_SRC, cdna4::OpSelSrc::OPR_SRC_TTMP_MIN + 2);
  cdna4::Operand imm32(32, cdna4::OperandType::OPR_SIMM32, 123);

  ASSERT_TRUE(sgpr.to_register_ref().has_value());
  EXPECT_EQ(*sgpr.to_register_ref(), (RegisterRef{RegClass::SGPR, 7, 1}));
  ASSERT_TRUE(vgpr.to_register_ref().has_value());
  EXPECT_EQ(*vgpr.to_register_ref(), (RegisterRef{RegClass::VGPR, 7, 1}));
  ASSERT_TRUE(acc.to_register_ref().has_value());
  EXPECT_EQ(*acc.to_register_ref(), (RegisterRef{RegClass::ACC_VGPR, 7, 1}));
  ASSERT_TRUE(ttmp.to_register_ref().has_value());
  EXPECT_EQ(*ttmp.to_register_ref(), (RegisterRef{RegClass::TTMP, 2, 2}));
  EXPECT_FALSE(imm32.to_register_ref().has_value());
}

TEST(RegisterSetAnalysis, Cdna4WritelaneDestinationIsUseAndDef) {
  constexpr std::array<uint32_t, 2> kWritelaneV141S4Lane2 = {0xd28a008du, 0x00010404u};
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);

  std::unique_ptr<Instruction> inst(decode_valid(*decoder, kWritelaneV141S4Lane2.data()));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "v_writelane_b32");

  InstDefUse du(*inst);
  EXPECT_TRUE(du.defs.contains({RegClass::VGPR, 141, 1}));
  EXPECT_TRUE(du.uses.contains({RegClass::VGPR, 141, 1}));
  EXPECT_TRUE(du.uses.contains({RegClass::SGPR, 4, 1}));
}

// ---------------------------------------------------------------------------
// Architectural special-register + memory effect API. Special registers are
// singleton members of the same RegisterSet as ordinary registers; consumers
// that drive scratch-allocation liveness project them out with ordinary_only().
// ---------------------------------------------------------------------------

// Collect a set's special members in ascending RegClass order, so a test can
// assert the *exact* set of special registers an instruction reads or writes,
// not just membership.
std::vector<RegClass> specials_in(const RegisterSet &set) {
  std::vector<RegClass> classes;
  set.for_each_special([&](RegClass c) { classes.push_back(c); });
  return classes;
}

// Expected special-register list, sorted to match specials_in()'s ascending
// iteration order.
std::vector<RegClass> special_regs(std::initializer_list<RegClass> classes) {
  std::vector<RegClass> sorted(classes);
  std::ranges::sort(sorted);
  return sorted;
}

TEST(RegisterSetSpecial, ClassifiesSpecialVersusOrdinaryClasses) {
  EXPECT_FALSE(is_special_reg_class(RegClass::SGPR));
  EXPECT_FALSE(is_special_reg_class(RegClass::VGPR));
  EXPECT_FALSE(is_special_reg_class(RegClass::ACC_VGPR));
  EXPECT_TRUE(is_special_reg_class(RegClass::EXEC));
  EXPECT_TRUE(is_special_reg_class(RegClass::VCC));
  EXPECT_TRUE(is_special_reg_class(RegClass::SCC));
  EXPECT_TRUE(is_special_reg_class(RegClass::M0));
  EXPECT_TRUE(is_special_reg_class(RegClass::FLAT_SCRATCH));
  EXPECT_TRUE(is_special_reg_class(RegClass::PC));
}

TEST(RegisterSetSpecial, OrdinaryAndSpecialInsertionCoexist) {
  RegisterSet s;
  s.expand({RegClass::SGPR, 4, 2});
  s.expand({RegClass::EXEC, 0, 1});
  s.expand({RegClass::VCC, 0, 1});

  // size() counts ordinary lanes plus special singletons; ordinary_size() sees
  // only the two SGPR lanes.
  EXPECT_EQ(s.ordinary_size(), 2u);
  EXPECT_EQ(s.size(), 4u);
  EXPECT_TRUE(s.has_specials());
  EXPECT_TRUE(s.contains({RegClass::SGPR, 4, 1}));
  EXPECT_TRUE(s.contains({RegClass::EXEC, 0, 1}));
  EXPECT_EQ(specials_in(s), special_regs({RegClass::EXEC, RegClass::VCC}));
}

TEST(RegisterSetSpecial, SpecialMembershipIgnoresIndexAndWidth) {
  // Special classes are singletons: any index/width refers to the same member.
  RegisterSet s;
  s.expand({RegClass::EXEC, 42, 8});
  EXPECT_TRUE(s.contains({RegClass::EXEC, 0, 1}));
  EXPECT_TRUE(s.contains({RegClass::EXEC, 99, 2}));
  EXPECT_EQ(s.size(), 1u);

  s.erase({RegClass::EXEC, 7, 4});
  EXPECT_FALSE(s.contains({RegClass::EXEC, 0, 1}));
  EXPECT_TRUE(s.none());
}

TEST(RegisterSetSpecial, FullIterationEmitsCanonicalSpecialRefs) {
  RegisterSet s;
  s.expand({RegClass::VGPR, 1, 1});
  s.expand({RegClass::PC, 5, 2});

  // for_each visits the ordinary lane first, then the special as a canonical
  // {cls, 0, 1} ref regardless of the index/width it was inserted with.
  std::vector<RegisterRef> seen;
  s.for_each([&](RegisterRef r) { seen.push_back(r); });
  ASSERT_EQ(seen.size(), 2u);
  EXPECT_EQ(seen[0], (RegisterRef{RegClass::VGPR, 1, 1}));
  EXPECT_EQ(seen[1], (RegisterRef{RegClass::PC, 0, 1}));

  // for_each_ordinary skips the special member entirely.
  std::vector<RegisterRef> ordinary;
  s.for_each_ordinary([&](RegisterRef r) { ordinary.push_back(r); });
  EXPECT_EQ(ordinary, (std::vector<RegisterRef>{{RegClass::VGPR, 1, 1}}));
}

TEST(RegisterSetSpecial, OrdinaryOnlyDropsSpecialMembers) {
  RegisterSet s;
  s.expand({RegClass::SGPR, 3, 1});
  s.expand({RegClass::SCC, 0, 1});

  const RegisterSet ordinary = s.ordinary_only();
  EXPECT_FALSE(ordinary.has_specials());
  EXPECT_EQ(ordinary.size(), 1u);
  EXPECT_TRUE(ordinary.contains({RegClass::SGPR, 3, 1}));
  EXPECT_TRUE(s.has_specials()); // source unchanged
}

TEST(RegisterSetSpecial, SetAlgebraSpansOrdinaryAndSpecial) {
  RegisterSet a;
  a.expand({RegClass::SGPR, 0, 1});
  a.expand({RegClass::EXEC, 0, 1});
  RegisterSet b;
  b.expand({RegClass::SGPR, 1, 1});
  b.expand({RegClass::EXEC, 0, 1});
  b.expand({RegClass::VCC, 0, 1});

  EXPECT_TRUE(a.intersects(b)); // shared EXEC

  const RegisterSet u = a | b;
  EXPECT_EQ(u.size(), 4u); // s0, s1, EXEC, VCC
  EXPECT_EQ(specials_in(u), special_regs({RegClass::EXEC, RegClass::VCC}));

  const RegisterSet i = a & b;
  EXPECT_EQ(specials_in(i), special_regs({RegClass::EXEC}));
  EXPECT_FALSE(i.contains({RegClass::SGPR, 0, 1})); // s0 only in a

  const RegisterSet d = b - a; // removes shared EXEC; keeps VCC and s1
  EXPECT_EQ(specials_in(d), special_regs({RegClass::VCC}));
  EXPECT_TRUE(d.contains({RegClass::SGPR, 1, 1}));

  RegisterSet cleared = u;
  cleared.clear_class(RegClass::EXEC);
  EXPECT_EQ(specials_in(cleared), special_regs({RegClass::VCC}));
  EXPECT_EQ(cleared.ordinary_size(), 2u); // clear_class(special) leaves ordinary intact
}

TEST(RegisterSetSpecial, EqualityDistinguishesSpecialMembership) {
  RegisterSet a;
  a.expand({RegClass::SGPR, 0, 1});
  RegisterSet b = a;
  EXPECT_EQ(a, b);
  b.expand({RegClass::EXEC, 0, 1});
  EXPECT_NE(a, b); // the special mask participates in equality
}

TEST(RegisterSetSpecial, HighValuedSpecialMaskBitsRoundTrip) {
  // FLAT_SCRATCH and PC are the highest RegClass values, where a mask-width bug
  // would first surface.
  RegisterSet s;
  s.expand({RegClass::FLAT_SCRATCH, 0, 1});
  s.expand({RegClass::PC, 0, 1});
  EXPECT_EQ(specials_in(s), special_regs({RegClass::FLAT_SCRATCH, RegClass::PC}));
  EXPECT_FALSE(s.contains({RegClass::SGPR, 0, 1})); // no aliasing onto bit 0
}

TEST(InstDefUseSpecialEffects, SpecialDefaultEmpty) {
  // A decoded instruction with only ordinary operands reports no special
  // members; ordinary def/use extraction is unchanged.
  TestInstruction inst("test_sp", {{RegClass::SGPR, 4, 1}}, {{RegClass::VGPR, 0, 1}});
  InstDefUse du(inst);

  EXPECT_TRUE(du.defs.contains({RegClass::SGPR, 4, 1}));
  EXPECT_TRUE(du.uses.contains({RegClass::VGPR, 0, 1}));
  EXPECT_FALSE(du.defs.has_specials());
  EXPECT_FALSE(du.uses.has_specials());
}

TEST(InstDefUseSpecialEffects, SpecialMembersStayOutOfOrdinaryProjection) {
  // Special singletons live in defs/uses alongside ordinary registers, but the
  // ordinary projection that drives scratch allocation is unaffected.
  TestInstruction inst("test_sp", {{RegClass::SGPR, 4, 1}}, {});
  InstDefUse du(inst);

  du.defs.expand({RegClass::EXEC, 0, 1});
  du.uses.expand({RegClass::VCC, 0, 1});

  EXPECT_EQ(du.defs.ordinary_size(), 1u);
  EXPECT_TRUE(du.defs.ordinary_only().contains({RegClass::SGPR, 4, 1}));
  EXPECT_FALSE(du.defs.ordinary_only().has_specials());
  EXPECT_TRUE(du.defs.contains({RegClass::EXEC, 0, 1}));
  EXPECT_TRUE(du.uses.contains({RegClass::VCC, 0, 1}));
}

TEST(SpecialEffectAnalysis, SaveexecReportsExecAndSccFromDecodedOperands) {
  // Real decoded instruction: s_and_saveexec_b64 s[0:1], s[0:1] (CDNA4). Its
  // fieldless special operands become singleton members of defs/uses by
  // direction:
  //   dst EXEC + dst SCC  -> special members of defs
  //   src EXEC (old exec) -> special member of uses
  // while the ordinary sdst/ssrc0 SGPRs stay in the same sets as ordinary
  // members. The ordinary projection (ordinary_only) sees only the SGPRs.
  const uint32_t words[] = {0xBE802000u, 0x00000000u};
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(inst->mnemonic(), "s_and_saveexec_b64");

  InstDefUse du(*inst);

  EXPECT_EQ(specials_in(du.defs), special_regs({RegClass::EXEC, RegClass::SCC}));
  EXPECT_EQ(specials_in(du.uses), special_regs({RegClass::EXEC}));

  // Ordinary sdst/ssrc0 (s[0:1]) still tracked as ordinary registers.
  EXPECT_TRUE(du.defs.contains({RegClass::SGPR, 0, 2}));
  EXPECT_TRUE(du.uses.contains({RegClass::SGPR, 0, 2}));

  // EXEC/SCC are members of du.defs but stay out of the ordinary projection
  // that feeds scratch allocation.
  EXPECT_TRUE(du.defs.contains({RegClass::EXEC, 0, 1}));
  EXPECT_FALSE(du.defs.ordinary_only().has_specials());
}

// Counterpart to the fieldless case above: a special register named through a
// *generic selector* field (s_mov_b32 exec_lo/exec_hi, where EXEC_LO/HI are
// encoding values 126/127 in the OPR_SDST selector) decodes to a special
// RegisterRef, but its class-wide singleton representation would collapse the
// LO/HI halves. InstDefUse deliberately drops such selector-encoded specials
// rather than record them imprecisely, so EXEC appears in neither defs nor uses.
TEST(SpecialEffectAnalysis, SelectorEncodedExecWriteIsNotSurfaced) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA2);
  ASSERT_NE(decoder, nullptr);

  for (const uint32_t word :
       {0xbefe0080u /* s_mov_b32 exec_lo, 0 */, 0xbeff0080u /* s_mov_b32 exec_hi, 0 */}) {
    const std::array<uint32_t, 2> words{word, 0u};
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
    ASSERT_NE(inst, nullptr);
    ASSERT_EQ(inst->mnemonic(), "s_mov_b32") << std::hex << word;

    InstDefUse du(*inst);
    EXPECT_FALSE(du.defs.has_specials()) << std::hex << word;
    EXPECT_FALSE(du.uses.has_specials()) << std::hex << word;
    // The dropped EXEC ref must not leak into the ordinary files either.
    EXPECT_EQ(du.defs.ordinary_size(), 0u) << std::hex << word;
  }
}

// The selector-drop policy must also cover implicit reads. s_cvt_f16_f32 writes
// the low 16 bits of sdst and preserves the high half, so its generated
// implicit_uses() hook surfaces a preserve-read of sdst via to_register_ref().
// With an ordinary sdst that read is a real ordinary use; with a selector-encoded
// special sdst (exec_lo) it collapses to a class-wide EXEC singleton and must be
// dropped, or EXEC would be used-but-never-defined (the explicit dst path already
// drops the selector-encoded def).
TEST(SpecialEffectAnalysis, ImplicitPreserveReadDropsSelectorEncodedSpecial) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);

  // Ordinary destination: the implicit preserve-read of s4 is a real use.
  {
    const auto built = rdna4::build_sop1(rdna4::kSCvtF16F32Sop1, {.ssrc0 = 0, .sdst = 4});
    const std::array<uint32_t, 2> words{built[0], 0u};
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
    ASSERT_NE(inst, nullptr);
    ASSERT_EQ(inst->mnemonic(), "s_cvt_f16_f32");

    InstDefUse du(*inst);
    EXPECT_TRUE(du.defs.contains({RegClass::SGPR, 4, 1}));
    EXPECT_TRUE(du.uses.contains({RegClass::SGPR, 0, 1}));
    EXPECT_TRUE(du.uses.contains({RegClass::SGPR, 4, 1})); // preserve-read via the hook
    EXPECT_FALSE(du.defs.has_specials());
    EXPECT_FALSE(du.uses.has_specials());
  }

  // Selector-encoded exec_lo destination (OPR_SDST value 126): the preserve-read
  // decodes to a class-wide EXEC ref and must be dropped from uses, and the def
  // is likewise dropped, leaving no special members and no ordinary destination.
  {
    const auto built = rdna4::build_sop1(rdna4::kSCvtF16F32Sop1, {.ssrc0 = 0, .sdst = 126});
    const std::array<uint32_t, 2> words{built[0], 0u};
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
    ASSERT_NE(inst, nullptr);
    ASSERT_EQ(inst->mnemonic(), "s_cvt_f16_f32");

    InstDefUse du(*inst);
    EXPECT_FALSE(du.uses.has_specials());
    EXPECT_FALSE(du.defs.has_specials());
    EXPECT_EQ(du.defs.ordinary_size(), 0u); // selector-encoded dst dropped
    EXPECT_EQ(du.uses.ordinary_size(), 1u); // only the ordinary ssrc0 (s0) remains
    EXPECT_TRUE(du.uses.contains({RegClass::SGPR, 0, 1}));
  }
}

TEST(SpecialEffectAnalysis, OrdinaryInstructionReportsNoSpecialEffects) {
  // A plain vector op with only ordinary register operands exposes no special
  // effects -- both special sets stay empty.
  constexpr std::array<uint32_t, 2> kWritelaneV141S4Lane2 = {0xd28a008du, 0x00010404u};
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, kWritelaneV141S4Lane2.data()));
  ASSERT_NE(inst, nullptr);

  InstDefUse du(*inst);
  EXPECT_FALSE(du.defs.has_specials());
  EXPECT_FALSE(du.uses.has_specials());
}

// Decode-time special-effect coverage for the hidden-side-effect instruction
// families. Encodings are built through the generated per-ISA encoders so they
// cannot silently drift, and each case locks the operand-driven special
// defs/uses so a change elsewhere cannot alter them unnoticed.

// V_CMPX writes EXEC on every ISA, and additionally VCC on CDNA/GFX9 (its
// operand list carries a VCC destination). gfx1250 is EXEC-only, so the same
// mnemonic has a genuinely different side-effect set across ISAs.
TEST(SpecialEffectAnalysis, CmpxWritesVccAndExecOnCdna) {
  const auto built = cdna3::build_vopc(cdna3::kVCmpxEqF32Vopc, {.src0 = 256, .vsrc1 = 1});
  const std::array<uint32_t, 2> words{built[0], 0u};
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(inst->mnemonic(), "v_cmpx_eq_f32_e32");

  InstDefUse du(*inst);
  EXPECT_EQ(specials_in(du.defs), special_regs({RegClass::EXEC, RegClass::VCC}));
  EXPECT_FALSE(du.uses.has_specials());
}

TEST(SpecialEffectAnalysis, CmpxWritesExecOnlyOnGfx1250) {
  const auto built = cdna5::build_vopc(cdna5::kVCmpxEqF32Vopc, {.src0 = 256, .vsrc1 = 1});
  const std::array<uint32_t, 2> words{built[0], 0u};
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(inst->mnemonic(), "v_cmpx_eq_f32_e32");

  InstDefUse du(*inst);
  EXPECT_EQ(specials_in(du.defs), special_regs({RegClass::EXEC}));
  EXPECT_FALSE(du.uses.has_specials());
}

// RDNA CMPX is EXEC-only like gfx1250 (and unlike CDNA), tested directly on an
// RDNA target rather than inferred from the CDNA-family gfx1250.
TEST(SpecialEffectAnalysis, CmpxWritesExecOnlyOnRdna) {
  const auto built = rdna4::build_vopc(rdna4::kVCmpxEqF32Vopc, {.src0 = 256, .vsrc1 = 1});
  const std::array<uint32_t, 2> words{built[0], 0u};
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(inst->mnemonic(), "v_cmpx_eq_f32_e32");

  InstDefUse du(*inst);
  EXPECT_EQ(specials_in(du.defs), special_regs({RegClass::EXEC}));
  EXPECT_FALSE(du.uses.has_specials());
}

// s_cbranch_{scc,vcc,exec}* read a special condition register and branch. The
// condition operand is a special use; nothing is defined.
TEST(SpecialEffectAnalysis, CbranchSccIsSpecialSccUse) {
  const uint32_t word = cdna3::build_sopp(cdna3::kSCbranchScc0Sopp, {.simm16 = 0})[0];
  const std::array<uint32_t, 2> words{word, 0u};
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(inst->mnemonic(), "s_cbranch_scc0");

  InstDefUse du(*inst);
  EXPECT_EQ(specials_in(du.uses), special_regs({RegClass::SCC}));
  EXPECT_FALSE(du.defs.has_specials());
}

TEST(SpecialEffectAnalysis, CbranchVccIsSpecialVccUse) {
  const uint32_t word = cdna3::build_sopp(cdna3::kSCbranchVcczSopp, {.simm16 = 0})[0];
  const std::array<uint32_t, 2> words{word, 0u};
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(inst->mnemonic(), "s_cbranch_vccz");

  InstDefUse du(*inst);
  EXPECT_EQ(specials_in(du.uses), special_regs({RegClass::VCC}));
  EXPECT_FALSE(du.defs.has_specials());
}

TEST(SpecialEffectAnalysis, CbranchExecIsSpecialExecUse) {
  const uint32_t word = cdna3::build_sopp(cdna3::kSCbranchExeczSopp, {.simm16 = 0})[0];
  const std::array<uint32_t, 2> words{word, 0u};
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(inst->mnemonic(), "s_cbranch_execz");

  InstDefUse du(*inst);
  EXPECT_EQ(specials_in(du.uses), special_regs({RegClass::EXEC}));
  EXPECT_FALSE(du.defs.has_specials());
}

// VOP2 carry reads and writes VCC implicitly. The mnemonic differs by ISA
// (CDNA/GFX9 v_addc_co_u32 vs RDNA/gfx1250 v_add_co_ci_u32) but both expose the
// same VCC carry-in use and carry-out def.
TEST(SpecialEffectAnalysis, Vop2CarryCdnaAddcUsesAndDefsVcc) {
  const auto built =
      cdna3::build_vop2(cdna3::kVAddcCoU32Vop2, {.src0 = 256, .vsrc1 = 1, .vdst = 2});
  const std::array<uint32_t, 2> words{built[0], 0u};
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(inst->mnemonic(), "v_addc_co_u32_e32");

  InstDefUse du(*inst);
  EXPECT_EQ(specials_in(du.defs), special_regs({RegClass::VCC}));
  EXPECT_EQ(specials_in(du.uses), special_regs({RegClass::VCC}));
  EXPECT_TRUE(du.defs.contains({RegClass::VGPR, 2, 1})); // ordinary vdst still tracked
}

TEST(SpecialEffectAnalysis, Vop2CarryGfx1250AddCoCiUsesAndDefsVcc) {
  const auto built =
      cdna5::build_vop2(cdna5::kVAddCoCiU32Vop2, {.src0 = 256, .vsrc1 = 1, .vdst = 2});
  const std::array<uint32_t, 2> words{built[0], 0u};
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(inst->mnemonic(), "v_add_co_ci_u32_e32");

  InstDefUse du(*inst);
  EXPECT_EQ(specials_in(du.defs), special_regs({RegClass::VCC}));
  EXPECT_EQ(specials_in(du.uses), special_regs({RegClass::VCC}));
}

// PC family: getpc reads PC (writes an ordinary SGPR pair), setpc writes PC
// (reads an ordinary SGPR pair), s_call does both.
TEST(SpecialEffectAnalysis, GetpcReadsPc) {
  const uint32_t word = cdna3::build_sop1(0x1c, {.ssrc0 = 0, .sdst = 8})[0]; // s_getpc_b64 s[8:9]
  const std::array<uint32_t, 2> words{word, 0u};
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(inst->mnemonic(), "s_getpc_b64");

  InstDefUse du(*inst);
  EXPECT_EQ(specials_in(du.uses), special_regs({RegClass::PC}));
  EXPECT_FALSE(du.defs.has_specials());
  EXPECT_TRUE(du.defs.contains({RegClass::SGPR, 8, 2}));
}

TEST(SpecialEffectAnalysis, SetpcWritesPc) {
  const uint32_t word = cdna3::build_sop1(0x1d, {.ssrc0 = 8, .sdst = 0})[0]; // s_setpc_b64 s[8:9]
  const std::array<uint32_t, 2> words{word, 0u};
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(inst->mnemonic(), "s_setpc_b64");

  InstDefUse du(*inst);
  EXPECT_EQ(specials_in(du.defs), special_regs({RegClass::PC}));
  EXPECT_FALSE(du.uses.has_specials());
  EXPECT_TRUE(du.uses.contains({RegClass::SGPR, 8, 2}));
}

TEST(SpecialEffectAnalysis, ScallReadsAndWritesPc) {
  const uint32_t word = build_s_call_b64(8, 0); // s_call_b64 s[8:9], <rel>
  const std::array<uint32_t, 2> words{word, 0u};
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(inst->mnemonic(), "s_call_b64");

  InstDefUse du(*inst);
  EXPECT_EQ(specials_in(du.defs), special_regs({RegClass::PC}));
  EXPECT_EQ(specials_in(du.uses), special_regs({RegClass::PC}));
  EXPECT_TRUE(du.defs.contains({RegClass::SGPR, 8, 2}));
}

// M0 family: s_movrels_b32 reads M0 as its relative index (special use), while
// s_set_gpr_idx_idx both reads M0 and writes it back (special use and def). The
// M0 operand is fieldless on both, so it surfaces only through the special path.
TEST(SpecialEffectAnalysis, MovrelsReadsM0) {
  const uint32_t word = cdna3::build_sop1(cdna3::kSMovrelsB32Sop1, {.ssrc0 = 0, .sdst = 8})[0];
  const std::array<uint32_t, 2> words{word, 0u};
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(inst->mnemonic(), "s_movrels_b32");

  InstDefUse du(*inst);
  EXPECT_EQ(specials_in(du.uses), special_regs({RegClass::M0}));
  EXPECT_FALSE(du.defs.has_specials());
  EXPECT_TRUE(du.defs.contains({RegClass::SGPR, 8, 1})); // ordinary sdst still tracked
}

TEST(SpecialEffectAnalysis, SetGprIdxIdxReadsAndWritesM0) {
  const uint32_t word = cdna3::build_sop1(cdna3::kSSetGprIdxIdxSop1, {.ssrc0 = 0})[0];
  const std::array<uint32_t, 2> words{word, 0u};
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(inst->mnemonic(), "s_set_gpr_idx_idx");

  InstDefUse du(*inst);
  EXPECT_EQ(specials_in(du.defs), special_regs({RegClass::M0}));
  EXPECT_EQ(specials_in(du.uses), special_regs({RegClass::M0}));
}

// A buffer load's fieldless GPUMEM operand is inert: it yields neither a special
// register class nor an ordinary register ref, so it contributes nothing to
// defs/uses.
TEST(SpecialEffectAnalysis, MemoryPseudoOperandProducesNoSpecialEffect) {
  const auto words = cdna5::build_vbuffer(cdna5::kBufferLoadB32Vbuffer, {.vdata = 1, .rsrc = 0});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(inst->mnemonic(), "buffer_load_b32");

  InstDefUse du(*inst);
  EXPECT_FALSE(du.defs.has_specials());
  EXPECT_FALSE(du.uses.has_specials());
}

// Liveness models ordinary scratch registers only: an instruction's special
// def/use appears in InstDefUse but must never reach BlockLiveness or
// live_before(), which drive scratch allocation.
TEST(SpecialEffectLiveness, SpecialEffectsAreAbsentFromOrdinaryLiveness) {
  // s_and_saveexec_b64 s[0:1], s[0:1] defs {s[0:1], EXEC, SCC} and uses
  // {s[0:1], EXEC}; s_endpgm terminates the single block.
  std::vector<uint32_t> words = {
      0xBE802000u, // s_and_saveexec_b64 s[0:1], s[0:1]
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_FALSE(blocks.empty());

  BasicBlock *entry = block_starting_at(blocks, 0);
  ASSERT_NE(entry, nullptr);
  auto &insts = entry->instructions();
  ASSERT_NE(insts.begin(), insts.end());
  const Instruction &saveexec = *insts.begin();
  ASSERT_EQ(saveexec.mnemonic(), "s_and_saveexec_b64");

  // InstDefUse records the special effects.
  const InstDefUse du(saveexec);
  EXPECT_TRUE(du.defs.contains({RegClass::EXEC, 0, 1}));
  EXPECT_TRUE(du.defs.contains({RegClass::SCC, 0, 1}));
  EXPECT_TRUE(du.uses.contains({RegClass::EXEC, 0, 1}));

  // Liveness projects them out of every ordinary set.
  const LivenessAnalysis liveness = analyze_scope(blocks);
  const BlockLiveness &bl = liveness.block_liveness(*entry);
  EXPECT_FALSE(bl.live_in.has_specials());
  EXPECT_FALSE(bl.live_out.has_specials());
  EXPECT_FALSE(bl.gen.has_specials());
  EXPECT_FALSE(bl.kill.has_specials());
  EXPECT_FALSE(liveness.live_before(saveexec).has_specials());
}

TEST(CfgAnalysis, LoopBackEdgeLinksPredecessor) {
  auto blocks = build_test_blocks({TestOpcode::Nop, TestOpcode::BranchBackToStart});

  ASSERT_EQ(blocks.size(), 1u);
  ASSERT_EQ(blocks[0]->successors().size(), 1u);
  EXPECT_EQ(blocks[0]->successors()[0], blocks[0].get());
  EXPECT_TRUE(has_predecessor(*blocks[0], blocks[0].get()));
}

TEST(CfgAnalysis, Gfx1250ZeroPaddingTerminatesFallthrough) {
  struct Case {
    const char *name;
    std::vector<uint32_t> words;
    bool has_implicit_terminator;
  };
  const std::array cases = {
      Case{"setup followed by padding", {0xb9800641u, 1u, 0}, true},
      Case{"setup sequence followed by padding",
           {0xee174000u, 0x00040000u, 0, 0x7e000000u, 0xb9800641u, 1u, 0},
           true},
      Case{
          "ordinary fallthrough into padding", {build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5), 0}, true},
      Case{"architectural terminator followed by padding",
           {build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), 0},
           false},
      Case{"unconditional branch followed by padding",
           {build_s_branch(-1, ROCJITSU_CODE_ARCH_CDNA5), 0},
           false},
  };

  for (const auto &test_case : cases) {
    SCOPED_TRACE(test_case.name);
    TestCodeObject co(test_case.words);
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
    ASSERT_NE(decoder, nullptr);
    auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);

    ASSERT_EQ(blocks.size(), 1u);
    EXPECT_EQ(blocks[0]->has_implicit_terminator(), test_case.has_implicit_terminator);
  }
}

TEST(CfgAnalysis, Gfx1250ConditionalBranchKeepsTakenEdgeWhenPaddingTerminatesFallthrough) {
  constexpr uint32_t kSCbranchScc0PlusOne = 0xbfa10001u;
  std::vector<uint32_t> words = {
      kSCbranchScc0PlusOne, // 0x00 -> 0x08; fallthrough reaches padding at 0x04.
      0,
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5),
  };
  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);

  ASSERT_EQ(blocks.size(), 2u);
  EXPECT_TRUE(blocks[0]->has_implicit_terminator());
  ASSERT_EQ(blocks[0]->successors().size(), 1u);
  EXPECT_EQ(blocks[0]->successors()[0], blocks[1].get());
}

TEST(CfgAnalysis, DirectCallToImplicitNonreturningTargetDropsFallthrough) {
  constexpr uint16_t kReturnSreg = 30;
  std::vector<uint32_t> words = {
      rocjitsu::build_s_call_b64(kReturnSreg, 1, ROCJITSU_CODE_ARCH_CDNA5),
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x04 continuation.
      0xb9800641u,
      1u,
      0, // 0x08 clang unreachable-stub target followed by padding.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);

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

// An implicit terminator cuts the fallthrough edge only. This callee is a conditional branch whose
// fallthrough runs into padding but whose TAKEN target leaves the section, so where control goes is
// not known. Treating the whole block as a program exit would skip the missing-target check, call
// the callee non-returning, and delete the caller's continuation -- silently removing the path the
// program actually takes when the branch is not taken.
TEST(CfgAnalysis, DirectCallToPaddingTerminatedBlockWithUnresolvedTakenTargetKeepsFallthrough) {
  constexpr uint16_t kReturnSreg = 30;
  constexpr uint32_t kSCbranchScc0FarOutOfSection = 0xbfa11000u;
  std::vector<uint32_t> words = {
      rocjitsu::build_s_call_b64(kReturnSreg, 1, ROCJITSU_CODE_ARCH_CDNA5),
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x04 continuation.
      kSCbranchScc0FarOutOfSection,             // 0x08 callee: taken target is not in .text.
      0,                                        // 0x0c padding terminates the fallthrough.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);

  auto *caller = block_starting_at(blocks, 0);
  auto *continuation = block_starting_at(blocks, 4);
  ASSERT_NE(caller, nullptr);
  ASSERT_NE(continuation, nullptr);

  EXPECT_TRUE(has_successor_start(*caller, continuation->start_offset()))
      << "an unresolved taken target must keep the callee's return path unknown";
  EXPECT_TRUE(has_predecessor(*continuation, caller));
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
  std::unique_ptr<Instruction> instruction(decode_valid(*decoder, &kNop));

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
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4, extra_leaders);

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
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

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
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

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
    auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

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
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4, extra_leaders);

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
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

  auto *consumer = block_starting_at(blocks, 28);
  ASSERT_NE(consumer, nullptr);
  ASSERT_EQ(consumer->static_indirect_call_fixups().size(), 1u);
  const auto &fixup = consumer->static_indirect_call_fixups()[0];
  EXPECT_EQ(fixup.source_target_offset, 36u);
  EXPECT_TRUE(fixup.source_incomplete)
      << "a consumer joined from an unconstrained path must be flagged incomplete";
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
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

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
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

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
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

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
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

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
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

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
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

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
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

  auto *consumer = block_starting_at(blocks, 24);
  ASSERT_NE(consumer, nullptr);
  ASSERT_EQ(consumer->static_indirect_call_fixups().size(), 1u);
  EXPECT_EQ(consumer->static_indirect_call_fixups()[0].source_target_offset, 28u);
  EXPECT_FALSE(consumer->static_indirect_call_fixups()[0].source_incomplete);
}

TEST(CfgAnalysis, MultipleUnorderedExplicitEntriesMakeIncomingPcBuilderIncomplete) {
  constexpr uint16_t kPcSreg = 8;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;

  // Entry A builds a static target and branches into entries B and C. Both are
  // separately launchable kernels, so their externally supplied s[8:9] values
  // must participate in the joins with A's concrete builder. Supply the entry
  // offsets out of order with a duplicate to exercise the ordered merge.
  std::vector<uint32_t> words = {
      pack_sop1(0x1c, kPcSreg, 0),                         // 0x00: s_getpc_b64.
      pack_sop2(0, kPcSreg, kPcSreg, kLiteralOperand),     // 0x04: s_add_u32.
      32,                                                  // 0x08: 0x04 + 32 = 0x24.
      pack_sop2(4, kPcSreg + 1, kPcSreg + 1, kInlineInt0), // 0x0c: s_addc_u32.
      pack_sopp(5, 2),                                     // 0x10: cbranch -> entry C at 0x1c.
      build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA4),         // 0x14 -> entry B at 0x18.
      pack_sop1(0x1d, 0, kPcSreg),                         // 0x18: entry B setpc.
      pack_sop1(0x1d, 0, kPcSreg),                         // 0x1c: entry C setpc.
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),            // 0x20: not a target.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),            // 0x24: A's target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  constexpr std::array<uint64_t, 3> extra_leaders{28, 24, 28};
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4, extra_leaders);

  for (uint64_t consumer_offset : {uint64_t{24}, uint64_t{28}}) {
    auto *consumer = block_starting_at(blocks, consumer_offset);
    ASSERT_NE(consumer, nullptr);
    ASSERT_EQ(consumer->static_indirect_call_fixups().size(), 1u);
    EXPECT_EQ(consumer->static_indirect_call_fixups()[0].source_target_offset, 36u);
    EXPECT_TRUE(consumer->static_indirect_call_fixups()[0].source_incomplete)
        << "each independently launchable entry must include unconstrained external SGPR state";
  }
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
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4, extra_leaders);

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
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4, {},
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
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

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
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

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
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

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
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

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
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

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
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

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
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4, extra_leaders);

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
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

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
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

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
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

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
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

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
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

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
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

  auto *continuation = block_starting_at(blocks, 20);
  auto *stale_target = block_starting_at(blocks, 32);
  ASSERT_NE(continuation, nullptr);
  ASSERT_NE(stale_target, nullptr);

  EXPECT_TRUE(continuation->static_indirect_call_fixups().empty());
  EXPECT_FALSE(has_successor_start(*continuation, stale_target->start_offset()));
}

TEST(CfgAnalysis, EitherHalfKillPredecessorPreventsRecoveredConsumer) {
  constexpr uint16_t kPcSreg = 8;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;
  constexpr uint32_t kOriginalGetpcDelta = 32;

  // Two paths reach the same setpc consumer:
  //
  //   * the fallthrough path builds a concrete PC target in s[8:9]
  //   * the branch path writes either s8 or s9 through ordinary scalar code,
  //     killing that pair for this analysis
  //
  // The concrete builder path alone is not enough to recover the consumer. A
  // real unmodeled write reaches the join, so the analysis must fail closed and
  // leave the setpc for the later DBT diagnostic.
  constexpr std::array<uint16_t, 2> clobbered_sregs{kPcSreg, kPcSreg + 1};
  for (uint16_t clobbered_sreg : clobbered_sregs) {
    SCOPED_TRACE(clobbered_sreg);
    std::vector<uint32_t> words = {
        pack_sopp(5, 5),                                           // 0x00 -> kill path at 0x18.
        pack_sop1(0x1c, kPcSreg, 0),                               // 0x04: s_getpc_b64.
        pack_sop2(0, kPcSreg, kPcSreg, kLiteralOperand),           // 0x08: s_add_u32.
        kOriginalGetpcDelta,                                       // 0x0c: target delta.
        pack_sop2(4, kPcSreg + 1, kPcSreg + 1, kInlineInt0),       // 0x10: s_addc_u32.
        build_s_branch(2, ROCJITSU_CODE_ARCH_CDNA4),               // 0x14 -> consumer at 0x20.
        pack_sop2(0, clobbered_sreg, clobbered_sreg, kInlineInt0), // 0x18: write.
        build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA4),               // 0x1c -> consumer.
        pack_sop1(0x1d, 0, kPcSreg),                               // 0x20: consumer.
        build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),                  // 0x24: not a target.
        build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),                  // 0x28: target.
    };

    TestCodeObject co(std::move(words));
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    ASSERT_NE(decoder, nullptr);
    constexpr std::array<uint64_t, 1> extra_leaders{40};
    auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4, extra_leaders);

    auto *consumer = block_starting_at(blocks, 32);
    auto *target = block_starting_at(blocks, 40);
    ASSERT_NE(consumer, nullptr);
    ASSERT_NE(target, nullptr);

    EXPECT_TRUE(consumer->static_indirect_call_fixups().empty());
    EXPECT_FALSE(has_successor_start(*consumer, target->start_offset()));
  }
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
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

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
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

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
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

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
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);

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
      cdna5::build_sop1(cdna5::kSGetPcI64Sop1,
                        {.ssrc0 = 0, .sdst = kPcSreg})[0], // 0x00: s_get_pc_i64 s[8:9].
      cdna5::build_sop2(cdna5::kSAddCoI32Sop2,
                        {.ssrc0 = kLiteralOperand, .ssrc1 = kInlineInt4, .sdst = kTmpSreg})[0],
      // 0x04: s_add_co_i32.
      kSignedDeltaLiteral, // 0x08: literal.
      cdna5::build_sopc(cdna5::kSCmpGeI32Sopc,
                        {.ssrc0 = kTmpSreg, .ssrc1 = kInlineInt0})[0], // 0x0c: s_cmp_ge_i32.
      cdna5::build_sopp(cdna5::kSCbranchScc1Sopp, {.simm16 = 7})[0],
      // 0x10 -> add half at 0x30.
      cdna5::build_sop1(cdna5::kSMovB32Sop1,
                        {.ssrc0 = 159, .sdst = 14})[0], // 0x14: s_mov_b32 s14, 31.
      0xF404A000u,
      0x1C000000u, // 0x18: s_prefetch_inst_pc_rel.
      cdna5::build_sop1(cdna5::kSAbsI32Sop1,
                        {.ssrc0 = kTmpSreg, .sdst = kTmpSreg})[0], // 0x20: s_abs_i32.
      cdna5::build_sop2(cdna5::kSSubCoU32Sop2,
                        {.ssrc0 = kPcSreg, .ssrc1 = kTmpSreg, .sdst = kPcSreg})[0],
      // 0x24: s_sub_co_u32.
      cdna5::build_sop2(cdna5::kSSubCoCiU32Sop2,
                        {.ssrc0 = kPcSreg + 1, .ssrc1 = kInlineInt0, .sdst = kPcSreg + 1})[0],
      // 0x28: s_sub_co_ci_u32.
      cdna5::build_sop1(cdna5::kSSetPcI64Sop1,
                        {.ssrc0 = kPcSreg, .sdst = 0})[0], // 0x2c: s_set_pc_i64.
      cdna5::build_sop2(cdna5::kSAddCoU32Sop2,
                        {.ssrc0 = kPcSreg, .ssrc1 = kTmpSreg, .sdst = kPcSreg})[0],
      // 0x30: s_add_co_u32.
      cdna5::build_sop1(cdna5::kSMovB32Sop1,
                        {.ssrc0 = 159, .sdst = 14})[0], // 0x34: s_mov_b32 s14, 31.
      0xF404A000u,
      0x1C000000u, // 0x38: s_prefetch_inst_pc_rel.
      cdna5::build_sop2(cdna5::kSAddCoCiU32Sop2,
                        {.ssrc0 = kPcSreg + 1, .ssrc1 = kInlineInt0, .sdst = kPcSreg + 1})[0],
      // 0x40: s_add_co_ci_u32.
      cdna5::build_sop1(cdna5::kSSetPcI64Sop1,
                        {.ssrc0 = kPcSreg, .sdst = 0})[0], // 0x44: s_set_pc_i64.
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5),            // 0x48: not a target.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5),            // 0x4c: shared target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);

  auto *sub_consumer = block_starting_at(blocks, 44);
  auto *add_consumer = block_starting_at(blocks, 68);
  auto *target = block_starting_at(blocks, 76);
  ASSERT_NE(sub_consumer, nullptr);
  ASSERT_NE(add_consumer, nullptr);
  ASSERT_NE(target, nullptr);

  ASSERT_EQ(sub_consumer->static_indirect_call_fixups().size(), 1u);
  EXPECT_EQ(sub_consumer->static_indirect_call_fixups()[0].source_target_offset, 76u);
  EXPECT_TRUE(has_successor_start(*sub_consumer, target->start_offset()));
  EXPECT_FALSE(sub_consumer->static_indirect_call_fixups()[0].source_requires_xcnt_drain);

  ASSERT_EQ(add_consumer->static_indirect_call_fixups().size(), 1u);
  EXPECT_EQ(add_consumer->static_indirect_call_fixups()[0].source_target_offset, 76u);
  EXPECT_TRUE(has_successor_start(*add_consumer, target->start_offset()));
  EXPECT_FALSE(add_consumer->static_indirect_call_fixups()[0].source_requires_xcnt_drain);
}

TEST(CfgAnalysis, Gfx1250RecoversSignedDeltaTemplateWithXcntWaitAndPrefetch) {
  constexpr uint16_t kPcSreg = 8;
  constexpr uint16_t kTmpSreg = 12;
  constexpr uint32_t kLiteralOperand = 255;
  constexpr uint32_t kInlineInt0 = 128;
  constexpr uint32_t kInlineInt4 = 132;
  constexpr uint32_t kSignedDeltaLiteral = 68;

  // gfx1250 compiler output drains XCNT immediately before each instruction
  // prefetch. These waits and prefetches do not modify the PC pair or temporary,
  // and must remain in the translated body while the two set-PC consumers are
  // recovered to their common static target.
  std::vector<uint32_t> words = {
      cdna5::build_sop1(cdna5::kSGetPcI64Sop1,
                        {.ssrc0 = 0, .sdst = kPcSreg})[0], // 0x00: s_get_pc_i64 s[8:9].
      cdna5::build_sop2(cdna5::kSAddCoI32Sop2,
                        {.ssrc0 = kLiteralOperand, .ssrc1 = kInlineInt4, .sdst = kTmpSreg})[0],
      // 0x04: s_add_co_i32.
      kSignedDeltaLiteral, // 0x08: literal.
      cdna5::build_sopc(cdna5::kSCmpGeI32Sopc,
                        {.ssrc0 = kTmpSreg, .ssrc1 = kInlineInt0})[0], // 0x0c: s_cmp_ge_i32.
      cdna5::build_sopp(cdna5::kSCbranchScc1Sopp, {.simm16 = 7})[0],
      // 0x10 -> add half at 0x30.
      cdna5::build_sopp(cdna5::kSWaitXcntSopp, {.simm16 = 0})[0],
      // 0x14: s_wait_xcnt 0.
      0xF404A000u,
      0x1C000000u, // 0x18: s_prefetch_inst_pc_rel.
      cdna5::build_sop1(cdna5::kSAbsI32Sop1,
                        {.ssrc0 = kTmpSreg, .sdst = kTmpSreg})[0], // 0x20: s_abs_i32.
      cdna5::build_sop2(cdna5::kSSubCoU32Sop2,
                        {.ssrc0 = kPcSreg, .ssrc1 = kTmpSreg, .sdst = kPcSreg})[0],
      // 0x24: s_sub_co_u32.
      cdna5::build_sop2(cdna5::kSSubCoCiU32Sop2,
                        {.ssrc0 = kPcSreg + 1, .ssrc1 = kInlineInt0, .sdst = kPcSreg + 1})[0],
      // 0x28: s_sub_co_ci_u32.
      cdna5::build_sop1(cdna5::kSSetPcI64Sop1,
                        {.ssrc0 = kPcSreg, .sdst = 0})[0], // 0x2c: subtract consumer.
      cdna5::build_sopp(cdna5::kSWaitXcntSopp, {.simm16 = 0})[0],
      // 0x30: s_wait_xcnt 0.
      0xF404A000u,
      0x1C000000u, // 0x34: s_prefetch_inst_pc_rel.
      cdna5::build_sop2(cdna5::kSAddCoU32Sop2,
                        {.ssrc0 = kPcSreg, .ssrc1 = kTmpSreg, .sdst = kPcSreg})[0],
      // 0x3c: s_add_co_u32.
      cdna5::build_sop2(cdna5::kSAddCoCiU32Sop2,
                        {.ssrc0 = kPcSreg + 1, .ssrc1 = kInlineInt0, .sdst = kPcSreg + 1})[0],
      // 0x40: s_add_co_ci_u32.
      cdna5::build_sop1(cdna5::kSSetPcI64Sop1,
                        {.ssrc0 = kPcSreg, .sdst = 0})[0], // 0x44: add consumer.
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5),            // 0x48: not a target.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5),            // 0x4c: shared target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);

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

  // The subtract half's drain is inside the range relocation overwrites, so
  // both consumers of that shared range must ask the rewrite to reproduce it.
  EXPECT_TRUE(sub_consumer->static_indirect_call_fixups()[0].source_requires_xcnt_drain);
  EXPECT_TRUE(add_consumer->static_indirect_call_fixups()[0].source_requires_xcnt_drain);
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
      cdna5::build_sop1(cdna5::kSGetPcI64Sop1,
                        {.ssrc0 = 0, .sdst = kPcSreg})[0], // 0x00: s_get_pc_i64 s[8:9].
      cdna5::build_sop2(cdna5::kSAddCoI32Sop2,
                        {.ssrc0 = kLiteralOperand, .ssrc1 = kInlineInt4, .sdst = kTmpSreg})[0],
      // 0x04: s_add_co_i32.
      kSignedDeltaLiteral, // 0x08: literal.
      cdna5::build_sopc(cdna5::kSCmpGeI32Sopc,
                        {.ssrc0 = kTmpSreg, .ssrc1 = kInlineInt0})[0], // 0x0c: s_cmp_ge_i32.
      cdna5::build_sopp(cdna5::kSCbranchScc1Sopp, {.simm16 = 7})[0],
      // 0x10 -> add half at 0x30.
      cdna5::build_sop1(cdna5::kSMovB32Sop1,
                        {.ssrc0 = 159, .sdst = kTmpSreg})[0], // 0x14: s_mov_b32 s12, 31 (CLOBBER).
      0xF404A000u,
      0x1C000000u, // 0x18: s_prefetch_inst_pc_rel.
      cdna5::build_sop1(cdna5::kSAbsI32Sop1,
                        {.ssrc0 = kTmpSreg, .sdst = kTmpSreg})[0], // 0x20: s_abs_i32.
      cdna5::build_sop2(cdna5::kSSubCoU32Sop2,
                        {.ssrc0 = kPcSreg, .ssrc1 = kTmpSreg, .sdst = kPcSreg})[0],
      // 0x24: s_sub_co_u32.
      cdna5::build_sop2(cdna5::kSSubCoCiU32Sop2,
                        {.ssrc0 = kPcSreg + 1, .ssrc1 = kInlineInt0, .sdst = kPcSreg + 1})[0],
      // 0x28: s_sub_co_ci_u32.
      cdna5::build_sop1(cdna5::kSSetPcI64Sop1,
                        {.ssrc0 = kPcSreg, .sdst = 0})[0], // 0x2c: s_set_pc_i64.
      cdna5::build_sop2(cdna5::kSAddCoU32Sop2,
                        {.ssrc0 = kPcSreg, .ssrc1 = kTmpSreg, .sdst = kPcSreg})[0],
      // 0x30: s_add_co_u32.
      cdna5::build_sop1(cdna5::kSMovB32Sop1,
                        {.ssrc0 = 159, .sdst = 14})[0], // 0x34: s_mov_b32 s14, 31.
      0xF404A000u,
      0x1C000000u, // 0x38: s_prefetch_inst_pc_rel.
      cdna5::build_sop2(cdna5::kSAddCoCiU32Sop2,
                        {.ssrc0 = kPcSreg + 1, .ssrc1 = kInlineInt0, .sdst = kPcSreg + 1})[0],
      // 0x40: s_add_co_ci_u32.
      cdna5::build_sop1(cdna5::kSSetPcI64Sop1,
                        {.ssrc0 = kPcSreg, .sdst = 0})[0], // 0x44: s_set_pc_i64.
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5),            // 0x48: not a target.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5),            // 0x4c: shared target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);

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
      0x0201032Cu,                              // 0x28: v_readlane_b32 s1, v44, 1.
      0xBE9E4900u,                              // 0x30: s_swap_pc_i64 s[30:31], s[0:1].
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x34: continuation.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x38: target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);

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
  constexpr auto clobber = cdna5::build_vop3(cdna5::kVMovB64Vop3, {.vdst = 44, .src0 = 256 + 46});
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
      0x0201032Cu,                              // 0x30: v_readlane_b32 s1, v44, 1.
      0xBE9E4900u,                              // 0x38: s_swap_pc_i64 s[30:31], s[0:1].
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x3c: continuation.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x40: would-be target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);

  size_t total_fixups = 0;
  for (const auto &block : blocks)
    total_fixups += block->static_indirect_call_fixups().size();
  EXPECT_EQ(total_fixups, 0u);
}

TEST(CfgAnalysis, Gfx1250ExplicitVgprWriteInvalidatesOnlyItsDestinationBank) {
  // Keep the lane stash in physical v256, temporarily switch the destination
  // bank to zero for a wide write to v[0:1], then select physical v256 as the
  // readlane source. The explicit bank-zero write must not invalidate the
  // bank-one lane table.
  constexpr auto set_dst_bank_one = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 0x40});
  constexpr auto set_bank_zero = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 0});
  constexpr auto set_src0_bank_one = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 1});
  constexpr auto write_bank_zero =
      cdna5::build_vop3(cdna5::kVMovB64Vop3, {.vdst = 0, .src0 = 256 + 2});
  std::vector<uint32_t> words = {
      0xBE804700u, // 0x00: s_get_pc_i64 s[0:1].
      0xA980FE00u,
      72u,
      0u,                  // 0x04: s_add_nc_u64 ..., lit64(72) -> target 0x4c.
      set_dst_bank_one[0], // 0x10: destination operands select bank one.
      0xD7610000u,
      0x02010000u, // 0x14: v_writelane_b32 physical v256, s0, 0.
      0xD7610000u,
      0x02010201u,      // 0x1c: v_writelane_b32 physical v256, s1, 1.
      set_bank_zero[0], // 0x24: explicit destinations select bank zero.
      write_bank_zero[0],
      write_bank_zero[1],   // 0x28: v_mov_b64 physical v[0:1], v[2:3].
      set_src0_bank_one[0], // 0x30: source-zero operands select bank one.
      0xD7600000u,
      0x02010100u, // 0x34: v_readlane_b32 s0, physical v256, 0.
      0xD7600001u,
      0x02010300u,                              // 0x3c: s1 <- physical v256 lane 1.
      0xBE9E4900u,                              // 0x44: s_swap_pc_i64 s[30:31], s[0:1].
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x48: continuation.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x4c: target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);

  auto *consumer = block_starting_at(blocks, 68);
  ASSERT_NE(consumer, nullptr);
  ASSERT_EQ(consumer->static_indirect_call_fixups().size(), 1u);
  EXPECT_EQ(consumer->static_indirect_call_fixups()[0].source_target_offset, 76u);
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
      0x02010201u,                                              // 0x18: v_writelane_b32 v44, s1, 1.
      cdna5::build_sopp(cdna5::kSBranchSopp, {.simm16 = 0})[0], // 0x20: s_branch -> 0x24.
      0xD7600000u,
      0x0201012Cu, // 0x24: v_readlane_b32 s0, v44, 0.
      0xD7600001u,
      0x0201032Cu,                              // 0x2c: v_readlane_b32 s1, v44, 1.
      0xBE9E4900u,                              // 0x34: s_swap_pc_i64 s[30:31], s[0:1].
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x38: continuation.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x3c: would-be target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);

  auto *consumer = block_starting_at(blocks, 52);
  ASSERT_NE(consumer, nullptr);
  ASSERT_EQ(consumer->static_indirect_call_fixups().size(), 1u);
  const IndirectCallFixup &restored = consumer->static_indirect_call_fixups()[0];
  EXPECT_EQ(restored.source_target_offset, 60u);
  // This stash restores into the pair the getpc wrote, so the producer and consumer registers
  // coincide. Pinning that is the counterpart to the split case below: the rewrite must name the
  // producer, and when the two are equal that has to stay indistinguishable from naming either.
  EXPECT_EQ(restored.source_builder_sreg, 0u);
  EXPECT_EQ(restored.source_call_sreg, 0u);
}

TEST(CfgAnalysis, Gfx1250UnreachablePostRocrAbortBlockDoesNotPoisonLaneStash) {
  constexpr auto live_branch = cdna5::build_sopp(cdna5::kSCbranchScc0Sopp, {.simm16 = 3});
  constexpr auto dead_branch = cdna5::build_sopp(cdna5::kSBranchSopp, {.simm16 = 1});
  constexpr auto call = cdna5::build_sop1(cdna5::kSSwapPcI64Sop1, {.ssrc0 = 0, .sdst = 30});

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
      0x02010201u,                               // 0x18: lane 1 <- s1.
      live_branch[0],                            // 0x20: -> join at 0x30.
      build_s_trap(ROCJITSU_CODE_ARCH_CDNA5, 2), // 0x24: abort terminator.
      dead_branch[0],                            // 0x28: dead edge -> 0x30.
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5),  // 0x2c: dead padding.
      0xD7600000u,
      0x0201012Cu, // 0x30: v_readlane_b32 s0, v44, 0.
      0xD7600001u,
      0x0201032Cu,                              // 0x38: lane 1 -> s1.
      call[0],                                  // 0x40: s_swap_pc_i64.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x44: continuation.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x48: target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5, {},
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
      0u,                                       // 0x04: s_add_nc_u64 ..., lit64(20) -> helper 0x18.
      0xBE9E4900u,                              // 0x10: outer s_swap_pc_i64 -> 0x18.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x14: outer continuation.
      0xBE804700u,                              // 0x18: helper s_get_pc_i64.
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
      0x0201032Cu,                              // 0x40: lane 1 -> s1.
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5), // 0x48: padding.
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5), // 0x4c: padding.
      0xBE9E4900u,                              // 0x50: inner call -> 0x58.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x54: inner continuation.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x58: inner target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5, {},
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
  constexpr auto clobber = cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 128, .vdst = 48});
  std::vector<uint32_t> words = {
      0xBE804700u, // 0x00: s_get_pc_i64 s[0:1].
      0xA980FE00u, 56u,
      0u, // 0x04: s_add_nc_u64 ..., lit64(56) -> stale target 0x3c.
      0xD7610030u,
      0x02010000u, // 0x10: v_writelane_b32 v48, s0, 0.
      0xD7610030u,
      0x02010201u, // 0x18: v_writelane_b32 v48, s1, 1.
      rocjitsu::build_s_call_b64(kReturnSreg, 7, ROCJITSU_CODE_ARCH_CDNA5),
      // 0x20: direct call -> callee at 0x40.
      0xD7600000u,
      0x02010130u, // 0x24: continuation reads the pre-call low half.
      0xD7600001u,
      0x02010330u,                              // 0x2c: continuation reads the high half.
      0xBE9E4900u,                              // 0x34: stale s_swap_pc_i64.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x38: continuation.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x3c: stale target.
      clobber[0],                               // 0x40: callee clobbers v48.
      rocjitsu::build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_CDNA5),
      // 0x44: callee return.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);

  size_t total_fixups = 0;
  for (const auto &block : blocks)
    total_fixups += block->static_indirect_call_fixups().size();
  EXPECT_EQ(total_fixups, 0u);
}

TEST(CfgAnalysis, Gfx1250ExactCalleeSummaryPreservesUnwrittenCallerSavedLaneStash) {
  constexpr uint16_t kReturnSreg = 30;
  constexpr auto unrelated_write = cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 128, .vdst = 2});
  for (const uint16_t stash_vgpr : {uint16_t{48}, uint16_t{192}}) {
    SCOPED_TRACE(stash_vgpr);
    std::vector<uint32_t> words = {
        0xBE804700u, // 0x00: s_get_pc_i64 s[0:1].
        0xA980FE00u, 56u,
        0u, // 0x04: s_add_nc_u64 ..., lit64(56) -> target 0x3c.
        0xD7610000u | stash_vgpr,
        0x02010000u, // 0x10: v_writelane_b32 stash_vgpr, s0, 0.
        0xD7610000u | stash_vgpr,
        0x02010201u, // 0x18: v_writelane_b32 stash_vgpr, s1, 1.
        rocjitsu::build_s_call_b64(kReturnSreg, 7, ROCJITSU_CODE_ARCH_CDNA5),
        // 0x20: direct call -> callee at 0x40.
        0xD7600000u,
        0x02010100u | stash_vgpr, // 0x24: continuation reads the low half.
        0xD7600001u,
        0x02010300u | stash_vgpr,                 // 0x2c: reads the high half.
        0xBE9E4900u,                              // 0x34: recovered s_swap_pc_i64.
        build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x38: continuation.
        build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x3c: stashed target.
        unrelated_write[0],                       // 0x40: callee writes only v2.
        rocjitsu::build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_CDNA5),
        // 0x44: callee return.
    };

    TestCodeObject co(std::move(words));
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
    ASSERT_NE(decoder, nullptr);
    auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);

    const IndirectCallFixup *continuation_fixup = nullptr;
    for (const auto &block : blocks) {
      for (const auto &fixup : block->static_indirect_call_fixups()) {
        if (fixup.source_call_offset == 52)
          continuation_fixup = &fixup;
      }
    }
    ASSERT_NE(continuation_fixup, nullptr);
    EXPECT_EQ(continuation_fixup->source_target_offset, 60u);
  }
}

TEST(CfgAnalysis, Gfx1250RelativeVgprDestinationDisablesExactCalleeSummary) {
  constexpr uint16_t kReturnSreg = 30;
  constexpr auto relative_write =
      cdna5::build_vop1(cdna5::kVMovreldB32Vop1, {.src0 = 0, .vdst = 2});
  std::vector<uint32_t> words = {
      0xBE804700u, // 0x00: s_get_pc_i64 s[0:1].
      0xA980FE00u, 56u,
      0u, // 0x04: s_add_nc_u64 ..., lit64(56) -> target 0x3c.
      0xD7610030u,
      0x02010000u, // 0x10: v_writelane_b32 v48, s0, 0.
      0xD7610030u,
      0x02010201u, // 0x18: v_writelane_b32 v48, s1, 1.
      rocjitsu::build_s_call_b64(kReturnSreg, 7, ROCJITSU_CODE_ARCH_CDNA5),
      // 0x20: direct call -> callee at 0x40.
      0xD7600000u,
      0x02010130u, // 0x24: continuation reads the low half.
      0xD7600001u,
      0x02010330u,                              // 0x2c: reads the high half.
      0xBE9E4900u,                              // 0x34: must not use a stale target.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x38: continuation.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x3c: stale target.
      relative_write[0],                        // 0x40: runtime-selected VGPR destination.
      rocjitsu::build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_CDNA5),
      // 0x44: callee return.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);

  for (const auto &block : blocks) {
    EXPECT_TRUE(std::ranges::none_of(
        block->static_indirect_call_fixups(),
        [](const IndirectCallFixup &fixup) { return fixup.source_call_offset == 52; }));
  }
}

TEST(CfgAnalysis, Gfx1250ModeBit27DoesNotInvalidateExactCalleeSummary) {
  constexpr uint16_t kReturnSreg = 30;
  constexpr uint16_t kModeGprIdxEnable = 1u | (27u << 6);
  constexpr auto enable_with_literal =
      cdna5::build_sopk(cdna5::kSSetregImm32B32Sopk, {.simm16 = kModeGprIdxEnable});
  constexpr auto enable_dynamically =
      cdna5::build_sopk(cdna5::kSSetregB32Sopk, {.simm16 = kModeGprIdxEnable, .sdst = 0});
  constexpr auto ordinary_write = cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 128, .vdst = 2});

  for (std::vector<uint32_t> callee :
       {std::vector<uint32_t>{enable_with_literal[0], 1u, ordinary_write[0]},
        std::vector<uint32_t>{enable_dynamically[0], ordinary_write[0]}}) {
    std::vector<uint32_t> words = {
        0xBE804700u, // 0x00: s_get_pc_i64 s[0:1].
        0xA980FE00u, 56u,
        0u, // 0x04: s_add_nc_u64 ..., lit64(56) -> target 0x3c.
        0xD7610030u,
        0x02010000u, // 0x10: v_writelane_b32 v48, s0, 0.
        0xD7610030u,
        0x02010201u, // 0x18: v_writelane_b32 v48, s1, 1.
        rocjitsu::build_s_call_b64(kReturnSreg, 7, ROCJITSU_CODE_ARCH_CDNA5),
        // 0x20: direct call -> callee at 0x40.
        0xD7600000u,
        0x02010130u, // 0x24: continuation reads the low half.
        0xD7600001u,
        0x02010330u,                              // 0x2c: reads the high half.
        0xBE9E4900u,                              // 0x34: must not use a stale target.
        build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x38: continuation.
        build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x3c: stale target.
    };
    words.insert(words.end(), callee.begin(), callee.end());
    words.push_back(rocjitsu::build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_CDNA5));

    TestCodeObject co(std::move(words));
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
    ASSERT_NE(decoder, nullptr);
    auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);

    const IndirectCallFixup *continuation_fixup = nullptr;
    for (const auto &block : blocks) {
      for (const auto &fixup : block->static_indirect_call_fixups()) {
        if (fixup.source_call_offset == 52)
          continuation_fixup = &fixup;
      }
    }
    ASSERT_NE(continuation_fixup, nullptr);
    EXPECT_EQ(continuation_fixup->source_target_offset, 60u);
  }
}

TEST(CfgAnalysis, Gfx1250RelativeSgprDestinationDisablesExactCalleeSummary) {
  constexpr uint16_t kReturnSreg = 30;
  constexpr uint16_t kRestoredPair = 20;
  constexpr auto relative_write =
      cdna5::build_sop1(cdna5::kSMovreldB32Sop1, {.ssrc0 = 2, .sdst = 8});
  std::vector<uint32_t> words = {
      0xBE804700u, // 0x00: s_get_pc_i64 s[0:1].
      0xA980FE00u, 64u,
      0u, // 0x04: s_add_nc_u64 ..., lit64(64) -> target 0x44.
      0xD761002Cu,
      0x02010000u, // 0x10: v_writelane_b32 v44, s0, 0.
      0xD761002Cu,
      0x02010201u, // 0x18: v_writelane_b32 v44, s1, 1.
      0xD7600000u | kRestoredPair,
      0x0201012Cu, // 0x20: v_readlane_b32 s20, v44, 0.
      0xD7600000u | static_cast<uint16_t>(kRestoredPair + 1),
      0x0201032Cu, // 0x28: v_readlane_b32 s21, v44, 1.
      rocjitsu::build_s_call_b64(kReturnSreg, 4, ROCJITSU_CODE_ARCH_CDNA5),
      // 0x30: direct call -> callee at 0x44.
      rocjitsu::build_s_swappc_b64(kReturnSreg, kRestoredPair, ROCJITSU_CODE_ARCH_CDNA5),
      // 0x34: must not use a possibly overwritten restored pair.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x38: continuation.
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5), // 0x3c: padding.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x40: padding.
      relative_write[0],                        // 0x44: runtime-selected SGPR destination.
      rocjitsu::build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_CDNA5),
      // 0x48: callee return.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);

  for (const auto &block : blocks) {
    EXPECT_TRUE(std::ranges::none_of(
        block->static_indirect_call_fixups(),
        [](const IndirectCallFixup &fixup) { return fixup.source_call_offset == 52; }));
  }
}

TEST(CfgAnalysis, Gfx1250CalleeSavedLaneStashSurvivesDirectCall) {
  // Same shape as Gfx1250DirectCallKillsCarriedLaneStash, but the stash lives
  // in v44 (CALLEE-saved under CSR_AMDGPU_VGPRs). The synthetic callee writes
  // only caller-saved v48, so the continuation must recover the call target
  // from the surviving v44 stash.
  constexpr uint16_t kReturnSreg = 30;
  constexpr auto clobber = cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 128, .vdst = 48});
  std::vector<uint32_t> words = {
      0xBE804700u, // 0x00: s_get_pc_i64 s[0:1].
      0xA980FE00u, 56u,
      0u, // 0x04: s_add_nc_u64 ..., lit64(56) -> stashed target 0x3c.
      0xD761002Cu,
      0x02010000u, // 0x10: v_writelane_b32 v44, s0, 0.
      0xD761002Cu,
      0x02010201u, // 0x18: v_writelane_b32 v44, s1, 1.
      rocjitsu::build_s_call_b64(kReturnSreg, 7, ROCJITSU_CODE_ARCH_CDNA5),
      // 0x20: direct call -> callee at 0x40.
      0xD7600000u,
      0x0201012Cu, // 0x24: continuation reads the surviving low half.
      0xD7600001u,
      0x0201032Cu,                              // 0x2c: reads the high half.
      0xBE9E4900u,                              // 0x34: recovered s_swap_pc_i64.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x38: continuation.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x3c: stashed target.
      clobber[0],                               // 0x40: callee clobbers caller-saved v48.
      rocjitsu::build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_CDNA5),
      // 0x44: callee return.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);

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

TEST(CfgAnalysis, Gfx1250ExactCalleeSummaryPreservesBankedLaneStash) {
  // Select bank 1 for both DST and SRC0, so the v44 operands below consistently
  // address physical v300. The ABI table does not cover physical VGPRs above
  // v255. This local callee has a complete CFG and writes the same low selector
  // only after selecting bank zero, so its exact physical-register summary
  // proves that the bank-one stash survives.
  //
  // s_set_vgpr_msb immediate byte is {DST[7:6], SRC2[5:4], SRC1[3:2], SRC0[1:0]};
  // 0x41 selects bank 1 for DST and SRC0.
  constexpr uint16_t kReturnSreg = 30;
  constexpr auto set_dst_src0_bank_one =
      cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 0x41});
  constexpr auto set_all_banks_zero = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 0});
  constexpr auto clobber_bank_zero =
      cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 128, .vdst = 44});
  std::vector<uint32_t> words = {
      0xBE804700u, // 0x00: s_get_pc_i64 s[0:1].
      0xA980FE00u, 60u,
      0u,                       // 0x04: s_add_nc_u64 ..., lit64(60) -> stale target 0x40.
      set_dst_src0_bank_one[0], // 0x10: v44 DST/SRC0 operands resolve to physical v300.
      0xD761002Cu,
      0x02010000u, // 0x14: v_writelane_b32 physical v300, s0, 0.
      0xD761002Cu,
      0x02010201u, // 0x1c: v_writelane_b32 physical v300, s1, 1.
      rocjitsu::build_s_call_b64(kReturnSreg, 7, ROCJITSU_CODE_ARCH_CDNA5),
      // 0x24: direct call -> callee at 0x44.
      0xD7600000u,
      0x0201012Cu, // 0x28: continuation reads physical v300 lane 0.
      0xD7600001u,
      0x0201032Cu,                              // 0x30: reads physical v300 lane 1.
      0xBE9E4900u,                              // 0x38: recovered from physical v300.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x3c: continuation.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x40: stale target.
      set_all_banks_zero[0],                    // 0x44: callee selects physical v44, not v300.
      clobber_bank_zero[0],                     // 0x48: same low selector in a different bank.
      set_dst_src0_bank_one[0],                 // 0x4c: restore the caller's bank selectors.
      rocjitsu::build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_CDNA5),
      // 0x50: callee return.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);

  const IndirectCallFixup *continuation_fixup = nullptr;
  for (const auto &block : blocks) {
    for (const auto &fixup : block->static_indirect_call_fixups()) {
      if (fixup.source_call_offset == 56)
        continuation_fixup = &fixup;
    }
  }
  ASSERT_NE(continuation_fixup, nullptr);
  EXPECT_EQ(continuation_fixup->source_target_offset, 64u);
}

TEST(CfgAnalysis, Gfx1250CalleeModeChangeInvalidatesContinuationBankSelection) {
  constexpr uint16_t kReturnSreg = 30;
  constexpr auto set_dst_src0_bank_one =
      cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 0x41});
  constexpr auto set_all_banks_zero = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 0});
  std::vector<uint32_t> words = {
      0xBE804700u, // 0x00: s_get_pc_i64 s[0:1].
      0xA980FE00u, 56u,
      0u,                       // 0x04: target 0x3c.
      set_dst_src0_bank_one[0], // 0x10: v44 resolves to physical v300.
      0xD761002Cu,
      0x02010000u, // 0x14: stash low half in physical v300.
      0xD761002Cu,
      0x02010201u, // 0x1c: stash high half in physical v300.
      rocjitsu::build_s_call_b64(kReturnSreg, 7, ROCJITSU_CODE_ARCH_CDNA5),
      // 0x24: direct call -> callee at 0x44.
      0xD7600000u,
      0x0201012Cu, // 0x28: MODE now maps this read to physical v44.
      0xD7600001u,
      0x0201032Cu,                              // 0x30: reads physical v44, not the stash.
      0xBE9E4900u,                              // 0x38: must not recover a stale target.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x3c: would-be target.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x40: continuation.
      set_all_banks_zero[0],                    // 0x44: persistent MODE change in callee.
      rocjitsu::build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_CDNA5),
      // 0x48: callee return without restoring MODE.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);

  for (const auto &block : blocks) {
    EXPECT_TRUE(std::ranges::none_of(
        block->static_indirect_call_fixups(),
        [](const IndirectCallFixup &fixup) { return fixup.source_call_offset == 56; }));
  }
}

TEST(CfgAnalysis, Gfx1250IndirectCallKillsCarriedLaneStash) {
  constexpr uint16_t kCallPcSreg = 8;
  constexpr uint16_t kReturnSreg = 30;
  constexpr uint16_t kStaleReturnSreg = 28;
  constexpr auto stale_getpc = cdna5::build_sop1(cdna5::kSGetPcI64Sop1, {.sdst = 0});
  constexpr auto call_getpc = cdna5::build_sop1(cdna5::kSGetPcI64Sop1, {.sdst = kCallPcSreg});
  constexpr auto call_add = cdna5::build_sop2(
      cdna5::kSAddNcU64Sop2, {.ssrc0 = kCallPcSreg, .ssrc1 = 254, .sdst = kCallPcSreg});
  constexpr auto call =
      cdna5::build_sop1(cdna5::kSSwapPcI64Sop1, {.ssrc0 = kCallPcSreg, .sdst = kReturnSreg});
  constexpr auto stale_call =
      cdna5::build_sop1(cdna5::kSSwapPcI64Sop1, {.ssrc0 = 0, .sdst = kStaleReturnSreg});
  constexpr auto clobber = cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 128, .vdst = 48});

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
      0x02010330u,                              // 0x3c: reads the high half.
      stale_call[0],                            // 0x44: stale indirect call.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x48: continuation.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x4c: stale target.
      clobber[0],                               // 0x50: callee clobbers v48.
      rocjitsu::build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_CDNA5),
      // 0x54: callee return.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);

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
  constexpr auto stale_getpc = cdna5::build_sop1(cdna5::kSGetPcI64Sop1, {.sdst = 0});
  constexpr auto call_getpc = cdna5::build_sop1(cdna5::kSGetPcI64Sop1, {.sdst = kCallPcSreg});
  constexpr auto call_add = cdna5::build_sop2(
      cdna5::kSAddNcU64Sop2, {.ssrc0 = kCallPcSreg, .ssrc1 = 254, .sdst = kCallPcSreg});
  constexpr auto call =
      cdna5::build_sop1(cdna5::kSSwapPcI64Sop1, {.ssrc0 = kCallPcSreg, .sdst = kReturnSreg});
  constexpr auto stale_call =
      cdna5::build_sop1(cdna5::kSSwapPcI64Sop1, {.ssrc0 = 0, .sdst = kStaleReturnSreg});
  constexpr auto clobber = cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 128, .vdst = 48});

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
      0x0201032Cu,                              // 0x3c: reads the high half.
      stale_call[0],                            // 0x44: continuation indirect call.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x48: continuation.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x4c: stashed target.
      clobber[0],                               // 0x50: callee clobbers caller-saved v48.
      rocjitsu::build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_CDNA5),
      // 0x54: callee return.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);

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
  constexpr auto getpc = cdna5::build_sop1(cdna5::kSGetPcI64Sop1, {.sdst = 0});
  constexpr auto branch = cdna5::build_sopp(cdna5::kSBranchSopp, {.simm16 = 0});
  constexpr auto backedge =
      cdna5::build_sopp(cdna5::kSCbranchScc0Sopp, {.simm16 = static_cast<uint16_t>(-10)});
  constexpr auto call = cdna5::build_sop1(cdna5::kSSwapPcI64Sop1, {.ssrc0 = 0, .sdst = 30});

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
      0x0201032Cu,                              // 0x30: v_readlane_b32 s1, v44, 1.
      call[0],                                  // 0x38: cross-block consumer.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x3c: continuation.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x40: target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);

  auto *consumer = block_starting_at(blocks, 56);
  ASSERT_NE(consumer, nullptr);
  ASSERT_EQ(consumer->static_indirect_call_fixups().size(), 1u);
  EXPECT_EQ(consumer->static_indirect_call_fixups()[0].source_target_offset, 64u);
}

TEST(CfgAnalysis, Gfx1250ExplicitKernelEntryClearsIncomingLaneStash) {
  constexpr auto getpc = cdna5::build_sop1(cdna5::kSGetPcI64Sop1, {.sdst = 0});
  constexpr auto branch = cdna5::build_sopp(cdna5::kSBranchSopp, {.simm16 = 1});
  constexpr auto call = cdna5::build_sop1(cdna5::kSSwapPcI64Sop1, {.ssrc0 = 0, .sdst = 30});

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
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5), // 0x24: skipped.
      0xD7600000u,
      0x0201012Cu, // 0x28: entry B reads v44 lane 0.
      0xD7600001u,
      0x0201032Cu,                              // 0x30: reads v44 lane 1.
      call[0],                                  // 0x38: must remain dynamic.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x3c: continuation.
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5), // 0x40: padding.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x44: A's target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  constexpr std::array<uint64_t, 1> extra_leaders{40};
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5, extra_leaders);

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
      cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 0x4400});
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
      0x0201032Cu,                              // 0x30: v_readlane_b32 s1, physical v44, 1.
      0xBE9E4900u,                              // 0x38: s_swap_pc_i64 s[30:31], s[0:1].
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x3c: target/continuation.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);

  auto *consumer = block_starting_at(blocks, 56);
  ASSERT_NE(consumer, nullptr);
  ASSERT_EQ(consumer->static_indirect_call_fixups().size(), 1u);
  EXPECT_EQ(consumer->static_indirect_call_fixups()[0].source_target_offset, 60u);
}

TEST(CfgAnalysis, Gfx1250ImmediateModeWriteKeepsLaneStashBankKnown) {
  // The hipTensor dispatcher writes WAVE_MODE bit 25 before restoring a
  // PC from fixed VGPR lanes. The write is disjoint from MODE.VGPR_MSB, so a
  // lane stash in bank one remains in physical v300 and is still recoverable.
  constexpr auto set_dst_src0_bank_one =
      cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 0x41});
  constexpr uint16_t kModeBit25Hwreg = 1u | (25u << 6);
  constexpr auto set_mode_bit25 =
      cdna5::build_sopk(cdna5::kSSetregImm32B32Sopk, {.simm16 = kModeBit25Hwreg});
  std::vector<uint32_t> words = {
      0xBE804700u, // 0x00: s_get_pc_i64 s[0:1].
      0xA980FE00u,
      60u,
      0u, // 0x04: s_add_nc_u64 ..., lit64(60) -> target 0x40.
      set_dst_src0_bank_one[0],
      0xD761002Cu,
      0x02010000u, // 0x14: v_writelane_b32 physical v300, s0, 0.
      0xD761002Cu,
      0x02010201u, // 0x1c: v_writelane_b32 physical v300, s1, 1.
      set_mode_bit25[0],
      1u, // 0x24: s_setreg_imm32_b32 hwreg(WAVE_MODE, 25, 1), 1.
      0xD7600000u,
      0x0201012Cu, // 0x2c: v_readlane_b32 s0, physical v300, 0.
      0xD7600001u,
      0x0201032Cu,                              // 0x34: v_readlane_b32 s1, physical v300, 1.
      0xBE9E4900u,                              // 0x3c: s_swap_pc_i64 s[30:31], s[0:1].
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x40: target/continuation.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);

  auto *consumer = block_starting_at(blocks, 60);
  ASSERT_NE(consumer, nullptr);
  ASSERT_EQ(consumer->static_indirect_call_fixups().size(), 1u);
  EXPECT_EQ(consumer->static_indirect_call_fixups()[0].source_target_offset, 64u);
}

TEST(CfgAnalysis, Gfx1250LaneRestoreReachesConsumerAcrossBranch) {
  constexpr auto branch = cdna5::build_sopp(cdna5::kSBranchSopp, {.simm16 = 0});
  std::vector<uint32_t> words = {
      0xBE804700u, // 0x00: s_get_pc_i64 s[0:1].
      0xA980FE00u,
      56u,
      0u, // 0x04: s_add_nc_u64 ..., lit64(56) -> target 0x3c.
      0xD761002Cu,
      0x02010000u, // 0x10: v_writelane_b32 v44, s0, 0.
      0xD761002Cu,
      0x02010201u, // 0x18: v_writelane_b32 v44, s1, 1.
      0xD7600014u,
      0x0201012Cu, // 0x20: v_readlane_b32 s20, v44, 0.
      0xD7600015u,
      0x0201032Cu,                              // 0x28: v_readlane_b32 s21, v44, 1.
      branch[0],                                // 0x30: split the restore and consumer blocks.
      0xBE9E4914u,                              // 0x34: s_swap_pc_i64 s[30:31], s[20:21].
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x38: continuation.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x3c: target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);

  auto *consumer = block_starting_at(blocks, 52);
  ASSERT_NE(consumer, nullptr);
  ASSERT_EQ(consumer->static_indirect_call_fixups().size(), 1u);
  const IndirectCallFixup &restored = consumer->static_indirect_call_fixups()[0];
  EXPECT_EQ(restored.source_target_offset, 60u);
  // Here the stash restores into a different pair than the getpc wrote, and the rewrite regenerates
  // the add against the producer. Asserting both is what makes this a discovery-path check: the
  // layout-level test builds its fixup by hand, so it would still pass if PcValue propagation or
  // fixup_for_value dropped the producer pair on the way through.
  EXPECT_EQ(restored.source_builder_sreg, 0u);
  EXPECT_EQ(restored.source_call_sreg, 20u);
}

TEST(CfgAnalysis, Gfx1250PcPairCopyReachesConsumer) {
  constexpr uint16_t kSourceSreg = 20;
  constexpr uint16_t kCopiedSreg = 54;
  constexpr uint16_t kReturnSreg = 30;
  constexpr auto getpc = cdna5::build_sop1(cdna5::kSGetPcI64Sop1, {.sdst = kSourceSreg});
  constexpr auto add = cdna5::build_sop2(cdna5::kSAddNcU64Sop2,
                                         {.ssrc0 = kSourceSreg, .ssrc1 = 254, .sdst = kSourceSreg});
  constexpr auto copy = cdna5::build_sop1(
      cdna5::kSMovB64Sop1, {.ssrc0 = static_cast<uint8_t>(kSourceSreg), .sdst = kCopiedSreg});
  std::vector<uint32_t> words = {
      getpc[0], // 0x00: s_get_pc_i64 s[20:21].
      add[0], 24u,
      0u,      // 0x04: s_add_nc_u64 ..., lit64(24) -> target 0x1c.
      copy[0], // 0x10: s_mov_b64 s[54:55], s[20:21].
      rocjitsu::build_s_swappc_b64(kReturnSreg, kCopiedSreg, ROCJITSU_CODE_ARCH_CDNA5),
      // 0x14: call through the copied pair.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x18: continuation.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x1c: target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);

  const IndirectCallFixup *copied_call = nullptr;
  for (const auto &block : blocks) {
    for (const auto &fixup : block->static_indirect_call_fixups()) {
      if (fixup.source_call_offset == 20)
        copied_call = &fixup;
    }
  }
  ASSERT_NE(copied_call, nullptr);
  EXPECT_EQ(copied_call->source_target_offset, 28u);
  // Same producer/consumer split, reached by a register-pair copy rather than a lane stash.
  EXPECT_EQ(copied_call->source_builder_sreg, kSourceSreg);
  EXPECT_EQ(copied_call->source_call_sreg, kCopiedSreg);
}

TEST(CfgAnalysis, Gfx1250DirectCallOverwritesPcBuilderInReturnPair) {
  constexpr uint16_t kReturnSreg = 30;
  constexpr auto getpc = cdna5::build_sop1(cdna5::kSGetPcI64Sop1, {.sdst = kReturnSreg});
  constexpr auto add = cdna5::build_sop2(cdna5::kSAddNcU64Sop2,
                                         {.ssrc0 = kReturnSreg, .ssrc1 = 254, .sdst = kReturnSreg});
  std::vector<uint32_t> words = {
      getpc[0], // 0x00: s_get_pc_i64 s[30:31].
      add[0], 32u,
      0u, // 0x04: s_add_nc_u64 ..., lit64(32) -> stale target 0x24.
      rocjitsu::build_s_call_b64(kReturnSreg, 2, ROCJITSU_CODE_ARCH_CDNA5),
      // 0x10: direct call overwrites s[30:31] and enters 0x1c.
      rocjitsu::build_s_swappc_b64(40, kReturnSreg, ROCJITSU_CODE_ARCH_CDNA5),
      // 0x14: s[30:31] contains the return PC, not the old builder.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x18: continuation.
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5), // 0x1c: callee body.
      rocjitsu::build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_CDNA5),
      // 0x20: callee return.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x24: stale target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);

  for (const auto &block : blocks) {
    EXPECT_TRUE(std::ranges::none_of(
        block->static_indirect_call_fixups(),
        [](const IndirectCallFixup &fixup) { return fixup.source_call_offset == 20; }));
  }
}

TEST(CfgAnalysis, Gfx1250SwapPcOverwritesPcBuilderInReturnPair) {
  constexpr uint16_t kTargetSreg = 0;
  constexpr auto getpc = cdna5::build_sop1(cdna5::kSGetPcI64Sop1, {.sdst = kTargetSreg});
  constexpr auto add = cdna5::build_sop2(cdna5::kSAddNcU64Sop2,
                                         {.ssrc0 = kTargetSreg, .ssrc1 = 254, .sdst = kTargetSreg});
  std::vector<uint32_t> words = {
      getpc[0], // 0x00: s_get_pc_i64 s[0:1].
      add[0], 24u,
      0u, // 0x04: s_add_nc_u64 ..., lit64(24) -> callee at 0x1c.
      rocjitsu::build_s_swappc_b64(kTargetSreg, kTargetSreg, ROCJITSU_CODE_ARCH_CDNA5),
      // 0x10: call through s[0:1], then overwrite it with return PC 0x14.
      rocjitsu::build_s_swappc_b64(30, kTargetSreg, ROCJITSU_CODE_ARCH_CDNA5),
      // 0x14: must not reuse the pre-call target.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x18: continuation.
      rocjitsu::build_s_setpc_b64(kTargetSreg, ROCJITSU_CODE_ARCH_CDNA5),
      // 0x1c: callee return.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);

  const IndirectCallFixup *first_call = nullptr;
  for (const auto &block : blocks) {
    for (const auto &fixup : block->static_indirect_call_fixups()) {
      if (fixup.source_call_offset == 16)
        first_call = &fixup;
      EXPECT_NE(fixup.source_call_offset, 20u);
    }
  }
  ASSERT_NE(first_call, nullptr);
  EXPECT_EQ(first_call->source_target_offset, 28u);
}

TEST(CfgAnalysis, Gfx1250CalleeSummaryRejectsRepurposedReturnPair) {
  constexpr uint16_t kTargetSreg = 0;
  constexpr uint16_t kReturnSreg = 30;
  constexpr uint16_t kSavedReturnSreg = 32;
  constexpr auto set_mode_zero = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 0});
  constexpr auto get_stashed_target =
      cdna5::build_sop1(cdna5::kSGetPcI64Sop1, {.sdst = kTargetSreg});
  constexpr auto add_stashed_target = cdna5::build_sop2(
      cdna5::kSAddNcU64Sop2, {.ssrc0 = kTargetSreg, .ssrc1 = 254, .sdst = kTargetSreg});
  constexpr auto save_return = cdna5::build_sop1(
      cdna5::kSMovB64Sop1, {.ssrc0 = static_cast<uint8_t>(kReturnSreg), .sdst = kSavedReturnSreg});
  constexpr auto get_tail_target = cdna5::build_sop1(cdna5::kSGetPcI64Sop1, {.sdst = kReturnSreg});
  constexpr auto add_tail_target = cdna5::build_sop2(
      cdna5::kSAddNcU64Sop2, {.ssrc0 = kReturnSreg, .ssrc1 = 254, .sdst = kReturnSreg});
  constexpr auto clobber_stash = cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 128, .vdst = 192});
  std::vector<uint32_t> words = {
      set_mode_zero[0],      // 0x00: use bank zero.
      get_stashed_target[0], // 0x04: s_get_pc_i64 s[0:1].
      add_stashed_target[0], 56u,
      0u, // 0x08: s_add_nc_u64 ..., lit64(56) -> target 0x40.
      0xD76100C0u,
      0x02010000u, // 0x14: v_writelane_b32 v192, s0, 0.
      0xD76100C0u,
      0x02010201u, // 0x1c: v_writelane_b32 v192, s1, 1.
      rocjitsu::build_s_call_b64(kReturnSreg, 7, ROCJITSU_CODE_ARCH_CDNA5),
      // 0x24: direct call -> thunk at 0x44.
      0xD7600000u,
      0x020101C0u, // 0x28: restore the low half after the call.
      0xD7600001u,
      0x020103C0u, // 0x30: restore the high half after the call.
      rocjitsu::build_s_swappc_b64(kReturnSreg, kTargetSreg, ROCJITSU_CODE_ARCH_CDNA5),
      // 0x38: must not recover the stash clobbered by the tail target.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x3c: continuation.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x40: stashed target.
      save_return[0],                           // 0x44: save caller return in s[32:33].
      get_tail_target[0],                       // 0x48: rebuild s[30:31].
      add_tail_target[0], 16u,
      0u, // 0x4c: s_add_nc_u64 ..., lit64(16) -> tail target 0x5c.
      rocjitsu::build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_CDNA5),
      // 0x58: tail transfer through the repurposed pair.
      clobber_stash[0], // 0x5c: tail target clobbers v192.
      rocjitsu::build_s_setpc_b64(kSavedReturnSreg, ROCJITSU_CODE_ARCH_CDNA5),
      // 0x60: return through the saved pair.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);

  for (const auto &block : blocks) {
    EXPECT_TRUE(std::ranges::none_of(
        block->static_indirect_call_fixups(),
        [](const IndirectCallFixup &fixup) { return fixup.source_call_offset == 56; }));
  }
}

TEST(CfgAnalysis, Gfx1250ExactCalleeSummaryPreservesUnwrittenRestoredSgprs) {
  constexpr uint16_t kReturnSreg = 30;
  for (const uint16_t restored_pair : {uint16_t{0}, uint16_t{20}}) {
    SCOPED_TRACE(restored_pair);
    std::vector<uint32_t> words = {
        0xBE804700u, // 0x00: s_get_pc_i64 s[0:1].
        0xA980FE00u, 60u,
        0u, // 0x04: s_add_nc_u64 ..., lit64(60) -> target 0x40.
        0xD761002Cu,
        0x02010000u, // 0x10: v_writelane_b32 v44, s0, 0.
        0xD761002Cu,
        0x02010201u, // 0x18: v_writelane_b32 v44, s1, 1.
        0xD7600000u | restored_pair,
        0x0201012Cu, // 0x20: v_readlane_b32 restored_pair, v44, 0.
        0xD7600000u | static_cast<uint16_t>(restored_pair + 1),
        0x0201032Cu, // 0x28: v_readlane_b32 restored_pair+1, v44, 1.
        rocjitsu::build_s_call_b64(kReturnSreg, 4, ROCJITSU_CODE_ARCH_CDNA5),
        // 0x30: direct call -> callee at 0x44.
        rocjitsu::build_s_swappc_b64(kReturnSreg, restored_pair, ROCJITSU_CODE_ARCH_CDNA5),
        // 0x34: restored target consumer after the call returns.
        build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x38: continuation.
        build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5), // 0x3c: padding.
        build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x40: stashed target.
        build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5), // 0x44: callee body.
        rocjitsu::build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_CDNA5),
        // 0x48: callee return.
    };

    TestCodeObject co(std::move(words));
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
    ASSERT_NE(decoder, nullptr);
    auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);

    const IndirectCallFixup *fixup = nullptr;
    for (const auto &block : blocks) {
      for (const auto &candidate : block->static_indirect_call_fixups()) {
        if (candidate.source_call_offset == 52)
          fixup = &candidate;
      }
    }
    ASSERT_NE(fixup, nullptr);
    EXPECT_EQ(fixup->source_target_offset, 64u);
  }
}

TEST(CfgAnalysis, Gfx1250CalleeSummaryVariantLimitFallsBackConservatively) {
  constexpr uint16_t kTargetSreg = 0;
  constexpr uint16_t kReturnSreg = 30;
  constexpr uint16_t kStashVgpr = 48;

  const auto discovers_fixup = [&](size_t variant_count) {
    const auto getpc = cdna5::build_sop1(cdna5::kSGetPcI64Sop1, {.sdst = kTargetSreg});
    const auto add = cdna5::build_sop2(cdna5::kSAddNcU64Sop2,
                                       {.ssrc0 = kTargetSreg, .ssrc1 = 254, .sdst = kTargetSreg});
    std::vector<uint32_t> words = {
        cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 0})[0],
        getpc[0],
        add[0],
        0u,
        0u,
        0xD7610000u | kStashVgpr,
        0x02010000u, // v_writelane_b32 v48, s0, 0.
        0xD7610000u | kStashVgpr,
        0x02010201u, // v_writelane_b32 v48, s1, 1.
    };

    std::vector<size_t> call_indices(variant_count);
    for (size_t variant = 0; variant < variant_count; ++variant) {
      words.push_back(cdna5::build_sopp(cdna5::kSSetVgprMsbSopp,
                                        {.simm16 = static_cast<uint16_t>(variant)})[0]);
      call_indices[variant] = words.size();
      words.push_back(0u);
    }
    words.push_back(cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 0})[0]);
    words.push_back(0xD7600000u | kTargetSreg);
    words.push_back(0x02010100u | kStashVgpr); // v_readlane_b32 s0, v48, 0.
    words.push_back(0xD7600000u | static_cast<uint16_t>(kTargetSreg + 1));
    words.push_back(0x02010300u | kStashVgpr); // v_readlane_b32 s1, v48, 1.
    const uint64_t consumer_offset = words.size() * sizeof(uint32_t);
    words.push_back(
        rocjitsu::build_s_swappc_b64(kReturnSreg, kTargetSreg, ROCJITSU_CODE_ARCH_CDNA5));
    words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5));
    const uint64_t stashed_target_offset = words.size() * sizeof(uint32_t);
    words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5));
    const uint64_t callee_offset = words.size() * sizeof(uint32_t);
    words.push_back(build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5));
    words.push_back(rocjitsu::build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_CDNA5));

    // S_GET_PC_I64 at byte offset 4 materializes byte offset 8. Its following
    // literal add therefore needs target-8. Each S_CALL immediate is measured in
    // dwords from the instruction following the call.
    words[3] = static_cast<uint32_t>(stashed_target_offset - 2 * sizeof(uint32_t));
    for (size_t call_index : call_indices) {
      const int64_t delta = static_cast<int64_t>(callee_offset) -
                            static_cast<int64_t>((call_index + 1) * sizeof(uint32_t));
      EXPECT_EQ(delta % static_cast<int64_t>(sizeof(uint32_t)), 0);
      words[call_index] = rocjitsu::build_s_call_b64(
          kReturnSreg, static_cast<int16_t>(delta / sizeof(uint32_t)), ROCJITSU_CODE_ARCH_CDNA5);
    }

    TestCodeObject co(std::move(words));
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
    EXPECT_NE(decoder, nullptr);
    auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);

    return std::ranges::any_of(blocks, [&](const auto &block) {
      return std::ranges::any_of(block->static_indirect_call_fixups(),
                                 [&](const IndirectCallFixup &fixup) {
                                   return fixup.source_call_offset == consumer_offset;
                                 });
    });
  };

  EXPECT_TRUE(discovers_fixup(8));
  EXPECT_FALSE(discovers_fixup(9));
}

TEST(CfgAnalysis, Gfx1250ExactCalleeSummaryDropsWrittenRestoredSgprs) {
  constexpr uint16_t kReturnSreg = 30;
  for (const uint16_t restored_pair : {uint16_t{0}, uint16_t{20}}) {
    SCOPED_TRACE(restored_pair);
    std::vector<uint32_t> words = {
        0xBE804700u, // 0x00: s_get_pc_i64 s[0:1].
        0xA980FE00u, 64u,
        0u, // 0x04: s_add_nc_u64 ..., lit64(64) -> target 0x44.
        0xD761002Cu,
        0x02010000u, // 0x10: v_writelane_b32 v44, s0, 0.
        0xD761002Cu,
        0x02010201u, // 0x18: v_writelane_b32 v44, s1, 1.
        0xD7600000u | restored_pair,
        0x0201012Cu, // 0x20: v_readlane_b32 restored_pair, v44, 0.
        0xD7600000u | static_cast<uint16_t>(restored_pair + 1),
        0x0201032Cu, // 0x28: v_readlane_b32 restored_pair+1, v44, 1.
        rocjitsu::build_s_call_b64(kReturnSreg, 4, ROCJITSU_CODE_ARCH_CDNA5),
        // 0x30: direct call -> callee at 0x44.
        rocjitsu::build_s_swappc_b64(kReturnSreg, restored_pair, ROCJITSU_CODE_ARCH_CDNA5),
        // 0x34: restored target consumer after the call returns.
        build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x38: continuation.
        build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5), // 0x3c: padding.
        build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5), // 0x40: padding.
        0xBE800080u,                              // 0x44: callee writes caller-saved s0.
        rocjitsu::build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_CDNA5),
        // 0x48: callee return.
    };

    TestCodeObject co(std::move(words));
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
    ASSERT_NE(decoder, nullptr);
    auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);

    const IndirectCallFixup *fixup = nullptr;
    for (const auto &block : blocks) {
      for (const auto &candidate : block->static_indirect_call_fixups()) {
        if (candidate.source_call_offset == 52)
          fixup = &candidate;
      }
    }
    if (restored_pair == 20) {
      ASSERT_NE(fixup, nullptr);
      EXPECT_EQ(fixup->source_target_offset, 68u);
    } else {
      EXPECT_EQ(fixup, nullptr);
    }
  }
}

TEST(CfgAnalysis, Gfx1250UnsupportedCalleeSummaryFallsBackToCallPreservedSgprs) {
  constexpr uint16_t kReturnSreg = 30;
  for (const uint16_t restored_pair : {uint16_t{20}, uint16_t{64}}) {
    SCOPED_TRACE(restored_pair);
    std::vector<uint32_t> words = {
        0xBE804700u, // 0x00: s_get_pc_i64 s[0:1].
        0xA980FE00u, 64u,
        0u, // 0x04: s_add_nc_u64 ..., lit64(64) -> target 0x44.
        0xD761002Cu,
        0x02010000u, // 0x10: v_writelane_b32 v44, s0, 0.
        0xD761002Cu,
        0x02010201u, // 0x18: v_writelane_b32 v44, s1, 1.
        0xD7600000u | restored_pair,
        0x0201012Cu, // 0x20: v_readlane_b32 restored_pair, v44, 0.
        0xD7600000u | static_cast<uint16_t>(restored_pair + 1),
        0x0201032Cu, // 0x28: v_readlane_b32 restored_pair+1, v44, 1.
        rocjitsu::build_s_call_b64(kReturnSreg, 4, ROCJITSU_CODE_ARCH_CDNA5),
        // 0x30: direct call -> callee at 0x44.
        rocjitsu::build_s_swappc_b64(kReturnSreg, restored_pair, ROCJITSU_CODE_ARCH_CDNA5),
        // 0x34: restored target consumer after the call returns.
        build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x38: continuation.
        build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5), // 0x3c: padding.
        build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5), // 0x40: padding.
        rocjitsu::build_s_swappc_b64(kReturnSreg, 2, ROCJITSU_CODE_ARCH_CDNA5),
        // 0x44: unresolved nested call prevents an exact summary.
        rocjitsu::build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_CDNA5),
        // 0x48: callee return.
    };

    TestCodeObject co(std::move(words));
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
    ASSERT_NE(decoder, nullptr);
    auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);

    const IndirectCallFixup *fixup = nullptr;
    for (const auto &block : blocks) {
      for (const auto &candidate : block->static_indirect_call_fixups()) {
        if (candidate.source_call_offset == 52)
          fixup = &candidate;
      }
    }
    if (restored_pair == 64) {
      ASSERT_NE(fixup, nullptr);
      EXPECT_EQ(fixup->source_target_offset, 68u);
    } else {
      EXPECT_EQ(fixup, nullptr);
    }
  }
}

TEST(CfgAnalysis, CalleeSavedSgprsUseCallingConventionIntersection) {
  EXPECT_FALSE(is_callee_saved_sgpr(20));
  EXPECT_TRUE(is_callee_saved_sgpr(30));
  EXPECT_TRUE(is_callee_saved_sgpr(64));
  EXPECT_FALSE(is_callee_saved_sgpr(72));
  EXPECT_TRUE(is_callee_saved_sgpr(80));
  EXPECT_FALSE(is_callee_saved_sgpr(88));
  EXPECT_TRUE(is_callee_saved_sgpr(96));
  EXPECT_FALSE(is_callee_saved_sgpr(106));
}

TEST(CfgAnalysis, Gfx1250RestashedLaneTargetRemainsRecoverable) {
  std::vector<uint32_t> words = {
      0xBE804700u, // 0x00: s_get_pc_i64 s[0:1].
      0xA980FE00u,
      84u,
      0u, // 0x04: s_add_nc_u64 ..., lit64(84) -> target 0x58.
      0xD761002Cu,
      0x02010000u, // 0x10: v_writelane_b32 v44, s0, 0.
      0xD761002Cu,
      0x02010201u, // 0x18: v_writelane_b32 v44, s1, 1.
      0xD760000Eu,
      0x0201012Cu, // 0x20: v_readlane_b32 s14, v44, 0.
      0xD760000Fu,
      0x0201032Cu, // 0x28: v_readlane_b32 s15, v44, 1.
      0xD7610030u,
      0x0201000Eu, // 0x30: v_writelane_b32 v48, s14, 0.
      0xD7610030u,
      0x0201020Fu, // 0x38: v_writelane_b32 v48, s15, 1.
      0xD7600000u,
      0x02010130u, // 0x40: v_readlane_b32 s0, v48, 0.
      0xD7600001u,
      0x02010330u,                              // 0x48: v_readlane_b32 s1, v48, 1.
      0xBE9E4900u,                              // 0x50: s_swap_pc_i64 s[30:31], s[0:1].
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x54: continuation.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x58: target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);

  auto *consumer = block_starting_at(blocks, 80);
  ASSERT_NE(consumer, nullptr);
  ASSERT_EQ(consumer->static_indirect_call_fixups().size(), 1u);
  EXPECT_EQ(consumer->static_indirect_call_fixups()[0].source_target_offset, 88u);
}

TEST(CfgAnalysis, Gfx1250CalleeSavedPcBuilderCanBeStashedAfterCall) {
  constexpr uint16_t kTargetSreg = 14;
  constexpr uint16_t kReturnSreg = 30;
  constexpr auto getpc = cdna5::build_sop1(cdna5::kSGetPcI64Sop1, {.sdst = kTargetSreg});
  constexpr auto add = cdna5::build_sop2(cdna5::kSAddNcU64Sop2,
                                         {.ssrc0 = kTargetSreg, .ssrc1 = 254, .sdst = kTargetSreg});
  std::vector<uint32_t> words = {
      getpc[0], // 0x00: s_get_pc_i64 s[14:15].
      add[0], 56u,
      0u, // 0x04: s_add_nc_u64 ..., lit64(56) -> callee at 0x3c.
      rocjitsu::build_s_swappc_b64(kReturnSreg, kTargetSreg, ROCJITSU_CODE_ARCH_CDNA5),
      // 0x10: first call through s[14:15].
      0xD7610030u,
      0x0201000Eu, // 0x14: v_writelane_b32 v48, s14, 0.
      0xD7610030u,
      0x0201020Fu, // 0x1c: v_writelane_b32 v48, s15, 1.
      0xD7600000u,
      0x02010130u, // 0x24: v_readlane_b32 s0, v48, 0.
      0xD7600001u,
      0x02010330u, // 0x2c: v_readlane_b32 s1, v48, 1.
      rocjitsu::build_s_swappc_b64(kReturnSreg, 0, ROCJITSU_CODE_ARCH_CDNA5),
      // 0x34: second call through the re-stashed target.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x38: continuation.
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5), // 0x3c: callee body.
      rocjitsu::build_s_setpc_b64(kReturnSreg, ROCJITSU_CODE_ARCH_CDNA5),
      // 0x40: callee return.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);

  const IndirectCallFixup *second_call = nullptr;
  for (const auto &block : blocks) {
    for (const auto &fixup : block->static_indirect_call_fixups()) {
      if (fixup.source_call_offset == 52)
        second_call = &fixup;
    }
  }
  ASSERT_NE(second_call, nullptr);
  EXPECT_EQ(second_call->source_target_offset, 60u);
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
  constexpr auto set_dst_bank_one = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 0x40});
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
      0x0201032Cu,                              // 0x2c: v_readlane_b32 s1, v44, 1.
      0xBE9E4900u,                              // 0x34: s_swap_pc_i64 s[30:31], s[0:1].
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x38: continuation.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x3c: would-be target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);

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
  constexpr auto set_bank_zero = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 0});
  constexpr auto branch_to_stash = cdna5::build_sopp(cdna5::kSBranchSopp, {.simm16 = 0});
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
      0xBE9E4900u,                              // 0x38: s_swap_pc_i64 s[30:31], s[0:1].
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x3c: continuation.
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x40: target.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);

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
  constexpr auto cbranch_to_consumer = cdna5::build_sopp(cdna5::kSCbranchScc1Sopp, {.simm16 = 9});
  constexpr auto set_bank_zero = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 0});
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
      0xBE9E4900u,                              // 0x38: s_swap_pc_i64 s[30:31], s[0:1].
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5), // 0x3c: would-be target / continuation.
  };

  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
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
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, &inst_data[pc], byte_offset));
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
      ROCJITSU_CODE_ARCH_CDNA5);
  EXPECT_TRUE(fixups.empty()) << "must-dataflow must not reuse a skipped-path stash";
}

// A getpc reseeding a pair the previous builder still owns is a stable point for that builder: no
// later arithmetic can reach the old chain. Publishing it is what lets code that materializes
// several function pointers through one scratch pair -- build, spill, rebuild -- still satisfy a
// whole-object "every code address is relocated" claim. A COMPLETED chain must therefore survive
// the reseed as a resolved builder.
/// @brief Decode a hand-assembled gfx1250 word list and run indirect-branch discovery over it.
///
/// @details Mirrors BasicBlock::build's decode loop, including its skip of zero alignment padding,
/// so a fixture written as raw encodings reaches the pass the same way real `.text` does.
void run_indirect_discovery_for_test(std::vector<uint32_t> words,
                                     std::vector<PcAddressBuilder> *builders) {
  TestCodeObject co(std::move(words));
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);

  const auto *sec = co.text_sections().front();
  const auto *inst_data = reinterpret_cast<const uint32_t *>(sec->data());
  const size_t inst_data_size = sec->size() / sizeof(uint32_t);
  std::vector<std::unique_ptr<Instruction>> owned;
  for (size_t pc = 0, byte_offset = 0; pc < inst_data_size;) {
    if (inst_data[pc] == 0) {
      ++pc;
      byte_offset += sizeof(uint32_t);
      continue;
    }
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, &inst_data[pc], byte_offset));
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

  (void)discover_indirect_branch_edges(
      std::span<const Instruction *const>(decoded_insts.data(), decoded_insts.size()), text,
      ROCJITSU_CODE_ARCH_CDNA5, {}, ExternalEntryPolicy::InferPredecessorless, builders);
}

TEST(IndirectBranchDiscovery, ReusedPairPublishesTheCompletedBuilderItReplaces) {
  constexpr uint16_t kPair = 0;
  constexpr uint16_t kLiteral = 255;
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA5;
  std::vector<uint32_t> words = {
      build_s_getpc_b64(kPair, kArch),
      build_s_add_u32(kPair, kPair, kLiteral, kArch),
      0x00000010u,                                        // literal
      build_s_addc_u32(kPair + 1, kPair + 1, 128, kArch), // closes the 64-bit edit
      build_s_getpc_b64(kPair, kArch),                    // reseeds the same pair
      build_s_add_u32(kPair, kPair, kLiteral, kArch),
      0x00000008u, // literal
      build_s_addc_u32(kPair + 1, kPair + 1, 128, kArch),
      // A consumer is what makes the pass report builders at all.
      build_s_setpc_b64(kPair, kArch),
      build_s_endpgm(kArch),
  };

  std::vector<PcAddressBuilder> builders;
  run_indirect_discovery_for_test(std::move(words), &builders);

  ASSERT_EQ(builders.size(), 2u);
  EXPECT_EQ(builders[0].source_getpc_offset, 0u);
  EXPECT_TRUE(builders[0].resolved) << "a completed chain must survive its pair being reused";
  EXPECT_FALSE(builders[0].poisoned);
}

// The same reseed, but the first chain never got its s_addc_u32. The high half is unwritten, so the
// pair does not hold the address the low add implies. Publishing it would let the patcher
// regenerate [begin, end) -- which stops before the missing carry -- and claim an address the
// program never computed, so this one has to fail closed instead.
TEST(IndirectBranchDiscovery, ReusedPairPoisonsAnAbandonedHalfBuiltBuilder) {
  constexpr uint16_t kPair = 0;
  constexpr uint16_t kLiteral = 255;
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA5;
  std::vector<uint32_t> words = {
      build_s_getpc_b64(kPair, kArch),
      build_s_add_u32(kPair, kPair, kLiteral, kArch), // low half only -- no s_addc_u32
      0x00000010u,                                    // literal
      build_s_getpc_b64(kPair, kArch),                // abandons the half-built chain
      build_s_add_u32(kPair, kPair, kLiteral, kArch),
      0x00000008u, // literal
      build_s_addc_u32(kPair + 1, kPair + 1, 128, kArch),
      build_s_setpc_b64(kPair, kArch),
      build_s_endpgm(kArch),
  };

  std::vector<PcAddressBuilder> builders;
  run_indirect_discovery_for_test(std::move(words), &builders);

  ASSERT_FALSE(builders.empty());
  const auto abandoned = std::ranges::find(builders, 0u, &PcAddressBuilder::source_getpc_offset);
  ASSERT_NE(abandoned, builders.end());
  EXPECT_TRUE(abandoned->poisoned) << "a chain abandoned before its carry must not be published";
  EXPECT_FALSE(abandoned->resolved);
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
  constexpr auto set_vgpr_msb = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 0x5a82});
  constexpr auto move = cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 257, .vdst = 0});
  constexpr auto end = cdna5::build_sopp(cdna5::kSEndpgmSopp);
  TestCodeObject co({set_vgpr_msb[0], move[0], end[0]});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_EQ(blocks.size(), 1u);
  auto scope = block_scope(blocks);

  LivenessAnalysisOptions options;
  options.arch = ROCJITSU_CODE_ARCH_CDNA5;
  options.entry_block = scope.front();
  options.text = text_span(co);
  const ExecMaskAnalysis exec(KernelBlockScope(scope), /*wave_size=*/64);
  LivenessAnalysis liveness(KernelBlockScope(scope), std::make_unique<ExecMaskAnalysis>(exec),
                            options);

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
  constexpr auto set_dst_bank_two = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 0x80});
  constexpr auto move = cdna5::build_vop1(cdna5::kVMovB16Vop1, {.src0 = 128, .vdst = 1});
  constexpr auto end = cdna5::build_sopp(cdna5::kSEndpgmSopp);
  TestCodeObject co({set_dst_bank_two[0], move[0], end[0]});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_EQ(blocks.size(), 1u);
  auto scope = block_scope(blocks);

  LivenessAnalysisOptions options;
  options.arch = ROCJITSU_CODE_ARCH_CDNA5;
  options.entry_block = scope.front();
  options.text = text_span(co);
  const ExecMaskAnalysis exec(KernelBlockScope(scope), /*wave_size=*/64);
  LivenessAnalysis liveness(KernelBlockScope(scope), std::make_unique<ExecMaskAnalysis>(exec),
                            options);

  auto instruction = blocks.front()->instructions().begin();
  ++instruction;
  ASSERT_NE(instruction, blocks.front()->instructions().end());
  EXPECT_EQ(liveness.vgpr_msb_bank_before(*instruction, amdgpu::VgprMsbRole::Dst), 2);
  EXPECT_TRUE(liveness.is_live_before(*instruction, {RegClass::VGPR, 513, 1}))
      << "implicit RMW read of the destination must resolve to the DST bank";
  EXPECT_FALSE(liveness.is_live_before(*instruction, {RegClass::VGPR, 1, 1}))
      << "the low-bank alias must not be treated as the read register";
}

TEST(LivenessAnalysis, Gfx1250D16LoadDoesNotReadDestination) {
  // gfx1250 SRAM ECC makes a D16 load a full-dword write by zero-filling the
  // unselected half. Even with DST bank 2 (0x80), neither physical v513 nor its
  // low-bank alias v1 is therefore live before the load.
  constexpr auto set_dst_bank_two = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 0x80});
  constexpr auto load = cdna5::build_vflat(cdna5::kFlatLoadD16U8Vflat, {.vdst = 1});
  constexpr auto end = cdna5::build_sopp(cdna5::kSEndpgmSopp);
  TestCodeObject co({set_dst_bank_two[0], load[0], load[1], load[2], end[0]});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_EQ(blocks.size(), 1u);
  auto scope = block_scope(blocks);

  LivenessAnalysisOptions options;
  options.arch = ROCJITSU_CODE_ARCH_CDNA5;
  options.entry_block = scope.front();
  options.text = text_span(co);
  LivenessAnalysis liveness(KernelBlockScope(scope), nullptr, options);

  auto instruction = blocks.front()->instructions().begin();
  ++instruction;
  ASSERT_NE(instruction, blocks.front()->instructions().end());
  ASSERT_EQ(std::string_view((*instruction).mnemonic()), "flat_load_d16_u8");
  EXPECT_EQ(liveness.vgpr_msb_bank_before(*instruction, amdgpu::VgprMsbRole::Dst), 2);
  EXPECT_FALSE(liveness.is_live_before(*instruction, {RegClass::VGPR, 513, 1}))
      << "SRAM ECC zero-fills the unselected half instead of preserving it";
  EXPECT_FALSE(liveness.is_live_before(*instruction, {RegClass::VGPR, 1, 1}))
      << "the low-bank alias must not be treated as a read register";
}

TEST(LivenessAnalysis, Gfx1250ImplicitVgprUseResolvesDespiteExplicitBank0Alias) {
  // Aliasing case: v_mov_b16 v1, v1 reads v1 as an explicit SRC0 (bank 0) and
  // also preserve-reads its destination v1 in DST bank 2 (physical v513). A
  // "newly-added bits" recovery would miss v513 because raw v1 is already present
  // from the explicit source; the per-operand path must add v513 regardless.
  // 0x80 selects DST bank 2, SRC0 bank 0.
  constexpr auto set_dst_bank_two = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 0x80});
  constexpr auto move = cdna5::build_vop1(cdna5::kVMovB16Vop1, {.src0 = 256 + 1, .vdst = 1});
  constexpr auto end = cdna5::build_sopp(cdna5::kSEndpgmSopp);
  TestCodeObject co({set_dst_bank_two[0], move[0], end[0]});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_EQ(blocks.size(), 1u);
  auto scope = block_scope(blocks);

  LivenessAnalysisOptions options;
  options.arch = ROCJITSU_CODE_ARCH_CDNA5;
  options.entry_block = scope.front();
  options.text = text_span(co);
  const ExecMaskAnalysis exec(KernelBlockScope(scope), /*wave_size=*/64);
  LivenessAnalysis liveness(KernelBlockScope(scope), std::make_unique<ExecMaskAnalysis>(exec),
                            options);

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
  constexpr auto set_banks = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 0x81});
  constexpr auto swap = cdna5::build_vop1(cdna5::kVSwapB16Vop1, {.src0 = 256 + 2, .vdst = 1});
  constexpr auto end = cdna5::build_sopp(cdna5::kSEndpgmSopp);
  TestCodeObject co({set_banks[0], swap[0], end[0]});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_EQ(blocks.size(), 1u);
  auto scope = block_scope(blocks);

  LivenessAnalysisOptions options;
  options.arch = ROCJITSU_CODE_ARCH_CDNA5;
  options.entry_block = scope.front();
  options.text = text_span(co);
  const ExecMaskAnalysis exec(KernelBlockScope(scope), /*wave_size=*/64);
  LivenessAnalysis liveness(KernelBlockScope(scope), std::make_unique<ExecMaskAnalysis>(exec),
                            options);

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
  constexpr auto set_banks = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 0x80});
  // VOP1 word0: enc[31:25]=0x3F, vdst[24:17]=5, op[15:9]=kVMovB32Vop1,
  // src0[8:0]=SRC_DPP. DPP word1: row_mask[31:28]=0x7 (partial),
  // bank_mask[27:24]=0xF, vsrc0[7:0]=2.
  constexpr uint32_t kDppMovWord0 =
      (0x3Fu << 25) | (5u << 17) | (uint32_t{cdna5::kVMovB32Vop1} << 9) | amdgpu::SRC_DPP;
  constexpr uint32_t kDppWord1Partial = (0x7u << 28) | (0xFu << 24) | 2u;
  constexpr auto end = cdna5::build_sopp(cdna5::kSEndpgmSopp);
  TestCodeObject co({set_banks[0], kDppMovWord0, kDppWord1Partial, end[0]});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_EQ(blocks.size(), 1u);
  auto scope = block_scope(blocks);

  LivenessAnalysisOptions options;
  options.arch = ROCJITSU_CODE_ARCH_CDNA5;
  options.entry_block = scope.front();
  options.text = text_span(co);
  const ExecMaskAnalysis exec(KernelBlockScope(scope), /*wave_size=*/64);
  LivenessAnalysis liveness(KernelBlockScope(scope), std::make_unique<ExecMaskAnalysis>(exec),
                            options);

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
      cdna5::build_sopk(cdna5::kSSetregB32Sopk, {.simm16 = kModeAllBanksHwreg, .sdst = 0});
  constexpr auto move = cdna5::build_vop1(cdna5::kVMovB16Vop1, {.src0 = 128, .vdst = 1});
  constexpr auto end = cdna5::build_sopp(cdna5::kSEndpgmSopp);
  TestCodeObject co({setreg[0], move[0], end[0]});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_EQ(blocks.size(), 1u);
  auto scope = block_scope(blocks);

  LivenessAnalysisOptions options;
  options.arch = ROCJITSU_CODE_ARCH_CDNA5;
  options.entry_block = scope.front();
  options.text = text_span(co);
  const ExecMaskAnalysis exec(KernelBlockScope(scope), /*wave_size=*/64);
  LivenessAnalysis liveness(KernelBlockScope(scope), std::make_unique<ExecMaskAnalysis>(exec),
                            options);

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
      cdna5::build_sopk(cdna5::kSSetregB32Sopk, {.simm16 = kModeAllBanksHwreg, .sdst = 0});
  constexpr auto move = cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 128, .vdst = 1});
  constexpr auto end = cdna5::build_sopp(cdna5::kSEndpgmSopp);
  TestCodeObject co({setreg[0], move[0], end[0]});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_EQ(blocks.size(), 1u);
  auto scope = block_scope(blocks);

  LivenessAnalysisOptions options;
  options.arch = ROCJITSU_CODE_ARCH_CDNA5;
  options.entry_block = scope.front();
  options.text = text_span(co);
  const ExecMaskAnalysis exec(KernelBlockScope(scope), /*wave_size=*/64);
  LivenessAnalysis liveness(KernelBlockScope(scope), std::make_unique<ExecMaskAnalysis>(exec),
                            options);

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
  constexpr auto move = cdna5::build_vop1(cdna5::kVMovrelsB32Vop1, {.src0 = 256, .vdst = 2});
  constexpr auto end = cdna5::build_sopp(cdna5::kSEndpgmSopp);
  TestCodeObject co({move[0], end[0]});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_EQ(blocks.size(), 1u);
  auto scope = block_scope(blocks);

  LivenessAnalysisOptions options;
  options.arch = ROCJITSU_CODE_ARCH_CDNA5;
  options.entry_block = scope.front();
  options.text = text_span(co);
  const ExecMaskAnalysis exec(KernelBlockScope(scope), /*wave_size=*/64);
  LivenessAnalysis liveness(KernelBlockScope(scope), std::make_unique<ExecMaskAnalysis>(exec),
                            options);

  const Instruction &instruction = *blocks.front()->instructions().begin();
  EXPECT_EQ(liveness.find_globally_unused_vgpr_run(&instruction, 1, 1, 1, 2), std::nullopt);
  EXPECT_FALSE(liveness.has_materialized_cfg_liveness());
}

TEST(LivenessAnalysis, Gfx1250SwaprelDisablesGlobalUnusedQuery) {
  constexpr auto swap = cdna5::build_vop1(cdna5::kVSwaprelB32Vop1, {.src0 = 256, .vdst = 2});
  constexpr auto end = cdna5::build_sopp(cdna5::kSEndpgmSopp);
  TestCodeObject co({swap[0], end[0]});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_EQ(blocks.size(), 1u);
  auto scope = block_scope(blocks);

  LivenessAnalysisOptions options;
  options.arch = ROCJITSU_CODE_ARCH_CDNA5;
  options.entry_block = scope.front();
  options.text = text_span(co);
  const ExecMaskAnalysis exec(KernelBlockScope(scope), /*wave_size=*/64);
  LivenessAnalysis liveness(KernelBlockScope(scope), std::make_unique<ExecMaskAnalysis>(exec),
                            options);

  const Instruction &instruction = *blocks.front()->instructions().begin();
  EXPECT_EQ(liveness.find_globally_unused_vgpr_run(&instruction, 1, 1, 1, 2), std::nullopt);
  EXPECT_FALSE(liveness.has_materialized_cfg_liveness());
}

TEST(LivenessAnalysis, Gfx1250DynamicModeBit27WriteDoesNotEnableGprIndexing) {
  constexpr uint16_t kModeGprIdxEnableHwreg = 1u | (27u << 6);
  constexpr auto setreg =
      cdna5::build_sopk(cdna5::kSSetregB32Sopk, {.simm16 = kModeGprIdxEnableHwreg, .sdst = 0});
  constexpr auto move = cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 256, .vdst = 0});
  constexpr auto end = cdna5::build_sopp(cdna5::kSEndpgmSopp);
  TestCodeObject co({setreg[0], move[0], end[0]});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_EQ(blocks.size(), 1u);
  auto scope = block_scope(blocks);

  LivenessAnalysisOptions options;
  options.arch = ROCJITSU_CODE_ARCH_CDNA5;
  options.entry_block = scope.front();
  options.text = text_span(co);
  const ExecMaskAnalysis exec(KernelBlockScope(scope), /*wave_size=*/64);
  LivenessAnalysis liveness(KernelBlockScope(scope), std::make_unique<ExecMaskAnalysis>(exec),
                            options);

  auto instruction = blocks.front()->instructions().begin();
  ++instruction;
  ASSERT_NE(instruction, blocks.front()->instructions().end());
  EXPECT_EQ(liveness.find_globally_unused_vgpr_run(&*instruction, 1, 1, 1, 2), 1);
  EXPECT_FALSE(liveness.has_materialized_cfg_liveness());
}

TEST(LivenessAnalysis, Rdna4DynamicModeBit27WriteDoesNotEnableGprIndexing) {
  constexpr uint16_t kModeDisablePerfHwreg = 1u | (27u << 6);
  constexpr auto setreg =
      rdna4::build_sopk(rdna4::kSSetregB32Sopk, {.simm16 = kModeDisablePerfHwreg, .sdst = 0});
  constexpr auto move = rdna4::build_vop1(rdna4::kVMovB32Vop1, {.src0 = 256, .vdst = 0});
  constexpr auto end = rdna4::build_sopp(rdna4::kSEndpgmSopp);
  TestCodeObject co({setreg[0], move[0], end[0]});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_EQ(blocks.size(), 1u);
  auto scope = block_scope(blocks);

  LivenessAnalysisOptions options;
  options.arch = ROCJITSU_CODE_ARCH_RDNA4;
  options.entry_block = scope.front();
  options.text = text_span(co);
  const ExecMaskAnalysis exec(KernelBlockScope(scope), /*wave_size=*/32);
  LivenessAnalysis liveness(KernelBlockScope(scope), std::make_unique<ExecMaskAnalysis>(exec),
                            options);

  auto instruction = blocks.front()->instructions().begin();
  ++instruction;
  ASSERT_NE(instruction, blocks.front()->instructions().end());
  EXPECT_EQ(liveness.find_globally_unused_vgpr_run(&*instruction, 1, 1, 1, 2), 1);
  EXPECT_FALSE(liveness.has_materialized_cfg_liveness());
}

TEST(LivenessAnalysis, Gfx1250ImmediateModeBit27WriteDoesNotEnableGprIndexing) {
  constexpr uint16_t kModeGprIdxEnableHwreg = 1u | (27u << 6);
  constexpr auto setreg =
      cdna5::build_sopk(cdna5::kSSetregImm32B32Sopk, {.simm16 = kModeGprIdxEnableHwreg});
  constexpr auto move = cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 256, .vdst = 0});
  constexpr auto end = cdna5::build_sopp(cdna5::kSEndpgmSopp);

  for (uint32_t literal : {0u, 1u}) {
    SCOPED_TRACE(literal);
    TestCodeObject co({setreg[0], literal, move[0], end[0]});
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
    ASSERT_NE(decoder, nullptr);
    auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);
    ASSERT_EQ(blocks.size(), 1u);
    auto scope = block_scope(blocks);

    LivenessAnalysisOptions options;
    options.arch = ROCJITSU_CODE_ARCH_CDNA5;
    options.entry_block = scope.front();
    options.text = text_span(co);
    const ExecMaskAnalysis exec(KernelBlockScope(scope), /*wave_size=*/64);
    LivenessAnalysis liveness(KernelBlockScope(scope), std::make_unique<ExecMaskAnalysis>(exec),
                              options);

    auto instruction = blocks.front()->instructions().begin();
    ++instruction;
    ASSERT_NE(instruction, blocks.front()->instructions().end());
    const auto unused = liveness.find_globally_unused_vgpr_run(&*instruction, 1, 1, 1, 2);
    EXPECT_EQ(unused, 1);
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
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_EQ(blocks.size(), 1u);
  auto scope = block_scope(blocks);

  LivenessAnalysisOptions options;
  options.arch = ROCJITSU_CODE_ARCH_CDNA4;
  options.text = text_span(co);
  const ExecMaskAnalysis exec(KernelBlockScope(scope), /*wave_size=*/64);
  LivenessAnalysis liveness(KernelBlockScope(scope), std::make_unique<ExecMaskAnalysis>(exec),
                            options);

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
    auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);
    ASSERT_EQ(blocks.size(), 1u);
    auto scope = block_scope(blocks);

    LivenessAnalysisOptions options;
    options.arch = ROCJITSU_CODE_ARCH_CDNA4;
    options.text = text_span(co);
    const ExecMaskAnalysis exec(KernelBlockScope(scope), /*wave_size=*/64);
    LivenessAnalysis liveness(KernelBlockScope(scope), std::make_unique<ExecMaskAnalysis>(exec),
                              options);

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
      cdna5::build_sopk(cdna5::kSSetregB32Sopk, {.simm16 = kModeSrc0Hwreg, .sdst = 0});
  constexpr auto move = cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 257, .vdst = 0});
  constexpr auto end = cdna5::build_sopp(cdna5::kSEndpgmSopp);
  TestCodeObject co({setreg[0], move[0], end[0]});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_EQ(blocks.size(), 1u);
  auto scope = block_scope(blocks);

  LivenessAnalysisOptions options;
  options.arch = ROCJITSU_CODE_ARCH_CDNA5;
  options.entry_block = scope.front();
  options.text = text_span(co);
  const ExecMaskAnalysis exec(KernelBlockScope(scope), /*wave_size=*/64);
  LivenessAnalysis liveness(KernelBlockScope(scope), std::make_unique<ExecMaskAnalysis>(exec),
                            options);

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
      cdna5::build_sopk(cdna5::kSSetregB32Sopk, {.simm16 = kModeSrc0Hwreg, .sdst = 0});
  constexpr auto literal_setreg =
      cdna5::build_sopk(cdna5::kSSetregImm32B32Sopk, {.simm16 = kModeSrc0Hwreg});
  constexpr auto move = cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 257, .vdst = 0});
  constexpr auto end = cdna5::build_sopp(cdna5::kSEndpgmSopp);
  TestCodeObject co({dynamic_setreg[0], literal_setreg[0], 2u, move[0], end[0]});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_EQ(blocks.size(), 1u);
  auto scope = block_scope(blocks);

  LivenessAnalysisOptions options;
  options.arch = ROCJITSU_CODE_ARCH_CDNA5;
  options.entry_block = scope.front();
  options.text = text_span(co);
  const ExecMaskAnalysis exec(KernelBlockScope(scope), /*wave_size=*/64);
  LivenessAnalysis liveness(KernelBlockScope(scope), std::make_unique<ExecMaskAnalysis>(exec),
                            options);

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
  constexpr auto set_bank_two = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 2});
  constexpr uint16_t kModeAllBanksHwreg = 1u | (12u << 6) | (7u << 11);
  constexpr auto literal_setreg =
      cdna5::build_sopk(cdna5::kSSetregImm32B32Sopk, {.simm16 = kModeAllBanksHwreg});
  constexpr auto move = cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 257, .vdst = 0});
  constexpr auto end = cdna5::build_sopp(cdna5::kSEndpgmSopp);
  // Full program (so decode sees a valid literal + terminator), but the analysis is
  // told the text ends right after the setreg encoding word at offset 4 (its
  // literal at offset 8 is out of range).
  TestCodeObject co({set_bank_two[0], literal_setreg[0], 0xe4u, move[0], end[0]});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_EQ(blocks.size(), 1u);
  auto scope = block_scope(blocks);

  LivenessAnalysisOptions options;
  options.arch = ROCJITSU_CODE_ARCH_CDNA5;
  options.entry_block = scope.front();
  // Truncate the text span to 8 bytes: the setreg (at offset 4) has no readable
  // literal at offset 8.
  const auto full = text_span(co);
  options.text = full.subspan(0, 8);
  const ExecMaskAnalysis exec(KernelBlockScope(scope), /*wave_size=*/64);
  LivenessAnalysis liveness(KernelBlockScope(scope), std::make_unique<ExecMaskAnalysis>(exec),
                            options);

  auto instruction = blocks.front()->instructions().begin();
  std::advance(instruction, 2);
  ASSERT_NE(instruction, blocks.front()->instructions().end());
  EXPECT_EQ(instruction.operator*().mnemonic(), "v_mov_b32_e32");
  // Bank 2 was set before the truncated mode write; because the mode write's
  // literal is unreadable, the Src0 bank must be ambiguous, not the pre-write 2.
  EXPECT_EQ(liveness.vgpr_msb_bank_before(*instruction, amdgpu::VgprMsbRole::Src0), std::nullopt);
}

TEST(LivenessAnalysis, Gfx1250PartialLiteralModeWritePreservesUntouchedBankBit) {
  constexpr auto set_bank_one = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 1});
  constexpr uint16_t kModeSrc0HighBitHwreg = 1u | (15u << 6);
  constexpr auto literal_setreg =
      cdna5::build_sopk(cdna5::kSSetregImm32B32Sopk, {.simm16 = kModeSrc0HighBitHwreg});
  constexpr auto move = cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 257, .vdst = 0});
  constexpr auto end = cdna5::build_sopp(cdna5::kSEndpgmSopp);
  TestCodeObject co({set_bank_one[0], literal_setreg[0], 1u, move[0], end[0]});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_EQ(blocks.size(), 1u);
  auto scope = block_scope(blocks);

  LivenessAnalysisOptions options;
  options.arch = ROCJITSU_CODE_ARCH_CDNA5;
  options.entry_block = scope.front();
  options.text = text_span(co);
  const ExecMaskAnalysis exec(KernelBlockScope(scope), /*wave_size=*/64);
  LivenessAnalysis liveness(KernelBlockScope(scope), std::make_unique<ExecMaskAnalysis>(exec),
                            options);

  auto instruction = blocks.front()->instructions().begin();
  std::advance(instruction, 2);
  ASSERT_NE(instruction, blocks.front()->instructions().end());
  EXPECT_EQ(instruction.operator*().mnemonic(), "v_mov_b32_e32");
  EXPECT_EQ(liveness.vgpr_msb_bank_before(*instruction, amdgpu::VgprMsbRole::Src0), 3);
  EXPECT_TRUE(liveness.is_live_before(*instruction, {RegClass::VGPR, 769, 1}));
}

TEST(LivenessAnalysis, Gfx1250ImmediateModeWritePreservesBanksOutsideRequestedSlice) {
  constexpr auto set_bank_one = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 1});
  // Request a write to MODE bit zero, disjoint from MODE.VGPR_MSB.
  constexpr uint16_t kModeBitZeroHwreg = 1u;
  constexpr auto literal_setreg =
      cdna5::build_sopk(cdna5::kSSetregImm32B32Sopk, {.simm16 = kModeBitZeroHwreg});
  constexpr auto move = cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 257, .vdst = 0});
  constexpr auto end = cdna5::build_sopp(cdna5::kSEndpgmSopp);
  TestCodeObject co({set_bank_one[0], literal_setreg[0], 0u, move[0], end[0]});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_EQ(blocks.size(), 1u);
  auto scope = block_scope(blocks);

  LivenessAnalysisOptions options;
  options.arch = ROCJITSU_CODE_ARCH_CDNA5;
  options.entry_block = scope.front();
  options.text = text_span(co);
  const ExecMaskAnalysis exec(KernelBlockScope(scope), /*wave_size=*/64);
  LivenessAnalysis liveness(KernelBlockScope(scope), std::make_unique<ExecMaskAnalysis>(exec),
                            options);

  auto instruction = blocks.front()->instructions().begin();
  std::advance(instruction, 2);
  ASSERT_NE(instruction, blocks.front()->instructions().end());
  EXPECT_EQ(liveness.vgpr_msb_bank_before(*instruction, amdgpu::VgprMsbRole::Src0), 1);
}

TEST(LivenessAnalysis, Gfx1250LiteralModeWriteTracksEveryRole) {
  constexpr uint16_t kAllVgprMsbFieldsHwreg = 1u | (12u << 6) | (7u << 11);
  constexpr auto literal_setreg =
      cdna5::build_sopk(cdna5::kSSetregImm32B32Sopk, {.simm16 = kAllVgprMsbFieldsHwreg});
  // MODE[19:12] is {src2=3, src1=2, src0=1, dst=0}.
  constexpr uint32_t kModeFields = 0xe4u;
  constexpr auto move = cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 257, .vdst = 0});
  constexpr auto end = cdna5::build_sopp(cdna5::kSEndpgmSopp);
  TestCodeObject co({literal_setreg[0], kModeFields, move[0], end[0]});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_EQ(blocks.size(), 1u);
  auto scope = block_scope(blocks);

  LivenessAnalysisOptions options;
  options.arch = ROCJITSU_CODE_ARCH_CDNA5;
  options.entry_block = scope.front();
  options.text = text_span(co);
  const ExecMaskAnalysis exec(KernelBlockScope(scope), /*wave_size=*/64);
  LivenessAnalysis liveness(KernelBlockScope(scope), std::make_unique<ExecMaskAnalysis>(exec),
                            options);

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
      cdna5::build_sopk(cdna5::kSSetregImm32B32Sopk, {.simm16 = kNonModeHwreg});
  constexpr auto move = cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 257, .vdst = 0});
  constexpr auto end = cdna5::build_sopp(cdna5::kSEndpgmSopp);
  TestCodeObject co({literal_setreg[0], 0x000ff000u, move[0], end[0]});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_EQ(blocks.size(), 1u);
  auto scope = block_scope(blocks);

  LivenessAnalysisOptions options;
  options.arch = ROCJITSU_CODE_ARCH_CDNA5;
  options.entry_block = scope.front();
  options.text = text_span(co);
  const ExecMaskAnalysis exec(KernelBlockScope(scope), /*wave_size=*/64);
  LivenessAnalysis liveness(KernelBlockScope(scope), std::make_unique<ExecMaskAnalysis>(exec),
                            options);

  auto instruction = blocks.front()->instructions().begin();
  ++instruction;
  ASSERT_NE(instruction, blocks.front()->instructions().end());
  EXPECT_EQ(liveness.vgpr_msb_bank_before(*instruction, amdgpu::VgprMsbRole::Src0), 0);
  EXPECT_EQ(liveness.vgpr_msb_bank_before(*instruction, amdgpu::VgprMsbRole::Src1), 0);
  EXPECT_EQ(liveness.vgpr_msb_bank_before(*instruction, amdgpu::VgprMsbRole::Src2), 0);
  EXPECT_EQ(liveness.vgpr_msb_bank_before(*instruction, amdgpu::VgprMsbRole::Dst), 0);
}

TEST(LivenessAnalysis, Gfx1250VgprMsbCfgJoinRequiresPredecessorsToAgree) {
  constexpr auto branch_to_else = cdna5::build_sopp(cdna5::kSCbranchScc0Sopp, {.simm16 = 2});
  constexpr auto set_bank_two = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 0x82});
  constexpr auto branch_to_join = cdna5::build_sopp(cdna5::kSBranchSopp, {.simm16 = 1});
  constexpr auto set_bank_zero = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp);
  constexpr auto move = cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 257, .vdst = 0});
  constexpr auto end = cdna5::build_sopp(cdna5::kSEndpgmSopp);
  TestCodeObject co(
      {branch_to_else[0], set_bank_two[0], branch_to_join[0], set_bank_zero[0], move[0], end[0]});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);
  auto scope = block_scope(blocks);
  BasicBlock *join = block_starting_at(blocks, 16);
  ASSERT_NE(join, nullptr);

  LivenessAnalysisOptions options;
  options.arch = ROCJITSU_CODE_ARCH_CDNA5;
  options.entry_block = scope.front();
  options.text = text_span(co);
  const ExecMaskAnalysis exec(KernelBlockScope(scope), /*wave_size=*/64);
  LivenessAnalysis liveness(KernelBlockScope(scope), std::make_unique<ExecMaskAnalysis>(exec),
                            options);

  const Instruction &joined_move = *join->instructions().begin();
  EXPECT_EQ(liveness.vgpr_msb_bank_before(joined_move, amdgpu::VgprMsbRole::Src0), std::nullopt);
  for (uint16_t bank = 0; bank < 4; ++bank)
    EXPECT_TRUE(liveness.is_live_before(
        joined_move, {RegClass::VGPR, static_cast<uint16_t>(1 + bank * 256), 1}));
}

TEST(LivenessAnalysis, Gfx1250VgprMsbCfgJoinPreservesAgreeingBank) {
  constexpr auto branch_to_else = cdna5::build_sopp(cdna5::kSCbranchScc0Sopp, {.simm16 = 2});
  constexpr auto set_bank_two = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 2});
  constexpr auto branch_to_join = cdna5::build_sopp(cdna5::kSBranchSopp, {.simm16 = 1});
  constexpr auto move = cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 257, .vdst = 0});
  constexpr auto end = cdna5::build_sopp(cdna5::kSEndpgmSopp);
  TestCodeObject co(
      {branch_to_else[0], set_bank_two[0], branch_to_join[0], set_bank_two[0], move[0], end[0]});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);
  auto scope = block_scope(blocks);
  BasicBlock *join = block_starting_at(blocks, 16);
  ASSERT_NE(join, nullptr);

  LivenessAnalysisOptions options;
  options.arch = ROCJITSU_CODE_ARCH_CDNA5;
  options.entry_block = scope.front();
  options.text = text_span(co);
  const ExecMaskAnalysis exec(KernelBlockScope(scope), /*wave_size=*/64);
  LivenessAnalysis liveness(KernelBlockScope(scope), std::make_unique<ExecMaskAnalysis>(exec),
                            options);

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
  constexpr auto branch_over = cdna5::build_sopp(cdna5::kSBranchSopp, {.simm16 = 1});
  constexpr auto set_bank_zero = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp);
  constexpr auto set_bank_two = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 2});
  constexpr auto move = cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 257, .vdst = 0});
  constexpr auto end = cdna5::build_sopp(cdna5::kSEndpgmSopp);
  TestCodeObject co({branch_over[0], set_bank_zero[0], set_bank_two[0], move[0], end[0]});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  auto blocks = build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA5);
  auto scope = block_scope(blocks);

  LivenessAnalysisOptions options;
  options.arch = ROCJITSU_CODE_ARCH_CDNA5;
  options.entry_block = scope.front();
  options.text = text_span(co);
  const ExecMaskAnalysis exec(KernelBlockScope(scope), /*wave_size=*/64);
  LivenessAnalysis liveness(KernelBlockScope(scope), std::make_unique<ExecMaskAnalysis>(exec),
                            options);

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
  return build_valid_blocks(co, *decoder, ROCJITSU_CODE_ARCH_CDNA4);
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
  ASSERT_TRUE(insts[0]->flags() & RESULT_COPY)
      << "s_mov lacks RESULT_COPY; regenerate ISA to enable EXEC-Full tracking";

  // EXEC is provably full at the def, so its vector write is a real kill.
  auto scope = block_scope(blocks);
  ExecMaskAnalysis exec{KernelBlockScope(scope), 64};
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
  ExecMaskAnalysis exec{KernelBlockScope(scope), 64};
  EXPECT_EQ(exec.before(def), ExecState::Unknown);

  LivenessAnalysis liveness = analyze_scope(blocks);
  EXPECT_TRUE(liveness.is_live_before(def, {RegClass::VGPR, 0, 1}));
}

// v_writelane_b32 writes a single lane, so even under Full EXEC it is a
// read-modify-write of its destination and must not be promoted to a kill.
TEST(ExecFlagsRealDecode, Cdna4WritelaneDestStaysLiveUnderFullExec) {
  // s_mov_b64 exec, -1 ; v_writelane_b32 v5, s4, s2 ; v_mov_b32 v2, v5 ; s_endpgm
  auto blocks =
      build_cdna4_blocks({0xBEFE01C1u, 0xD28A0005u, 0x00000404u, 0x7E040305u, 0xBF810000u});
  ASSERT_FALSE(blocks.empty());
  auto insts = insts_of(*blocks[0]);
  ASSERT_GE(insts.size(), 3u);

  // Sanity-check the decode so an encoding typo fails loudly.
  EXPECT_EQ(insts[0]->mnemonic(), "s_mov_b64");
  const Instruction &writelane = *insts[1];
  EXPECT_EQ(writelane.mnemonic(), "v_writelane_b32");
  ASSERT_NE(writelane.dst_operand(0), nullptr);
  auto dst_ref = writelane.dst_operand(0)->to_register_ref();
  ASSERT_TRUE(dst_ref.has_value());
  EXPECT_EQ(*dst_ref, (RegisterRef{RegClass::VGPR, 5, 1}));

  // Full-EXEC tracking needs the regenerated s_mov RESULT_COPY metadata.
  ASSERT_TRUE(insts[0]->flags() & RESULT_COPY)
      << "s_mov lacks RESULT_COPY; regenerate ISA to enable EXEC-Full tracking";

  auto scope = block_scope(blocks);
  ExecMaskAnalysis exec{KernelBlockScope(scope), 64};
  EXPECT_EQ(exec.before(writelane), ExecState::Full);

  // Contrast Cdna4SMovExecAllOnesPromotesVgprDefToKill: the RMW dst stays live.
  LivenessAnalysis liveness = analyze_scope(blocks);
  EXPECT_TRUE(liveness.is_live_before(writelane, {RegClass::VGPR, 5, 1}));
}

TEST(ExecMaskAnalysis, EntryIsUnknownAllOnesIsFullNarrowingIsUnknown) {
  // exec=all-ones; v0=...; exec=narrow; v0=...; end
  auto blocks =
      build_test_blocks({TestOpcode::WriteExecFull, TestOpcode::DefVgpr0,
                         TestOpcode::WriteExecNarrow, TestOpcode::DefVgpr0, TestOpcode::End});
  auto scope = block_scope(blocks);
  ExecMaskAnalysis exec{KernelBlockScope(scope), 64};

  auto insts = insts_of(*blocks[0]);
  ASSERT_EQ(insts.size(), 5u);
  EXPECT_EQ(exec.before(*insts[0]), ExecState::Unknown); // kernel entry
  EXPECT_EQ(exec.before(*insts[1]), ExecState::Full);    // after exec=all-ones
  EXPECT_EQ(exec.before(*insts[2]), ExecState::Full);    // narrowing not yet applied
  EXPECT_EQ(exec.before(*insts[3]), ExecState::Unknown); // after narrowing write
}

TEST(ExecMaskAnalysis, LoopHeaderEntryWithBackedgeIsPinnedUnknown) {
  // block0 @0: End -- a predecessor-less natural entry, so the scope-leader
  // fallback does not fire. block1 @4: a self-looping "kernel entry" that writes
  // EXEC all-ones and conditionally branches back to itself; its only in-scope
  // predecessor is that backedge. Without pinning it, the forward meet carries
  // the backedge's Full state into the header and wrongly derives Full there.
  auto blocks = build_test_blocks({TestOpcode::End, TestOpcode::WriteExecFull,
                                   TestOpcode::CBranchBackToUseSgpr4, TestOpcode::End});
  auto scope = block_scope(blocks);
  BasicBlock *header = block_starting_at(blocks, 4);
  ASSERT_NE(header, nullptr);
  ASSERT_TRUE(has_predecessor(*header, header)) << "header must have its own backedge";
  const Instruction &header_first = *insts_of(*header).front();
  EXPECT_TRUE(header_first.mnemonic().starts_with("test_write_exec_full"));

  // Without pinning, the backedge makes the header wrongly Full.
  ExecMaskAnalysis unpinned{KernelBlockScope(scope), 64};
  EXPECT_EQ(unpinned.before(header_first), ExecState::Full);

  // Pinning the real entry keeps it Unknown, as at any kernel entry.
  const BasicBlock *const entries[] = {header};
  ExecMaskAnalysis pinned{KernelBlockScope(scope), 64, /*extra_edges=*/{}, entries};
  EXPECT_EQ(pinned.before(header_first), ExecState::Unknown);
}

TEST(ExecMaskAnalysis, SecondHardwareEntryReachedAfterFullIsPinnedUnknown) {
  // Models a kernel with two hardware entries (descriptor entry + kernarg-preload
  // firmware entry): block0 @0 establishes Full and falls through to block1 @4, a
  // second entry (forced to be a leader). block1 is reachable with Full via the
  // ordinary path, but hardware may enter it directly with unknown EXEC, so both
  // entries must be seeded -- unlike the single-entry tests above.
  const uint64_t firmware_leader[] = {4};
  auto blocks = build_test_blocks(
      {TestOpcode::WriteExecFull, TestOpcode::DefVgpr0, TestOpcode::End}, firmware_leader);
  auto scope = block_scope(blocks);
  BasicBlock *firmware = block_starting_at(blocks, 4);
  ASSERT_NE(firmware, nullptr);
  ASSERT_TRUE(has_predecessor(*firmware, blocks.front().get()));
  const Instruction &firmware_first = *insts_of(*firmware).front();

  // Without pinning, the ordinary fallthrough carries Full into the second entry.
  ExecMaskAnalysis unpinned{KernelBlockScope(scope), 64};
  EXPECT_EQ(unpinned.before(firmware_first), ExecState::Full);

  // Seeding both entries keeps the second Unknown despite the incoming Full.
  const BasicBlock *const entries[] = {blocks.front().get(), firmware};
  ExecMaskAnalysis pinned{KernelBlockScope(scope), 64, /*extra_edges=*/{}, entries};
  EXPECT_EQ(pinned.before(firmware_first), ExecState::Unknown);
}

TEST(ExecMaskAnalysis, OrWithAllOnesConstantIsFull) {
  // exec = exec | -1 -> all-ones regardless of the prior EXEC (RESULT_OR with an
  // all-ones source operand).
  auto blocks =
      build_test_blocks({TestOpcode::WriteExecOrAllOnes, TestOpcode::DefVgpr0, TestOpcode::End});
  auto scope = block_scope(blocks);
  ExecMaskAnalysis exec{KernelBlockScope(scope), 64};

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
  ExecMaskAnalysis exec{KernelBlockScope(scope), 64};

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
    ExecMaskAnalysis exec{KernelBlockScope(scope), 64};
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
    ExecMaskAnalysis exec{KernelBlockScope(scope), 64};
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
  ExecMaskAnalysis exec{KernelBlockScope(scope), 32};
  auto insts = insts_of(*blocks[0]);
  ASSERT_GE(insts.size(), 2u);
  EXPECT_EQ(exec.before(*insts[1]), ExecState::Full);
}

TEST(ExecMaskAnalysis, ExecHiWriteNeverEstablishesFull) {
  // s_mov_b32 exec_hi, -1 writes bits [32,64), which never cover the active
  // lanes. It must not establish Full from Unknown on either wave size — on
  // Wave32 the written bits lie entirely outside the mask, on Wave64 they are a
  // partial upper half.
  for (uint8_t wave_size : {uint8_t{32}, uint8_t{64}}) {
    auto blocks =
        build_test_blocks({TestOpcode::WriteExecHiHalf, TestOpcode::DefVgpr0, TestOpcode::End});
    auto scope = block_scope(blocks);
    ExecMaskAnalysis exec{KernelBlockScope(scope), wave_size};
    auto insts = insts_of(*blocks[0]);
    ASSERT_GE(insts.size(), 2u);
    EXPECT_EQ(exec.before(*insts[1]), ExecState::Unknown);
  }
}

TEST(ExecMaskAnalysis, ExecHiWritePreservesButDoesNotNarrowFull) {
  // From an already-Full mask, an all-ones exec_hi write keeps Full (it only
  // rewrites in-range or upper bits with ones), the same as an exec_lo half.
  auto blocks = build_test_blocks({TestOpcode::WriteExecFull, TestOpcode::WriteExecHiHalf,
                                   TestOpcode::DefVgpr0, TestOpcode::End});
  auto scope = block_scope(blocks);
  ExecMaskAnalysis exec{KernelBlockScope(scope), 64};
  auto insts = insts_of(*blocks[0]);
  ASSERT_GE(insts.size(), 3u);
  EXPECT_EQ(exec.before(*insts[1]), ExecState::Full); // entering the hi write
  EXPECT_EQ(exec.before(*insts[2]), ExecState::Full); // hi all-ones preserved Full
}

TEST(ExecMaskAnalysis, Wave32ExecHiZeroWritePreservesFull) {
  // On Wave32 exec_hi lies outside the active mask, so an exec_hi write preserves
  // the current state even for a non-all-ones value like `s_mov_b32 exec_hi, 0`
  // (which previously narrowed an established Full to Unknown).
  auto blocks = build_test_blocks({TestOpcode::WriteExecFull, TestOpcode::WriteExecHiHalfZero,
                                   TestOpcode::DefVgpr0, TestOpcode::UseVgpr0, TestOpcode::End});
  auto scope = block_scope(blocks);
  ExecMaskAnalysis exec{KernelBlockScope(scope), 32};
  auto insts = insts_of(*blocks[0]);
  ASSERT_GE(insts.size(), 4u);
  EXPECT_EQ(exec.before(*insts[1]), ExecState::Full); // entering the exec_hi=0 write
  EXPECT_EQ(exec.before(*insts[2]), ExecState::Full); // exec_hi=0 preserved Full

  const ExecMaskAnalysis exec_for_liveness{KernelBlockScope(scope), /*wave_size=*/32};
  const LivenessAnalysis liveness{KernelBlockScope(scope),
                                  std::make_unique<ExecMaskAnalysis>(exec_for_liveness)};
  EXPECT_FALSE(liveness.is_live_before(*insts[2], {RegClass::VGPR, 0, 1}));
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
  ExecMaskAnalysis exec{KernelBlockScope(scope), 64};

  BasicBlock *join = block_starting_at(blocks, 12);
  ASSERT_NE(join, nullptr);
  const Instruction &def = *join->instructions().begin();
  EXPECT_EQ(exec.before(def), ExecState::Unknown);

  LivenessAnalysis liveness = analyze_scope(blocks);
  EXPECT_TRUE(liveness.is_live_before(def, {RegClass::VGPR, 0, 1}));
}

// Regression: EXEC must flow across the scoped caller->callee edge like liveness
// does. Without the edge the callee looks like a scope entry (Unknown) and its
// vector def is wrongly left a non-kill; with it EXEC is Full and the def kills.
TEST(ExecMaskAnalysis, CallEdgeFlowsExecFullIntoCallee) {
  // Caller: exec=all-ones; end.   Callee: def v0; use v0; end. The caller's End
  // splits the callee into its own block, reachable only via the scoped edge.
  auto blocks = build_test_blocks({TestOpcode::WriteExecFull, TestOpcode::End, TestOpcode::DefVgpr0,
                                   TestOpcode::UseVgpr0, TestOpcode::End});
  ASSERT_EQ(blocks.size(), 2u);
  BasicBlock *caller = blocks[0].get();
  BasicBlock *callee = blocks[1].get();
  const Instruction &def = *callee->instructions().begin();

  const std::array<ScopedCfgEdge, 1> edges{ScopedCfgEdge{.from = caller, .to = callee}};
  auto scope = block_scope(blocks);

  // No edge: callee is a scope entry, EXEC Unknown, def stays live.
  {
    ExecMaskAnalysis exec{KernelBlockScope(scope), 64};
    EXPECT_EQ(exec.before(def), ExecState::Unknown);
    LivenessAnalysis liveness = analyze_scope(blocks);
    EXPECT_TRUE(liveness.is_live_before(def, {RegClass::VGPR, 0, 1}));
  }

  // Same edge in both: EXEC Full at the def, so it overwrites every lane and kills v0.
  {
    ExecMaskAnalysis exec{KernelBlockScope(scope), 64, edges};
    EXPECT_EQ(exec.before(def), ExecState::Full);
    LivenessAnalysis liveness = analyze_scope_with_edges(blocks, edges);
    EXPECT_FALSE(liveness.is_live_before(def, {RegClass::VGPR, 0, 1}));
  }
}

// Regression: a scoped return->continuation edge must pull the continuation to
// Unknown when the returning path narrowed EXEC. Omitting it from exec leaves the
// continuation looking Full (from its local branch pred) and unsoundly kills.
TEST(ExecMaskAnalysis, ReturnEdgeToContinuationMeetsToUnknown) {
  //   0:  WriteExecFull         (P0)
  //   4:  CBranchToElseAfterTwo (delta +8 -> branch target 16)
  //   8:  WriteExecNarrow       (P1) return block: narrows EXEC
  //   12: End                    P1 terminates -> reaches cont only via the edge
  //   16: DefVgpr0              (cont) branch target of P0; 20: UseVgpr0; 24: End
  auto blocks = build_test_blocks({TestOpcode::WriteExecFull, TestOpcode::CBranchToElseAfterTwo,
                                   TestOpcode::WriteExecNarrow, TestOpcode::End,
                                   TestOpcode::DefVgpr0, TestOpcode::UseVgpr0, TestOpcode::End});
  BasicBlock *ret_block = block_starting_at(blocks, 8);
  BasicBlock *cont = block_starting_at(blocks, 16);
  ASSERT_NE(ret_block, nullptr);
  ASSERT_NE(cont, nullptr);
  const Instruction &def = *cont->instructions().begin();

  const std::array<ScopedCfgEdge, 1> edges{ScopedCfgEdge{.from = ret_block, .to = cont}};
  auto scope = block_scope(blocks);

  // No edge: cont sees only its Full branch pred, so EXEC looks Full and def kills.
  {
    ExecMaskAnalysis exec{KernelBlockScope(scope), 64};
    EXPECT_EQ(exec.before(def), ExecState::Full);
    LivenessAnalysis liveness = analyze_scope(blocks);
    EXPECT_FALSE(liveness.is_live_before(def, {RegClass::VGPR, 0, 1}));
  }

  // With edge: meet of the Full branch and the narrowed return path is Unknown,
  // so the def is not a kill and v0 stays live.
  {
    ExecMaskAnalysis exec{KernelBlockScope(scope), 64, edges};
    EXPECT_EQ(exec.before(def), ExecState::Unknown);
    LivenessAnalysis liveness = analyze_scope_with_edges(blocks, edges);
    EXPECT_TRUE(liveness.is_live_before(def, {RegClass::VGPR, 0, 1}));
  }
}

// Regression: a back-edge into the scope entry gives block 0 a predecessor, but
// EXEC at kernel entry is set by dispatch and must stay Unknown -- the loop
// back-edge must not let interior blocks' optimistic `Full` seed leak in. Guards
// the `i == 0` pin in the predecessor-based entry detection.
TEST(ExecMaskAnalysis, BackEdgeIntoScopeEntryStaysUnknown) {
  //   0: CBranchToElse (+4)    header/entry: loop or exit at 8
  //   4: BranchBackToStart(-8) latch: unconditional back-edge to offset 0
  //   8: End                   loop exit
  auto blocks = build_test_blocks(
      {TestOpcode::CBranchToElse, TestOpcode::BranchBackToStart, TestOpcode::End});
  BasicBlock *entry = blocks[0].get();
  BasicBlock *latch = block_starting_at(blocks, 4);
  ASSERT_NE(latch, nullptr);

  // The latch back-edges into the entry, so it is not caught by the empty-pred
  // case -- only the i==0 pin keeps it an entry.
  EXPECT_TRUE(has_predecessor(*entry, latch));

  auto scope = block_scope(blocks);
  ExecMaskAnalysis exec{KernelBlockScope(scope), 64};
  EXPECT_EQ(exec.before(*entry->instructions().begin()), ExecState::Unknown);
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

  const ExecMaskAnalysis exec(KernelBlockScope(scope), /*wave_size=*/64);
  LivenessAnalysis liveness(KernelBlockScope(scope), std::make_unique<ExecMaskAnalysis>(exec),
                            options);

  const Instruction &use = *blocks[0]->instructions().begin();
  EXPECT_FALSE(liveness.is_live_before(use, {RegClass::VGPR, 0, 4}));
  EXPECT_EQ(liveness.find_free_sgpr(&use, 0), 0);
  EXPECT_EQ(liveness.find_free_run(&use, 1, 0), 4);
  EXPECT_EQ(liveness.find_free_run(&use, 1, 7), 7);
}

TEST(LivenessAnalysis, GloballyUnusedRunHonorsMinFreeVgprFloor) {
  auto blocks = build_test_blocks({TestOpcode::UseSgpr4, TestOpcode::End});
  auto scope = block_scope(blocks);
  LivenessAnalysisOptions options;
  options.min_free_vgpr = 4;
  const ExecMaskAnalysis exec(KernelBlockScope(scope), /*wave_size=*/64);
  LivenessAnalysis liveness(KernelBlockScope(scope), std::make_unique<ExecMaskAnalysis>(exec),
                            options);

  const Instruction &use = *blocks[0]->instructions().begin();
  EXPECT_EQ(liveness.find_globally_unused_vgpr_run(&use, 1, 0, 1, 8), 4);
  EXPECT_EQ(liveness.find_globally_unused_vgpr_run(&use, 1, 0, 1, 4), std::nullopt);
  EXPECT_FALSE(liveness.has_materialized_cfg_liveness());
}

TEST(LivenessAnalysis, FindsGloballyUnusedRunBeforeSiteDeadFallback) {
  auto blocks = build_test_blocks({TestOpcode::UseVgpr0, TestOpcode::Nop, TestOpcode::End});
  auto scope = block_scope(blocks);
  const ExecMaskAnalysis exec{KernelBlockScope(scope), /*wave_size=*/64};
  const LivenessAnalysis liveness{KernelBlockScope(scope),
                                  std::make_unique<ExecMaskAnalysis>(exec)};

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
  const ExecMaskAnalysis exec(KernelBlockScope(scope), /*wave_size=*/64);

  LivenessAnalysisOptions limited_options;
  limited_options.min_free_vgpr = 256;
  LivenessAnalysis limited(KernelBlockScope(scope), std::make_unique<ExecMaskAnalysis>(exec),
                           limited_options);
  EXPECT_EQ(limited.find_free_run(&use, 1), std::nullopt);

  LivenessAnalysisOptions gfx1250_options;
  gfx1250_options.min_free_vgpr = 256;
  gfx1250_options.max_free_vgpr = 1024;
  LivenessAnalysis gfx1250(KernelBlockScope(scope), std::make_unique<ExecMaskAnalysis>(exec),
                           gfx1250_options);
  EXPECT_EQ(gfx1250.find_free_run(&use, 1), 256);
}

TEST(LivenessAnalysis, FreeVgprAllocationRejectsUnnameableRunWidth) {
  // A candidate run is tested as one RegisterRef, whose width is a uint8_t.
  // These queries take the width as a uint16_t, so a wider run is expressible
  // and must fail closed: truncating 256 to 0 would test only the run's first
  // lane and hand back a base whose other 255 lanes were never checked.
  auto blocks = build_test_blocks({TestOpcode::UseVgpr0, TestOpcode::End});
  auto scope = block_scope(blocks);
  const Instruction &use = *blocks[0]->instructions().begin();
  const ExecMaskAnalysis exec(KernelBlockScope(scope), /*wave_size=*/64);

  LivenessAnalysisOptions options;
  options.max_free_vgpr = 1024;
  LivenessAnalysis liveness(KernelBlockScope(scope), std::make_unique<ExecMaskAnalysis>(exec),
                            options);

  // v0 is live and globally used, so the widest nameable run starts at v1. The
  // 255/256 pair is what separates the width gate from an ordinary no-fit.
  EXPECT_EQ(liveness.find_free_run(&use, 255), 1);
  EXPECT_EQ(liveness.find_free_run(&use, 256), std::nullopt);
  EXPECT_EQ(liveness.find_globally_unused_vgpr_run(&use, 255, 0, 1, 1024), 1);
  EXPECT_EQ(liveness.find_globally_unused_vgpr_run(&use, 256, 0, 1, 1024), std::nullopt);
}

TEST(LivenessAnalysis, FindFreeRunHonorsBaseAlignment) {
  auto blocks = build_test_blocks({TestOpcode::UseSgpr4, TestOpcode::End});
  auto scope = block_scope(blocks);

  LivenessAnalysisOptions options;
  options.min_free_vgpr = 93;

  const ExecMaskAnalysis exec(KernelBlockScope(scope), /*wave_size=*/64);
  LivenessAnalysis liveness(KernelBlockScope(scope), std::make_unique<ExecMaskAnalysis>(exec),
                            options);

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
  const ExecMaskAnalysis kernel_exec(KernelBlockScope(kernel_blocks), /*wave_size=*/64);
  LivenessAnalysis kernel_liveness{KernelBlockScope(kernel_blocks),
                                   std::make_unique<ExecMaskAnalysis>(kernel_exec)};
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
                                  std::nullopt, {}, std::nullopt, std::nullopt,
                                  /*def_size_bits=*/16);
  InstDefUse idu(test_inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::SGPR, 4, 1}));
  EXPECT_TRUE(idu.uses.contains({RegClass::SGPR, 4, 1}));
}

TEST(InstDefUse, FullWidthDefIsNotUse) {
  const TestInstruction test_inst("test_def_s4", {{RegClass::SGPR, 4, 1}}, {}, 0, std::nullopt, {},
                                  std::nullopt, std::nullopt, /*def_size_bits=*/32);
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
  return std::unique_ptr<Instruction>(decoder ? decode_valid(*decoder, words.data()) : nullptr);
}

std::unique_ptr<Instruction> decode_gfx1250(const std::array<uint32_t, 2> &words) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  return std::unique_ptr<Instruction>(decoder ? decode_valid(*decoder, words.data()) : nullptr);
}

std::unique_ptr<Instruction> decode_gfx1250_compound(const std::array<uint32_t, 4> &words) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  return std::unique_ptr<Instruction>(decoder ? decode_valid(*decoder, words.data()) : nullptr);
}

std::array<uint32_t, 4> gfx1250_scale16_words(uint16_t scale_src0, uint16_t scale_src1) {
  constexpr uint16_t kVgprEncoding = 256;
  const auto prefix =
      cdna5::build_vop3p(0x3a, {.src0 = scale_src0, .src1 = scale_src1, .src2 = 256});
  const auto matrix = cdna5::build_vop3p(
      cdna5::kVWmmaF3216x16x128F8f6f4Vop3p,
      {.vdst = 96, .src0 = kVgprEncoding, .src1 = kVgprEncoding + 32, .src2 = 256 + 96});
  return {prefix[0], prefix[1], matrix[0], matrix[1]};
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

TEST(GeneratedInstDefUse, Gfx1250Scale16ScalarSourcesUseSingleSgprs) {
  auto inst = decode_gfx1250_compound(gfx1250_scale16_words(0, 2));
  ASSERT_NE(inst, nullptr);

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.uses.contains({RegClass::SGPR, 0, 1}));
  EXPECT_TRUE(idu.uses.contains({RegClass::SGPR, 2, 1}));
  EXPECT_FALSE(idu.uses.contains({RegClass::SGPR, 1, 1}));
  EXPECT_FALSE(idu.uses.contains({RegClass::SGPR, 3, 1}));
}

TEST(GeneratedInstDefUse, Gfx1250Scale16VectorSourcesUseVgprPairs) {
  auto inst = decode_gfx1250_compound(gfx1250_scale16_words(256 + 64, 256 + 66));
  ASSERT_NE(inst, nullptr);

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.uses.contains({RegClass::VGPR, 64, 2}));
  EXPECT_TRUE(idu.uses.contains({RegClass::VGPR, 66, 2}));
}

TEST(GeneratedInstDefUse, Gfx1250Scale16InlineSourcesUseNoRegisters) {
  auto inst = decode_gfx1250_compound(gfx1250_scale16_words(128, 128));
  ASSERT_NE(inst, nullptr);

  InstDefUse idu(*inst);
  EXPECT_FALSE(idu.uses.contains({RegClass::SGPR, 0, 1}));
  EXPECT_FALSE(idu.uses.contains({RegClass::VGPR, 128, 1}));
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
  // v_mov_b64_e32 writes a VGPR pair (v[6:7]). A partial DPP row mask preserves
  // the whole 64-bit destination, so the implicit use must match the width-2
  // def -- not just the low dword.
  // CDNA4 VOP1 word0: encoding[31:25]=0x3F, vdst[24:17]=6, op[16:9]=56
  // (v_mov_b64), src0[8:0]=250 (SRC_DPP).
  constexpr uint32_t kVop1MovB64Word0Dpp = (0x3Fu << 25) | (6u << 17) | (56u << 9) | 250u;
  constexpr uint32_t kDppCtrlRowShare0 = 0x150u << 8;
  auto inst = decode_cdna4({kVop1MovB64Word0Dpp, (0x7u << 28) | (0xFu << 24) | kDppCtrlRowShare0});
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()).substr(0, 9), "v_mov_b64");

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
// both a VGPR-dest op and an SGPR-dest compare. CDNA VOP3 encodings do not
// support this DPP form, so these decode for RDNA4.
//
// RDNA4 VOP3 word0: encoding[31:26]=53, op[25:16], clamp[15], opsel[14:11],
// abs[10:8], vdst[7:0]. word1: src0[8:0]=marker (250=SRC_DPP), src1[17:9]. The
// DPP16 word2 layout matches VOP1/VOP2, so its bit fields are reused.
constexpr uint32_t kVop3Enc = 53u << 26;
constexpr uint32_t kVop3AddF32Op = 259u << 16;  // v_add_f32_e64 (VGPR vdst)
constexpr uint32_t kVop3CmpLtF32Op = 17u << 16; // v_cmp_lt_f32_e64 (SGPR vdst)
// word1: src0=SRC_DPP, src1=VGPR3.
constexpr uint32_t kVop3DppWord1 = (3u << 9) | 250u;
constexpr uint32_t kVop3DppFi = 1u << 18;

// VOP3 DPP16 is 3 dwords and a FLAT (D16) load can decode as a 3-dword
// instruction, so the buffer is zero-padded to avoid out-of-bounds reads during
// decode when a caller supplies fewer words than the decoded length.
std::unique_ptr<Instruction> decode_rdna4(std::initializer_list<uint32_t> words) {
  std::array<uint32_t, 4> buf{};
  std::copy(words.begin(), words.end(), buf.begin());
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  return std::unique_ptr<Instruction>(decoder ? decode_valid(*decoder, buf.data()) : nullptr);
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
  // Full masks, FI=1, dpp_ctrl=0 (quad_perm, never OOB) -> every lane written.
  auto inst = decode_rdna4({kVop3Enc | kVop3AddF32Op | 5u, kVop3DppWord1,
                            (0xFu << 28) | (0xFu << 24) | kVop3DppFi | 2u});
  ASSERT_NE(inst, nullptr);

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::VGPR, 5, 1}));
  EXPECT_FALSE(idu.uses.contains({RegClass::VGPR, 5, 1}));
}

TEST(GeneratedInstDefUse, Vop3DppInactiveSourceReadsVgprDestinationWithFullMasks) {
  // Full destination masks and an in-range quad permutation isolate FI=0:
  // BOUND_CTRL=0 can suppress writes selected from inactive source lanes.
  auto inst = decode_rdna4({kVop3Enc | kVop3AddF32Op | 5u, kVop3DppWord1, kDppFullMasks | 2u});
  ASSERT_NE(inst, nullptr);

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::VGPR, 5, 1}));
  EXPECT_TRUE(idu.uses.contains({RegClass::VGPR, 5, 1}));
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
  // (s[8:9]). Row/bank-masked compare bits are cleared, so a partial destination
  // mask alone does not read the old SGPR result.
  auto inst = decode_rdna4({kVop3Enc | kVop3CmpLtF32Op | 8u, kVop3DppWord1,
                            (0x7u << 28) | (0xFu << 24) | kVop3DppFi | 2u});
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()).substr(0, 9), "v_cmp_lt_");

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::SGPR, 8, 2}));
  EXPECT_FALSE(idu.uses.contains({RegClass::SGPR, 8, 2}));
  EXPECT_FALSE(idu.uses.contains({RegClass::VGPR, 8, 1}));
}

TEST(GeneratedInstDefUse, Vop3CmpDppInvalidSourceDoesNotReadScalarDestination) {
  // With FI=1 and BOUND_CTRL=0, row_shr:1 has OOB row-edge sources. Active
  // compare destinations at those edges receive zero rather than preserving
  // their old SGPR result bits.
  auto inst = decode_rdna4({kVop3Enc | kVop3CmpLtF32Op | 8u, kVop3DppWord1,
                            kDppFullMasks | kVop3DppFi | kDppCtrlRowShr1 | 2u});
  ASSERT_NE(inst, nullptr);

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::SGPR, 8, 2}));
  EXPECT_FALSE(idu.uses.contains({RegClass::SGPR, 8, 2}));
  EXPECT_FALSE(idu.uses.contains({RegClass::VGPR, 8, 1}));
}

TEST(GeneratedInstDefUse, Vop3CmpDppInactiveSourceDoesNotReadScalarDestination) {
  // As above, full masks and an in-range permutation isolate FI=0 as the
  // reason an invalid-source result bit is forced to zero.
  auto inst = decode_rdna4({kVop3Enc | kVop3CmpLtF32Op | 8u, kVop3DppWord1, kDppFullMasks | 2u});
  ASSERT_NE(inst, nullptr);

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::SGPR, 8, 2}));
  EXPECT_FALSE(idu.uses.contains({RegClass::SGPR, 8, 2}));
  EXPECT_FALSE(idu.uses.contains({RegClass::VGPR, 8, 1}));
}

TEST(GeneratedInstDefUse, Vop3pDppPartialRowMaskReadsDestination) {
  // v_fma_mix_f32 (VOP3P, VGPR vdst=6) is a legal DPP VOP3P operation. A
  // partial row mask preserves the VGPR dst.
  // RDNA4 VOP3P word0: encoding[31:24]=204, op[22:16]=32 (v_fma_mix_f32),
  // vdst[7:0]=6. word1: src0[8:0]=250 (SRC_DPP), src1[17:9]=3 (VGPR3).
  // Modern DPP requires low-half selection for all low results and high-half
  // selection for all high results. The third high selector occupies word0 bit
  // 14; the first two occupy word1 bits 27:28.
  constexpr uint32_t kVop3pFmaMixWord0 = (204u << 24) | (32u << 16) | (1u << 14) | 6u;
  constexpr uint32_t kVop3pDppWord1 = (3u << 27) | (3u << 9) | 250u;
  auto inst = decode_rdna4({kVop3pFmaMixWord0, kVop3pDppWord1, (0x7u << 28) | (0xFu << 24) | 2u});
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()).substr(0, 13), "v_fma_mix_f32");

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::VGPR, 6, 1}));
  EXPECT_TRUE(idu.uses.contains({RegClass::VGPR, 6, 1}));
}

TEST(GeneratedInstDefUse, Vop3SdstEncDppPartialRowMaskReadsOnlyVgprResult) {
  // v_add_co_ci_u32_e64 (VOP3_SDST_ENC) writes TWO destinations: a VGPR result
  // (v6) and an SGPR carry-out (s[8:9]). Row/bank masking preserves only the
  // VGPR result. BC=1 eliminates invalid-source write suppression, so the SGPR
  // carry remains a pure def.
  // RDNA4 VOP3_SDST_ENC word0: encoding[31:26]=53, op[25:16]=288, sdst[14:8]=8,
  // vdst[7:0]=6. word1: src0=250 (SRC_DPP), src1[17:9]=3, src2[26:18]=10 (carry).
  constexpr uint32_t kVop3SdstWord0 = (53u << 26) | (288u << 16) | (8u << 8) | 6u;
  constexpr uint32_t kVop3SdstWord1 = (10u << 18) | (3u << 9) | 250u;
  auto inst =
      decode_rdna4({kVop3SdstWord0, kVop3SdstWord1, (0x7u << 28) | (0xFu << 24) | (1u << 19) | 2u});
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()).substr(0, 14), "v_add_co_ci_u3");

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::VGPR, 6, 1}));
  EXPECT_TRUE(idu.defs.contains({RegClass::SGPR, 8, 2}));
  EXPECT_TRUE(idu.uses.contains({RegClass::VGPR, 6, 1}));
  EXPECT_FALSE(idu.uses.contains({RegClass::SGPR, 8, 2}));
}

// v_writelane_b32 writes one lane and preserves the rest of vdst, so it reads
// the old value. On CDNA4/gfx1250 (XML marks vdst output-only) that read is
// surfaced via implicit_uses. word0 = VOP3 writelane opcode, vdst=5; word1 =
// src0=s4, src1=s2 (2 << 9).
TEST(GeneratedInstDefUse, WritelaneReadsDestinationCdna4) {
  auto inst = decode_cdna4({0xD28A0005U, 0x00000404U}); // v_writelane_b32 v5, s4, s2
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()), "v_writelane_b32");

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::VGPR, 5, 1}));
  EXPECT_TRUE(idu.uses.contains({RegClass::VGPR, 5, 1}));
}

// --- D16 memory loads: sub-dword loads preserve half of vdst (real decode) ---
//
// A D16(_HI) load writes one 16-bit half of the destination VGPR and preserves
// the other, so it reads the old vdst -- a read-modify-write. The generator
// emits an implicit_uses override for these (see _d16_load_reads_dst), so the
// decoded instruction reports vdst in both defs and uses while keeping it out
// of the printed operand list. A regular (non-D16) load fully overwrites vdst
// and must not report it as a use.
//
// Encodings are the canonical forms from rdna4/test_encodings.h with word1's
// low byte set to vdst=5 (FLAT VDST is word1[7:0]); vaddr stays v0 (word2=0),
// so v5 is distinct from the address source.
TEST(GeneratedInstDefUse, D16HiLoadReadsDestination) {
  auto inst = decode_rdna4({0xEC084000U, 0x00000005U}); // flat_load_d16_hi_u8, vdst=5
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()), "flat_load_d16_hi_u8");

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::VGPR, 5, 1}));
  EXPECT_TRUE(idu.uses.contains({RegClass::VGPR, 5, 1}));
}

TEST(GeneratedInstDefUse, WritelaneReadsDestinationGfx1250) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::array<uint32_t, 4> words{0xD7610005U, 0x00000404U, 0U, 0U};
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()), "v_writelane_b32");

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::VGPR, 5, 1}));
  EXPECT_TRUE(idu.uses.contains({RegClass::VGPR, 5, 1}));
}

TEST(GeneratedInstDefUse, D16LoLoadReadsDestination) {
  auto inst = decode_rdna4({0xEC078000U, 0x00000005U}); // flat_load_d16_u8, vdst=5
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()), "flat_load_d16_u8");

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::VGPR, 5, 1}));
  EXPECT_TRUE(idu.uses.contains({RegClass::VGPR, 5, 1}));
}

TEST(GeneratedInstDefUse, RegularLoadDoesNotReadDestination) {
  auto inst = decode_rdna4({0xEC050000U, 0x00000005U}); // flat_load_b32, vdst=5 (full write)
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()), "flat_load_b32");

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::VGPR, 5, 1}));
  EXPECT_FALSE(idu.uses.contains({RegClass::VGPR, 5, 1}));
}

// The remaining non-FLAT D16 load classes exercise the other paths of
// _d16_load_reads_dst. MUBUF (buffer) names the destination 'vdata' at
// word1[7:0]; DS names it 'vdst' at word1[31:24]. vaddr/addr stay 0 (v0), so v5
// is distinct from the address source.

TEST(GeneratedInstDefUse, D16BufferLoadReadsDestination) {
  auto inst = decode_rdna4({0xC4078000U, 0x00000005U}); // buffer_load_d16_u8, vdata=5
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()), "buffer_load_d16_u8");

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::VGPR, 5, 1}));
  EXPECT_TRUE(idu.uses.contains({RegClass::VGPR, 5, 1}));
}

// Odd-count FORMAT D16 load: xyz packs 3 halfwords into two VGPRs. The first
// (v5) is fully written (x|y); only the last (v6) holds one 16-bit half (z) and
// preserves its upper 16 bits, so only v6 is read.
TEST(GeneratedInstDefUse, D16FormatXyzLoadReadsOnlyLastDestination) {
  auto inst = decode_rdna4({0xC4028000U, 0x00000005U}); // buffer_load_d16_format_xyz, vdata=5
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()), "buffer_load_d16_format_xyz");

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::VGPR, 5, 2}));  // writes v5:v6
  EXPECT_TRUE(idu.uses.contains({RegClass::VGPR, 6, 1}));  // last reg partial
  EXPECT_FALSE(idu.uses.contains({RegClass::VGPR, 5, 1})); // first reg fully written
}

// Even-count FORMAT D16 load: xyzw fills two whole VGPRs and preserves nothing.
TEST(GeneratedInstDefUse, D16FormatXyzwLoadDoesNotReadDestination) {
  auto inst = decode_rdna4({0xC402C000U, 0x00000005U}); // buffer_load_d16_format_xyzw, vdata=5
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()), "buffer_load_d16_format_xyzw");

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::VGPR, 5, 2}));
  EXPECT_FALSE(idu.uses.contains({RegClass::VGPR, 5, 2}));
  // contains() over a range requires every lane, so assert each register
  // individually to catch a regression that reads only v5 or only v6.
  EXPECT_FALSE(idu.uses.contains({RegClass::VGPR, 5, 1}));
  EXPECT_FALSE(idu.uses.contains({RegClass::VGPR, 6, 1}));
}

TEST(GeneratedInstDefUse, D16DsLoadReadsDestination) {
  auto inst = decode_rdna4({0xDA880000U, 0x05000000U}); // ds_load_u8_d16, vdst=5
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()), "ds_load_u8_d16");

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::VGPR, 5, 1}));
  EXPECT_TRUE(idu.uses.contains({RegClass::VGPR, 5, 1}));
}

// tbuffer (MTBUF) is a separate encoding only on ISAs before the unified
// VBUFFER (RDNA4 folds MUBUF/MTBUF into VBUFFER and routes typed-buffer ops
// through the untyped path, leaving them unclassified). Exercise the
// 'tbuffer_load' path on CDNA3, where MTBUF is distinct and its 4-bit opcode
// distinguishes the D16 variant (RDNA2's 3-bit opcode cannot, aliasing D16 back
// to the non-D16 form). Its dest 'vdata' is at word1[8:15]; vaddr at word1[7:0]
// stays 0 (v0).
std::unique_ptr<Instruction> decode_cdna3(std::initializer_list<uint32_t> words) {
  std::array<uint32_t, 4> buf{};
  std::copy(words.begin(), words.end(), buf.begin());
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  return std::unique_ptr<Instruction>(decoder ? decode_valid(*decoder, buf.data()) : nullptr);
}

std::unique_ptr<Instruction> decode_cdna1(std::initializer_list<uint32_t> words) {
  std::array<uint32_t, 4> buf{};
  std::copy(words.begin(), words.end(), buf.begin());
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA1);
  return std::unique_ptr<Instruction>(decoder ? decode_valid(*decoder, buf.data()) : nullptr);
}

TEST(GeneratedInstDefUse, D16TbufferLoadDoesNotReadDestinationWithSramEcc) {
  auto inst = decode_cdna3({0xE8040000U, 0x00000500U}); // tbuffer_load_format_d16_x, vdata=5
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()), "tbuffer_load_format_d16_x");

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::VGPR, 5, 1}));
  EXPECT_FALSE(idu.uses.contains({RegClass::VGPR, 5, 1}));
}

// Negative case: D16 stores share the d16 flags but are not in
// _D16_LOAD_CLASSES, so no implicit_uses override is emitted. The data operand
// (FLAT 'vsrc' at word1[23:30], set to v5 = 5 << 23) is a plain source: it must
// be a use, never a def. vaddr at word2[7:0] stays v0.
TEST(GeneratedInstDefUse, D16StoreDoesNotDefineData) {
  auto inst = decode_rdna4({0xEC094000U, 0x02800000U}); // flat_store_d16_hi_b16, vsrc=5
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()), "flat_store_d16_hi_b16");

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.uses.contains({RegClass::VGPR, 5, 1}));
  EXPECT_FALSE(idu.defs.contains({RegClass::VGPR, 5, 1}));
}

// RDNA4 folds typed buffers into ENC_VBUFFER, routed through the untyped
// derivation. That path now recognizes TBUFFER_* as well, so a typed D16
// FORMAT_X load (op 136) reports its preserved destination like its untyped and
// pre-RDNA4 MTBUF counterparts. Before the fix it decoded as an unclassified nop
// with no implicit_uses override. VDATA is word1[0:7] (=5).
TEST(GeneratedInstDefUse, D16TypedFormatLoadUnderVbufferReadsDestination) {
  auto inst = decode_rdna4({0xC4220000U, 0x00000005U}); // tbuffer_load_d16_format_x, vdata=5
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()), "tbuffer_load_d16_format_x");

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::VGPR, 5, 1}));
  EXPECT_TRUE(idu.uses.contains({RegClass::VGPR, 5, 1}));
}

// Odd-count typed FORMAT under VBUFFER: xyz packs 3 halfwords into two VGPRs,
// so only the final register's (v6) upper half is preserved. Mirrors the
// untyped xyz case but through the typed path. VDATA is word1[0:7] (=5).
TEST(GeneratedInstDefUse, D16TypedFormatXyzLoadUnderVbufferReadsOnlyLastDestination) {
  auto inst = decode_rdna4({0xC4228000U, 0x00000005U}); // tbuffer_load_d16_format_xyz, vdata=5
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()), "tbuffer_load_d16_format_xyz");

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::VGPR, 5, 2}));  // writes v5:v6
  EXPECT_TRUE(idu.uses.contains({RegClass::VGPR, 6, 1}));  // last reg partial
  EXPECT_FALSE(idu.uses.contains({RegClass::VGPR, 5, 1})); // first reg fully written
}

// On older MUBUF encodings the LDS bit (word0 bit 16) redirects the loaded data
// to LDS, leaving no VGPR destination -- so the preserved-destination read must
// be suppressed there, or liveness invents a false live range. Same opcode
// (buffer_load_short_d16, MUBUF op 36 on CDNA1), toggling only LDS. VDATA is
// word1[8:15] (=5).
TEST(GeneratedInstDefUse, D16BufferLoadLdsBitSuppressesDestinationRead) {
  auto normal = decode_cdna1({0xE0900000U, 0x00000500U}); // buffer_load_short_d16, vdata=5, lds=0
  ASSERT_NE(normal, nullptr);
  ASSERT_EQ(std::string_view(normal->mnemonic()), "buffer_load_short_d16");
  InstDefUse normal_idu(*normal);
  EXPECT_TRUE(normal_idu.uses.contains({RegClass::VGPR, 5, 1}))
      << "LDS clear: the preserved half of vdata is a read";

  auto lds = decode_cdna1({0xE0910000U, 0x00000500U}); // ...same, lds=1 (bit 16 set)
  ASSERT_NE(lds, nullptr);
  ASSERT_EQ(std::string_view(lds->mnemonic()), "buffer_load_short_d16");
  InstDefUse lds_idu(*lds);
  EXPECT_FALSE(lds_idu.uses.contains({RegClass::VGPR, 5, 1}))
      << "LDS set: data goes to LDS, so vdata is not a preserved read";
}

TEST(GeneratedInstDefUse, Vop3SdstEncDppInvalidSourceCanReadScalarResult) {
  constexpr uint32_t kVop3SdstWord0 = (53u << 26) | (288u << 16) | (8u << 8) | 6u;
  constexpr uint32_t kVop3SdstWord1 = (10u << 18) | (3u << 9) | 250u;
  auto inst = decode_rdna4({kVop3SdstWord0, kVop3SdstWord1, (0xFu << 28) | (0xFu << 24) | 2u});
  ASSERT_NE(inst, nullptr);

  InstDefUse idu(*inst);
  EXPECT_TRUE(idu.defs.contains({RegClass::SGPR, 8, 2}));
  EXPECT_TRUE(idu.uses.contains({RegClass::SGPR, 8, 2}));
}

} // namespace
} // namespace rocjitsu

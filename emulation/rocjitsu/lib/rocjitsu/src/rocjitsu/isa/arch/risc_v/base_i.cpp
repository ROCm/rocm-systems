// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/risc_v/base_i.h"
#include "rocjitsu/vm/risc_v/hart_state.h"
#include "rocjitsu/vm/risc_v/memory.h"

namespace rocjitsu {
namespace risc_v {
namespace detail {
namespace {

int64_t arithmetic_right_shift(int64_t val, uint32_t shamt) {
  // Portable arithmetic right shift via unsigned shift + sign extension.
  if (shamt >= 64)
    return (val < 0) ? -1 : 0;
  uint64_t uval = static_cast<uint64_t>(val);
  uint64_t shifted = uval >> shamt;
  // Sign-extend if negative
  if (val < 0 && shamt > 0)
    shifted |= ~uint64_t{0} << (64 - shamt);
  return static_cast<int64_t>(shifted);
}

int32_t arithmetic_right_shift_32(int32_t val, uint32_t shamt) {
  if (shamt >= 32)
    return (val < 0) ? -1 : 0;
  uint32_t uval = static_cast<uint32_t>(val);
  uint32_t shifted = uval >> shamt;
  if (val < 0 && shamt > 0)
    shifted |= ~uint32_t{0} << (32 - shamt);
  return static_cast<int32_t>(shifted);
}

} // namespace

// U-type instructions

LuiInst::LuiInst(uint32_t raw)
    : UType("lui", raw, make_exec_fn<LuiInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      imm_op(32, OperandType::OPR_IMM, imm()) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &imm_op;
  num_src_ = 1;
  num_dst_ = 1;
}

void LuiInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(imm()));
}

AuipcInst::AuipcInst(uint32_t raw)
    : UType("auipc", raw, make_exec_fn<AuipcInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      imm_op(32, OperandType::OPR_IMM, imm()) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &imm_op;
  num_src_ = 1;
  num_dst_ = 1;
}

void AuipcInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(h->pc + static_cast<uint64_t>(imm())));
}

// J-type instructions

JalInst::JalInst(uint32_t raw)
    : JType("jal", raw, make_exec_fn<JalInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      offset(21, OperandType::OPR_IMM, imm()) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &offset;
  num_src_ = 1;
  num_dst_ = 1;
}

void JalInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(h->pc + 4));
  h->next_pc = h->pc + static_cast<uint64_t>(imm());
}

// I-type jump instructions

JalrInst::JalrInst(uint32_t raw)
    : IType("jalr", raw, make_exec_fn<JalrInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), imm_op(12, OperandType::OPR_IMM, imm()) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &imm_op;
  num_src_ = 2;
  num_dst_ = 1;
}

void JalrInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  int64_t base = h->read_xreg(rs1.encoding_value_);
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(h->pc + 4));
  h->next_pc = static_cast<uint64_t>((base + imm()) & ~int64_t{1});
}

// B-type branch instructions

BeqInst::BeqInst(uint32_t raw)
    : BType("beq", raw, make_exec_fn<BeqInst>()), rs1_op(64, OperandType::OPR_GPR, inst_.rs1),
      rs2_op(64, OperandType::OPR_GPR, inst_.rs2), offset(13, OperandType::OPR_IMM, imm()) {
  src_operands_[0] = &rs1_op;
  src_operands_[1] = &rs2_op;
  src_operands_[2] = &offset;
  num_src_ = 3;
  num_dst_ = 0;
}

void BeqInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  if (h->read_xreg(rs1_op.encoding_value_) == h->read_xreg(rs2_op.encoding_value_))
    h->next_pc = h->pc + static_cast<uint64_t>(imm());
}

BneInst::BneInst(uint32_t raw)
    : BType("bne", raw, make_exec_fn<BneInst>()), rs1_op(64, OperandType::OPR_GPR, inst_.rs1),
      rs2_op(64, OperandType::OPR_GPR, inst_.rs2), offset(13, OperandType::OPR_IMM, imm()) {
  src_operands_[0] = &rs1_op;
  src_operands_[1] = &rs2_op;
  src_operands_[2] = &offset;
  num_src_ = 3;
  num_dst_ = 0;
}

void BneInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  if (h->read_xreg(rs1_op.encoding_value_) != h->read_xreg(rs2_op.encoding_value_))
    h->next_pc = h->pc + static_cast<uint64_t>(imm());
}

BltInst::BltInst(uint32_t raw)
    : BType("blt", raw, make_exec_fn<BltInst>()), rs1_op(64, OperandType::OPR_GPR, inst_.rs1),
      rs2_op(64, OperandType::OPR_GPR, inst_.rs2), offset(13, OperandType::OPR_IMM, imm()) {
  src_operands_[0] = &rs1_op;
  src_operands_[1] = &rs2_op;
  src_operands_[2] = &offset;
  num_src_ = 3;
  num_dst_ = 0;
}

void BltInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  if (h->read_xreg(rs1_op.encoding_value_) < h->read_xreg(rs2_op.encoding_value_))
    h->next_pc = h->pc + static_cast<uint64_t>(imm());
}

BgeInst::BgeInst(uint32_t raw)
    : BType("bge", raw, make_exec_fn<BgeInst>()), rs1_op(64, OperandType::OPR_GPR, inst_.rs1),
      rs2_op(64, OperandType::OPR_GPR, inst_.rs2), offset(13, OperandType::OPR_IMM, imm()) {
  src_operands_[0] = &rs1_op;
  src_operands_[1] = &rs2_op;
  src_operands_[2] = &offset;
  num_src_ = 3;
  num_dst_ = 0;
}

void BgeInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  if (h->read_xreg(rs1_op.encoding_value_) >= h->read_xreg(rs2_op.encoding_value_))
    h->next_pc = h->pc + static_cast<uint64_t>(imm());
}

BltuInst::BltuInst(uint32_t raw)
    : BType("bltu", raw, make_exec_fn<BltuInst>()), rs1_op(64, OperandType::OPR_GPR, inst_.rs1),
      rs2_op(64, OperandType::OPR_GPR, inst_.rs2), offset(13, OperandType::OPR_IMM, imm()) {
  src_operands_[0] = &rs1_op;
  src_operands_[1] = &rs2_op;
  src_operands_[2] = &offset;
  num_src_ = 3;
  num_dst_ = 0;
}

void BltuInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  if (static_cast<uint64_t>(h->read_xreg(rs1_op.encoding_value_)) <
      static_cast<uint64_t>(h->read_xreg(rs2_op.encoding_value_)))
    h->next_pc = h->pc + static_cast<uint64_t>(imm());
}

BgeuInst::BgeuInst(uint32_t raw)
    : BType("bgeu", raw, make_exec_fn<BgeuInst>()), rs1_op(64, OperandType::OPR_GPR, inst_.rs1),
      rs2_op(64, OperandType::OPR_GPR, inst_.rs2), offset(13, OperandType::OPR_IMM, imm()) {
  src_operands_[0] = &rs1_op;
  src_operands_[1] = &rs2_op;
  src_operands_[2] = &offset;
  num_src_ = 3;
  num_dst_ = 0;
}

void BgeuInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  if (static_cast<uint64_t>(h->read_xreg(rs1_op.encoding_value_)) >=
      static_cast<uint64_t>(h->read_xreg(rs2_op.encoding_value_)))
    h->next_pc = h->pc + static_cast<uint64_t>(imm());
}

// I-type load instructions

LbInst::LbInst(uint32_t raw)
    : IType("lb", raw, make_exec_fn<LbInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1_op(64, OperandType::OPR_GPR, inst_.rs1), offset(12, OperandType::OPR_IMM, imm()) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1_op;
  src_operands_[1] = &offset;
  num_src_ = 2;
  num_dst_ = 1;
}

void LbInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = static_cast<uint64_t>(h->read_xreg(rs1_op.encoding_value_) + imm());
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(static_cast<int8_t>(m->read8(addr))));
}

LhInst::LhInst(uint32_t raw)
    : IType("lh", raw, make_exec_fn<LhInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1_op(64, OperandType::OPR_GPR, inst_.rs1), offset(12, OperandType::OPR_IMM, imm()) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1_op;
  src_operands_[1] = &offset;
  num_src_ = 2;
  num_dst_ = 1;
}

void LhInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = static_cast<uint64_t>(h->read_xreg(rs1_op.encoding_value_) + imm());
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(static_cast<int16_t>(m->read16(addr))));
}

LwInst::LwInst(uint32_t raw)
    : IType("lw", raw, make_exec_fn<LwInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1_op(64, OperandType::OPR_GPR, inst_.rs1), offset(12, OperandType::OPR_IMM, imm()) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1_op;
  src_operands_[1] = &offset;
  num_src_ = 2;
  num_dst_ = 1;
}

void LwInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = static_cast<uint64_t>(h->read_xreg(rs1_op.encoding_value_) + imm());
  h->write_xreg(rd.encoding_value_, sext32(static_cast<int32_t>(m->read32(addr))));
}

LdInst::LdInst(uint32_t raw)
    : IType("ld", raw, make_exec_fn<LdInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1_op(64, OperandType::OPR_GPR, inst_.rs1), offset(12, OperandType::OPR_IMM, imm()) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1_op;
  src_operands_[1] = &offset;
  num_src_ = 2;
  num_dst_ = 1;
}

void LdInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = static_cast<uint64_t>(h->read_xreg(rs1_op.encoding_value_) + imm());
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(m->read64(addr)));
}

LbuInst::LbuInst(uint32_t raw)
    : IType("lbu", raw, make_exec_fn<LbuInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1_op(64, OperandType::OPR_GPR, inst_.rs1), offset(12, OperandType::OPR_IMM, imm()) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1_op;
  src_operands_[1] = &offset;
  num_src_ = 2;
  num_dst_ = 1;
}

void LbuInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = static_cast<uint64_t>(h->read_xreg(rs1_op.encoding_value_) + imm());
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(m->read8(addr)));
}

LhuInst::LhuInst(uint32_t raw)
    : IType("lhu", raw, make_exec_fn<LhuInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1_op(64, OperandType::OPR_GPR, inst_.rs1), offset(12, OperandType::OPR_IMM, imm()) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1_op;
  src_operands_[1] = &offset;
  num_src_ = 2;
  num_dst_ = 1;
}

void LhuInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = static_cast<uint64_t>(h->read_xreg(rs1_op.encoding_value_) + imm());
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(m->read16(addr)));
}

LwuInst::LwuInst(uint32_t raw)
    : IType("lwu", raw, make_exec_fn<LwuInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1_op(64, OperandType::OPR_GPR, inst_.rs1), offset(12, OperandType::OPR_IMM, imm()) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1_op;
  src_operands_[1] = &offset;
  num_src_ = 2;
  num_dst_ = 1;
}

void LwuInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = static_cast<uint64_t>(h->read_xreg(rs1_op.encoding_value_) + imm());
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(m->read32(addr)));
}

// S-type store instructions

SbInst::SbInst(uint32_t raw)
    : SType("sb", raw, make_exec_fn<SbInst>()), rs2_op(64, OperandType::OPR_GPR, inst_.rs2),
      rs1_op(64, OperandType::OPR_GPR, inst_.rs1), offset(12, OperandType::OPR_IMM, imm()) {
  src_operands_[0] = &rs2_op;
  src_operands_[1] = &offset;
  src_operands_[2] = &rs1_op;
  num_src_ = 3;
  num_dst_ = 0;
}

void SbInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = static_cast<uint64_t>(h->read_xreg(rs1_op.encoding_value_) + imm());
  m->write8(addr, static_cast<uint8_t>(h->read_xreg(rs2_op.encoding_value_)));
}

ShInst::ShInst(uint32_t raw)
    : SType("sh", raw, make_exec_fn<ShInst>()), rs2_op(64, OperandType::OPR_GPR, inst_.rs2),
      rs1_op(64, OperandType::OPR_GPR, inst_.rs1), offset(12, OperandType::OPR_IMM, imm()) {
  src_operands_[0] = &rs2_op;
  src_operands_[1] = &offset;
  src_operands_[2] = &rs1_op;
  num_src_ = 3;
  num_dst_ = 0;
}

void ShInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = static_cast<uint64_t>(h->read_xreg(rs1_op.encoding_value_) + imm());
  m->write16(addr, static_cast<uint16_t>(h->read_xreg(rs2_op.encoding_value_)));
}

SwInst::SwInst(uint32_t raw)
    : SType("sw", raw, make_exec_fn<SwInst>()), rs2_op(64, OperandType::OPR_GPR, inst_.rs2),
      rs1_op(64, OperandType::OPR_GPR, inst_.rs1), offset(12, OperandType::OPR_IMM, imm()) {
  src_operands_[0] = &rs2_op;
  src_operands_[1] = &offset;
  src_operands_[2] = &rs1_op;
  num_src_ = 3;
  num_dst_ = 0;
}

void SwInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = static_cast<uint64_t>(h->read_xreg(rs1_op.encoding_value_) + imm());
  m->write32(addr, static_cast<uint32_t>(h->read_xreg(rs2_op.encoding_value_)));
}

SdInst::SdInst(uint32_t raw)
    : SType("sd", raw, make_exec_fn<SdInst>()), rs2_op(64, OperandType::OPR_GPR, inst_.rs2),
      rs1_op(64, OperandType::OPR_GPR, inst_.rs1), offset(12, OperandType::OPR_IMM, imm()) {
  src_operands_[0] = &rs2_op;
  src_operands_[1] = &offset;
  src_operands_[2] = &rs1_op;
  num_src_ = 3;
  num_dst_ = 0;
}

void SdInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = static_cast<uint64_t>(h->read_xreg(rs1_op.encoding_value_) + imm());
  m->write64(addr, static_cast<uint64_t>(h->read_xreg(rs2_op.encoding_value_)));
}

// I-type ALU instructions

AddiInst::AddiInst(uint32_t raw)
    : IType("addi", raw, make_exec_fn<AddiInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), imm_op(12, OperandType::OPR_IMM, imm()) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &imm_op;
  num_src_ = 2;
  num_dst_ = 1;
}

void AddiInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  h->write_xreg(rd.encoding_value_, h->read_xreg(rs1.encoding_value_) + imm());
}

SltiInst::SltiInst(uint32_t raw)
    : IType("slti", raw, make_exec_fn<SltiInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), imm_op(12, OperandType::OPR_IMM, imm()) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &imm_op;
  num_src_ = 2;
  num_dst_ = 1;
}

void SltiInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  h->write_xreg(rd.encoding_value_, (h->read_xreg(rs1.encoding_value_) < imm()) ? 1 : 0);
}

SltiuInst::SltiuInst(uint32_t raw)
    : IType("sltiu", raw, make_exec_fn<SltiuInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), imm_op(12, OperandType::OPR_IMM, imm()) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &imm_op;
  num_src_ = 2;
  num_dst_ = 1;
}

void SltiuInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  h->write_xreg(rd.encoding_value_, (static_cast<uint64_t>(h->read_xreg(rs1.encoding_value_)) <
                                     static_cast<uint64_t>(static_cast<int64_t>(imm())))
                                        ? 1
                                        : 0);
}

XoriInst::XoriInst(uint32_t raw)
    : IType("xori", raw, make_exec_fn<XoriInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), imm_op(12, OperandType::OPR_IMM, imm()) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &imm_op;
  num_src_ = 2;
  num_dst_ = 1;
}

void XoriInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  h->write_xreg(rd.encoding_value_, h->read_xreg(rs1.encoding_value_) ^ imm());
}

OriInst::OriInst(uint32_t raw)
    : IType("ori", raw, make_exec_fn<OriInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), imm_op(12, OperandType::OPR_IMM, imm()) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &imm_op;
  num_src_ = 2;
  num_dst_ = 1;
}

void OriInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  h->write_xreg(rd.encoding_value_, h->read_xreg(rs1.encoding_value_) | imm());
}

AndiInst::AndiInst(uint32_t raw)
    : IType("andi", raw, make_exec_fn<AndiInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), imm_op(12, OperandType::OPR_IMM, imm()) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &imm_op;
  num_src_ = 2;
  num_dst_ = 1;
}

void AndiInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  h->write_xreg(rd.encoding_value_, h->read_xreg(rs1.encoding_value_) & imm());
}

// I-type shift instructions

SlliInst::SlliInst(uint32_t raw)
    : IType("slli", raw, make_exec_fn<SlliInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), imm_op(12, OperandType::OPR_IMM, imm()) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &imm_op;
  num_src_ = 2;
  num_dst_ = 1;
}

void SlliInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  int shamt = imm() & 0x3f;
  h->write_xreg(rd.encoding_value_, h->read_xreg(rs1.encoding_value_) << shamt);
}

SrliInst::SrliInst(uint32_t raw)
    : IType("srli", raw, make_exec_fn<SrliInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), imm_op(12, OperandType::OPR_IMM, imm()) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &imm_op;
  num_src_ = 2;
  num_dst_ = 1;
}

void SrliInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  int shamt = imm() & 0x3f;
  h->write_xreg(
      rd.encoding_value_,
      static_cast<int64_t>(static_cast<uint64_t>(h->read_xreg(rs1.encoding_value_)) >> shamt));
}

SraiInst::SraiInst(uint32_t raw)
    : IType("srai", raw, make_exec_fn<SraiInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), imm_op(12, OperandType::OPR_IMM, imm()) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &imm_op;
  num_src_ = 2;
  num_dst_ = 1;
}

void SraiInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  int shamt = imm() & 0x3f;
  h->write_xreg(rd.encoding_value_,
                arithmetic_right_shift(h->read_xreg(rs1.encoding_value_), shamt));
}

// R-type ALU instructions

AddInst::AddInst(uint32_t raw)
    : RType("add", raw, make_exec_fn<AddInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}

void AddInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  h->write_xreg(rd.encoding_value_,
                h->read_xreg(rs1.encoding_value_) + h->read_xreg(rs2.encoding_value_));
}

SubInst::SubInst(uint32_t raw)
    : RType("sub", raw, make_exec_fn<SubInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}

void SubInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  h->write_xreg(rd.encoding_value_,
                h->read_xreg(rs1.encoding_value_) - h->read_xreg(rs2.encoding_value_));
}

SllInst::SllInst(uint32_t raw)
    : RType("sll", raw, make_exec_fn<SllInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}

void SllInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  int shamt = h->read_xreg(rs2.encoding_value_) & 0x3f;
  h->write_xreg(rd.encoding_value_, h->read_xreg(rs1.encoding_value_) << shamt);
}

SltInst::SltInst(uint32_t raw)
    : RType("slt", raw, make_exec_fn<SltInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}

void SltInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  h->write_xreg(rd.encoding_value_,
                (h->read_xreg(rs1.encoding_value_) < h->read_xreg(rs2.encoding_value_)) ? 1 : 0);
}

SltuInst::SltuInst(uint32_t raw)
    : RType("sltu", raw, make_exec_fn<SltuInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}

void SltuInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  h->write_xreg(rd.encoding_value_, (static_cast<uint64_t>(h->read_xreg(rs1.encoding_value_)) <
                                     static_cast<uint64_t>(h->read_xreg(rs2.encoding_value_)))
                                        ? 1
                                        : 0);
}

XorInst::XorInst(uint32_t raw)
    : RType("xor", raw, make_exec_fn<XorInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}

void XorInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  h->write_xreg(rd.encoding_value_,
                h->read_xreg(rs1.encoding_value_) ^ h->read_xreg(rs2.encoding_value_));
}

SrlInst::SrlInst(uint32_t raw)
    : RType("srl", raw, make_exec_fn<SrlInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}

void SrlInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  int shamt = h->read_xreg(rs2.encoding_value_) & 0x3f;
  h->write_xreg(
      rd.encoding_value_,
      static_cast<int64_t>(static_cast<uint64_t>(h->read_xreg(rs1.encoding_value_)) >> shamt));
}

SraInst::SraInst(uint32_t raw)
    : RType("sra", raw, make_exec_fn<SraInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}

void SraInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  int shamt = h->read_xreg(rs2.encoding_value_) & 0x3f;
  h->write_xreg(rd.encoding_value_,
                arithmetic_right_shift(h->read_xreg(rs1.encoding_value_), shamt));
}

OrInst::OrInst(uint32_t raw)
    : RType("or", raw, make_exec_fn<OrInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}

void OrInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  h->write_xreg(rd.encoding_value_,
                h->read_xreg(rs1.encoding_value_) | h->read_xreg(rs2.encoding_value_));
}

AndInst::AndInst(uint32_t raw)
    : RType("and", raw, make_exec_fn<AndInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}

void AndInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  h->write_xreg(rd.encoding_value_,
                h->read_xreg(rs1.encoding_value_) & h->read_xreg(rs2.encoding_value_));
}

// I-type W-ALU instructions (RV64I)

AddiwInst::AddiwInst(uint32_t raw)
    : IType("addiw", raw, make_exec_fn<AddiwInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), imm_op(12, OperandType::OPR_IMM, imm()) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &imm_op;
  num_src_ = 2;
  num_dst_ = 1;
}

void AddiwInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  int32_t result = static_cast<int32_t>(h->read_xreg(rs1.encoding_value_)) + imm();
  h->write_xreg(rd.encoding_value_, sext32(result));
}

SlliwInst::SlliwInst(uint32_t raw)
    : IType("slliw", raw, make_exec_fn<SlliwInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), imm_op(12, OperandType::OPR_IMM, imm()) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &imm_op;
  num_src_ = 2;
  num_dst_ = 1;
}

void SlliwInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  int shamt = imm() & 0x1f;
  int32_t result = static_cast<int32_t>(h->read_xreg(rs1.encoding_value_)) << shamt;
  h->write_xreg(rd.encoding_value_, sext32(result));
}

SrliwInst::SrliwInst(uint32_t raw)
    : IType("srliw", raw, make_exec_fn<SrliwInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), imm_op(12, OperandType::OPR_IMM, imm()) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &imm_op;
  num_src_ = 2;
  num_dst_ = 1;
}

void SrliwInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  int shamt = imm() & 0x1f;
  uint32_t val = static_cast<uint32_t>(h->read_xreg(rs1.encoding_value_));
  h->write_xreg(rd.encoding_value_, sext32(static_cast<int32_t>(val >> shamt)));
}

SraiwInst::SraiwInst(uint32_t raw)
    : IType("sraiw", raw, make_exec_fn<SraiwInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), imm_op(12, OperandType::OPR_IMM, imm()) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &imm_op;
  num_src_ = 2;
  num_dst_ = 1;
}

void SraiwInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  int shamt = imm() & 0x1f;
  int32_t val = static_cast<int32_t>(h->read_xreg(rs1.encoding_value_));
  h->write_xreg(rd.encoding_value_, sext32(arithmetic_right_shift_32(val, shamt)));
}

// R-type W-ALU instructions (RV64I)

AddwInst::AddwInst(uint32_t raw)
    : RType("addw", raw, make_exec_fn<AddwInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}

void AddwInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  int32_t result = static_cast<int32_t>(h->read_xreg(rs1.encoding_value_)) +
                   static_cast<int32_t>(h->read_xreg(rs2.encoding_value_));
  h->write_xreg(rd.encoding_value_, sext32(result));
}

SubwInst::SubwInst(uint32_t raw)
    : RType("subw", raw, make_exec_fn<SubwInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}

void SubwInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  int32_t result = static_cast<int32_t>(h->read_xreg(rs1.encoding_value_)) -
                   static_cast<int32_t>(h->read_xreg(rs2.encoding_value_));
  h->write_xreg(rd.encoding_value_, sext32(result));
}

SllwInst::SllwInst(uint32_t raw)
    : RType("sllw", raw, make_exec_fn<SllwInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}

void SllwInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  int shamt = h->read_xreg(rs2.encoding_value_) & 0x1f;
  int32_t result = static_cast<int32_t>(h->read_xreg(rs1.encoding_value_)) << shamt;
  h->write_xreg(rd.encoding_value_, sext32(result));
}

SrlwInst::SrlwInst(uint32_t raw)
    : RType("srlw", raw, make_exec_fn<SrlwInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}

void SrlwInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  int shamt = h->read_xreg(rs2.encoding_value_) & 0x1f;
  uint32_t val = static_cast<uint32_t>(h->read_xreg(rs1.encoding_value_));
  h->write_xreg(rd.encoding_value_, sext32(static_cast<int32_t>(val >> shamt)));
}

SrawInst::SrawInst(uint32_t raw)
    : RType("sraw", raw, make_exec_fn<SrawInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}

void SrawInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  int shamt = h->read_xreg(rs2.encoding_value_) & 0x1f;
  int32_t val = static_cast<int32_t>(h->read_xreg(rs1.encoding_value_));
  h->write_xreg(rd.encoding_value_, sext32(arithmetic_right_shift_32(val, shamt)));
}

// I-type system instructions

FenceInst::FenceInst(uint32_t raw)
    : IType("fence", raw, make_exec_fn<FenceInst>()), imm_op(12, OperandType::OPR_IMM, imm()) {
  src_operands_[0] = &imm_op;
  num_src_ = 1;
  num_dst_ = 0;
}

void FenceInst::execute_impl(HartState &ctx) {
  (void)ctx; // No-op for single-hart simulation.
}

// I-type environment instructions

EcallInst::EcallInst(uint32_t raw) : IType("ecall", raw, make_exec_fn<EcallInst>()) {}

void EcallInst::execute_impl(HartState &ctx) { as_hart(ctx)->halted = true; }

EbreakInst::EbreakInst(uint32_t raw) : IType("ebreak", raw, make_exec_fn<EbreakInst>()) {}

void EbreakInst::execute_impl(HartState &ctx) { as_hart(ctx)->halted = true; }

} // namespace detail
} // namespace risc_v
} // namespace rocjitsu

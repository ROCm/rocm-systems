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
    : UType("lui", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      imm_op(32, OperandType::OPR_IMM, imm()) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&imm_op);
}

void LuiInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(imm()));
}

AuipcInst::AuipcInst(uint32_t raw)
    : UType("auipc", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      imm_op(32, OperandType::OPR_IMM, imm()) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&imm_op);
}

void AuipcInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(h->pc + static_cast<uint64_t>(imm())));
}

// J-type instructions

JalInst::JalInst(uint32_t raw)
    : JType("jal", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      offset(21, OperandType::OPR_IMM, imm()) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&offset);
}

void JalInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(h->pc + 4));
  h->next_pc = h->pc + static_cast<uint64_t>(imm());
}

// I-type jump instructions

JalrInst::JalrInst(uint32_t raw)
    : IType("jalr", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), imm_op(12, OperandType::OPR_IMM, imm()) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&imm_op);
}

void JalrInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  int64_t base = h->read_xreg(rs1.encoding_value_);
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(h->pc + 4));
  h->next_pc = static_cast<uint64_t>((base + imm()) & ~int64_t{1});
}

// B-type branch instructions

BeqInst::BeqInst(uint32_t raw)
    : BType("beq", raw), rs1_op(64, OperandType::OPR_GPR, inst_.rs1),
      rs2_op(64, OperandType::OPR_GPR, inst_.rs2), offset(13, OperandType::OPR_IMM, imm()) {
  src_operands_.emplace_back(&rs1_op);
  src_operands_.emplace_back(&rs2_op);
  src_operands_.emplace_back(&offset);
}

void BeqInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  if (h->read_xreg(rs1_op.encoding_value_) == h->read_xreg(rs2_op.encoding_value_))
    h->next_pc = h->pc + static_cast<uint64_t>(imm());
}

BneInst::BneInst(uint32_t raw)
    : BType("bne", raw), rs1_op(64, OperandType::OPR_GPR, inst_.rs1),
      rs2_op(64, OperandType::OPR_GPR, inst_.rs2), offset(13, OperandType::OPR_IMM, imm()) {
  src_operands_.emplace_back(&rs1_op);
  src_operands_.emplace_back(&rs2_op);
  src_operands_.emplace_back(&offset);
}

void BneInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  if (h->read_xreg(rs1_op.encoding_value_) != h->read_xreg(rs2_op.encoding_value_))
    h->next_pc = h->pc + static_cast<uint64_t>(imm());
}

BltInst::BltInst(uint32_t raw)
    : BType("blt", raw), rs1_op(64, OperandType::OPR_GPR, inst_.rs1),
      rs2_op(64, OperandType::OPR_GPR, inst_.rs2), offset(13, OperandType::OPR_IMM, imm()) {
  src_operands_.emplace_back(&rs1_op);
  src_operands_.emplace_back(&rs2_op);
  src_operands_.emplace_back(&offset);
}

void BltInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  if (h->read_xreg(rs1_op.encoding_value_) < h->read_xreg(rs2_op.encoding_value_))
    h->next_pc = h->pc + static_cast<uint64_t>(imm());
}

BgeInst::BgeInst(uint32_t raw)
    : BType("bge", raw), rs1_op(64, OperandType::OPR_GPR, inst_.rs1),
      rs2_op(64, OperandType::OPR_GPR, inst_.rs2), offset(13, OperandType::OPR_IMM, imm()) {
  src_operands_.emplace_back(&rs1_op);
  src_operands_.emplace_back(&rs2_op);
  src_operands_.emplace_back(&offset);
}

void BgeInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  if (h->read_xreg(rs1_op.encoding_value_) >= h->read_xreg(rs2_op.encoding_value_))
    h->next_pc = h->pc + static_cast<uint64_t>(imm());
}

BltuInst::BltuInst(uint32_t raw)
    : BType("bltu", raw), rs1_op(64, OperandType::OPR_GPR, inst_.rs1),
      rs2_op(64, OperandType::OPR_GPR, inst_.rs2), offset(13, OperandType::OPR_IMM, imm()) {
  src_operands_.emplace_back(&rs1_op);
  src_operands_.emplace_back(&rs2_op);
  src_operands_.emplace_back(&offset);
}

void BltuInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  if (static_cast<uint64_t>(h->read_xreg(rs1_op.encoding_value_)) <
      static_cast<uint64_t>(h->read_xreg(rs2_op.encoding_value_)))
    h->next_pc = h->pc + static_cast<uint64_t>(imm());
}

BgeuInst::BgeuInst(uint32_t raw)
    : BType("bgeu", raw), rs1_op(64, OperandType::OPR_GPR, inst_.rs1),
      rs2_op(64, OperandType::OPR_GPR, inst_.rs2), offset(13, OperandType::OPR_IMM, imm()) {
  src_operands_.emplace_back(&rs1_op);
  src_operands_.emplace_back(&rs2_op);
  src_operands_.emplace_back(&offset);
}

void BgeuInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  if (static_cast<uint64_t>(h->read_xreg(rs1_op.encoding_value_)) >=
      static_cast<uint64_t>(h->read_xreg(rs2_op.encoding_value_)))
    h->next_pc = h->pc + static_cast<uint64_t>(imm());
}

// I-type load instructions

LbInst::LbInst(uint32_t raw)
    : IType("lb", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1_op(64, OperandType::OPR_GPR, inst_.rs1), offset(12, OperandType::OPR_IMM, imm()) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1_op);
  src_operands_.emplace_back(&offset);
}

void LbInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = static_cast<uint64_t>(h->read_xreg(rs1_op.encoding_value_) + imm());
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(static_cast<int8_t>(m->read8(addr))));
}

LhInst::LhInst(uint32_t raw)
    : IType("lh", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1_op(64, OperandType::OPR_GPR, inst_.rs1), offset(12, OperandType::OPR_IMM, imm()) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1_op);
  src_operands_.emplace_back(&offset);
}

void LhInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = static_cast<uint64_t>(h->read_xreg(rs1_op.encoding_value_) + imm());
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(static_cast<int16_t>(m->read16(addr))));
}

LwInst::LwInst(uint32_t raw)
    : IType("lw", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1_op(64, OperandType::OPR_GPR, inst_.rs1), offset(12, OperandType::OPR_IMM, imm()) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1_op);
  src_operands_.emplace_back(&offset);
}

void LwInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = static_cast<uint64_t>(h->read_xreg(rs1_op.encoding_value_) + imm());
  h->write_xreg(rd.encoding_value_, sext32(static_cast<int32_t>(m->read32(addr))));
}

LdInst::LdInst(uint32_t raw)
    : IType("ld", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1_op(64, OperandType::OPR_GPR, inst_.rs1), offset(12, OperandType::OPR_IMM, imm()) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1_op);
  src_operands_.emplace_back(&offset);
}

void LdInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = static_cast<uint64_t>(h->read_xreg(rs1_op.encoding_value_) + imm());
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(m->read64(addr)));
}

LbuInst::LbuInst(uint32_t raw)
    : IType("lbu", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1_op(64, OperandType::OPR_GPR, inst_.rs1), offset(12, OperandType::OPR_IMM, imm()) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1_op);
  src_operands_.emplace_back(&offset);
}

void LbuInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = static_cast<uint64_t>(h->read_xreg(rs1_op.encoding_value_) + imm());
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(m->read8(addr)));
}

LhuInst::LhuInst(uint32_t raw)
    : IType("lhu", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1_op(64, OperandType::OPR_GPR, inst_.rs1), offset(12, OperandType::OPR_IMM, imm()) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1_op);
  src_operands_.emplace_back(&offset);
}

void LhuInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = static_cast<uint64_t>(h->read_xreg(rs1_op.encoding_value_) + imm());
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(m->read16(addr)));
}

LwuInst::LwuInst(uint32_t raw)
    : IType("lwu", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1_op(64, OperandType::OPR_GPR, inst_.rs1), offset(12, OperandType::OPR_IMM, imm()) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1_op);
  src_operands_.emplace_back(&offset);
}

void LwuInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = static_cast<uint64_t>(h->read_xreg(rs1_op.encoding_value_) + imm());
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(m->read32(addr)));
}

// S-type store instructions

SbInst::SbInst(uint32_t raw)
    : SType("sb", raw), rs2_op(64, OperandType::OPR_GPR, inst_.rs2),
      rs1_op(64, OperandType::OPR_GPR, inst_.rs1), offset(12, OperandType::OPR_IMM, imm()) {
  src_operands_.emplace_back(&rs2_op);
  src_operands_.emplace_back(&offset);
  src_operands_.emplace_back(&rs1_op);
}

void SbInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = static_cast<uint64_t>(h->read_xreg(rs1_op.encoding_value_) + imm());
  m->write8(addr, static_cast<uint8_t>(h->read_xreg(rs2_op.encoding_value_)));
}

ShInst::ShInst(uint32_t raw)
    : SType("sh", raw), rs2_op(64, OperandType::OPR_GPR, inst_.rs2),
      rs1_op(64, OperandType::OPR_GPR, inst_.rs1), offset(12, OperandType::OPR_IMM, imm()) {
  src_operands_.emplace_back(&rs2_op);
  src_operands_.emplace_back(&offset);
  src_operands_.emplace_back(&rs1_op);
}

void ShInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = static_cast<uint64_t>(h->read_xreg(rs1_op.encoding_value_) + imm());
  m->write16(addr, static_cast<uint16_t>(h->read_xreg(rs2_op.encoding_value_)));
}

SwInst::SwInst(uint32_t raw)
    : SType("sw", raw), rs2_op(64, OperandType::OPR_GPR, inst_.rs2),
      rs1_op(64, OperandType::OPR_GPR, inst_.rs1), offset(12, OperandType::OPR_IMM, imm()) {
  src_operands_.emplace_back(&rs2_op);
  src_operands_.emplace_back(&offset);
  src_operands_.emplace_back(&rs1_op);
}

void SwInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = static_cast<uint64_t>(h->read_xreg(rs1_op.encoding_value_) + imm());
  m->write32(addr, static_cast<uint32_t>(h->read_xreg(rs2_op.encoding_value_)));
}

SdInst::SdInst(uint32_t raw)
    : SType("sd", raw), rs2_op(64, OperandType::OPR_GPR, inst_.rs2),
      rs1_op(64, OperandType::OPR_GPR, inst_.rs1), offset(12, OperandType::OPR_IMM, imm()) {
  src_operands_.emplace_back(&rs2_op);
  src_operands_.emplace_back(&offset);
  src_operands_.emplace_back(&rs1_op);
}

void SdInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = static_cast<uint64_t>(h->read_xreg(rs1_op.encoding_value_) + imm());
  m->write64(addr, static_cast<uint64_t>(h->read_xreg(rs2_op.encoding_value_)));
}

// I-type ALU instructions

AddiInst::AddiInst(uint32_t raw)
    : IType("addi", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), imm_op(12, OperandType::OPR_IMM, imm()) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&imm_op);
}

void AddiInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  h->write_xreg(rd.encoding_value_, h->read_xreg(rs1.encoding_value_) + imm());
}

SltiInst::SltiInst(uint32_t raw)
    : IType("slti", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), imm_op(12, OperandType::OPR_IMM, imm()) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&imm_op);
}

void SltiInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  h->write_xreg(rd.encoding_value_, (h->read_xreg(rs1.encoding_value_) < imm()) ? 1 : 0);
}

SltiuInst::SltiuInst(uint32_t raw)
    : IType("sltiu", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), imm_op(12, OperandType::OPR_IMM, imm()) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&imm_op);
}

void SltiuInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  h->write_xreg(rd.encoding_value_, (static_cast<uint64_t>(h->read_xreg(rs1.encoding_value_)) <
                                     static_cast<uint64_t>(static_cast<int64_t>(imm())))
                                        ? 1
                                        : 0);
}

XoriInst::XoriInst(uint32_t raw)
    : IType("xori", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), imm_op(12, OperandType::OPR_IMM, imm()) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&imm_op);
}

void XoriInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  h->write_xreg(rd.encoding_value_, h->read_xreg(rs1.encoding_value_) ^ imm());
}

OriInst::OriInst(uint32_t raw)
    : IType("ori", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), imm_op(12, OperandType::OPR_IMM, imm()) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&imm_op);
}

void OriInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  h->write_xreg(rd.encoding_value_, h->read_xreg(rs1.encoding_value_) | imm());
}

AndiInst::AndiInst(uint32_t raw)
    : IType("andi", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), imm_op(12, OperandType::OPR_IMM, imm()) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&imm_op);
}

void AndiInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  h->write_xreg(rd.encoding_value_, h->read_xreg(rs1.encoding_value_) & imm());
}

// I-type shift instructions

SlliInst::SlliInst(uint32_t raw)
    : IType("slli", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), imm_op(12, OperandType::OPR_IMM, imm()) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&imm_op);
}

void SlliInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  int shamt = imm() & 0x3f;
  h->write_xreg(rd.encoding_value_, h->read_xreg(rs1.encoding_value_) << shamt);
}

SrliInst::SrliInst(uint32_t raw)
    : IType("srli", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), imm_op(12, OperandType::OPR_IMM, imm()) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&imm_op);
}

void SrliInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  int shamt = imm() & 0x3f;
  h->write_xreg(
      rd.encoding_value_,
      static_cast<int64_t>(static_cast<uint64_t>(h->read_xreg(rs1.encoding_value_)) >> shamt));
}

SraiInst::SraiInst(uint32_t raw)
    : IType("srai", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), imm_op(12, OperandType::OPR_IMM, imm()) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&imm_op);
}

void SraiInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  int shamt = imm() & 0x3f;
  h->write_xreg(rd.encoding_value_,
                arithmetic_right_shift(h->read_xreg(rs1.encoding_value_), shamt));
}

// R-type ALU instructions

AddInst::AddInst(uint32_t raw)
    : RType("add", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}

void AddInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  h->write_xreg(rd.encoding_value_,
                h->read_xreg(rs1.encoding_value_) + h->read_xreg(rs2.encoding_value_));
}

SubInst::SubInst(uint32_t raw)
    : RType("sub", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}

void SubInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  h->write_xreg(rd.encoding_value_,
                h->read_xreg(rs1.encoding_value_) - h->read_xreg(rs2.encoding_value_));
}

SllInst::SllInst(uint32_t raw)
    : RType("sll", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}

void SllInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  int shamt = h->read_xreg(rs2.encoding_value_) & 0x3f;
  h->write_xreg(rd.encoding_value_, h->read_xreg(rs1.encoding_value_) << shamt);
}

SltInst::SltInst(uint32_t raw)
    : RType("slt", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}

void SltInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  h->write_xreg(rd.encoding_value_,
                (h->read_xreg(rs1.encoding_value_) < h->read_xreg(rs2.encoding_value_)) ? 1 : 0);
}

SltuInst::SltuInst(uint32_t raw)
    : RType("sltu", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}

void SltuInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  h->write_xreg(rd.encoding_value_, (static_cast<uint64_t>(h->read_xreg(rs1.encoding_value_)) <
                                     static_cast<uint64_t>(h->read_xreg(rs2.encoding_value_)))
                                        ? 1
                                        : 0);
}

XorInst::XorInst(uint32_t raw)
    : RType("xor", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}

void XorInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  h->write_xreg(rd.encoding_value_,
                h->read_xreg(rs1.encoding_value_) ^ h->read_xreg(rs2.encoding_value_));
}

SrlInst::SrlInst(uint32_t raw)
    : RType("srl", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}

void SrlInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  int shamt = h->read_xreg(rs2.encoding_value_) & 0x3f;
  h->write_xreg(
      rd.encoding_value_,
      static_cast<int64_t>(static_cast<uint64_t>(h->read_xreg(rs1.encoding_value_)) >> shamt));
}

SraInst::SraInst(uint32_t raw)
    : RType("sra", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}

void SraInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  int shamt = h->read_xreg(rs2.encoding_value_) & 0x3f;
  h->write_xreg(rd.encoding_value_,
                arithmetic_right_shift(h->read_xreg(rs1.encoding_value_), shamt));
}

OrInst::OrInst(uint32_t raw)
    : RType("or", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}

void OrInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  h->write_xreg(rd.encoding_value_,
                h->read_xreg(rs1.encoding_value_) | h->read_xreg(rs2.encoding_value_));
}

AndInst::AndInst(uint32_t raw)
    : RType("and", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}

void AndInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  h->write_xreg(rd.encoding_value_,
                h->read_xreg(rs1.encoding_value_) & h->read_xreg(rs2.encoding_value_));
}

// I-type W-ALU instructions (RV64I)

AddiwInst::AddiwInst(uint32_t raw)
    : IType("addiw", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), imm_op(12, OperandType::OPR_IMM, imm()) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&imm_op);
}

void AddiwInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  int32_t result = static_cast<int32_t>(h->read_xreg(rs1.encoding_value_)) + imm();
  h->write_xreg(rd.encoding_value_, sext32(result));
}

SlliwInst::SlliwInst(uint32_t raw)
    : IType("slliw", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), imm_op(12, OperandType::OPR_IMM, imm()) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&imm_op);
}

void SlliwInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  int shamt = imm() & 0x1f;
  int32_t result = static_cast<int32_t>(h->read_xreg(rs1.encoding_value_)) << shamt;
  h->write_xreg(rd.encoding_value_, sext32(result));
}

SrliwInst::SrliwInst(uint32_t raw)
    : IType("srliw", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), imm_op(12, OperandType::OPR_IMM, imm()) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&imm_op);
}

void SrliwInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  int shamt = imm() & 0x1f;
  uint32_t val = static_cast<uint32_t>(h->read_xreg(rs1.encoding_value_));
  h->write_xreg(rd.encoding_value_, sext32(static_cast<int32_t>(val >> shamt)));
}

SraiwInst::SraiwInst(uint32_t raw)
    : IType("sraiw", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), imm_op(12, OperandType::OPR_IMM, imm()) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&imm_op);
}

void SraiwInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  int shamt = imm() & 0x1f;
  int32_t val = static_cast<int32_t>(h->read_xreg(rs1.encoding_value_));
  h->write_xreg(rd.encoding_value_, sext32(arithmetic_right_shift_32(val, shamt)));
}

// R-type W-ALU instructions (RV64I)

AddwInst::AddwInst(uint32_t raw)
    : RType("addw", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}

void AddwInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  int32_t result = static_cast<int32_t>(h->read_xreg(rs1.encoding_value_)) +
                   static_cast<int32_t>(h->read_xreg(rs2.encoding_value_));
  h->write_xreg(rd.encoding_value_, sext32(result));
}

SubwInst::SubwInst(uint32_t raw)
    : RType("subw", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}

void SubwInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  int32_t result = static_cast<int32_t>(h->read_xreg(rs1.encoding_value_)) -
                   static_cast<int32_t>(h->read_xreg(rs2.encoding_value_));
  h->write_xreg(rd.encoding_value_, sext32(result));
}

SllwInst::SllwInst(uint32_t raw)
    : RType("sllw", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}

void SllwInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  int shamt = h->read_xreg(rs2.encoding_value_) & 0x1f;
  int32_t result = static_cast<int32_t>(h->read_xreg(rs1.encoding_value_)) << shamt;
  h->write_xreg(rd.encoding_value_, sext32(result));
}

SrlwInst::SrlwInst(uint32_t raw)
    : RType("srlw", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}

void SrlwInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  int shamt = h->read_xreg(rs2.encoding_value_) & 0x1f;
  uint32_t val = static_cast<uint32_t>(h->read_xreg(rs1.encoding_value_));
  h->write_xreg(rd.encoding_value_, sext32(static_cast<int32_t>(val >> shamt)));
}

SrawInst::SrawInst(uint32_t raw)
    : RType("sraw", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}

void SrawInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  int shamt = h->read_xreg(rs2.encoding_value_) & 0x1f;
  int32_t val = static_cast<int32_t>(h->read_xreg(rs1.encoding_value_));
  h->write_xreg(rd.encoding_value_, sext32(arithmetic_right_shift_32(val, shamt)));
}

// I-type system instructions

FenceInst::FenceInst(uint32_t raw) : IType("fence", raw), imm_op(12, OperandType::OPR_IMM, imm()) {
  src_operands_.emplace_back(&imm_op);
}

void FenceInst::execute(HartState &ctx) {
  (void)ctx; // No-op for single-hart simulation.
}

// I-type environment instructions

EcallInst::EcallInst(uint32_t raw) : IType("ecall", raw) {}

void EcallInst::execute(HartState &ctx) { as_hart(ctx)->halted = true; }

EbreakInst::EbreakInst(uint32_t raw) : IType("ebreak", raw) {}

void EbreakInst::execute(HartState &ctx) { as_hart(ctx)->halted = true; }

} // namespace detail
} // namespace risc_v
} // namespace rocjitsu

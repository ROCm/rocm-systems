// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/risc_v/a_ext.h"

#include "rocjitsu/vm/risc_v/hart_state.h"
#include "rocjitsu/vm/risc_v/memory.h"

#include <algorithm>

namespace rocjitsu {
namespace risc_v {
namespace detail {

LrWInst::LrWInst(uint32_t raw)
    : RType("lr.w", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  if (inst_.funct7 & 0x02)
    modifiers_ += " aq";
  if (inst_.funct7 & 0x01)
    modifiers_ += " rl";
}
void LrWInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = h->read_xreg(rs1.encoding_value_);
  int64_t val = sext32(static_cast<int32_t>(m->read32(addr)));
  h->write_xreg(rd.encoding_value_, val);
  h->reservation_valid = true;
  h->reservation_addr = addr;
}

ScWInst::ScWInst(uint32_t raw)
    : RType("sc.w", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
  if (inst_.funct7 & 0x02)
    modifiers_ += " aq";
  if (inst_.funct7 & 0x01)
    modifiers_ += " rl";
}
void ScWInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = h->read_xreg(rs1.encoding_value_);
  if (h->reservation_valid && h->reservation_addr == addr) {
    m->write32(addr, static_cast<uint32_t>(h->read_xreg(rs2.encoding_value_)));
    h->write_xreg(rd.encoding_value_, 0);
  } else {
    h->write_xreg(rd.encoding_value_, 1);
  }
  h->reservation_valid = false;
}

AmoswapWInst::AmoswapWInst(uint32_t raw)
    : RType("amoswap.w", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
  if (inst_.funct7 & 0x02)
    modifiers_ += " aq";
  if (inst_.funct7 & 0x01)
    modifiers_ += " rl";
}
void AmoswapWInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = h->read_xreg(rs1.encoding_value_);
  int64_t old_val = sext32(static_cast<int32_t>(m->read32(addr)));
  int64_t rs2_val = h->read_xreg(rs2.encoding_value_);
  int64_t new_val = rs2_val;
  m->write32(addr, static_cast<uint32_t>(new_val));
  h->write_xreg(rd.encoding_value_, old_val);
}

AmoaddWInst::AmoaddWInst(uint32_t raw)
    : RType("amoadd.w", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
  if (inst_.funct7 & 0x02)
    modifiers_ += " aq";
  if (inst_.funct7 & 0x01)
    modifiers_ += " rl";
}
void AmoaddWInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = h->read_xreg(rs1.encoding_value_);
  int64_t old_val = sext32(static_cast<int32_t>(m->read32(addr)));
  int64_t rs2_val = h->read_xreg(rs2.encoding_value_);
  int64_t new_val = static_cast<int32_t>(static_cast<uint32_t>(old_val + rs2_val));
  m->write32(addr, static_cast<uint32_t>(new_val));
  h->write_xreg(rd.encoding_value_, old_val);
}

AmoxorWInst::AmoxorWInst(uint32_t raw)
    : RType("amoxor.w", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
  if (inst_.funct7 & 0x02)
    modifiers_ += " aq";
  if (inst_.funct7 & 0x01)
    modifiers_ += " rl";
}
void AmoxorWInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = h->read_xreg(rs1.encoding_value_);
  int64_t old_val = sext32(static_cast<int32_t>(m->read32(addr)));
  int64_t rs2_val = h->read_xreg(rs2.encoding_value_);
  int64_t new_val = static_cast<int32_t>(static_cast<uint32_t>(old_val ^ rs2_val));
  m->write32(addr, static_cast<uint32_t>(new_val));
  h->write_xreg(rd.encoding_value_, old_val);
}

AmoandWInst::AmoandWInst(uint32_t raw)
    : RType("amoand.w", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
  if (inst_.funct7 & 0x02)
    modifiers_ += " aq";
  if (inst_.funct7 & 0x01)
    modifiers_ += " rl";
}
void AmoandWInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = h->read_xreg(rs1.encoding_value_);
  int64_t old_val = sext32(static_cast<int32_t>(m->read32(addr)));
  int64_t rs2_val = h->read_xreg(rs2.encoding_value_);
  int64_t new_val = static_cast<int32_t>(static_cast<uint32_t>(old_val & rs2_val));
  m->write32(addr, static_cast<uint32_t>(new_val));
  h->write_xreg(rd.encoding_value_, old_val);
}

AmoorWInst::AmoorWInst(uint32_t raw)
    : RType("amoor.w", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
  if (inst_.funct7 & 0x02)
    modifiers_ += " aq";
  if (inst_.funct7 & 0x01)
    modifiers_ += " rl";
}
void AmoorWInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = h->read_xreg(rs1.encoding_value_);
  int64_t old_val = sext32(static_cast<int32_t>(m->read32(addr)));
  int64_t rs2_val = h->read_xreg(rs2.encoding_value_);
  int64_t new_val = static_cast<int32_t>(static_cast<uint32_t>(old_val | rs2_val));
  m->write32(addr, static_cast<uint32_t>(new_val));
  h->write_xreg(rd.encoding_value_, old_val);
}

AmominWInst::AmominWInst(uint32_t raw)
    : RType("amomin.w", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
  if (inst_.funct7 & 0x02)
    modifiers_ += " aq";
  if (inst_.funct7 & 0x01)
    modifiers_ += " rl";
}
void AmominWInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = h->read_xreg(rs1.encoding_value_);
  int64_t old_val = sext32(static_cast<int32_t>(m->read32(addr)));
  int64_t rs2_val = h->read_xreg(rs2.encoding_value_);
  int64_t new_val = std::min(old_val, rs2_val);
  m->write32(addr, static_cast<uint32_t>(new_val));
  h->write_xreg(rd.encoding_value_, old_val);
}

AmomaxWInst::AmomaxWInst(uint32_t raw)
    : RType("amomax.w", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
  if (inst_.funct7 & 0x02)
    modifiers_ += " aq";
  if (inst_.funct7 & 0x01)
    modifiers_ += " rl";
}
void AmomaxWInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = h->read_xreg(rs1.encoding_value_);
  int64_t old_val = sext32(static_cast<int32_t>(m->read32(addr)));
  int64_t rs2_val = h->read_xreg(rs2.encoding_value_);
  int64_t new_val = std::max(old_val, rs2_val);
  m->write32(addr, static_cast<uint32_t>(new_val));
  h->write_xreg(rd.encoding_value_, old_val);
}

AmominuWInst::AmominuWInst(uint32_t raw)
    : RType("amominu.w", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
  if (inst_.funct7 & 0x02)
    modifiers_ += " aq";
  if (inst_.funct7 & 0x01)
    modifiers_ += " rl";
}
void AmominuWInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = h->read_xreg(rs1.encoding_value_);
  int64_t old_val = sext32(static_cast<int32_t>(m->read32(addr)));
  int64_t rs2_val = h->read_xreg(rs2.encoding_value_);
  int64_t new_val = static_cast<int64_t>(
      std::min(static_cast<uint64_t>(old_val), static_cast<uint64_t>(rs2_val)));
  m->write32(addr, static_cast<uint32_t>(new_val));
  h->write_xreg(rd.encoding_value_, old_val);
}

AmomaxuWInst::AmomaxuWInst(uint32_t raw)
    : RType("amomaxu.w", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
  if (inst_.funct7 & 0x02)
    modifiers_ += " aq";
  if (inst_.funct7 & 0x01)
    modifiers_ += " rl";
}
void AmomaxuWInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = h->read_xreg(rs1.encoding_value_);
  int64_t old_val = sext32(static_cast<int32_t>(m->read32(addr)));
  int64_t rs2_val = h->read_xreg(rs2.encoding_value_);
  int64_t new_val = static_cast<int64_t>(
      std::max(static_cast<uint64_t>(old_val), static_cast<uint64_t>(rs2_val)));
  m->write32(addr, static_cast<uint32_t>(new_val));
  h->write_xreg(rd.encoding_value_, old_val);
}

LrDInst::LrDInst(uint32_t raw)
    : RType("lr.d", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  if (inst_.funct7 & 0x02)
    modifiers_ += " aq";
  if (inst_.funct7 & 0x01)
    modifiers_ += " rl";
}
void LrDInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = h->read_xreg(rs1.encoding_value_);
  int64_t val = static_cast<int64_t>(m->read64(addr));
  h->write_xreg(rd.encoding_value_, val);
  h->reservation_valid = true;
  h->reservation_addr = addr;
}

ScDInst::ScDInst(uint32_t raw)
    : RType("sc.d", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
  if (inst_.funct7 & 0x02)
    modifiers_ += " aq";
  if (inst_.funct7 & 0x01)
    modifiers_ += " rl";
}
void ScDInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = h->read_xreg(rs1.encoding_value_);
  if (h->reservation_valid && h->reservation_addr == addr) {
    m->write64(addr, static_cast<uint64_t>(h->read_xreg(rs2.encoding_value_)));
    h->write_xreg(rd.encoding_value_, 0);
  } else {
    h->write_xreg(rd.encoding_value_, 1);
  }
  h->reservation_valid = false;
}

AmoswapDInst::AmoswapDInst(uint32_t raw)
    : RType("amoswap.d", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
  if (inst_.funct7 & 0x02)
    modifiers_ += " aq";
  if (inst_.funct7 & 0x01)
    modifiers_ += " rl";
}
void AmoswapDInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = h->read_xreg(rs1.encoding_value_);
  int64_t old_val = static_cast<int64_t>(m->read64(addr));
  int64_t rs2_val = h->read_xreg(rs2.encoding_value_);
  int64_t new_val = rs2_val;
  m->write64(addr, static_cast<uint64_t>(new_val));
  h->write_xreg(rd.encoding_value_, old_val);
}

AmoaddDInst::AmoaddDInst(uint32_t raw)
    : RType("amoadd.d", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
  if (inst_.funct7 & 0x02)
    modifiers_ += " aq";
  if (inst_.funct7 & 0x01)
    modifiers_ += " rl";
}
void AmoaddDInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = h->read_xreg(rs1.encoding_value_);
  int64_t old_val = static_cast<int64_t>(m->read64(addr));
  int64_t rs2_val = h->read_xreg(rs2.encoding_value_);
  int64_t new_val = old_val + rs2_val;
  m->write64(addr, static_cast<uint64_t>(new_val));
  h->write_xreg(rd.encoding_value_, old_val);
}

AmoxorDInst::AmoxorDInst(uint32_t raw)
    : RType("amoxor.d", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
  if (inst_.funct7 & 0x02)
    modifiers_ += " aq";
  if (inst_.funct7 & 0x01)
    modifiers_ += " rl";
}
void AmoxorDInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = h->read_xreg(rs1.encoding_value_);
  int64_t old_val = static_cast<int64_t>(m->read64(addr));
  int64_t rs2_val = h->read_xreg(rs2.encoding_value_);
  int64_t new_val = old_val ^ rs2_val;
  m->write64(addr, static_cast<uint64_t>(new_val));
  h->write_xreg(rd.encoding_value_, old_val);
}

AmoandDInst::AmoandDInst(uint32_t raw)
    : RType("amoand.d", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
  if (inst_.funct7 & 0x02)
    modifiers_ += " aq";
  if (inst_.funct7 & 0x01)
    modifiers_ += " rl";
}
void AmoandDInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = h->read_xreg(rs1.encoding_value_);
  int64_t old_val = static_cast<int64_t>(m->read64(addr));
  int64_t rs2_val = h->read_xreg(rs2.encoding_value_);
  int64_t new_val = old_val & rs2_val;
  m->write64(addr, static_cast<uint64_t>(new_val));
  h->write_xreg(rd.encoding_value_, old_val);
}

AmoorDInst::AmoorDInst(uint32_t raw)
    : RType("amoor.d", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
  if (inst_.funct7 & 0x02)
    modifiers_ += " aq";
  if (inst_.funct7 & 0x01)
    modifiers_ += " rl";
}
void AmoorDInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = h->read_xreg(rs1.encoding_value_);
  int64_t old_val = static_cast<int64_t>(m->read64(addr));
  int64_t rs2_val = h->read_xreg(rs2.encoding_value_);
  int64_t new_val = old_val | rs2_val;
  m->write64(addr, static_cast<uint64_t>(new_val));
  h->write_xreg(rd.encoding_value_, old_val);
}

AmominDInst::AmominDInst(uint32_t raw)
    : RType("amomin.d", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
  if (inst_.funct7 & 0x02)
    modifiers_ += " aq";
  if (inst_.funct7 & 0x01)
    modifiers_ += " rl";
}
void AmominDInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = h->read_xreg(rs1.encoding_value_);
  int64_t old_val = static_cast<int64_t>(m->read64(addr));
  int64_t rs2_val = h->read_xreg(rs2.encoding_value_);
  int64_t new_val = std::min(old_val, rs2_val);
  m->write64(addr, static_cast<uint64_t>(new_val));
  h->write_xreg(rd.encoding_value_, old_val);
}

AmomaxDInst::AmomaxDInst(uint32_t raw)
    : RType("amomax.d", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
  if (inst_.funct7 & 0x02)
    modifiers_ += " aq";
  if (inst_.funct7 & 0x01)
    modifiers_ += " rl";
}
void AmomaxDInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = h->read_xreg(rs1.encoding_value_);
  int64_t old_val = static_cast<int64_t>(m->read64(addr));
  int64_t rs2_val = h->read_xreg(rs2.encoding_value_);
  int64_t new_val = std::max(old_val, rs2_val);
  m->write64(addr, static_cast<uint64_t>(new_val));
  h->write_xreg(rd.encoding_value_, old_val);
}

AmominuDInst::AmominuDInst(uint32_t raw)
    : RType("amominu.d", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
  if (inst_.funct7 & 0x02)
    modifiers_ += " aq";
  if (inst_.funct7 & 0x01)
    modifiers_ += " rl";
}
void AmominuDInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = h->read_xreg(rs1.encoding_value_);
  int64_t old_val = static_cast<int64_t>(m->read64(addr));
  int64_t rs2_val = h->read_xreg(rs2.encoding_value_);
  int64_t new_val = static_cast<int64_t>(
      std::min(static_cast<uint64_t>(old_val), static_cast<uint64_t>(rs2_val)));
  m->write64(addr, static_cast<uint64_t>(new_val));
  h->write_xreg(rd.encoding_value_, old_val);
}

AmomaxuDInst::AmomaxuDInst(uint32_t raw)
    : RType("amomaxu.d", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
  if (inst_.funct7 & 0x02)
    modifiers_ += " aq";
  if (inst_.funct7 & 0x01)
    modifiers_ += " rl";
}
void AmomaxuDInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = h->read_xreg(rs1.encoding_value_);
  int64_t old_val = static_cast<int64_t>(m->read64(addr));
  int64_t rs2_val = h->read_xreg(rs2.encoding_value_);
  int64_t new_val = static_cast<int64_t>(
      std::max(static_cast<uint64_t>(old_val), static_cast<uint64_t>(rs2_val)));
  m->write64(addr, static_cast<uint64_t>(new_val));
  h->write_xreg(rd.encoding_value_, old_val);
}

} // namespace detail
} // namespace risc_v
} // namespace rocjitsu

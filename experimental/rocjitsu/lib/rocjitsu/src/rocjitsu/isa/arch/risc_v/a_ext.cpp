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
    : RType("lr.w", raw, make_exec_fn<LrWInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;

  num_src_ = 1;
  num_dst_ = 1;
}
void LrWInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = h->read_xreg(rs1.encoding_value_);
  int64_t val = sext32(static_cast<int32_t>(m->read32(addr)));
  h->write_xreg(rd.encoding_value_, val);
  h->reservation_valid = true;
  h->reservation_addr = addr;
}

ScWInst::ScWInst(uint32_t raw)
    : RType("sc.w", raw, make_exec_fn<ScWInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;

  num_src_ = 2;
  num_dst_ = 1;
}
void ScWInst::execute_impl(HartState &ctx) {
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
    : RType("amoswap.w", raw, make_exec_fn<AmoswapWInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;

  num_src_ = 2;
  num_dst_ = 1;
}
void AmoswapWInst::execute_impl(HartState &ctx) {
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
    : RType("amoadd.w", raw, make_exec_fn<AmoaddWInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;

  num_src_ = 2;
  num_dst_ = 1;
}
void AmoaddWInst::execute_impl(HartState &ctx) {
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
    : RType("amoxor.w", raw, make_exec_fn<AmoxorWInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;

  num_src_ = 2;
  num_dst_ = 1;
}
void AmoxorWInst::execute_impl(HartState &ctx) {
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
    : RType("amoand.w", raw, make_exec_fn<AmoandWInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;

  num_src_ = 2;
  num_dst_ = 1;
}
void AmoandWInst::execute_impl(HartState &ctx) {
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
    : RType("amoor.w", raw, make_exec_fn<AmoorWInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;

  num_src_ = 2;
  num_dst_ = 1;
}
void AmoorWInst::execute_impl(HartState &ctx) {
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
    : RType("amomin.w", raw, make_exec_fn<AmominWInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;

  num_src_ = 2;
  num_dst_ = 1;
}
void AmominWInst::execute_impl(HartState &ctx) {
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
    : RType("amomax.w", raw, make_exec_fn<AmomaxWInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;

  num_src_ = 2;
  num_dst_ = 1;
}
void AmomaxWInst::execute_impl(HartState &ctx) {
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
    : RType("amominu.w", raw, make_exec_fn<AmominuWInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;

  num_src_ = 2;
  num_dst_ = 1;
}
void AmominuWInst::execute_impl(HartState &ctx) {
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
    : RType("amomaxu.w", raw, make_exec_fn<AmomaxuWInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;

  num_src_ = 2;
  num_dst_ = 1;
}
void AmomaxuWInst::execute_impl(HartState &ctx) {
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
    : RType("lr.d", raw, make_exec_fn<LrDInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;

  num_src_ = 1;
  num_dst_ = 1;
}
void LrDInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = h->read_xreg(rs1.encoding_value_);
  int64_t val = static_cast<int64_t>(m->read64(addr));
  h->write_xreg(rd.encoding_value_, val);
  h->reservation_valid = true;
  h->reservation_addr = addr;
}

ScDInst::ScDInst(uint32_t raw)
    : RType("sc.d", raw, make_exec_fn<ScDInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;

  num_src_ = 2;
  num_dst_ = 1;
}
void ScDInst::execute_impl(HartState &ctx) {
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
    : RType("amoswap.d", raw, make_exec_fn<AmoswapDInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;

  num_src_ = 2;
  num_dst_ = 1;
}
void AmoswapDInst::execute_impl(HartState &ctx) {
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
    : RType("amoadd.d", raw, make_exec_fn<AmoaddDInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;

  num_src_ = 2;
  num_dst_ = 1;
}
void AmoaddDInst::execute_impl(HartState &ctx) {
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
    : RType("amoxor.d", raw, make_exec_fn<AmoxorDInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;

  num_src_ = 2;
  num_dst_ = 1;
}
void AmoxorDInst::execute_impl(HartState &ctx) {
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
    : RType("amoand.d", raw, make_exec_fn<AmoandDInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;

  num_src_ = 2;
  num_dst_ = 1;
}
void AmoandDInst::execute_impl(HartState &ctx) {
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
    : RType("amoor.d", raw, make_exec_fn<AmoorDInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;

  num_src_ = 2;
  num_dst_ = 1;
}
void AmoorDInst::execute_impl(HartState &ctx) {
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
    : RType("amomin.d", raw, make_exec_fn<AmominDInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;

  num_src_ = 2;
  num_dst_ = 1;
}
void AmominDInst::execute_impl(HartState &ctx) {
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
    : RType("amomax.d", raw, make_exec_fn<AmomaxDInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;

  num_src_ = 2;
  num_dst_ = 1;
}
void AmomaxDInst::execute_impl(HartState &ctx) {
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
    : RType("amominu.d", raw, make_exec_fn<AmominuDInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;

  num_src_ = 2;
  num_dst_ = 1;
}
void AmominuDInst::execute_impl(HartState &ctx) {
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
    : RType("amomaxu.d", raw, make_exec_fn<AmomaxuDInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;

  num_src_ = 2;
  num_dst_ = 1;
}
void AmomaxuDInst::execute_impl(HartState &ctx) {
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

// All A-extension atomics share the same aq/rl modifier pattern.
#define RV_ATOMIC_BUILD_MODIFIERS(cls)                                                             \
  void cls::build_modifiers(std::string &out) const {                                              \
    if (inst_.funct7 & 0x02)                                                                       \
      out += " aq";                                                                                \
    if (inst_.funct7 & 0x01)                                                                       \
      out += " rl";                                                                                \
  }

RV_ATOMIC_BUILD_MODIFIERS(LrWInst)
RV_ATOMIC_BUILD_MODIFIERS(ScWInst)
RV_ATOMIC_BUILD_MODIFIERS(AmoswapWInst)
RV_ATOMIC_BUILD_MODIFIERS(AmoaddWInst)
RV_ATOMIC_BUILD_MODIFIERS(AmoxorWInst)
RV_ATOMIC_BUILD_MODIFIERS(AmoandWInst)
RV_ATOMIC_BUILD_MODIFIERS(AmoorWInst)
RV_ATOMIC_BUILD_MODIFIERS(AmominWInst)
RV_ATOMIC_BUILD_MODIFIERS(AmomaxWInst)
RV_ATOMIC_BUILD_MODIFIERS(AmominuWInst)
RV_ATOMIC_BUILD_MODIFIERS(AmomaxuWInst)
RV_ATOMIC_BUILD_MODIFIERS(LrDInst)
RV_ATOMIC_BUILD_MODIFIERS(ScDInst)
RV_ATOMIC_BUILD_MODIFIERS(AmoswapDInst)
RV_ATOMIC_BUILD_MODIFIERS(AmoaddDInst)
RV_ATOMIC_BUILD_MODIFIERS(AmoxorDInst)
RV_ATOMIC_BUILD_MODIFIERS(AmoandDInst)
RV_ATOMIC_BUILD_MODIFIERS(AmoorDInst)
RV_ATOMIC_BUILD_MODIFIERS(AmominDInst)
RV_ATOMIC_BUILD_MODIFIERS(AmomaxDInst)
RV_ATOMIC_BUILD_MODIFIERS(AmominuDInst)
RV_ATOMIC_BUILD_MODIFIERS(AmomaxuDInst)

#undef RV_ATOMIC_BUILD_MODIFIERS

} // namespace detail
} // namespace risc_v
} // namespace rocjitsu

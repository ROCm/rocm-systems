// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/risc_v/d_ext.h"
#include "rocjitsu/vm/risc_v/hart_state.h"
#include "rocjitsu/vm/risc_v/memory.h"

#include <algorithm>
#include <bit>
#include <cfenv>
#include <climits>
#include <cmath>
#include <cstdint>

namespace rocjitsu {
namespace risc_v {
namespace detail {

// I-type FP load instruction

FldInst::FldInst(uint32_t raw)
    : IType("fld", raw, make_exec_fn<FldInst>()), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), offset(12, OperandType::OPR_IMM, imm()) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &offset;
  src_operands_[1] = &rs1;
  num_src_ = 2;
  num_dst_ = 1;
}

void FldInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = static_cast<uint64_t>(h->read_xreg(rs1.encoding_value_) + imm());
  h->write_freg(rd.encoding_value_, m->read64(addr));
}

// S-type FP store instruction

FsdInst::FsdInst(uint32_t raw)
    : SType("fsd", raw, make_exec_fn<FsdInst>()), rs2_op(64, OperandType::OPR_FPR, inst_.rs2),
      rs1_op(64, OperandType::OPR_GPR, inst_.rs1), offset(12, OperandType::OPR_IMM, imm()) {
  src_operands_[0] = &rs2_op;
  src_operands_[1] = &offset;
  src_operands_[2] = &rs1_op;
  num_src_ = 3;
  num_dst_ = 0;
}

void FsdInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = static_cast<uint64_t>(h->read_xreg(rs1_op.encoding_value_) + imm());
  m->write64(addr, h->read_freg(rs2_op.encoding_value_));
}

// R-type FP compute instructions (FPR -> FPR)

FaddDInst::FaddDInst(uint32_t raw)
    : RType("fadd.d", raw, make_exec_fn<FaddDInst>()), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1), rs2(64, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}

void FaddDInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  double d1 = std::bit_cast<double>(h->read_freg(rs1.encoding_value_));
  double d2 = std::bit_cast<double>(h->read_freg(rs2.encoding_value_));
  double result = d1 + d2;
  h->write_freg(rd.encoding_value_, std::bit_cast<uint64_t>(result));
}

FsubDInst::FsubDInst(uint32_t raw)
    : RType("fsub.d", raw, make_exec_fn<FsubDInst>()), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1), rs2(64, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}

void FsubDInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  double d1 = std::bit_cast<double>(h->read_freg(rs1.encoding_value_));
  double d2 = std::bit_cast<double>(h->read_freg(rs2.encoding_value_));
  double result = d1 - d2;
  h->write_freg(rd.encoding_value_, std::bit_cast<uint64_t>(result));
}

FmulDInst::FmulDInst(uint32_t raw)
    : RType("fmul.d", raw, make_exec_fn<FmulDInst>()), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1), rs2(64, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}

void FmulDInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  double d1 = std::bit_cast<double>(h->read_freg(rs1.encoding_value_));
  double d2 = std::bit_cast<double>(h->read_freg(rs2.encoding_value_));
  double result = d1 * d2;
  h->write_freg(rd.encoding_value_, std::bit_cast<uint64_t>(result));
}

FdivDInst::FdivDInst(uint32_t raw)
    : RType("fdiv.d", raw, make_exec_fn<FdivDInst>()), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1), rs2(64, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}

void FdivDInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  double d1 = std::bit_cast<double>(h->read_freg(rs1.encoding_value_));
  double d2 = std::bit_cast<double>(h->read_freg(rs2.encoding_value_));
  double result = d1 / d2;
  h->write_freg(rd.encoding_value_, std::bit_cast<uint64_t>(result));
}

FsgnjDInst::FsgnjDInst(uint32_t raw)
    : RType("fsgnj.d", raw, make_exec_fn<FsgnjDInst>()), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1), rs2(64, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}

void FsgnjDInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  uint64_t b1 = h->read_freg(rs1.encoding_value_);
  uint64_t b2 = h->read_freg(rs2.encoding_value_);
  uint64_t result = (b1 & 0x7FFFFFFFFFFFFFFFULL) | (b2 & 0x8000000000000000ULL);
  h->write_freg(rd.encoding_value_, result);
}

FsgnjnDInst::FsgnjnDInst(uint32_t raw)
    : RType("fsgnjn.d", raw, make_exec_fn<FsgnjnDInst>()), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1), rs2(64, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}

void FsgnjnDInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  uint64_t b1 = h->read_freg(rs1.encoding_value_);
  uint64_t b2 = h->read_freg(rs2.encoding_value_);
  uint64_t result = (b1 & 0x7FFFFFFFFFFFFFFFULL) | (~b2 & 0x8000000000000000ULL);
  h->write_freg(rd.encoding_value_, result);
}

FsgnjxDInst::FsgnjxDInst(uint32_t raw)
    : RType("fsgnjx.d", raw, make_exec_fn<FsgnjxDInst>()), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1), rs2(64, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}

void FsgnjxDInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  uint64_t b1 = h->read_freg(rs1.encoding_value_);
  uint64_t b2 = h->read_freg(rs2.encoding_value_);
  uint64_t result = b1 ^ (b2 & 0x8000000000000000ULL);
  h->write_freg(rd.encoding_value_, result);
}

FminDInst::FminDInst(uint32_t raw)
    : RType("fmin.d", raw, make_exec_fn<FminDInst>()), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1), rs2(64, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}

void FminDInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  double d1 = std::bit_cast<double>(h->read_freg(rs1.encoding_value_));
  double d2 = std::bit_cast<double>(h->read_freg(rs2.encoding_value_));
  double result;
  if (std::isnan(d1) && std::isnan(d2)) {
    result = std::bit_cast<double>(uint64_t{0x7FF8000000000000ULL}); // canonical NaN
  } else if (std::isnan(d1)) {
    result = d2;
  } else if (std::isnan(d2)) {
    result = d1;
  } else {
    // -0.0 is less than +0.0 per RISC-V spec
    if (d1 == d2) {
      uint64_t b1 = std::bit_cast<uint64_t>(d1);
      result = (b1 & 0x8000000000000000ULL) ? d1 : d2;
    } else {
      result = (d1 < d2) ? d1 : d2;
    }
  }
  h->write_freg(rd.encoding_value_, std::bit_cast<uint64_t>(result));
}

FmaxDInst::FmaxDInst(uint32_t raw)
    : RType("fmax.d", raw, make_exec_fn<FmaxDInst>()), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1), rs2(64, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}

void FmaxDInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  double d1 = std::bit_cast<double>(h->read_freg(rs1.encoding_value_));
  double d2 = std::bit_cast<double>(h->read_freg(rs2.encoding_value_));
  double result;
  if (std::isnan(d1) && std::isnan(d2)) {
    result = std::bit_cast<double>(uint64_t{0x7FF8000000000000ULL}); // canonical NaN
  } else if (std::isnan(d1)) {
    result = d2;
  } else if (std::isnan(d2)) {
    result = d1;
  } else {
    // +0.0 is greater than -0.0 per RISC-V spec
    if (d1 == d2) {
      uint64_t b1 = std::bit_cast<uint64_t>(d1);
      result = (b1 & 0x8000000000000000ULL) ? d2 : d1;
    } else {
      result = (d1 > d2) ? d1 : d2;
    }
  }
  h->write_freg(rd.encoding_value_, std::bit_cast<uint64_t>(result));
}

// R-type FP unary instruction (single source)

FsqrtDInst::FsqrtDInst(uint32_t raw)
    : RType("fsqrt.d", raw, make_exec_fn<FsqrtDInst>()), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  num_src_ = 1;
  num_dst_ = 1;
}

void FsqrtDInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  double d1 = std::bit_cast<double>(h->read_freg(rs1.encoding_value_));
  double result = std::sqrt(d1);
  h->write_freg(rd.encoding_value_, std::bit_cast<uint64_t>(result));
}

// R-type FP-to-int conversions (FPR -> GPR)

FcvtWDInst::FcvtWDInst(uint32_t raw)
    : RType("fcvt.w.d", raw, make_exec_fn<FcvtWDInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  num_src_ = 1;
  num_dst_ = 1;
}

void FcvtWDInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  double d1 = std::bit_cast<double>(h->read_freg(rs1.encoding_value_));
  int32_t result;
  if (std::isnan(d1)) {
    result = INT32_MAX;
  } else if (d1 >= static_cast<double>(INT32_MAX)) {
    result = INT32_MAX;
  } else if (d1 <= static_cast<double>(INT32_MIN)) {
    result = INT32_MIN;
  } else {
    result = static_cast<int32_t>(d1);
  }
  h->write_xreg(rd.encoding_value_, sext32(result));
}

FcvtWuDInst::FcvtWuDInst(uint32_t raw)
    : RType("fcvt.wu.d", raw, make_exec_fn<FcvtWuDInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  num_src_ = 1;
  num_dst_ = 1;
}

void FcvtWuDInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  double d1 = std::bit_cast<double>(h->read_freg(rs1.encoding_value_));
  uint32_t result;
  if (std::isnan(d1)) {
    result = UINT32_MAX;
  } else if (d1 >= static_cast<double>(UINT32_MAX)) {
    result = UINT32_MAX;
  } else if (d1 <= 0.0) {
    result = 0;
  } else {
    result = static_cast<uint32_t>(d1);
  }
  // Per RISC-V spec: sign-extend the 32-bit result to 64 bits
  h->write_xreg(rd.encoding_value_, sext32(static_cast<int32_t>(result)));
}

FcvtLDInst::FcvtLDInst(uint32_t raw)
    : RType("fcvt.l.d", raw, make_exec_fn<FcvtLDInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  num_src_ = 1;
  num_dst_ = 1;
}

void FcvtLDInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  double d1 = std::bit_cast<double>(h->read_freg(rs1.encoding_value_));
  int64_t result;
  if (std::isnan(d1)) {
    result = INT64_MAX;
  } else if (d1 >= static_cast<double>(INT64_MAX)) {
    result = INT64_MAX;
  } else if (d1 <= static_cast<double>(INT64_MIN)) {
    result = INT64_MIN;
  } else {
    result = static_cast<int64_t>(d1);
  }
  h->write_xreg(rd.encoding_value_, result);
}

FcvtLuDInst::FcvtLuDInst(uint32_t raw)
    : RType("fcvt.lu.d", raw, make_exec_fn<FcvtLuDInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  num_src_ = 1;
  num_dst_ = 1;
}

void FcvtLuDInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  double d1 = std::bit_cast<double>(h->read_freg(rs1.encoding_value_));
  uint64_t result;
  if (std::isnan(d1)) {
    result = UINT64_MAX;
  } else if (d1 >= static_cast<double>(UINT64_MAX)) {
    result = UINT64_MAX;
  } else if (d1 <= 0.0) {
    result = 0;
  } else {
    result = static_cast<uint64_t>(d1);
  }
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(result));
}

// R-type int-to-FP conversions (GPR -> FPR)

FcvtDWInst::FcvtDWInst(uint32_t raw)
    : RType("fcvt.d.w", raw, make_exec_fn<FcvtDWInst>()), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  num_src_ = 1;
  num_dst_ = 1;
}

void FcvtDWInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  int32_t val = static_cast<int32_t>(h->read_xreg(rs1.encoding_value_));
  double result = static_cast<double>(val);
  h->write_freg(rd.encoding_value_, std::bit_cast<uint64_t>(result));
}

FcvtDWuInst::FcvtDWuInst(uint32_t raw)
    : RType("fcvt.d.wu", raw, make_exec_fn<FcvtDWuInst>()), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  num_src_ = 1;
  num_dst_ = 1;
}

void FcvtDWuInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  uint32_t val = static_cast<uint32_t>(h->read_xreg(rs1.encoding_value_));
  double result = static_cast<double>(val);
  h->write_freg(rd.encoding_value_, std::bit_cast<uint64_t>(result));
}

FcvtDLInst::FcvtDLInst(uint32_t raw)
    : RType("fcvt.d.l", raw, make_exec_fn<FcvtDLInst>()), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  num_src_ = 1;
  num_dst_ = 1;
}

void FcvtDLInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  int64_t val = h->read_xreg(rs1.encoding_value_);
  double result = static_cast<double>(val);
  h->write_freg(rd.encoding_value_, std::bit_cast<uint64_t>(result));
}

FcvtDLuInst::FcvtDLuInst(uint32_t raw)
    : RType("fcvt.d.lu", raw, make_exec_fn<FcvtDLuInst>()), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  num_src_ = 1;
  num_dst_ = 1;
}

void FcvtDLuInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  uint64_t val = static_cast<uint64_t>(h->read_xreg(rs1.encoding_value_));
  double result = static_cast<double>(val);
  h->write_freg(rd.encoding_value_, std::bit_cast<uint64_t>(result));
}

// R-type FP-FP conversion instructions

FcvtSDInst::FcvtSDInst(uint32_t raw)
    : RType("fcvt.s.d", raw, make_exec_fn<FcvtSDInst>()), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  num_src_ = 1;
  num_dst_ = 1;
}

void FcvtSDInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  double d = std::bit_cast<double>(h->read_freg(rs1.encoding_value_));
  float f = static_cast<float>(d);
  h->write_freg(rd.encoding_value_, nan_box(std::bit_cast<uint32_t>(f)));
}

FcvtDSInst::FcvtDSInst(uint32_t raw)
    : RType("fcvt.d.s", raw, make_exec_fn<FcvtDSInst>()), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  num_src_ = 1;
  num_dst_ = 1;
}

void FcvtDSInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  uint32_t bits = unbox(h->read_freg(rs1.encoding_value_));
  float f = std::bit_cast<float>(bits);
  double d = static_cast<double>(f);
  h->write_freg(rd.encoding_value_, std::bit_cast<uint64_t>(d));
}

// R-type move/classify (FPR -> GPR)

FmvXDInst::FmvXDInst(uint32_t raw)
    : RType("fmv.x.d", raw, make_exec_fn<FmvXDInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  num_src_ = 1;
  num_dst_ = 1;
}

void FmvXDInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(h->read_freg(rs1.encoding_value_)));
}

FclassDInst::FclassDInst(uint32_t raw)
    : RType("fclass.d", raw, make_exec_fn<FclassDInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  num_src_ = 1;
  num_dst_ = 1;
}

void FclassDInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  uint64_t bits = h->read_freg(rs1.encoding_value_);

  uint64_t result = 0;
  bool sign = (bits >> 63) != 0;
  uint64_t exp = (bits >> 52) & 0x7FF;
  uint64_t frac = bits & 0x000FFFFFFFFFFFFFULL;

  if (exp == 0x7FF && frac != 0) {
    // NaN
    if (frac & 0x0008000000000000ULL) {
      result = 1 << 9; // quiet NaN
    } else {
      result = 1 << 8; // signaling NaN
    }
  } else if (exp == 0x7FF && frac == 0) {
    // infinity
    result = sign ? (1 << 0) : (1 << 7);
  } else if (exp == 0 && frac == 0) {
    // zero
    result = sign ? (1 << 3) : (1 << 4);
  } else if (exp == 0 && frac != 0) {
    // subnormal
    result = sign ? (1 << 2) : (1 << 5);
  } else {
    // normal
    result = sign ? (1 << 1) : (1 << 6);
  }

  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(result));
}

// R-type move (GPR -> FPR)

FmvDXInst::FmvDXInst(uint32_t raw)
    : RType("fmv.d.x", raw, make_exec_fn<FmvDXInst>()), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  num_src_ = 1;
  num_dst_ = 1;
}

void FmvDXInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  h->write_freg(rd.encoding_value_, static_cast<uint64_t>(h->read_xreg(rs1.encoding_value_)));
}

// R-type FP compare instructions (FPR -> GPR)

FeqDInst::FeqDInst(uint32_t raw)
    : RType("feq.d", raw, make_exec_fn<FeqDInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1), rs2(64, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}

void FeqDInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  double d1 = std::bit_cast<double>(h->read_freg(rs1.encoding_value_));
  double d2 = std::bit_cast<double>(h->read_freg(rs2.encoding_value_));
  h->write_xreg(rd.encoding_value_, (d1 == d2) ? 1 : 0);
}

FltDInst::FltDInst(uint32_t raw)
    : RType("flt.d", raw, make_exec_fn<FltDInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1), rs2(64, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}

void FltDInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  double d1 = std::bit_cast<double>(h->read_freg(rs1.encoding_value_));
  double d2 = std::bit_cast<double>(h->read_freg(rs2.encoding_value_));
  h->write_xreg(rd.encoding_value_, (d1 < d2) ? 1 : 0);
}

FleDInst::FleDInst(uint32_t raw)
    : RType("fle.d", raw, make_exec_fn<FleDInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1), rs2(64, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}

void FleDInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  double d1 = std::bit_cast<double>(h->read_freg(rs1.encoding_value_));
  double d2 = std::bit_cast<double>(h->read_freg(rs2.encoding_value_));
  h->write_xreg(rd.encoding_value_, (d1 <= d2) ? 1 : 0);
}

// R4-type fused multiply-add instructions

FmaddDInst::FmaddDInst(uint32_t raw)
    : R4Type("fmadd.d", raw, make_exec_fn<FmaddDInst>()), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1), rs2(64, OperandType::OPR_FPR, inst_.rs2),
      rs3(64, OperandType::OPR_FPR, inst_.rs3) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  src_operands_[2] = &rs3;
  num_src_ = 3;
  num_dst_ = 1;
}

void FmaddDInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  double d1 = std::bit_cast<double>(h->read_freg(rs1.encoding_value_));
  double d2 = std::bit_cast<double>(h->read_freg(rs2.encoding_value_));
  double d3 = std::bit_cast<double>(h->read_freg(rs3.encoding_value_));
  double result = std::fma(d1, d2, d3);
  h->write_freg(rd.encoding_value_, std::bit_cast<uint64_t>(result));
}

FmsubDInst::FmsubDInst(uint32_t raw)
    : R4Type("fmsub.d", raw, make_exec_fn<FmsubDInst>()), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1), rs2(64, OperandType::OPR_FPR, inst_.rs2),
      rs3(64, OperandType::OPR_FPR, inst_.rs3) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  src_operands_[2] = &rs3;
  num_src_ = 3;
  num_dst_ = 1;
}

void FmsubDInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  double d1 = std::bit_cast<double>(h->read_freg(rs1.encoding_value_));
  double d2 = std::bit_cast<double>(h->read_freg(rs2.encoding_value_));
  double d3 = std::bit_cast<double>(h->read_freg(rs3.encoding_value_));
  double result = std::fma(d1, d2, -d3);
  h->write_freg(rd.encoding_value_, std::bit_cast<uint64_t>(result));
}

FnmsubDInst::FnmsubDInst(uint32_t raw)
    : R4Type("fnmsub.d", raw, make_exec_fn<FnmsubDInst>()), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1), rs2(64, OperandType::OPR_FPR, inst_.rs2),
      rs3(64, OperandType::OPR_FPR, inst_.rs3) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  src_operands_[2] = &rs3;
  num_src_ = 3;
  num_dst_ = 1;
}

void FnmsubDInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  double d1 = std::bit_cast<double>(h->read_freg(rs1.encoding_value_));
  double d2 = std::bit_cast<double>(h->read_freg(rs2.encoding_value_));
  double d3 = std::bit_cast<double>(h->read_freg(rs3.encoding_value_));
  // FNMSUB.D: -(rs1*rs2) + rs3 = fma(-rs1, rs2, rs3)
  double result = std::fma(-d1, d2, d3);
  h->write_freg(rd.encoding_value_, std::bit_cast<uint64_t>(result));
}

FnmaddDInst::FnmaddDInst(uint32_t raw)
    : R4Type("fnmadd.d", raw, make_exec_fn<FnmaddDInst>()), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1), rs2(64, OperandType::OPR_FPR, inst_.rs2),
      rs3(64, OperandType::OPR_FPR, inst_.rs3) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  src_operands_[2] = &rs3;
  num_src_ = 3;
  num_dst_ = 1;
}

void FnmaddDInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  double d1 = std::bit_cast<double>(h->read_freg(rs1.encoding_value_));
  double d2 = std::bit_cast<double>(h->read_freg(rs2.encoding_value_));
  double d3 = std::bit_cast<double>(h->read_freg(rs3.encoding_value_));
  // FNMADD.D: -(rs1*rs2) - rs3 = -fma(rs1, rs2, rs3)
  double result = -std::fma(d1, d2, d3);
  h->write_freg(rd.encoding_value_, std::bit_cast<uint64_t>(result));
}

} // namespace detail
} // namespace risc_v
} // namespace rocjitsu

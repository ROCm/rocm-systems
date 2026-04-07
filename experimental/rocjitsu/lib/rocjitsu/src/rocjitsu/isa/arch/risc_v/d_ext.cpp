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
    : IType("fld", raw), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), offset(12, OperandType::OPR_IMM, imm()) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&offset);
  src_operands_.emplace_back(&rs1);
}

void FldInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = static_cast<uint64_t>(h->read_xreg(rs1.encoding_value_) + imm());
  h->write_freg(rd.encoding_value_, m->read64(addr));
}

// S-type FP store instruction

FsdInst::FsdInst(uint32_t raw)
    : SType("fsd", raw), rs2_op(64, OperandType::OPR_FPR, inst_.rs2),
      rs1_op(64, OperandType::OPR_GPR, inst_.rs1), offset(12, OperandType::OPR_IMM, imm()) {
  src_operands_.emplace_back(&rs2_op);
  src_operands_.emplace_back(&offset);
  src_operands_.emplace_back(&rs1_op);
}

void FsdInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = static_cast<uint64_t>(h->read_xreg(rs1_op.encoding_value_) + imm());
  m->write64(addr, h->read_freg(rs2_op.encoding_value_));
}

// R-type FP compute instructions (FPR -> FPR)

FaddDInst::FaddDInst(uint32_t raw)
    : RType("fadd.d", raw), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1), rs2(64, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}

void FaddDInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  double d1 = std::bit_cast<double>(h->read_freg(rs1.encoding_value_));
  double d2 = std::bit_cast<double>(h->read_freg(rs2.encoding_value_));
  double result = d1 + d2;
  h->write_freg(rd.encoding_value_, std::bit_cast<uint64_t>(result));
}

FsubDInst::FsubDInst(uint32_t raw)
    : RType("fsub.d", raw), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1), rs2(64, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}

void FsubDInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  double d1 = std::bit_cast<double>(h->read_freg(rs1.encoding_value_));
  double d2 = std::bit_cast<double>(h->read_freg(rs2.encoding_value_));
  double result = d1 - d2;
  h->write_freg(rd.encoding_value_, std::bit_cast<uint64_t>(result));
}

FmulDInst::FmulDInst(uint32_t raw)
    : RType("fmul.d", raw), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1), rs2(64, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}

void FmulDInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  double d1 = std::bit_cast<double>(h->read_freg(rs1.encoding_value_));
  double d2 = std::bit_cast<double>(h->read_freg(rs2.encoding_value_));
  double result = d1 * d2;
  h->write_freg(rd.encoding_value_, std::bit_cast<uint64_t>(result));
}

FdivDInst::FdivDInst(uint32_t raw)
    : RType("fdiv.d", raw), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1), rs2(64, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}

void FdivDInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  double d1 = std::bit_cast<double>(h->read_freg(rs1.encoding_value_));
  double d2 = std::bit_cast<double>(h->read_freg(rs2.encoding_value_));
  double result = d1 / d2;
  h->write_freg(rd.encoding_value_, std::bit_cast<uint64_t>(result));
}

FsgnjDInst::FsgnjDInst(uint32_t raw)
    : RType("fsgnj.d", raw), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1), rs2(64, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}

void FsgnjDInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  uint64_t b1 = h->read_freg(rs1.encoding_value_);
  uint64_t b2 = h->read_freg(rs2.encoding_value_);
  uint64_t result = (b1 & 0x7FFFFFFFFFFFFFFFULL) | (b2 & 0x8000000000000000ULL);
  h->write_freg(rd.encoding_value_, result);
}

FsgnjnDInst::FsgnjnDInst(uint32_t raw)
    : RType("fsgnjn.d", raw), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1), rs2(64, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}

void FsgnjnDInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  uint64_t b1 = h->read_freg(rs1.encoding_value_);
  uint64_t b2 = h->read_freg(rs2.encoding_value_);
  uint64_t result = (b1 & 0x7FFFFFFFFFFFFFFFULL) | (~b2 & 0x8000000000000000ULL);
  h->write_freg(rd.encoding_value_, result);
}

FsgnjxDInst::FsgnjxDInst(uint32_t raw)
    : RType("fsgnjx.d", raw), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1), rs2(64, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}

void FsgnjxDInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  uint64_t b1 = h->read_freg(rs1.encoding_value_);
  uint64_t b2 = h->read_freg(rs2.encoding_value_);
  uint64_t result = b1 ^ (b2 & 0x8000000000000000ULL);
  h->write_freg(rd.encoding_value_, result);
}

FminDInst::FminDInst(uint32_t raw)
    : RType("fmin.d", raw), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1), rs2(64, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}

void FminDInst::execute(HartState &ctx) {
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
    : RType("fmax.d", raw), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1), rs2(64, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}

void FmaxDInst::execute(HartState &ctx) {
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
    : RType("fsqrt.d", raw), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
}

void FsqrtDInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  double d1 = std::bit_cast<double>(h->read_freg(rs1.encoding_value_));
  double result = std::sqrt(d1);
  h->write_freg(rd.encoding_value_, std::bit_cast<uint64_t>(result));
}

// R-type FP-to-int conversions (FPR -> GPR)

FcvtWDInst::FcvtWDInst(uint32_t raw)
    : RType("fcvt.w.d", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
}

void FcvtWDInst::execute(HartState &ctx) {
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
    : RType("fcvt.wu.d", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
}

void FcvtWuDInst::execute(HartState &ctx) {
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
    : RType("fcvt.l.d", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
}

void FcvtLDInst::execute(HartState &ctx) {
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
    : RType("fcvt.lu.d", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
}

void FcvtLuDInst::execute(HartState &ctx) {
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
    : RType("fcvt.d.w", raw), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
}

void FcvtDWInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  int32_t val = static_cast<int32_t>(h->read_xreg(rs1.encoding_value_));
  double result = static_cast<double>(val);
  h->write_freg(rd.encoding_value_, std::bit_cast<uint64_t>(result));
}

FcvtDWuInst::FcvtDWuInst(uint32_t raw)
    : RType("fcvt.d.wu", raw), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
}

void FcvtDWuInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  uint32_t val = static_cast<uint32_t>(h->read_xreg(rs1.encoding_value_));
  double result = static_cast<double>(val);
  h->write_freg(rd.encoding_value_, std::bit_cast<uint64_t>(result));
}

FcvtDLInst::FcvtDLInst(uint32_t raw)
    : RType("fcvt.d.l", raw), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
}

void FcvtDLInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  int64_t val = h->read_xreg(rs1.encoding_value_);
  double result = static_cast<double>(val);
  h->write_freg(rd.encoding_value_, std::bit_cast<uint64_t>(result));
}

FcvtDLuInst::FcvtDLuInst(uint32_t raw)
    : RType("fcvt.d.lu", raw), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
}

void FcvtDLuInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  uint64_t val = static_cast<uint64_t>(h->read_xreg(rs1.encoding_value_));
  double result = static_cast<double>(val);
  h->write_freg(rd.encoding_value_, std::bit_cast<uint64_t>(result));
}

// R-type FP-FP conversion instructions

FcvtSDInst::FcvtSDInst(uint32_t raw)
    : RType("fcvt.s.d", raw), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
}

void FcvtSDInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  double d = std::bit_cast<double>(h->read_freg(rs1.encoding_value_));
  float f = static_cast<float>(d);
  h->write_freg(rd.encoding_value_, nan_box(std::bit_cast<uint32_t>(f)));
}

FcvtDSInst::FcvtDSInst(uint32_t raw)
    : RType("fcvt.d.s", raw), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
}

void FcvtDSInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  uint32_t bits = unbox(h->read_freg(rs1.encoding_value_));
  float f = std::bit_cast<float>(bits);
  double d = static_cast<double>(f);
  h->write_freg(rd.encoding_value_, std::bit_cast<uint64_t>(d));
}

// R-type move/classify (FPR -> GPR)

FmvXDInst::FmvXDInst(uint32_t raw)
    : RType("fmv.x.d", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
}

void FmvXDInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(h->read_freg(rs1.encoding_value_)));
}

FclassDInst::FclassDInst(uint32_t raw)
    : RType("fclass.d", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
}

void FclassDInst::execute(HartState &ctx) {
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
    : RType("fmv.d.x", raw), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
}

void FmvDXInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  h->write_freg(rd.encoding_value_, static_cast<uint64_t>(h->read_xreg(rs1.encoding_value_)));
}

// R-type FP compare instructions (FPR -> GPR)

FeqDInst::FeqDInst(uint32_t raw)
    : RType("feq.d", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1), rs2(64, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}

void FeqDInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  double d1 = std::bit_cast<double>(h->read_freg(rs1.encoding_value_));
  double d2 = std::bit_cast<double>(h->read_freg(rs2.encoding_value_));
  h->write_xreg(rd.encoding_value_, (d1 == d2) ? 1 : 0);
}

FltDInst::FltDInst(uint32_t raw)
    : RType("flt.d", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1), rs2(64, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}

void FltDInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  double d1 = std::bit_cast<double>(h->read_freg(rs1.encoding_value_));
  double d2 = std::bit_cast<double>(h->read_freg(rs2.encoding_value_));
  h->write_xreg(rd.encoding_value_, (d1 < d2) ? 1 : 0);
}

FleDInst::FleDInst(uint32_t raw)
    : RType("fle.d", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1), rs2(64, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}

void FleDInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  double d1 = std::bit_cast<double>(h->read_freg(rs1.encoding_value_));
  double d2 = std::bit_cast<double>(h->read_freg(rs2.encoding_value_));
  h->write_xreg(rd.encoding_value_, (d1 <= d2) ? 1 : 0);
}

// R4-type fused multiply-add instructions

FmaddDInst::FmaddDInst(uint32_t raw)
    : R4Type("fmadd.d", raw), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1), rs2(64, OperandType::OPR_FPR, inst_.rs2),
      rs3(64, OperandType::OPR_FPR, inst_.rs3) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
  src_operands_.emplace_back(&rs3);
}

void FmaddDInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  double d1 = std::bit_cast<double>(h->read_freg(rs1.encoding_value_));
  double d2 = std::bit_cast<double>(h->read_freg(rs2.encoding_value_));
  double d3 = std::bit_cast<double>(h->read_freg(rs3.encoding_value_));
  double result = std::fma(d1, d2, d3);
  h->write_freg(rd.encoding_value_, std::bit_cast<uint64_t>(result));
}

FmsubDInst::FmsubDInst(uint32_t raw)
    : R4Type("fmsub.d", raw), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1), rs2(64, OperandType::OPR_FPR, inst_.rs2),
      rs3(64, OperandType::OPR_FPR, inst_.rs3) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
  src_operands_.emplace_back(&rs3);
}

void FmsubDInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  double d1 = std::bit_cast<double>(h->read_freg(rs1.encoding_value_));
  double d2 = std::bit_cast<double>(h->read_freg(rs2.encoding_value_));
  double d3 = std::bit_cast<double>(h->read_freg(rs3.encoding_value_));
  double result = std::fma(d1, d2, -d3);
  h->write_freg(rd.encoding_value_, std::bit_cast<uint64_t>(result));
}

FnmsubDInst::FnmsubDInst(uint32_t raw)
    : R4Type("fnmsub.d", raw), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1), rs2(64, OperandType::OPR_FPR, inst_.rs2),
      rs3(64, OperandType::OPR_FPR, inst_.rs3) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
  src_operands_.emplace_back(&rs3);
}

void FnmsubDInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  double d1 = std::bit_cast<double>(h->read_freg(rs1.encoding_value_));
  double d2 = std::bit_cast<double>(h->read_freg(rs2.encoding_value_));
  double d3 = std::bit_cast<double>(h->read_freg(rs3.encoding_value_));
  // FNMSUB.D: -(rs1*rs2) + rs3 = fma(-rs1, rs2, rs3)
  double result = std::fma(-d1, d2, d3);
  h->write_freg(rd.encoding_value_, std::bit_cast<uint64_t>(result));
}

FnmaddDInst::FnmaddDInst(uint32_t raw)
    : R4Type("fnmadd.d", raw), rd(64, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_FPR, inst_.rs1), rs2(64, OperandType::OPR_FPR, inst_.rs2),
      rs3(64, OperandType::OPR_FPR, inst_.rs3) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
  src_operands_.emplace_back(&rs3);
}

void FnmaddDInst::execute(HartState &ctx) {
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

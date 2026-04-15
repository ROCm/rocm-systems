// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/risc_v/f_ext.h"
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

FlwInst::FlwInst(uint32_t raw)
    : IType("flw", raw), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), offset(12, OperandType::OPR_IMM, imm()) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&offset);
  src_operands_.emplace_back(&rs1);
}

void FlwInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = static_cast<uint64_t>(h->read_xreg(rs1.encoding_value_) + imm());
  uint32_t bits = m->read32(addr);
  h->write_freg(rd.encoding_value_, nan_box(bits));
}

// S-type FP store instruction

FswInst::FswInst(uint32_t raw)
    : SType("fsw", raw), rs2_op(32, OperandType::OPR_FPR, inst_.rs2),
      rs1_op(64, OperandType::OPR_GPR, inst_.rs1), offset(12, OperandType::OPR_IMM, imm()) {
  src_operands_.emplace_back(&rs2_op);
  src_operands_.emplace_back(&offset);
  src_operands_.emplace_back(&rs1_op);
}

void FswInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = static_cast<uint64_t>(h->read_xreg(rs1_op.encoding_value_) + imm());
  m->write32(addr, unbox(h->read_freg(rs2_op.encoding_value_)));
}

// R-type FP compute instructions (FPR -> FPR)

FaddSInst::FaddSInst(uint32_t raw)
    : RType("fadd.s", raw), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1), rs2(32, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}

void FaddSInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  float f1 = std::bit_cast<float>(unbox(h->read_freg(rs1.encoding_value_)));
  float f2 = std::bit_cast<float>(unbox(h->read_freg(rs2.encoding_value_)));
  float result = f1 + f2;
  h->write_freg(rd.encoding_value_, nan_box(std::bit_cast<uint32_t>(result)));
}

FsubSInst::FsubSInst(uint32_t raw)
    : RType("fsub.s", raw), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1), rs2(32, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}

void FsubSInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  float f1 = std::bit_cast<float>(unbox(h->read_freg(rs1.encoding_value_)));
  float f2 = std::bit_cast<float>(unbox(h->read_freg(rs2.encoding_value_)));
  float result = f1 - f2;
  h->write_freg(rd.encoding_value_, nan_box(std::bit_cast<uint32_t>(result)));
}

FmulSInst::FmulSInst(uint32_t raw)
    : RType("fmul.s", raw), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1), rs2(32, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}

void FmulSInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  float f1 = std::bit_cast<float>(unbox(h->read_freg(rs1.encoding_value_)));
  float f2 = std::bit_cast<float>(unbox(h->read_freg(rs2.encoding_value_)));
  float result = f1 * f2;
  h->write_freg(rd.encoding_value_, nan_box(std::bit_cast<uint32_t>(result)));
}

FdivSInst::FdivSInst(uint32_t raw)
    : RType("fdiv.s", raw), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1), rs2(32, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}

void FdivSInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  float f1 = std::bit_cast<float>(unbox(h->read_freg(rs1.encoding_value_)));
  float f2 = std::bit_cast<float>(unbox(h->read_freg(rs2.encoding_value_)));
  float result = f1 / f2;
  h->write_freg(rd.encoding_value_, nan_box(std::bit_cast<uint32_t>(result)));
}

FsgnjSInst::FsgnjSInst(uint32_t raw)
    : RType("fsgnj.s", raw), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1), rs2(32, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}

void FsgnjSInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  uint32_t b1 = unbox(h->read_freg(rs1.encoding_value_));
  uint32_t b2 = unbox(h->read_freg(rs2.encoding_value_));
  uint32_t result = (b1 & 0x7FFFFFFFU) | (b2 & 0x80000000U);
  h->write_freg(rd.encoding_value_, nan_box(result));
}

FsgnjnSInst::FsgnjnSInst(uint32_t raw)
    : RType("fsgnjn.s", raw), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1), rs2(32, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}

void FsgnjnSInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  uint32_t b1 = unbox(h->read_freg(rs1.encoding_value_));
  uint32_t b2 = unbox(h->read_freg(rs2.encoding_value_));
  uint32_t result = (b1 & 0x7FFFFFFFU) | (~b2 & 0x80000000U);
  h->write_freg(rd.encoding_value_, nan_box(result));
}

FsgnjxSInst::FsgnjxSInst(uint32_t raw)
    : RType("fsgnjx.s", raw), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1), rs2(32, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}

void FsgnjxSInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  uint32_t b1 = unbox(h->read_freg(rs1.encoding_value_));
  uint32_t b2 = unbox(h->read_freg(rs2.encoding_value_));
  uint32_t result = b1 ^ (b2 & 0x80000000U);
  h->write_freg(rd.encoding_value_, nan_box(result));
}

FminSInst::FminSInst(uint32_t raw)
    : RType("fmin.s", raw), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1), rs2(32, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}

void FminSInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  float f1 = std::bit_cast<float>(unbox(h->read_freg(rs1.encoding_value_)));
  float f2 = std::bit_cast<float>(unbox(h->read_freg(rs2.encoding_value_)));
  float result;
  if (std::isnan(f1) && std::isnan(f2)) {
    result = std::bit_cast<float>(uint32_t{0x7FC00000U}); // canonical NaN
  } else if (std::isnan(f1)) {
    result = f2;
  } else if (std::isnan(f2)) {
    result = f1;
  } else {
    // -0.0 is less than +0.0 per RISC-V spec
    if (f1 == f2) {
      uint32_t b1 = std::bit_cast<uint32_t>(f1);
      result = (b1 & 0x80000000U) ? f1 : f2;
    } else {
      result = (f1 < f2) ? f1 : f2;
    }
  }
  h->write_freg(rd.encoding_value_, nan_box(std::bit_cast<uint32_t>(result)));
}

FmaxSInst::FmaxSInst(uint32_t raw)
    : RType("fmax.s", raw), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1), rs2(32, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}

void FmaxSInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  float f1 = std::bit_cast<float>(unbox(h->read_freg(rs1.encoding_value_)));
  float f2 = std::bit_cast<float>(unbox(h->read_freg(rs2.encoding_value_)));
  float result;
  if (std::isnan(f1) && std::isnan(f2)) {
    result = std::bit_cast<float>(uint32_t{0x7FC00000U}); // canonical NaN
  } else if (std::isnan(f1)) {
    result = f2;
  } else if (std::isnan(f2)) {
    result = f1;
  } else {
    // +0.0 is greater than -0.0 per RISC-V spec
    if (f1 == f2) {
      uint32_t b1 = std::bit_cast<uint32_t>(f1);
      result = (b1 & 0x80000000U) ? f2 : f1;
    } else {
      result = (f1 > f2) ? f1 : f2;
    }
  }
  h->write_freg(rd.encoding_value_, nan_box(std::bit_cast<uint32_t>(result)));
}

// R-type FP unary instruction (single source)

FsqrtSInst::FsqrtSInst(uint32_t raw)
    : RType("fsqrt.s", raw), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
}

void FsqrtSInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  float f1 = std::bit_cast<float>(unbox(h->read_freg(rs1.encoding_value_)));
  float result = std::sqrt(f1);
  h->write_freg(rd.encoding_value_, nan_box(std::bit_cast<uint32_t>(result)));
}

// R-type FP-to-int conversions (FPR -> GPR)

FcvtWSInst::FcvtWSInst(uint32_t raw)
    : RType("fcvt.w.s", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
}

void FcvtWSInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  float f1 = std::bit_cast<float>(unbox(h->read_freg(rs1.encoding_value_)));
  int32_t result;
  if (std::isnan(f1)) {
    result = INT32_MAX;
  } else if (f1 >= static_cast<float>(INT32_MAX)) {
    result = INT32_MAX;
  } else if (f1 <= static_cast<float>(INT32_MIN)) {
    result = INT32_MIN;
  } else {
    result = static_cast<int32_t>(f1);
  }
  h->write_xreg(rd.encoding_value_, sext32(result));
}

FcvtWuSInst::FcvtWuSInst(uint32_t raw)
    : RType("fcvt.wu.s", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
}

void FcvtWuSInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  float f1 = std::bit_cast<float>(unbox(h->read_freg(rs1.encoding_value_)));
  uint32_t result;
  if (std::isnan(f1)) {
    result = UINT32_MAX;
  } else if (f1 >= static_cast<float>(UINT32_MAX)) {
    result = UINT32_MAX;
  } else if (f1 <= 0.0f) {
    result = 0;
  } else {
    result = static_cast<uint32_t>(f1);
  }
  // Per RISC-V spec: sign-extend the 32-bit result to 64 bits
  h->write_xreg(rd.encoding_value_, sext32(static_cast<int32_t>(result)));
}

FcvtLSInst::FcvtLSInst(uint32_t raw)
    : RType("fcvt.l.s", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
}

void FcvtLSInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  float f1 = std::bit_cast<float>(unbox(h->read_freg(rs1.encoding_value_)));
  int64_t result;
  if (std::isnan(f1)) {
    result = INT64_MAX;
  } else if (f1 >= static_cast<float>(INT64_MAX)) {
    result = INT64_MAX;
  } else if (f1 <= static_cast<float>(INT64_MIN)) {
    result = INT64_MIN;
  } else {
    result = static_cast<int64_t>(f1);
  }
  h->write_xreg(rd.encoding_value_, result);
}

FcvtLuSInst::FcvtLuSInst(uint32_t raw)
    : RType("fcvt.lu.s", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
}

void FcvtLuSInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  float f1 = std::bit_cast<float>(unbox(h->read_freg(rs1.encoding_value_)));
  uint64_t result;
  if (std::isnan(f1)) {
    result = UINT64_MAX;
  } else if (f1 >= static_cast<float>(UINT64_MAX)) {
    result = UINT64_MAX;
  } else if (f1 <= 0.0f) {
    result = 0;
  } else {
    result = static_cast<uint64_t>(f1);
  }
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(result));
}

// R-type int-to-FP conversions (GPR -> FPR)

FcvtSWInst::FcvtSWInst(uint32_t raw)
    : RType("fcvt.s.w", raw), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
}

void FcvtSWInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  int32_t val = static_cast<int32_t>(h->read_xreg(rs1.encoding_value_));
  float result = static_cast<float>(val);
  h->write_freg(rd.encoding_value_, nan_box(std::bit_cast<uint32_t>(result)));
}

FcvtSWuInst::FcvtSWuInst(uint32_t raw)
    : RType("fcvt.s.wu", raw), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
}

void FcvtSWuInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  uint32_t val = static_cast<uint32_t>(h->read_xreg(rs1.encoding_value_));
  float result = static_cast<float>(val);
  h->write_freg(rd.encoding_value_, nan_box(std::bit_cast<uint32_t>(result)));
}

FcvtSLInst::FcvtSLInst(uint32_t raw)
    : RType("fcvt.s.l", raw), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
}

void FcvtSLInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  int64_t val = h->read_xreg(rs1.encoding_value_);
  float result = static_cast<float>(val);
  h->write_freg(rd.encoding_value_, nan_box(std::bit_cast<uint32_t>(result)));
}

FcvtSLuInst::FcvtSLuInst(uint32_t raw)
    : RType("fcvt.s.lu", raw), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
}

void FcvtSLuInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  uint64_t val = static_cast<uint64_t>(h->read_xreg(rs1.encoding_value_));
  float result = static_cast<float>(val);
  h->write_freg(rd.encoding_value_, nan_box(std::bit_cast<uint32_t>(result)));
}

// R-type move/classify (FPR -> GPR)

FmvXWInst::FmvXWInst(uint32_t raw)
    : RType("fmv.x.w", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
}

void FmvXWInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  uint32_t bits = unbox(h->read_freg(rs1.encoding_value_));
  // Sign-extend the 32-bit value to 64 bits per RISC-V spec
  h->write_xreg(rd.encoding_value_, sext32(static_cast<int32_t>(bits)));
}

FclassSInst::FclassSInst(uint32_t raw)
    : RType("fclass.s", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
}

void FclassSInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  uint32_t bits = unbox(h->read_freg(rs1.encoding_value_));

  uint64_t result = 0;
  bool sign = (bits >> 31) != 0;
  uint32_t exp = (bits >> 23) & 0xFF;
  uint32_t frac = bits & 0x007FFFFFU;

  if (exp == 0xFF && frac != 0) {
    // NaN
    if (frac & 0x00400000U) {
      result = 1 << 9; // quiet NaN
    } else {
      result = 1 << 8; // signaling NaN
    }
  } else if (exp == 0xFF && frac == 0) {
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

FmvWXInst::FmvWXInst(uint32_t raw)
    : RType("fmv.w.x", raw), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
}

void FmvWXInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  uint32_t bits = static_cast<uint32_t>(h->read_xreg(rs1.encoding_value_));
  h->write_freg(rd.encoding_value_, nan_box(bits));
}

// R-type FP compare instructions (FPR -> GPR)

FeqSInst::FeqSInst(uint32_t raw)
    : RType("feq.s", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1), rs2(32, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}

void FeqSInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  float f1 = std::bit_cast<float>(unbox(h->read_freg(rs1.encoding_value_)));
  float f2 = std::bit_cast<float>(unbox(h->read_freg(rs2.encoding_value_)));
  h->write_xreg(rd.encoding_value_, (f1 == f2) ? 1 : 0);
}

FltSInst::FltSInst(uint32_t raw)
    : RType("flt.s", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1), rs2(32, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}

void FltSInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  float f1 = std::bit_cast<float>(unbox(h->read_freg(rs1.encoding_value_)));
  float f2 = std::bit_cast<float>(unbox(h->read_freg(rs2.encoding_value_)));
  h->write_xreg(rd.encoding_value_, (f1 < f2) ? 1 : 0);
}

FleSInst::FleSInst(uint32_t raw)
    : RType("fle.s", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1), rs2(32, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}

void FleSInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  float f1 = std::bit_cast<float>(unbox(h->read_freg(rs1.encoding_value_)));
  float f2 = std::bit_cast<float>(unbox(h->read_freg(rs2.encoding_value_)));
  h->write_xreg(rd.encoding_value_, (f1 <= f2) ? 1 : 0);
}

// R4-type fused multiply-add instructions

FmaddSInst::FmaddSInst(uint32_t raw)
    : R4Type("fmadd.s", raw), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1), rs2(32, OperandType::OPR_FPR, inst_.rs2),
      rs3(32, OperandType::OPR_FPR, inst_.rs3) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
  src_operands_.emplace_back(&rs3);
}

void FmaddSInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  float f1 = std::bit_cast<float>(unbox(h->read_freg(rs1.encoding_value_)));
  float f2 = std::bit_cast<float>(unbox(h->read_freg(rs2.encoding_value_)));
  float f3 = std::bit_cast<float>(unbox(h->read_freg(rs3.encoding_value_)));
  float result = std::fma(f1, f2, f3);
  h->write_freg(rd.encoding_value_, nan_box(std::bit_cast<uint32_t>(result)));
}

FmsubSInst::FmsubSInst(uint32_t raw)
    : R4Type("fmsub.s", raw), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1), rs2(32, OperandType::OPR_FPR, inst_.rs2),
      rs3(32, OperandType::OPR_FPR, inst_.rs3) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
  src_operands_.emplace_back(&rs3);
}

void FmsubSInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  float f1 = std::bit_cast<float>(unbox(h->read_freg(rs1.encoding_value_)));
  float f2 = std::bit_cast<float>(unbox(h->read_freg(rs2.encoding_value_)));
  float f3 = std::bit_cast<float>(unbox(h->read_freg(rs3.encoding_value_)));
  float result = std::fma(f1, f2, -f3);
  h->write_freg(rd.encoding_value_, nan_box(std::bit_cast<uint32_t>(result)));
}

FnmsubSInst::FnmsubSInst(uint32_t raw)
    : R4Type("fnmsub.s", raw), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1), rs2(32, OperandType::OPR_FPR, inst_.rs2),
      rs3(32, OperandType::OPR_FPR, inst_.rs3) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
  src_operands_.emplace_back(&rs3);
}

void FnmsubSInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  float f1 = std::bit_cast<float>(unbox(h->read_freg(rs1.encoding_value_)));
  float f2 = std::bit_cast<float>(unbox(h->read_freg(rs2.encoding_value_)));
  float f3 = std::bit_cast<float>(unbox(h->read_freg(rs3.encoding_value_)));
  // FNMSUB.S: -(rs1*rs2) + rs3 = fma(-rs1, rs2, rs3)
  float result = std::fma(-f1, f2, f3);
  h->write_freg(rd.encoding_value_, nan_box(std::bit_cast<uint32_t>(result)));
}

FnmaddSInst::FnmaddSInst(uint32_t raw)
    : R4Type("fnmadd.s", raw), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1), rs2(32, OperandType::OPR_FPR, inst_.rs2),
      rs3(32, OperandType::OPR_FPR, inst_.rs3) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
  src_operands_.emplace_back(&rs3);
}

void FnmaddSInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  float f1 = std::bit_cast<float>(unbox(h->read_freg(rs1.encoding_value_)));
  float f2 = std::bit_cast<float>(unbox(h->read_freg(rs2.encoding_value_)));
  float f3 = std::bit_cast<float>(unbox(h->read_freg(rs3.encoding_value_)));
  // FNMADD.S: -(rs1*rs2) - rs3 = -fma(rs1, rs2, rs3)
  float result = -std::fma(f1, f2, f3);
  h->write_freg(rd.encoding_value_, nan_box(std::bit_cast<uint32_t>(result)));
}

} // namespace detail
} // namespace risc_v
} // namespace rocjitsu

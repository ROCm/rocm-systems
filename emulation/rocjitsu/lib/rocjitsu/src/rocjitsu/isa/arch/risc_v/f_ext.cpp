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
    : IType("flw", raw, make_exec_fn<FlwInst>()), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), offset(12, OperandType::OPR_IMM, imm()) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &offset;
  src_operands_[1] = &rs1;
  num_src_ = 2;
  num_dst_ = 1;
}

void FlwInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = static_cast<uint64_t>(h->read_xreg(rs1.encoding_value_) + imm());
  uint32_t bits = m->read32(addr);
  h->write_freg(rd.encoding_value_, nan_box(bits));
}

// S-type FP store instruction

FswInst::FswInst(uint32_t raw)
    : SType("fsw", raw, make_exec_fn<FswInst>()), rs2_op(32, OperandType::OPR_FPR, inst_.rs2),
      rs1_op(64, OperandType::OPR_GPR, inst_.rs1), offset(12, OperandType::OPR_IMM, imm()) {
  src_operands_[0] = &rs2_op;
  src_operands_[1] = &offset;
  src_operands_[2] = &rs1_op;
  num_src_ = 3;
  num_dst_ = 0;
}

void FswInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto *m = current_memory();
  uint64_t addr = static_cast<uint64_t>(h->read_xreg(rs1_op.encoding_value_) + imm());
  m->write32(addr, unbox(h->read_freg(rs2_op.encoding_value_)));
}

// R-type FP compute instructions (FPR -> FPR)

FaddSInst::FaddSInst(uint32_t raw)
    : RType("fadd.s", raw, make_exec_fn<FaddSInst>()), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1), rs2(32, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}

void FaddSInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  float f1 = std::bit_cast<float>(unbox(h->read_freg(rs1.encoding_value_)));
  float f2 = std::bit_cast<float>(unbox(h->read_freg(rs2.encoding_value_)));
  float result = f1 + f2;
  h->write_freg(rd.encoding_value_, nan_box(std::bit_cast<uint32_t>(result)));
}

FsubSInst::FsubSInst(uint32_t raw)
    : RType("fsub.s", raw, make_exec_fn<FsubSInst>()), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1), rs2(32, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}

void FsubSInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  float f1 = std::bit_cast<float>(unbox(h->read_freg(rs1.encoding_value_)));
  float f2 = std::bit_cast<float>(unbox(h->read_freg(rs2.encoding_value_)));
  float result = f1 - f2;
  h->write_freg(rd.encoding_value_, nan_box(std::bit_cast<uint32_t>(result)));
}

FmulSInst::FmulSInst(uint32_t raw)
    : RType("fmul.s", raw, make_exec_fn<FmulSInst>()), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1), rs2(32, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}

void FmulSInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  float f1 = std::bit_cast<float>(unbox(h->read_freg(rs1.encoding_value_)));
  float f2 = std::bit_cast<float>(unbox(h->read_freg(rs2.encoding_value_)));
  float result = f1 * f2;
  h->write_freg(rd.encoding_value_, nan_box(std::bit_cast<uint32_t>(result)));
}

FdivSInst::FdivSInst(uint32_t raw)
    : RType("fdiv.s", raw, make_exec_fn<FdivSInst>()), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1), rs2(32, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}

void FdivSInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  float f1 = std::bit_cast<float>(unbox(h->read_freg(rs1.encoding_value_)));
  float f2 = std::bit_cast<float>(unbox(h->read_freg(rs2.encoding_value_)));
  float result = f1 / f2;
  h->write_freg(rd.encoding_value_, nan_box(std::bit_cast<uint32_t>(result)));
}

FsgnjSInst::FsgnjSInst(uint32_t raw)
    : RType("fsgnj.s", raw, make_exec_fn<FsgnjSInst>()), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1), rs2(32, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}

void FsgnjSInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  uint32_t b1 = unbox(h->read_freg(rs1.encoding_value_));
  uint32_t b2 = unbox(h->read_freg(rs2.encoding_value_));
  uint32_t result = (b1 & 0x7FFFFFFFU) | (b2 & 0x80000000U);
  h->write_freg(rd.encoding_value_, nan_box(result));
}

FsgnjnSInst::FsgnjnSInst(uint32_t raw)
    : RType("fsgnjn.s", raw, make_exec_fn<FsgnjnSInst>()), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1), rs2(32, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}

void FsgnjnSInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  uint32_t b1 = unbox(h->read_freg(rs1.encoding_value_));
  uint32_t b2 = unbox(h->read_freg(rs2.encoding_value_));
  uint32_t result = (b1 & 0x7FFFFFFFU) | (~b2 & 0x80000000U);
  h->write_freg(rd.encoding_value_, nan_box(result));
}

FsgnjxSInst::FsgnjxSInst(uint32_t raw)
    : RType("fsgnjx.s", raw, make_exec_fn<FsgnjxSInst>()), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1), rs2(32, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}

void FsgnjxSInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  uint32_t b1 = unbox(h->read_freg(rs1.encoding_value_));
  uint32_t b2 = unbox(h->read_freg(rs2.encoding_value_));
  uint32_t result = b1 ^ (b2 & 0x80000000U);
  h->write_freg(rd.encoding_value_, nan_box(result));
}

FminSInst::FminSInst(uint32_t raw)
    : RType("fmin.s", raw, make_exec_fn<FminSInst>()), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1), rs2(32, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}

void FminSInst::execute_impl(HartState &ctx) {
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
    : RType("fmax.s", raw, make_exec_fn<FmaxSInst>()), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1), rs2(32, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}

void FmaxSInst::execute_impl(HartState &ctx) {
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
    : RType("fsqrt.s", raw, make_exec_fn<FsqrtSInst>()), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  num_src_ = 1;
  num_dst_ = 1;
}

void FsqrtSInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  float f1 = std::bit_cast<float>(unbox(h->read_freg(rs1.encoding_value_)));
  float result = std::sqrt(f1);
  h->write_freg(rd.encoding_value_, nan_box(std::bit_cast<uint32_t>(result)));
}

// R-type FP-to-int conversions (FPR -> GPR)

FcvtWSInst::FcvtWSInst(uint32_t raw)
    : RType("fcvt.w.s", raw, make_exec_fn<FcvtWSInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  num_src_ = 1;
  num_dst_ = 1;
}

void FcvtWSInst::execute_impl(HartState &ctx) {
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
    : RType("fcvt.wu.s", raw, make_exec_fn<FcvtWuSInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  num_src_ = 1;
  num_dst_ = 1;
}

void FcvtWuSInst::execute_impl(HartState &ctx) {
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
    : RType("fcvt.l.s", raw, make_exec_fn<FcvtLSInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  num_src_ = 1;
  num_dst_ = 1;
}

void FcvtLSInst::execute_impl(HartState &ctx) {
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
    : RType("fcvt.lu.s", raw, make_exec_fn<FcvtLuSInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  num_src_ = 1;
  num_dst_ = 1;
}

void FcvtLuSInst::execute_impl(HartState &ctx) {
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
    : RType("fcvt.s.w", raw, make_exec_fn<FcvtSWInst>()), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  num_src_ = 1;
  num_dst_ = 1;
}

void FcvtSWInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  int32_t val = static_cast<int32_t>(h->read_xreg(rs1.encoding_value_));
  float result = static_cast<float>(val);
  h->write_freg(rd.encoding_value_, nan_box(std::bit_cast<uint32_t>(result)));
}

FcvtSWuInst::FcvtSWuInst(uint32_t raw)
    : RType("fcvt.s.wu", raw, make_exec_fn<FcvtSWuInst>()), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  num_src_ = 1;
  num_dst_ = 1;
}

void FcvtSWuInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  uint32_t val = static_cast<uint32_t>(h->read_xreg(rs1.encoding_value_));
  float result = static_cast<float>(val);
  h->write_freg(rd.encoding_value_, nan_box(std::bit_cast<uint32_t>(result)));
}

FcvtSLInst::FcvtSLInst(uint32_t raw)
    : RType("fcvt.s.l", raw, make_exec_fn<FcvtSLInst>()), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  num_src_ = 1;
  num_dst_ = 1;
}

void FcvtSLInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  int64_t val = h->read_xreg(rs1.encoding_value_);
  float result = static_cast<float>(val);
  h->write_freg(rd.encoding_value_, nan_box(std::bit_cast<uint32_t>(result)));
}

FcvtSLuInst::FcvtSLuInst(uint32_t raw)
    : RType("fcvt.s.lu", raw, make_exec_fn<FcvtSLuInst>()), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  num_src_ = 1;
  num_dst_ = 1;
}

void FcvtSLuInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  uint64_t val = static_cast<uint64_t>(h->read_xreg(rs1.encoding_value_));
  float result = static_cast<float>(val);
  h->write_freg(rd.encoding_value_, nan_box(std::bit_cast<uint32_t>(result)));
}

// R-type move/classify (FPR -> GPR)

FmvXWInst::FmvXWInst(uint32_t raw)
    : RType("fmv.x.w", raw, make_exec_fn<FmvXWInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  num_src_ = 1;
  num_dst_ = 1;
}

void FmvXWInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  uint32_t bits = unbox(h->read_freg(rs1.encoding_value_));
  // Sign-extend the 32-bit value to 64 bits per RISC-V spec
  h->write_xreg(rd.encoding_value_, sext32(static_cast<int32_t>(bits)));
}

FclassSInst::FclassSInst(uint32_t raw)
    : RType("fclass.s", raw, make_exec_fn<FclassSInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  num_src_ = 1;
  num_dst_ = 1;
}

void FclassSInst::execute_impl(HartState &ctx) {
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
    : RType("fmv.w.x", raw, make_exec_fn<FmvWXInst>()), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  num_src_ = 1;
  num_dst_ = 1;
}

void FmvWXInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  uint32_t bits = static_cast<uint32_t>(h->read_xreg(rs1.encoding_value_));
  h->write_freg(rd.encoding_value_, nan_box(bits));
}

// R-type FP compare instructions (FPR -> GPR)

FeqSInst::FeqSInst(uint32_t raw)
    : RType("feq.s", raw, make_exec_fn<FeqSInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1), rs2(32, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}

void FeqSInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  float f1 = std::bit_cast<float>(unbox(h->read_freg(rs1.encoding_value_)));
  float f2 = std::bit_cast<float>(unbox(h->read_freg(rs2.encoding_value_)));
  h->write_xreg(rd.encoding_value_, (f1 == f2) ? 1 : 0);
}

FltSInst::FltSInst(uint32_t raw)
    : RType("flt.s", raw, make_exec_fn<FltSInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1), rs2(32, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}

void FltSInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  float f1 = std::bit_cast<float>(unbox(h->read_freg(rs1.encoding_value_)));
  float f2 = std::bit_cast<float>(unbox(h->read_freg(rs2.encoding_value_)));
  h->write_xreg(rd.encoding_value_, (f1 < f2) ? 1 : 0);
}

FleSInst::FleSInst(uint32_t raw)
    : RType("fle.s", raw, make_exec_fn<FleSInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1), rs2(32, OperandType::OPR_FPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}

void FleSInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  float f1 = std::bit_cast<float>(unbox(h->read_freg(rs1.encoding_value_)));
  float f2 = std::bit_cast<float>(unbox(h->read_freg(rs2.encoding_value_)));
  h->write_xreg(rd.encoding_value_, (f1 <= f2) ? 1 : 0);
}

// R4-type fused multiply-add instructions

FmaddSInst::FmaddSInst(uint32_t raw)
    : R4Type("fmadd.s", raw, make_exec_fn<FmaddSInst>()), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1), rs2(32, OperandType::OPR_FPR, inst_.rs2),
      rs3(32, OperandType::OPR_FPR, inst_.rs3) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  src_operands_[2] = &rs3;
  num_src_ = 3;
  num_dst_ = 1;
}

void FmaddSInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  float f1 = std::bit_cast<float>(unbox(h->read_freg(rs1.encoding_value_)));
  float f2 = std::bit_cast<float>(unbox(h->read_freg(rs2.encoding_value_)));
  float f3 = std::bit_cast<float>(unbox(h->read_freg(rs3.encoding_value_)));
  float result = std::fma(f1, f2, f3);
  h->write_freg(rd.encoding_value_, nan_box(std::bit_cast<uint32_t>(result)));
}

FmsubSInst::FmsubSInst(uint32_t raw)
    : R4Type("fmsub.s", raw, make_exec_fn<FmsubSInst>()), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1), rs2(32, OperandType::OPR_FPR, inst_.rs2),
      rs3(32, OperandType::OPR_FPR, inst_.rs3) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  src_operands_[2] = &rs3;
  num_src_ = 3;
  num_dst_ = 1;
}

void FmsubSInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  float f1 = std::bit_cast<float>(unbox(h->read_freg(rs1.encoding_value_)));
  float f2 = std::bit_cast<float>(unbox(h->read_freg(rs2.encoding_value_)));
  float f3 = std::bit_cast<float>(unbox(h->read_freg(rs3.encoding_value_)));
  float result = std::fma(f1, f2, -f3);
  h->write_freg(rd.encoding_value_, nan_box(std::bit_cast<uint32_t>(result)));
}

FnmsubSInst::FnmsubSInst(uint32_t raw)
    : R4Type("fnmsub.s", raw, make_exec_fn<FnmsubSInst>()), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1), rs2(32, OperandType::OPR_FPR, inst_.rs2),
      rs3(32, OperandType::OPR_FPR, inst_.rs3) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  src_operands_[2] = &rs3;
  num_src_ = 3;
  num_dst_ = 1;
}

void FnmsubSInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  float f1 = std::bit_cast<float>(unbox(h->read_freg(rs1.encoding_value_)));
  float f2 = std::bit_cast<float>(unbox(h->read_freg(rs2.encoding_value_)));
  float f3 = std::bit_cast<float>(unbox(h->read_freg(rs3.encoding_value_)));
  // FNMSUB.S: -(rs1*rs2) + rs3 = fma(-rs1, rs2, rs3)
  float result = std::fma(-f1, f2, f3);
  h->write_freg(rd.encoding_value_, nan_box(std::bit_cast<uint32_t>(result)));
}

FnmaddSInst::FnmaddSInst(uint32_t raw)
    : R4Type("fnmadd.s", raw, make_exec_fn<FnmaddSInst>()), rd(32, OperandType::OPR_FPR, inst_.rd),
      rs1(32, OperandType::OPR_FPR, inst_.rs1), rs2(32, OperandType::OPR_FPR, inst_.rs2),
      rs3(32, OperandType::OPR_FPR, inst_.rs3) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  src_operands_[2] = &rs3;
  num_src_ = 3;
  num_dst_ = 1;
}

void FnmaddSInst::execute_impl(HartState &ctx) {
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

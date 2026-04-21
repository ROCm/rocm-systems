// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/risc_v/m_ext.h"

#include "rocjitsu/vm/risc_v/hart_state.h"

// MULH/MULHSU/MULHU require __int128 for 128-bit multiply.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"

namespace rocjitsu {
namespace risc_v {
namespace detail {

MulInst::MulInst(uint32_t raw)
    : RType("mul", raw, make_exec_fn<MulInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}
void MulInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  int64_t a = h->read_xreg(rs1.encoding_value_);
  int64_t b = h->read_xreg(rs2.encoding_value_);
  h->write_xreg(rd.encoding_value_, a * b);
}

MulhInst::MulhInst(uint32_t raw)
    : RType("mulh", raw, make_exec_fn<MulhInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}
void MulhInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  __int128 a = static_cast<int64_t>(h->read_xreg(rs1.encoding_value_));
  __int128 b = static_cast<int64_t>(h->read_xreg(rs2.encoding_value_));
  __int128 result = a * b;
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(result >> 64));
}

MulhsuInst::MulhsuInst(uint32_t raw)
    : RType("mulhsu", raw, make_exec_fn<MulhsuInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}
void MulhsuInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  __int128 a = static_cast<int64_t>(h->read_xreg(rs1.encoding_value_));
  __int128 b = static_cast<__int128>(static_cast<uint64_t>(h->read_xreg(rs2.encoding_value_)));
  __int128 result = a * b;
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(result >> 64));
}

MulhuInst::MulhuInst(uint32_t raw)
    : RType("mulhu", raw, make_exec_fn<MulhuInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}
void MulhuInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  unsigned __int128 a = static_cast<uint64_t>(h->read_xreg(rs1.encoding_value_));
  unsigned __int128 b = static_cast<uint64_t>(h->read_xreg(rs2.encoding_value_));
  unsigned __int128 result = a * b;
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(static_cast<uint64_t>(result >> 64)));
}

DivInst::DivInst(uint32_t raw)
    : RType("div", raw, make_exec_fn<DivInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}
void DivInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  int64_t a = h->read_xreg(rs1.encoding_value_);
  int64_t b = h->read_xreg(rs2.encoding_value_);
  int64_t result;
  if (b == 0) {
    result = -1;
  } else if (a == INT64_MIN && b == -1) {
    result = INT64_MIN;
  } else {
    result = a / b;
  }
  h->write_xreg(rd.encoding_value_, result);
}

DivuInst::DivuInst(uint32_t raw)
    : RType("divu", raw, make_exec_fn<DivuInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}
void DivuInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  uint64_t a = static_cast<uint64_t>(h->read_xreg(rs1.encoding_value_));
  uint64_t b = static_cast<uint64_t>(h->read_xreg(rs2.encoding_value_));
  uint64_t result;
  if (b == 0) {
    result = UINT64_MAX;
  } else {
    result = a / b;
  }
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(result));
}

RemInst::RemInst(uint32_t raw)
    : RType("rem", raw, make_exec_fn<RemInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}
void RemInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  int64_t a = h->read_xreg(rs1.encoding_value_);
  int64_t b = h->read_xreg(rs2.encoding_value_);
  int64_t result;
  if (b == 0) {
    result = a;
  } else if (a == INT64_MIN && b == -1) {
    result = 0;
  } else {
    result = a % b;
  }
  h->write_xreg(rd.encoding_value_, result);
}

RemuInst::RemuInst(uint32_t raw)
    : RType("remu", raw, make_exec_fn<RemuInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}
void RemuInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  uint64_t a = static_cast<uint64_t>(h->read_xreg(rs1.encoding_value_));
  uint64_t b = static_cast<uint64_t>(h->read_xreg(rs2.encoding_value_));
  uint64_t result;
  if (b == 0) {
    result = a;
  } else {
    result = a % b;
  }
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(result));
}

MulwInst::MulwInst(uint32_t raw)
    : RType("mulw", raw, make_exec_fn<MulwInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}
void MulwInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  int32_t a = static_cast<int32_t>(h->read_xreg(rs1.encoding_value_));
  int32_t b = static_cast<int32_t>(h->read_xreg(rs2.encoding_value_));
  h->write_xreg(rd.encoding_value_, sext32(a * b));
}

DivwInst::DivwInst(uint32_t raw)
    : RType("divw", raw, make_exec_fn<DivwInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}
void DivwInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  int32_t a = static_cast<int32_t>(h->read_xreg(rs1.encoding_value_));
  int32_t b = static_cast<int32_t>(h->read_xreg(rs2.encoding_value_));
  int32_t result;
  if (b == 0) {
    result = -1;
  } else if (a == INT32_MIN && b == -1) {
    result = INT32_MIN;
  } else {
    result = a / b;
  }
  h->write_xreg(rd.encoding_value_, sext32(result));
}

DivuwInst::DivuwInst(uint32_t raw)
    : RType("divuw", raw, make_exec_fn<DivuwInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}
void DivuwInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  uint32_t a = static_cast<uint32_t>(h->read_xreg(rs1.encoding_value_));
  uint32_t b = static_cast<uint32_t>(h->read_xreg(rs2.encoding_value_));
  int32_t result;
  if (b == 0) {
    result = static_cast<int32_t>(UINT32_MAX);
  } else {
    result = static_cast<int32_t>(a / b);
  }
  h->write_xreg(rd.encoding_value_, sext32(result));
}

RemwInst::RemwInst(uint32_t raw)
    : RType("remw", raw, make_exec_fn<RemwInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}
void RemwInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  int32_t a = static_cast<int32_t>(h->read_xreg(rs1.encoding_value_));
  int32_t b = static_cast<int32_t>(h->read_xreg(rs2.encoding_value_));
  int32_t result;
  if (b == 0) {
    result = a;
  } else if (a == INT32_MIN && b == -1) {
    result = 0;
  } else {
    result = a % b;
  }
  h->write_xreg(rd.encoding_value_, sext32(result));
}

RemuwInst::RemuwInst(uint32_t raw)
    : RType("remuw", raw, make_exec_fn<RemuwInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 1;
}
void RemuwInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  uint32_t a = static_cast<uint32_t>(h->read_xreg(rs1.encoding_value_));
  uint32_t b = static_cast<uint32_t>(h->read_xreg(rs2.encoding_value_));
  int32_t result;
  if (b == 0) {
    result = static_cast<int32_t>(a);
  } else {
    result = static_cast<int32_t>(a % b);
  }
  h->write_xreg(rd.encoding_value_, sext32(result));
}

} // namespace detail
} // namespace risc_v
} // namespace rocjitsu

#pragma GCC diagnostic pop

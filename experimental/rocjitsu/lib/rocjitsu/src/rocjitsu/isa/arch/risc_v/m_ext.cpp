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
    : RType("mul", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}
void MulInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  int64_t a = h->read_xreg(rs1.encoding_value_);
  int64_t b = h->read_xreg(rs2.encoding_value_);
  h->write_xreg(rd.encoding_value_, a * b);
}

MulhInst::MulhInst(uint32_t raw)
    : RType("mulh", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}
void MulhInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  __int128 a = static_cast<int64_t>(h->read_xreg(rs1.encoding_value_));
  __int128 b = static_cast<int64_t>(h->read_xreg(rs2.encoding_value_));
  __int128 result = a * b;
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(result >> 64));
}

MulhsuInst::MulhsuInst(uint32_t raw)
    : RType("mulhsu", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}
void MulhsuInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  __int128 a = static_cast<int64_t>(h->read_xreg(rs1.encoding_value_));
  __int128 b = static_cast<__int128>(static_cast<uint64_t>(h->read_xreg(rs2.encoding_value_)));
  __int128 result = a * b;
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(result >> 64));
}

MulhuInst::MulhuInst(uint32_t raw)
    : RType("mulhu", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}
void MulhuInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  unsigned __int128 a = static_cast<uint64_t>(h->read_xreg(rs1.encoding_value_));
  unsigned __int128 b = static_cast<uint64_t>(h->read_xreg(rs2.encoding_value_));
  unsigned __int128 result = a * b;
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(static_cast<uint64_t>(result >> 64)));
}

DivInst::DivInst(uint32_t raw)
    : RType("div", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}
void DivInst::execute(HartState &ctx) {
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
    : RType("divu", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}
void DivuInst::execute(HartState &ctx) {
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
    : RType("rem", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}
void RemInst::execute(HartState &ctx) {
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
    : RType("remu", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}
void RemuInst::execute(HartState &ctx) {
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
    : RType("mulw", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}
void MulwInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  int32_t a = static_cast<int32_t>(h->read_xreg(rs1.encoding_value_));
  int32_t b = static_cast<int32_t>(h->read_xreg(rs2.encoding_value_));
  h->write_xreg(rd.encoding_value_, sext32(a * b));
}

DivwInst::DivwInst(uint32_t raw)
    : RType("divw", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}
void DivwInst::execute(HartState &ctx) {
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
    : RType("divuw", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}
void DivuwInst::execute(HartState &ctx) {
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
    : RType("remw", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}
void RemwInst::execute(HartState &ctx) {
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
    : RType("remuw", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}
void RemuwInst::execute(HartState &ctx) {
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

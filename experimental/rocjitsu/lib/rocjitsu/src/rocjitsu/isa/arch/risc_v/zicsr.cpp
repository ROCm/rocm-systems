// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/risc_v/zicsr.h"
#include "rocjitsu/vm/risc_v/hart_state.h"

namespace rocjitsu {
namespace risc_v {
namespace detail {

CsrrwInst::CsrrwInst(uint32_t raw)
    : IType("csrrw", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), csr(12, OperandType::OPR_CSR, inst_.imm11_0) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&csr);
}
void CsrrwInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto csr_addr = static_cast<uint16_t>(csr.encoding_value_);
  uint64_t old = h->read_csr(csr_addr);
  h->write_csr(csr_addr, static_cast<uint64_t>(h->read_xreg(rs1.encoding_value_)));
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(old));
}

CsrrsInst::CsrrsInst(uint32_t raw)
    : IType("csrrs", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), csr(12, OperandType::OPR_CSR, inst_.imm11_0) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&csr);
}
void CsrrsInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto csr_addr = static_cast<uint16_t>(csr.encoding_value_);
  uint64_t old = h->read_csr(csr_addr);
  h->write_csr(csr_addr, old | static_cast<uint64_t>(h->read_xreg(rs1.encoding_value_)));
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(old));
}

CsrrcInst::CsrrcInst(uint32_t raw)
    : IType("csrrc", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), csr(12, OperandType::OPR_CSR, inst_.imm11_0) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&csr);
}
void CsrrcInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto csr_addr = static_cast<uint16_t>(csr.encoding_value_);
  uint64_t old = h->read_csr(csr_addr);
  h->write_csr(csr_addr, old & ~static_cast<uint64_t>(h->read_xreg(rs1.encoding_value_)));
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(old));
}

CsrrwiInst::CsrrwiInst(uint32_t raw)
    : IType("csrrwi", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      uimm(5, OperandType::OPR_IMM, inst_.rs1), csr(12, OperandType::OPR_CSR, inst_.imm11_0) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&uimm);
  src_operands_.emplace_back(&csr);
}
void CsrrwiInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto csr_addr = static_cast<uint16_t>(csr.encoding_value_);
  uint64_t old = h->read_csr(csr_addr);
  h->write_csr(csr_addr, static_cast<uint64_t>(uimm.encoding_value_));
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(old));
}

CsrrsiInst::CsrrsiInst(uint32_t raw)
    : IType("csrrsi", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      uimm(5, OperandType::OPR_IMM, inst_.rs1), csr(12, OperandType::OPR_CSR, inst_.imm11_0) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&uimm);
  src_operands_.emplace_back(&csr);
}
void CsrrsiInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto csr_addr = static_cast<uint16_t>(csr.encoding_value_);
  uint64_t old = h->read_csr(csr_addr);
  h->write_csr(csr_addr, old | static_cast<uint64_t>(uimm.encoding_value_));
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(old));
}

CsrrciInst::CsrrciInst(uint32_t raw)
    : IType("csrrci", raw), rd(64, OperandType::OPR_GPR, inst_.rd),
      uimm(5, OperandType::OPR_IMM, inst_.rs1), csr(12, OperandType::OPR_CSR, inst_.imm11_0) {
  dst_operands_.emplace_back(&rd);
  src_operands_.emplace_back(&uimm);
  src_operands_.emplace_back(&csr);
}
void CsrrciInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto csr_addr = static_cast<uint16_t>(csr.encoding_value_);
  uint64_t old = h->read_csr(csr_addr);
  h->write_csr(csr_addr, old & ~static_cast<uint64_t>(uimm.encoding_value_));
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(old));
}

} // namespace detail
} // namespace risc_v
} // namespace rocjitsu

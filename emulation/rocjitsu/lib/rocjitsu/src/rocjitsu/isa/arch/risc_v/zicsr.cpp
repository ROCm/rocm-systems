// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/risc_v/zicsr.h"
#include "rocjitsu/vm/risc_v/hart_state.h"

namespace rocjitsu {
namespace risc_v {
namespace detail {

CsrrwInst::CsrrwInst(uint32_t raw)
    : IType("csrrw", raw, make_exec_fn<CsrrwInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), csr(12, OperandType::OPR_CSR, inst_.imm11_0) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &csr;
  num_src_ = 2;
  num_dst_ = 1;
}
void CsrrwInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto csr_addr = static_cast<uint16_t>(csr.encoding_value_);
  uint64_t old = h->read_csr(csr_addr);
  h->write_csr(csr_addr, static_cast<uint64_t>(h->read_xreg(rs1.encoding_value_)));
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(old));
}

CsrrsInst::CsrrsInst(uint32_t raw)
    : IType("csrrs", raw, make_exec_fn<CsrrsInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), csr(12, OperandType::OPR_CSR, inst_.imm11_0) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &csr;
  num_src_ = 2;
  num_dst_ = 1;
}
void CsrrsInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto csr_addr = static_cast<uint16_t>(csr.encoding_value_);
  uint64_t old = h->read_csr(csr_addr);
  h->write_csr(csr_addr, old | static_cast<uint64_t>(h->read_xreg(rs1.encoding_value_)));
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(old));
}

CsrrcInst::CsrrcInst(uint32_t raw)
    : IType("csrrc", raw, make_exec_fn<CsrrcInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), csr(12, OperandType::OPR_CSR, inst_.imm11_0) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &rs1;
  src_operands_[1] = &csr;
  num_src_ = 2;
  num_dst_ = 1;
}
void CsrrcInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto csr_addr = static_cast<uint16_t>(csr.encoding_value_);
  uint64_t old = h->read_csr(csr_addr);
  h->write_csr(csr_addr, old & ~static_cast<uint64_t>(h->read_xreg(rs1.encoding_value_)));
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(old));
}

CsrrwiInst::CsrrwiInst(uint32_t raw)
    : IType("csrrwi", raw, make_exec_fn<CsrrwiInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      uimm(5, OperandType::OPR_IMM, inst_.rs1), csr(12, OperandType::OPR_CSR, inst_.imm11_0) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &uimm;
  src_operands_[1] = &csr;
  num_src_ = 2;
  num_dst_ = 1;
}
void CsrrwiInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto csr_addr = static_cast<uint16_t>(csr.encoding_value_);
  uint64_t old = h->read_csr(csr_addr);
  h->write_csr(csr_addr, static_cast<uint64_t>(uimm.encoding_value_));
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(old));
}

CsrrsiInst::CsrrsiInst(uint32_t raw)
    : IType("csrrsi", raw, make_exec_fn<CsrrsiInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      uimm(5, OperandType::OPR_IMM, inst_.rs1), csr(12, OperandType::OPR_CSR, inst_.imm11_0) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &uimm;
  src_operands_[1] = &csr;
  num_src_ = 2;
  num_dst_ = 1;
}
void CsrrsiInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto csr_addr = static_cast<uint16_t>(csr.encoding_value_);
  uint64_t old = h->read_csr(csr_addr);
  h->write_csr(csr_addr, old | static_cast<uint64_t>(uimm.encoding_value_));
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(old));
}

CsrrciInst::CsrrciInst(uint32_t raw)
    : IType("csrrci", raw, make_exec_fn<CsrrciInst>()), rd(64, OperandType::OPR_GPR, inst_.rd),
      uimm(5, OperandType::OPR_IMM, inst_.rs1), csr(12, OperandType::OPR_CSR, inst_.imm11_0) {
  dst_operands_[0] = &rd;
  src_operands_[0] = &uimm;
  src_operands_[1] = &csr;
  num_src_ = 2;
  num_dst_ = 1;
}
void CsrrciInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  auto csr_addr = static_cast<uint16_t>(csr.encoding_value_);
  uint64_t old = h->read_csr(csr_addr);
  h->write_csr(csr_addr, old & ~static_cast<uint64_t>(uimm.encoding_value_));
  h->write_xreg(rd.encoding_value_, static_cast<int64_t>(old));
}

} // namespace detail
} // namespace risc_v
} // namespace rocjitsu

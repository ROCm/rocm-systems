// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/arch/risc_v/privileged.h"
#include "rocjitsu/vm/risc_v/hart_state.h"

namespace rocjitsu {
namespace risc_v {
namespace detail {

namespace {
// CSR addresses for privileged instructions.
constexpr uint16_t CSR_SEPC = 0x141;
constexpr uint16_t CSR_MEPC = 0x341;
} // namespace

SretInst::SretInst(uint32_t raw) : RType("sret", raw, make_exec_fn<SretInst>()) {}

void SretInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  h->next_pc = h->read_csr(CSR_SEPC);
}

MretInst::MretInst(uint32_t raw) : RType("mret", raw, make_exec_fn<MretInst>()) {}

void MretInst::execute_impl(HartState &ctx) {
  auto *h = as_hart(ctx);
  h->next_pc = h->read_csr(CSR_MEPC);
}

WfiInst::WfiInst(uint32_t raw) : RType("wfi", raw, make_exec_fn<WfiInst>()) {}

void WfiInst::execute_impl(HartState &ctx) {
  (void)ctx; // No-op for single-hart simulation.
}

SfenceVmaInst::SfenceVmaInst(uint32_t raw)
    : RType("sfence.vma", raw, make_exec_fn<SfenceVmaInst>()),
      rs1(64, OperandType::OPR_GPR, inst_.rs1), rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  src_operands_[0] = &rs1;
  src_operands_[1] = &rs2;
  num_src_ = 2;
  num_dst_ = 0;
}
void SfenceVmaInst::execute_impl(HartState &ctx) {
  (void)ctx; // No-op: no TLB in this simulation.
}

} // namespace detail
} // namespace risc_v
} // namespace rocjitsu

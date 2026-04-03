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

SretInst::SretInst(uint32_t raw) : RType("sret", raw) {}
void SretInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  h->next_pc = h->read_csr(CSR_SEPC);
}

MretInst::MretInst(uint32_t raw) : RType("mret", raw) {}
void MretInst::execute(HartState &ctx) {
  auto *h = as_hart(ctx);
  h->next_pc = h->read_csr(CSR_MEPC);
}

WfiInst::WfiInst(uint32_t raw) : RType("wfi", raw) {}
void WfiInst::execute(HartState &ctx) {
  (void)ctx; // No-op for single-hart simulation.
}

SfenceVmaInst::SfenceVmaInst(uint32_t raw)
    : RType("sfence.vma", raw), rs1(64, OperandType::OPR_GPR, inst_.rs1),
      rs2(64, OperandType::OPR_GPR, inst_.rs2) {
  src_operands_.emplace_back(&rs1);
  src_operands_.emplace_back(&rs2);
}
void SfenceVmaInst::execute(HartState &ctx) {
  (void)ctx; // No-op: no TLB in this simulation.
}

} // namespace detail
} // namespace risc_v
} // namespace rocjitsu

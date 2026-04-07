// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/cdna3/sopp.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"
#include "util/except.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

namespace rocjitsu {
namespace cdna3 {

SNopSopp::SNopSopp(const MachineInst *inst)
    : Sopp("s_nop", reinterpret_cast<const OpEncoding *>(inst)),
      simm16(32, OperandType::OPR_SIMM16, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  src_operands_.emplace_back(&simm16);
}

void SNopSopp::execute(amdgpu::Wavefront &wf) { (void)wf; }

SEndpgmSopp::SEndpgmSopp(const MachineInst *inst)
    : Sopp("s_endpgm", reinterpret_cast<const OpEncoding *>(inst)) {}

void SEndpgmSopp::execute(amdgpu::Wavefront &wf) { wf.halt(); }

SBranchSopp::SBranchSopp(const MachineInst *inst)
    : Sopp("s_branch", reinterpret_cast<const OpEncoding *>(inst)),
      simm16(32, OperandType::OPR_LABEL, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  src_operands_.emplace_back(&simm16);
}

void SBranchSopp::execute(amdgpu::Wavefront &wf) {
  int16_t offset = static_cast<int16_t>(simm16.encoding_value_);
  wf.pc = wf.pc + 4 + static_cast<int64_t>(offset) * 4 - size_;
}

SWakeupSopp::SWakeupSopp(const MachineInst *inst)
    : Sopp("s_wakeup", reinterpret_cast<const OpEncoding *>(inst)) {}

void SWakeupSopp::execute(amdgpu::Wavefront &wf) { (void)wf; }

SCbranchScc0Sopp::SCbranchScc0Sopp(const MachineInst *inst)
    : Sopp("s_cbranch_scc0", reinterpret_cast<const OpEncoding *>(inst)),
      simm16(32, OperandType::OPR_LABEL, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  src_operands_.emplace_back(&simm16);
}

void SCbranchScc0Sopp::execute(amdgpu::Wavefront &wf) {
  if (!wf.read_scc()) {
    int16_t offset = static_cast<int16_t>(simm16.encoding_value_);
    wf.pc = wf.pc + 4 + static_cast<int64_t>(offset) * 4 - size_;
  }
}

SCbranchScc1Sopp::SCbranchScc1Sopp(const MachineInst *inst)
    : Sopp("s_cbranch_scc1", reinterpret_cast<const OpEncoding *>(inst)),
      simm16(32, OperandType::OPR_LABEL, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  src_operands_.emplace_back(&simm16);
}

void SCbranchScc1Sopp::execute(amdgpu::Wavefront &wf) {
  if (wf.read_scc()) {
    int16_t offset = static_cast<int16_t>(simm16.encoding_value_);
    wf.pc = wf.pc + 4 + static_cast<int64_t>(offset) * 4 - size_;
  }
}

SCbranchVcczSopp::SCbranchVcczSopp(const MachineInst *inst)
    : Sopp("s_cbranch_vccz", reinterpret_cast<const OpEncoding *>(inst)),
      simm16(32, OperandType::OPR_LABEL, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  src_operands_.emplace_back(&simm16);
}

void SCbranchVcczSopp::execute(amdgpu::Wavefront &wf) {
  if (wf.vcc() == 0) {
    int16_t offset = static_cast<int16_t>(simm16.encoding_value_);
    wf.pc = wf.pc + 4 + static_cast<int64_t>(offset) * 4 - size_;
  }
}

SCbranchVccnzSopp::SCbranchVccnzSopp(const MachineInst *inst)
    : Sopp("s_cbranch_vccnz", reinterpret_cast<const OpEncoding *>(inst)),
      simm16(32, OperandType::OPR_LABEL, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  src_operands_.emplace_back(&simm16);
}

void SCbranchVccnzSopp::execute(amdgpu::Wavefront &wf) {
  if (wf.vcc() != 0) {
    int16_t offset = static_cast<int16_t>(simm16.encoding_value_);
    wf.pc = wf.pc + 4 + static_cast<int64_t>(offset) * 4 - size_;
  }
}

SCbranchExeczSopp::SCbranchExeczSopp(const MachineInst *inst)
    : Sopp("s_cbranch_execz", reinterpret_cast<const OpEncoding *>(inst)),
      simm16(32, OperandType::OPR_LABEL, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  src_operands_.emplace_back(&simm16);
}

void SCbranchExeczSopp::execute(amdgpu::Wavefront &wf) {
  if (wf.exec() == 0) {
    int16_t offset = static_cast<int16_t>(simm16.encoding_value_);
    wf.pc = wf.pc + 4 + static_cast<int64_t>(offset) * 4 - size_;
  }
}

SCbranchExecnzSopp::SCbranchExecnzSopp(const MachineInst *inst)
    : Sopp("s_cbranch_execnz", reinterpret_cast<const OpEncoding *>(inst)),
      simm16(32, OperandType::OPR_LABEL, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  src_operands_.emplace_back(&simm16);
}

void SCbranchExecnzSopp::execute(amdgpu::Wavefront &wf) {
  if (wf.exec() != 0) {
    int16_t offset = static_cast<int16_t>(simm16.encoding_value_);
    wf.pc = wf.pc + 4 + static_cast<int64_t>(offset) * 4 - size_;
  }
}

SBarrierSopp::SBarrierSopp(const MachineInst *inst)
    : Sopp("s_barrier", reinterpret_cast<const OpEncoding *>(inst)) {}

void SBarrierSopp::execute(amdgpu::Wavefront &wf) { (void)wf; }

SSetkillSopp::SSetkillSopp(const MachineInst *inst)
    : Sopp("s_setkill", reinterpret_cast<const OpEncoding *>(inst)),
      simm16(32, OperandType::OPR_SIMM16, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  src_operands_.emplace_back(&simm16);
}

void SSetkillSopp::execute(amdgpu::Wavefront &wf) { (void)wf; }

SWaitcntSopp::SWaitcntSopp(const MachineInst *inst)
    : Sopp("s_waitcnt", reinterpret_cast<const OpEncoding *>(inst)),
      simm16(32, OperandType::OPR_WAITCNT, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  src_operands_.emplace_back(&simm16);
}

void SWaitcntSopp::execute(amdgpu::Wavefront &wf) {
  uint16_t imm = static_cast<uint16_t>(simm16.encoding_value_);
  uint8_t vm = (imm & 0xF) | ((imm >> 10) & 0x30);
  uint8_t exp = (imm >> 4) & 0x7;
  uint8_t lgkm = (imm >> 8) & 0x1F;
  wf.set_wait_target(vm, lgkm, exp);
  if (!wf.wait_satisfied())
    wf.set_state(amdgpu::WfState::WAITCNT);
}

SSethaltSopp::SSethaltSopp(const MachineInst *inst)
    : Sopp("s_sethalt", reinterpret_cast<const OpEncoding *>(inst)),
      simm16(32, OperandType::OPR_SIMM16, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  src_operands_.emplace_back(&simm16);
}

void SSethaltSopp::execute(amdgpu::Wavefront &wf) { (void)wf; }

SSleepSopp::SSleepSopp(const MachineInst *inst)
    : Sopp("s_sleep", reinterpret_cast<const OpEncoding *>(inst)),
      simm16(32, OperandType::OPR_SIMM16, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  src_operands_.emplace_back(&simm16);
}

void SSleepSopp::execute(amdgpu::Wavefront &wf) { (void)wf; }

SSetprioSopp::SSetprioSopp(const MachineInst *inst)
    : Sopp("s_setprio", reinterpret_cast<const OpEncoding *>(inst)),
      simm16(32, OperandType::OPR_SIMM16, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  src_operands_.emplace_back(&simm16);
}

void SSetprioSopp::execute(amdgpu::Wavefront &wf) { (void)wf; }

SSendmsgSopp::SSendmsgSopp(const MachineInst *inst)
    : Sopp("s_sendmsg", reinterpret_cast<const OpEncoding *>(inst)),
      simm16(32, OperandType::OPR_SENDMSG, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  src_operands_.emplace_back(&simm16);
}

void SSendmsgSopp::execute(amdgpu::Wavefront &wf) { (void)wf; }

SSendmsghaltSopp::SSendmsghaltSopp(const MachineInst *inst)
    : Sopp("s_sendmsghalt", reinterpret_cast<const OpEncoding *>(inst)),
      simm16(32, OperandType::OPR_SENDMSG, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  src_operands_.emplace_back(&simm16);
}

void SSendmsghaltSopp::execute(amdgpu::Wavefront &wf) { (void)wf; }

STrapSopp::STrapSopp(const MachineInst *inst)
    : Sopp("s_trap", reinterpret_cast<const OpEncoding *>(inst)),
      simm16(32, OperandType::OPR_SIMM16, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  src_operands_.emplace_back(&simm16);
}

void STrapSopp::execute(amdgpu::Wavefront &wf) { (void)wf; }

SIcacheInvSopp::SIcacheInvSopp(const MachineInst *inst)
    : Sopp("s_icache_inv", reinterpret_cast<const OpEncoding *>(inst)) {}

void SIcacheInvSopp::execute(amdgpu::Wavefront &wf) { (void)wf; }

SIncperflevelSopp::SIncperflevelSopp(const MachineInst *inst)
    : Sopp("s_incperflevel", reinterpret_cast<const OpEncoding *>(inst)),
      simm16(32, OperandType::OPR_SIMM16, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  src_operands_.emplace_back(&simm16);
}

void SIncperflevelSopp::execute(amdgpu::Wavefront &wf) { (void)wf; }

SDecperflevelSopp::SDecperflevelSopp(const MachineInst *inst)
    : Sopp("s_decperflevel", reinterpret_cast<const OpEncoding *>(inst)),
      simm16(32, OperandType::OPR_SIMM16, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  src_operands_.emplace_back(&simm16);
}

void SDecperflevelSopp::execute(amdgpu::Wavefront &wf) { (void)wf; }

STtracedataSopp::STtracedataSopp(const MachineInst *inst)
    : Sopp("s_ttracedata", reinterpret_cast<const OpEncoding *>(inst)) {}

void STtracedataSopp::execute(amdgpu::Wavefront &wf) { (void)wf; }

SCbranchCdbgsysSopp::SCbranchCdbgsysSopp(const MachineInst *inst)
    : Sopp("s_cbranch_cdbgsys", reinterpret_cast<const OpEncoding *>(inst)),
      simm16(32, OperandType::OPR_LABEL, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  src_operands_.emplace_back(&simm16);
}

void SCbranchCdbgsysSopp::execute(amdgpu::Wavefront &wf) { (void)wf; }

SCbranchCdbguserSopp::SCbranchCdbguserSopp(const MachineInst *inst)
    : Sopp("s_cbranch_cdbguser", reinterpret_cast<const OpEncoding *>(inst)),
      simm16(32, OperandType::OPR_LABEL, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  src_operands_.emplace_back(&simm16);
}

void SCbranchCdbguserSopp::execute(amdgpu::Wavefront &wf) { (void)wf; }

SCbranchCdbgsysOrUserSopp::SCbranchCdbgsysOrUserSopp(const MachineInst *inst)
    : Sopp("s_cbranch_cdbgsys_or_user", reinterpret_cast<const OpEncoding *>(inst)),
      simm16(32, OperandType::OPR_LABEL, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  src_operands_.emplace_back(&simm16);
}

void SCbranchCdbgsysOrUserSopp::execute(amdgpu::Wavefront &wf) { (void)wf; }

SCbranchCdbgsysAndUserSopp::SCbranchCdbgsysAndUserSopp(const MachineInst *inst)
    : Sopp("s_cbranch_cdbgsys_and_user", reinterpret_cast<const OpEncoding *>(inst)),
      simm16(32, OperandType::OPR_LABEL, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  src_operands_.emplace_back(&simm16);
}

void SCbranchCdbgsysAndUserSopp::execute(amdgpu::Wavefront &wf) { (void)wf; }

SEndpgmSavedSopp::SEndpgmSavedSopp(const MachineInst *inst)
    : Sopp("s_endpgm_saved", reinterpret_cast<const OpEncoding *>(inst)) {}

void SEndpgmSavedSopp::execute(amdgpu::Wavefront &wf) { wf.halt(); }

SSetGprIdxOffSopp::SSetGprIdxOffSopp(const MachineInst *inst)
    : Sopp("s_set_gpr_idx_off", reinterpret_cast<const OpEncoding *>(inst)) {}

void SSetGprIdxOffSopp::execute(amdgpu::Wavefront &wf) { (void)wf; }

SSetGprIdxModeSopp::SSetGprIdxModeSopp(const MachineInst *inst)
    : Sopp("s_set_gpr_idx_mode", reinterpret_cast<const OpEncoding *>(inst)),
      simm16(32, OperandType::OPR_SIMM16, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  src_operands_.emplace_back(&simm16);
}

void SSetGprIdxModeSopp::execute(amdgpu::Wavefront &wf) { (void)wf; }

SEndpgmOrderedPsDoneSopp::SEndpgmOrderedPsDoneSopp(const MachineInst *inst)
    : Sopp("s_endpgm_ordered_ps_done", reinterpret_cast<const OpEncoding *>(inst)) {}

void SEndpgmOrderedPsDoneSopp::execute(amdgpu::Wavefront &wf) { wf.halt(); }

SSetValuCoexecModeSopp::SSetValuCoexecModeSopp(const MachineInst *inst)
    : Sopp("s_set_valu_coexec_mode", reinterpret_cast<const OpEncoding *>(inst)),
      simm16(32, OperandType::OPR_SIMM16, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  src_operands_.emplace_back(&simm16);
}

void SSetValuCoexecModeSopp::execute(amdgpu::Wavefront &wf) { (void)wf; }

} // namespace cdna3
} // namespace rocjitsu

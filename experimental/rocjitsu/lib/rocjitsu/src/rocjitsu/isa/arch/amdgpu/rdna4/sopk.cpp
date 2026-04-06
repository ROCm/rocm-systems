// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/rdna4/sopk.h"
#include "rocjitsu/isa/arch/amdgpu/shared/execute_shared.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"
#include "util/except.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

namespace rocjitsu {
namespace rdna4 {

SMovkI32Sopk::SMovkI32Sopk(const MachineInst *inst)
    : Sopk("s_movk_i32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      simm16(16, OperandType::OPR_SIMM16, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&simm16);
}

void SMovkI32Sopk::execute(amdgpu::Wavefront &wf) {
  sdst.write_scalar(wf, static_cast<uint32_t>(
                            static_cast<int32_t>(static_cast<int16_t>(simm16.encoding_value_))));
}

SVersionSopk::SVersionSopk(const MachineInst *inst)
    : Sopk("s_version", reinterpret_cast<const OpEncoding *>(inst)),
      simm16(16, OperandType::OPR_VERSION, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  src_operands_.emplace_back(&simm16);
}

void SVersionSopk::execute(amdgpu::Wavefront &wf) { (void)wf; }

SCmovkI32Sopk::SCmovkI32Sopk(const MachineInst *inst)
    : Sopk("s_cmovk_i32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      simm16(16, OperandType::OPR_SIMM16, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  src_operands_.emplace_back(&sdst);
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&simm16);
}

void SCmovkI32Sopk::execute(amdgpu::Wavefront &wf) { amdgpu::execute_s_cmovk_i32_sopk(*this, wf); }

SAddkCoI32Sopk::SAddkCoI32Sopk(const MachineInst *inst)
    : Sopk("s_addk_co_i32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      simm16(16, OperandType::OPR_SIMM16, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  src_operands_.emplace_back(&sdst);
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&simm16);
}

void SAddkCoI32Sopk::execute(amdgpu::Wavefront &wf) { (void)wf; }

SMulkI32Sopk::SMulkI32Sopk(const MachineInst *inst)
    : Sopk("s_mulk_i32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      simm16(16, OperandType::OPR_SIMM16, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  src_operands_.emplace_back(&sdst);
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&simm16);
}

void SMulkI32Sopk::execute(amdgpu::Wavefront &wf) { amdgpu::execute_s_mulk_i32_sopk(*this, wf); }

SGetregB32Sopk::SGetregB32Sopk(const MachineInst *inst)
    : Sopk("s_getreg_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      simm16(16, OperandType::OPR_HWREG, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&simm16);
}

void SGetregB32Sopk::execute(amdgpu::Wavefront &wf) { (void)wf; }

SSetregB32Sopk::SSetregB32Sopk(const MachineInst *inst)
    : Sopk("s_setreg_b32", reinterpret_cast<const OpEncoding *>(inst)),
      simm16(16, OperandType::OPR_HWREG, reinterpret_cast<const OpEncoding *>(inst)->simm16),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst) {
  dst_operands_.emplace_back(&simm16);
  src_operands_.emplace_back(&sdst);
}

void SSetregB32Sopk::execute(amdgpu::Wavefront &wf) { (void)wf; }

SSetregImm32B32Sopk::SSetregImm32B32Sopk(const MachineInst *inst)
    : Sopk("s_setreg_imm32_b32", reinterpret_cast<const OpEncoding *>(inst)),
      simm16(16, OperandType::OPR_HWREG, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  dst_operands_.emplace_back(&simm16);
}

void SSetregImm32B32Sopk::execute(amdgpu::Wavefront &wf) { (void)wf; }

SCallB64Sopk::SCallB64Sopk(const MachineInst *inst)
    : Sopk("s_call_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      simm16(16, OperandType::OPR_LABEL, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&simm16);
}

void SCallB64Sopk::execute(amdgpu::Wavefront &wf) {
  sdst.write_scalar64(wf, wf.pc + size_);
  int16_t offset = static_cast<int16_t>(simm16.encoding_value_);
  wf.pc = wf.pc + static_cast<int64_t>(offset) * 4 - size_;
}

} // namespace rdna4
} // namespace rocjitsu

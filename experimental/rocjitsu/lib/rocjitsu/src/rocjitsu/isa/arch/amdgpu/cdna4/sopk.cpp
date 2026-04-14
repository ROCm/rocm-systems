// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/cdna4/sopk.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"
#include "util/except.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

namespace rocjitsu {
namespace cdna4 {

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

SCmovkI32Sopk::SCmovkI32Sopk(const MachineInst *inst)
    : Sopk("s_cmovk_i32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      simm16(16, OperandType::OPR_SIMM16, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&simm16);
}

void SCmovkI32Sopk::execute(amdgpu::Wavefront &wf) {
  if (wf.read_scc())
    sdst.write_scalar(wf, static_cast<uint32_t>(
                              static_cast<int32_t>(static_cast<int16_t>(simm16.encoding_value_))));
}

SCmpkEqI32Sopk::SCmpkEqI32Sopk(const MachineInst *inst)
    : Sopk("s_cmpk_eq_i32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      simm16(16, OperandType::OPR_SIMM16, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  src_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&simm16);
}

void SCmpkEqI32Sopk::execute(amdgpu::Wavefront &wf) {
  int32_t s0 = static_cast<int32_t>(sdst.read_scalar(wf));
  int32_t imm = static_cast<int16_t>(simm16.encoding_value_);
  wf.write_scc(s0 == imm);
}

SCmpkLgI32Sopk::SCmpkLgI32Sopk(const MachineInst *inst)
    : Sopk("s_cmpk_lg_i32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      simm16(16, OperandType::OPR_SIMM16, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  src_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&simm16);
}

void SCmpkLgI32Sopk::execute(amdgpu::Wavefront &wf) {
  int32_t s0 = static_cast<int32_t>(sdst.read_scalar(wf));
  int32_t imm = static_cast<int16_t>(simm16.encoding_value_);
  wf.write_scc(s0 != imm);
}

SCmpkGtI32Sopk::SCmpkGtI32Sopk(const MachineInst *inst)
    : Sopk("s_cmpk_gt_i32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      simm16(16, OperandType::OPR_SIMM16, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  src_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&simm16);
}

void SCmpkGtI32Sopk::execute(amdgpu::Wavefront &wf) {
  int32_t s0 = static_cast<int32_t>(sdst.read_scalar(wf));
  int32_t imm = static_cast<int16_t>(simm16.encoding_value_);
  wf.write_scc(s0 > imm);
}

SCmpkGeI32Sopk::SCmpkGeI32Sopk(const MachineInst *inst)
    : Sopk("s_cmpk_ge_i32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      simm16(16, OperandType::OPR_SIMM16, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  src_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&simm16);
}

void SCmpkGeI32Sopk::execute(amdgpu::Wavefront &wf) {
  int32_t s0 = static_cast<int32_t>(sdst.read_scalar(wf));
  int32_t imm = static_cast<int16_t>(simm16.encoding_value_);
  wf.write_scc(s0 >= imm);
}

SCmpkLtI32Sopk::SCmpkLtI32Sopk(const MachineInst *inst)
    : Sopk("s_cmpk_lt_i32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      simm16(16, OperandType::OPR_SIMM16, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  src_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&simm16);
}

void SCmpkLtI32Sopk::execute(amdgpu::Wavefront &wf) {
  int32_t s0 = static_cast<int32_t>(sdst.read_scalar(wf));
  int32_t imm = static_cast<int16_t>(simm16.encoding_value_);
  wf.write_scc(s0 < imm);
}

SCmpkLeI32Sopk::SCmpkLeI32Sopk(const MachineInst *inst)
    : Sopk("s_cmpk_le_i32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      simm16(16, OperandType::OPR_SIMM16, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  src_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&simm16);
}

void SCmpkLeI32Sopk::execute(amdgpu::Wavefront &wf) {
  int32_t s0 = static_cast<int32_t>(sdst.read_scalar(wf));
  int32_t imm = static_cast<int16_t>(simm16.encoding_value_);
  wf.write_scc(s0 <= imm);
}

SCmpkEqU32Sopk::SCmpkEqU32Sopk(const MachineInst *inst)
    : Sopk("s_cmpk_eq_u32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      simm16(16, OperandType::OPR_SIMM16, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  src_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&simm16);
}

void SCmpkEqU32Sopk::execute(amdgpu::Wavefront &wf) {
  uint32_t s0 = sdst.read_scalar(wf);
  uint32_t imm = static_cast<uint32_t>(static_cast<uint16_t>(simm16.encoding_value_));
  wf.write_scc(s0 == imm);
}

SCmpkLgU32Sopk::SCmpkLgU32Sopk(const MachineInst *inst)
    : Sopk("s_cmpk_lg_u32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      simm16(16, OperandType::OPR_SIMM16, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  src_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&simm16);
}

void SCmpkLgU32Sopk::execute(amdgpu::Wavefront &wf) {
  uint32_t s0 = sdst.read_scalar(wf);
  uint32_t imm = static_cast<uint32_t>(static_cast<uint16_t>(simm16.encoding_value_));
  wf.write_scc(s0 != imm);
}

SCmpkGtU32Sopk::SCmpkGtU32Sopk(const MachineInst *inst)
    : Sopk("s_cmpk_gt_u32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      simm16(16, OperandType::OPR_SIMM16, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  src_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&simm16);
}

void SCmpkGtU32Sopk::execute(amdgpu::Wavefront &wf) {
  uint32_t s0 = sdst.read_scalar(wf);
  uint32_t imm = static_cast<uint32_t>(static_cast<uint16_t>(simm16.encoding_value_));
  wf.write_scc(s0 > imm);
}

SCmpkGeU32Sopk::SCmpkGeU32Sopk(const MachineInst *inst)
    : Sopk("s_cmpk_ge_u32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      simm16(16, OperandType::OPR_SIMM16, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  src_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&simm16);
}

void SCmpkGeU32Sopk::execute(amdgpu::Wavefront &wf) {
  uint32_t s0 = sdst.read_scalar(wf);
  uint32_t imm = static_cast<uint32_t>(static_cast<uint16_t>(simm16.encoding_value_));
  wf.write_scc(s0 >= imm);
}

SCmpkLtU32Sopk::SCmpkLtU32Sopk(const MachineInst *inst)
    : Sopk("s_cmpk_lt_u32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      simm16(16, OperandType::OPR_SIMM16, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  src_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&simm16);
}

void SCmpkLtU32Sopk::execute(amdgpu::Wavefront &wf) {
  uint32_t s0 = sdst.read_scalar(wf);
  uint32_t imm = static_cast<uint32_t>(static_cast<uint16_t>(simm16.encoding_value_));
  wf.write_scc(s0 < imm);
}

SCmpkLeU32Sopk::SCmpkLeU32Sopk(const MachineInst *inst)
    : Sopk("s_cmpk_le_u32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      simm16(16, OperandType::OPR_SIMM16, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  src_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&simm16);
}

void SCmpkLeU32Sopk::execute(amdgpu::Wavefront &wf) {
  uint32_t s0 = sdst.read_scalar(wf);
  uint32_t imm = static_cast<uint32_t>(static_cast<uint16_t>(simm16.encoding_value_));
  wf.write_scc(s0 <= imm);
}

SAddkI32Sopk::SAddkI32Sopk(const MachineInst *inst)
    : Sopk("s_addk_i32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      simm16(16, OperandType::OPR_SIMM16, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&simm16);
}

void SAddkI32Sopk::execute(amdgpu::Wavefront &wf) {
  int32_t s0 = static_cast<int32_t>(sdst.read_scalar(wf));
  int32_t imm = static_cast<int16_t>(simm16.encoding_value_);
  int64_t wide = static_cast<int64_t>(s0) + static_cast<int64_t>(imm);
  int32_t result = static_cast<int32_t>(wide);
  sdst.write_scalar(wf, static_cast<uint32_t>(result));
  wf.write_scc(wide != static_cast<int64_t>(result));
}

SMulkI32Sopk::SMulkI32Sopk(const MachineInst *inst)
    : Sopk("s_mulk_i32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      simm16(16, OperandType::OPR_SIMM16, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&simm16);
}

void SMulkI32Sopk::execute(amdgpu::Wavefront &wf) {
  int32_t s0 = static_cast<int32_t>(sdst.read_scalar(wf));
  int32_t imm = static_cast<int16_t>(simm16.encoding_value_);
  sdst.write_scalar(wf, static_cast<uint32_t>(s0 * imm));
}

SCbranchIForkSopk::SCbranchIForkSopk(const MachineInst *inst)
    : Sopk("s_cbranch_i_fork", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      simm16(16, OperandType::OPR_LABEL, reinterpret_cast<const OpEncoding *>(inst)->simm16) {
  src_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&simm16);
}

void SCbranchIForkSopk::execute(amdgpu::Wavefront &wf) { (void)wf; }

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
  sdst.write_scalar64(wf, wf.pc);
  int16_t offset = static_cast<int16_t>(simm16.encoding_value_);
  wf.pc = wf.pc + static_cast<int64_t>(offset) * 4 - size_;
}

} // namespace cdna4
} // namespace rocjitsu

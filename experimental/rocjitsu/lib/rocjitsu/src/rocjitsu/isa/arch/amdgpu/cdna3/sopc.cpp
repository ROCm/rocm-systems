// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/cdna3/sopc.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"
#include "util/except.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

namespace rocjitsu {
namespace cdna3 {

SCmpEqI32Sopc::SCmpEqI32Sopc(const MachineInst *inst)
    : Sopc("s_cmp_eq_i32", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
}

void SCmpEqI32Sopc::execute(amdgpu::Wavefront &wf) {
  int32_t s0 = static_cast<int32_t>(ssrc0.read_scalar(wf));
  int32_t s1 = static_cast<int32_t>(ssrc1.read_scalar(wf));
  wf.write_scc(s0 == s1);
}

SCmpLgI32Sopc::SCmpLgI32Sopc(const MachineInst *inst)
    : Sopc("s_cmp_lg_i32", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
}

void SCmpLgI32Sopc::execute(amdgpu::Wavefront &wf) {
  int32_t s0 = static_cast<int32_t>(ssrc0.read_scalar(wf));
  int32_t s1 = static_cast<int32_t>(ssrc1.read_scalar(wf));
  wf.write_scc(s0 != s1);
}

SCmpGtI32Sopc::SCmpGtI32Sopc(const MachineInst *inst)
    : Sopc("s_cmp_gt_i32", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
}

void SCmpGtI32Sopc::execute(amdgpu::Wavefront &wf) {
  int32_t s0 = static_cast<int32_t>(ssrc0.read_scalar(wf));
  int32_t s1 = static_cast<int32_t>(ssrc1.read_scalar(wf));
  wf.write_scc(s0 > s1);
}

SCmpGeI32Sopc::SCmpGeI32Sopc(const MachineInst *inst)
    : Sopc("s_cmp_ge_i32", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
}

void SCmpGeI32Sopc::execute(amdgpu::Wavefront &wf) {
  int32_t s0 = static_cast<int32_t>(ssrc0.read_scalar(wf));
  int32_t s1 = static_cast<int32_t>(ssrc1.read_scalar(wf));
  wf.write_scc(s0 >= s1);
}

SCmpLtI32Sopc::SCmpLtI32Sopc(const MachineInst *inst)
    : Sopc("s_cmp_lt_i32", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
}

void SCmpLtI32Sopc::execute(amdgpu::Wavefront &wf) {
  int32_t s0 = static_cast<int32_t>(ssrc0.read_scalar(wf));
  int32_t s1 = static_cast<int32_t>(ssrc1.read_scalar(wf));
  wf.write_scc(s0 < s1);
}

SCmpLeI32Sopc::SCmpLeI32Sopc(const MachineInst *inst)
    : Sopc("s_cmp_le_i32", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
}

void SCmpLeI32Sopc::execute(amdgpu::Wavefront &wf) {
  int32_t s0 = static_cast<int32_t>(ssrc0.read_scalar(wf));
  int32_t s1 = static_cast<int32_t>(ssrc1.read_scalar(wf));
  wf.write_scc(s0 <= s1);
}

SCmpEqU32Sopc::SCmpEqU32Sopc(const MachineInst *inst)
    : Sopc("s_cmp_eq_u32", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
}

void SCmpEqU32Sopc::execute(amdgpu::Wavefront &wf) {
  uint32_t s0 = ssrc0.read_scalar(wf);
  uint32_t s1 = ssrc1.read_scalar(wf);
  wf.write_scc(s0 == s1);
}

SCmpLgU32Sopc::SCmpLgU32Sopc(const MachineInst *inst)
    : Sopc("s_cmp_lg_u32", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
}

void SCmpLgU32Sopc::execute(amdgpu::Wavefront &wf) {
  uint32_t s0 = ssrc0.read_scalar(wf);
  uint32_t s1 = ssrc1.read_scalar(wf);
  wf.write_scc(s0 != s1);
}

SCmpGtU32Sopc::SCmpGtU32Sopc(const MachineInst *inst)
    : Sopc("s_cmp_gt_u32", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
}

void SCmpGtU32Sopc::execute(amdgpu::Wavefront &wf) {
  uint32_t s0 = ssrc0.read_scalar(wf);
  uint32_t s1 = ssrc1.read_scalar(wf);
  wf.write_scc(s0 > s1);
}

SCmpGeU32Sopc::SCmpGeU32Sopc(const MachineInst *inst)
    : Sopc("s_cmp_ge_u32", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
}

void SCmpGeU32Sopc::execute(amdgpu::Wavefront &wf) {
  uint32_t s0 = ssrc0.read_scalar(wf);
  uint32_t s1 = ssrc1.read_scalar(wf);
  wf.write_scc(s0 >= s1);
}

SCmpLtU32Sopc::SCmpLtU32Sopc(const MachineInst *inst)
    : Sopc("s_cmp_lt_u32", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
}

void SCmpLtU32Sopc::execute(amdgpu::Wavefront &wf) {
  uint32_t s0 = ssrc0.read_scalar(wf);
  uint32_t s1 = ssrc1.read_scalar(wf);
  wf.write_scc(s0 < s1);
}

SCmpLeU32Sopc::SCmpLeU32Sopc(const MachineInst *inst)
    : Sopc("s_cmp_le_u32", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
}

void SCmpLeU32Sopc::execute(amdgpu::Wavefront &wf) {
  uint32_t s0 = ssrc0.read_scalar(wf);
  uint32_t s1 = ssrc1.read_scalar(wf);
  wf.write_scc(s0 <= s1);
}

SBitcmp0B32Sopc::SBitcmp0B32Sopc(const MachineInst *inst)
    : Sopc("s_bitcmp0_b32", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
}

void SBitcmp0B32Sopc::execute(amdgpu::Wavefront &wf) {
  uint32_t val = ssrc0.read_scalar(wf);
  uint32_t bit = ssrc1.read_scalar(wf) & 31u;
  wf.write_scc(!(val & (1ULL << bit)));
}

SBitcmp1B32Sopc::SBitcmp1B32Sopc(const MachineInst *inst)
    : Sopc("s_bitcmp1_b32", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
}

void SBitcmp1B32Sopc::execute(amdgpu::Wavefront &wf) {
  uint32_t val = ssrc0.read_scalar(wf);
  uint32_t bit = ssrc1.read_scalar(wf) & 31u;
  wf.write_scc((val & (1ULL << bit)) != 0);
}

SBitcmp0B64Sopc::SBitcmp0B64Sopc(const MachineInst *inst)
    : Sopc("s_bitcmp0_b64", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
}

void SBitcmp0B64Sopc::execute(amdgpu::Wavefront &wf) {
  uint64_t val = ssrc0.read_scalar64(wf);
  uint32_t bit = ssrc1.read_scalar(wf) & 63u;
  wf.write_scc(!(val & (1ULL << bit)));
}

SBitcmp1B64Sopc::SBitcmp1B64Sopc(const MachineInst *inst)
    : Sopc("s_bitcmp1_b64", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
}

void SBitcmp1B64Sopc::execute(amdgpu::Wavefront &wf) {
  uint64_t val = ssrc0.read_scalar64(wf);
  uint32_t bit = ssrc1.read_scalar(wf) & 63u;
  wf.write_scc((val & (1ULL << bit)) != 0);
}

SSetvskipSopc::SSetvskipSopc(const MachineInst *inst)
    : Sopc("s_setvskip", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
}

void SSetvskipSopc::execute(amdgpu::Wavefront &wf) { (void)wf; }

SSetGprIdxOnSopc::SSetGprIdxOnSopc(const MachineInst *inst)
    : Sopc("s_set_gpr_idx_on", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SIMM4, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
}

void SSetGprIdxOnSopc::execute(amdgpu::Wavefront &wf) { (void)wf; }

SCmpEqU64Sopc::SCmpEqU64Sopc(const MachineInst *inst)
    : Sopc("s_cmp_eq_u64", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
}

void SCmpEqU64Sopc::execute(amdgpu::Wavefront &wf) {
  uint64_t s0 = ssrc0.read_scalar64(wf);
  uint64_t s1 = ssrc1.read_scalar64(wf);
  wf.write_scc(s0 == s1);
}

SCmpLgU64Sopc::SCmpLgU64Sopc(const MachineInst *inst)
    : Sopc("s_cmp_lg_u64", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
}

void SCmpLgU64Sopc::execute(amdgpu::Wavefront &wf) {
  uint64_t s0 = ssrc0.read_scalar64(wf);
  uint64_t s1 = ssrc1.read_scalar64(wf);
  wf.write_scc(s0 != s1);
}

} // namespace cdna3
} // namespace rocjitsu

// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/cdna3/sopc.h"
#include "rocjitsu/isa/arch/amdgpu/shared/execute_shared.h"
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

void SCmpEqI32Sopc::execute(amdgpu::Wavefront &wf) { amdgpu::execute_s_cmp_eq_i32_sopc(*this, wf); }

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

void SCmpLgI32Sopc::execute(amdgpu::Wavefront &wf) { amdgpu::execute_s_cmp_lg_i32_sopc(*this, wf); }

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

void SCmpGtI32Sopc::execute(amdgpu::Wavefront &wf) { amdgpu::execute_s_cmp_gt_i32_sopc(*this, wf); }

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

void SCmpGeI32Sopc::execute(amdgpu::Wavefront &wf) { amdgpu::execute_s_cmp_ge_i32_sopc(*this, wf); }

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

void SCmpLtI32Sopc::execute(amdgpu::Wavefront &wf) { amdgpu::execute_s_cmp_lt_i32_sopc(*this, wf); }

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

void SCmpLeI32Sopc::execute(amdgpu::Wavefront &wf) { amdgpu::execute_s_cmp_le_i32_sopc(*this, wf); }

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

void SCmpEqU32Sopc::execute(amdgpu::Wavefront &wf) { amdgpu::execute_s_cmp_eq_u32_sopc(*this, wf); }

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

void SCmpLgU32Sopc::execute(amdgpu::Wavefront &wf) { amdgpu::execute_s_cmp_lg_u32_sopc(*this, wf); }

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

void SCmpGtU32Sopc::execute(amdgpu::Wavefront &wf) { amdgpu::execute_s_cmp_gt_u32_sopc(*this, wf); }

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

void SCmpGeU32Sopc::execute(amdgpu::Wavefront &wf) { amdgpu::execute_s_cmp_ge_u32_sopc(*this, wf); }

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

void SCmpLtU32Sopc::execute(amdgpu::Wavefront &wf) { amdgpu::execute_s_cmp_lt_u32_sopc(*this, wf); }

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

void SCmpLeU32Sopc::execute(amdgpu::Wavefront &wf) { amdgpu::execute_s_cmp_le_u32_sopc(*this, wf); }

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
  amdgpu::execute_s_bitcmp0_b32_sopc(*this, wf);
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
  amdgpu::execute_s_bitcmp1_b32_sopc(*this, wf);
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
  amdgpu::execute_s_bitcmp0_b64_sopc(*this, wf);
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
  amdgpu::execute_s_bitcmp1_b64_sopc(*this, wf);
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

void SCmpEqU64Sopc::execute(amdgpu::Wavefront &wf) { amdgpu::execute_s_cmp_eq_u64_sopc(*this, wf); }

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

void SCmpLgU64Sopc::execute(amdgpu::Wavefront &wf) { amdgpu::execute_s_cmp_lg_u64_sopc(*this, wf); }

} // namespace cdna3
} // namespace rocjitsu

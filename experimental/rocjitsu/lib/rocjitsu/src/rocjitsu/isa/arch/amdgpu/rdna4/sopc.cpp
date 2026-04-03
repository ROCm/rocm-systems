// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/rdna4/sopc.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"
#include "util/except.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

namespace rocjitsu {
namespace rdna4 {

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

SCmpLtF32Sopc::SCmpLtF32Sopc(const MachineInst *inst)
    : Sopc("s_cmp_lt_f32", reinterpret_cast<const OpEncoding *>(inst)),
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

void SCmpLtF32Sopc::execute(amdgpu::Wavefront &wf) { (void)wf; }

SCmpLtF16Sopc::SCmpLtF16Sopc(const MachineInst *inst)
    : Sopc("s_cmp_lt_f16", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(16, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(16, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
}

void SCmpLtF16Sopc::execute(amdgpu::Wavefront &wf) { (void)wf; }

SCmpEqF32Sopc::SCmpEqF32Sopc(const MachineInst *inst)
    : Sopc("s_cmp_eq_f32", reinterpret_cast<const OpEncoding *>(inst)),
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

void SCmpEqF32Sopc::execute(amdgpu::Wavefront &wf) { (void)wf; }

SCmpEqF16Sopc::SCmpEqF16Sopc(const MachineInst *inst)
    : Sopc("s_cmp_eq_f16", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(16, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(16, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
}

void SCmpEqF16Sopc::execute(amdgpu::Wavefront &wf) { (void)wf; }

SCmpLeF32Sopc::SCmpLeF32Sopc(const MachineInst *inst)
    : Sopc("s_cmp_le_f32", reinterpret_cast<const OpEncoding *>(inst)),
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

void SCmpLeF32Sopc::execute(amdgpu::Wavefront &wf) { (void)wf; }

SCmpLeF16Sopc::SCmpLeF16Sopc(const MachineInst *inst)
    : Sopc("s_cmp_le_f16", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(16, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(16, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
}

void SCmpLeF16Sopc::execute(amdgpu::Wavefront &wf) { (void)wf; }

SCmpGtF32Sopc::SCmpGtF32Sopc(const MachineInst *inst)
    : Sopc("s_cmp_gt_f32", reinterpret_cast<const OpEncoding *>(inst)),
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

void SCmpGtF32Sopc::execute(amdgpu::Wavefront &wf) { (void)wf; }

SCmpGtF16Sopc::SCmpGtF16Sopc(const MachineInst *inst)
    : Sopc("s_cmp_gt_f16", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(16, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(16, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
}

void SCmpGtF16Sopc::execute(amdgpu::Wavefront &wf) { (void)wf; }

SCmpLgF32Sopc::SCmpLgF32Sopc(const MachineInst *inst)
    : Sopc("s_cmp_lg_f32", reinterpret_cast<const OpEncoding *>(inst)),
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

void SCmpLgF32Sopc::execute(amdgpu::Wavefront &wf) { (void)wf; }

SCmpLgF16Sopc::SCmpLgF16Sopc(const MachineInst *inst)
    : Sopc("s_cmp_lg_f16", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(16, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(16, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
}

void SCmpLgF16Sopc::execute(amdgpu::Wavefront &wf) { (void)wf; }

SCmpGeF32Sopc::SCmpGeF32Sopc(const MachineInst *inst)
    : Sopc("s_cmp_ge_f32", reinterpret_cast<const OpEncoding *>(inst)),
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

void SCmpGeF32Sopc::execute(amdgpu::Wavefront &wf) { (void)wf; }

SCmpGeF16Sopc::SCmpGeF16Sopc(const MachineInst *inst)
    : Sopc("s_cmp_ge_f16", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(16, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(16, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
}

void SCmpGeF16Sopc::execute(amdgpu::Wavefront &wf) { (void)wf; }

SCmpOF32Sopc::SCmpOF32Sopc(const MachineInst *inst)
    : Sopc("s_cmp_o_f32", reinterpret_cast<const OpEncoding *>(inst)),
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

void SCmpOF32Sopc::execute(amdgpu::Wavefront &wf) { (void)wf; }

SCmpOF16Sopc::SCmpOF16Sopc(const MachineInst *inst)
    : Sopc("s_cmp_o_f16", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(16, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(16, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
}

void SCmpOF16Sopc::execute(amdgpu::Wavefront &wf) { (void)wf; }

SCmpUF32Sopc::SCmpUF32Sopc(const MachineInst *inst)
    : Sopc("s_cmp_u_f32", reinterpret_cast<const OpEncoding *>(inst)),
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

void SCmpUF32Sopc::execute(amdgpu::Wavefront &wf) { (void)wf; }

SCmpUF16Sopc::SCmpUF16Sopc(const MachineInst *inst)
    : Sopc("s_cmp_u_f16", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(16, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(16, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
}

void SCmpUF16Sopc::execute(amdgpu::Wavefront &wf) { (void)wf; }

SCmpNgeF32Sopc::SCmpNgeF32Sopc(const MachineInst *inst)
    : Sopc("s_cmp_nge_f32", reinterpret_cast<const OpEncoding *>(inst)),
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

void SCmpNgeF32Sopc::execute(amdgpu::Wavefront &wf) { (void)wf; }

SCmpNgeF16Sopc::SCmpNgeF16Sopc(const MachineInst *inst)
    : Sopc("s_cmp_nge_f16", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(16, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(16, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
}

void SCmpNgeF16Sopc::execute(amdgpu::Wavefront &wf) { (void)wf; }

SCmpNlgF32Sopc::SCmpNlgF32Sopc(const MachineInst *inst)
    : Sopc("s_cmp_nlg_f32", reinterpret_cast<const OpEncoding *>(inst)),
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

void SCmpNlgF32Sopc::execute(amdgpu::Wavefront &wf) { (void)wf; }

SCmpNlgF16Sopc::SCmpNlgF16Sopc(const MachineInst *inst)
    : Sopc("s_cmp_nlg_f16", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(16, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(16, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
}

void SCmpNlgF16Sopc::execute(amdgpu::Wavefront &wf) { (void)wf; }

SCmpNgtF32Sopc::SCmpNgtF32Sopc(const MachineInst *inst)
    : Sopc("s_cmp_ngt_f32", reinterpret_cast<const OpEncoding *>(inst)),
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

void SCmpNgtF32Sopc::execute(amdgpu::Wavefront &wf) { (void)wf; }

SCmpNgtF16Sopc::SCmpNgtF16Sopc(const MachineInst *inst)
    : Sopc("s_cmp_ngt_f16", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(16, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(16, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
}

void SCmpNgtF16Sopc::execute(amdgpu::Wavefront &wf) { (void)wf; }

SCmpNleF32Sopc::SCmpNleF32Sopc(const MachineInst *inst)
    : Sopc("s_cmp_nle_f32", reinterpret_cast<const OpEncoding *>(inst)),
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

void SCmpNleF32Sopc::execute(amdgpu::Wavefront &wf) { (void)wf; }

SCmpNleF16Sopc::SCmpNleF16Sopc(const MachineInst *inst)
    : Sopc("s_cmp_nle_f16", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(16, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(16, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
}

void SCmpNleF16Sopc::execute(amdgpu::Wavefront &wf) { (void)wf; }

SCmpNeqF32Sopc::SCmpNeqF32Sopc(const MachineInst *inst)
    : Sopc("s_cmp_neq_f32", reinterpret_cast<const OpEncoding *>(inst)),
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

void SCmpNeqF32Sopc::execute(amdgpu::Wavefront &wf) { (void)wf; }

SCmpNeqF16Sopc::SCmpNeqF16Sopc(const MachineInst *inst)
    : Sopc("s_cmp_neq_f16", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(16, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(16, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
}

void SCmpNeqF16Sopc::execute(amdgpu::Wavefront &wf) { (void)wf; }

SCmpNltF32Sopc::SCmpNltF32Sopc(const MachineInst *inst)
    : Sopc("s_cmp_nlt_f32", reinterpret_cast<const OpEncoding *>(inst)),
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

void SCmpNltF32Sopc::execute(amdgpu::Wavefront &wf) { (void)wf; }

SCmpNltF16Sopc::SCmpNltF16Sopc(const MachineInst *inst)
    : Sopc("s_cmp_nlt_f16", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(16, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(16, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const SopcInstLiteralMachineInst *>(inst)->simm32));
}

void SCmpNltF16Sopc::execute(amdgpu::Wavefront &wf) { (void)wf; }

} // namespace rdna4
} // namespace rocjitsu

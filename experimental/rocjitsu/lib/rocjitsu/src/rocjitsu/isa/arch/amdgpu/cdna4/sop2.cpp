// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/cdna4/sop2.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"
#include "util/except.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

namespace rocjitsu {
namespace cdna4 {

SAddU32Sop2::SAddU32Sop2(const MachineInst *inst)
    : Sop2("s_add_u32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SAddU32Sop2::execute(amdgpu::Wavefront &wf) {
  uint32_t s0 = ssrc0.read_scalar(wf);
  uint32_t s1 = ssrc1.read_scalar(wf);
  uint64_t wide = static_cast<uint64_t>(s0) + static_cast<uint64_t>(s1);
  sdst.write_scalar(wf, static_cast<uint32_t>(wide));
  wf.write_scc(wide > 0xFFFFFFFFULL);
}

SSubU32Sop2::SSubU32Sop2(const MachineInst *inst)
    : Sop2("s_sub_u32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SSubU32Sop2::execute(amdgpu::Wavefront &wf) {
  uint32_t s0 = ssrc0.read_scalar(wf);
  uint32_t s1 = ssrc1.read_scalar(wf);
  sdst.write_scalar(wf, s0 - s1);
  wf.write_scc(s0 < s1);
}

SAddI32Sop2::SAddI32Sop2(const MachineInst *inst)
    : Sop2("s_add_i32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SAddI32Sop2::execute(amdgpu::Wavefront &wf) {
  int32_t s0 = static_cast<int32_t>(ssrc0.read_scalar(wf));
  int32_t s1 = static_cast<int32_t>(ssrc1.read_scalar(wf));
  int64_t wide = static_cast<int64_t>(s0) + static_cast<int64_t>(s1);
  int32_t result = static_cast<int32_t>(wide);
  sdst.write_scalar(wf, static_cast<uint32_t>(result));
  wf.write_scc(wide != static_cast<int64_t>(result));
}

SSubI32Sop2::SSubI32Sop2(const MachineInst *inst)
    : Sop2("s_sub_i32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SSubI32Sop2::execute(amdgpu::Wavefront &wf) {
  int32_t s0 = static_cast<int32_t>(ssrc0.read_scalar(wf));
  int32_t s1 = static_cast<int32_t>(ssrc1.read_scalar(wf));
  int64_t wide = static_cast<int64_t>(s0) - static_cast<int64_t>(s1);
  int32_t result = static_cast<int32_t>(wide);
  sdst.write_scalar(wf, static_cast<uint32_t>(result));
  wf.write_scc(wide != static_cast<int64_t>(result));
}

SAddcU32Sop2::SAddcU32Sop2(const MachineInst *inst)
    : Sop2("s_addc_u32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SAddcU32Sop2::execute(amdgpu::Wavefront &wf) {
  uint32_t s0 = ssrc0.read_scalar(wf);
  uint32_t s1 = ssrc1.read_scalar(wf);
  uint64_t wide = static_cast<uint64_t>(s0) + static_cast<uint64_t>(s1) + (wf.read_scc() ? 1u : 0u);
  sdst.write_scalar(wf, static_cast<uint32_t>(wide));
  wf.write_scc(wide > 0xFFFFFFFFULL);
}

SSubbU32Sop2::SSubbU32Sop2(const MachineInst *inst)
    : Sop2("s_subb_u32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SSubbU32Sop2::execute(amdgpu::Wavefront &wf) {
  uint32_t s0 = ssrc0.read_scalar(wf);
  uint32_t s1 = ssrc1.read_scalar(wf);
  uint32_t bin = wf.read_scc() ? 1u : 0u;
  uint64_t wide = static_cast<uint64_t>(s0) - static_cast<uint64_t>(s1) - bin;
  sdst.write_scalar(wf, static_cast<uint32_t>(wide));
  wf.write_scc(static_cast<uint64_t>(s0) < static_cast<uint64_t>(s1) + bin);
}

SMinI32Sop2::SMinI32Sop2(const MachineInst *inst)
    : Sop2("s_min_i32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SMinI32Sop2::execute(amdgpu::Wavefront &wf) {
  int32_t s0 = static_cast<int32_t>(ssrc0.read_scalar(wf));
  int32_t s1 = static_cast<int32_t>(ssrc1.read_scalar(wf));
  int32_t result = s0 < s1 ? s0 : s1;
  sdst.write_scalar(wf, static_cast<uint32_t>(result));
  wf.write_scc(s0 < s1);
}

SMinU32Sop2::SMinU32Sop2(const MachineInst *inst)
    : Sop2("s_min_u32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SMinU32Sop2::execute(amdgpu::Wavefront &wf) {
  uint32_t s0 = ssrc0.read_scalar(wf);
  uint32_t s1 = ssrc1.read_scalar(wf);
  uint32_t result = s0 < s1 ? s0 : s1;
  sdst.write_scalar(wf, result);
  wf.write_scc(s0 < s1);
}

SMaxI32Sop2::SMaxI32Sop2(const MachineInst *inst)
    : Sop2("s_max_i32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SMaxI32Sop2::execute(amdgpu::Wavefront &wf) {
  int32_t s0 = static_cast<int32_t>(ssrc0.read_scalar(wf));
  int32_t s1 = static_cast<int32_t>(ssrc1.read_scalar(wf));
  int32_t result = s0 > s1 ? s0 : s1;
  sdst.write_scalar(wf, static_cast<uint32_t>(result));
  wf.write_scc(s0 > s1);
}

SMaxU32Sop2::SMaxU32Sop2(const MachineInst *inst)
    : Sop2("s_max_u32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SMaxU32Sop2::execute(amdgpu::Wavefront &wf) {
  uint32_t s0 = ssrc0.read_scalar(wf);
  uint32_t s1 = ssrc1.read_scalar(wf);
  uint32_t result = s0 > s1 ? s0 : s1;
  sdst.write_scalar(wf, result);
  wf.write_scc(s0 > s1);
}

SCselectB32Sop2::SCselectB32Sop2(const MachineInst *inst)
    : Sop2("s_cselect_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SCselectB32Sop2::execute(amdgpu::Wavefront &wf) {
  sdst.write_scalar(wf, wf.read_scc() ? ssrc0.read_scalar(wf) : ssrc1.read_scalar(wf));
}

SCselectB64Sop2::SCselectB64Sop2(const MachineInst *inst)
    : Sop2("s_cselect_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SCselectB64Sop2::execute(amdgpu::Wavefront &wf) {
  sdst.write_scalar64(wf, wf.read_scc() ? ssrc0.read_scalar64(wf) : ssrc1.read_scalar64(wf));
}

SAndB32Sop2::SAndB32Sop2(const MachineInst *inst)
    : Sop2("s_and_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SAndB32Sop2::execute(amdgpu::Wavefront &wf) {
  uint32_t s0 = ssrc0.read_scalar(wf);
  uint32_t s1 = ssrc1.read_scalar(wf);
  uint32_t result = s0 & s1;
  sdst.write_scalar(wf, result);
  wf.write_scc(result != 0);
}

SAndB64Sop2::SAndB64Sop2(const MachineInst *inst)
    : Sop2("s_and_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SAndB64Sop2::execute(amdgpu::Wavefront &wf) {
  uint64_t s0 = ssrc0.read_scalar64(wf);
  uint64_t s1 = ssrc1.read_scalar64(wf);
  uint64_t result = s0 & s1;
  sdst.write_scalar64(wf, result);
  wf.write_scc(result != 0);
}

SOrB32Sop2::SOrB32Sop2(const MachineInst *inst)
    : Sop2("s_or_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SOrB32Sop2::execute(amdgpu::Wavefront &wf) {
  uint32_t s0 = ssrc0.read_scalar(wf);
  uint32_t s1 = ssrc1.read_scalar(wf);
  uint32_t result = s0 | s1;
  sdst.write_scalar(wf, result);
  wf.write_scc(result != 0);
}

SOrB64Sop2::SOrB64Sop2(const MachineInst *inst)
    : Sop2("s_or_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SOrB64Sop2::execute(amdgpu::Wavefront &wf) {
  uint64_t s0 = ssrc0.read_scalar64(wf);
  uint64_t s1 = ssrc1.read_scalar64(wf);
  uint64_t result = s0 | s1;
  sdst.write_scalar64(wf, result);
  wf.write_scc(result != 0);
}

SXorB32Sop2::SXorB32Sop2(const MachineInst *inst)
    : Sop2("s_xor_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SXorB32Sop2::execute(amdgpu::Wavefront &wf) {
  uint32_t s0 = ssrc0.read_scalar(wf);
  uint32_t s1 = ssrc1.read_scalar(wf);
  uint32_t result = s0 ^ s1;
  sdst.write_scalar(wf, result);
  wf.write_scc(result != 0);
}

SXorB64Sop2::SXorB64Sop2(const MachineInst *inst)
    : Sop2("s_xor_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SXorB64Sop2::execute(amdgpu::Wavefront &wf) {
  uint64_t s0 = ssrc0.read_scalar64(wf);
  uint64_t s1 = ssrc1.read_scalar64(wf);
  uint64_t result = s0 ^ s1;
  sdst.write_scalar64(wf, result);
  wf.write_scc(result != 0);
}

SAndn2B32Sop2::SAndn2B32Sop2(const MachineInst *inst)
    : Sop2("s_andn2_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SAndn2B32Sop2::execute(amdgpu::Wavefront &wf) {
  uint32_t s0 = ssrc0.read_scalar(wf);
  uint32_t s1 = ssrc1.read_scalar(wf);
  uint32_t result = s0 & ~s1;
  sdst.write_scalar(wf, result);
  wf.write_scc(result != 0);
}

SAndn2B64Sop2::SAndn2B64Sop2(const MachineInst *inst)
    : Sop2("s_andn2_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SAndn2B64Sop2::execute(amdgpu::Wavefront &wf) {
  uint64_t s0 = ssrc0.read_scalar64(wf);
  uint64_t s1 = ssrc1.read_scalar64(wf);
  uint64_t result = s0 & ~s1;
  sdst.write_scalar64(wf, result);
  wf.write_scc(result != 0);
}

SOrn2B32Sop2::SOrn2B32Sop2(const MachineInst *inst)
    : Sop2("s_orn2_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SOrn2B32Sop2::execute(amdgpu::Wavefront &wf) {
  uint32_t s0 = ssrc0.read_scalar(wf);
  uint32_t s1 = ssrc1.read_scalar(wf);
  uint32_t result = s0 | ~s1;
  sdst.write_scalar(wf, result);
  wf.write_scc(result != 0);
}

SOrn2B64Sop2::SOrn2B64Sop2(const MachineInst *inst)
    : Sop2("s_orn2_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SOrn2B64Sop2::execute(amdgpu::Wavefront &wf) {
  uint64_t s0 = ssrc0.read_scalar64(wf);
  uint64_t s1 = ssrc1.read_scalar64(wf);
  uint64_t result = s0 | ~s1;
  sdst.write_scalar64(wf, result);
  wf.write_scc(result != 0);
}

SNandB32Sop2::SNandB32Sop2(const MachineInst *inst)
    : Sop2("s_nand_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SNandB32Sop2::execute(amdgpu::Wavefront &wf) {
  uint32_t s0 = ssrc0.read_scalar(wf);
  uint32_t s1 = ssrc1.read_scalar(wf);
  uint32_t result = ~(s0 & s1);
  sdst.write_scalar(wf, result);
  wf.write_scc(result != 0);
}

SNandB64Sop2::SNandB64Sop2(const MachineInst *inst)
    : Sop2("s_nand_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SNandB64Sop2::execute(amdgpu::Wavefront &wf) {
  uint64_t s0 = ssrc0.read_scalar64(wf);
  uint64_t s1 = ssrc1.read_scalar64(wf);
  uint64_t result = ~(s0 & s1);
  sdst.write_scalar64(wf, result);
  wf.write_scc(result != 0);
}

SNorB32Sop2::SNorB32Sop2(const MachineInst *inst)
    : Sop2("s_nor_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SNorB32Sop2::execute(amdgpu::Wavefront &wf) {
  uint32_t s0 = ssrc0.read_scalar(wf);
  uint32_t s1 = ssrc1.read_scalar(wf);
  uint32_t result = ~(s0 | s1);
  sdst.write_scalar(wf, result);
  wf.write_scc(result != 0);
}

SNorB64Sop2::SNorB64Sop2(const MachineInst *inst)
    : Sop2("s_nor_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SNorB64Sop2::execute(amdgpu::Wavefront &wf) {
  uint64_t s0 = ssrc0.read_scalar64(wf);
  uint64_t s1 = ssrc1.read_scalar64(wf);
  uint64_t result = ~(s0 | s1);
  sdst.write_scalar64(wf, result);
  wf.write_scc(result != 0);
}

SXnorB32Sop2::SXnorB32Sop2(const MachineInst *inst)
    : Sop2("s_xnor_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SXnorB32Sop2::execute(amdgpu::Wavefront &wf) {
  uint32_t s0 = ssrc0.read_scalar(wf);
  uint32_t s1 = ssrc1.read_scalar(wf);
  uint32_t result = ~(s0 ^ s1);
  sdst.write_scalar(wf, result);
  wf.write_scc(result != 0);
}

SXnorB64Sop2::SXnorB64Sop2(const MachineInst *inst)
    : Sop2("s_xnor_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SXnorB64Sop2::execute(amdgpu::Wavefront &wf) {
  uint64_t s0 = ssrc0.read_scalar64(wf);
  uint64_t s1 = ssrc1.read_scalar64(wf);
  uint64_t result = ~(s0 ^ s1);
  sdst.write_scalar64(wf, result);
  wf.write_scc(result != 0);
}

SLshlB32Sop2::SLshlB32Sop2(const MachineInst *inst)
    : Sop2("s_lshl_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SLshlB32Sop2::execute(amdgpu::Wavefront &wf) {
  uint32_t s0 = ssrc0.read_scalar(wf);
  uint32_t s1 = ssrc1.read_scalar(wf);
  uint32_t result = s0 << (s1 & 31u);
  sdst.write_scalar(wf, result);
  wf.write_scc(result != 0);
}

SLshlB64Sop2::SLshlB64Sop2(const MachineInst *inst)
    : Sop2("s_lshl_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SLshlB64Sop2::execute(amdgpu::Wavefront &wf) {
  uint64_t s0 = ssrc0.read_scalar64(wf);
  uint64_t s1 = ssrc1.read_scalar64(wf);
  uint64_t result = s0 << (s1 & 63u);
  sdst.write_scalar64(wf, result);
  wf.write_scc(result != 0);
}

SLshrB32Sop2::SLshrB32Sop2(const MachineInst *inst)
    : Sop2("s_lshr_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SLshrB32Sop2::execute(amdgpu::Wavefront &wf) {
  uint32_t s0 = ssrc0.read_scalar(wf);
  uint32_t s1 = ssrc1.read_scalar(wf);
  uint32_t result = s0 >> (s1 & 31u);
  sdst.write_scalar(wf, result);
  wf.write_scc(result != 0);
}

SLshrB64Sop2::SLshrB64Sop2(const MachineInst *inst)
    : Sop2("s_lshr_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SLshrB64Sop2::execute(amdgpu::Wavefront &wf) {
  uint64_t s0 = ssrc0.read_scalar64(wf);
  uint64_t s1 = ssrc1.read_scalar64(wf);
  uint64_t result = s0 >> (s1 & 63u);
  sdst.write_scalar64(wf, result);
  wf.write_scc(result != 0);
}

SAshrI32Sop2::SAshrI32Sop2(const MachineInst *inst)
    : Sop2("s_ashr_i32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SAshrI32Sop2::execute(amdgpu::Wavefront &wf) {
  int32_t s0 = static_cast<int32_t>(ssrc0.read_scalar(wf));
  int32_t s1 = static_cast<int32_t>(ssrc1.read_scalar(wf));
  int32_t result = s0 >> (s1 & 31);
  sdst.write_scalar(wf, static_cast<uint32_t>(result));
  wf.write_scc(result != 0);
}

SAshrI64Sop2::SAshrI64Sop2(const MachineInst *inst)
    : Sop2("s_ashr_i64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SAshrI64Sop2::execute(amdgpu::Wavefront &wf) {
  int64_t s0 = static_cast<int64_t>(ssrc0.read_scalar64(wf));
  int64_t s1 = static_cast<int64_t>(ssrc1.read_scalar64(wf));
  int64_t result = s0 >> (s1 & 63);
  sdst.write_scalar64(wf, static_cast<uint64_t>(result));
  wf.write_scc(result != 0);
}

SBfmB32Sop2::SBfmB32Sop2(const MachineInst *inst)
    : Sop2("s_bfm_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SBfmB32Sop2::execute(amdgpu::Wavefront &wf) {
  uint32_t s0 = ssrc0.read_scalar(wf);
  uint32_t s1 = ssrc1.read_scalar(wf);
  uint32_t count = s0 & 31u;
  uint32_t offset = s1 & 31u;
  uint32_t result = count == 0 ? 0 : ((1u << count) - 1) << offset;
  sdst.write_scalar(wf, result);
}

SBfmB64Sop2::SBfmB64Sop2(const MachineInst *inst)
    : Sop2("s_bfm_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SBfmB64Sop2::execute(amdgpu::Wavefront &wf) {
  uint64_t s0 = ssrc0.read_scalar64(wf);
  uint64_t s1 = ssrc1.read_scalar64(wf);
  uint64_t count = s0 & 63u;
  uint64_t offset = s1 & 63u;
  uint64_t result = count == 0 ? 0 : ((1ULL << count) - 1) << offset;
  sdst.write_scalar64(wf, result);
}

SMulI32Sop2::SMulI32Sop2(const MachineInst *inst)
    : Sop2("s_mul_i32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SMulI32Sop2::execute(amdgpu::Wavefront &wf) {
  int32_t s0 = static_cast<int32_t>(ssrc0.read_scalar(wf));
  int32_t s1 = static_cast<int32_t>(ssrc1.read_scalar(wf));
  sdst.write_scalar(wf,
                    static_cast<uint32_t>(static_cast<uint32_t>(s0) * static_cast<uint32_t>(s1)));
}

SBfeU32Sop2::SBfeU32Sop2(const MachineInst *inst)
    : Sop2("s_bfe_u32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SBfeU32Sop2::execute(amdgpu::Wavefront &wf) {
  uint32_t base = ssrc0.read_scalar(wf);
  uint32_t field = ssrc1.read_scalar(wf);
  uint32_t offset = field & 31u;
  uint32_t width = (field >> 16) & 127u;
  if (width == 0) {
    sdst.write_scalar(wf, 0);
    wf.write_scc(false);
  } else {
    uint32_t mask = width >= 32 ? ~0u : ((1u << width) - 1);
    uint32_t extracted = (base >> offset) & mask;
    sdst.write_scalar(wf, extracted);
    wf.write_scc(extracted != 0);
  }
}

SBfeI32Sop2::SBfeI32Sop2(const MachineInst *inst)
    : Sop2("s_bfe_i32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SBfeI32Sop2::execute(amdgpu::Wavefront &wf) {
  uint32_t base = ssrc0.read_scalar(wf);
  uint32_t field = ssrc1.read_scalar(wf);
  uint32_t offset = field & 31u;
  uint32_t width = (field >> 16) & 127u;
  if (width == 0) {
    sdst.write_scalar(wf, 0);
    wf.write_scc(false);
  } else {
    uint32_t mask = width >= 32 ? ~0u : ((1u << width) - 1);
    uint32_t extracted = (base >> offset) & mask;
    if (width < 32 && (extracted & (1u << (width - 1))))
      extracted |= ~mask;
    sdst.write_scalar(wf, extracted);
    wf.write_scc(extracted != 0);
  }
}

SBfeU64Sop2::SBfeU64Sop2(const MachineInst *inst)
    : Sop2("s_bfe_u64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SBfeU64Sop2::execute(amdgpu::Wavefront &wf) {
  uint64_t base = ssrc0.read_scalar64(wf);
  uint32_t field = ssrc1.read_scalar(wf);
  uint32_t offset = field & 63u;
  uint32_t width = (field >> 16) & 127u;
  if (width == 0) {
    sdst.write_scalar64(wf, 0);
    wf.write_scc(false);
  } else {
    uint64_t mask = width >= 64 ? ~0ULL : ((1ULL << width) - 1);
    uint64_t extracted = (base >> offset) & mask;
    sdst.write_scalar64(wf, extracted);
    wf.write_scc(extracted != 0);
  }
}

SBfeI64Sop2::SBfeI64Sop2(const MachineInst *inst)
    : Sop2("s_bfe_i64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SBfeI64Sop2::execute(amdgpu::Wavefront &wf) {
  uint64_t base = ssrc0.read_scalar64(wf);
  uint32_t field = ssrc1.read_scalar(wf);
  uint32_t offset = field & 63u;
  uint32_t width = (field >> 16) & 127u;
  if (width == 0) {
    sdst.write_scalar64(wf, 0);
    wf.write_scc(false);
  } else {
    uint64_t mask = width >= 64 ? ~0ULL : ((1ULL << width) - 1);
    uint64_t extracted = (base >> offset) & mask;
    if (width < 64 && (extracted & (1ULL << (width - 1))))
      extracted |= ~mask;
    sdst.write_scalar64(wf, extracted);
    wf.write_scc(extracted != 0);
  }
}

SCbranchGForkSop2::SCbranchGForkSop2(const MachineInst *inst)
    : Sop2("s_cbranch_g_fork", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(64, OperandType::OPR_SSRC_NOLIT, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(64, OperandType::OPR_SSRC_NOLIT, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SCbranchGForkSop2::execute(amdgpu::Wavefront &wf) { (void)wf; }

SAbsdiffI32Sop2::SAbsdiffI32Sop2(const MachineInst *inst)
    : Sop2("s_absdiff_i32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SAbsdiffI32Sop2::execute(amdgpu::Wavefront &wf) {
  int32_t s0 = static_cast<int32_t>(ssrc0.read_scalar(wf));
  int32_t s1 = static_cast<int32_t>(ssrc1.read_scalar(wf));
  int32_t result = s0 > s1 ? s0 - s1 : s1 - s0;
  sdst.write_scalar(wf, static_cast<uint32_t>(result));
  wf.write_scc(result != 0);
}

SRfeRestoreB64Sop2::SRfeRestoreB64Sop2(const MachineInst *inst)
    : Sop2("s_rfe_restore_b64", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SRfeRestoreB64Sop2::execute(amdgpu::Wavefront &wf) { (void)wf; }

SMulHiU32Sop2::SMulHiU32Sop2(const MachineInst *inst)
    : Sop2("s_mul_hi_u32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SMulHiU32Sop2::execute(amdgpu::Wavefront &wf) {
  uint32_t s0 = ssrc0.read_scalar(wf);
  uint32_t s1 = ssrc1.read_scalar(wf);
  uint64_t wide = static_cast<uint64_t>(s0) * static_cast<uint64_t>(s1);
  sdst.write_scalar(wf, static_cast<uint32_t>(wide >> 32));
}

SMulHiI32Sop2::SMulHiI32Sop2(const MachineInst *inst)
    : Sop2("s_mul_hi_i32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SMulHiI32Sop2::execute(amdgpu::Wavefront &wf) {
  int32_t s0 = static_cast<int32_t>(ssrc0.read_scalar(wf));
  int32_t s1 = static_cast<int32_t>(ssrc1.read_scalar(wf));
  int64_t wide = static_cast<int64_t>(s0) * static_cast<int64_t>(s1);
  sdst.write_scalar(wf, static_cast<uint32_t>(static_cast<uint64_t>(wide) >> 32));
}

SLshl1AddU32Sop2::SLshl1AddU32Sop2(const MachineInst *inst)
    : Sop2("s_lshl1_add_u32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SLshl1AddU32Sop2::execute(amdgpu::Wavefront &wf) {
  uint32_t s0 = ssrc0.read_scalar(wf);
  uint32_t s1 = ssrc1.read_scalar(wf);
  uint64_t wide = (static_cast<uint64_t>(s0) << 1u) + static_cast<uint64_t>(s1);
  sdst.write_scalar(wf, static_cast<uint32_t>(wide));
  wf.write_scc(wide > 0xFFFFFFFFULL);
}

SLshl2AddU32Sop2::SLshl2AddU32Sop2(const MachineInst *inst)
    : Sop2("s_lshl2_add_u32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SLshl2AddU32Sop2::execute(amdgpu::Wavefront &wf) {
  uint32_t s0 = ssrc0.read_scalar(wf);
  uint32_t s1 = ssrc1.read_scalar(wf);
  uint64_t wide = (static_cast<uint64_t>(s0) << 2u) + static_cast<uint64_t>(s1);
  sdst.write_scalar(wf, static_cast<uint32_t>(wide));
  wf.write_scc(wide > 0xFFFFFFFFULL);
}

SLshl3AddU32Sop2::SLshl3AddU32Sop2(const MachineInst *inst)
    : Sop2("s_lshl3_add_u32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SLshl3AddU32Sop2::execute(amdgpu::Wavefront &wf) {
  uint32_t s0 = ssrc0.read_scalar(wf);
  uint32_t s1 = ssrc1.read_scalar(wf);
  uint64_t wide = (static_cast<uint64_t>(s0) << 3u) + static_cast<uint64_t>(s1);
  sdst.write_scalar(wf, static_cast<uint32_t>(wide));
  wf.write_scc(wide > 0xFFFFFFFFULL);
}

SLshl4AddU32Sop2::SLshl4AddU32Sop2(const MachineInst *inst)
    : Sop2("s_lshl4_add_u32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SLshl4AddU32Sop2::execute(amdgpu::Wavefront &wf) {
  uint32_t s0 = ssrc0.read_scalar(wf);
  uint32_t s1 = ssrc1.read_scalar(wf);
  uint64_t wide = (static_cast<uint64_t>(s0) << 4u) + static_cast<uint64_t>(s1);
  sdst.write_scalar(wf, static_cast<uint32_t>(wide));
  wf.write_scc(wide > 0xFFFFFFFFULL);
}

SPackLlB32B16Sop2::SPackLlB32B16Sop2(const MachineInst *inst)
    : Sop2("s_pack_ll_b32_b16", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(16, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(16, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SPackLlB32B16Sop2::execute(amdgpu::Wavefront &wf) {
  uint32_t s0 = ssrc0.read_scalar(wf);
  uint32_t s1 = ssrc1.read_scalar(wf);
  sdst.write_scalar(wf, (s0 & 0xFFFFu) | ((s1 & 0xFFFFu) << 16));
}

SPackLhB32B16Sop2::SPackLhB32B16Sop2(const MachineInst *inst)
    : Sop2("s_pack_lh_b32_b16", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(16, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SPackLhB32B16Sop2::execute(amdgpu::Wavefront &wf) {
  uint32_t s0 = ssrc0.read_scalar(wf);
  uint32_t s1 = ssrc1.read_scalar(wf);
  sdst.write_scalar(wf, (s0 & 0xFFFFu) | (s1 & 0xFFFF0000u));
}

SPackHhB32B16Sop2::SPackHhB32B16Sop2(const MachineInst *inst)
    : Sop2("s_pack_hh_b32_b16", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0),
      ssrc1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc1) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  src_operands_.emplace_back(&ssrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc1 == 255)
    ssrc1 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop2InstLiteralMachineInst *>(inst)->simm32));
}

void SPackHhB32B16Sop2::execute(amdgpu::Wavefront &wf) {
  uint32_t s0 = ssrc0.read_scalar(wf);
  uint32_t s1 = ssrc1.read_scalar(wf);
  sdst.write_scalar(wf, ((s0 >> 16) & 0xFFFFu) | (s1 & 0xFFFF0000u));
}

} // namespace cdna4
} // namespace rocjitsu

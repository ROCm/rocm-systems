// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/cdna4/sop1.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"
#include "util/except.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

namespace rocjitsu {
namespace cdna4 {

SMovB32Sop1::SMovB32Sop1(const MachineInst *inst)
    : Sop1("s_mov_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SMovB32Sop1::execute(amdgpu::Wavefront &wf) { sdst.write_scalar(wf, ssrc0.read_scalar(wf)); }

SMovB64Sop1::SMovB64Sop1(const MachineInst *inst)
    : Sop1("s_mov_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SMovB64Sop1::execute(amdgpu::Wavefront &wf) {
  sdst.write_scalar64(wf, ssrc0.read_scalar64(wf));
}

SCmovB32Sop1::SCmovB32Sop1(const MachineInst *inst)
    : Sop1("s_cmov_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SCmovB32Sop1::execute(amdgpu::Wavefront &wf) {
  if (wf.read_scc())
    sdst.write_scalar(wf, ssrc0.read_scalar(wf));
}

SCmovB64Sop1::SCmovB64Sop1(const MachineInst *inst)
    : Sop1("s_cmov_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SCmovB64Sop1::execute(amdgpu::Wavefront &wf) {
  if (wf.read_scc())
    sdst.write_scalar64(wf, ssrc0.read_scalar64(wf));
}

SNotB32Sop1::SNotB32Sop1(const MachineInst *inst)
    : Sop1("s_not_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SNotB32Sop1::execute(amdgpu::Wavefront &wf) {
  uint32_t val = ssrc0.read_scalar(wf);
  uint32_t result = ~val;
  sdst.write_scalar(wf, result);
  wf.write_scc(result != 0);
}

SNotB64Sop1::SNotB64Sop1(const MachineInst *inst)
    : Sop1("s_not_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SNotB64Sop1::execute(amdgpu::Wavefront &wf) {
  uint64_t val = ssrc0.read_scalar64(wf);
  uint64_t result = ~val;
  sdst.write_scalar64(wf, result);
  wf.write_scc(result != 0);
}

SWqmB32Sop1::SWqmB32Sop1(const MachineInst *inst)
    : Sop1("s_wqm_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SWqmB32Sop1::execute(amdgpu::Wavefront &wf) {
  uint32_t val = ssrc0.read_scalar(wf);
  uint32_t result = 0;
  for (int q = 0; q < 8; ++q)
    if (val & (0xFu << (q * 4)))
      result |= (0xFu << (q * 4));
  sdst.write_scalar(wf, result);
  wf.write_scc(result != 0);
}

SWqmB64Sop1::SWqmB64Sop1(const MachineInst *inst)
    : Sop1("s_wqm_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SWqmB64Sop1::execute(amdgpu::Wavefront &wf) {
  uint64_t val = ssrc0.read_scalar64(wf);
  uint64_t result = 0;
  for (int q = 0; q < 16; ++q)
    if (val & (0xFULL << (q * 4)))
      result |= (0xFULL << (q * 4));
  sdst.write_scalar64(wf, result);
  wf.write_scc(result != 0);
}

SBrevB32Sop1::SBrevB32Sop1(const MachineInst *inst)
    : Sop1("s_brev_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SBrevB32Sop1::execute(amdgpu::Wavefront &wf) {
  uint32_t val = ssrc0.read_scalar(wf);
  uint32_t result = 0;
  for (int i = 0; i < 32; ++i)
    result |= ((val >> i) & 1) << (31 - i);
  sdst.write_scalar(wf, result);
  wf.write_scc(result != 0);
}

SBrevB64Sop1::SBrevB64Sop1(const MachineInst *inst)
    : Sop1("s_brev_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SBrevB64Sop1::execute(amdgpu::Wavefront &wf) {
  uint64_t val = ssrc0.read_scalar64(wf);
  uint64_t result = 0;
  for (int i = 0; i < 64; ++i)
    result |= ((val >> i) & 1) << (63 - i);
  sdst.write_scalar64(wf, result);
  wf.write_scc(result != 0);
}

SBcnt0I32B32Sop1::SBcnt0I32B32Sop1(const MachineInst *inst)
    : Sop1("s_bcnt0_i32_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SBcnt0I32B32Sop1::execute(amdgpu::Wavefront &wf) {
  uint32_t val = ssrc0.read_scalar(wf);
  uint32_t result = static_cast<uint32_t>(std::popcount(~val));
  sdst.write_scalar(wf, result);
  wf.write_scc(result != 0);
}

SBcnt0I32B64Sop1::SBcnt0I32B64Sop1(const MachineInst *inst)
    : Sop1("s_bcnt0_i32_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SBcnt0I32B64Sop1::execute(amdgpu::Wavefront &wf) {
  uint64_t val = ssrc0.read_scalar64(wf);
  uint64_t result = static_cast<uint64_t>(std::popcount(~val));
  sdst.write_scalar64(wf, result);
  wf.write_scc(result != 0);
}

SBcnt1I32B32Sop1::SBcnt1I32B32Sop1(const MachineInst *inst)
    : Sop1("s_bcnt1_i32_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SBcnt1I32B32Sop1::execute(amdgpu::Wavefront &wf) {
  uint32_t val = ssrc0.read_scalar(wf);
  uint32_t result = static_cast<uint32_t>(std::popcount(val));
  sdst.write_scalar(wf, result);
  wf.write_scc(result != 0);
}

SBcnt1I32B64Sop1::SBcnt1I32B64Sop1(const MachineInst *inst)
    : Sop1("s_bcnt1_i32_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SBcnt1I32B64Sop1::execute(amdgpu::Wavefront &wf) {
  uint64_t val = ssrc0.read_scalar64(wf);
  uint64_t result = static_cast<uint64_t>(std::popcount(val));
  sdst.write_scalar64(wf, result);
  wf.write_scc(result != 0);
}

SFf0I32B32Sop1::SFf0I32B32Sop1(const MachineInst *inst)
    : Sop1("s_ff0_i32_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SFf0I32B32Sop1::execute(amdgpu::Wavefront &wf) {
  uint32_t val = ssrc0.read_scalar(wf);
  uint32_t result =
      val == ~0u ? static_cast<uint32_t>(-1) : static_cast<uint32_t>(std::countr_zero(~val));
  sdst.write_scalar(wf, result);
  wf.write_scc(result != 0);
}

SFf0I32B64Sop1::SFf0I32B64Sop1(const MachineInst *inst)
    : Sop1("s_ff0_i32_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SFf0I32B64Sop1::execute(amdgpu::Wavefront &wf) {
  uint64_t val = ssrc0.read_scalar64(wf);
  uint64_t result = static_cast<uint64_t>(val == ~0ULL ? -1 : std::countr_zero(~val));
  sdst.write_scalar64(wf, result);
  wf.write_scc(result != 0);
}

SFf1I32B32Sop1::SFf1I32B32Sop1(const MachineInst *inst)
    : Sop1("s_ff1_i32_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SFf1I32B32Sop1::execute(amdgpu::Wavefront &wf) {
  uint32_t val = ssrc0.read_scalar(wf);
  uint32_t result =
      val == 0 ? static_cast<uint32_t>(-1) : static_cast<uint32_t>(std::countr_zero(val));
  sdst.write_scalar(wf, result);
  wf.write_scc(result != 0);
}

SFf1I32B64Sop1::SFf1I32B64Sop1(const MachineInst *inst)
    : Sop1("s_ff1_i32_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SFf1I32B64Sop1::execute(amdgpu::Wavefront &wf) {
  uint64_t val = ssrc0.read_scalar64(wf);
  uint64_t result = static_cast<uint64_t>(val == 0 ? -1 : std::countr_zero(val));
  sdst.write_scalar64(wf, result);
  wf.write_scc(result != 0);
}

SFlbitI32B32Sop1::SFlbitI32B32Sop1(const MachineInst *inst)
    : Sop1("s_flbit_i32_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SFlbitI32B32Sop1::execute(amdgpu::Wavefront &wf) {
  uint32_t val = ssrc0.read_scalar(wf);
  uint32_t result =
      val == 0 ? static_cast<uint32_t>(-1) : static_cast<uint32_t>(std::countl_zero(val));
  sdst.write_scalar(wf, result);
  wf.write_scc(result != 0);
}

SFlbitI32B64Sop1::SFlbitI32B64Sop1(const MachineInst *inst)
    : Sop1("s_flbit_i32_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SFlbitI32B64Sop1::execute(amdgpu::Wavefront &wf) {
  uint64_t val = ssrc0.read_scalar64(wf);
  uint64_t result = static_cast<uint64_t>(val == 0 ? -1 : std::countl_zero(val));
  sdst.write_scalar64(wf, result);
  wf.write_scc(result != 0);
}

SFlbitI32Sop1::SFlbitI32Sop1(const MachineInst *inst)
    : Sop1("s_flbit_i32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SFlbitI32Sop1::execute(amdgpu::Wavefront &wf) {
  int32_t sval = static_cast<int32_t>(ssrc0.read_scalar(wf));
  uint32_t val = sval < 0 ? ~static_cast<uint32_t>(sval) : static_cast<uint32_t>(sval);
  uint32_t result =
      val == 0 ? static_cast<uint32_t>(-1) : static_cast<uint32_t>(std::countl_zero(val));
  sdst.write_scalar(wf, result);
  wf.write_scc(result != 0);
}

SFlbitI32I64Sop1::SFlbitI32I64Sop1(const MachineInst *inst)
    : Sop1("s_flbit_i32_i64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SFlbitI32I64Sop1::execute(amdgpu::Wavefront &wf) {
  int64_t sval = static_cast<int64_t>(ssrc0.read_scalar64(wf));
  uint64_t uval = sval < 0 ? ~static_cast<uint64_t>(sval) : static_cast<uint64_t>(sval);
  uint32_t result =
      uval == 0 ? static_cast<uint32_t>(-1) : static_cast<uint32_t>(std::countl_zero(uval));
  sdst.write_scalar(wf, result);
  wf.write_scc(result != 0);
}

SSextI32I8Sop1::SSextI32I8Sop1(const MachineInst *inst)
    : Sop1("s_sext_i32_i8", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(8, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        8, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SSextI32I8Sop1::execute(amdgpu::Wavefront &wf) {
  uint32_t val = ssrc0.read_scalar(wf);
  uint32_t result = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(val & 0xFF)));
  sdst.write_scalar(wf, result);
}

SSextI32I16Sop1::SSextI32I16Sop1(const MachineInst *inst)
    : Sop1("s_sext_i32_i16", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(16, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SSextI32I16Sop1::execute(amdgpu::Wavefront &wf) {
  uint32_t val = ssrc0.read_scalar(wf);
  uint32_t result = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(val & 0xFFFF)));
  sdst.write_scalar(wf, result);
}

SBitset0B32Sop1::SBitset0B32Sop1(const MachineInst *inst)
    : Sop1("s_bitset0_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  src_operands_.emplace_back(&sdst);
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SBitset0B32Sop1::execute(amdgpu::Wavefront &wf) {
  uint32_t bit = ssrc0.read_scalar(wf);
  uint32_t result = sdst.read_scalar(wf) & ~(1u << (bit & 31));
  sdst.write_scalar(wf, result);
}

SBitset0B64Sop1::SBitset0B64Sop1(const MachineInst *inst)
    : Sop1("s_bitset0_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  src_operands_.emplace_back(&sdst);
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SBitset0B64Sop1::execute(amdgpu::Wavefront &wf) {
  uint32_t bit = ssrc0.read_scalar(wf);
  uint64_t result = sdst.read_scalar64(wf) & ~(1ULL << (bit & 63));
  sdst.write_scalar64(wf, result);
}

SBitset1B32Sop1::SBitset1B32Sop1(const MachineInst *inst)
    : Sop1("s_bitset1_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  src_operands_.emplace_back(&sdst);
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SBitset1B32Sop1::execute(amdgpu::Wavefront &wf) {
  uint32_t bit = ssrc0.read_scalar(wf);
  uint32_t result = sdst.read_scalar(wf) | (1u << (bit & 31));
  sdst.write_scalar(wf, result);
}

SBitset1B64Sop1::SBitset1B64Sop1(const MachineInst *inst)
    : Sop1("s_bitset1_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  src_operands_.emplace_back(&sdst);
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SBitset1B64Sop1::execute(amdgpu::Wavefront &wf) {
  uint32_t bit = ssrc0.read_scalar(wf);
  uint64_t result = sdst.read_scalar64(wf) | (1ULL << (bit & 63));
  sdst.write_scalar64(wf, result);
}

SGetpcB64Sop1::SGetpcB64Sop1(const MachineInst *inst)
    : Sop1("s_getpc_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst) {
  dst_operands_.emplace_back(&sdst);
}

void SGetpcB64Sop1::execute(amdgpu::Wavefront &wf) { sdst.write_scalar64(wf, wf.pc); }

SSetpcB64Sop1::SSetpcB64Sop1(const MachineInst *inst)
    : Sop1("s_setpc_b64", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SSetpcB64Sop1::execute(amdgpu::Wavefront &wf) { wf.pc = ssrc0.read_scalar64(wf) - size_; }

SSwappcB64Sop1::SSwappcB64Sop1(const MachineInst *inst)
    : Sop1("s_swappc_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SSwappcB64Sop1::execute(amdgpu::Wavefront &wf) {
  uint64_t old_pc = wf.pc;
  wf.pc = ssrc0.read_scalar64(wf) - size_;
  sdst.write_scalar64(wf, old_pc);
}

SRfeB64Sop1::SRfeB64Sop1(const MachineInst *inst)
    : Sop1("s_rfe_b64", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SRfeB64Sop1::execute(amdgpu::Wavefront &wf) { (void)wf; }

SAndSaveexecB64Sop1::SAndSaveexecB64Sop1(const MachineInst *inst)
    : Sop1("s_and_saveexec_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SAndSaveexecB64Sop1::execute(amdgpu::Wavefront &wf) {
  uint64_t old_exec = wf.exec();
  sdst.write_scalar64(wf, old_exec);
  uint64_t src = ssrc0.read_scalar64(wf);
  uint64_t result = old_exec & src;
  wf.set_exec(result);
  wf.write_scc(result != 0);
}

SOrSaveexecB64Sop1::SOrSaveexecB64Sop1(const MachineInst *inst)
    : Sop1("s_or_saveexec_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SOrSaveexecB64Sop1::execute(amdgpu::Wavefront &wf) {
  uint64_t old_exec = wf.exec();
  sdst.write_scalar64(wf, old_exec);
  uint64_t src = ssrc0.read_scalar64(wf);
  uint64_t result = old_exec | src;
  wf.set_exec(result);
  wf.write_scc(result != 0);
}

SXorSaveexecB64Sop1::SXorSaveexecB64Sop1(const MachineInst *inst)
    : Sop1("s_xor_saveexec_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SXorSaveexecB64Sop1::execute(amdgpu::Wavefront &wf) {
  uint64_t old_exec = wf.exec();
  sdst.write_scalar64(wf, old_exec);
  uint64_t src = ssrc0.read_scalar64(wf);
  uint64_t result = old_exec ^ src;
  wf.set_exec(result);
  wf.write_scc(result != 0);
}

SAndn2SaveexecB64Sop1::SAndn2SaveexecB64Sop1(const MachineInst *inst)
    : Sop1("s_andn2_saveexec_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SAndn2SaveexecB64Sop1::execute(amdgpu::Wavefront &wf) {
  uint64_t old_exec = wf.exec();
  sdst.write_scalar64(wf, old_exec);
  uint64_t src = ssrc0.read_scalar64(wf);
  uint64_t result = old_exec & ~src;
  wf.set_exec(result);
  wf.write_scc(result != 0);
}

SOrn2SaveexecB64Sop1::SOrn2SaveexecB64Sop1(const MachineInst *inst)
    : Sop1("s_orn2_saveexec_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SOrn2SaveexecB64Sop1::execute(amdgpu::Wavefront &wf) {
  uint64_t old_exec = wf.exec();
  sdst.write_scalar64(wf, old_exec);
  uint64_t src = ssrc0.read_scalar64(wf);
  uint64_t result = old_exec | ~src;
  wf.set_exec(result);
  wf.write_scc(result != 0);
}

SNandSaveexecB64Sop1::SNandSaveexecB64Sop1(const MachineInst *inst)
    : Sop1("s_nand_saveexec_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SNandSaveexecB64Sop1::execute(amdgpu::Wavefront &wf) {
  uint64_t old_exec = wf.exec();
  sdst.write_scalar64(wf, old_exec);
  uint64_t src = ssrc0.read_scalar64(wf);
  uint64_t result = ~(old_exec & src);
  wf.set_exec(result);
  wf.write_scc(result != 0);
}

SNorSaveexecB64Sop1::SNorSaveexecB64Sop1(const MachineInst *inst)
    : Sop1("s_nor_saveexec_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SNorSaveexecB64Sop1::execute(amdgpu::Wavefront &wf) {
  uint64_t old_exec = wf.exec();
  sdst.write_scalar64(wf, old_exec);
  uint64_t src = ssrc0.read_scalar64(wf);
  uint64_t result = ~(old_exec | src);
  wf.set_exec(result);
  wf.write_scc(result != 0);
}

SXnorSaveexecB64Sop1::SXnorSaveexecB64Sop1(const MachineInst *inst)
    : Sop1("s_xnor_saveexec_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SXnorSaveexecB64Sop1::execute(amdgpu::Wavefront &wf) {
  uint64_t old_exec = wf.exec();
  sdst.write_scalar64(wf, old_exec);
  uint64_t src = ssrc0.read_scalar64(wf);
  uint64_t result = ~(old_exec ^ src);
  wf.set_exec(result);
  wf.write_scc(result != 0);
}

SQuadmaskB32Sop1::SQuadmaskB32Sop1(const MachineInst *inst)
    : Sop1("s_quadmask_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SQuadmaskB32Sop1::execute(amdgpu::Wavefront &wf) {
  uint32_t val = ssrc0.read_scalar(wf);
  uint32_t result = 0;
  for (int q = 0; q < 8; ++q)
    if (val & (0xFu << (q * 4)))
      result |= (1u << q);
  sdst.write_scalar(wf, result);
  wf.write_scc(result != 0);
}

SQuadmaskB64Sop1::SQuadmaskB64Sop1(const MachineInst *inst)
    : Sop1("s_quadmask_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SQuadmaskB64Sop1::execute(amdgpu::Wavefront &wf) {
  uint64_t val = ssrc0.read_scalar64(wf);
  uint64_t result = 0;
  for (int q = 0; q < 16; ++q)
    if (val & (0xFULL << (q * 4)))
      result |= (1ULL << q);
  sdst.write_scalar64(wf, result);
  wf.write_scc(result != 0);
}

SMovrelsB32Sop1::SMovrelsB32Sop1(const MachineInst *inst)
    : Sop1("s_movrels_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SMovrelsB32Sop1::execute(amdgpu::Wavefront &wf) { (void)wf; }

SMovrelsB64Sop1::SMovrelsB64Sop1(const MachineInst *inst)
    : Sop1("s_movrels_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SMovrelsB64Sop1::execute(amdgpu::Wavefront &wf) { (void)wf; }

SMovreldB32Sop1::SMovreldB32Sop1(const MachineInst *inst)
    : Sop1("s_movreld_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SMovreldB32Sop1::execute(amdgpu::Wavefront &wf) { (void)wf; }

SMovreldB64Sop1::SMovreldB64Sop1(const MachineInst *inst)
    : Sop1("s_movreld_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SMovreldB64Sop1::execute(amdgpu::Wavefront &wf) { (void)wf; }

SCbranchJoinSop1::SCbranchJoinSop1(const MachineInst *inst)
    : Sop1("s_cbranch_join", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SCbranchJoinSop1::execute(amdgpu::Wavefront &wf) { (void)wf; }

SAbsI32Sop1::SAbsI32Sop1(const MachineInst *inst)
    : Sop1("s_abs_i32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SAbsI32Sop1::execute(amdgpu::Wavefront &wf) {
  int32_t val = static_cast<int32_t>(ssrc0.read_scalar(wf));
  uint32_t uval = static_cast<uint32_t>(val);
  uint32_t result = val < 0 ? (0u - uval) : uval;
  sdst.write_scalar(wf, result);
  wf.write_scc(result != 0);
}

SSetGprIdxIdxSop1::SSetGprIdxIdxSop1(const MachineInst *inst)
    : Sop1("s_set_gpr_idx_idx", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SSetGprIdxIdxSop1::execute(amdgpu::Wavefront &wf) { (void)wf; }

SAndn1SaveexecB64Sop1::SAndn1SaveexecB64Sop1(const MachineInst *inst)
    : Sop1("s_andn1_saveexec_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SAndn1SaveexecB64Sop1::execute(amdgpu::Wavefront &wf) {
  uint64_t old_exec = wf.exec();
  sdst.write_scalar64(wf, old_exec);
  uint64_t src = ssrc0.read_scalar64(wf);
  uint64_t result = src & ~old_exec;
  wf.set_exec(result);
  wf.write_scc(result != 0);
}

SOrn1SaveexecB64Sop1::SOrn1SaveexecB64Sop1(const MachineInst *inst)
    : Sop1("s_orn1_saveexec_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SOrn1SaveexecB64Sop1::execute(amdgpu::Wavefront &wf) {
  uint64_t old_exec = wf.exec();
  sdst.write_scalar64(wf, old_exec);
  uint64_t src = ssrc0.read_scalar64(wf);
  uint64_t result = src | ~old_exec;
  wf.set_exec(result);
  wf.write_scc(result != 0);
}

SAndn1WrexecB64Sop1::SAndn1WrexecB64Sop1(const MachineInst *inst)
    : Sop1("s_andn1_wrexec_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SAndn1WrexecB64Sop1::execute(amdgpu::Wavefront &wf) {
  uint64_t src = ssrc0.read_scalar64(wf);
  wf.set_exec(src & ~wf.exec());
}

SAndn2WrexecB64Sop1::SAndn2WrexecB64Sop1(const MachineInst *inst)
    : Sop1("s_andn2_wrexec_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SAndn2WrexecB64Sop1::execute(amdgpu::Wavefront &wf) {
  uint64_t src = ssrc0.read_scalar64(wf);
  wf.set_exec(wf.exec() & ~src);
}

SBitreplicateB64B32Sop1::SBitreplicateB64B32Sop1(const MachineInst *inst)
    : Sop1("s_bitreplicate_b64_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SBitreplicateB64B32Sop1::execute(amdgpu::Wavefront &wf) { (void)wf; }

} // namespace cdna4
} // namespace rocjitsu

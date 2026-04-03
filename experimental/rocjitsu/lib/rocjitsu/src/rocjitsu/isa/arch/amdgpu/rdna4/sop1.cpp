// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/rdna4/sop1.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"
#include "util/except.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

namespace rocjitsu {
namespace rdna4 {

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
  src_operands_.emplace_back(&sdst);
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
  src_operands_.emplace_back(&sdst);
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

SCtzI32B32Sop1::SCtzI32B32Sop1(const MachineInst *inst)
    : Sop1("s_ctz_i32_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SCtzI32B32Sop1::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

SCtzI32B64Sop1::SCtzI32B64Sop1(const MachineInst *inst)
    : Sop1("s_ctz_i32_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SCtzI32B64Sop1::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

SClzI32U32Sop1::SClzI32U32Sop1(const MachineInst *inst)
    : Sop1("s_clz_i32_u32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SClzI32U32Sop1::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

SClzI32U64Sop1::SClzI32U64Sop1(const MachineInst *inst)
    : Sop1("s_clz_i32_u64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SClzI32U64Sop1::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

SClsI32Sop1::SClsI32Sop1(const MachineInst *inst)
    : Sop1("s_cls_i32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SClsI32Sop1::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

SClsI32I64Sop1::SClsI32I64Sop1(const MachineInst *inst)
    : Sop1("s_cls_i32_i64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SClsI32I64Sop1::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

SSextI32I8Sop1::SSextI32I8Sop1(const MachineInst *inst)
    : Sop1("s_sext_i32_i8", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(16, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        16, OperandType::OPR_SIMM32,
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

SAndSaveexecB32Sop1::SAndSaveexecB32Sop1(const MachineInst *inst)
    : Sop1("s_and_saveexec_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SAndSaveexecB32Sop1::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

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

SOrSaveexecB32Sop1::SOrSaveexecB32Sop1(const MachineInst *inst)
    : Sop1("s_or_saveexec_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SOrSaveexecB32Sop1::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
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

SXorSaveexecB32Sop1::SXorSaveexecB32Sop1(const MachineInst *inst)
    : Sop1("s_xor_saveexec_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SXorSaveexecB32Sop1::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
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

SNandSaveexecB32Sop1::SNandSaveexecB32Sop1(const MachineInst *inst)
    : Sop1("s_nand_saveexec_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SNandSaveexecB32Sop1::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
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

SNorSaveexecB32Sop1::SNorSaveexecB32Sop1(const MachineInst *inst)
    : Sop1("s_nor_saveexec_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SNorSaveexecB32Sop1::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
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

SXnorSaveexecB32Sop1::SXnorSaveexecB32Sop1(const MachineInst *inst)
    : Sop1("s_xnor_saveexec_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SXnorSaveexecB32Sop1::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
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

SAndNot0SaveexecB32Sop1::SAndNot0SaveexecB32Sop1(const MachineInst *inst)
    : Sop1("s_and_not0_saveexec_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SAndNot0SaveexecB32Sop1::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

SAndNot0SaveexecB64Sop1::SAndNot0SaveexecB64Sop1(const MachineInst *inst)
    : Sop1("s_and_not0_saveexec_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SAndNot0SaveexecB64Sop1::execute(amdgpu::Wavefront &wf) {
  uint64_t old_exec = wf.exec();
  sdst.write_scalar64(wf, old_exec);
  uint64_t src = ssrc0.read_scalar64(wf);
  uint64_t result = old_exec & ~src;
  wf.set_exec(result);
  wf.write_scc(result != 0);
}

SOrNot0SaveexecB32Sop1::SOrNot0SaveexecB32Sop1(const MachineInst *inst)
    : Sop1("s_or_not0_saveexec_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SOrNot0SaveexecB32Sop1::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

SOrNot0SaveexecB64Sop1::SOrNot0SaveexecB64Sop1(const MachineInst *inst)
    : Sop1("s_or_not0_saveexec_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SOrNot0SaveexecB64Sop1::execute(amdgpu::Wavefront &wf) {
  uint64_t old_exec = wf.exec();
  sdst.write_scalar64(wf, old_exec);
  uint64_t src = ssrc0.read_scalar64(wf);
  uint64_t result = old_exec | ~src;
  wf.set_exec(result);
  wf.write_scc(result != 0);
}

SAndNot1SaveexecB32Sop1::SAndNot1SaveexecB32Sop1(const MachineInst *inst)
    : Sop1("s_and_not1_saveexec_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SAndNot1SaveexecB32Sop1::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

SAndNot1SaveexecB64Sop1::SAndNot1SaveexecB64Sop1(const MachineInst *inst)
    : Sop1("s_and_not1_saveexec_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SAndNot1SaveexecB64Sop1::execute(amdgpu::Wavefront &wf) {
  uint64_t old_exec = wf.exec();
  sdst.write_scalar64(wf, old_exec);
  uint64_t src = ssrc0.read_scalar64(wf);
  uint64_t result = ~src & ~old_exec;
  wf.set_exec(result);
  wf.write_scc(result != 0);
}

SOrNot1SaveexecB32Sop1::SOrNot1SaveexecB32Sop1(const MachineInst *inst)
    : Sop1("s_or_not1_saveexec_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SOrNot1SaveexecB32Sop1::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

SOrNot1SaveexecB64Sop1::SOrNot1SaveexecB64Sop1(const MachineInst *inst)
    : Sop1("s_or_not1_saveexec_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SOrNot1SaveexecB64Sop1::execute(amdgpu::Wavefront &wf) {
  uint64_t old_exec = wf.exec();
  sdst.write_scalar64(wf, old_exec);
  uint64_t src = ssrc0.read_scalar64(wf);
  uint64_t result = ~src | old_exec;
  wf.set_exec(result);
  wf.write_scc(result != 0);
}

SAndNot0WrexecB32Sop1::SAndNot0WrexecB32Sop1(const MachineInst *inst)
    : Sop1("s_and_not0_wrexec_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SAndNot0WrexecB32Sop1::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

SAndNot0WrexecB64Sop1::SAndNot0WrexecB64Sop1(const MachineInst *inst)
    : Sop1("s_and_not0_wrexec_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SAndNot0WrexecB64Sop1::execute(amdgpu::Wavefront &wf) {
  uint64_t src = ssrc0.read_scalar64(wf);
  wf.set_exec(src); // TODO: and_not0
}

SAndNot1WrexecB32Sop1::SAndNot1WrexecB32Sop1(const MachineInst *inst)
    : Sop1("s_and_not1_wrexec_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SAndNot1WrexecB32Sop1::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

SAndNot1WrexecB64Sop1::SAndNot1WrexecB64Sop1(const MachineInst *inst)
    : Sop1("s_and_not1_wrexec_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(64, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SAndNot1WrexecB64Sop1::execute(amdgpu::Wavefront &wf) {
  uint64_t src = ssrc0.read_scalar64(wf);
  wf.set_exec(src); // TODO: and_not1
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

SMovrelsd2B32Sop1::SMovrelsd2B32Sop1(const MachineInst *inst)
    : Sop1("s_movrelsd_2_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SMovrelsd2B32Sop1::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
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

SSendmsgRtnB32Sop1::SSendmsgRtnB32Sop1(const MachineInst *inst)
    : Sop1("s_sendmsg_rtn_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SENDMSG_RTN, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SSendmsgRtnB32Sop1::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

SSendmsgRtnB64Sop1::SSendmsgRtnB64Sop1(const MachineInst *inst)
    : Sop1("s_sendmsg_rtn_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(64, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SENDMSG_RTN, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SSendmsgRtnB64Sop1::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

SBarrierSignalSop1::SBarrierSignalSop1(const MachineInst *inst)
    : Sop1("s_barrier_signal", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(32, OperandType::OPR_SSRC_BARRIER_ID,
            reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SBarrierSignalSop1::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

SBarrierSignalIsfirstSop1::SBarrierSignalIsfirstSop1(const MachineInst *inst)
    : Sop1("s_barrier_signal_isfirst", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(32, OperandType::OPR_SSRC_BARRIER_ID,
            reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SBarrierSignalIsfirstSop1::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

SAllocVgprSop1::SAllocVgprSop1(const MachineInst *inst)
    : Sop1("s_alloc_vgpr", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SAllocVgprSop1::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

SSleepVarSop1::SSleepVarSop1(const MachineInst *inst)
    : Sop1("s_sleep_var", reinterpret_cast<const OpEncoding *>(inst)),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SSleepVarSop1::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

SCeilF32Sop1::SCeilF32Sop1(const MachineInst *inst)
    : Sop1("s_ceil_f32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SCeilF32Sop1::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

SFloorF32Sop1::SFloorF32Sop1(const MachineInst *inst)
    : Sop1("s_floor_f32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SFloorF32Sop1::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

STruncF32Sop1::STruncF32Sop1(const MachineInst *inst)
    : Sop1("s_trunc_f32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void STruncF32Sop1::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

SRndneF32Sop1::SRndneF32Sop1(const MachineInst *inst)
    : Sop1("s_rndne_f32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SRndneF32Sop1::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

SCvtF32I32Sop1::SCvtF32I32Sop1(const MachineInst *inst)
    : Sop1("s_cvt_f32_i32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SCvtF32I32Sop1::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

SCvtF32U32Sop1::SCvtF32U32Sop1(const MachineInst *inst)
    : Sop1("s_cvt_f32_u32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SCvtF32U32Sop1::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

SCvtI32F32Sop1::SCvtI32F32Sop1(const MachineInst *inst)
    : Sop1("s_cvt_i32_f32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SCvtI32F32Sop1::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

SCvtU32F32Sop1::SCvtU32F32Sop1(const MachineInst *inst)
    : Sop1("s_cvt_u32_f32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SCvtU32F32Sop1::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

SCvtF16F32Sop1::SCvtF16F32Sop1(const MachineInst *inst)
    : Sop1("s_cvt_f16_f32", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(16, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SCvtF16F32Sop1::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

SCvtF32F16Sop1::SCvtF32F16Sop1(const MachineInst *inst)
    : Sop1("s_cvt_f32_f16", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(16, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SCvtF32F16Sop1::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

SCvtHiF32F16Sop1::SCvtHiF32F16Sop1(const MachineInst *inst)
    : Sop1("s_cvt_hi_f32_f16", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(32, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(16, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SCvtHiF32F16Sop1::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

SCeilF16Sop1::SCeilF16Sop1(const MachineInst *inst)
    : Sop1("s_ceil_f16", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(16, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(16, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SCeilF16Sop1::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

SFloorF16Sop1::SFloorF16Sop1(const MachineInst *inst)
    : Sop1("s_floor_f16", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(16, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(16, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SFloorF16Sop1::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

STruncF16Sop1::STruncF16Sop1(const MachineInst *inst)
    : Sop1("s_trunc_f16", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(16, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(16, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void STruncF16Sop1::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

SRndneF16Sop1::SRndneF16Sop1(const MachineInst *inst)
    : Sop1("s_rndne_f16", reinterpret_cast<const OpEncoding *>(inst)),
      sdst(16, OperandType::OPR_SDST, reinterpret_cast<const OpEncoding *>(inst)->sdst),
      ssrc0(16, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->ssrc0) {
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&ssrc0);
  if (reinterpret_cast<const OpEncoding *>(inst)->ssrc0 == 255)
    ssrc0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Sop1InstLiteralMachineInst *>(inst)->simm32));
}

void SRndneF16Sop1::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

} // namespace rdna4
} // namespace rocjitsu

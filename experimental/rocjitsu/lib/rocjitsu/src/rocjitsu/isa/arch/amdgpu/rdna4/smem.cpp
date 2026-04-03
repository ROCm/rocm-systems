// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/rdna4/smem.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/addr_calc.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"
#include "util/except.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>

namespace rocjitsu {
namespace rdna4 {

namespace {
Operand make_smem_offset(const Smem::OpEncoding *enc) {
  return Operand(32, OperandType::OPR_SIMM32, static_cast<int>(enc->ioffset));
}
} // namespace

SLoadB32Smem::SLoadB32Smem(const MachineInst *inst)
    : Smem("s_load_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SLoadB32Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SLoadB64Smem::SLoadB64Smem(const MachineInst *inst)
    : Smem("s_load_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SLoadB64Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SLoadB128Smem::SLoadB128Smem(const MachineInst *inst)
    : Smem("s_load_b128", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SLoadB128Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SLoadB256Smem::SLoadB256Smem(const MachineInst *inst)
    : Smem("s_load_b256", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SLoadB256Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SLoadB512Smem::SLoadB512Smem(const MachineInst *inst)
    : Smem("s_load_b512", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(512, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SLoadB512Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SLoadB96Smem::SLoadB96Smem(const MachineInst *inst)
    : Smem("s_load_b96", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(96, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SLoadB96Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SLoadI8Smem::SLoadI8Smem(const MachineInst *inst)
    : Smem("s_load_i8", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SLoadI8Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SLoadU8Smem::SLoadU8Smem(const MachineInst *inst)
    : Smem("s_load_u8", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SLoadU8Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SLoadI16Smem::SLoadI16Smem(const MachineInst *inst)
    : Smem("s_load_i16", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SLoadI16Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SLoadU16Smem::SLoadU16Smem(const MachineInst *inst)
    : Smem("s_load_u16", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SLoadU16Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SBufferLoadB32Smem::SBufferLoadB32Smem(const MachineInst *inst)
    : Smem("s_buffer_load_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SBufferLoadB32Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SBufferLoadB64Smem::SBufferLoadB64Smem(const MachineInst *inst)
    : Smem("s_buffer_load_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SBufferLoadB64Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SBufferLoadB128Smem::SBufferLoadB128Smem(const MachineInst *inst)
    : Smem("s_buffer_load_b128", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SBufferLoadB128Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SBufferLoadB256Smem::SBufferLoadB256Smem(const MachineInst *inst)
    : Smem("s_buffer_load_b256", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SBufferLoadB256Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SBufferLoadB512Smem::SBufferLoadB512Smem(const MachineInst *inst)
    : Smem("s_buffer_load_b512", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(512, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SBufferLoadB512Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SBufferLoadB96Smem::SBufferLoadB96Smem(const MachineInst *inst)
    : Smem("s_buffer_load_b96", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(96, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SBufferLoadB96Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SBufferLoadI8Smem::SBufferLoadI8Smem(const MachineInst *inst)
    : Smem("s_buffer_load_i8", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SBufferLoadI8Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SBufferLoadU8Smem::SBufferLoadU8Smem(const MachineInst *inst)
    : Smem("s_buffer_load_u8", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SBufferLoadU8Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SBufferLoadI16Smem::SBufferLoadI16Smem(const MachineInst *inst)
    : Smem("s_buffer_load_i16", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SBufferLoadI16Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SBufferLoadU16Smem::SBufferLoadU16Smem(const MachineInst *inst)
    : Smem("s_buffer_load_u16", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SBufferLoadU16Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SDcacheInvSmem::SDcacheInvSmem(const MachineInst *inst)
    : Smem("s_dcache_inv", reinterpret_cast<const OpEncoding *>(inst)) {}

void SDcacheInvSmem::execute(amdgpu::Wavefront &wf) { wf.cu().l1_scalar().invalidate_all(); }

SAtcProbeSmem::SAtcProbeSmem(const MachineInst *inst)
    : Smem("s_atc_probe", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(8, OperandType::OPR_SIMM8, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  src_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SAtcProbeSmem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SAtcProbeBufferSmem::SAtcProbeBufferSmem(const MachineInst *inst)
    : Smem("s_atc_probe_buffer", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(8, OperandType::OPR_SIMM8, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  src_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SAtcProbeBufferSmem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SPrefetchInstSmem::SPrefetchInstSmem(const MachineInst *inst)
    : Smem("s_prefetch_inst", reinterpret_cast<const OpEncoding *>(inst)),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      ioffset(32, OperandType::OPR_SIMM24, reinterpret_cast<const OpEncoding *>(inst)->ioffset),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))),
      sdata(32, OperandType::OPR_SIMM5, reinterpret_cast<const OpEncoding *>(inst)->sdata) {
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&ioffset);
  src_operands_.emplace_back(&soffset);
  src_operands_.emplace_back(&sdata);
}

void SPrefetchInstSmem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SPrefetchInstPcRelSmem::SPrefetchInstPcRelSmem(const MachineInst *inst)
    : Smem("s_prefetch_inst_pc_rel", reinterpret_cast<const OpEncoding *>(inst)),
      ioffset(32, OperandType::OPR_SIMM24, reinterpret_cast<const OpEncoding *>(inst)->ioffset),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))),
      sdata(32, OperandType::OPR_SIMM5, reinterpret_cast<const OpEncoding *>(inst)->sdata) {
  src_operands_.emplace_back(&ioffset);
  src_operands_.emplace_back(&soffset);
  src_operands_.emplace_back(&sdata);
}

void SPrefetchInstPcRelSmem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SPrefetchDataSmem::SPrefetchDataSmem(const MachineInst *inst)
    : Smem("s_prefetch_data", reinterpret_cast<const OpEncoding *>(inst)),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      ioffset(32, OperandType::OPR_SIMM24, reinterpret_cast<const OpEncoding *>(inst)->ioffset),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))),
      sdata(32, OperandType::OPR_SIMM5, reinterpret_cast<const OpEncoding *>(inst)->sdata) {
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&ioffset);
  src_operands_.emplace_back(&soffset);
  src_operands_.emplace_back(&sdata);
}

void SPrefetchDataSmem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SBufferPrefetchDataSmem::SBufferPrefetchDataSmem(const MachineInst *inst)
    : Smem("s_buffer_prefetch_data", reinterpret_cast<const OpEncoding *>(inst)),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      ioffset(32, OperandType::OPR_SIMM24, reinterpret_cast<const OpEncoding *>(inst)->ioffset),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))),
      sdata(32, OperandType::OPR_SIMM5, reinterpret_cast<const OpEncoding *>(inst)->sdata) {
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&ioffset);
  src_operands_.emplace_back(&soffset);
  src_operands_.emplace_back(&sdata);
}

void SBufferPrefetchDataSmem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SPrefetchDataPcRelSmem::SPrefetchDataPcRelSmem(const MachineInst *inst)
    : Smem("s_prefetch_data_pc_rel", reinterpret_cast<const OpEncoding *>(inst)),
      ioffset(32, OperandType::OPR_SIMM24, reinterpret_cast<const OpEncoding *>(inst)->ioffset),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))),
      sdata(32, OperandType::OPR_SIMM5, reinterpret_cast<const OpEncoding *>(inst)->sdata) {
  src_operands_.emplace_back(&ioffset);
  src_operands_.emplace_back(&soffset);
  src_operands_.emplace_back(&sdata);
}

void SPrefetchDataPcRelSmem::execute(amdgpu::Wavefront &wf) { (void)wf; }

} // namespace rdna4
} // namespace rocjitsu

// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/rdna3_5/smem.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3_5/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/shared/gfx11_cache_flags.h"
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
namespace rdna3_5 {

namespace {
Operand make_smem_offset(const Smem::OpEncoding *enc) {
  return Operand(32, OperandType::OPR_SIMM32, static_cast<int>(enc->offset));
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
  flags_ |= MEMORY_OP;
}

void SLoadB32Smem::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::ScalarMemState>();
  d->dst_reg_base = wf.sgpr_alloc().base + inst_.sdata;
  d->num_dwords = 1;
  d->is_load = true;
  d->mtype = amdgpu::mtype_from_flags_gfx11(inst_.glc, inst_.dlc, false);
  d->addr = smem_calculate_address(inst_, wf);
  set_data(std::move(d));
}

SLoadB64Smem::SLoadB64Smem(const MachineInst *inst)
    : Smem("s_load_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void SLoadB64Smem::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::ScalarMemState>();
  d->dst_reg_base = wf.sgpr_alloc().base + inst_.sdata;
  d->num_dwords = 2;
  d->is_load = true;
  d->mtype = amdgpu::mtype_from_flags_gfx11(inst_.glc, inst_.dlc, false);
  d->addr = smem_calculate_address(inst_, wf);
  set_data(std::move(d));
}

SLoadB128Smem::SLoadB128Smem(const MachineInst *inst)
    : Smem("s_load_b128", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void SLoadB128Smem::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::ScalarMemState>();
  d->dst_reg_base = wf.sgpr_alloc().base + inst_.sdata;
  d->num_dwords = 4;
  d->is_load = true;
  d->mtype = amdgpu::mtype_from_flags_gfx11(inst_.glc, inst_.dlc, false);
  d->addr = smem_calculate_address(inst_, wf);
  set_data(std::move(d));
}

SLoadB256Smem::SLoadB256Smem(const MachineInst *inst)
    : Smem("s_load_b256", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void SLoadB256Smem::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::ScalarMemState>();
  d->dst_reg_base = wf.sgpr_alloc().base + inst_.sdata;
  d->num_dwords = 8;
  d->is_load = true;
  d->mtype = amdgpu::mtype_from_flags_gfx11(inst_.glc, inst_.dlc, false);
  d->addr = smem_calculate_address(inst_, wf);
  set_data(std::move(d));
}

SLoadB512Smem::SLoadB512Smem(const MachineInst *inst)
    : Smem("s_load_b512", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(512, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void SLoadB512Smem::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::ScalarMemState>();
  d->dst_reg_base = wf.sgpr_alloc().base + inst_.sdata;
  d->num_dwords = 16;
  d->is_load = true;
  d->mtype = amdgpu::mtype_from_flags_gfx11(inst_.glc, inst_.dlc, false);
  d->addr = smem_calculate_address(inst_, wf);
  set_data(std::move(d));
}

SBufferLoadB32Smem::SBufferLoadB32Smem(const MachineInst *inst)
    : Smem("s_buffer_load_b32", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void SBufferLoadB32Smem::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::ScalarMemState>();
  d->dst_reg_base = wf.sgpr_alloc().base + inst_.sdata;
  d->num_dwords = 1;
  d->is_load = true;
  d->mtype = amdgpu::mtype_from_flags_gfx11(inst_.glc, inst_.dlc, false);
  d->addr = smem_calculate_address(inst_, wf);
  set_data(std::move(d));
}

SBufferLoadB64Smem::SBufferLoadB64Smem(const MachineInst *inst)
    : Smem("s_buffer_load_b64", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void SBufferLoadB64Smem::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::ScalarMemState>();
  d->dst_reg_base = wf.sgpr_alloc().base + inst_.sdata;
  d->num_dwords = 2;
  d->is_load = true;
  d->mtype = amdgpu::mtype_from_flags_gfx11(inst_.glc, inst_.dlc, false);
  d->addr = smem_calculate_address(inst_, wf);
  set_data(std::move(d));
}

SBufferLoadB128Smem::SBufferLoadB128Smem(const MachineInst *inst)
    : Smem("s_buffer_load_b128", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void SBufferLoadB128Smem::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::ScalarMemState>();
  d->dst_reg_base = wf.sgpr_alloc().base + inst_.sdata;
  d->num_dwords = 4;
  d->is_load = true;
  d->mtype = amdgpu::mtype_from_flags_gfx11(inst_.glc, inst_.dlc, false);
  d->addr = smem_calculate_address(inst_, wf);
  set_data(std::move(d));
}

SBufferLoadB256Smem::SBufferLoadB256Smem(const MachineInst *inst)
    : Smem("s_buffer_load_b256", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void SBufferLoadB256Smem::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::ScalarMemState>();
  d->dst_reg_base = wf.sgpr_alloc().base + inst_.sdata;
  d->num_dwords = 8;
  d->is_load = true;
  d->mtype = amdgpu::mtype_from_flags_gfx11(inst_.glc, inst_.dlc, false);
  d->addr = smem_calculate_address(inst_, wf);
  set_data(std::move(d));
}

SBufferLoadB512Smem::SBufferLoadB512Smem(const MachineInst *inst)
    : Smem("s_buffer_load_b512", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(512, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void SBufferLoadB512Smem::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::ScalarMemState>();
  d->dst_reg_base = wf.sgpr_alloc().base + inst_.sdata;
  d->num_dwords = 16;
  d->is_load = true;
  d->mtype = amdgpu::mtype_from_flags_gfx11(inst_.glc, inst_.dlc, false);
  d->addr = smem_calculate_address(inst_, wf);
  set_data(std::move(d));
}

SGl1InvSmem::SGl1InvSmem(const MachineInst *inst)
    : Smem("s_gl1_inv", reinterpret_cast<const OpEncoding *>(inst)) {}

void SGl1InvSmem::execute(amdgpu::Wavefront &wf) { wf.cu().l1_vector().invalidate_all(); }

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

} // namespace rdna3_5
} // namespace rocjitsu

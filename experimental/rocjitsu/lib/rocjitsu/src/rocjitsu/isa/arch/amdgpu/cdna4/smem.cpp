// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/cdna4/smem.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/addr_calc.h"
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
namespace cdna4 {

namespace {
Operand make_smem_offset(const Smem::OpEncoding *enc) {
  // SOFFSET_EN and IMM are independent: SOFFSET_EN gates the
  // SGPR field, IMM gates the 21-bit immediate field.
  // When both are set the hardware adds SGPR + immediate;
  // we show the SGPR as the operand and the immediate as
  // an offset modifier.
  if (enc->soffset_en)
    return Operand(32, OperandType::OPR_SMEM_OFFSET, static_cast<int>(enc->soffset));
  if (enc->imm)
    return Operand(32, OperandType::OPR_SIMM32, static_cast<int>(enc->offset));
  return Operand(32, OperandType::OPR_SIMM32, 0);
}
} // namespace

SLoadDwordSmem::SLoadDwordSmem(const MachineInst *inst)
    : Smem("s_load_dword", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void SLoadDwordSmem::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::ScalarMemState>();
  d->dst_reg_base = wf.sgpr_alloc().base + inst_.sdata;
  d->num_dwords = 1;
  d->is_load = true;
  d->addr = smem_calculate_address(inst_, wf);
  set_data(std::move(d));
}

SLoadDwordx2Smem::SLoadDwordx2Smem(const MachineInst *inst)
    : Smem("s_load_dwordx2", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void SLoadDwordx2Smem::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::ScalarMemState>();
  d->dst_reg_base = wf.sgpr_alloc().base + inst_.sdata;
  d->num_dwords = 2;
  d->is_load = true;
  d->addr = smem_calculate_address(inst_, wf);
  set_data(std::move(d));
}

SLoadDwordx4Smem::SLoadDwordx4Smem(const MachineInst *inst)
    : Smem("s_load_dwordx4", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void SLoadDwordx4Smem::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::ScalarMemState>();
  d->dst_reg_base = wf.sgpr_alloc().base + inst_.sdata;
  d->num_dwords = 4;
  d->is_load = true;
  d->addr = smem_calculate_address(inst_, wf);
  set_data(std::move(d));
}

SLoadDwordx8Smem::SLoadDwordx8Smem(const MachineInst *inst)
    : Smem("s_load_dwordx8", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void SLoadDwordx8Smem::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::ScalarMemState>();
  d->dst_reg_base = wf.sgpr_alloc().base + inst_.sdata;
  d->num_dwords = 8;
  d->is_load = true;
  d->addr = smem_calculate_address(inst_, wf);
  set_data(std::move(d));
}

SLoadDwordx16Smem::SLoadDwordx16Smem(const MachineInst *inst)
    : Smem("s_load_dwordx16", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(512, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void SLoadDwordx16Smem::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::ScalarMemState>();
  d->dst_reg_base = wf.sgpr_alloc().base + inst_.sdata;
  d->num_dwords = 16;
  d->is_load = true;
  d->addr = smem_calculate_address(inst_, wf);
  set_data(std::move(d));
}

SScratchLoadDwordSmem::SScratchLoadDwordSmem(const MachineInst *inst)
    : Smem("s_scratch_load_dword", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void SScratchLoadDwordSmem::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::ScalarMemState>();
  d->dst_reg_base = wf.sgpr_alloc().base + inst_.sdata;
  d->num_dwords = 1;
  d->is_load = true;
  d->addr = smem_calculate_address(inst_, wf);
  set_data(std::move(d));
}

SScratchLoadDwordx2Smem::SScratchLoadDwordx2Smem(const MachineInst *inst)
    : Smem("s_scratch_load_dwordx2", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void SScratchLoadDwordx2Smem::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::ScalarMemState>();
  d->dst_reg_base = wf.sgpr_alloc().base + inst_.sdata;
  d->num_dwords = 2;
  d->is_load = true;
  d->addr = smem_calculate_address(inst_, wf);
  set_data(std::move(d));
}

SScratchLoadDwordx4Smem::SScratchLoadDwordx4Smem(const MachineInst *inst)
    : Smem("s_scratch_load_dwordx4", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void SScratchLoadDwordx4Smem::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::ScalarMemState>();
  d->dst_reg_base = wf.sgpr_alloc().base + inst_.sdata;
  d->num_dwords = 4;
  d->is_load = true;
  d->addr = smem_calculate_address(inst_, wf);
  set_data(std::move(d));
}

SBufferLoadDwordSmem::SBufferLoadDwordSmem(const MachineInst *inst)
    : Smem("s_buffer_load_dword", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void SBufferLoadDwordSmem::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::ScalarMemState>();
  d->dst_reg_base = wf.sgpr_alloc().base + inst_.sdata;
  d->num_dwords = 1;
  d->is_load = true;
  d->addr = smem_calculate_address(inst_, wf);
  set_data(std::move(d));
}

SBufferLoadDwordx2Smem::SBufferLoadDwordx2Smem(const MachineInst *inst)
    : Smem("s_buffer_load_dwordx2", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void SBufferLoadDwordx2Smem::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::ScalarMemState>();
  d->dst_reg_base = wf.sgpr_alloc().base + inst_.sdata;
  d->num_dwords = 2;
  d->is_load = true;
  d->addr = smem_calculate_address(inst_, wf);
  set_data(std::move(d));
}

SBufferLoadDwordx4Smem::SBufferLoadDwordx4Smem(const MachineInst *inst)
    : Smem("s_buffer_load_dwordx4", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void SBufferLoadDwordx4Smem::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::ScalarMemState>();
  d->dst_reg_base = wf.sgpr_alloc().base + inst_.sdata;
  d->num_dwords = 4;
  d->is_load = true;
  d->addr = smem_calculate_address(inst_, wf);
  set_data(std::move(d));
}

SBufferLoadDwordx8Smem::SBufferLoadDwordx8Smem(const MachineInst *inst)
    : Smem("s_buffer_load_dwordx8", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(256, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void SBufferLoadDwordx8Smem::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::ScalarMemState>();
  d->dst_reg_base = wf.sgpr_alloc().base + inst_.sdata;
  d->num_dwords = 8;
  d->is_load = true;
  d->addr = smem_calculate_address(inst_, wf);
  set_data(std::move(d));
}

SBufferLoadDwordx16Smem::SBufferLoadDwordx16Smem(const MachineInst *inst)
    : Smem("s_buffer_load_dwordx16", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(512, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void SBufferLoadDwordx16Smem::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::ScalarMemState>();
  d->dst_reg_base = wf.sgpr_alloc().base + inst_.sdata;
  d->num_dwords = 16;
  d->is_load = true;
  d->addr = smem_calculate_address(inst_, wf);
  set_data(std::move(d));
}

SStoreDwordSmem::SStoreDwordSmem(const MachineInst *inst)
    : Smem("s_store_dword", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  src_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void SStoreDwordSmem::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::ScalarMemState>();
  d->num_dwords = 1;
  d->is_load = false;
  auto &cu = wf.cu();
  uint32_t sdata_base = wf.sgpr_alloc().base + inst_.sdata;
  for (uint32_t i = 0; i < 1; ++i)
    d->store_data[i] = cu.read_sgpr(sdata_base + i);
  d->addr = smem_calculate_address(inst_, wf);
  set_data(std::move(d));
}

SStoreDwordx2Smem::SStoreDwordx2Smem(const MachineInst *inst)
    : Smem("s_store_dwordx2", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  src_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void SStoreDwordx2Smem::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::ScalarMemState>();
  d->num_dwords = 2;
  d->is_load = false;
  auto &cu = wf.cu();
  uint32_t sdata_base = wf.sgpr_alloc().base + inst_.sdata;
  for (uint32_t i = 0; i < 2; ++i)
    d->store_data[i] = cu.read_sgpr(sdata_base + i);
  d->addr = smem_calculate_address(inst_, wf);
  set_data(std::move(d));
}

SStoreDwordx4Smem::SStoreDwordx4Smem(const MachineInst *inst)
    : Smem("s_store_dwordx4", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  src_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void SStoreDwordx4Smem::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::ScalarMemState>();
  d->num_dwords = 4;
  d->is_load = false;
  auto &cu = wf.cu();
  uint32_t sdata_base = wf.sgpr_alloc().base + inst_.sdata;
  for (uint32_t i = 0; i < 4; ++i)
    d->store_data[i] = cu.read_sgpr(sdata_base + i);
  d->addr = smem_calculate_address(inst_, wf);
  set_data(std::move(d));
}

SScratchStoreDwordSmem::SScratchStoreDwordSmem(const MachineInst *inst)
    : Smem("s_scratch_store_dword", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  src_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void SScratchStoreDwordSmem::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::ScalarMemState>();
  d->num_dwords = 1;
  d->is_load = false;
  auto &cu = wf.cu();
  uint32_t sdata_base = wf.sgpr_alloc().base + inst_.sdata;
  for (uint32_t i = 0; i < 1; ++i)
    d->store_data[i] = cu.read_sgpr(sdata_base + i);
  d->addr = smem_calculate_address(inst_, wf);
  set_data(std::move(d));
}

SScratchStoreDwordx2Smem::SScratchStoreDwordx2Smem(const MachineInst *inst)
    : Smem("s_scratch_store_dwordx2", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  src_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void SScratchStoreDwordx2Smem::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::ScalarMemState>();
  d->num_dwords = 2;
  d->is_load = false;
  auto &cu = wf.cu();
  uint32_t sdata_base = wf.sgpr_alloc().base + inst_.sdata;
  for (uint32_t i = 0; i < 2; ++i)
    d->store_data[i] = cu.read_sgpr(sdata_base + i);
  d->addr = smem_calculate_address(inst_, wf);
  set_data(std::move(d));
}

SScratchStoreDwordx4Smem::SScratchStoreDwordx4Smem(const MachineInst *inst)
    : Smem("s_scratch_store_dwordx4", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  src_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void SScratchStoreDwordx4Smem::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::ScalarMemState>();
  d->num_dwords = 4;
  d->is_load = false;
  auto &cu = wf.cu();
  uint32_t sdata_base = wf.sgpr_alloc().base + inst_.sdata;
  for (uint32_t i = 0; i < 4; ++i)
    d->store_data[i] = cu.read_sgpr(sdata_base + i);
  d->addr = smem_calculate_address(inst_, wf);
  set_data(std::move(d));
}

SBufferStoreDwordSmem::SBufferStoreDwordSmem(const MachineInst *inst)
    : Smem("s_buffer_store_dword", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  src_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void SBufferStoreDwordSmem::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::ScalarMemState>();
  d->num_dwords = 1;
  d->is_load = false;
  auto &cu = wf.cu();
  uint32_t sdata_base = wf.sgpr_alloc().base + inst_.sdata;
  for (uint32_t i = 0; i < 1; ++i)
    d->store_data[i] = cu.read_sgpr(sdata_base + i);
  d->addr = smem_calculate_address(inst_, wf);
  set_data(std::move(d));
}

SBufferStoreDwordx2Smem::SBufferStoreDwordx2Smem(const MachineInst *inst)
    : Smem("s_buffer_store_dwordx2", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  src_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void SBufferStoreDwordx2Smem::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::ScalarMemState>();
  d->num_dwords = 2;
  d->is_load = false;
  auto &cu = wf.cu();
  uint32_t sdata_base = wf.sgpr_alloc().base + inst_.sdata;
  for (uint32_t i = 0; i < 2; ++i)
    d->store_data[i] = cu.read_sgpr(sdata_base + i);
  d->addr = smem_calculate_address(inst_, wf);
  set_data(std::move(d));
}

SBufferStoreDwordx4Smem::SBufferStoreDwordx4Smem(const MachineInst *inst)
    : Smem("s_buffer_store_dwordx4", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  src_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void SBufferStoreDwordx4Smem::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::ScalarMemState>();
  d->num_dwords = 4;
  d->is_load = false;
  auto &cu = wf.cu();
  uint32_t sdata_base = wf.sgpr_alloc().base + inst_.sdata;
  for (uint32_t i = 0; i < 4; ++i)
    d->store_data[i] = cu.read_sgpr(sdata_base + i);
  d->addr = smem_calculate_address(inst_, wf);
  set_data(std::move(d));
}

SDcacheInvSmem::SDcacheInvSmem(const MachineInst *inst)
    : Smem("s_dcache_inv", reinterpret_cast<const OpEncoding *>(inst)) {}

void SDcacheInvSmem::execute(amdgpu::Wavefront &wf) { wf.cu().l1_scalar().invalidate_all(); }

SDcacheWbSmem::SDcacheWbSmem(const MachineInst *inst)
    : Smem("s_dcache_wb", reinterpret_cast<const OpEncoding *>(inst)) {}

void SDcacheWbSmem::execute(amdgpu::Wavefront &wf) { wf.cu().l1_scalar().writeback_all(); }

SDcacheInvVolSmem::SDcacheInvVolSmem(const MachineInst *inst)
    : Smem("s_dcache_inv_vol", reinterpret_cast<const OpEncoding *>(inst)) {}

void SDcacheInvVolSmem::execute(amdgpu::Wavefront &wf) { wf.cu().l1_scalar().invalidate_all(); }

SDcacheWbVolSmem::SDcacheWbVolSmem(const MachineInst *inst)
    : Smem("s_dcache_wb_vol", reinterpret_cast<const OpEncoding *>(inst)) {}

void SDcacheWbVolSmem::execute(amdgpu::Wavefront &wf) { wf.cu().l1_scalar().writeback_all(); }

SMemtimeSmem::SMemtimeSmem(const MachineInst *inst)
    : Smem("s_memtime", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata) {
  dst_operands_.emplace_back(&sdata);
}

void SMemtimeSmem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SMemrealtimeSmem::SMemrealtimeSmem(const MachineInst *inst)
    : Smem("s_memrealtime", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata) {
  dst_operands_.emplace_back(&sdata);
}

void SMemrealtimeSmem::execute(amdgpu::Wavefront &wf) { (void)wf; }

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

SDcacheDiscardSmem::SDcacheDiscardSmem(const MachineInst *inst)
    : Smem("s_dcache_discard", reinterpret_cast<const OpEncoding *>(inst)),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SDcacheDiscardSmem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SDcacheDiscardX2Smem::SDcacheDiscardX2Smem(const MachineInst *inst)
    : Smem("s_dcache_discard_x2", reinterpret_cast<const OpEncoding *>(inst)),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SDcacheDiscardX2Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SBufferAtomicSwapSmem::SBufferAtomicSwapSmem(const MachineInst *inst)
    : Smem("s_buffer_atomic_swap", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SBufferAtomicSwapSmem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SBufferAtomicCmpswapSmem::SBufferAtomicCmpswapSmem(const MachineInst *inst)
    : Smem("s_buffer_atomic_cmpswap", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SBufferAtomicCmpswapSmem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SBufferAtomicAddSmem::SBufferAtomicAddSmem(const MachineInst *inst)
    : Smem("s_buffer_atomic_add", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SBufferAtomicAddSmem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SBufferAtomicSubSmem::SBufferAtomicSubSmem(const MachineInst *inst)
    : Smem("s_buffer_atomic_sub", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SBufferAtomicSubSmem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SBufferAtomicSminSmem::SBufferAtomicSminSmem(const MachineInst *inst)
    : Smem("s_buffer_atomic_smin", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SBufferAtomicSminSmem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SBufferAtomicUminSmem::SBufferAtomicUminSmem(const MachineInst *inst)
    : Smem("s_buffer_atomic_umin", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SBufferAtomicUminSmem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SBufferAtomicSmaxSmem::SBufferAtomicSmaxSmem(const MachineInst *inst)
    : Smem("s_buffer_atomic_smax", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SBufferAtomicSmaxSmem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SBufferAtomicUmaxSmem::SBufferAtomicUmaxSmem(const MachineInst *inst)
    : Smem("s_buffer_atomic_umax", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SBufferAtomicUmaxSmem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SBufferAtomicAndSmem::SBufferAtomicAndSmem(const MachineInst *inst)
    : Smem("s_buffer_atomic_and", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SBufferAtomicAndSmem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SBufferAtomicOrSmem::SBufferAtomicOrSmem(const MachineInst *inst)
    : Smem("s_buffer_atomic_or", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SBufferAtomicOrSmem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SBufferAtomicXorSmem::SBufferAtomicXorSmem(const MachineInst *inst)
    : Smem("s_buffer_atomic_xor", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SBufferAtomicXorSmem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SBufferAtomicIncSmem::SBufferAtomicIncSmem(const MachineInst *inst)
    : Smem("s_buffer_atomic_inc", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SBufferAtomicIncSmem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SBufferAtomicDecSmem::SBufferAtomicDecSmem(const MachineInst *inst)
    : Smem("s_buffer_atomic_dec", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SBufferAtomicDecSmem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SBufferAtomicSwapX2Smem::SBufferAtomicSwapX2Smem(const MachineInst *inst)
    : Smem("s_buffer_atomic_swap_x2", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SBufferAtomicSwapX2Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SBufferAtomicCmpswapX2Smem::SBufferAtomicCmpswapX2Smem(const MachineInst *inst)
    : Smem("s_buffer_atomic_cmpswap_x2", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SBufferAtomicCmpswapX2Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SBufferAtomicAddX2Smem::SBufferAtomicAddX2Smem(const MachineInst *inst)
    : Smem("s_buffer_atomic_add_x2", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SBufferAtomicAddX2Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SBufferAtomicSubX2Smem::SBufferAtomicSubX2Smem(const MachineInst *inst)
    : Smem("s_buffer_atomic_sub_x2", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SBufferAtomicSubX2Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SBufferAtomicSminX2Smem::SBufferAtomicSminX2Smem(const MachineInst *inst)
    : Smem("s_buffer_atomic_smin_x2", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SBufferAtomicSminX2Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SBufferAtomicUminX2Smem::SBufferAtomicUminX2Smem(const MachineInst *inst)
    : Smem("s_buffer_atomic_umin_x2", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SBufferAtomicUminX2Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SBufferAtomicSmaxX2Smem::SBufferAtomicSmaxX2Smem(const MachineInst *inst)
    : Smem("s_buffer_atomic_smax_x2", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SBufferAtomicSmaxX2Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SBufferAtomicUmaxX2Smem::SBufferAtomicUmaxX2Smem(const MachineInst *inst)
    : Smem("s_buffer_atomic_umax_x2", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SBufferAtomicUmaxX2Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SBufferAtomicAndX2Smem::SBufferAtomicAndX2Smem(const MachineInst *inst)
    : Smem("s_buffer_atomic_and_x2", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SBufferAtomicAndX2Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SBufferAtomicOrX2Smem::SBufferAtomicOrX2Smem(const MachineInst *inst)
    : Smem("s_buffer_atomic_or_x2", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SBufferAtomicOrX2Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SBufferAtomicXorX2Smem::SBufferAtomicXorX2Smem(const MachineInst *inst)
    : Smem("s_buffer_atomic_xor_x2", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SBufferAtomicXorX2Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SBufferAtomicIncX2Smem::SBufferAtomicIncX2Smem(const MachineInst *inst)
    : Smem("s_buffer_atomic_inc_x2", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SBufferAtomicIncX2Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SBufferAtomicDecX2Smem::SBufferAtomicDecX2Smem(const MachineInst *inst)
    : Smem("s_buffer_atomic_dec_x2", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SBufferAtomicDecX2Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SAtomicSwapSmem::SAtomicSwapSmem(const MachineInst *inst)
    : Smem("s_atomic_swap", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SAtomicSwapSmem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SAtomicCmpswapSmem::SAtomicCmpswapSmem(const MachineInst *inst)
    : Smem("s_atomic_cmpswap", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SAtomicCmpswapSmem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SAtomicAddSmem::SAtomicAddSmem(const MachineInst *inst)
    : Smem("s_atomic_add", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SAtomicAddSmem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SAtomicSubSmem::SAtomicSubSmem(const MachineInst *inst)
    : Smem("s_atomic_sub", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SAtomicSubSmem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SAtomicSminSmem::SAtomicSminSmem(const MachineInst *inst)
    : Smem("s_atomic_smin", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SAtomicSminSmem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SAtomicUminSmem::SAtomicUminSmem(const MachineInst *inst)
    : Smem("s_atomic_umin", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SAtomicUminSmem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SAtomicSmaxSmem::SAtomicSmaxSmem(const MachineInst *inst)
    : Smem("s_atomic_smax", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SAtomicSmaxSmem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SAtomicUmaxSmem::SAtomicUmaxSmem(const MachineInst *inst)
    : Smem("s_atomic_umax", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SAtomicUmaxSmem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SAtomicAndSmem::SAtomicAndSmem(const MachineInst *inst)
    : Smem("s_atomic_and", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SAtomicAndSmem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SAtomicOrSmem::SAtomicOrSmem(const MachineInst *inst)
    : Smem("s_atomic_or", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SAtomicOrSmem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SAtomicXorSmem::SAtomicXorSmem(const MachineInst *inst)
    : Smem("s_atomic_xor", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SAtomicXorSmem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SAtomicIncSmem::SAtomicIncSmem(const MachineInst *inst)
    : Smem("s_atomic_inc", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SAtomicIncSmem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SAtomicDecSmem::SAtomicDecSmem(const MachineInst *inst)
    : Smem("s_atomic_dec", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SAtomicDecSmem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SAtomicSwapX2Smem::SAtomicSwapX2Smem(const MachineInst *inst)
    : Smem("s_atomic_swap_x2", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SAtomicSwapX2Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SAtomicCmpswapX2Smem::SAtomicCmpswapX2Smem(const MachineInst *inst)
    : Smem("s_atomic_cmpswap_x2", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SAtomicCmpswapX2Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SAtomicAddX2Smem::SAtomicAddX2Smem(const MachineInst *inst)
    : Smem("s_atomic_add_x2", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SAtomicAddX2Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SAtomicSubX2Smem::SAtomicSubX2Smem(const MachineInst *inst)
    : Smem("s_atomic_sub_x2", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SAtomicSubX2Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SAtomicSminX2Smem::SAtomicSminX2Smem(const MachineInst *inst)
    : Smem("s_atomic_smin_x2", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SAtomicSminX2Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SAtomicUminX2Smem::SAtomicUminX2Smem(const MachineInst *inst)
    : Smem("s_atomic_umin_x2", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SAtomicUminX2Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SAtomicSmaxX2Smem::SAtomicSmaxX2Smem(const MachineInst *inst)
    : Smem("s_atomic_smax_x2", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SAtomicSmaxX2Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SAtomicUmaxX2Smem::SAtomicUmaxX2Smem(const MachineInst *inst)
    : Smem("s_atomic_umax_x2", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SAtomicUmaxX2Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SAtomicAndX2Smem::SAtomicAndX2Smem(const MachineInst *inst)
    : Smem("s_atomic_and_x2", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SAtomicAndX2Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SAtomicOrX2Smem::SAtomicOrX2Smem(const MachineInst *inst)
    : Smem("s_atomic_or_x2", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SAtomicOrX2Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SAtomicXorX2Smem::SAtomicXorX2Smem(const MachineInst *inst)
    : Smem("s_atomic_xor_x2", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SAtomicXorX2Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SAtomicIncX2Smem::SAtomicIncX2Smem(const MachineInst *inst)
    : Smem("s_atomic_inc_x2", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SAtomicIncX2Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

SAtomicDecX2Smem::SAtomicDecX2Smem(const MachineInst *inst)
    : Smem("s_atomic_dec_x2", reinterpret_cast<const OpEncoding *>(inst)),
      sdata(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sdata),
      sbase(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->sbase),
      soffset(make_smem_offset(reinterpret_cast<const OpEncoding *>(inst))) {
  dst_operands_.emplace_back(&sdata);
  src_operands_.emplace_back(&sbase);
  src_operands_.emplace_back(&soffset);
}

void SAtomicDecX2Smem::execute(amdgpu::Wavefront &wf) { (void)wf; }

} // namespace cdna4
} // namespace rocjitsu

// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/rdna4/vglobal.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/shared/gfx12_cache_flags.h"
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

GlobalLoadU8Vglobal::GlobalLoadU8Vglobal(const MachineInst *inst)
    : Vglobal("global_load_u8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalLoadU8Vglobal::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
  d->non_temporal = 0;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

GlobalLoadI8Vglobal::GlobalLoadI8Vglobal(const MachineInst *inst)
    : Vglobal("global_load_i8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalLoadI8Vglobal::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->sign_extend = true;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
  d->non_temporal = 0;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

GlobalLoadU16Vglobal::GlobalLoadU16Vglobal(const MachineInst *inst)
    : Vglobal("global_load_u16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalLoadU16Vglobal::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
  d->non_temporal = 0;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

GlobalLoadI16Vglobal::GlobalLoadI16Vglobal(const MachineInst *inst)
    : Vglobal("global_load_i16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalLoadI16Vglobal::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = true;
  d->sign_extend = true;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
  d->non_temporal = 0;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

GlobalLoadB32Vglobal::GlobalLoadB32Vglobal(const MachineInst *inst)
    : Vglobal("global_load_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalLoadB32Vglobal::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
  d->non_temporal = 0;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

GlobalLoadB64Vglobal::GlobalLoadB64Vglobal(const MachineInst *inst)
    : Vglobal("global_load_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalLoadB64Vglobal::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 2;
  d->is_load = true;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
  d->non_temporal = 0;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

GlobalLoadB96Vglobal::GlobalLoadB96Vglobal(const MachineInst *inst)
    : Vglobal("global_load_b96", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(96, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalLoadB96Vglobal::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 3;
  d->is_load = true;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
  d->non_temporal = 0;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

GlobalLoadB128Vglobal::GlobalLoadB128Vglobal(const MachineInst *inst)
    : Vglobal("global_load_b128", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalLoadB128Vglobal::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 4;
  d->is_load = true;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
  d->non_temporal = 0;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

GlobalStoreB8Vglobal::GlobalStoreB8Vglobal(const MachineInst *inst)
    : Vglobal("global_store_b8", reinterpret_cast<const OpEncoding *>(inst)),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalStoreB8Vglobal::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = false;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
  d->non_temporal = 0;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 1);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.vsrc, lane);
    d->store_data[lane * 1 + 0] = static_cast<uint8_t>(val0);
  }
  set_data(std::move(d));
}

GlobalStoreB16Vglobal::GlobalStoreB16Vglobal(const MachineInst *inst)
    : Vglobal("global_store_b16", reinterpret_cast<const OpEncoding *>(inst)),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalStoreB16Vglobal::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = false;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
  d->non_temporal = 0;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 2);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.vsrc, lane);
    std::memcpy(&d->store_data[lane * 2 + 0], &val0, 2);
  }
  set_data(std::move(d));
}

GlobalStoreB32Vglobal::GlobalStoreB32Vglobal(const MachineInst *inst)
    : Vglobal("global_store_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalStoreB32Vglobal::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = false;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
  d->non_temporal = 0;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.vsrc + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

GlobalStoreB64Vglobal::GlobalStoreB64Vglobal(const MachineInst *inst)
    : Vglobal("global_store_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalStoreB64Vglobal::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 4;
  d->num_elems = 2;
  d->is_load = false;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
  d->non_temporal = 0;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 8);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.vsrc + 0, lane);
    std::memcpy(&d->store_data[lane * 8 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.vsrc + 1, lane);
    std::memcpy(&d->store_data[lane * 8 + 4], &val1, 4);
  }
  set_data(std::move(d));
}

GlobalStoreB96Vglobal::GlobalStoreB96Vglobal(const MachineInst *inst)
    : Vglobal("global_store_b96", reinterpret_cast<const OpEncoding *>(inst)),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(96, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalStoreB96Vglobal::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 4;
  d->num_elems = 3;
  d->is_load = false;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
  d->non_temporal = 0;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 12);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.vsrc + 0, lane);
    std::memcpy(&d->store_data[lane * 12 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.vsrc + 1, lane);
    std::memcpy(&d->store_data[lane * 12 + 4], &val1, 4);
    uint32_t val2 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.vsrc + 2, lane);
    std::memcpy(&d->store_data[lane * 12 + 8], &val2, 4);
  }
  set_data(std::move(d));
}

GlobalStoreB128Vglobal::GlobalStoreB128Vglobal(const MachineInst *inst)
    : Vglobal("global_store_b128", reinterpret_cast<const OpEncoding *>(inst)),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalStoreB128Vglobal::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 4;
  d->num_elems = 4;
  d->is_load = false;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
  d->non_temporal = 0;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 16);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.vsrc + 0, lane);
    std::memcpy(&d->store_data[lane * 16 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.vsrc + 1, lane);
    std::memcpy(&d->store_data[lane * 16 + 4], &val1, 4);
    uint32_t val2 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.vsrc + 2, lane);
    std::memcpy(&d->store_data[lane * 16 + 8], &val2, 4);
    uint32_t val3 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.vsrc + 3, lane);
    std::memcpy(&d->store_data[lane * 16 + 12], &val3, 4);
  }
  set_data(std::move(d));
}

GlobalLoadD16U8Vglobal::GlobalLoadD16U8Vglobal(const MachineInst *inst)
    : Vglobal("global_load_d16_u8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalLoadD16U8Vglobal::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
  d->non_temporal = 0;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

GlobalLoadD16I8Vglobal::GlobalLoadD16I8Vglobal(const MachineInst *inst)
    : Vglobal("global_load_d16_i8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalLoadD16I8Vglobal::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->sign_extend = true;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
  d->non_temporal = 0;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

GlobalLoadD16B16Vglobal::GlobalLoadD16B16Vglobal(const MachineInst *inst)
    : Vglobal("global_load_d16_b16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalLoadD16B16Vglobal::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
  d->non_temporal = 0;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

GlobalLoadD16HiU8Vglobal::GlobalLoadD16HiU8Vglobal(const MachineInst *inst)
    : Vglobal("global_load_d16_hi_u8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalLoadD16HiU8Vglobal::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
  d->non_temporal = 0;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

GlobalLoadD16HiI8Vglobal::GlobalLoadD16HiI8Vglobal(const MachineInst *inst)
    : Vglobal("global_load_d16_hi_i8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalLoadD16HiI8Vglobal::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->sign_extend = true;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
  d->non_temporal = 0;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

GlobalLoadD16HiB16Vglobal::GlobalLoadD16HiB16Vglobal(const MachineInst *inst)
    : Vglobal("global_load_d16_hi_b16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalLoadD16HiB16Vglobal::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
  d->non_temporal = 0;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

GlobalStoreD16HiB8Vglobal::GlobalStoreD16HiB8Vglobal(const MachineInst *inst)
    : Vglobal("global_store_d16_hi_b8", reinterpret_cast<const OpEncoding *>(inst)),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalStoreD16HiB8Vglobal::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = false;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
  d->non_temporal = 0;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 1);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.vsrc, lane);
    d->store_data[lane * 1 + 0] = static_cast<uint8_t>(val0);
  }
  set_data(std::move(d));
}

GlobalStoreD16HiB16Vglobal::GlobalStoreD16HiB16Vglobal(const MachineInst *inst)
    : Vglobal("global_store_d16_hi_b16", reinterpret_cast<const OpEncoding *>(inst)),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalStoreD16HiB16Vglobal::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = false;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
  d->non_temporal = 0;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 2);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.vsrc, lane);
    std::memcpy(&d->store_data[lane * 2 + 0], &val0, 2);
  }
  set_data(std::move(d));
}

GlobalLoadAddtidB32Vglobal::GlobalLoadAddtidB32Vglobal(const MachineInst *inst)
    : Vglobal("global_load_addtid_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&saddr);
}

void GlobalLoadAddtidB32Vglobal::execute(amdgpu::Wavefront &wf) { (void)wf; }

GlobalStoreAddtidB32Vglobal::GlobalStoreAddtidB32Vglobal(const MachineInst *inst)
    : Vglobal("global_store_addtid_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
}

void GlobalStoreAddtidB32Vglobal::execute(amdgpu::Wavefront &wf) { (void)wf; }

GlobalInvVglobal::GlobalInvVglobal(const MachineInst *inst)
    : Vglobal("global_inv", reinterpret_cast<const OpEncoding *>(inst)) {}

void GlobalInvVglobal::execute(amdgpu::Wavefront &wf) { (void)wf; }

GlobalWbVglobal::GlobalWbVglobal(const MachineInst *inst)
    : Vglobal("global_wb", reinterpret_cast<const OpEncoding *>(inst)) {}

void GlobalWbVglobal::execute(amdgpu::Wavefront &wf) { (void)wf; }

GlobalAtomicSwapB32Vglobal::GlobalAtomicSwapB32Vglobal(const MachineInst *inst)
    : Vglobal("global_atomic_swap_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalAtomicSwapB32Vglobal::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (GLOBAL_ATOMIC_SWAP_B32)
}

GlobalAtomicCmpswapB32Vglobal::GlobalAtomicCmpswapB32Vglobal(const MachineInst *inst)
    : Vglobal("global_atomic_cmpswap_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalAtomicCmpswapB32Vglobal::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (GLOBAL_ATOMIC_CMPSWAP_B32)
}

GlobalAtomicAddU32Vglobal::GlobalAtomicAddU32Vglobal(const MachineInst *inst)
    : Vglobal("global_atomic_add_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalAtomicAddU32Vglobal::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (GLOBAL_ATOMIC_ADD_U32)
}

GlobalAtomicSubU32Vglobal::GlobalAtomicSubU32Vglobal(const MachineInst *inst)
    : Vglobal("global_atomic_sub_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalAtomicSubU32Vglobal::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (GLOBAL_ATOMIC_SUB_U32)
}

GlobalAtomicSubClampU32Vglobal::GlobalAtomicSubClampU32Vglobal(const MachineInst *inst)
    : Vglobal("global_atomic_sub_clamp_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalAtomicSubClampU32Vglobal::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (GLOBAL_ATOMIC_SUB_CLAMP_U32)
}

GlobalAtomicMinI32Vglobal::GlobalAtomicMinI32Vglobal(const MachineInst *inst)
    : Vglobal("global_atomic_min_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalAtomicMinI32Vglobal::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (GLOBAL_ATOMIC_MIN_I32)
}

GlobalAtomicMinU32Vglobal::GlobalAtomicMinU32Vglobal(const MachineInst *inst)
    : Vglobal("global_atomic_min_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalAtomicMinU32Vglobal::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (GLOBAL_ATOMIC_MIN_U32)
}

GlobalAtomicMaxI32Vglobal::GlobalAtomicMaxI32Vglobal(const MachineInst *inst)
    : Vglobal("global_atomic_max_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalAtomicMaxI32Vglobal::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (GLOBAL_ATOMIC_MAX_I32)
}

GlobalAtomicMaxU32Vglobal::GlobalAtomicMaxU32Vglobal(const MachineInst *inst)
    : Vglobal("global_atomic_max_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalAtomicMaxU32Vglobal::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (GLOBAL_ATOMIC_MAX_U32)
}

GlobalAtomicAndB32Vglobal::GlobalAtomicAndB32Vglobal(const MachineInst *inst)
    : Vglobal("global_atomic_and_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalAtomicAndB32Vglobal::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (GLOBAL_ATOMIC_AND_B32)
}

GlobalAtomicOrB32Vglobal::GlobalAtomicOrB32Vglobal(const MachineInst *inst)
    : Vglobal("global_atomic_or_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalAtomicOrB32Vglobal::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (GLOBAL_ATOMIC_OR_B32)
}

GlobalAtomicXorB32Vglobal::GlobalAtomicXorB32Vglobal(const MachineInst *inst)
    : Vglobal("global_atomic_xor_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalAtomicXorB32Vglobal::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (GLOBAL_ATOMIC_XOR_B32)
}

GlobalAtomicIncU32Vglobal::GlobalAtomicIncU32Vglobal(const MachineInst *inst)
    : Vglobal("global_atomic_inc_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalAtomicIncU32Vglobal::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (GLOBAL_ATOMIC_INC_U32)
}

GlobalAtomicDecU32Vglobal::GlobalAtomicDecU32Vglobal(const MachineInst *inst)
    : Vglobal("global_atomic_dec_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalAtomicDecU32Vglobal::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (GLOBAL_ATOMIC_DEC_U32)
}

GlobalAtomicSwapB64Vglobal::GlobalAtomicSwapB64Vglobal(const MachineInst *inst)
    : Vglobal("global_atomic_swap_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalAtomicSwapB64Vglobal::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (GLOBAL_ATOMIC_SWAP_B64)
}

GlobalAtomicCmpswapB64Vglobal::GlobalAtomicCmpswapB64Vglobal(const MachineInst *inst)
    : Vglobal("global_atomic_cmpswap_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalAtomicCmpswapB64Vglobal::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (GLOBAL_ATOMIC_CMPSWAP_B64)
}

GlobalAtomicAddU64Vglobal::GlobalAtomicAddU64Vglobal(const MachineInst *inst)
    : Vglobal("global_atomic_add_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalAtomicAddU64Vglobal::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (GLOBAL_ATOMIC_ADD_U64)
}

GlobalAtomicSubU64Vglobal::GlobalAtomicSubU64Vglobal(const MachineInst *inst)
    : Vglobal("global_atomic_sub_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalAtomicSubU64Vglobal::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (GLOBAL_ATOMIC_SUB_U64)
}

GlobalAtomicMinI64Vglobal::GlobalAtomicMinI64Vglobal(const MachineInst *inst)
    : Vglobal("global_atomic_min_i64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalAtomicMinI64Vglobal::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (GLOBAL_ATOMIC_MIN_I64)
}

GlobalAtomicMinU64Vglobal::GlobalAtomicMinU64Vglobal(const MachineInst *inst)
    : Vglobal("global_atomic_min_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalAtomicMinU64Vglobal::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (GLOBAL_ATOMIC_MIN_U64)
}

GlobalAtomicMaxI64Vglobal::GlobalAtomicMaxI64Vglobal(const MachineInst *inst)
    : Vglobal("global_atomic_max_i64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalAtomicMaxI64Vglobal::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (GLOBAL_ATOMIC_MAX_I64)
}

GlobalAtomicMaxU64Vglobal::GlobalAtomicMaxU64Vglobal(const MachineInst *inst)
    : Vglobal("global_atomic_max_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalAtomicMaxU64Vglobal::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (GLOBAL_ATOMIC_MAX_U64)
}

GlobalAtomicAndB64Vglobal::GlobalAtomicAndB64Vglobal(const MachineInst *inst)
    : Vglobal("global_atomic_and_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalAtomicAndB64Vglobal::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (GLOBAL_ATOMIC_AND_B64)
}

GlobalAtomicOrB64Vglobal::GlobalAtomicOrB64Vglobal(const MachineInst *inst)
    : Vglobal("global_atomic_or_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalAtomicOrB64Vglobal::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (GLOBAL_ATOMIC_OR_B64)
}

GlobalAtomicXorB64Vglobal::GlobalAtomicXorB64Vglobal(const MachineInst *inst)
    : Vglobal("global_atomic_xor_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalAtomicXorB64Vglobal::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (GLOBAL_ATOMIC_XOR_B64)
}

GlobalAtomicIncU64Vglobal::GlobalAtomicIncU64Vglobal(const MachineInst *inst)
    : Vglobal("global_atomic_inc_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalAtomicIncU64Vglobal::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (GLOBAL_ATOMIC_INC_U64)
}

GlobalAtomicDecU64Vglobal::GlobalAtomicDecU64Vglobal(const MachineInst *inst)
    : Vglobal("global_atomic_dec_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalAtomicDecU64Vglobal::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (GLOBAL_ATOMIC_DEC_U64)
}

GlobalWbinvVglobal::GlobalWbinvVglobal(const MachineInst *inst)
    : Vglobal("global_wbinv", reinterpret_cast<const OpEncoding *>(inst)) {}

void GlobalWbinvVglobal::execute(amdgpu::Wavefront &wf) { (void)wf; }

GlobalAtomicCondSubU32Vglobal::GlobalAtomicCondSubU32Vglobal(const MachineInst *inst)
    : Vglobal("global_atomic_cond_sub_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalAtomicCondSubU32Vglobal::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (GLOBAL_ATOMIC_COND_SUB_U32)
}

GlobalAtomicMinNumF32Vglobal::GlobalAtomicMinNumF32Vglobal(const MachineInst *inst)
    : Vglobal("global_atomic_min_num_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalAtomicMinNumF32Vglobal::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (GLOBAL_ATOMIC_MIN_NUM_F32)
}

GlobalAtomicMaxNumF32Vglobal::GlobalAtomicMaxNumF32Vglobal(const MachineInst *inst)
    : Vglobal("global_atomic_max_num_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalAtomicMaxNumF32Vglobal::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (GLOBAL_ATOMIC_MAX_NUM_F32)
}

GlobalLoadBlockVglobal::GlobalLoadBlockVglobal(const MachineInst *inst)
    : Vglobal("global_load_block", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(1024, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&saddr);
}

void GlobalLoadBlockVglobal::execute(amdgpu::Wavefront &wf) { (void)wf; }

GlobalStoreBlockVglobal::GlobalStoreBlockVglobal(const MachineInst *inst)
    : Vglobal("global_store_block", reinterpret_cast<const OpEncoding *>(inst)),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(1024, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
}

void GlobalStoreBlockVglobal::execute(amdgpu::Wavefront &wf) { (void)wf; }

GlobalAtomicAddF32Vglobal::GlobalAtomicAddF32Vglobal(const MachineInst *inst)
    : Vglobal("global_atomic_add_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalAtomicAddF32Vglobal::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.nv != 0);
  d->atomic_op = amdgpu::AtomicOp::FADD;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
  d->non_temporal = 0;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.vsrc + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

GlobalLoadTrB128Vglobal::GlobalLoadTrB128Vglobal(const MachineInst *inst)
    : Vglobal("global_load_tr_b128", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&saddr);
}

void GlobalLoadTrB128Vglobal::execute(amdgpu::Wavefront &wf) { (void)wf; }

GlobalLoadTrB64Vglobal::GlobalLoadTrB64Vglobal(const MachineInst *inst)
    : Vglobal("global_load_tr_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&saddr);
}

void GlobalLoadTrB64Vglobal::execute(amdgpu::Wavefront &wf) { (void)wf; }

GlobalAtomicPkAddF16Vglobal::GlobalAtomicPkAddF16Vglobal(const MachineInst *inst)
    : Vglobal("global_atomic_pk_add_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalAtomicPkAddF16Vglobal::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.nv != 0);
  d->atomic_op = amdgpu::AtomicOp::FADD;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
  d->non_temporal = 0;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.vsrc + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

GlobalAtomicPkAddBf16Vglobal::GlobalAtomicPkAddBf16Vglobal(const MachineInst *inst)
    : Vglobal("global_atomic_pk_add_bf16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalAtomicPkAddBf16Vglobal::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.nv != 0);
  d->atomic_op = amdgpu::AtomicOp::FADD;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
  d->non_temporal = 0;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.vsrc + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

GlobalAtomicOrderedAddB64Vglobal::GlobalAtomicOrderedAddB64Vglobal(const MachineInst *inst)
    : Vglobal("global_atomic_ordered_add_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      saddr(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->saddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&saddr);
  flags_ |= MEMORY_OP;
}

void GlobalAtomicOrderedAddB64Vglobal::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (GLOBAL_ATOMIC_ORDERED_ADD_B64)
}

} // namespace rdna4
} // namespace rocjitsu

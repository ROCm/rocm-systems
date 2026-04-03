// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/rdna4/vflat.h"
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

FlatLoadU8Vflat::FlatLoadU8Vflat(const MachineInst *inst)
    : Vflat("flat_load_u8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  flags_ |= MEMORY_OP;
}

void FlatLoadU8Vflat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = mtype_from_bits(inst_.nv, inst_.nv);
  d->non_temporal = 0;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

FlatLoadI8Vflat::FlatLoadI8Vflat(const MachineInst *inst)
    : Vflat("flat_load_i8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  flags_ |= MEMORY_OP;
}

void FlatLoadI8Vflat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->sign_extend = true;
  d->mtype = mtype_from_bits(inst_.nv, inst_.nv);
  d->non_temporal = 0;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

FlatLoadU16Vflat::FlatLoadU16Vflat(const MachineInst *inst)
    : Vflat("flat_load_u16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  flags_ |= MEMORY_OP;
}

void FlatLoadU16Vflat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = mtype_from_bits(inst_.nv, inst_.nv);
  d->non_temporal = 0;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

FlatLoadI16Vflat::FlatLoadI16Vflat(const MachineInst *inst)
    : Vflat("flat_load_i16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  flags_ |= MEMORY_OP;
}

void FlatLoadI16Vflat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = true;
  d->sign_extend = true;
  d->mtype = mtype_from_bits(inst_.nv, inst_.nv);
  d->non_temporal = 0;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

FlatLoadB32Vflat::FlatLoadB32Vflat(const MachineInst *inst)
    : Vflat("flat_load_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  flags_ |= MEMORY_OP;
}

void FlatLoadB32Vflat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = mtype_from_bits(inst_.nv, inst_.nv);
  d->non_temporal = 0;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

FlatLoadB64Vflat::FlatLoadB64Vflat(const MachineInst *inst)
    : Vflat("flat_load_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  flags_ |= MEMORY_OP;
}

void FlatLoadB64Vflat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 2;
  d->is_load = true;
  d->mtype = mtype_from_bits(inst_.nv, inst_.nv);
  d->non_temporal = 0;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

FlatLoadB96Vflat::FlatLoadB96Vflat(const MachineInst *inst)
    : Vflat("flat_load_b96", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(96, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  flags_ |= MEMORY_OP;
}

void FlatLoadB96Vflat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 3;
  d->is_load = true;
  d->mtype = mtype_from_bits(inst_.nv, inst_.nv);
  d->non_temporal = 0;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

FlatLoadB128Vflat::FlatLoadB128Vflat(const MachineInst *inst)
    : Vflat("flat_load_b128", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  flags_ |= MEMORY_OP;
}

void FlatLoadB128Vflat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 4;
  d->is_load = true;
  d->mtype = mtype_from_bits(inst_.nv, inst_.nv);
  d->non_temporal = 0;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

FlatStoreB8Vflat::FlatStoreB8Vflat(const MachineInst *inst)
    : Vflat("flat_store_b8", reinterpret_cast<const OpEncoding *>(inst)),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc) {
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  flags_ |= MEMORY_OP;
}

void FlatStoreB8Vflat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = false;
  d->mtype = mtype_from_bits(inst_.nv, inst_.nv);
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

FlatStoreB16Vflat::FlatStoreB16Vflat(const MachineInst *inst)
    : Vflat("flat_store_b16", reinterpret_cast<const OpEncoding *>(inst)),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc) {
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  flags_ |= MEMORY_OP;
}

void FlatStoreB16Vflat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = false;
  d->mtype = mtype_from_bits(inst_.nv, inst_.nv);
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

FlatStoreB32Vflat::FlatStoreB32Vflat(const MachineInst *inst)
    : Vflat("flat_store_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc) {
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  flags_ |= MEMORY_OP;
}

void FlatStoreB32Vflat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = false;
  d->mtype = mtype_from_bits(inst_.nv, inst_.nv);
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

FlatStoreB64Vflat::FlatStoreB64Vflat(const MachineInst *inst)
    : Vflat("flat_store_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc) {
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  flags_ |= MEMORY_OP;
}

void FlatStoreB64Vflat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 4;
  d->num_elems = 2;
  d->is_load = false;
  d->mtype = mtype_from_bits(inst_.nv, inst_.nv);
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

FlatStoreB96Vflat::FlatStoreB96Vflat(const MachineInst *inst)
    : Vflat("flat_store_b96", reinterpret_cast<const OpEncoding *>(inst)),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(96, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc) {
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  flags_ |= MEMORY_OP;
}

void FlatStoreB96Vflat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 4;
  d->num_elems = 3;
  d->is_load = false;
  d->mtype = mtype_from_bits(inst_.nv, inst_.nv);
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

FlatStoreB128Vflat::FlatStoreB128Vflat(const MachineInst *inst)
    : Vflat("flat_store_b128", reinterpret_cast<const OpEncoding *>(inst)),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc) {
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  flags_ |= MEMORY_OP;
}

void FlatStoreB128Vflat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 4;
  d->num_elems = 4;
  d->is_load = false;
  d->mtype = mtype_from_bits(inst_.nv, inst_.nv);
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

FlatLoadD16U8Vflat::FlatLoadD16U8Vflat(const MachineInst *inst)
    : Vflat("flat_load_d16_u8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  flags_ |= MEMORY_OP;
}

void FlatLoadD16U8Vflat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = mtype_from_bits(inst_.nv, inst_.nv);
  d->non_temporal = 0;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

FlatLoadD16I8Vflat::FlatLoadD16I8Vflat(const MachineInst *inst)
    : Vflat("flat_load_d16_i8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  flags_ |= MEMORY_OP;
}

void FlatLoadD16I8Vflat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->sign_extend = true;
  d->mtype = mtype_from_bits(inst_.nv, inst_.nv);
  d->non_temporal = 0;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

FlatLoadD16B16Vflat::FlatLoadD16B16Vflat(const MachineInst *inst)
    : Vflat("flat_load_d16_b16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  flags_ |= MEMORY_OP;
}

void FlatLoadD16B16Vflat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = mtype_from_bits(inst_.nv, inst_.nv);
  d->non_temporal = 0;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

FlatLoadD16HiU8Vflat::FlatLoadD16HiU8Vflat(const MachineInst *inst)
    : Vflat("flat_load_d16_hi_u8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  flags_ |= MEMORY_OP;
}

void FlatLoadD16HiU8Vflat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = mtype_from_bits(inst_.nv, inst_.nv);
  d->non_temporal = 0;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

FlatLoadD16HiI8Vflat::FlatLoadD16HiI8Vflat(const MachineInst *inst)
    : Vflat("flat_load_d16_hi_i8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  flags_ |= MEMORY_OP;
}

void FlatLoadD16HiI8Vflat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->sign_extend = true;
  d->mtype = mtype_from_bits(inst_.nv, inst_.nv);
  d->non_temporal = 0;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

FlatLoadD16HiB16Vflat::FlatLoadD16HiB16Vflat(const MachineInst *inst)
    : Vflat("flat_load_d16_hi_b16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  flags_ |= MEMORY_OP;
}

void FlatLoadD16HiB16Vflat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = mtype_from_bits(inst_.nv, inst_.nv);
  d->non_temporal = 0;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

FlatStoreD16HiB8Vflat::FlatStoreD16HiB8Vflat(const MachineInst *inst)
    : Vflat("flat_store_d16_hi_b8", reinterpret_cast<const OpEncoding *>(inst)),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc) {
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  flags_ |= MEMORY_OP;
}

void FlatStoreD16HiB8Vflat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = false;
  d->mtype = mtype_from_bits(inst_.nv, inst_.nv);
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

FlatStoreD16HiB16Vflat::FlatStoreD16HiB16Vflat(const MachineInst *inst)
    : Vflat("flat_store_d16_hi_b16", reinterpret_cast<const OpEncoding *>(inst)),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc) {
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  flags_ |= MEMORY_OP;
}

void FlatStoreD16HiB16Vflat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = false;
  d->mtype = mtype_from_bits(inst_.nv, inst_.nv);
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

FlatAtomicSwapB32Vflat::FlatAtomicSwapB32Vflat(const MachineInst *inst)
    : Vflat("flat_atomic_swap_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  flags_ |= MEMORY_OP;
}

void FlatAtomicSwapB32Vflat::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (FLAT_ATOMIC_SWAP_B32)
}

FlatAtomicCmpswapB32Vflat::FlatAtomicCmpswapB32Vflat(const MachineInst *inst)
    : Vflat("flat_atomic_cmpswap_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  flags_ |= MEMORY_OP;
}

void FlatAtomicCmpswapB32Vflat::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (FLAT_ATOMIC_CMPSWAP_B32)
}

FlatAtomicAddU32Vflat::FlatAtomicAddU32Vflat(const MachineInst *inst)
    : Vflat("flat_atomic_add_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  flags_ |= MEMORY_OP;
}

void FlatAtomicAddU32Vflat::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (FLAT_ATOMIC_ADD_U32)
}

FlatAtomicSubU32Vflat::FlatAtomicSubU32Vflat(const MachineInst *inst)
    : Vflat("flat_atomic_sub_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  flags_ |= MEMORY_OP;
}

void FlatAtomicSubU32Vflat::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (FLAT_ATOMIC_SUB_U32)
}

FlatAtomicSubClampU32Vflat::FlatAtomicSubClampU32Vflat(const MachineInst *inst)
    : Vflat("flat_atomic_sub_clamp_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  flags_ |= MEMORY_OP;
}

void FlatAtomicSubClampU32Vflat::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (FLAT_ATOMIC_SUB_CLAMP_U32)
}

FlatAtomicMinI32Vflat::FlatAtomicMinI32Vflat(const MachineInst *inst)
    : Vflat("flat_atomic_min_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  flags_ |= MEMORY_OP;
}

void FlatAtomicMinI32Vflat::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (FLAT_ATOMIC_MIN_I32)
}

FlatAtomicMinU32Vflat::FlatAtomicMinU32Vflat(const MachineInst *inst)
    : Vflat("flat_atomic_min_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  flags_ |= MEMORY_OP;
}

void FlatAtomicMinU32Vflat::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (FLAT_ATOMIC_MIN_U32)
}

FlatAtomicMaxI32Vflat::FlatAtomicMaxI32Vflat(const MachineInst *inst)
    : Vflat("flat_atomic_max_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  flags_ |= MEMORY_OP;
}

void FlatAtomicMaxI32Vflat::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (FLAT_ATOMIC_MAX_I32)
}

FlatAtomicMaxU32Vflat::FlatAtomicMaxU32Vflat(const MachineInst *inst)
    : Vflat("flat_atomic_max_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  flags_ |= MEMORY_OP;
}

void FlatAtomicMaxU32Vflat::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (FLAT_ATOMIC_MAX_U32)
}

FlatAtomicAndB32Vflat::FlatAtomicAndB32Vflat(const MachineInst *inst)
    : Vflat("flat_atomic_and_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  flags_ |= MEMORY_OP;
}

void FlatAtomicAndB32Vflat::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (FLAT_ATOMIC_AND_B32)
}

FlatAtomicOrB32Vflat::FlatAtomicOrB32Vflat(const MachineInst *inst)
    : Vflat("flat_atomic_or_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  flags_ |= MEMORY_OP;
}

void FlatAtomicOrB32Vflat::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (FLAT_ATOMIC_OR_B32)
}

FlatAtomicXorB32Vflat::FlatAtomicXorB32Vflat(const MachineInst *inst)
    : Vflat("flat_atomic_xor_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  flags_ |= MEMORY_OP;
}

void FlatAtomicXorB32Vflat::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (FLAT_ATOMIC_XOR_B32)
}

FlatAtomicIncU32Vflat::FlatAtomicIncU32Vflat(const MachineInst *inst)
    : Vflat("flat_atomic_inc_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  flags_ |= MEMORY_OP;
}

void FlatAtomicIncU32Vflat::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (FLAT_ATOMIC_INC_U32)
}

FlatAtomicDecU32Vflat::FlatAtomicDecU32Vflat(const MachineInst *inst)
    : Vflat("flat_atomic_dec_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  flags_ |= MEMORY_OP;
}

void FlatAtomicDecU32Vflat::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (FLAT_ATOMIC_DEC_U32)
}

FlatAtomicSwapB64Vflat::FlatAtomicSwapB64Vflat(const MachineInst *inst)
    : Vflat("flat_atomic_swap_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  flags_ |= MEMORY_OP;
}

void FlatAtomicSwapB64Vflat::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (FLAT_ATOMIC_SWAP_B64)
}

FlatAtomicCmpswapB64Vflat::FlatAtomicCmpswapB64Vflat(const MachineInst *inst)
    : Vflat("flat_atomic_cmpswap_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  flags_ |= MEMORY_OP;
}

void FlatAtomicCmpswapB64Vflat::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (FLAT_ATOMIC_CMPSWAP_B64)
}

FlatAtomicAddU64Vflat::FlatAtomicAddU64Vflat(const MachineInst *inst)
    : Vflat("flat_atomic_add_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  flags_ |= MEMORY_OP;
}

void FlatAtomicAddU64Vflat::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (FLAT_ATOMIC_ADD_U64)
}

FlatAtomicSubU64Vflat::FlatAtomicSubU64Vflat(const MachineInst *inst)
    : Vflat("flat_atomic_sub_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  flags_ |= MEMORY_OP;
}

void FlatAtomicSubU64Vflat::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (FLAT_ATOMIC_SUB_U64)
}

FlatAtomicMinI64Vflat::FlatAtomicMinI64Vflat(const MachineInst *inst)
    : Vflat("flat_atomic_min_i64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  flags_ |= MEMORY_OP;
}

void FlatAtomicMinI64Vflat::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (FLAT_ATOMIC_MIN_I64)
}

FlatAtomicMinU64Vflat::FlatAtomicMinU64Vflat(const MachineInst *inst)
    : Vflat("flat_atomic_min_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  flags_ |= MEMORY_OP;
}

void FlatAtomicMinU64Vflat::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (FLAT_ATOMIC_MIN_U64)
}

FlatAtomicMaxI64Vflat::FlatAtomicMaxI64Vflat(const MachineInst *inst)
    : Vflat("flat_atomic_max_i64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  flags_ |= MEMORY_OP;
}

void FlatAtomicMaxI64Vflat::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (FLAT_ATOMIC_MAX_I64)
}

FlatAtomicMaxU64Vflat::FlatAtomicMaxU64Vflat(const MachineInst *inst)
    : Vflat("flat_atomic_max_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  flags_ |= MEMORY_OP;
}

void FlatAtomicMaxU64Vflat::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (FLAT_ATOMIC_MAX_U64)
}

FlatAtomicAndB64Vflat::FlatAtomicAndB64Vflat(const MachineInst *inst)
    : Vflat("flat_atomic_and_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  flags_ |= MEMORY_OP;
}

void FlatAtomicAndB64Vflat::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (FLAT_ATOMIC_AND_B64)
}

FlatAtomicOrB64Vflat::FlatAtomicOrB64Vflat(const MachineInst *inst)
    : Vflat("flat_atomic_or_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  flags_ |= MEMORY_OP;
}

void FlatAtomicOrB64Vflat::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (FLAT_ATOMIC_OR_B64)
}

FlatAtomicXorB64Vflat::FlatAtomicXorB64Vflat(const MachineInst *inst)
    : Vflat("flat_atomic_xor_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  flags_ |= MEMORY_OP;
}

void FlatAtomicXorB64Vflat::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (FLAT_ATOMIC_XOR_B64)
}

FlatAtomicIncU64Vflat::FlatAtomicIncU64Vflat(const MachineInst *inst)
    : Vflat("flat_atomic_inc_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  flags_ |= MEMORY_OP;
}

void FlatAtomicIncU64Vflat::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (FLAT_ATOMIC_INC_U64)
}

FlatAtomicDecU64Vflat::FlatAtomicDecU64Vflat(const MachineInst *inst)
    : Vflat("flat_atomic_dec_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  flags_ |= MEMORY_OP;
}

void FlatAtomicDecU64Vflat::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (FLAT_ATOMIC_DEC_U64)
}

FlatAtomicCondSubU32Vflat::FlatAtomicCondSubU32Vflat(const MachineInst *inst)
    : Vflat("flat_atomic_cond_sub_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  flags_ |= MEMORY_OP;
}

void FlatAtomicCondSubU32Vflat::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (FLAT_ATOMIC_COND_SUB_U32)
}

FlatAtomicMinNumF32Vflat::FlatAtomicMinNumF32Vflat(const MachineInst *inst)
    : Vflat("flat_atomic_min_num_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  flags_ |= MEMORY_OP;
}

void FlatAtomicMinNumF32Vflat::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (FLAT_ATOMIC_MIN_NUM_F32)
}

FlatAtomicMaxNumF32Vflat::FlatAtomicMaxNumF32Vflat(const MachineInst *inst)
    : Vflat("flat_atomic_max_num_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  flags_ |= MEMORY_OP;
}

void FlatAtomicMaxNumF32Vflat::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled flat_atomic variant (FLAT_ATOMIC_MAX_NUM_F32)
}

FlatAtomicAddF32Vflat::FlatAtomicAddF32Vflat(const MachineInst *inst)
    : Vflat("flat_atomic_add_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  flags_ |= MEMORY_OP;
}

void FlatAtomicAddF32Vflat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.nv != 0);
  d->atomic_op = amdgpu::AtomicOp::FADD;
  d->mtype = mtype_from_bits(inst_.nv, inst_.nv);
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

FlatAtomicPkAddF16Vflat::FlatAtomicPkAddF16Vflat(const MachineInst *inst)
    : Vflat("flat_atomic_pk_add_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  flags_ |= MEMORY_OP;
}

void FlatAtomicPkAddF16Vflat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.nv != 0);
  d->atomic_op = amdgpu::AtomicOp::FADD;
  d->mtype = mtype_from_bits(inst_.nv, inst_.nv);
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

FlatAtomicPkAddBf16Vflat::FlatAtomicPkAddBf16Vflat(const MachineInst *inst)
    : Vflat("flat_atomic_pk_add_bf16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&vsrc);
  flags_ |= MEMORY_OP;
}

void FlatAtomicPkAddBf16Vflat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.nv != 0);
  d->atomic_op = amdgpu::AtomicOp::FADD;
  d->mtype = mtype_from_bits(inst_.nv, inst_.nv);
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

} // namespace rdna4
} // namespace rocjitsu

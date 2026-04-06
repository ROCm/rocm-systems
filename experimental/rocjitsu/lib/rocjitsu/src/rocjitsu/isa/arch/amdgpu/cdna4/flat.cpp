// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/cdna4/flat.h"
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

FlatLoadUbyteFlat::FlatLoadUbyteFlat(const MachineInst *inst)
    : Flat("flat_load_ubyte", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void FlatLoadUbyteFlat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

FlatLoadSbyteFlat::FlatLoadSbyteFlat(const MachineInst *inst)
    : Flat("flat_load_sbyte", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void FlatLoadSbyteFlat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->sign_extend = true;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

FlatLoadUshortFlat::FlatLoadUshortFlat(const MachineInst *inst)
    : Flat("flat_load_ushort", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void FlatLoadUshortFlat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

FlatLoadSshortFlat::FlatLoadSshortFlat(const MachineInst *inst)
    : Flat("flat_load_sshort", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void FlatLoadSshortFlat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = true;
  d->sign_extend = true;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

FlatLoadDwordFlat::FlatLoadDwordFlat(const MachineInst *inst)
    : Flat("flat_load_dword", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void FlatLoadDwordFlat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

FlatLoadDwordx2Flat::FlatLoadDwordx2Flat(const MachineInst *inst)
    : Flat("flat_load_dwordx2", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void FlatLoadDwordx2Flat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 2;
  d->is_load = true;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

FlatLoadDwordx3Flat::FlatLoadDwordx3Flat(const MachineInst *inst)
    : Flat("flat_load_dwordx3", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(96, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void FlatLoadDwordx3Flat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 3;
  d->is_load = true;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

FlatLoadDwordx4Flat::FlatLoadDwordx4Flat(const MachineInst *inst)
    : Flat("flat_load_dwordx4", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void FlatLoadDwordx4Flat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 4;
  d->is_load = true;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

FlatStoreByteFlat::FlatStoreByteFlat(const MachineInst *inst)
    : Flat("flat_store_byte", reinterpret_cast<const OpEncoding *>(inst)),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->data) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data);
  flags_ |= MEMORY_OP;
}

void FlatStoreByteFlat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = false;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 1);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data, lane);
    d->store_data[lane * 1 + 0] = static_cast<uint8_t>(val0);
  }
  set_data(std::move(d));
}

FlatStoreByteD16HiFlat::FlatStoreByteD16HiFlat(const MachineInst *inst)
    : Flat("flat_store_byte_d16_hi", reinterpret_cast<const OpEncoding *>(inst)),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->data) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data);
  flags_ |= MEMORY_OP;
}

void FlatStoreByteD16HiFlat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = false;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 1);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data, lane);
    d->store_data[lane * 1 + 0] = static_cast<uint8_t>(val0);
  }
  set_data(std::move(d));
}

FlatStoreShortFlat::FlatStoreShortFlat(const MachineInst *inst)
    : Flat("flat_store_short", reinterpret_cast<const OpEncoding *>(inst)),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->data) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data);
  flags_ |= MEMORY_OP;
}

void FlatStoreShortFlat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = false;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 2);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data, lane);
    std::memcpy(&d->store_data[lane * 2 + 0], &val0, 2);
  }
  set_data(std::move(d));
}

FlatStoreShortD16HiFlat::FlatStoreShortD16HiFlat(const MachineInst *inst)
    : Flat("flat_store_short_d16_hi", reinterpret_cast<const OpEncoding *>(inst)),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->data) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data);
  flags_ |= MEMORY_OP;
}

void FlatStoreShortD16HiFlat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = false;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 2);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data, lane);
    std::memcpy(&d->store_data[lane * 2 + 0], &val0, 2);
  }
  set_data(std::move(d));
}

FlatStoreDwordFlat::FlatStoreDwordFlat(const MachineInst *inst)
    : Flat("flat_store_dword", reinterpret_cast<const OpEncoding *>(inst)),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->data) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data);
  flags_ |= MEMORY_OP;
}

void FlatStoreDwordFlat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = false;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

FlatStoreDwordx2Flat::FlatStoreDwordx2Flat(const MachineInst *inst)
    : Flat("flat_store_dwordx2", reinterpret_cast<const OpEncoding *>(inst)),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->data) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data);
  flags_ |= MEMORY_OP;
}

void FlatStoreDwordx2Flat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 4;
  d->num_elems = 2;
  d->is_load = false;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 8);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 0, lane);
    std::memcpy(&d->store_data[lane * 8 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 1, lane);
    std::memcpy(&d->store_data[lane * 8 + 4], &val1, 4);
  }
  set_data(std::move(d));
}

FlatStoreDwordx3Flat::FlatStoreDwordx3Flat(const MachineInst *inst)
    : Flat("flat_store_dwordx3", reinterpret_cast<const OpEncoding *>(inst)),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data(96, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->data) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data);
  flags_ |= MEMORY_OP;
}

void FlatStoreDwordx3Flat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 4;
  d->num_elems = 3;
  d->is_load = false;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 12);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 0, lane);
    std::memcpy(&d->store_data[lane * 12 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 1, lane);
    std::memcpy(&d->store_data[lane * 12 + 4], &val1, 4);
    uint32_t val2 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 2, lane);
    std::memcpy(&d->store_data[lane * 12 + 8], &val2, 4);
  }
  set_data(std::move(d));
}

FlatStoreDwordx4Flat::FlatStoreDwordx4Flat(const MachineInst *inst)
    : Flat("flat_store_dwordx4", reinterpret_cast<const OpEncoding *>(inst)),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data(128, OperandType::OPR_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->data) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data);
  flags_ |= MEMORY_OP;
}

void FlatStoreDwordx4Flat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 4;
  d->num_elems = 4;
  d->is_load = false;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 16);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 0, lane);
    std::memcpy(&d->store_data[lane * 16 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 1, lane);
    std::memcpy(&d->store_data[lane * 16 + 4], &val1, 4);
    uint32_t val2 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 2, lane);
    std::memcpy(&d->store_data[lane * 16 + 8], &val2, 4);
    uint32_t val3 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 3, lane);
    std::memcpy(&d->store_data[lane * 16 + 12], &val3, 4);
  }
  set_data(std::move(d));
}

FlatLoadUbyteD16Flat::FlatLoadUbyteD16Flat(const MachineInst *inst)
    : Flat("flat_load_ubyte_d16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void FlatLoadUbyteD16Flat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

FlatLoadUbyteD16HiFlat::FlatLoadUbyteD16HiFlat(const MachineInst *inst)
    : Flat("flat_load_ubyte_d16_hi", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void FlatLoadUbyteD16HiFlat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

FlatLoadSbyteD16Flat::FlatLoadSbyteD16Flat(const MachineInst *inst)
    : Flat("flat_load_sbyte_d16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void FlatLoadSbyteD16Flat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->sign_extend = true;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

FlatLoadSbyteD16HiFlat::FlatLoadSbyteD16HiFlat(const MachineInst *inst)
    : Flat("flat_load_sbyte_d16_hi", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void FlatLoadSbyteD16HiFlat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->sign_extend = true;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

FlatLoadShortD16Flat::FlatLoadShortD16Flat(const MachineInst *inst)
    : Flat("flat_load_short_d16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void FlatLoadShortD16Flat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

FlatLoadShortD16HiFlat::FlatLoadShortD16HiFlat(const MachineInst *inst)
    : Flat("flat_load_short_d16_hi", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void FlatLoadShortD16HiFlat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

FlatAtomicSwapFlat::FlatAtomicSwapFlat(const MachineInst *inst)
    : Flat("flat_atomic_swap", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->data) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data);
  flags_ |= MEMORY_OP;
}

void FlatAtomicSwapFlat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::SWAP;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

FlatAtomicCmpswapFlat::FlatAtomicCmpswapFlat(const MachineInst *inst)
    : Flat("flat_atomic_cmpswap", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->data) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data);
  flags_ |= MEMORY_OP;
}

void FlatAtomicCmpswapFlat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::CMPSWAP;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 8);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 0, lane);
    std::memcpy(&d->store_data[lane * 8 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 1, lane);
    std::memcpy(&d->store_data[lane * 8 + 4], &val1, 4);
  }
  set_data(std::move(d));
}

FlatAtomicAddFlat::FlatAtomicAddFlat(const MachineInst *inst)
    : Flat("flat_atomic_add", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->data) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data);
  flags_ |= MEMORY_OP;
}

void FlatAtomicAddFlat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::ADD;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

FlatAtomicSubFlat::FlatAtomicSubFlat(const MachineInst *inst)
    : Flat("flat_atomic_sub", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->data) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data);
  flags_ |= MEMORY_OP;
}

void FlatAtomicSubFlat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::SUB;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

FlatAtomicSminFlat::FlatAtomicSminFlat(const MachineInst *inst)
    : Flat("flat_atomic_smin", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->data) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data);
  flags_ |= MEMORY_OP;
}

void FlatAtomicSminFlat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::SMIN;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

FlatAtomicUminFlat::FlatAtomicUminFlat(const MachineInst *inst)
    : Flat("flat_atomic_umin", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->data) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data);
  flags_ |= MEMORY_OP;
}

void FlatAtomicUminFlat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::UMIN;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

FlatAtomicSmaxFlat::FlatAtomicSmaxFlat(const MachineInst *inst)
    : Flat("flat_atomic_smax", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->data) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data);
  flags_ |= MEMORY_OP;
}

void FlatAtomicSmaxFlat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::SMAX;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

FlatAtomicUmaxFlat::FlatAtomicUmaxFlat(const MachineInst *inst)
    : Flat("flat_atomic_umax", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->data) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data);
  flags_ |= MEMORY_OP;
}

void FlatAtomicUmaxFlat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::UMAX;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

FlatAtomicAndFlat::FlatAtomicAndFlat(const MachineInst *inst)
    : Flat("flat_atomic_and", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->data) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data);
  flags_ |= MEMORY_OP;
}

void FlatAtomicAndFlat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::AND;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

FlatAtomicOrFlat::FlatAtomicOrFlat(const MachineInst *inst)
    : Flat("flat_atomic_or", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->data) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data);
  flags_ |= MEMORY_OP;
}

void FlatAtomicOrFlat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::OR;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

FlatAtomicXorFlat::FlatAtomicXorFlat(const MachineInst *inst)
    : Flat("flat_atomic_xor", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->data) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data);
  flags_ |= MEMORY_OP;
}

void FlatAtomicXorFlat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::XOR;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

FlatAtomicIncFlat::FlatAtomicIncFlat(const MachineInst *inst)
    : Flat("flat_atomic_inc", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->data) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data);
  flags_ |= MEMORY_OP;
}

void FlatAtomicIncFlat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::INC;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

FlatAtomicDecFlat::FlatAtomicDecFlat(const MachineInst *inst)
    : Flat("flat_atomic_dec", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->data) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data);
  flags_ |= MEMORY_OP;
}

void FlatAtomicDecFlat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::DEC;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

FlatAtomicAddF32Flat::FlatAtomicAddF32Flat(const MachineInst *inst)
    : Flat("flat_atomic_add_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->data) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data);
  flags_ |= MEMORY_OP;
}

void FlatAtomicAddF32Flat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::FADD;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

FlatAtomicPkAddF16Flat::FlatAtomicPkAddF16Flat(const MachineInst *inst)
    : Flat("flat_atomic_pk_add_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->data) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data);
  flags_ |= MEMORY_OP;
}

void FlatAtomicPkAddF16Flat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::FADD;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

FlatAtomicAddF64Flat::FlatAtomicAddF64Flat(const MachineInst *inst)
    : Flat("flat_atomic_add_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->data) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data);
  flags_ |= MEMORY_OP;
}

void FlatAtomicAddF64Flat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::FADD;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 8);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 0, lane);
    std::memcpy(&d->store_data[lane * 8 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 1, lane);
    std::memcpy(&d->store_data[lane * 8 + 4], &val1, 4);
  }
  set_data(std::move(d));
}

FlatAtomicMinF64Flat::FlatAtomicMinF64Flat(const MachineInst *inst)
    : Flat("flat_atomic_min_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->data) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data);
  flags_ |= MEMORY_OP;
}

void FlatAtomicMinF64Flat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::FMIN;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 8);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 0, lane);
    std::memcpy(&d->store_data[lane * 8 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 1, lane);
    std::memcpy(&d->store_data[lane * 8 + 4], &val1, 4);
  }
  set_data(std::move(d));
}

FlatAtomicMaxF64Flat::FlatAtomicMaxF64Flat(const MachineInst *inst)
    : Flat("flat_atomic_max_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->data) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data);
  flags_ |= MEMORY_OP;
}

void FlatAtomicMaxF64Flat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::FMAX;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 8);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 0, lane);
    std::memcpy(&d->store_data[lane * 8 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 1, lane);
    std::memcpy(&d->store_data[lane * 8 + 4], &val1, 4);
  }
  set_data(std::move(d));
}

FlatAtomicPkAddBf16Flat::FlatAtomicPkAddBf16Flat(const MachineInst *inst)
    : Flat("flat_atomic_pk_add_bf16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->data) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data);
  flags_ |= MEMORY_OP;
}

void FlatAtomicPkAddBf16Flat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::FADD;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

FlatAtomicSwapX2Flat::FlatAtomicSwapX2Flat(const MachineInst *inst)
    : Flat("flat_atomic_swap_x2", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->data) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data);
  flags_ |= MEMORY_OP;
}

void FlatAtomicSwapX2Flat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::SWAP;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 8);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 0, lane);
    std::memcpy(&d->store_data[lane * 8 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 1, lane);
    std::memcpy(&d->store_data[lane * 8 + 4], &val1, 4);
  }
  set_data(std::move(d));
}

FlatAtomicCmpswapX2Flat::FlatAtomicCmpswapX2Flat(const MachineInst *inst)
    : Flat("flat_atomic_cmpswap_x2", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data(128, OperandType::OPR_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->data) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data);
  flags_ |= MEMORY_OP;
}

void FlatAtomicCmpswapX2Flat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::CMPSWAP;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 16);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 0, lane);
    std::memcpy(&d->store_data[lane * 16 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 1, lane);
    std::memcpy(&d->store_data[lane * 16 + 4], &val1, 4);
    uint32_t val2 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 2, lane);
    std::memcpy(&d->store_data[lane * 16 + 8], &val2, 4);
    uint32_t val3 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 3, lane);
    std::memcpy(&d->store_data[lane * 16 + 12], &val3, 4);
  }
  set_data(std::move(d));
}

FlatAtomicAddX2Flat::FlatAtomicAddX2Flat(const MachineInst *inst)
    : Flat("flat_atomic_add_x2", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->data) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data);
  flags_ |= MEMORY_OP;
}

void FlatAtomicAddX2Flat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::ADD;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 8);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 0, lane);
    std::memcpy(&d->store_data[lane * 8 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 1, lane);
    std::memcpy(&d->store_data[lane * 8 + 4], &val1, 4);
  }
  set_data(std::move(d));
}

FlatAtomicSubX2Flat::FlatAtomicSubX2Flat(const MachineInst *inst)
    : Flat("flat_atomic_sub_x2", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->data) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data);
  flags_ |= MEMORY_OP;
}

void FlatAtomicSubX2Flat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::SUB;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 8);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 0, lane);
    std::memcpy(&d->store_data[lane * 8 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 1, lane);
    std::memcpy(&d->store_data[lane * 8 + 4], &val1, 4);
  }
  set_data(std::move(d));
}

FlatAtomicSminX2Flat::FlatAtomicSminX2Flat(const MachineInst *inst)
    : Flat("flat_atomic_smin_x2", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->data) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data);
  flags_ |= MEMORY_OP;
}

void FlatAtomicSminX2Flat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::SMIN;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 8);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 0, lane);
    std::memcpy(&d->store_data[lane * 8 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 1, lane);
    std::memcpy(&d->store_data[lane * 8 + 4], &val1, 4);
  }
  set_data(std::move(d));
}

FlatAtomicUminX2Flat::FlatAtomicUminX2Flat(const MachineInst *inst)
    : Flat("flat_atomic_umin_x2", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->data) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data);
  flags_ |= MEMORY_OP;
}

void FlatAtomicUminX2Flat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::UMIN;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 8);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 0, lane);
    std::memcpy(&d->store_data[lane * 8 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 1, lane);
    std::memcpy(&d->store_data[lane * 8 + 4], &val1, 4);
  }
  set_data(std::move(d));
}

FlatAtomicSmaxX2Flat::FlatAtomicSmaxX2Flat(const MachineInst *inst)
    : Flat("flat_atomic_smax_x2", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->data) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data);
  flags_ |= MEMORY_OP;
}

void FlatAtomicSmaxX2Flat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::SMAX;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 8);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 0, lane);
    std::memcpy(&d->store_data[lane * 8 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 1, lane);
    std::memcpy(&d->store_data[lane * 8 + 4], &val1, 4);
  }
  set_data(std::move(d));
}

FlatAtomicUmaxX2Flat::FlatAtomicUmaxX2Flat(const MachineInst *inst)
    : Flat("flat_atomic_umax_x2", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->data) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data);
  flags_ |= MEMORY_OP;
}

void FlatAtomicUmaxX2Flat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::UMAX;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 8);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 0, lane);
    std::memcpy(&d->store_data[lane * 8 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 1, lane);
    std::memcpy(&d->store_data[lane * 8 + 4], &val1, 4);
  }
  set_data(std::move(d));
}

FlatAtomicAndX2Flat::FlatAtomicAndX2Flat(const MachineInst *inst)
    : Flat("flat_atomic_and_x2", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->data) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data);
  flags_ |= MEMORY_OP;
}

void FlatAtomicAndX2Flat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::AND;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 8);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 0, lane);
    std::memcpy(&d->store_data[lane * 8 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 1, lane);
    std::memcpy(&d->store_data[lane * 8 + 4], &val1, 4);
  }
  set_data(std::move(d));
}

FlatAtomicOrX2Flat::FlatAtomicOrX2Flat(const MachineInst *inst)
    : Flat("flat_atomic_or_x2", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->data) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data);
  flags_ |= MEMORY_OP;
}

void FlatAtomicOrX2Flat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::OR;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 8);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 0, lane);
    std::memcpy(&d->store_data[lane * 8 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 1, lane);
    std::memcpy(&d->store_data[lane * 8 + 4], &val1, 4);
  }
  set_data(std::move(d));
}

FlatAtomicXorX2Flat::FlatAtomicXorX2Flat(const MachineInst *inst)
    : Flat("flat_atomic_xor_x2", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->data) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data);
  flags_ |= MEMORY_OP;
}

void FlatAtomicXorX2Flat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::XOR;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 8);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 0, lane);
    std::memcpy(&d->store_data[lane * 8 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 1, lane);
    std::memcpy(&d->store_data[lane * 8 + 4], &val1, 4);
  }
  set_data(std::move(d));
}

FlatAtomicIncX2Flat::FlatAtomicIncX2Flat(const MachineInst *inst)
    : Flat("flat_atomic_inc_x2", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->data) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data);
  flags_ |= MEMORY_OP;
}

void FlatAtomicIncX2Flat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::INC;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 8);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 0, lane);
    std::memcpy(&d->store_data[lane * 8 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 1, lane);
    std::memcpy(&d->store_data[lane * 8 + 4], &val1, 4);
  }
  set_data(std::move(d));
}

FlatAtomicDecX2Flat::FlatAtomicDecX2Flat(const MachineInst *inst)
    : Flat("flat_atomic_dec_x2", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->data) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data);
  flags_ |= MEMORY_OP;
}

void FlatAtomicDecX2Flat::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::DEC;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  flat_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 8);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 0, lane);
    std::memcpy(&d->store_data[lane * 8 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data + 1, lane);
    std::memcpy(&d->store_data[lane * 8 + 4], &val1, 4);
  }
  set_data(std::move(d));
}

} // namespace cdna4
} // namespace rocjitsu

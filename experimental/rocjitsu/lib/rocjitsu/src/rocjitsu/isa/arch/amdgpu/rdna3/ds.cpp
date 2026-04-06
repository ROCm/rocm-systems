// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/rdna3/ds.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/shared/execute_shared.h"
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
namespace rdna3 {

DsAddU32Ds::DsAddU32Ds(const MachineInst *inst)
    : Ds("ds_add_u32", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsAddU32Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = true;
  d->atomic_op = amdgpu::AtomicOp::ADD;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

DsSubU32Ds::DsSubU32Ds(const MachineInst *inst)
    : Ds("ds_sub_u32", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsSubU32Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = true;
  d->atomic_op = amdgpu::AtomicOp::SUB;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

DsRsubU32Ds::DsRsubU32Ds(const MachineInst *inst)
    : Ds("ds_rsub_u32", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsRsubU32Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = true;
  d->atomic_op = amdgpu::AtomicOp::SUB;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

DsIncU32Ds::DsIncU32Ds(const MachineInst *inst)
    : Ds("ds_inc_u32", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsIncU32Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = true;
  d->atomic_op = amdgpu::AtomicOp::INC;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

DsDecU32Ds::DsDecU32Ds(const MachineInst *inst)
    : Ds("ds_dec_u32", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsDecU32Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = true;
  d->atomic_op = amdgpu::AtomicOp::DEC;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

DsMinI32Ds::DsMinI32Ds(const MachineInst *inst)
    : Ds("ds_min_i32", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsMinI32Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = true;
  d->atomic_op = amdgpu::AtomicOp::SMIN;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

DsMaxI32Ds::DsMaxI32Ds(const MachineInst *inst)
    : Ds("ds_max_i32", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsMaxI32Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = true;
  d->atomic_op = amdgpu::AtomicOp::SMAX;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

DsMinU32Ds::DsMinU32Ds(const MachineInst *inst)
    : Ds("ds_min_u32", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsMinU32Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = true;
  d->atomic_op = amdgpu::AtomicOp::UMIN;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

DsMaxU32Ds::DsMaxU32Ds(const MachineInst *inst)
    : Ds("ds_max_u32", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsMaxU32Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = true;
  d->atomic_op = amdgpu::AtomicOp::UMAX;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

DsAndB32Ds::DsAndB32Ds(const MachineInst *inst)
    : Ds("ds_and_b32", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsAndB32Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = true;
  d->atomic_op = amdgpu::AtomicOp::AND;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

DsOrB32Ds::DsOrB32Ds(const MachineInst *inst)
    : Ds("ds_or_b32", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsOrB32Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = true;
  d->atomic_op = amdgpu::AtomicOp::OR;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

DsXorB32Ds::DsXorB32Ds(const MachineInst *inst)
    : Ds("ds_xor_b32", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsXorB32Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = true;
  d->atomic_op = amdgpu::AtomicOp::XOR;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

DsMskorB32Ds::DsMskorB32Ds(const MachineInst *inst)
    : Ds("ds_mskor_b32", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data1) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
}

void DsMskorB32Ds::execute(amdgpu::Wavefront &wf) { (void)wf; }

DsStoreB32Ds::DsStoreB32Ds(const MachineInst *inst)
    : Ds("ds_store_b32", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsStoreB32Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = false;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

DsStore2addrB32Ds::DsStore2addrB32Ds(const MachineInst *inst)
    : Ds("ds_store_2addr_b32", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data1) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
  flags_ |= MEMORY_OP;
}

void DsStore2addrB32Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = false;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

DsStore2addrStride64B32Ds::DsStore2addrStride64B32Ds(const MachineInst *inst)
    : Ds("ds_store_2addr_stride64_b32", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data1) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
  flags_ |= MEMORY_OP;
}

void DsStore2addrStride64B32Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = false;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

DsCmpstoreB32Ds::DsCmpstoreB32Ds(const MachineInst *inst)
    : Ds("ds_cmpstore_b32", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data1) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
  flags_ |= MEMORY_OP;
}

void DsCmpstoreB32Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = true;
  d->atomic_op = amdgpu::AtomicOp::CMPSWAP;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 8);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    std::memcpy(&d->store_data[lane * 8 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 1, lane);
    std::memcpy(&d->store_data[lane * 8 + 4], &val1, 4);
  }
  set_data(std::move(d));
}

DsCmpstoreF32Ds::DsCmpstoreF32Ds(const MachineInst *inst)
    : Ds("ds_cmpstore_f32", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data1) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
  flags_ |= MEMORY_OP;
}

void DsCmpstoreF32Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant (DS_CMPSTORE_F32)
}

DsMinF32Ds::DsMinF32Ds(const MachineInst *inst)
    : Ds("ds_min_f32", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsMinF32Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = true;
  d->atomic_op = amdgpu::AtomicOp::FMIN;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

DsMaxF32Ds::DsMaxF32Ds(const MachineInst *inst)
    : Ds("ds_max_f32", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsMaxF32Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = true;
  d->atomic_op = amdgpu::AtomicOp::FMAX;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

DsNopDs::DsNopDs(const MachineInst *inst)
    : Ds("ds_nop", reinterpret_cast<const OpEncoding *>(inst)) {}

void DsNopDs::execute(amdgpu::Wavefront &wf) { (void)wf; }

DsAddF32Ds::DsAddF32Ds(const MachineInst *inst)
    : Ds("ds_add_f32", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsAddF32Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = true;
  d->atomic_op = amdgpu::AtomicOp::FADD;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

DsGwsSemaReleaseAllDs::DsGwsSemaReleaseAllDs(const MachineInst *inst)
    : Ds("ds_gws_sema_release_all", reinterpret_cast<const OpEncoding *>(inst)) {}

void DsGwsSemaReleaseAllDs::execute(amdgpu::Wavefront &wf) { (void)wf; }

DsGwsInitDs::DsGwsInitDs(const MachineInst *inst)
    : Ds("ds_gws_init", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  src_operands_.emplace_back(&addr);
}

void DsGwsInitDs::execute(amdgpu::Wavefront &wf) { (void)wf; }

DsGwsSemaVDs::DsGwsSemaVDs(const MachineInst *inst)
    : Ds("ds_gws_sema_v", reinterpret_cast<const OpEncoding *>(inst)) {}

void DsGwsSemaVDs::execute(amdgpu::Wavefront &wf) { (void)wf; }

DsGwsSemaBrDs::DsGwsSemaBrDs(const MachineInst *inst)
    : Ds("ds_gws_sema_br", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  src_operands_.emplace_back(&addr);
}

void DsGwsSemaBrDs::execute(amdgpu::Wavefront &wf) { (void)wf; }

DsGwsSemaPDs::DsGwsSemaPDs(const MachineInst *inst)
    : Ds("ds_gws_sema_p", reinterpret_cast<const OpEncoding *>(inst)) {}

void DsGwsSemaPDs::execute(amdgpu::Wavefront &wf) { (void)wf; }

DsGwsBarrierDs::DsGwsBarrierDs(const MachineInst *inst)
    : Ds("ds_gws_barrier", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  src_operands_.emplace_back(&addr);
}

void DsGwsBarrierDs::execute(amdgpu::Wavefront &wf) { (void)wf; }

DsStoreB8Ds::DsStoreB8Ds(const MachineInst *inst)
    : Ds("ds_store_b8", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(8, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsStoreB8Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = false;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 1);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0, lane);
    d->store_data[lane * 1 + 0] = static_cast<uint8_t>(val0);
  }
  set_data(std::move(d));
}

DsStoreB16Ds::DsStoreB16Ds(const MachineInst *inst)
    : Ds("ds_store_b16", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsStoreB16Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = false;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 2);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0, lane);
    std::memcpy(&d->store_data[lane * 2 + 0], &val0, 2);
  }
  set_data(std::move(d));
}

DsAddRtnU32Ds::DsAddRtnU32Ds(const MachineInst *inst)
    : Ds("ds_add_rtn_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsAddRtnU32Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant (DS_ADD_RTN_U32)
}

DsSubRtnU32Ds::DsSubRtnU32Ds(const MachineInst *inst)
    : Ds("ds_sub_rtn_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsSubRtnU32Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant (DS_SUB_RTN_U32)
}

DsRsubRtnU32Ds::DsRsubRtnU32Ds(const MachineInst *inst)
    : Ds("ds_rsub_rtn_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsRsubRtnU32Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant (DS_RSUB_RTN_U32)
}

DsIncRtnU32Ds::DsIncRtnU32Ds(const MachineInst *inst)
    : Ds("ds_inc_rtn_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsIncRtnU32Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant (DS_INC_RTN_U32)
}

DsDecRtnU32Ds::DsDecRtnU32Ds(const MachineInst *inst)
    : Ds("ds_dec_rtn_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsDecRtnU32Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant (DS_DEC_RTN_U32)
}

DsMinRtnI32Ds::DsMinRtnI32Ds(const MachineInst *inst)
    : Ds("ds_min_rtn_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsMinRtnI32Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant (DS_MIN_RTN_I32)
}

DsMaxRtnI32Ds::DsMaxRtnI32Ds(const MachineInst *inst)
    : Ds("ds_max_rtn_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsMaxRtnI32Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant (DS_MAX_RTN_I32)
}

DsMinRtnU32Ds::DsMinRtnU32Ds(const MachineInst *inst)
    : Ds("ds_min_rtn_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsMinRtnU32Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant (DS_MIN_RTN_U32)
}

DsMaxRtnU32Ds::DsMaxRtnU32Ds(const MachineInst *inst)
    : Ds("ds_max_rtn_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsMaxRtnU32Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant (DS_MAX_RTN_U32)
}

DsAndRtnB32Ds::DsAndRtnB32Ds(const MachineInst *inst)
    : Ds("ds_and_rtn_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsAndRtnB32Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant (DS_AND_RTN_B32)
}

DsOrRtnB32Ds::DsOrRtnB32Ds(const MachineInst *inst)
    : Ds("ds_or_rtn_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsOrRtnB32Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant (DS_OR_RTN_B32)
}

DsXorRtnB32Ds::DsXorRtnB32Ds(const MachineInst *inst)
    : Ds("ds_xor_rtn_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsXorRtnB32Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant (DS_XOR_RTN_B32)
}

DsMskorRtnB32Ds::DsMskorRtnB32Ds(const MachineInst *inst)
    : Ds("ds_mskor_rtn_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
}

void DsMskorRtnB32Ds::execute(amdgpu::Wavefront &wf) { (void)wf; }

DsStorexchgRtnB32Ds::DsStorexchgRtnB32Ds(const MachineInst *inst)
    : Ds("ds_storexchg_rtn_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsStorexchgRtnB32Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = true;
  d->atomic_op = amdgpu::AtomicOp::SWAP;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

DsStorexchg2addrRtnB32Ds::DsStorexchg2addrRtnB32Ds(const MachineInst *inst)
    : Ds("ds_storexchg_2addr_rtn_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
  flags_ |= MEMORY_OP;
}

void DsStorexchg2addrRtnB32Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled ds_atomic variant (DS_STOREXCHG_2ADDR_RTN_B32)
}

DsStorexchg2addrStride64RtnB32Ds::DsStorexchg2addrStride64RtnB32Ds(const MachineInst *inst)
    : Ds("ds_storexchg_2addr_stride64_rtn_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
  flags_ |= MEMORY_OP;
}

void DsStorexchg2addrStride64RtnB32Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled ds_atomic variant (DS_STOREXCHG_2ADDR_STRIDE64_RTN_B32)
}

DsCmpstoreRtnB32Ds::DsCmpstoreRtnB32Ds(const MachineInst *inst)
    : Ds("ds_cmpstore_rtn_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
  flags_ |= MEMORY_OP;
}

void DsCmpstoreRtnB32Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = true;
  d->atomic_op = amdgpu::AtomicOp::CMPSWAP;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 8);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    std::memcpy(&d->store_data[lane * 8 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 1, lane);
    std::memcpy(&d->store_data[lane * 8 + 4], &val1, 4);
  }
  set_data(std::move(d));
}

DsCmpstoreRtnF32Ds::DsCmpstoreRtnF32Ds(const MachineInst *inst)
    : Ds("ds_cmpstore_rtn_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
  flags_ |= MEMORY_OP;
}

void DsCmpstoreRtnF32Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled ds_atomic variant (DS_CMPSTORE_RTN_F32)
}

DsMinRtnF32Ds::DsMinRtnF32Ds(const MachineInst *inst)
    : Ds("ds_min_rtn_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsMinRtnF32Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant (DS_MIN_RTN_F32)
}

DsMaxRtnF32Ds::DsMaxRtnF32Ds(const MachineInst *inst)
    : Ds("ds_max_rtn_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsMaxRtnF32Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant (DS_MAX_RTN_F32)
}

DsWrapRtnB32Ds::DsWrapRtnB32Ds(const MachineInst *inst)
    : Ds("ds_wrap_rtn_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
}

void DsWrapRtnB32Ds::execute(amdgpu::Wavefront &wf) { (void)wf; }

DsSwizzleB32Ds::DsSwizzleB32Ds(const MachineInst *inst)
    : Ds("ds_swizzle_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
}

void DsSwizzleB32Ds::execute(amdgpu::Wavefront &wf) {
  amdgpu::execute_ds_swizzle_b32_ds(*this, wf);
}

DsLoadB32Ds::DsLoadB32Ds(const MachineInst *inst)
    : Ds("ds_load_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void DsLoadB32Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = true;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

DsLoad2addrB32Ds::DsLoad2addrB32Ds(const MachineInst *inst)
    : Ds("ds_load_2addr_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void DsLoad2addrB32Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = true;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

DsLoad2addrStride64B32Ds::DsLoad2addrStride64B32Ds(const MachineInst *inst)
    : Ds("ds_load_2addr_stride64_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void DsLoad2addrStride64B32Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = true;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

DsLoadI8Ds::DsLoadI8Ds(const MachineInst *inst)
    : Ds("ds_load_i8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void DsLoadI8Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->sign_extend = true;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

DsLoadU8Ds::DsLoadU8Ds(const MachineInst *inst)
    : Ds("ds_load_u8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void DsLoadU8Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

DsLoadI16Ds::DsLoadI16Ds(const MachineInst *inst)
    : Ds("ds_load_i16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void DsLoadI16Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = true;
  d->sign_extend = true;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

DsLoadU16Ds::DsLoadU16Ds(const MachineInst *inst)
    : Ds("ds_load_u16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void DsLoadU16Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = true;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

DsConsumeDs::DsConsumeDs(const MachineInst *inst)
    : Ds("ds_consume", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst) {
  dst_operands_.emplace_back(&vdst);
}

void DsConsumeDs::execute(amdgpu::Wavefront &wf) { (void)wf; }

DsAppendDs::DsAppendDs(const MachineInst *inst)
    : Ds("ds_append", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst) {
  dst_operands_.emplace_back(&vdst);
}

void DsAppendDs::execute(amdgpu::Wavefront &wf) { (void)wf; }

DsOrderedCountDs::DsOrderedCountDs(const MachineInst *inst)
    : Ds("ds_ordered_count", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
}

void DsOrderedCountDs::execute(amdgpu::Wavefront &wf) { (void)wf; }

DsAddU64Ds::DsAddU64Ds(const MachineInst *inst)
    : Ds("ds_add_u64", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsAddU64Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = true;
  d->atomic_op = amdgpu::AtomicOp::ADD;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 8);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    std::memcpy(&d->store_data[lane * 8 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 1, lane);
    std::memcpy(&d->store_data[lane * 8 + 4], &val1, 4);
  }
  set_data(std::move(d));
}

DsSubU64Ds::DsSubU64Ds(const MachineInst *inst)
    : Ds("ds_sub_u64", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsSubU64Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = true;
  d->atomic_op = amdgpu::AtomicOp::SUB;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 8);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    std::memcpy(&d->store_data[lane * 8 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 1, lane);
    std::memcpy(&d->store_data[lane * 8 + 4], &val1, 4);
  }
  set_data(std::move(d));
}

DsRsubU64Ds::DsRsubU64Ds(const MachineInst *inst)
    : Ds("ds_rsub_u64", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsRsubU64Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = true;
  d->atomic_op = amdgpu::AtomicOp::SUB;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 8);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    std::memcpy(&d->store_data[lane * 8 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 1, lane);
    std::memcpy(&d->store_data[lane * 8 + 4], &val1, 4);
  }
  set_data(std::move(d));
}

DsIncU64Ds::DsIncU64Ds(const MachineInst *inst)
    : Ds("ds_inc_u64", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsIncU64Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = true;
  d->atomic_op = amdgpu::AtomicOp::INC;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 8);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    std::memcpy(&d->store_data[lane * 8 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 1, lane);
    std::memcpy(&d->store_data[lane * 8 + 4], &val1, 4);
  }
  set_data(std::move(d));
}

DsDecU64Ds::DsDecU64Ds(const MachineInst *inst)
    : Ds("ds_dec_u64", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsDecU64Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = true;
  d->atomic_op = amdgpu::AtomicOp::DEC;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 8);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    std::memcpy(&d->store_data[lane * 8 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 1, lane);
    std::memcpy(&d->store_data[lane * 8 + 4], &val1, 4);
  }
  set_data(std::move(d));
}

DsMinI64Ds::DsMinI64Ds(const MachineInst *inst)
    : Ds("ds_min_i64", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsMinI64Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = true;
  d->atomic_op = amdgpu::AtomicOp::SMIN;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 8);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    std::memcpy(&d->store_data[lane * 8 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 1, lane);
    std::memcpy(&d->store_data[lane * 8 + 4], &val1, 4);
  }
  set_data(std::move(d));
}

DsMaxI64Ds::DsMaxI64Ds(const MachineInst *inst)
    : Ds("ds_max_i64", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsMaxI64Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = true;
  d->atomic_op = amdgpu::AtomicOp::SMAX;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 8);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    std::memcpy(&d->store_data[lane * 8 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 1, lane);
    std::memcpy(&d->store_data[lane * 8 + 4], &val1, 4);
  }
  set_data(std::move(d));
}

DsMinU64Ds::DsMinU64Ds(const MachineInst *inst)
    : Ds("ds_min_u64", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsMinU64Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = true;
  d->atomic_op = amdgpu::AtomicOp::UMIN;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 8);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    std::memcpy(&d->store_data[lane * 8 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 1, lane);
    std::memcpy(&d->store_data[lane * 8 + 4], &val1, 4);
  }
  set_data(std::move(d));
}

DsMaxU64Ds::DsMaxU64Ds(const MachineInst *inst)
    : Ds("ds_max_u64", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsMaxU64Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = true;
  d->atomic_op = amdgpu::AtomicOp::UMAX;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 8);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    std::memcpy(&d->store_data[lane * 8 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 1, lane);
    std::memcpy(&d->store_data[lane * 8 + 4], &val1, 4);
  }
  set_data(std::move(d));
}

DsAndB64Ds::DsAndB64Ds(const MachineInst *inst)
    : Ds("ds_and_b64", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsAndB64Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = true;
  d->atomic_op = amdgpu::AtomicOp::AND;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 8);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    std::memcpy(&d->store_data[lane * 8 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 1, lane);
    std::memcpy(&d->store_data[lane * 8 + 4], &val1, 4);
  }
  set_data(std::move(d));
}

DsOrB64Ds::DsOrB64Ds(const MachineInst *inst)
    : Ds("ds_or_b64", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsOrB64Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = true;
  d->atomic_op = amdgpu::AtomicOp::OR;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 8);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    std::memcpy(&d->store_data[lane * 8 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 1, lane);
    std::memcpy(&d->store_data[lane * 8 + 4], &val1, 4);
  }
  set_data(std::move(d));
}

DsXorB64Ds::DsXorB64Ds(const MachineInst *inst)
    : Ds("ds_xor_b64", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsXorB64Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = true;
  d->atomic_op = amdgpu::AtomicOp::XOR;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 8);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    std::memcpy(&d->store_data[lane * 8 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 1, lane);
    std::memcpy(&d->store_data[lane * 8 + 4], &val1, 4);
  }
  set_data(std::move(d));
}

DsMskorB64Ds::DsMskorB64Ds(const MachineInst *inst)
    : Ds("ds_mskor_b64", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data1) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
}

void DsMskorB64Ds::execute(amdgpu::Wavefront &wf) { (void)wf; }

DsStoreB64Ds::DsStoreB64Ds(const MachineInst *inst)
    : Ds("ds_store_b64", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsStoreB64Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = false;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 8);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t lo0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    uint32_t hi0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 1, lane);
    std::memcpy(&d->store_data[lane * 8 + 0], &lo0, 4);
    std::memcpy(&d->store_data[lane * 8 + 4], &hi0, 4);
  }
  set_data(std::move(d));
}

DsStore2addrB64Ds::DsStore2addrB64Ds(const MachineInst *inst)
    : Ds("ds_store_2addr_b64", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data1) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
  flags_ |= MEMORY_OP;
}

void DsStore2addrB64Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = false;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 8);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t lo0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    uint32_t hi0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 1, lane);
    std::memcpy(&d->store_data[lane * 8 + 0], &lo0, 4);
    std::memcpy(&d->store_data[lane * 8 + 4], &hi0, 4);
  }
  set_data(std::move(d));
}

DsStore2addrStride64B64Ds::DsStore2addrStride64B64Ds(const MachineInst *inst)
    : Ds("ds_store_2addr_stride64_b64", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data1) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
  flags_ |= MEMORY_OP;
}

void DsStore2addrStride64B64Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = false;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 8);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t lo0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    uint32_t hi0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 1, lane);
    std::memcpy(&d->store_data[lane * 8 + 0], &lo0, 4);
    std::memcpy(&d->store_data[lane * 8 + 4], &hi0, 4);
  }
  set_data(std::move(d));
}

DsCmpstoreB64Ds::DsCmpstoreB64Ds(const MachineInst *inst)
    : Ds("ds_cmpstore_b64", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data1) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
  flags_ |= MEMORY_OP;
}

void DsCmpstoreB64Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = true;
  d->atomic_op = amdgpu::AtomicOp::CMPSWAP;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 16);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    std::memcpy(&d->store_data[lane * 16 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 1, lane);
    std::memcpy(&d->store_data[lane * 16 + 4], &val1, 4);
    uint32_t val2 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 2, lane);
    std::memcpy(&d->store_data[lane * 16 + 8], &val2, 4);
    uint32_t val3 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 3, lane);
    std::memcpy(&d->store_data[lane * 16 + 12], &val3, 4);
  }
  set_data(std::move(d));
}

DsCmpstoreF64Ds::DsCmpstoreF64Ds(const MachineInst *inst)
    : Ds("ds_cmpstore_f64", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data1) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
  flags_ |= MEMORY_OP;
}

void DsCmpstoreF64Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant (DS_CMPSTORE_F64)
}

DsMinF64Ds::DsMinF64Ds(const MachineInst *inst)
    : Ds("ds_min_f64", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsMinF64Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = true;
  d->atomic_op = amdgpu::AtomicOp::FMIN;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 8);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    std::memcpy(&d->store_data[lane * 8 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 1, lane);
    std::memcpy(&d->store_data[lane * 8 + 4], &val1, 4);
  }
  set_data(std::move(d));
}

DsMaxF64Ds::DsMaxF64Ds(const MachineInst *inst)
    : Ds("ds_max_f64", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsMaxF64Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = true;
  d->atomic_op = amdgpu::AtomicOp::FMAX;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 8);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    std::memcpy(&d->store_data[lane * 8 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 1, lane);
    std::memcpy(&d->store_data[lane * 8 + 4], &val1, 4);
  }
  set_data(std::move(d));
}

DsAddRtnU64Ds::DsAddRtnU64Ds(const MachineInst *inst)
    : Ds("ds_add_rtn_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsAddRtnU64Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant (DS_ADD_RTN_U64)
}

DsSubRtnU64Ds::DsSubRtnU64Ds(const MachineInst *inst)
    : Ds("ds_sub_rtn_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsSubRtnU64Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant (DS_SUB_RTN_U64)
}

DsRsubRtnU64Ds::DsRsubRtnU64Ds(const MachineInst *inst)
    : Ds("ds_rsub_rtn_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsRsubRtnU64Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant (DS_RSUB_RTN_U64)
}

DsIncRtnU64Ds::DsIncRtnU64Ds(const MachineInst *inst)
    : Ds("ds_inc_rtn_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsIncRtnU64Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant (DS_INC_RTN_U64)
}

DsDecRtnU64Ds::DsDecRtnU64Ds(const MachineInst *inst)
    : Ds("ds_dec_rtn_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsDecRtnU64Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant (DS_DEC_RTN_U64)
}

DsMinRtnI64Ds::DsMinRtnI64Ds(const MachineInst *inst)
    : Ds("ds_min_rtn_i64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsMinRtnI64Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant (DS_MIN_RTN_I64)
}

DsMaxRtnI64Ds::DsMaxRtnI64Ds(const MachineInst *inst)
    : Ds("ds_max_rtn_i64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsMaxRtnI64Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant (DS_MAX_RTN_I64)
}

DsMinRtnU64Ds::DsMinRtnU64Ds(const MachineInst *inst)
    : Ds("ds_min_rtn_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsMinRtnU64Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant (DS_MIN_RTN_U64)
}

DsMaxRtnU64Ds::DsMaxRtnU64Ds(const MachineInst *inst)
    : Ds("ds_max_rtn_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsMaxRtnU64Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant (DS_MAX_RTN_U64)
}

DsAndRtnB64Ds::DsAndRtnB64Ds(const MachineInst *inst)
    : Ds("ds_and_rtn_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsAndRtnB64Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant (DS_AND_RTN_B64)
}

DsOrRtnB64Ds::DsOrRtnB64Ds(const MachineInst *inst)
    : Ds("ds_or_rtn_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsOrRtnB64Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant (DS_OR_RTN_B64)
}

DsXorRtnB64Ds::DsXorRtnB64Ds(const MachineInst *inst)
    : Ds("ds_xor_rtn_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsXorRtnB64Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant (DS_XOR_RTN_B64)
}

DsMskorRtnB64Ds::DsMskorRtnB64Ds(const MachineInst *inst)
    : Ds("ds_mskor_rtn_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
}

void DsMskorRtnB64Ds::execute(amdgpu::Wavefront &wf) { (void)wf; }

DsStorexchgRtnB64Ds::DsStorexchgRtnB64Ds(const MachineInst *inst)
    : Ds("ds_storexchg_rtn_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsStorexchgRtnB64Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = true;
  d->atomic_op = amdgpu::AtomicOp::SWAP;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 8);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    std::memcpy(&d->store_data[lane * 8 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 1, lane);
    std::memcpy(&d->store_data[lane * 8 + 4], &val1, 4);
  }
  set_data(std::move(d));
}

DsStorexchg2addrRtnB64Ds::DsStorexchg2addrRtnB64Ds(const MachineInst *inst)
    : Ds("ds_storexchg_2addr_rtn_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
  flags_ |= MEMORY_OP;
}

void DsStorexchg2addrRtnB64Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled ds_atomic variant (DS_STOREXCHG_2ADDR_RTN_B64)
}

DsStorexchg2addrStride64RtnB64Ds::DsStorexchg2addrStride64RtnB64Ds(const MachineInst *inst)
    : Ds("ds_storexchg_2addr_stride64_rtn_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
  flags_ |= MEMORY_OP;
}

void DsStorexchg2addrStride64RtnB64Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled ds_atomic variant (DS_STOREXCHG_2ADDR_STRIDE64_RTN_B64)
}

DsCmpstoreRtnB64Ds::DsCmpstoreRtnB64Ds(const MachineInst *inst)
    : Ds("ds_cmpstore_rtn_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
  flags_ |= MEMORY_OP;
}

void DsCmpstoreRtnB64Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = true;
  d->atomic_op = amdgpu::AtomicOp::CMPSWAP;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 16);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    std::memcpy(&d->store_data[lane * 16 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 1, lane);
    std::memcpy(&d->store_data[lane * 16 + 4], &val1, 4);
    uint32_t val2 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 2, lane);
    std::memcpy(&d->store_data[lane * 16 + 8], &val2, 4);
    uint32_t val3 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 3, lane);
    std::memcpy(&d->store_data[lane * 16 + 12], &val3, 4);
  }
  set_data(std::move(d));
}

DsCmpstoreRtnF64Ds::DsCmpstoreRtnF64Ds(const MachineInst *inst)
    : Ds("ds_cmpstore_rtn_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
  flags_ |= MEMORY_OP;
}

void DsCmpstoreRtnF64Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled ds_atomic variant (DS_CMPSTORE_RTN_F64)
}

DsMinRtnF64Ds::DsMinRtnF64Ds(const MachineInst *inst)
    : Ds("ds_min_rtn_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsMinRtnF64Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant (DS_MIN_RTN_F64)
}

DsMaxRtnF64Ds::DsMaxRtnF64Ds(const MachineInst *inst)
    : Ds("ds_max_rtn_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsMaxRtnF64Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant (DS_MAX_RTN_F64)
}

DsLoadB64Ds::DsLoadB64Ds(const MachineInst *inst)
    : Ds("ds_load_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void DsLoadB64Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = true;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

DsLoad2addrB64Ds::DsLoad2addrB64Ds(const MachineInst *inst)
    : Ds("ds_load_2addr_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void DsLoad2addrB64Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = true;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

DsLoad2addrStride64B64Ds::DsLoad2addrStride64B64Ds(const MachineInst *inst)
    : Ds("ds_load_2addr_stride64_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void DsLoad2addrStride64B64Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = true;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

DsAddRtnF32Ds::DsAddRtnF32Ds(const MachineInst *inst)
    : Ds("ds_add_rtn_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsAddRtnF32Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant (DS_ADD_RTN_F32)
}

DsAddGsRegRtnDs::DsAddGsRegRtnDs(const MachineInst *inst)
    : Ds("ds_add_gs_reg_rtn", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&data0);
}

void DsAddGsRegRtnDs::execute(amdgpu::Wavefront &wf) { (void)wf; }

DsSubGsRegRtnDs::DsSubGsRegRtnDs(const MachineInst *inst)
    : Ds("ds_sub_gs_reg_rtn", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&data0);
}

void DsSubGsRegRtnDs::execute(amdgpu::Wavefront &wf) { (void)wf; }

DsCondxchg32RtnB64Ds::DsCondxchg32RtnB64Ds(const MachineInst *inst)
    : Ds("ds_condxchg32_rtn_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsCondxchg32RtnB64Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled ds_atomic variant (DS_CONDXCHG32_RTN_B64)
}

DsStoreB8D16HiDs::DsStoreB8D16HiDs(const MachineInst *inst)
    : Ds("ds_store_b8_d16_hi", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsStoreB8D16HiDs::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = false;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 1);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0, lane);
    d->store_data[lane * 1 + 0] = static_cast<uint8_t>(val0);
  }
  set_data(std::move(d));
}

DsStoreB16D16HiDs::DsStoreB16D16HiDs(const MachineInst *inst)
    : Ds("ds_store_b16_d16_hi", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsStoreB16D16HiDs::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = false;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 2);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0, lane);
    std::memcpy(&d->store_data[lane * 2 + 0], &val0, 2);
  }
  set_data(std::move(d));
}

DsLoadU8D16Ds::DsLoadU8D16Ds(const MachineInst *inst)
    : Ds("ds_load_u8_d16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void DsLoadU8D16Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

DsLoadU8D16HiDs::DsLoadU8D16HiDs(const MachineInst *inst)
    : Ds("ds_load_u8_d16_hi", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void DsLoadU8D16HiDs::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

DsLoadI8D16Ds::DsLoadI8D16Ds(const MachineInst *inst)
    : Ds("ds_load_i8_d16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void DsLoadI8D16Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->sign_extend = true;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

DsLoadI8D16HiDs::DsLoadI8D16HiDs(const MachineInst *inst)
    : Ds("ds_load_i8_d16_hi", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void DsLoadI8D16HiDs::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->sign_extend = true;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

DsLoadU16D16Ds::DsLoadU16D16Ds(const MachineInst *inst)
    : Ds("ds_load_u16_d16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void DsLoadU16D16Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = true;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

DsLoadU16D16HiDs::DsLoadU16D16HiDs(const MachineInst *inst)
    : Ds("ds_load_u16_d16_hi", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void DsLoadU16D16HiDs::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = true;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

DsBvhStackRtnB32Ds::DsBvhStackRtnB32Ds(const MachineInst *inst)
    : Ds("ds_bvh_stack_rtn_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  dst_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
}

void DsBvhStackRtnB32Ds::execute(amdgpu::Wavefront &wf) { (void)wf; }

DsStoreAddtidB32Ds::DsStoreAddtidB32Ds(const MachineInst *inst)
    : Ds("ds_store_addtid_b32", reinterpret_cast<const OpEncoding *>(inst)),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsStoreAddtidB32Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = false;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

DsLoadAddtidB32Ds::DsLoadAddtidB32Ds(const MachineInst *inst)
    : Ds("ds_load_addtid_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst) {
  dst_operands_.emplace_back(&vdst);
  flags_ |= MEMORY_OP;
}

void DsLoadAddtidB32Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = true;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

DsPermuteB32Ds::DsPermuteB32Ds(const MachineInst *inst)
    : Ds("ds_permute_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
}

void DsPermuteB32Ds::execute(amdgpu::Wavefront &wf) {
  amdgpu::execute_ds_permute_b32_ds(*this, wf);
}

DsBpermuteB32Ds::DsBpermuteB32Ds(const MachineInst *inst)
    : Ds("ds_bpermute_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
}

void DsBpermuteB32Ds::execute(amdgpu::Wavefront &wf) {
  amdgpu::execute_ds_bpermute_b32_ds(*this, wf);
}

DsStoreB96Ds::DsStoreB96Ds(const MachineInst *inst)
    : Ds("ds_store_b96", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(96, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsStoreB96Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->elem_size = 4;
  d->num_elems = 3;
  d->is_load = false;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 12);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    std::memcpy(&d->store_data[lane * 12 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 1, lane);
    std::memcpy(&d->store_data[lane * 12 + 4], &val1, 4);
    uint32_t val2 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 2, lane);
    std::memcpy(&d->store_data[lane * 12 + 8], &val2, 4);
  }
  set_data(std::move(d));
}

DsStoreB128Ds::DsStoreB128Ds(const MachineInst *inst)
    : Ds("ds_store_b128", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsStoreB128Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->elem_size = 4;
  d->num_elems = 4;
  d->is_load = false;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 16);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 0, lane);
    std::memcpy(&d->store_data[lane * 16 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 1, lane);
    std::memcpy(&d->store_data[lane * 16 + 4], &val1, 4);
    uint32_t val2 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 2, lane);
    std::memcpy(&d->store_data[lane * 16 + 8], &val2, 4);
    uint32_t val3 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.data0 + 3, lane);
    std::memcpy(&d->store_data[lane * 16 + 12], &val3, 4);
  }
  set_data(std::move(d));
}

DsLoadB96Ds::DsLoadB96Ds(const MachineInst *inst)
    : Ds("ds_load_b96", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(96, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void DsLoadB96Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 3;
  d->is_load = true;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

DsLoadB128Ds::DsLoadB128Ds(const MachineInst *inst)
    : Ds("ds_load_b128", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void DsLoadB128Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 4;
  d->is_load = true;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

} // namespace rdna3
} // namespace rocjitsu

// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/cdna3/ds.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/addr_calc.h"
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
namespace cdna3 {

DsAddU32Ds::DsAddU32Ds(const MachineInst *inst)
    : Ds("ds_add_u32", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data1) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
}

void DsMskorB32Ds::execute(amdgpu::Wavefront &wf) { (void)wf; }

DsWriteB32Ds::DsWriteB32Ds(const MachineInst *inst)
    : Ds("ds_write_b32", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsWriteB32Ds::execute(amdgpu::Wavefront &wf) {
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

DsWrite2B32Ds::DsWrite2B32Ds(const MachineInst *inst)
    : Ds("ds_write2_b32", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data1) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
  flags_ |= MEMORY_OP;
}

void DsWrite2B32Ds::execute(amdgpu::Wavefront &wf) {
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

DsWrite2st64B32Ds::DsWrite2st64B32Ds(const MachineInst *inst)
    : Ds("ds_write2st64_b32", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data1) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
}

void DsWrite2st64B32Ds::execute(amdgpu::Wavefront &wf) { (void)wf; }

DsCmpstB32Ds::DsCmpstB32Ds(const MachineInst *inst)
    : Ds("ds_cmpst_b32", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data1) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
  flags_ |= MEMORY_OP;
}

void DsCmpstB32Ds::execute(amdgpu::Wavefront &wf) {
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

DsCmpstF32Ds::DsCmpstF32Ds(const MachineInst *inst)
    : Ds("ds_cmpst_f32", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data1) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
  flags_ |= MEMORY_OP;
}

void DsCmpstF32Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant (DS_CMPST_F32)
}

DsMinF32Ds::DsMinF32Ds(const MachineInst *inst)
    : Ds("ds_min_f32", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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

DsPkAddF16Ds::DsPkAddF16Ds(const MachineInst *inst)
    : Ds("ds_pk_add_f16", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsPkAddF16Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant (DS_PK_ADD_F16)
}

DsPkAddBf16Ds::DsPkAddBf16Ds(const MachineInst *inst)
    : Ds("ds_pk_add_bf16", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsPkAddBf16Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant (DS_PK_ADD_BF16)
}

DsWriteAddtidB32Ds::DsWriteAddtidB32Ds(const MachineInst *inst)
    : Ds("ds_write_addtid_b32", reinterpret_cast<const OpEncoding *>(inst)),
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsWriteAddtidB32Ds::execute(amdgpu::Wavefront &wf) {
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

DsWriteB8Ds::DsWriteB8Ds(const MachineInst *inst)
    : Ds("ds_write_b8", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsWriteB8Ds::execute(amdgpu::Wavefront &wf) {
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

DsWriteB16Ds::DsWriteB16Ds(const MachineInst *inst)
    : Ds("ds_write_b16", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsWriteB16Ds::execute(amdgpu::Wavefront &wf) {
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
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
}

void DsMskorRtnB32Ds::execute(amdgpu::Wavefront &wf) { (void)wf; }

DsWrxchgRtnB32Ds::DsWrxchgRtnB32Ds(const MachineInst *inst)
    : Ds("ds_wrxchg_rtn_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsWrxchgRtnB32Ds::execute(amdgpu::Wavefront &wf) {
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

DsWrxchg2RtnB32Ds::DsWrxchg2RtnB32Ds(const MachineInst *inst)
    : Ds("ds_wrxchg2_rtn_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
  flags_ |= MEMORY_OP;
}

void DsWrxchg2RtnB32Ds::execute(amdgpu::Wavefront &wf) {
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

DsWrxchg2st64RtnB32Ds::DsWrxchg2st64RtnB32Ds(const MachineInst *inst)
    : Ds("ds_wrxchg2st64_rtn_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
  flags_ |= MEMORY_OP;
}

void DsWrxchg2st64RtnB32Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled ds_atomic variant (DS_WRXCHG2ST64_RTN_B32)
}

DsCmpstRtnB32Ds::DsCmpstRtnB32Ds(const MachineInst *inst)
    : Ds("ds_cmpst_rtn_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
  flags_ |= MEMORY_OP;
}

void DsCmpstRtnB32Ds::execute(amdgpu::Wavefront &wf) {
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

DsCmpstRtnF32Ds::DsCmpstRtnF32Ds(const MachineInst *inst)
    : Ds("ds_cmpst_rtn_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
  flags_ |= MEMORY_OP;
}

void DsCmpstRtnF32Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant (DS_CMPST_RTN_F32)
}

DsMinRtnF32Ds::DsMinRtnF32Ds(const MachineInst *inst)
    : Ds("ds_min_rtn_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
}

void DsWrapRtnB32Ds::execute(amdgpu::Wavefront &wf) { (void)wf; }

DsAddRtnF32Ds::DsAddRtnF32Ds(const MachineInst *inst)
    : Ds("ds_add_rtn_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsAddRtnF32Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant (DS_ADD_RTN_F32)
}

DsReadB32Ds::DsReadB32Ds(const MachineInst *inst)
    : Ds("ds_read_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void DsReadB32Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = true;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

DsRead2B32Ds::DsRead2B32Ds(const MachineInst *inst)
    : Ds("ds_read2_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void DsRead2B32Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = true;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

DsRead2st64B32Ds::DsRead2st64B32Ds(const MachineInst *inst)
    : Ds("ds_read2st64_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
}

void DsRead2st64B32Ds::execute(amdgpu::Wavefront &wf) { (void)wf; }

DsReadI8Ds::DsReadI8Ds(const MachineInst *inst)
    : Ds("ds_read_i8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void DsReadI8Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->sign_extend = true;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

DsReadU8Ds::DsReadU8Ds(const MachineInst *inst)
    : Ds("ds_read_u8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void DsReadU8Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

DsReadI16Ds::DsReadI16Ds(const MachineInst *inst)
    : Ds("ds_read_i16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void DsReadI16Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = true;
  d->sign_extend = true;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

DsReadU16Ds::DsReadU16Ds(const MachineInst *inst)
    : Ds("ds_read_u16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void DsReadU16Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = true;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

DsSwizzleB32Ds::DsSwizzleB32Ds(const MachineInst *inst)
    : Ds("ds_swizzle_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
}

void DsSwizzleB32Ds::execute(amdgpu::Wavefront &wf) { (void)wf; }

DsPermuteB32Ds::DsPermuteB32Ds(const MachineInst *inst)
    : Ds("ds_permute_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
}

void DsPermuteB32Ds::execute(amdgpu::Wavefront &wf) { (void)wf; }

DsBpermuteB32Ds::DsBpermuteB32Ds(const MachineInst *inst)
    : Ds("ds_bpermute_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
}

void DsBpermuteB32Ds::execute(amdgpu::Wavefront &wf) { (void)wf; }

DsAddU64Ds::DsAddU64Ds(const MachineInst *inst)
    : Ds("ds_add_u64", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data1) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
}

void DsMskorB64Ds::execute(amdgpu::Wavefront &wf) { (void)wf; }

DsWriteB64Ds::DsWriteB64Ds(const MachineInst *inst)
    : Ds("ds_write_b64", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsWriteB64Ds::execute(amdgpu::Wavefront &wf) {
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

DsWrite2B64Ds::DsWrite2B64Ds(const MachineInst *inst)
    : Ds("ds_write2_b64", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data1) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
  flags_ |= MEMORY_OP;
}

void DsWrite2B64Ds::execute(amdgpu::Wavefront &wf) {
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

DsWrite2st64B64Ds::DsWrite2st64B64Ds(const MachineInst *inst)
    : Ds("ds_write2st64_b64", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data1) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
}

void DsWrite2st64B64Ds::execute(amdgpu::Wavefront &wf) { (void)wf; }

DsCmpstB64Ds::DsCmpstB64Ds(const MachineInst *inst)
    : Ds("ds_cmpst_b64", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data1) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
  flags_ |= MEMORY_OP;
}

void DsCmpstB64Ds::execute(amdgpu::Wavefront &wf) {
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

DsCmpstF64Ds::DsCmpstF64Ds(const MachineInst *inst)
    : Ds("ds_cmpst_f64", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data1) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
  flags_ |= MEMORY_OP;
}

void DsCmpstF64Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant (DS_CMPST_F64)
}

DsMinF64Ds::DsMinF64Ds(const MachineInst *inst)
    : Ds("ds_min_f64", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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

DsWriteB8D16HiDs::DsWriteB8D16HiDs(const MachineInst *inst)
    : Ds("ds_write_b8_d16_hi", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsWriteB8D16HiDs::execute(amdgpu::Wavefront &wf) {
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

DsWriteB16D16HiDs::DsWriteB16D16HiDs(const MachineInst *inst)
    : Ds("ds_write_b16_d16_hi", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsWriteB16D16HiDs::execute(amdgpu::Wavefront &wf) {
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

DsReadU8D16Ds::DsReadU8D16Ds(const MachineInst *inst)
    : Ds("ds_read_u8_d16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void DsReadU8D16Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

DsReadU8D16HiDs::DsReadU8D16HiDs(const MachineInst *inst)
    : Ds("ds_read_u8_d16_hi", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void DsReadU8D16HiDs::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

DsReadI8D16Ds::DsReadI8D16Ds(const MachineInst *inst)
    : Ds("ds_read_i8_d16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void DsReadI8D16Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->sign_extend = true;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

DsReadI8D16HiDs::DsReadI8D16HiDs(const MachineInst *inst)
    : Ds("ds_read_i8_d16_hi", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void DsReadI8D16HiDs::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->sign_extend = true;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

DsReadU16D16Ds::DsReadU16D16Ds(const MachineInst *inst)
    : Ds("ds_read_u16_d16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void DsReadU16D16Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = true;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

DsReadU16D16HiDs::DsReadU16D16HiDs(const MachineInst *inst)
    : Ds("ds_read_u16_d16_hi", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void DsReadU16D16HiDs::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = true;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

DsAddF64Ds::DsAddF64Ds(const MachineInst *inst)
    : Ds("ds_add_f64", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsAddF64Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant (DS_ADD_F64)
}

DsAddRtnU64Ds::DsAddRtnU64Ds(const MachineInst *inst)
    : Ds("ds_add_rtn_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
}

void DsMskorRtnB64Ds::execute(amdgpu::Wavefront &wf) { (void)wf; }

DsWrxchgRtnB64Ds::DsWrxchgRtnB64Ds(const MachineInst *inst)
    : Ds("ds_wrxchg_rtn_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsWrxchgRtnB64Ds::execute(amdgpu::Wavefront &wf) {
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

DsWrxchg2RtnB64Ds::DsWrxchg2RtnB64Ds(const MachineInst *inst)
    : Ds("ds_wrxchg2_rtn_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
  flags_ |= MEMORY_OP;
}

void DsWrxchg2RtnB64Ds::execute(amdgpu::Wavefront &wf) {
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

DsWrxchg2st64RtnB64Ds::DsWrxchg2st64RtnB64Ds(const MachineInst *inst)
    : Ds("ds_wrxchg2st64_rtn_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
  flags_ |= MEMORY_OP;
}

void DsWrxchg2st64RtnB64Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled ds_atomic variant (DS_WRXCHG2ST64_RTN_B64)
}

DsCmpstRtnB64Ds::DsCmpstRtnB64Ds(const MachineInst *inst)
    : Ds("ds_cmpst_rtn_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
  flags_ |= MEMORY_OP;
}

void DsCmpstRtnB64Ds::execute(amdgpu::Wavefront &wf) {
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

DsCmpstRtnF64Ds::DsCmpstRtnF64Ds(const MachineInst *inst)
    : Ds("ds_cmpst_rtn_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0),
      data1(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  src_operands_.emplace_back(&data1);
  flags_ |= MEMORY_OP;
}

void DsCmpstRtnF64Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant (DS_CMPST_RTN_F64)
}

DsMinRtnF64Ds::DsMinRtnF64Ds(const MachineInst *inst)
    : Ds("ds_min_rtn_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsMaxRtnF64Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant (DS_MAX_RTN_F64)
}

DsReadB64Ds::DsReadB64Ds(const MachineInst *inst)
    : Ds("ds_read_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void DsReadB64Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = true;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

DsRead2B64Ds::DsRead2B64Ds(const MachineInst *inst)
    : Ds("ds_read2_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void DsRead2B64Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = true;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

DsRead2st64B64Ds::DsRead2st64B64Ds(const MachineInst *inst)
    : Ds("ds_read2st64_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
}

void DsRead2st64B64Ds::execute(amdgpu::Wavefront &wf) { (void)wf; }

DsAddRtnF64Ds::DsAddRtnF64Ds(const MachineInst *inst)
    : Ds("ds_add_rtn_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsAddRtnF64Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // TODO: unhandled ds_atomic variant (DS_ADD_RTN_F64)
}

DsCondxchg32RtnB64Ds::DsCondxchg32RtnB64Ds(const MachineInst *inst)
    : Ds("ds_condxchg32_rtn_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
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

DsReadAddtidB32Ds::DsReadAddtidB32Ds(const MachineInst *inst)
    : Ds("ds_read_addtid_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst) {
  dst_operands_.emplace_back(&vdst);
  flags_ |= MEMORY_OP;
}

void DsReadAddtidB32Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = true;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

DsPkAddRtnF16Ds::DsPkAddRtnF16Ds(const MachineInst *inst)
    : Ds("ds_pk_add_rtn_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsPkAddRtnF16Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled ds_atomic variant (DS_PK_ADD_RTN_F16)
}

DsPkAddRtnBf16Ds::DsPkAddRtnBf16Ds(const MachineInst *inst)
    : Ds("ds_pk_add_rtn_bf16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsPkAddRtnBf16Ds::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled ds_atomic variant (DS_PK_ADD_RTN_BF16)
}

DsConsumeDs::DsConsumeDs(const MachineInst *inst)
    : Ds("ds_consume", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst) {
  dst_operands_.emplace_back(&vdst);
}

void DsConsumeDs::execute(amdgpu::Wavefront &wf) { (void)wf; }

DsAppendDs::DsAppendDs(const MachineInst *inst)
    : Ds("ds_append", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst) {
  dst_operands_.emplace_back(&vdst);
}

void DsAppendDs::execute(amdgpu::Wavefront &wf) { (void)wf; }

DsWriteB96Ds::DsWriteB96Ds(const MachineInst *inst)
    : Ds("ds_write_b96", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(96, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsWriteB96Ds::execute(amdgpu::Wavefront &wf) {
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

DsWriteB128Ds::DsWriteB128Ds(const MachineInst *inst)
    : Ds("ds_write_b128", reinterpret_cast<const OpEncoding *>(inst)),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr),
      data0(128, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->data0) {
  src_operands_.emplace_back(&addr);
  src_operands_.emplace_back(&data0);
  flags_ |= MEMORY_OP;
}

void DsWriteB128Ds::execute(amdgpu::Wavefront &wf) {
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

DsReadB96Ds::DsReadB96Ds(const MachineInst *inst)
    : Ds("ds_read_b96", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(96, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void DsReadB96Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 3;
  d->is_load = true;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

DsReadB128Ds::DsReadB128Ds(const MachineInst *inst)
    : Ds("ds_read_b128", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      addr(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->addr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&addr);
  flags_ |= MEMORY_OP;
}

void DsReadB128Ds::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::LOCAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdst;
  d->elem_size = 4;
  d->num_elems = 4;
  d->is_load = true;
  ds_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

} // namespace cdna3
} // namespace rocjitsu

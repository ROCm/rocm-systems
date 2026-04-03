// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/rdna3/mubuf.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/addr_calc.h"
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

BufferLoadFormatXMubuf::BufferLoadFormatXMubuf(const MachineInst *inst)
    : Mubuf("buffer_load_format_x", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferLoadFormatXMubuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferLoadFormatXyMubuf::BufferLoadFormatXyMubuf(const MachineInst *inst)
    : Mubuf("buffer_load_format_xy", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferLoadFormatXyMubuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferLoadFormatXyzMubuf::BufferLoadFormatXyzMubuf(const MachineInst *inst)
    : Mubuf("buffer_load_format_xyz", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(96, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferLoadFormatXyzMubuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferLoadFormatXyzwMubuf::BufferLoadFormatXyzwMubuf(const MachineInst *inst)
    : Mubuf("buffer_load_format_xyzw", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferLoadFormatXyzwMubuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferStoreFormatXMubuf::BufferStoreFormatXMubuf(const MachineInst *inst)
    : Mubuf("buffer_store_format_x", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferStoreFormatXMubuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferStoreFormatXyMubuf::BufferStoreFormatXyMubuf(const MachineInst *inst)
    : Mubuf("buffer_store_format_xy", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferStoreFormatXyMubuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferStoreFormatXyzMubuf::BufferStoreFormatXyzMubuf(const MachineInst *inst)
    : Mubuf("buffer_store_format_xyz", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(96, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferStoreFormatXyzMubuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferStoreFormatXyzwMubuf::BufferStoreFormatXyzwMubuf(const MachineInst *inst)
    : Mubuf("buffer_store_format_xyzw", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferStoreFormatXyzwMubuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferLoadD16FormatXMubuf::BufferLoadD16FormatXMubuf(const MachineInst *inst)
    : Mubuf("buffer_load_d16_format_x", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferLoadD16FormatXMubuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferLoadD16FormatXyMubuf::BufferLoadD16FormatXyMubuf(const MachineInst *inst)
    : Mubuf("buffer_load_d16_format_xy", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferLoadD16FormatXyMubuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferLoadD16FormatXyzMubuf::BufferLoadD16FormatXyzMubuf(const MachineInst *inst)
    : Mubuf("buffer_load_d16_format_xyz", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferLoadD16FormatXyzMubuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferLoadD16FormatXyzwMubuf::BufferLoadD16FormatXyzwMubuf(const MachineInst *inst)
    : Mubuf("buffer_load_d16_format_xyzw", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferLoadD16FormatXyzwMubuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferStoreD16FormatXMubuf::BufferStoreD16FormatXMubuf(const MachineInst *inst)
    : Mubuf("buffer_store_d16_format_x", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferStoreD16FormatXMubuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferStoreD16FormatXyMubuf::BufferStoreD16FormatXyMubuf(const MachineInst *inst)
    : Mubuf("buffer_store_d16_format_xy", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferStoreD16FormatXyMubuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferStoreD16FormatXyzMubuf::BufferStoreD16FormatXyzMubuf(const MachineInst *inst)
    : Mubuf("buffer_store_d16_format_xyz", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferStoreD16FormatXyzMubuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferStoreD16FormatXyzwMubuf::BufferStoreD16FormatXyzwMubuf(const MachineInst *inst)
    : Mubuf("buffer_store_d16_format_xyzw", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferStoreD16FormatXyzwMubuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferLoadU8Mubuf::BufferLoadU8Mubuf(const MachineInst *inst)
    : Mubuf("buffer_load_u8", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadU8Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = mtype_from_bits(inst_.glc, inst_.slc);
  d->non_temporal = 0;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferLoadI8Mubuf::BufferLoadI8Mubuf(const MachineInst *inst)
    : Mubuf("buffer_load_i8", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadI8Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->sign_extend = true;
  d->mtype = mtype_from_bits(inst_.glc, inst_.slc);
  d->non_temporal = 0;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferLoadU16Mubuf::BufferLoadU16Mubuf(const MachineInst *inst)
    : Mubuf("buffer_load_u16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadU16Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = mtype_from_bits(inst_.glc, inst_.slc);
  d->non_temporal = 0;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferLoadI16Mubuf::BufferLoadI16Mubuf(const MachineInst *inst)
    : Mubuf("buffer_load_i16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadI16Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = true;
  d->sign_extend = true;
  d->mtype = mtype_from_bits(inst_.glc, inst_.slc);
  d->non_temporal = 0;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferLoadB32Mubuf::BufferLoadB32Mubuf(const MachineInst *inst)
    : Mubuf("buffer_load_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadB32Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = mtype_from_bits(inst_.glc, inst_.slc);
  d->non_temporal = 0;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferLoadB64Mubuf::BufferLoadB64Mubuf(const MachineInst *inst)
    : Mubuf("buffer_load_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadB64Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 4;
  d->num_elems = 2;
  d->is_load = true;
  d->mtype = mtype_from_bits(inst_.glc, inst_.slc);
  d->non_temporal = 0;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferLoadB96Mubuf::BufferLoadB96Mubuf(const MachineInst *inst)
    : Mubuf("buffer_load_b96", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(96, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadB96Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 4;
  d->num_elems = 3;
  d->is_load = true;
  d->mtype = mtype_from_bits(inst_.glc, inst_.slc);
  d->non_temporal = 0;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferLoadB128Mubuf::BufferLoadB128Mubuf(const MachineInst *inst)
    : Mubuf("buffer_load_b128", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadB128Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 4;
  d->num_elems = 4;
  d->is_load = true;
  d->mtype = mtype_from_bits(inst_.glc, inst_.slc);
  d->non_temporal = 0;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferStoreB8Mubuf::BufferStoreB8Mubuf(const MachineInst *inst)
    : Mubuf("buffer_store_b8", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferStoreB8Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = false;
  d->mtype = mtype_from_bits(inst_.glc, inst_.slc);
  d->non_temporal = 0;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 1);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.vdata, lane);
    d->store_data[lane * 1 + 0] = static_cast<uint8_t>(val0);
  }
  set_data(std::move(d));
}

BufferStoreB16Mubuf::BufferStoreB16Mubuf(const MachineInst *inst)
    : Mubuf("buffer_store_b16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferStoreB16Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = false;
  d->mtype = mtype_from_bits(inst_.glc, inst_.slc);
  d->non_temporal = 0;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 2);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.vdata, lane);
    std::memcpy(&d->store_data[lane * 2 + 0], &val0, 2);
  }
  set_data(std::move(d));
}

BufferStoreB32Mubuf::BufferStoreB32Mubuf(const MachineInst *inst)
    : Mubuf("buffer_store_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferStoreB32Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = false;
  d->mtype = mtype_from_bits(inst_.glc, inst_.slc);
  d->non_temporal = 0;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.vdata + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

BufferStoreB64Mubuf::BufferStoreB64Mubuf(const MachineInst *inst)
    : Mubuf("buffer_store_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferStoreB64Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 4;
  d->num_elems = 2;
  d->is_load = false;
  d->mtype = mtype_from_bits(inst_.glc, inst_.slc);
  d->non_temporal = 0;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 8);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.vdata + 0, lane);
    std::memcpy(&d->store_data[lane * 8 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.vdata + 1, lane);
    std::memcpy(&d->store_data[lane * 8 + 4], &val1, 4);
  }
  set_data(std::move(d));
}

BufferStoreB96Mubuf::BufferStoreB96Mubuf(const MachineInst *inst)
    : Mubuf("buffer_store_b96", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(96, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferStoreB96Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 4;
  d->num_elems = 3;
  d->is_load = false;
  d->mtype = mtype_from_bits(inst_.glc, inst_.slc);
  d->non_temporal = 0;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 12);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.vdata + 0, lane);
    std::memcpy(&d->store_data[lane * 12 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.vdata + 1, lane);
    std::memcpy(&d->store_data[lane * 12 + 4], &val1, 4);
    uint32_t val2 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.vdata + 2, lane);
    std::memcpy(&d->store_data[lane * 12 + 8], &val2, 4);
  }
  set_data(std::move(d));
}

BufferStoreB128Mubuf::BufferStoreB128Mubuf(const MachineInst *inst)
    : Mubuf("buffer_store_b128", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferStoreB128Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 4;
  d->num_elems = 4;
  d->is_load = false;
  d->mtype = mtype_from_bits(inst_.glc, inst_.slc);
  d->non_temporal = 0;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 16);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.vdata + 0, lane);
    std::memcpy(&d->store_data[lane * 16 + 0], &val0, 4);
    uint32_t val1 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.vdata + 1, lane);
    std::memcpy(&d->store_data[lane * 16 + 4], &val1, 4);
    uint32_t val2 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.vdata + 2, lane);
    std::memcpy(&d->store_data[lane * 16 + 8], &val2, 4);
    uint32_t val3 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.vdata + 3, lane);
    std::memcpy(&d->store_data[lane * 16 + 12], &val3, 4);
  }
  set_data(std::move(d));
}

BufferLoadD16U8Mubuf::BufferLoadD16U8Mubuf(const MachineInst *inst)
    : Mubuf("buffer_load_d16_u8", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadD16U8Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = mtype_from_bits(inst_.glc, inst_.slc);
  d->non_temporal = 0;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferLoadD16I8Mubuf::BufferLoadD16I8Mubuf(const MachineInst *inst)
    : Mubuf("buffer_load_d16_i8", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadD16I8Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->sign_extend = true;
  d->mtype = mtype_from_bits(inst_.glc, inst_.slc);
  d->non_temporal = 0;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferLoadD16B16Mubuf::BufferLoadD16B16Mubuf(const MachineInst *inst)
    : Mubuf("buffer_load_d16_b16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadD16B16Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = mtype_from_bits(inst_.glc, inst_.slc);
  d->non_temporal = 0;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferLoadD16HiU8Mubuf::BufferLoadD16HiU8Mubuf(const MachineInst *inst)
    : Mubuf("buffer_load_d16_hi_u8", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadD16HiU8Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = mtype_from_bits(inst_.glc, inst_.slc);
  d->non_temporal = 0;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferLoadD16HiI8Mubuf::BufferLoadD16HiI8Mubuf(const MachineInst *inst)
    : Mubuf("buffer_load_d16_hi_i8", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadD16HiI8Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->sign_extend = true;
  d->mtype = mtype_from_bits(inst_.glc, inst_.slc);
  d->non_temporal = 0;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferLoadD16HiB16Mubuf::BufferLoadD16HiB16Mubuf(const MachineInst *inst)
    : Mubuf("buffer_load_d16_hi_b16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadD16HiB16Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = mtype_from_bits(inst_.glc, inst_.slc);
  d->non_temporal = 0;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferStoreD16HiB8Mubuf::BufferStoreD16HiB8Mubuf(const MachineInst *inst)
    : Mubuf("buffer_store_d16_hi_b8", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferStoreD16HiB8Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = false;
  d->mtype = mtype_from_bits(inst_.glc, inst_.slc);
  d->non_temporal = 0;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 1);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.vdata, lane);
    d->store_data[lane * 1 + 0] = static_cast<uint8_t>(val0);
  }
  set_data(std::move(d));
}

BufferStoreD16HiB16Mubuf::BufferStoreD16HiB16Mubuf(const MachineInst *inst)
    : Mubuf("buffer_store_d16_hi_b16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferStoreD16HiB16Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = false;
  d->mtype = mtype_from_bits(inst_.glc, inst_.slc);
  d->non_temporal = 0;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 2);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.vdata, lane);
    std::memcpy(&d->store_data[lane * 2 + 0], &val0, 2);
  }
  set_data(std::move(d));
}

BufferLoadD16HiFormatXMubuf::BufferLoadD16HiFormatXMubuf(const MachineInst *inst)
    : Mubuf("buffer_load_d16_hi_format_x", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferLoadD16HiFormatXMubuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferStoreD16HiFormatXMubuf::BufferStoreD16HiFormatXMubuf(const MachineInst *inst)
    : Mubuf("buffer_store_d16_hi_format_x", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferStoreD16HiFormatXMubuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferGl0InvMubuf::BufferGl0InvMubuf(const MachineInst *inst)
    : Mubuf("buffer_gl0_inv", reinterpret_cast<const OpEncoding *>(inst)) {}

void BufferGl0InvMubuf::execute(amdgpu::Wavefront &wf) { wf.cu().l1_scalar().invalidate_all(); }

BufferGl1InvMubuf::BufferGl1InvMubuf(const MachineInst *inst)
    : Mubuf("buffer_gl1_inv", reinterpret_cast<const OpEncoding *>(inst)) {}

void BufferGl1InvMubuf::execute(amdgpu::Wavefront &wf) { wf.cu().l1_scalar().invalidate_all(); }

BufferAtomicSwapB32Mubuf::BufferAtomicSwapB32Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_swap_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicSwapB32Mubuf::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_SWAP_B32)
}

BufferAtomicCmpswapB32Mubuf::BufferAtomicCmpswapB32Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_cmpswap_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicCmpswapB32Mubuf::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_CMPSWAP_B32)
}

BufferAtomicAddU32Mubuf::BufferAtomicAddU32Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_add_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicAddU32Mubuf::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_ADD_U32)
}

BufferAtomicSubU32Mubuf::BufferAtomicSubU32Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_sub_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicSubU32Mubuf::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_SUB_U32)
}

BufferAtomicCsubU32Mubuf::BufferAtomicCsubU32Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_csub_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicCsubU32Mubuf::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_CSUB_U32)
}

BufferAtomicMinI32Mubuf::BufferAtomicMinI32Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_min_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicMinI32Mubuf::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_MIN_I32)
}

BufferAtomicMinU32Mubuf::BufferAtomicMinU32Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_min_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicMinU32Mubuf::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_MIN_U32)
}

BufferAtomicMaxI32Mubuf::BufferAtomicMaxI32Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_max_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicMaxI32Mubuf::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_MAX_I32)
}

BufferAtomicMaxU32Mubuf::BufferAtomicMaxU32Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_max_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicMaxU32Mubuf::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_MAX_U32)
}

BufferAtomicAndB32Mubuf::BufferAtomicAndB32Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_and_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicAndB32Mubuf::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_AND_B32)
}

BufferAtomicOrB32Mubuf::BufferAtomicOrB32Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_or_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicOrB32Mubuf::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_OR_B32)
}

BufferAtomicXorB32Mubuf::BufferAtomicXorB32Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_xor_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicXorB32Mubuf::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_XOR_B32)
}

BufferAtomicIncU32Mubuf::BufferAtomicIncU32Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_inc_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicIncU32Mubuf::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_INC_U32)
}

BufferAtomicDecU32Mubuf::BufferAtomicDecU32Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_dec_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicDecU32Mubuf::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_DEC_U32)
}

BufferAtomicSwapB64Mubuf::BufferAtomicSwapB64Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_swap_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicSwapB64Mubuf::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_SWAP_B64)
}

BufferAtomicCmpswapB64Mubuf::BufferAtomicCmpswapB64Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_cmpswap_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicCmpswapB64Mubuf::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_CMPSWAP_B64)
}

BufferAtomicAddU64Mubuf::BufferAtomicAddU64Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_add_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicAddU64Mubuf::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_ADD_U64)
}

BufferAtomicSubU64Mubuf::BufferAtomicSubU64Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_sub_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicSubU64Mubuf::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_SUB_U64)
}

BufferAtomicMinI64Mubuf::BufferAtomicMinI64Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_min_i64", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicMinI64Mubuf::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_MIN_I64)
}

BufferAtomicMinU64Mubuf::BufferAtomicMinU64Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_min_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicMinU64Mubuf::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_MIN_U64)
}

BufferAtomicMaxI64Mubuf::BufferAtomicMaxI64Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_max_i64", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicMaxI64Mubuf::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_MAX_I64)
}

BufferAtomicMaxU64Mubuf::BufferAtomicMaxU64Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_max_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicMaxU64Mubuf::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_MAX_U64)
}

BufferAtomicAndB64Mubuf::BufferAtomicAndB64Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_and_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicAndB64Mubuf::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_AND_B64)
}

BufferAtomicOrB64Mubuf::BufferAtomicOrB64Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_or_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicOrB64Mubuf::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_OR_B64)
}

BufferAtomicXorB64Mubuf::BufferAtomicXorB64Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_xor_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicXorB64Mubuf::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_XOR_B64)
}

BufferAtomicIncU64Mubuf::BufferAtomicIncU64Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_inc_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicIncU64Mubuf::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_INC_U64)
}

BufferAtomicDecU64Mubuf::BufferAtomicDecU64Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_dec_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicDecU64Mubuf::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_DEC_U64)
}

BufferAtomicCmpswapF32Mubuf::BufferAtomicCmpswapF32Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_cmpswap_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicCmpswapF32Mubuf::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_CMPSWAP_F32)
}

BufferAtomicMinF32Mubuf::BufferAtomicMinF32Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_min_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicMinF32Mubuf::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_MIN_F32)
}

BufferAtomicMaxF32Mubuf::BufferAtomicMaxF32Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_max_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicMaxF32Mubuf::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_MAX_F32)
}

BufferAtomicAddF32Mubuf::BufferAtomicAddF32Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_add_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SREG_M0_INL,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicAddF32Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.glc != 0);
  d->atomic_op = amdgpu::AtomicOp::FADD;
  d->mtype = mtype_from_bits(inst_.glc, inst_.slc);
  d->non_temporal = 0;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  auto &cu = wf.cu();
  uint64_t exec = wf.exec();
  d->store_data.resize(wf.wf_size() * 4);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val0 = cu.read_vgpr(wf.vgpr_alloc().base + inst_.vdata + 0, lane);
    std::memcpy(&d->store_data[lane * 4 + 0], &val0, 4);
  }
  set_data(std::move(d));
}

} // namespace rdna3
} // namespace rocjitsu

// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/rdna4/vbuffer.h"
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

BufferLoadFormatXVbuffer::BufferLoadFormatXVbuffer(const MachineInst *inst)
    : Vbuffer("buffer_load_format_x", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferLoadFormatXVbuffer::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferLoadFormatXyVbuffer::BufferLoadFormatXyVbuffer(const MachineInst *inst)
    : Vbuffer("buffer_load_format_xy", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferLoadFormatXyVbuffer::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferLoadFormatXyzVbuffer::BufferLoadFormatXyzVbuffer(const MachineInst *inst)
    : Vbuffer("buffer_load_format_xyz", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(96, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferLoadFormatXyzVbuffer::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferLoadFormatXyzwVbuffer::BufferLoadFormatXyzwVbuffer(const MachineInst *inst)
    : Vbuffer("buffer_load_format_xyzw", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferLoadFormatXyzwVbuffer::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferStoreFormatXVbuffer::BufferStoreFormatXVbuffer(const MachineInst *inst)
    : Vbuffer("buffer_store_format_x", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferStoreFormatXVbuffer::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferStoreFormatXyVbuffer::BufferStoreFormatXyVbuffer(const MachineInst *inst)
    : Vbuffer("buffer_store_format_xy", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferStoreFormatXyVbuffer::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferStoreFormatXyzVbuffer::BufferStoreFormatXyzVbuffer(const MachineInst *inst)
    : Vbuffer("buffer_store_format_xyz", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(96, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferStoreFormatXyzVbuffer::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferStoreFormatXyzwVbuffer::BufferStoreFormatXyzwVbuffer(const MachineInst *inst)
    : Vbuffer("buffer_store_format_xyzw", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferStoreFormatXyzwVbuffer::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferLoadD16FormatXVbuffer::BufferLoadD16FormatXVbuffer(const MachineInst *inst)
    : Vbuffer("buffer_load_d16_format_x", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferLoadD16FormatXVbuffer::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferLoadD16FormatXyVbuffer::BufferLoadD16FormatXyVbuffer(const MachineInst *inst)
    : Vbuffer("buffer_load_d16_format_xy", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferLoadD16FormatXyVbuffer::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferLoadD16FormatXyzVbuffer::BufferLoadD16FormatXyzVbuffer(const MachineInst *inst)
    : Vbuffer("buffer_load_d16_format_xyz", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferLoadD16FormatXyzVbuffer::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferLoadD16FormatXyzwVbuffer::BufferLoadD16FormatXyzwVbuffer(const MachineInst *inst)
    : Vbuffer("buffer_load_d16_format_xyzw", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferLoadD16FormatXyzwVbuffer::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferStoreD16FormatXVbuffer::BufferStoreD16FormatXVbuffer(const MachineInst *inst)
    : Vbuffer("buffer_store_d16_format_x", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferStoreD16FormatXVbuffer::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferStoreD16FormatXyVbuffer::BufferStoreD16FormatXyVbuffer(const MachineInst *inst)
    : Vbuffer("buffer_store_d16_format_xy", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferStoreD16FormatXyVbuffer::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferStoreD16FormatXyzVbuffer::BufferStoreD16FormatXyzVbuffer(const MachineInst *inst)
    : Vbuffer("buffer_store_d16_format_xyz", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferStoreD16FormatXyzVbuffer::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferStoreD16FormatXyzwVbuffer::BufferStoreD16FormatXyzwVbuffer(const MachineInst *inst)
    : Vbuffer("buffer_store_d16_format_xyzw", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferStoreD16FormatXyzwVbuffer::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferLoadU8Vbuffer::BufferLoadU8Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_load_u8", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadU8Vbuffer::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
  d->non_temporal = 0;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferLoadI8Vbuffer::BufferLoadI8Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_load_i8", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadI8Vbuffer::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->sign_extend = true;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
  d->non_temporal = 0;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferLoadU16Vbuffer::BufferLoadU16Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_load_u16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadU16Vbuffer::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
  d->non_temporal = 0;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferLoadI16Vbuffer::BufferLoadI16Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_load_i16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadI16Vbuffer::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = true;
  d->sign_extend = true;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
  d->non_temporal = 0;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferLoadB32Vbuffer::BufferLoadB32Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_load_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadB32Vbuffer::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
  d->non_temporal = 0;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferLoadB64Vbuffer::BufferLoadB64Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_load_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadB64Vbuffer::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 4;
  d->num_elems = 2;
  d->is_load = true;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
  d->non_temporal = 0;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferLoadB96Vbuffer::BufferLoadB96Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_load_b96", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(96, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadB96Vbuffer::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 4;
  d->num_elems = 3;
  d->is_load = true;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
  d->non_temporal = 0;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferLoadB128Vbuffer::BufferLoadB128Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_load_b128", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadB128Vbuffer::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 4;
  d->num_elems = 4;
  d->is_load = true;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
  d->non_temporal = 0;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferStoreB8Vbuffer::BufferStoreB8Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_store_b8", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferStoreB8Vbuffer::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = false;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
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

BufferStoreB16Vbuffer::BufferStoreB16Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_store_b16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferStoreB16Vbuffer::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = false;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
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

BufferStoreB32Vbuffer::BufferStoreB32Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_store_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferStoreB32Vbuffer::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = false;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
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

BufferStoreB64Vbuffer::BufferStoreB64Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_store_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferStoreB64Vbuffer::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 4;
  d->num_elems = 2;
  d->is_load = false;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
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

BufferStoreB96Vbuffer::BufferStoreB96Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_store_b96", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(96, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferStoreB96Vbuffer::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 4;
  d->num_elems = 3;
  d->is_load = false;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
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

BufferStoreB128Vbuffer::BufferStoreB128Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_store_b128", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferStoreB128Vbuffer::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 4;
  d->num_elems = 4;
  d->is_load = false;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
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

BufferLoadD16U8Vbuffer::BufferLoadD16U8Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_load_d16_u8", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadD16U8Vbuffer::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
  d->non_temporal = 0;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferLoadD16I8Vbuffer::BufferLoadD16I8Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_load_d16_i8", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadD16I8Vbuffer::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->sign_extend = true;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
  d->non_temporal = 0;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferLoadD16B16Vbuffer::BufferLoadD16B16Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_load_d16_b16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadD16B16Vbuffer::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
  d->non_temporal = 0;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferLoadD16HiU8Vbuffer::BufferLoadD16HiU8Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_load_d16_hi_u8", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadD16HiU8Vbuffer::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
  d->non_temporal = 0;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferLoadD16HiI8Vbuffer::BufferLoadD16HiI8Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_load_d16_hi_i8", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadD16HiI8Vbuffer::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->sign_extend = true;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
  d->non_temporal = 0;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferLoadD16HiB16Vbuffer::BufferLoadD16HiB16Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_load_d16_hi_b16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadD16HiB16Vbuffer::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
  d->non_temporal = 0;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferStoreD16HiB8Vbuffer::BufferStoreD16HiB8Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_store_d16_hi_b8", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferStoreD16HiB8Vbuffer::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = false;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
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

BufferStoreD16HiB16Vbuffer::BufferStoreD16HiB16Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_store_d16_hi_b16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferStoreD16HiB16Vbuffer::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = false;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
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

BufferLoadD16HiFormatXVbuffer::BufferLoadD16HiFormatXVbuffer(const MachineInst *inst)
    : Vbuffer("buffer_load_d16_hi_format_x", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferLoadD16HiFormatXVbuffer::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferStoreD16HiFormatXVbuffer::BufferStoreD16HiFormatXVbuffer(const MachineInst *inst)
    : Vbuffer("buffer_store_d16_hi_format_x", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferStoreD16HiFormatXVbuffer::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferAtomicSwapB32Vbuffer::BufferAtomicSwapB32Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_atomic_swap_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicSwapB32Vbuffer::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_SWAP_B32)
}

BufferAtomicCmpswapB32Vbuffer::BufferAtomicCmpswapB32Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_atomic_cmpswap_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicCmpswapB32Vbuffer::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_CMPSWAP_B32)
}

BufferAtomicAddU32Vbuffer::BufferAtomicAddU32Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_atomic_add_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicAddU32Vbuffer::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_ADD_U32)
}

BufferAtomicSubU32Vbuffer::BufferAtomicSubU32Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_atomic_sub_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicSubU32Vbuffer::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_SUB_U32)
}

BufferAtomicSubClampU32Vbuffer::BufferAtomicSubClampU32Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_atomic_sub_clamp_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicSubClampU32Vbuffer::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_SUB_CLAMP_U32)
}

BufferAtomicMinI32Vbuffer::BufferAtomicMinI32Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_atomic_min_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicMinI32Vbuffer::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_MIN_I32)
}

BufferAtomicMinU32Vbuffer::BufferAtomicMinU32Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_atomic_min_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicMinU32Vbuffer::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_MIN_U32)
}

BufferAtomicMaxI32Vbuffer::BufferAtomicMaxI32Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_atomic_max_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicMaxI32Vbuffer::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_MAX_I32)
}

BufferAtomicMaxU32Vbuffer::BufferAtomicMaxU32Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_atomic_max_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicMaxU32Vbuffer::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_MAX_U32)
}

BufferAtomicAndB32Vbuffer::BufferAtomicAndB32Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_atomic_and_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicAndB32Vbuffer::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_AND_B32)
}

BufferAtomicOrB32Vbuffer::BufferAtomicOrB32Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_atomic_or_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicOrB32Vbuffer::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_OR_B32)
}

BufferAtomicXorB32Vbuffer::BufferAtomicXorB32Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_atomic_xor_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicXorB32Vbuffer::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_XOR_B32)
}

BufferAtomicIncU32Vbuffer::BufferAtomicIncU32Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_atomic_inc_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicIncU32Vbuffer::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_INC_U32)
}

BufferAtomicDecU32Vbuffer::BufferAtomicDecU32Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_atomic_dec_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicDecU32Vbuffer::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_DEC_U32)
}

BufferAtomicSwapB64Vbuffer::BufferAtomicSwapB64Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_atomic_swap_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicSwapB64Vbuffer::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_SWAP_B64)
}

BufferAtomicCmpswapB64Vbuffer::BufferAtomicCmpswapB64Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_atomic_cmpswap_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicCmpswapB64Vbuffer::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_CMPSWAP_B64)
}

BufferAtomicAddU64Vbuffer::BufferAtomicAddU64Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_atomic_add_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicAddU64Vbuffer::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_ADD_U64)
}

BufferAtomicSubU64Vbuffer::BufferAtomicSubU64Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_atomic_sub_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicSubU64Vbuffer::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_SUB_U64)
}

BufferAtomicMinI64Vbuffer::BufferAtomicMinI64Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_atomic_min_i64", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicMinI64Vbuffer::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_MIN_I64)
}

BufferAtomicMinU64Vbuffer::BufferAtomicMinU64Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_atomic_min_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicMinU64Vbuffer::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_MIN_U64)
}

BufferAtomicMaxI64Vbuffer::BufferAtomicMaxI64Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_atomic_max_i64", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicMaxI64Vbuffer::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_MAX_I64)
}

BufferAtomicMaxU64Vbuffer::BufferAtomicMaxU64Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_atomic_max_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicMaxU64Vbuffer::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_MAX_U64)
}

BufferAtomicAndB64Vbuffer::BufferAtomicAndB64Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_atomic_and_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicAndB64Vbuffer::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_AND_B64)
}

BufferAtomicOrB64Vbuffer::BufferAtomicOrB64Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_atomic_or_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicOrB64Vbuffer::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_OR_B64)
}

BufferAtomicXorB64Vbuffer::BufferAtomicXorB64Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_atomic_xor_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicXorB64Vbuffer::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_XOR_B64)
}

BufferAtomicIncU64Vbuffer::BufferAtomicIncU64Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_atomic_inc_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicIncU64Vbuffer::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_INC_U64)
}

BufferAtomicDecU64Vbuffer::BufferAtomicDecU64Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_atomic_dec_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicDecU64Vbuffer::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_DEC_U64)
}

BufferAtomicCondSubU32Vbuffer::BufferAtomicCondSubU32Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_atomic_cond_sub_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicCondSubU32Vbuffer::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_COND_SUB_U32)
}

BufferAtomicMinNumF32Vbuffer::BufferAtomicMinNumF32Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_atomic_min_num_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicMinNumF32Vbuffer::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_MIN_NUM_F32)
}

BufferAtomicMaxNumF32Vbuffer::BufferAtomicMaxNumF32Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_atomic_max_num_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicMaxNumF32Vbuffer::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(
      mnemonic()); // TODO: unhandled buffer_atomic variant (BUFFER_ATOMIC_MAX_NUM_F32)
}

BufferAtomicAddF32Vbuffer::BufferAtomicAddF32Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_atomic_add_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicAddF32Vbuffer::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.nv != 0);
  d->atomic_op = amdgpu::AtomicOp::FADD;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
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

BufferAtomicPkAddF16Vbuffer::BufferAtomicPkAddF16Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_atomic_pk_add_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicPkAddF16Vbuffer::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.nv != 0);
  d->atomic_op = amdgpu::AtomicOp::FADD;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
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

BufferAtomicPkAddBf16Vbuffer::BufferAtomicPkAddBf16Vbuffer(const MachineInst *inst)
    : Vbuffer("buffer_atomic_pk_add_bf16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicPkAddBf16Vbuffer::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.nv != 0);
  d->atomic_op = amdgpu::AtomicOp::FADD;
  d->mtype = amdgpu::mtype_from_flags_gfx12(inst_.scope, inst_.th);
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

TbufferLoadFormatXVbuffer::TbufferLoadFormatXVbuffer(const MachineInst *inst)
    : Vbuffer("tbuffer_load_format_x", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
}

void TbufferLoadFormatXVbuffer::execute(amdgpu::Wavefront &wf) { (void)wf; }

TbufferLoadFormatXyVbuffer::TbufferLoadFormatXyVbuffer(const MachineInst *inst)
    : Vbuffer("tbuffer_load_format_xy", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
}

void TbufferLoadFormatXyVbuffer::execute(amdgpu::Wavefront &wf) { (void)wf; }

TbufferLoadFormatXyzVbuffer::TbufferLoadFormatXyzVbuffer(const MachineInst *inst)
    : Vbuffer("tbuffer_load_format_xyz", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(96, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
}

void TbufferLoadFormatXyzVbuffer::execute(amdgpu::Wavefront &wf) { (void)wf; }

TbufferLoadFormatXyzwVbuffer::TbufferLoadFormatXyzwVbuffer(const MachineInst *inst)
    : Vbuffer("tbuffer_load_format_xyzw", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
}

void TbufferLoadFormatXyzwVbuffer::execute(amdgpu::Wavefront &wf) { (void)wf; }

TbufferStoreFormatXVbuffer::TbufferStoreFormatXVbuffer(const MachineInst *inst)
    : Vbuffer("tbuffer_store_format_x", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
}

void TbufferStoreFormatXVbuffer::execute(amdgpu::Wavefront &wf) { (void)wf; }

TbufferStoreFormatXyVbuffer::TbufferStoreFormatXyVbuffer(const MachineInst *inst)
    : Vbuffer("tbuffer_store_format_xy", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
}

void TbufferStoreFormatXyVbuffer::execute(amdgpu::Wavefront &wf) { (void)wf; }

TbufferStoreFormatXyzVbuffer::TbufferStoreFormatXyzVbuffer(const MachineInst *inst)
    : Vbuffer("tbuffer_store_format_xyz", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(96, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
}

void TbufferStoreFormatXyzVbuffer::execute(amdgpu::Wavefront &wf) { (void)wf; }

TbufferStoreFormatXyzwVbuffer::TbufferStoreFormatXyzwVbuffer(const MachineInst *inst)
    : Vbuffer("tbuffer_store_format_xyzw", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
}

void TbufferStoreFormatXyzwVbuffer::execute(amdgpu::Wavefront &wf) { (void)wf; }

TbufferLoadD16FormatXVbuffer::TbufferLoadD16FormatXVbuffer(const MachineInst *inst)
    : Vbuffer("tbuffer_load_d16_format_x", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
}

void TbufferLoadD16FormatXVbuffer::execute(amdgpu::Wavefront &wf) { (void)wf; }

TbufferLoadD16FormatXyVbuffer::TbufferLoadD16FormatXyVbuffer(const MachineInst *inst)
    : Vbuffer("tbuffer_load_d16_format_xy", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
}

void TbufferLoadD16FormatXyVbuffer::execute(amdgpu::Wavefront &wf) { (void)wf; }

TbufferLoadD16FormatXyzVbuffer::TbufferLoadD16FormatXyzVbuffer(const MachineInst *inst)
    : Vbuffer("tbuffer_load_d16_format_xyz", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
}

void TbufferLoadD16FormatXyzVbuffer::execute(amdgpu::Wavefront &wf) { (void)wf; }

TbufferLoadD16FormatXyzwVbuffer::TbufferLoadD16FormatXyzwVbuffer(const MachineInst *inst)
    : Vbuffer("tbuffer_load_d16_format_xyzw", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
}

void TbufferLoadD16FormatXyzwVbuffer::execute(amdgpu::Wavefront &wf) { (void)wf; }

TbufferStoreD16FormatXVbuffer::TbufferStoreD16FormatXVbuffer(const MachineInst *inst)
    : Vbuffer("tbuffer_store_d16_format_x", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
}

void TbufferStoreD16FormatXVbuffer::execute(amdgpu::Wavefront &wf) { (void)wf; }

TbufferStoreD16FormatXyVbuffer::TbufferStoreD16FormatXyVbuffer(const MachineInst *inst)
    : Vbuffer("tbuffer_store_d16_format_xy", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
}

void TbufferStoreD16FormatXyVbuffer::execute(amdgpu::Wavefront &wf) { (void)wf; }

TbufferStoreD16FormatXyzVbuffer::TbufferStoreD16FormatXyzVbuffer(const MachineInst *inst)
    : Vbuffer("tbuffer_store_d16_format_xyz", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
}

void TbufferStoreD16FormatXyzVbuffer::execute(amdgpu::Wavefront &wf) { (void)wf; }

TbufferStoreD16FormatXyzwVbuffer::TbufferStoreD16FormatXyzwVbuffer(const MachineInst *inst)
    : Vbuffer("tbuffer_store_d16_format_xyzw", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      rsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->rsrc),
      soffset(32, OperandType::OPR_SREG_M0, reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&rsrc);
  src_operands_.emplace_back(&soffset);
}

void TbufferStoreD16FormatXyzwVbuffer::execute(amdgpu::Wavefront &wf) { (void)wf; }

} // namespace rdna4
} // namespace rocjitsu

// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/cdna3/mubuf.h"
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

BufferLoadFormatXMubuf::BufferLoadFormatXMubuf(const MachineInst *inst)
    : Mubuf("buffer_load_format_x", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferLoadFormatXMubuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferLoadFormatXyMubuf::BufferLoadFormatXyMubuf(const MachineInst *inst)
    : Mubuf("buffer_load_format_xy", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferLoadFormatXyMubuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferLoadFormatXyzMubuf::BufferLoadFormatXyzMubuf(const MachineInst *inst)
    : Mubuf("buffer_load_format_xyz", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(96, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferLoadFormatXyzMubuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferLoadFormatXyzwMubuf::BufferLoadFormatXyzwMubuf(const MachineInst *inst)
    : Mubuf("buffer_load_format_xyzw", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferLoadFormatXyzwMubuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferStoreFormatXMubuf::BufferStoreFormatXMubuf(const MachineInst *inst)
    : Mubuf("buffer_store_format_x", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferStoreFormatXMubuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferStoreFormatXyMubuf::BufferStoreFormatXyMubuf(const MachineInst *inst)
    : Mubuf("buffer_store_format_xy", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferStoreFormatXyMubuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferStoreFormatXyzMubuf::BufferStoreFormatXyzMubuf(const MachineInst *inst)
    : Mubuf("buffer_store_format_xyz", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(96, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferStoreFormatXyzMubuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferStoreFormatXyzwMubuf::BufferStoreFormatXyzwMubuf(const MachineInst *inst)
    : Mubuf("buffer_store_format_xyzw", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferStoreFormatXyzwMubuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferLoadFormatD16XMubuf::BufferLoadFormatD16XMubuf(const MachineInst *inst)
    : Mubuf("buffer_load_format_d16_x", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferLoadFormatD16XMubuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferLoadFormatD16XyMubuf::BufferLoadFormatD16XyMubuf(const MachineInst *inst)
    : Mubuf("buffer_load_format_d16_xy", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferLoadFormatD16XyMubuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferLoadFormatD16XyzMubuf::BufferLoadFormatD16XyzMubuf(const MachineInst *inst)
    : Mubuf("buffer_load_format_d16_xyz", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferLoadFormatD16XyzMubuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferLoadFormatD16XyzwMubuf::BufferLoadFormatD16XyzwMubuf(const MachineInst *inst)
    : Mubuf("buffer_load_format_d16_xyzw", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferLoadFormatD16XyzwMubuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferStoreFormatD16XMubuf::BufferStoreFormatD16XMubuf(const MachineInst *inst)
    : Mubuf("buffer_store_format_d16_x", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferStoreFormatD16XMubuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferStoreFormatD16XyMubuf::BufferStoreFormatD16XyMubuf(const MachineInst *inst)
    : Mubuf("buffer_store_format_d16_xy", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferStoreFormatD16XyMubuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferStoreFormatD16XyzMubuf::BufferStoreFormatD16XyzMubuf(const MachineInst *inst)
    : Mubuf("buffer_store_format_d16_xyz", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferStoreFormatD16XyzMubuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferStoreFormatD16XyzwMubuf::BufferStoreFormatD16XyzwMubuf(const MachineInst *inst)
    : Mubuf("buffer_store_format_d16_xyzw", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferStoreFormatD16XyzwMubuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferLoadUbyteMubuf::BufferLoadUbyteMubuf(const MachineInst *inst)
    : Mubuf("buffer_load_ubyte", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadUbyteMubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferLoadSbyteMubuf::BufferLoadSbyteMubuf(const MachineInst *inst)
    : Mubuf("buffer_load_sbyte", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadSbyteMubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->sign_extend = true;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferLoadUshortMubuf::BufferLoadUshortMubuf(const MachineInst *inst)
    : Mubuf("buffer_load_ushort", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadUshortMubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferLoadSshortMubuf::BufferLoadSshortMubuf(const MachineInst *inst)
    : Mubuf("buffer_load_sshort", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadSshortMubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = true;
  d->sign_extend = true;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferLoadDwordMubuf::BufferLoadDwordMubuf(const MachineInst *inst)
    : Mubuf("buffer_load_dword", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadDwordMubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferLoadDwordx2Mubuf::BufferLoadDwordx2Mubuf(const MachineInst *inst)
    : Mubuf("buffer_load_dwordx2", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadDwordx2Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 4;
  d->num_elems = 2;
  d->is_load = true;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferLoadDwordx3Mubuf::BufferLoadDwordx3Mubuf(const MachineInst *inst)
    : Mubuf("buffer_load_dwordx3", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(96, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadDwordx3Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 4;
  d->num_elems = 3;
  d->is_load = true;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferLoadDwordx4Mubuf::BufferLoadDwordx4Mubuf(const MachineInst *inst)
    : Mubuf("buffer_load_dwordx4", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadDwordx4Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 4;
  d->num_elems = 4;
  d->is_load = true;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferStoreByteMubuf::BufferStoreByteMubuf(const MachineInst *inst)
    : Mubuf("buffer_store_byte", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferStoreByteMubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = false;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
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

BufferStoreByteD16HiMubuf::BufferStoreByteD16HiMubuf(const MachineInst *inst)
    : Mubuf("buffer_store_byte_d16_hi", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferStoreByteD16HiMubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = false;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
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

BufferStoreShortMubuf::BufferStoreShortMubuf(const MachineInst *inst)
    : Mubuf("buffer_store_short", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferStoreShortMubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = false;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
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

BufferStoreShortD16HiMubuf::BufferStoreShortD16HiMubuf(const MachineInst *inst)
    : Mubuf("buffer_store_short_d16_hi", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferStoreShortD16HiMubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = false;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
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

BufferStoreDwordMubuf::BufferStoreDwordMubuf(const MachineInst *inst)
    : Mubuf("buffer_store_dword", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferStoreDwordMubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = false;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
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

BufferStoreDwordx2Mubuf::BufferStoreDwordx2Mubuf(const MachineInst *inst)
    : Mubuf("buffer_store_dwordx2", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferStoreDwordx2Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 4;
  d->num_elems = 2;
  d->is_load = false;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
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

BufferStoreDwordx3Mubuf::BufferStoreDwordx3Mubuf(const MachineInst *inst)
    : Mubuf("buffer_store_dwordx3", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(96, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferStoreDwordx3Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 4;
  d->num_elems = 3;
  d->is_load = false;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
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

BufferStoreDwordx4Mubuf::BufferStoreDwordx4Mubuf(const MachineInst *inst)
    : Mubuf("buffer_store_dwordx4", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferStoreDwordx4Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->elem_size = 4;
  d->num_elems = 4;
  d->is_load = false;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
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

BufferLoadUbyteD16Mubuf::BufferLoadUbyteD16Mubuf(const MachineInst *inst)
    : Mubuf("buffer_load_ubyte_d16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadUbyteD16Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferLoadUbyteD16HiMubuf::BufferLoadUbyteD16HiMubuf(const MachineInst *inst)
    : Mubuf("buffer_load_ubyte_d16_hi", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadUbyteD16HiMubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferLoadSbyteD16Mubuf::BufferLoadSbyteD16Mubuf(const MachineInst *inst)
    : Mubuf("buffer_load_sbyte_d16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadSbyteD16Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->sign_extend = true;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferLoadSbyteD16HiMubuf::BufferLoadSbyteD16HiMubuf(const MachineInst *inst)
    : Mubuf("buffer_load_sbyte_d16_hi", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadSbyteD16HiMubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 1;
  d->num_elems = 1;
  d->is_load = true;
  d->sign_extend = true;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferLoadShortD16Mubuf::BufferLoadShortD16Mubuf(const MachineInst *inst)
    : Mubuf("buffer_load_short_d16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadShortD16Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferLoadShortD16HiMubuf::BufferLoadShortD16HiMubuf(const MachineInst *inst)
    : Mubuf("buffer_load_short_d16_hi", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferLoadShortD16HiMubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 2;
  d->num_elems = 1;
  d->is_load = true;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
  mubuf_calculate_addresses(inst_, wf, d->per_lane_addr, d->lane_mask);
  set_data(std::move(d));
}

BufferLoadFormatD16HiXMubuf::BufferLoadFormatD16HiXMubuf(const MachineInst *inst)
    : Mubuf("buffer_load_format_d16_hi_x", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferLoadFormatD16HiXMubuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferStoreFormatD16HiXMubuf::BufferStoreFormatD16HiXMubuf(const MachineInst *inst)
    : Mubuf("buffer_store_format_d16_hi_x", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
}

void BufferStoreFormatD16HiXMubuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferWbl2Mubuf::BufferWbl2Mubuf(const MachineInst *inst)
    : Mubuf("buffer_wbl2", reinterpret_cast<const OpEncoding *>(inst)) {}

void BufferWbl2Mubuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferInvMubuf::BufferInvMubuf(const MachineInst *inst)
    : Mubuf("buffer_inv", reinterpret_cast<const OpEncoding *>(inst)) {}

void BufferInvMubuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

BufferAtomicSwapMubuf::BufferAtomicSwapMubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_swap", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicSwapMubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::SWAP;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
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

BufferAtomicCmpswapMubuf::BufferAtomicCmpswapMubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_cmpswap", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicCmpswapMubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::CMPSWAP;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
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

BufferAtomicAddMubuf::BufferAtomicAddMubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_add", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicAddMubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::ADD;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
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

BufferAtomicSubMubuf::BufferAtomicSubMubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_sub", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicSubMubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::SUB;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
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

BufferAtomicSminMubuf::BufferAtomicSminMubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_smin", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicSminMubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::SMIN;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
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

BufferAtomicUminMubuf::BufferAtomicUminMubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_umin", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicUminMubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::UMIN;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
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

BufferAtomicSmaxMubuf::BufferAtomicSmaxMubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_smax", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicSmaxMubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::SMAX;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
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

BufferAtomicUmaxMubuf::BufferAtomicUmaxMubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_umax", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicUmaxMubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::UMAX;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
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

BufferAtomicAndMubuf::BufferAtomicAndMubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_and", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicAndMubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::AND;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
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

BufferAtomicOrMubuf::BufferAtomicOrMubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_or", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicOrMubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::OR;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
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

BufferAtomicXorMubuf::BufferAtomicXorMubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_xor", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicXorMubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::XOR;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
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

BufferAtomicIncMubuf::BufferAtomicIncMubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_inc", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicIncMubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::INC;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
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

BufferAtomicDecMubuf::BufferAtomicDecMubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_dec", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicDecMubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::DEC;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
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

BufferAtomicAddF32Mubuf::BufferAtomicAddF32Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_add_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
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
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::FADD;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
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

BufferAtomicPkAddF16Mubuf::BufferAtomicPkAddF16Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_pk_add_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicPkAddF16Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::FADD;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
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

BufferAtomicAddF64Mubuf::BufferAtomicAddF64Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_add_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicAddF64Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::FADD;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
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

BufferAtomicMinF64Mubuf::BufferAtomicMinF64Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_min_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicMinF64Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::FMIN;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
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

BufferAtomicMaxF64Mubuf::BufferAtomicMaxF64Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_max_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicMaxF64Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 4;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::FMAX;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
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

BufferAtomicSwapX2Mubuf::BufferAtomicSwapX2Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_swap_x2", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicSwapX2Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::SWAP;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
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

BufferAtomicCmpswapX2Mubuf::BufferAtomicCmpswapX2Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_cmpswap_x2", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(128, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicCmpswapX2Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::CMPSWAP;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
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

BufferAtomicAddX2Mubuf::BufferAtomicAddX2Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_add_x2", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicAddX2Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::ADD;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
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

BufferAtomicSubX2Mubuf::BufferAtomicSubX2Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_sub_x2", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicSubX2Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::SUB;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
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

BufferAtomicSminX2Mubuf::BufferAtomicSminX2Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_smin_x2", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicSminX2Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::SMIN;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
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

BufferAtomicUminX2Mubuf::BufferAtomicUminX2Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_umin_x2", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicUminX2Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::UMIN;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
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

BufferAtomicSmaxX2Mubuf::BufferAtomicSmaxX2Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_smax_x2", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicSmaxX2Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::SMAX;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
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

BufferAtomicUmaxX2Mubuf::BufferAtomicUmaxX2Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_umax_x2", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicUmaxX2Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::UMAX;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
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

BufferAtomicAndX2Mubuf::BufferAtomicAndX2Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_and_x2", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicAndX2Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::AND;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
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

BufferAtomicOrX2Mubuf::BufferAtomicOrX2Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_or_x2", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicOrX2Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::OR;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
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

BufferAtomicXorX2Mubuf::BufferAtomicXorX2Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_xor_x2", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicXorX2Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::XOR;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
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

BufferAtomicIncX2Mubuf::BufferAtomicIncX2Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_inc_x2", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicIncX2Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::INC;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
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

BufferAtomicDecX2Mubuf::BufferAtomicDecX2Mubuf(const MachineInst *inst)
    : Mubuf("buffer_atomic_dec_x2", reinterpret_cast<const OpEncoding *>(inst)),
      vdata(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_.emplace_back(&vdata);
  dst_operands_.emplace_back(&vdata);
  src_operands_.emplace_back(&vaddr);
  src_operands_.emplace_back(&srsrc);
  src_operands_.emplace_back(&soffset);
  flags_ |= MEMORY_OP;
}

void BufferAtomicDecX2Mubuf::execute(amdgpu::Wavefront &wf) {
  auto d = std::make_unique<amdgpu::VectorMemState>(amdgpu::GLOBAL_MEM);
  d->dst_reg_base = wf.vgpr_alloc().base + inst_.vdata;
  d->elem_size = 8;
  d->num_elems = 1;
  d->is_load = (inst_.sc0 != 0);
  d->atomic_op = amdgpu::AtomicOp::DEC;
  d->mtype = mtype_from_bits(inst_.sc0, inst_.sc1);
  d->non_temporal = inst_.nt;
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

} // namespace cdna3
} // namespace rocjitsu

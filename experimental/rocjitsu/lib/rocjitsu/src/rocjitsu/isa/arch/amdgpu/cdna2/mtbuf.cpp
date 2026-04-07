// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/cdna2/mtbuf.h"
#include "rocjitsu/isa/arch/amdgpu/cdna2/addr_calc.h"
#include "rocjitsu/isa/arch/amdgpu/shared/gfx9_cache_flags.h"
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
namespace cdna2 {

TbufferLoadFormatXMtbuf::TbufferLoadFormatXMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_load_format_x", reinterpret_cast<const OpEncoding *>(inst),
            make_exec_fn<TbufferLoadFormatXMtbuf>()),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_[0] = &vdata;
  src_operands_[0] = &vaddr;
  src_operands_[1] = &srsrc;
  src_operands_[2] = &soffset;
  num_src_ = 3;
  num_dst_ = 1;
}

void TbufferLoadFormatXMtbuf::execute_impl(amdgpu::Wavefront &wf) { (void)wf; }

TbufferLoadFormatXyMtbuf::TbufferLoadFormatXyMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_load_format_xy", reinterpret_cast<const OpEncoding *>(inst),
            make_exec_fn<TbufferLoadFormatXyMtbuf>()),
      vdata(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_[0] = &vdata;
  src_operands_[0] = &vaddr;
  src_operands_[1] = &srsrc;
  src_operands_[2] = &soffset;
  num_src_ = 3;
  num_dst_ = 1;
}

void TbufferLoadFormatXyMtbuf::execute_impl(amdgpu::Wavefront &wf) { (void)wf; }

TbufferLoadFormatXyzMtbuf::TbufferLoadFormatXyzMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_load_format_xyz", reinterpret_cast<const OpEncoding *>(inst),
            make_exec_fn<TbufferLoadFormatXyzMtbuf>()),
      vdata(96, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_[0] = &vdata;
  src_operands_[0] = &vaddr;
  src_operands_[1] = &srsrc;
  src_operands_[2] = &soffset;
  num_src_ = 3;
  num_dst_ = 1;
}

void TbufferLoadFormatXyzMtbuf::execute_impl(amdgpu::Wavefront &wf) { (void)wf; }

TbufferLoadFormatXyzwMtbuf::TbufferLoadFormatXyzwMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_load_format_xyzw", reinterpret_cast<const OpEncoding *>(inst),
            make_exec_fn<TbufferLoadFormatXyzwMtbuf>()),
      vdata(128, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_[0] = &vdata;
  src_operands_[0] = &vaddr;
  src_operands_[1] = &srsrc;
  src_operands_[2] = &soffset;
  num_src_ = 3;
  num_dst_ = 1;
}

void TbufferLoadFormatXyzwMtbuf::execute_impl(amdgpu::Wavefront &wf) { (void)wf; }

TbufferStoreFormatXMtbuf::TbufferStoreFormatXMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_store_format_x", reinterpret_cast<const OpEncoding *>(inst),
            make_exec_fn<TbufferStoreFormatXMtbuf>()),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_[0] = &vdata;
  src_operands_[1] = &vaddr;
  src_operands_[2] = &srsrc;
  src_operands_[3] = &soffset;
  num_src_ = 4;
  num_dst_ = 0;
}

void TbufferStoreFormatXMtbuf::execute_impl(amdgpu::Wavefront &wf) { (void)wf; }

TbufferStoreFormatXyMtbuf::TbufferStoreFormatXyMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_store_format_xy", reinterpret_cast<const OpEncoding *>(inst),
            make_exec_fn<TbufferStoreFormatXyMtbuf>()),
      vdata(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_[0] = &vdata;
  src_operands_[1] = &vaddr;
  src_operands_[2] = &srsrc;
  src_operands_[3] = &soffset;
  num_src_ = 4;
  num_dst_ = 0;
}

void TbufferStoreFormatXyMtbuf::execute_impl(amdgpu::Wavefront &wf) { (void)wf; }

TbufferStoreFormatXyzMtbuf::TbufferStoreFormatXyzMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_store_format_xyz", reinterpret_cast<const OpEncoding *>(inst),
            make_exec_fn<TbufferStoreFormatXyzMtbuf>()),
      vdata(96, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_[0] = &vdata;
  src_operands_[1] = &vaddr;
  src_operands_[2] = &srsrc;
  src_operands_[3] = &soffset;
  num_src_ = 4;
  num_dst_ = 0;
}

void TbufferStoreFormatXyzMtbuf::execute_impl(amdgpu::Wavefront &wf) { (void)wf; }

TbufferStoreFormatXyzwMtbuf::TbufferStoreFormatXyzwMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_store_format_xyzw", reinterpret_cast<const OpEncoding *>(inst),
            make_exec_fn<TbufferStoreFormatXyzwMtbuf>()),
      vdata(128, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_[0] = &vdata;
  src_operands_[1] = &vaddr;
  src_operands_[2] = &srsrc;
  src_operands_[3] = &soffset;
  num_src_ = 4;
  num_dst_ = 0;
}

void TbufferStoreFormatXyzwMtbuf::execute_impl(amdgpu::Wavefront &wf) { (void)wf; }

TbufferLoadFormatD16XMtbuf::TbufferLoadFormatD16XMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_load_format_d16_x", reinterpret_cast<const OpEncoding *>(inst),
            make_exec_fn<TbufferLoadFormatD16XMtbuf>()),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_[0] = &vdata;
  src_operands_[0] = &vaddr;
  src_operands_[1] = &srsrc;
  src_operands_[2] = &soffset;
  num_src_ = 3;
  num_dst_ = 1;
}

void TbufferLoadFormatD16XMtbuf::execute_impl(amdgpu::Wavefront &wf) { (void)wf; }

TbufferLoadFormatD16XyMtbuf::TbufferLoadFormatD16XyMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_load_format_d16_xy", reinterpret_cast<const OpEncoding *>(inst),
            make_exec_fn<TbufferLoadFormatD16XyMtbuf>()),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_[0] = &vdata;
  src_operands_[0] = &vaddr;
  src_operands_[1] = &srsrc;
  src_operands_[2] = &soffset;
  num_src_ = 3;
  num_dst_ = 1;
}

void TbufferLoadFormatD16XyMtbuf::execute_impl(amdgpu::Wavefront &wf) { (void)wf; }

TbufferLoadFormatD16XyzMtbuf::TbufferLoadFormatD16XyzMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_load_format_d16_xyz", reinterpret_cast<const OpEncoding *>(inst),
            make_exec_fn<TbufferLoadFormatD16XyzMtbuf>()),
      vdata(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_[0] = &vdata;
  src_operands_[0] = &vaddr;
  src_operands_[1] = &srsrc;
  src_operands_[2] = &soffset;
  num_src_ = 3;
  num_dst_ = 1;
}

void TbufferLoadFormatD16XyzMtbuf::execute_impl(amdgpu::Wavefront &wf) { (void)wf; }

TbufferLoadFormatD16XyzwMtbuf::TbufferLoadFormatD16XyzwMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_load_format_d16_xyzw", reinterpret_cast<const OpEncoding *>(inst),
            make_exec_fn<TbufferLoadFormatD16XyzwMtbuf>()),
      vdata(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  dst_operands_[0] = &vdata;
  src_operands_[0] = &vaddr;
  src_operands_[1] = &srsrc;
  src_operands_[2] = &soffset;
  num_src_ = 3;
  num_dst_ = 1;
}

void TbufferLoadFormatD16XyzwMtbuf::execute_impl(amdgpu::Wavefront &wf) { (void)wf; }

TbufferStoreFormatD16XMtbuf::TbufferStoreFormatD16XMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_store_format_d16_x", reinterpret_cast<const OpEncoding *>(inst),
            make_exec_fn<TbufferStoreFormatD16XMtbuf>()),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_[0] = &vdata;
  src_operands_[1] = &vaddr;
  src_operands_[2] = &srsrc;
  src_operands_[3] = &soffset;
  num_src_ = 4;
  num_dst_ = 0;
}

void TbufferStoreFormatD16XMtbuf::execute_impl(amdgpu::Wavefront &wf) { (void)wf; }

TbufferStoreFormatD16XyMtbuf::TbufferStoreFormatD16XyMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_store_format_d16_xy", reinterpret_cast<const OpEncoding *>(inst),
            make_exec_fn<TbufferStoreFormatD16XyMtbuf>()),
      vdata(32, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_[0] = &vdata;
  src_operands_[1] = &vaddr;
  src_operands_[2] = &srsrc;
  src_operands_[3] = &soffset;
  num_src_ = 4;
  num_dst_ = 0;
}

void TbufferStoreFormatD16XyMtbuf::execute_impl(amdgpu::Wavefront &wf) { (void)wf; }

TbufferStoreFormatD16XyzMtbuf::TbufferStoreFormatD16XyzMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_store_format_d16_xyz", reinterpret_cast<const OpEncoding *>(inst),
            make_exec_fn<TbufferStoreFormatD16XyzMtbuf>()),
      vdata(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_[0] = &vdata;
  src_operands_[1] = &vaddr;
  src_operands_[2] = &srsrc;
  src_operands_[3] = &soffset;
  num_src_ = 4;
  num_dst_ = 0;
}

void TbufferStoreFormatD16XyzMtbuf::execute_impl(amdgpu::Wavefront &wf) { (void)wf; }

TbufferStoreFormatD16XyzwMtbuf::TbufferStoreFormatD16XyzwMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_store_format_d16_xyzw", reinterpret_cast<const OpEncoding *>(inst),
            make_exec_fn<TbufferStoreFormatD16XyzwMtbuf>()),
      vdata(64, OperandType::OPR_VGPR_OR_ACCVGPR,
            reinterpret_cast<const OpEncoding *>(inst)->vdata),
      vaddr(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vaddr),
      srsrc(128, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->srsrc),
      soffset(32, OperandType::OPR_SSRC_NOLIT,
              reinterpret_cast<const OpEncoding *>(inst)->soffset) {
  src_operands_[0] = &vdata;
  src_operands_[1] = &vaddr;
  src_operands_[2] = &srsrc;
  src_operands_[3] = &soffset;
  num_src_ = 4;
  num_dst_ = 0;
}

void TbufferStoreFormatD16XyzwMtbuf::execute_impl(amdgpu::Wavefront &wf) { (void)wf; }

} // namespace cdna2
} // namespace rocjitsu

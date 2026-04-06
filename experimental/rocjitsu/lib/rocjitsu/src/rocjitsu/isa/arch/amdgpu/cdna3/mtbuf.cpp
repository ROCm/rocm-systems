// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/cdna3/mtbuf.h"
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

TbufferLoadFormatXMtbuf::TbufferLoadFormatXMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_load_format_x", reinterpret_cast<const OpEncoding *>(inst)),
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

void TbufferLoadFormatXMtbuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

TbufferLoadFormatXyMtbuf::TbufferLoadFormatXyMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_load_format_xy", reinterpret_cast<const OpEncoding *>(inst)),
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

void TbufferLoadFormatXyMtbuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

TbufferLoadFormatXyzMtbuf::TbufferLoadFormatXyzMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_load_format_xyz", reinterpret_cast<const OpEncoding *>(inst)),
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

void TbufferLoadFormatXyzMtbuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

TbufferLoadFormatXyzwMtbuf::TbufferLoadFormatXyzwMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_load_format_xyzw", reinterpret_cast<const OpEncoding *>(inst)),
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

void TbufferLoadFormatXyzwMtbuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

TbufferStoreFormatXMtbuf::TbufferStoreFormatXMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_store_format_x", reinterpret_cast<const OpEncoding *>(inst)),
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

void TbufferStoreFormatXMtbuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

TbufferStoreFormatXyMtbuf::TbufferStoreFormatXyMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_store_format_xy", reinterpret_cast<const OpEncoding *>(inst)),
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

void TbufferStoreFormatXyMtbuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

TbufferStoreFormatXyzMtbuf::TbufferStoreFormatXyzMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_store_format_xyz", reinterpret_cast<const OpEncoding *>(inst)),
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

void TbufferStoreFormatXyzMtbuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

TbufferStoreFormatXyzwMtbuf::TbufferStoreFormatXyzwMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_store_format_xyzw", reinterpret_cast<const OpEncoding *>(inst)),
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

void TbufferStoreFormatXyzwMtbuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

TbufferLoadFormatD16XMtbuf::TbufferLoadFormatD16XMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_load_format_d16_x", reinterpret_cast<const OpEncoding *>(inst)),
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

void TbufferLoadFormatD16XMtbuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

TbufferLoadFormatD16XyMtbuf::TbufferLoadFormatD16XyMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_load_format_d16_xy", reinterpret_cast<const OpEncoding *>(inst)),
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

void TbufferLoadFormatD16XyMtbuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

TbufferLoadFormatD16XyzMtbuf::TbufferLoadFormatD16XyzMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_load_format_d16_xyz", reinterpret_cast<const OpEncoding *>(inst)),
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

void TbufferLoadFormatD16XyzMtbuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

TbufferLoadFormatD16XyzwMtbuf::TbufferLoadFormatD16XyzwMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_load_format_d16_xyzw", reinterpret_cast<const OpEncoding *>(inst)),
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

void TbufferLoadFormatD16XyzwMtbuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

TbufferStoreFormatD16XMtbuf::TbufferStoreFormatD16XMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_store_format_d16_x", reinterpret_cast<const OpEncoding *>(inst)),
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

void TbufferStoreFormatD16XMtbuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

TbufferStoreFormatD16XyMtbuf::TbufferStoreFormatD16XyMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_store_format_d16_xy", reinterpret_cast<const OpEncoding *>(inst)),
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

void TbufferStoreFormatD16XyMtbuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

TbufferStoreFormatD16XyzMtbuf::TbufferStoreFormatD16XyzMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_store_format_d16_xyz", reinterpret_cast<const OpEncoding *>(inst)),
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

void TbufferStoreFormatD16XyzMtbuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

TbufferStoreFormatD16XyzwMtbuf::TbufferStoreFormatD16XyzwMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_store_format_d16_xyzw", reinterpret_cast<const OpEncoding *>(inst)),
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

void TbufferStoreFormatD16XyzwMtbuf::execute(amdgpu::Wavefront &wf) { (void)wf; }

} // namespace cdna3
} // namespace rocjitsu

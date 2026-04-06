// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/rdna3/mtbuf.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/addr_calc.h"
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

TbufferLoadFormatXMtbuf::TbufferLoadFormatXMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_load_format_x", reinterpret_cast<const OpEncoding *>(inst),
            make_exec_fn<TbufferLoadFormatXMtbuf>()),
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

void TbufferLoadFormatXMtbuf::execute_impl(amdgpu::Wavefront &wf) { (void)wf; }

TbufferLoadFormatXyMtbuf::TbufferLoadFormatXyMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_load_format_xy", reinterpret_cast<const OpEncoding *>(inst),
            make_exec_fn<TbufferLoadFormatXyMtbuf>()),
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

void TbufferLoadFormatXyMtbuf::execute_impl(amdgpu::Wavefront &wf) { (void)wf; }

TbufferLoadFormatXyzMtbuf::TbufferLoadFormatXyzMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_load_format_xyz", reinterpret_cast<const OpEncoding *>(inst),
            make_exec_fn<TbufferLoadFormatXyzMtbuf>()),
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

void TbufferLoadFormatXyzMtbuf::execute_impl(amdgpu::Wavefront &wf) { (void)wf; }

TbufferLoadFormatXyzwMtbuf::TbufferLoadFormatXyzwMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_load_format_xyzw", reinterpret_cast<const OpEncoding *>(inst),
            make_exec_fn<TbufferLoadFormatXyzwMtbuf>()),
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

void TbufferLoadFormatXyzwMtbuf::execute_impl(amdgpu::Wavefront &wf) { (void)wf; }

TbufferStoreFormatXMtbuf::TbufferStoreFormatXMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_store_format_x", reinterpret_cast<const OpEncoding *>(inst),
            make_exec_fn<TbufferStoreFormatXMtbuf>()),
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

void TbufferStoreFormatXMtbuf::execute_impl(amdgpu::Wavefront &wf) { (void)wf; }

TbufferStoreFormatXyMtbuf::TbufferStoreFormatXyMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_store_format_xy", reinterpret_cast<const OpEncoding *>(inst),
            make_exec_fn<TbufferStoreFormatXyMtbuf>()),
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

void TbufferStoreFormatXyMtbuf::execute_impl(amdgpu::Wavefront &wf) { (void)wf; }

TbufferStoreFormatXyzMtbuf::TbufferStoreFormatXyzMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_store_format_xyz", reinterpret_cast<const OpEncoding *>(inst),
            make_exec_fn<TbufferStoreFormatXyzMtbuf>()),
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

void TbufferStoreFormatXyzMtbuf::execute_impl(amdgpu::Wavefront &wf) { (void)wf; }

TbufferStoreFormatXyzwMtbuf::TbufferStoreFormatXyzwMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_store_format_xyzw", reinterpret_cast<const OpEncoding *>(inst),
            make_exec_fn<TbufferStoreFormatXyzwMtbuf>()),
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

void TbufferStoreFormatXyzwMtbuf::execute_impl(amdgpu::Wavefront &wf) { (void)wf; }

TbufferLoadD16FormatXMtbuf::TbufferLoadD16FormatXMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_load_d16_format_x", reinterpret_cast<const OpEncoding *>(inst),
            make_exec_fn<TbufferLoadD16FormatXMtbuf>()),
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

void TbufferLoadD16FormatXMtbuf::execute_impl(amdgpu::Wavefront &wf) { (void)wf; }

TbufferLoadD16FormatXyMtbuf::TbufferLoadD16FormatXyMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_load_d16_format_xy", reinterpret_cast<const OpEncoding *>(inst),
            make_exec_fn<TbufferLoadD16FormatXyMtbuf>()),
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

void TbufferLoadD16FormatXyMtbuf::execute_impl(amdgpu::Wavefront &wf) { (void)wf; }

TbufferLoadD16FormatXyzMtbuf::TbufferLoadD16FormatXyzMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_load_d16_format_xyz", reinterpret_cast<const OpEncoding *>(inst),
            make_exec_fn<TbufferLoadD16FormatXyzMtbuf>()),
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

void TbufferLoadD16FormatXyzMtbuf::execute_impl(amdgpu::Wavefront &wf) { (void)wf; }

TbufferLoadD16FormatXyzwMtbuf::TbufferLoadD16FormatXyzwMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_load_d16_format_xyzw", reinterpret_cast<const OpEncoding *>(inst),
            make_exec_fn<TbufferLoadD16FormatXyzwMtbuf>()),
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

void TbufferLoadD16FormatXyzwMtbuf::execute_impl(amdgpu::Wavefront &wf) { (void)wf; }

TbufferStoreD16FormatXMtbuf::TbufferStoreD16FormatXMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_store_d16_format_x", reinterpret_cast<const OpEncoding *>(inst),
            make_exec_fn<TbufferStoreD16FormatXMtbuf>()),
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

void TbufferStoreD16FormatXMtbuf::execute_impl(amdgpu::Wavefront &wf) { (void)wf; }

TbufferStoreD16FormatXyMtbuf::TbufferStoreD16FormatXyMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_store_d16_format_xy", reinterpret_cast<const OpEncoding *>(inst),
            make_exec_fn<TbufferStoreD16FormatXyMtbuf>()),
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

void TbufferStoreD16FormatXyMtbuf::execute_impl(amdgpu::Wavefront &wf) { (void)wf; }

TbufferStoreD16FormatXyzMtbuf::TbufferStoreD16FormatXyzMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_store_d16_format_xyz", reinterpret_cast<const OpEncoding *>(inst),
            make_exec_fn<TbufferStoreD16FormatXyzMtbuf>()),
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

void TbufferStoreD16FormatXyzMtbuf::execute_impl(amdgpu::Wavefront &wf) { (void)wf; }

TbufferStoreD16FormatXyzwMtbuf::TbufferStoreD16FormatXyzwMtbuf(const MachineInst *inst)
    : Mtbuf("tbuffer_store_d16_format_xyzw", reinterpret_cast<const OpEncoding *>(inst),
            make_exec_fn<TbufferStoreD16FormatXyzwMtbuf>()),
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

void TbufferStoreD16FormatXyzwMtbuf::execute_impl(amdgpu::Wavefront &wf) { (void)wf; }

} // namespace rdna3
} // namespace rocjitsu

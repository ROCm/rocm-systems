// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/rdna1/vintrp.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"
#include "util/except.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

namespace rocjitsu {
namespace rdna1 {

VInterpP1F32Vintrp::VInterpP1F32Vintrp(const MachineInst *inst)
    : Vintrp("v_interp_p1_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      attr(32, OperandType::OPR_ATTR, reinterpret_cast<const OpEncoding *>(inst)->attr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&attr);
}

void VInterpP1F32Vintrp::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: interp
}

VInterpP2F32Vintrp::VInterpP2F32Vintrp(const MachineInst *inst)
    : Vintrp("v_interp_p2_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vsrc(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      attr(32, OperandType::OPR_ATTR, reinterpret_cast<const OpEncoding *>(inst)->attr) {
  src_operands_.emplace_back(&vdst);
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&attr);
}

void VInterpP2F32Vintrp::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: interp
}

VInterpMovF32Vintrp::VInterpMovF32Vintrp(const MachineInst *inst)
    : Vintrp("v_interp_mov_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      vsrc(32, OperandType::OPR_PARAM, reinterpret_cast<const OpEncoding *>(inst)->vsrc),
      attr(32, OperandType::OPR_ATTR, reinterpret_cast<const OpEncoding *>(inst)->attr) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&vsrc);
  src_operands_.emplace_back(&attr);
}

void VInterpMovF32Vintrp::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic()); // unhandled semantic class: interp
}

} // namespace rdna1
} // namespace rocjitsu

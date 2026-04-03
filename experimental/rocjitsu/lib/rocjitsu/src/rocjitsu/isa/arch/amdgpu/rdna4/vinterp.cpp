// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/rdna4/vinterp.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"
#include "util/except.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

namespace rocjitsu {
namespace rdna4 {

VInterpP10F32Vinterp::VInterpP10F32Vinterp(const MachineInst *inst)
    : Vinterp("v_interp_p10_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VInterpP10F32Vinterp::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Interpolation/LDS-direct: no-op in compute simulation.
}

VInterpP2F32Vinterp::VInterpP2F32Vinterp(const MachineInst *inst)
    : Vinterp("v_interp_p2_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VInterpP2F32Vinterp::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Interpolation/LDS-direct: no-op in compute simulation.
}

VInterpP10F16F32Vinterp::VInterpP10F16F32Vinterp(const MachineInst *inst)
    : Vinterp("v_interp_p10_f16_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(16, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VInterpP10F16F32Vinterp::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Interpolation/LDS-direct: no-op in compute simulation.
}

VInterpP2F16F32Vinterp::VInterpP2F16F32Vinterp(const MachineInst *inst)
    : Vinterp("v_interp_p2_f16_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VInterpP2F16F32Vinterp::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Interpolation/LDS-direct: no-op in compute simulation.
}

VInterpP10RtzF16F32Vinterp::VInterpP10RtzF16F32Vinterp(const MachineInst *inst)
    : Vinterp("v_interp_p10_rtz_f16_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(16, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VInterpP10RtzF16F32Vinterp::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Interpolation/LDS-direct: no-op in compute simulation.
}

VInterpP2RtzF16F32Vinterp::VInterpP2RtzF16F32Vinterp(const MachineInst *inst)
    : Vinterp("v_interp_p2_rtz_f16_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VInterpP2RtzF16F32Vinterp::execute(amdgpu::Wavefront &wf) {
  (void)wf; // Interpolation/LDS-direct: no-op in compute simulation.
}

} // namespace rdna4
} // namespace rocjitsu

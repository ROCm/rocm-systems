// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/rdna3/vop2.h"
#include "rocjitsu/isa/arch/amdgpu/shared/execute_shared.h"
#include "rocjitsu/isa/arch/amdgpu/shared/transcendental.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"
#include "util/except.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

namespace rocjitsu {
namespace rdna3 {

VCndmaskB32Vop2::VCndmaskB32Vop2(const MachineInst *inst)
    : Vop2("v_cndmask_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VCndmaskB32Vop2::execute(amdgpu::Wavefront &wf) {
  amdgpu::execute_v_cndmask_b32_vop2(*this, wf);
}

VDot2accF32F16Vop2::VDot2accF32F16Vop2(const MachineInst *inst)
    : Vop2("v_dot2acc_f32_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&vdst);
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VDot2accF32F16Vop2::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VAddF32Vop2::VAddF32Vop2(const MachineInst *inst)
    : Vop2("v_add_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VAddF32Vop2::execute(amdgpu::Wavefront &wf) { amdgpu::execute_v_add_f32_vop2(*this, wf); }

VSubF32Vop2::VSubF32Vop2(const MachineInst *inst)
    : Vop2("v_sub_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VSubF32Vop2::execute(amdgpu::Wavefront &wf) { amdgpu::execute_v_sub_f32_vop2(*this, wf); }

VSubrevF32Vop2::VSubrevF32Vop2(const MachineInst *inst)
    : Vop2("v_subrev_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VSubrevF32Vop2::execute(amdgpu::Wavefront &wf) {
  amdgpu::execute_v_subrev_f32_vop2(*this, wf);
}

VFmacDx9ZeroF32Vop2::VFmacDx9ZeroF32Vop2(const MachineInst *inst)
    : Vop2("v_fmac_dx9_zero_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&vdst);
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VFmacDx9ZeroF32Vop2::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VMulDx9ZeroF32Vop2::VMulDx9ZeroF32Vop2(const MachineInst *inst)
    : Vop2("v_mul_dx9_zero_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VMulDx9ZeroF32Vop2::execute(amdgpu::Wavefront &wf) {
  amdgpu::execute_v_mul_dx9_zero_f32_vop2(*this, wf);
}

VMulF32Vop2::VMulF32Vop2(const MachineInst *inst)
    : Vop2("v_mul_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VMulF32Vop2::execute(amdgpu::Wavefront &wf) { amdgpu::execute_v_mul_f32_vop2(*this, wf); }

VMulI32I24Vop2::VMulI32I24Vop2(const MachineInst *inst)
    : Vop2("v_mul_i32_i24", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VMulI32I24Vop2::execute(amdgpu::Wavefront &wf) {
  amdgpu::execute_v_mul_i32_i24_vop2(*this, wf);
}

VMulHiI32I24Vop2::VMulHiI32I24Vop2(const MachineInst *inst)
    : Vop2("v_mul_hi_i32_i24", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VMulHiI32I24Vop2::execute(amdgpu::Wavefront &wf) {
  amdgpu::execute_v_mul_hi_i32_i24_vop2(*this, wf);
}

VMulU32U24Vop2::VMulU32U24Vop2(const MachineInst *inst)
    : Vop2("v_mul_u32_u24", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VMulU32U24Vop2::execute(amdgpu::Wavefront &wf) {
  amdgpu::execute_v_mul_u32_u24_vop2(*this, wf);
}

VMulHiU32U24Vop2::VMulHiU32U24Vop2(const MachineInst *inst)
    : Vop2("v_mul_hi_u32_u24", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VMulHiU32U24Vop2::execute(amdgpu::Wavefront &wf) {
  amdgpu::execute_v_mul_hi_u32_u24_vop2(*this, wf);
}

VMinF32Vop2::VMinF32Vop2(const MachineInst *inst)
    : Vop2("v_min_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VMinF32Vop2::execute(amdgpu::Wavefront &wf) { amdgpu::execute_v_min_f32_vop2(*this, wf); }

VMaxF32Vop2::VMaxF32Vop2(const MachineInst *inst)
    : Vop2("v_max_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VMaxF32Vop2::execute(amdgpu::Wavefront &wf) { amdgpu::execute_v_max_f32_vop2(*this, wf); }

VMinI32Vop2::VMinI32Vop2(const MachineInst *inst)
    : Vop2("v_min_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VMinI32Vop2::execute(amdgpu::Wavefront &wf) { amdgpu::execute_v_min_i32_vop2(*this, wf); }

VMaxI32Vop2::VMaxI32Vop2(const MachineInst *inst)
    : Vop2("v_max_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VMaxI32Vop2::execute(amdgpu::Wavefront &wf) { amdgpu::execute_v_max_i32_vop2(*this, wf); }

VMinU32Vop2::VMinU32Vop2(const MachineInst *inst)
    : Vop2("v_min_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VMinU32Vop2::execute(amdgpu::Wavefront &wf) { amdgpu::execute_v_min_u32_vop2(*this, wf); }

VMaxU32Vop2::VMaxU32Vop2(const MachineInst *inst)
    : Vop2("v_max_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VMaxU32Vop2::execute(amdgpu::Wavefront &wf) { amdgpu::execute_v_max_u32_vop2(*this, wf); }

VLshlrevB32Vop2::VLshlrevB32Vop2(const MachineInst *inst)
    : Vop2("v_lshlrev_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VLshlrevB32Vop2::execute(amdgpu::Wavefront &wf) {
  amdgpu::execute_v_lshlrev_b32_vop2(*this, wf);
}

VLshrrevB32Vop2::VLshrrevB32Vop2(const MachineInst *inst)
    : Vop2("v_lshrrev_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VLshrrevB32Vop2::execute(amdgpu::Wavefront &wf) {
  amdgpu::execute_v_lshrrev_b32_vop2(*this, wf);
}

VAshrrevI32Vop2::VAshrrevI32Vop2(const MachineInst *inst)
    : Vop2("v_ashrrev_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VAshrrevI32Vop2::execute(amdgpu::Wavefront &wf) {
  amdgpu::execute_v_ashrrev_i32_vop2(*this, wf);
}

VAndB32Vop2::VAndB32Vop2(const MachineInst *inst)
    : Vop2("v_and_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VAndB32Vop2::execute(amdgpu::Wavefront &wf) { amdgpu::execute_v_and_b32_vop2(*this, wf); }

VOrB32Vop2::VOrB32Vop2(const MachineInst *inst)
    : Vop2("v_or_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VOrB32Vop2::execute(amdgpu::Wavefront &wf) { amdgpu::execute_v_or_b32_vop2(*this, wf); }

VXorB32Vop2::VXorB32Vop2(const MachineInst *inst)
    : Vop2("v_xor_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VXorB32Vop2::execute(amdgpu::Wavefront &wf) { amdgpu::execute_v_xor_b32_vop2(*this, wf); }

VXnorB32Vop2::VXnorB32Vop2(const MachineInst *inst)
    : Vop2("v_xnor_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VXnorB32Vop2::execute(amdgpu::Wavefront &wf) { amdgpu::execute_v_xnor_b32_vop2(*this, wf); }

VAddCoCiU32Vop2::VAddCoCiU32Vop2(const MachineInst *inst)
    : Vop2("v_add_co_ci_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VAddCoCiU32Vop2::execute(amdgpu::Wavefront &wf) {
  amdgpu::execute_v_add_co_ci_u32_vop2(*this, wf);
}

VSubCoCiU32Vop2::VSubCoCiU32Vop2(const MachineInst *inst)
    : Vop2("v_sub_co_ci_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VSubCoCiU32Vop2::execute(amdgpu::Wavefront &wf) {
  amdgpu::execute_v_sub_co_ci_u32_vop2(*this, wf);
}

VSubrevCoCiU32Vop2::VSubrevCoCiU32Vop2(const MachineInst *inst)
    : Vop2("v_subrev_co_ci_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VSubrevCoCiU32Vop2::execute(amdgpu::Wavefront &wf) {
  amdgpu::execute_v_subrev_co_ci_u32_vop2(*this, wf);
}

VAddNcU32Vop2::VAddNcU32Vop2(const MachineInst *inst)
    : Vop2("v_add_nc_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VAddNcU32Vop2::execute(amdgpu::Wavefront &wf) { amdgpu::execute_v_add_nc_u32_vop2(*this, wf); }

VSubNcU32Vop2::VSubNcU32Vop2(const MachineInst *inst)
    : Vop2("v_sub_nc_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VSubNcU32Vop2::execute(amdgpu::Wavefront &wf) { amdgpu::execute_v_sub_nc_u32_vop2(*this, wf); }

VSubrevNcU32Vop2::VSubrevNcU32Vop2(const MachineInst *inst)
    : Vop2("v_subrev_nc_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VSubrevNcU32Vop2::execute(amdgpu::Wavefront &wf) {
  amdgpu::execute_v_subrev_nc_u32_vop2(*this, wf);
}

VFmacF32Vop2::VFmacF32Vop2(const MachineInst *inst)
    : Vop2("v_fmac_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&vdst);
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VFmacF32Vop2::execute(amdgpu::Wavefront &wf) { amdgpu::execute_v_fmac_f32_vop2(*this, wf); }

VFmamkF32Vop2::VFmamkF32Vop2(const MachineInst *inst)
    : Vop2("v_fmamk_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1),
      simm32_(0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
  simm32_ = reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32;
}

void VFmamkF32Vop2::execute(amdgpu::Wavefront &wf) { amdgpu::execute_v_fmamk_f32_vop2(*this, wf); }

VFmaakF32Vop2::VFmaakF32Vop2(const MachineInst *inst)
    : Vop2("v_fmaak_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1),
      simm32_(0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
  simm32_ = reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32;
}

void VFmaakF32Vop2::execute(amdgpu::Wavefront &wf) { amdgpu::execute_v_fmaak_f32_vop2(*this, wf); }

VCvtPkRtzF16F32Vop2::VCvtPkRtzF16F32Vop2(const MachineInst *inst)
    : Vop2("v_cvt_pk_rtz_f16_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VCvtPkRtzF16F32Vop2::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VAddF16Vop2::VAddF16Vop2(const MachineInst *inst)
    : Vop2("v_add_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VAddF16Vop2::execute(amdgpu::Wavefront &wf) { amdgpu::execute_v_add_f16_vop2(*this, wf); }

VSubF16Vop2::VSubF16Vop2(const MachineInst *inst)
    : Vop2("v_sub_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VSubF16Vop2::execute(amdgpu::Wavefront &wf) { amdgpu::execute_v_sub_f16_vop2(*this, wf); }

VSubrevF16Vop2::VSubrevF16Vop2(const MachineInst *inst)
    : Vop2("v_subrev_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VSubrevF16Vop2::execute(amdgpu::Wavefront &wf) {
  amdgpu::execute_v_subrev_f16_vop2(*this, wf);
}

VMulF16Vop2::VMulF16Vop2(const MachineInst *inst)
    : Vop2("v_mul_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VMulF16Vop2::execute(amdgpu::Wavefront &wf) { amdgpu::execute_v_mul_f16_vop2(*this, wf); }

VFmacF16Vop2::VFmacF16Vop2(const MachineInst *inst)
    : Vop2("v_fmac_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&vdst);
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VFmacF16Vop2::execute(amdgpu::Wavefront &wf) { amdgpu::execute_v_fmac_f16_vop2(*this, wf); }

VFmamkF16Vop2::VFmamkF16Vop2(const MachineInst *inst)
    : Vop2("v_fmamk_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1),
      simm32_(0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
  simm32_ = reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32;
}

void VFmamkF16Vop2::execute(amdgpu::Wavefront &wf) { amdgpu::execute_v_fmamk_f16_vop2(*this, wf); }

VFmaakF16Vop2::VFmaakF16Vop2(const MachineInst *inst)
    : Vop2("v_fmaak_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1),
      simm32_(0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
  simm32_ = reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32;
}

void VFmaakF16Vop2::execute(amdgpu::Wavefront &wf) { amdgpu::execute_v_fmaak_f16_vop2(*this, wf); }

VMaxF16Vop2::VMaxF16Vop2(const MachineInst *inst)
    : Vop2("v_max_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VMaxF16Vop2::execute(amdgpu::Wavefront &wf) { amdgpu::execute_v_max_f16_vop2(*this, wf); }

VMinF16Vop2::VMinF16Vop2(const MachineInst *inst)
    : Vop2("v_min_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VMinF16Vop2::execute(amdgpu::Wavefront &wf) { amdgpu::execute_v_min_f16_vop2(*this, wf); }

VLdexpF16Vop2::VLdexpF16Vop2(const MachineInst *inst)
    : Vop2("v_ldexp_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VLdexpF16Vop2::execute(amdgpu::Wavefront &wf) { amdgpu::execute_v_ldexp_f16_vop2(*this, wf); }

VPkFmacF16Vop2::VPkFmacF16Vop2(const MachineInst *inst)
    : Vop2("v_pk_fmac_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&vdst);
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VPkFmacF16Vop2::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

} // namespace rdna3
} // namespace rocjitsu

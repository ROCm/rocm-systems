// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/cdna2/vop2.h"
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
namespace cdna2 {

VCndmaskB32Vop2::VCndmaskB32Vop2(const MachineInst *inst)
    : Vop2("v_cndmask_b32_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VCndmaskB32Vop2>()),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VCndmaskB32Vop2::execute_impl(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t val = (vcc & (1ULL << lane)) ? vsrc1.read_lane(wf, lane) : src0.read_lane(wf, lane);
    vdst.write_lane(wf, lane, val);
  }
}

VAddF32Vop2::VAddF32Vop2(const MachineInst *inst)
    : Vop2("v_add_f32_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VAddF32Vop2>()),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VAddF32Vop2::execute_impl(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float sv0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float sv1 = std::bit_cast<float>(vsrc1.read_lane(wf, lane));
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(sv0 + sv1));
  }
}

VSubF32Vop2::VSubF32Vop2(const MachineInst *inst)
    : Vop2("v_sub_f32_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VSubF32Vop2>()),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VSubF32Vop2::execute_impl(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float sv0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float sv1 = std::bit_cast<float>(vsrc1.read_lane(wf, lane));
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(sv0 - sv1));
  }
}

VSubrevF32Vop2::VSubrevF32Vop2(const MachineInst *inst)
    : Vop2("v_subrev_f32_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VSubrevF32Vop2>()),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VSubrevF32Vop2::execute_impl(amdgpu::Wavefront &wf) {
  amdgpu::execute_v_subrev_f32_vop2(*this, wf);
}

VFmacF64Vop2::VFmacF64Vop2(const MachineInst *inst)
    : Vop2("v_fmac_f64_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VFmacF64Vop2>()),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_[0] = &vdst;
  dst_operands_[0] = &vdst;
  src_operands_[1] = &src0;
  src_operands_[2] = &vsrc1;
  num_src_ = 3;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VFmacF64Vop2::execute_impl(amdgpu::Wavefront &wf) {
  amdgpu::execute_v_fmac_f64_vop2(*this, wf);
}

VMulF32Vop2::VMulF32Vop2(const MachineInst *inst)
    : Vop2("v_mul_f32_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VMulF32Vop2>()),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VMulF32Vop2::execute_impl(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float sv0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float sv1 = std::bit_cast<float>(vsrc1.read_lane(wf, lane));
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(sv0 * sv1));
  }
}

VMulI32I24Vop2::VMulI32I24Vop2(const MachineInst *inst)
    : Vop2("v_mul_i32_i24_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VMulI32I24Vop2>()),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VMulI32I24Vop2::execute_impl(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t sv0 = static_cast<int32_t>(src0.read_lane(wf, lane) << 8) >> 8;
    int32_t sv1 = static_cast<int32_t>(vsrc1.read_lane(wf, lane) << 8) >> 8;
    vdst.write_lane(wf, lane, static_cast<uint32_t>(sv0 * sv1));
  }
}

VMulHiI32I24Vop2::VMulHiI32I24Vop2(const MachineInst *inst)
    : Vop2("v_mul_hi_i32_i24_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VMulHiI32I24Vop2>()),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VMulHiI32I24Vop2::execute_impl(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t sv0 = static_cast<int32_t>(src0.read_lane(wf, lane) << 8) >> 8;
    int32_t sv1 = static_cast<int32_t>(vsrc1.read_lane(wf, lane) << 8) >> 8;
    vdst.write_lane(wf, lane, static_cast<uint32_t>((static_cast<int64_t>(sv0) * sv1) >> 32));
  }
}

VMulU32U24Vop2::VMulU32U24Vop2(const MachineInst *inst)
    : Vop2("v_mul_u32_u24_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VMulU32U24Vop2>()),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VMulU32U24Vop2::execute_impl(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane) & 0x00FFFFFFu;
    uint32_t sv1 = vsrc1.read_lane(wf, lane) & 0x00FFFFFFu;
    vdst.write_lane(wf, lane, sv0 * sv1);
  }
}

VMulHiU32U24Vop2::VMulHiU32U24Vop2(const MachineInst *inst)
    : Vop2("v_mul_hi_u32_u24_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VMulHiU32U24Vop2>()),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VMulHiU32U24Vop2::execute_impl(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane) & 0x00FFFFFFu;
    uint32_t sv1 = vsrc1.read_lane(wf, lane) & 0x00FFFFFFu;
    vdst.write_lane(wf, lane, static_cast<uint32_t>((static_cast<uint64_t>(sv0) * sv1) >> 32));
  }
}

VMinF32Vop2::VMinF32Vop2(const MachineInst *inst)
    : Vop2("v_min_f32_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VMinF32Vop2>()),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VMinF32Vop2::execute_impl(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float sv0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float sv1 = std::bit_cast<float>(vsrc1.read_lane(wf, lane));
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(std::fmin(sv0, sv1)));
  }
}

VMaxF32Vop2::VMaxF32Vop2(const MachineInst *inst)
    : Vop2("v_max_f32_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VMaxF32Vop2>()),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VMaxF32Vop2::execute_impl(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float sv0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float sv1 = std::bit_cast<float>(vsrc1.read_lane(wf, lane));
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(std::fmax(sv0, sv1)));
  }
}

VMinI32Vop2::VMinI32Vop2(const MachineInst *inst)
    : Vop2("v_min_i32_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VMinI32Vop2>()),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VMinI32Vop2::execute_impl(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t sv0 = static_cast<int32_t>(src0.read_lane(wf, lane));
    int32_t sv1 = static_cast<int32_t>(vsrc1.read_lane(wf, lane));
    vdst.write_lane(wf, lane, static_cast<uint32_t>(sv0 < sv1 ? sv0 : sv1));
  }
}

VMaxI32Vop2::VMaxI32Vop2(const MachineInst *inst)
    : Vop2("v_max_i32_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VMaxI32Vop2>()),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VMaxI32Vop2::execute_impl(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t sv0 = static_cast<int32_t>(src0.read_lane(wf, lane));
    int32_t sv1 = static_cast<int32_t>(vsrc1.read_lane(wf, lane));
    vdst.write_lane(wf, lane, static_cast<uint32_t>(sv0 > sv1 ? sv0 : sv1));
  }
}

VMinU32Vop2::VMinU32Vop2(const MachineInst *inst)
    : Vop2("v_min_u32_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VMinU32Vop2>()),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VMinU32Vop2::execute_impl(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane);
    uint32_t sv1 = vsrc1.read_lane(wf, lane);
    vdst.write_lane(wf, lane, sv0 < sv1 ? sv0 : sv1);
  }
}

VMaxU32Vop2::VMaxU32Vop2(const MachineInst *inst)
    : Vop2("v_max_u32_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VMaxU32Vop2>()),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VMaxU32Vop2::execute_impl(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane);
    uint32_t sv1 = vsrc1.read_lane(wf, lane);
    vdst.write_lane(wf, lane, sv0 > sv1 ? sv0 : sv1);
  }
}

VLshrrevB32Vop2::VLshrrevB32Vop2(const MachineInst *inst)
    : Vop2("v_lshrrev_b32_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VLshrrevB32Vop2>()),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_NOLDS, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VLshrrevB32Vop2::execute_impl(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane);
    uint32_t sv1 = vsrc1.read_lane(wf, lane);
    vdst.write_lane(wf, lane, sv1 >> (sv0 & 31u));
  }
}

VAshrrevI32Vop2::VAshrrevI32Vop2(const MachineInst *inst)
    : Vop2("v_ashrrev_i32_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VAshrrevI32Vop2>()),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_NOLDS, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VAshrrevI32Vop2::execute_impl(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t sv0 = static_cast<int32_t>(src0.read_lane(wf, lane));
    int32_t sv1 = static_cast<int32_t>(vsrc1.read_lane(wf, lane));
    vdst.write_lane(wf, lane, static_cast<uint32_t>(static_cast<int32_t>(sv1) >> (sv0 & 31)));
  }
}

VLshlrevB32Vop2::VLshlrevB32Vop2(const MachineInst *inst)
    : Vop2("v_lshlrev_b32_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VLshlrevB32Vop2>()),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_NOLDS, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VLshlrevB32Vop2::execute_impl(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane);
    uint32_t sv1 = vsrc1.read_lane(wf, lane);
    vdst.write_lane(wf, lane, sv1 << (sv0 & 31u));
  }
}

VAndB32Vop2::VAndB32Vop2(const MachineInst *inst)
    : Vop2("v_and_b32_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VAndB32Vop2>()),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VAndB32Vop2::execute_impl(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane);
    uint32_t sv1 = vsrc1.read_lane(wf, lane);
    vdst.write_lane(wf, lane, sv0 & sv1);
  }
}

VOrB32Vop2::VOrB32Vop2(const MachineInst *inst)
    : Vop2("v_or_b32_e32", reinterpret_cast<const OpEncoding *>(inst), make_exec_fn<VOrB32Vop2>()),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VOrB32Vop2::execute_impl(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane);
    uint32_t sv1 = vsrc1.read_lane(wf, lane);
    vdst.write_lane(wf, lane, sv0 | sv1);
  }
}

VXorB32Vop2::VXorB32Vop2(const MachineInst *inst)
    : Vop2("v_xor_b32_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VXorB32Vop2>()),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VXorB32Vop2::execute_impl(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane);
    uint32_t sv1 = vsrc1.read_lane(wf, lane);
    vdst.write_lane(wf, lane, sv0 ^ sv1);
  }
}

VMacF32Vop2::VMacF32Vop2(const MachineInst *inst)
    : Vop2("v_mac_f32_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VMacF32Vop2>()),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_[0] = &vdst;
  dst_operands_[0] = &vdst;
  src_operands_[1] = &src0;
  src_operands_[2] = &vsrc1;
  num_src_ = 3;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VMacF32Vop2::execute_impl(amdgpu::Wavefront &wf) { amdgpu::execute_v_mac_f32_vop2(*this, wf); }

VMadmkF32Vop2::VMadmkF32Vop2(const MachineInst *inst)
    : Vop2("v_madmk_f32_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VMadmkF32Vop2>()),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      simm32(32, OperandType::OPR_SIMM32, 0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &simm32;
  src_operands_[2] = &vsrc1;
  num_src_ = 3;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VMadmkF32Vop2::execute_impl(amdgpu::Wavefront &wf) {
  amdgpu::execute_v_madmk_f32_vop2(*this, wf);
}

VMadakF32Vop2::VMadakF32Vop2(const MachineInst *inst)
    : Vop2("v_madak_f32_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VMadakF32Vop2>()),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1),
      simm32(32, OperandType::OPR_SIMM32, 0) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  src_operands_[2] = &simm32;
  num_src_ = 3;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VMadakF32Vop2::execute_impl(amdgpu::Wavefront &wf) {
  amdgpu::execute_v_madak_f32_vop2(*this, wf);
}

VAddCoU32Vop2::VAddCoU32Vop2(const MachineInst *inst)
    : Vop2("v_add_co_u32_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VAddCoU32Vop2>()),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VAddCoU32Vop2::execute_impl(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane);
    uint32_t sv1 = vsrc1.read_lane(wf, lane);
    uint64_t wide = static_cast<uint64_t>(sv0) + static_cast<uint64_t>(sv1);
    vdst.write_lane(wf, lane, static_cast<uint32_t>(wide));
    if (wide > 0xFFFFFFFFULL)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VSubCoU32Vop2::VSubCoU32Vop2(const MachineInst *inst)
    : Vop2("v_sub_co_u32_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VSubCoU32Vop2>()),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VSubCoU32Vop2::execute_impl(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane);
    uint32_t sv1 = vsrc1.read_lane(wf, lane);
    uint64_t wide = static_cast<uint64_t>(sv0) - static_cast<uint64_t>(sv1);
    bool borrow = sv0 < sv1;
    vdst.write_lane(wf, lane, static_cast<uint32_t>(wide));
    if (borrow)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VSubrevCoU32Vop2::VSubrevCoU32Vop2(const MachineInst *inst)
    : Vop2("v_subrev_co_u32_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VSubrevCoU32Vop2>()),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_NOLDS, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VSubrevCoU32Vop2::execute_impl(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane);
    uint32_t sv1 = vsrc1.read_lane(wf, lane);
    uint64_t wide = static_cast<uint64_t>(sv1) - static_cast<uint64_t>(sv0);
    bool borrow = sv1 < sv0;
    vdst.write_lane(wf, lane, static_cast<uint32_t>(wide));
    if (borrow)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VAddcCoU32Vop2::VAddcCoU32Vop2(const MachineInst *inst)
    : Vop2("v_addc_co_u32_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VAddcCoU32Vop2>()),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VAddcCoU32Vop2::execute_impl(amdgpu::Wavefront &wf) {
  amdgpu::execute_v_addc_co_u32_vop2(*this, wf);
}

VSubbCoU32Vop2::VSubbCoU32Vop2(const MachineInst *inst)
    : Vop2("v_subb_co_u32_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VSubbCoU32Vop2>()),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VSubbCoU32Vop2::execute_impl(amdgpu::Wavefront &wf) {
  amdgpu::execute_v_subb_co_u32_vop2(*this, wf);
}

VSubbrevCoU32Vop2::VSubbrevCoU32Vop2(const MachineInst *inst)
    : Vop2("v_subbrev_co_u32_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VSubbrevCoU32Vop2>()),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_NOLDS, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VSubbrevCoU32Vop2::execute_impl(amdgpu::Wavefront &wf) {
  amdgpu::execute_v_subbrev_co_u32_vop2(*this, wf);
}

VAddF16Vop2::VAddF16Vop2(const MachineInst *inst)
    : Vop2("v_add_f16_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VAddF16Vop2>()),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VAddF16Vop2::execute_impl(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float sv0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float sv1 = util::f16_to_f32(static_cast<uint16_t>(vsrc1.read_lane(wf, lane)));
    vdst.write_lane(wf, lane, util::f32_to_f16(sv0 + sv1));
  }
}

VSubF16Vop2::VSubF16Vop2(const MachineInst *inst)
    : Vop2("v_sub_f16_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VSubF16Vop2>()),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VSubF16Vop2::execute_impl(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float sv0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float sv1 = util::f16_to_f32(static_cast<uint16_t>(vsrc1.read_lane(wf, lane)));
    vdst.write_lane(wf, lane, util::f32_to_f16(sv0 - sv1));
  }
}

VSubrevF16Vop2::VSubrevF16Vop2(const MachineInst *inst)
    : Vop2("v_subrev_f16_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VSubrevF16Vop2>()),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC_NOLDS, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VSubrevF16Vop2::execute_impl(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float sv0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float sv1 = util::f16_to_f32(static_cast<uint16_t>(vsrc1.read_lane(wf, lane)));
    vdst.write_lane(wf, lane, util::f32_to_f16(sv1 - sv0));
  }
}

VMulF16Vop2::VMulF16Vop2(const MachineInst *inst)
    : Vop2("v_mul_f16_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VMulF16Vop2>()),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VMulF16Vop2::execute_impl(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float sv0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float sv1 = util::f16_to_f32(static_cast<uint16_t>(vsrc1.read_lane(wf, lane)));
    vdst.write_lane(wf, lane, util::f32_to_f16(sv0 * sv1));
  }
}

VMacF16Vop2::VMacF16Vop2(const MachineInst *inst)
    : Vop2("v_mac_f16_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VMacF16Vop2>()),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_[0] = &vdst;
  dst_operands_[0] = &vdst;
  src_operands_[1] = &src0;
  src_operands_[2] = &vsrc1;
  num_src_ = 3;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VMacF16Vop2::execute_impl(amdgpu::Wavefront &wf) { amdgpu::execute_v_mac_f16_vop2(*this, wf); }

VMadmkF16Vop2::VMadmkF16Vop2(const MachineInst *inst)
    : Vop2("v_madmk_f16_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VMadmkF16Vop2>()),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      simm32(16, OperandType::OPR_SIMM32, 0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &simm32;
  src_operands_[2] = &vsrc1;
  num_src_ = 3;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VMadmkF16Vop2::execute_impl(amdgpu::Wavefront &wf) {
  amdgpu::execute_v_madmk_f16_vop2(*this, wf);
}

VMadakF16Vop2::VMadakF16Vop2(const MachineInst *inst)
    : Vop2("v_madak_f16_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VMadakF16Vop2>()),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1),
      simm32(16, OperandType::OPR_SIMM32, 0) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  src_operands_[2] = &simm32;
  num_src_ = 3;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VMadakF16Vop2::execute_impl(amdgpu::Wavefront &wf) {
  amdgpu::execute_v_madak_f16_vop2(*this, wf);
}

VAddU16Vop2::VAddU16Vop2(const MachineInst *inst)
    : Vop2("v_add_u16_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VAddU16Vop2>()),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VAddU16Vop2::execute_impl(amdgpu::Wavefront &wf) { amdgpu::execute_v_add_u16_vop2(*this, wf); }

VSubU16Vop2::VSubU16Vop2(const MachineInst *inst)
    : Vop2("v_sub_u16_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VSubU16Vop2>()),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VSubU16Vop2::execute_impl(amdgpu::Wavefront &wf) { amdgpu::execute_v_sub_u16_vop2(*this, wf); }

VSubrevU16Vop2::VSubrevU16Vop2(const MachineInst *inst)
    : Vop2("v_subrev_u16_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VSubrevU16Vop2>()),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC_NOLDS, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VSubrevU16Vop2::execute_impl(amdgpu::Wavefront &wf) {
  amdgpu::execute_v_subrev_u16_vop2(*this, wf);
}

VMulLoU16Vop2::VMulLoU16Vop2(const MachineInst *inst)
    : Vop2("v_mul_lo_u16_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VMulLoU16Vop2>()),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VMulLoU16Vop2::execute_impl(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t sv0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t sv1 = static_cast<uint16_t>(vsrc1.read_lane(wf, lane));
    vdst.write_lane(wf, lane, static_cast<uint32_t>(static_cast<uint16_t>(sv0 * sv1)));
  }
}

VLshlrevB16Vop2::VLshlrevB16Vop2(const MachineInst *inst)
    : Vop2("v_lshlrev_b16_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VLshlrevB16Vop2>()),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC_NOLDS, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VLshlrevB16Vop2::execute_impl(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t sv0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t sv1 = static_cast<uint16_t>(vsrc1.read_lane(wf, lane));
    vdst.write_lane(wf, lane, static_cast<uint32_t>(static_cast<uint16_t>(sv1 << (sv0 & 15u))));
  }
}

VLshrrevB16Vop2::VLshrrevB16Vop2(const MachineInst *inst)
    : Vop2("v_lshrrev_b16_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VLshrrevB16Vop2>()),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC_NOLDS, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VLshrrevB16Vop2::execute_impl(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t sv0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t sv1 = static_cast<uint16_t>(vsrc1.read_lane(wf, lane));
    vdst.write_lane(wf, lane, static_cast<uint32_t>(static_cast<uint16_t>(sv1 >> (sv0 & 15u))));
  }
}

VAshrrevI16Vop2::VAshrrevI16Vop2(const MachineInst *inst)
    : Vop2("v_ashrrev_i16_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VAshrrevI16Vop2>()),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC_NOLDS, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VAshrrevI16Vop2::execute_impl(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int16_t sv0 = static_cast<int16_t>(src0.read_lane(wf, lane) & 0xFFFF);
    int16_t sv1 = static_cast<int16_t>(vsrc1.read_lane(wf, lane) & 0xFFFF);
    vdst.write_lane(
        wf, lane,
        static_cast<uint32_t>(static_cast<uint16_t>(static_cast<int16_t>(sv1 >> (sv0 & 15)))));
  }
}

VMaxF16Vop2::VMaxF16Vop2(const MachineInst *inst)
    : Vop2("v_max_f16_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VMaxF16Vop2>()),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VMaxF16Vop2::execute_impl(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float sv0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float sv1 = util::f16_to_f32(static_cast<uint16_t>(vsrc1.read_lane(wf, lane)));
    vdst.write_lane(wf, lane, util::f32_to_f16(std::fmax(sv0, sv1)));
  }
}

VMinF16Vop2::VMinF16Vop2(const MachineInst *inst)
    : Vop2("v_min_f16_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VMinF16Vop2>()),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VMinF16Vop2::execute_impl(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float sv0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float sv1 = util::f16_to_f32(static_cast<uint16_t>(vsrc1.read_lane(wf, lane)));
    vdst.write_lane(wf, lane, util::f32_to_f16(std::fmin(sv0, sv1)));
  }
}

VMaxU16Vop2::VMaxU16Vop2(const MachineInst *inst)
    : Vop2("v_max_u16_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VMaxU16Vop2>()),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VMaxU16Vop2::execute_impl(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t sv0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t sv1 = static_cast<uint16_t>(vsrc1.read_lane(wf, lane));
    vdst.write_lane(wf, lane, static_cast<uint32_t>(sv0 > sv1 ? sv0 : sv1));
  }
}

VMaxI16Vop2::VMaxI16Vop2(const MachineInst *inst)
    : Vop2("v_max_i16_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VMaxI16Vop2>()),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VMaxI16Vop2::execute_impl(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int16_t sv0 = static_cast<int16_t>(src0.read_lane(wf, lane) & 0xFFFF);
    int16_t sv1 = static_cast<int16_t>(vsrc1.read_lane(wf, lane) & 0xFFFF);
    vdst.write_lane(wf, lane, static_cast<uint32_t>(static_cast<uint16_t>(sv0 > sv1 ? sv0 : sv1)));
  }
}

VMinU16Vop2::VMinU16Vop2(const MachineInst *inst)
    : Vop2("v_min_u16_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VMinU16Vop2>()),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VMinU16Vop2::execute_impl(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t sv0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t sv1 = static_cast<uint16_t>(vsrc1.read_lane(wf, lane));
    vdst.write_lane(wf, lane, static_cast<uint32_t>(sv0 < sv1 ? sv0 : sv1));
  }
}

VMinI16Vop2::VMinI16Vop2(const MachineInst *inst)
    : Vop2("v_min_i16_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VMinI16Vop2>()),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VMinI16Vop2::execute_impl(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int16_t sv0 = static_cast<int16_t>(src0.read_lane(wf, lane) & 0xFFFF);
    int16_t sv1 = static_cast<int16_t>(vsrc1.read_lane(wf, lane) & 0xFFFF);
    vdst.write_lane(wf, lane, static_cast<uint32_t>(static_cast<uint16_t>(sv0 < sv1 ? sv0 : sv1)));
  }
}

VLdexpF16Vop2::VLdexpF16Vop2(const MachineInst *inst)
    : Vop2("v_ldexp_f16_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VLdexpF16Vop2>()),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VLdexpF16Vop2::execute_impl(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float sv0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    int32_t sv1_i = static_cast<int32_t>(
        static_cast<int16_t>(static_cast<uint16_t>(vsrc1.read_lane(wf, lane))));
    vdst.write_lane(wf, lane, util::f32_to_f16(std::ldexp(sv0, static_cast<int>(sv1_i))));
  }
}

VAddU32Vop2::VAddU32Vop2(const MachineInst *inst)
    : Vop2("v_add_u32_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VAddU32Vop2>()),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VAddU32Vop2::execute_impl(amdgpu::Wavefront &wf) { amdgpu::execute_v_add_u32_vop2(*this, wf); }

VSubU32Vop2::VSubU32Vop2(const MachineInst *inst)
    : Vop2("v_sub_u32_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VSubU32Vop2>()),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VSubU32Vop2::execute_impl(amdgpu::Wavefront &wf) { amdgpu::execute_v_sub_u32_vop2(*this, wf); }

VSubrevU32Vop2::VSubrevU32Vop2(const MachineInst *inst)
    : Vop2("v_subrev_u32_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VSubrevU32Vop2>()),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_NOLDS, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VSubrevU32Vop2::execute_impl(amdgpu::Wavefront &wf) {
  amdgpu::execute_v_subrev_u32_vop2(*this, wf);
}

VDot2cF32F16Vop2::VDot2cF32F16Vop2(const MachineInst *inst)
    : Vop2("v_dot2c_f32_f16_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VDot2cF32F16Vop2>()),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_[0] = &vdst;
  dst_operands_[0] = &vdst;
  src_operands_[1] = &src0;
  src_operands_[2] = &vsrc1;
  num_src_ = 3;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VDot2cF32F16Vop2::execute_impl(amdgpu::Wavefront &wf) {
  amdgpu::execute_v_dot2c_f32_f16_vop2(*this, wf);
}

VDot2cI32I16Vop2::VDot2cI32I16Vop2(const MachineInst *inst)
    : Vop2("v_dot2c_i32_i16_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VDot2cI32I16Vop2>()),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_[0] = &vdst;
  dst_operands_[0] = &vdst;
  src_operands_[1] = &src0;
  src_operands_[2] = &vsrc1;
  num_src_ = 3;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VDot2cI32I16Vop2::execute_impl(amdgpu::Wavefront &wf) {
  amdgpu::execute_v_dot2c_i32_i16_vop2(*this, wf);
}

VDot4cI32I8Vop2::VDot4cI32I8Vop2(const MachineInst *inst)
    : Vop2("v_dot4c_i32_i8_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VDot4cI32I8Vop2>()),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_[0] = &vdst;
  dst_operands_[0] = &vdst;
  src_operands_[1] = &src0;
  src_operands_[2] = &vsrc1;
  num_src_ = 3;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VDot4cI32I8Vop2::execute_impl(amdgpu::Wavefront &wf) {
  amdgpu::execute_v_dot4c_i32_i8_vop2(*this, wf);
}

VDot8cI32I4Vop2::VDot8cI32I4Vop2(const MachineInst *inst)
    : Vop2("v_dot8c_i32_i4_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VDot8cI32I4Vop2>()),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_[0] = &vdst;
  dst_operands_[0] = &vdst;
  src_operands_[1] = &src0;
  src_operands_[2] = &vsrc1;
  num_src_ = 3;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VDot8cI32I4Vop2::execute_impl(amdgpu::Wavefront &wf) {
  amdgpu::execute_v_dot8c_i32_i4_vop2(*this, wf);
}

VFmacF32Vop2::VFmacF32Vop2(const MachineInst *inst)
    : Vop2("v_fmac_f32_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VFmacF32Vop2>()),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_[0] = &vdst;
  dst_operands_[0] = &vdst;
  src_operands_[1] = &src0;
  src_operands_[2] = &vsrc1;
  num_src_ = 3;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VFmacF32Vop2::execute_impl(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float sv0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float sv1 = std::bit_cast<float>(vsrc1.read_lane(wf, lane));
    vdst.write_lane(
        wf, lane,
        std::bit_cast<uint32_t>(sv0 * sv1 + std::bit_cast<float>(vdst.read_lane(wf, lane))));
  }
}

VPkFmacF16Vop2::VPkFmacF16Vop2(const MachineInst *inst)
    : Vop2("v_pk_fmac_f16_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VPkFmacF16Vop2>()),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_[0] = &vdst;
  dst_operands_[0] = &vdst;
  src_operands_[1] = &src0;
  src_operands_[2] = &vsrc1;
  num_src_ = 3;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VPkFmacF16Vop2::execute_impl(amdgpu::Wavefront &wf) { (void)wf; }

VXnorB32Vop2::VXnorB32Vop2(const MachineInst *inst)
    : Vop2("v_xnor_b32_e32", reinterpret_cast<const OpEncoding *>(inst),
           make_exec_fn<VXnorB32Vop2>()),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_[0] = &vdst;
  src_operands_[0] = &src0;
  src_operands_[1] = &vsrc1;
  num_src_ = 2;
  num_dst_ = 1;
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VXnorB32Vop2::execute_impl(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane);
    uint32_t sv1 = vsrc1.read_lane(wf, lane);
    vdst.write_lane(wf, lane, ~(sv0 ^ sv1));
  }
}

} // namespace cdna2
} // namespace rocjitsu

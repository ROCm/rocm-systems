// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/cdna4/vop2.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"
#include "util/except.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

namespace rocjitsu {
namespace cdna4 {

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

void VAddF32Vop2::execute(amdgpu::Wavefront &wf) {
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

void VSubF32Vop2::execute(amdgpu::Wavefront &wf) {
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
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float sv0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float sv1 = std::bit_cast<float>(vsrc1.read_lane(wf, lane));
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(sv1 - sv0));
  }
}

VFmacF64Vop2::VFmacF64Vop2(const MachineInst *inst)
    : Vop2("v_fmac_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&vdst);
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VFmacF64Vop2::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double sv0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double sv1 = std::bit_cast<double>(vsrc1.read_lane64(wf, lane));
    vdst.write_lane64(
        wf, lane,
        std::bit_cast<uint64_t>(sv0 * sv1 + std::bit_cast<double>(vdst.read_lane64(wf, lane))));
  }
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

void VMulF32Vop2::execute(amdgpu::Wavefront &wf) {
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

void VMinF32Vop2::execute(amdgpu::Wavefront &wf) {
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

void VMaxF32Vop2::execute(amdgpu::Wavefront &wf) {
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

void VMinI32Vop2::execute(amdgpu::Wavefront &wf) {
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

void VMaxI32Vop2::execute(amdgpu::Wavefront &wf) {
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

void VMinU32Vop2::execute(amdgpu::Wavefront &wf) {
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

void VMaxU32Vop2::execute(amdgpu::Wavefront &wf) {
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
    : Vop2("v_lshrrev_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_NOLDS, reinterpret_cast<const OpEncoding *>(inst)->src0),
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
    : Vop2("v_ashrrev_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_NOLDS, reinterpret_cast<const OpEncoding *>(inst)->src0),
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
    : Vop2("v_lshlrev_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_NOLDS, reinterpret_cast<const OpEncoding *>(inst)->src0),
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

void VAndB32Vop2::execute(amdgpu::Wavefront &wf) {
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

void VOrB32Vop2::execute(amdgpu::Wavefront &wf) {
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

void VXorB32Vop2::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane);
    uint32_t sv1 = vsrc1.read_lane(wf, lane);
    vdst.write_lane(wf, lane, sv0 ^ sv1);
  }
}

VDot2cF32Bf16Vop2::VDot2cF32Bf16Vop2(const MachineInst *inst)
    : Vop2("v_dot2c_f32_bf16", reinterpret_cast<const OpEncoding *>(inst)),
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

void VDot2cF32Bf16Vop2::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

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

void VFmamkF32Vop2::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float k = std::bit_cast<float>(simm32_);
    float s2 = std::bit_cast<float>(vsrc1.read_lane(wf, lane));
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(std::fma(s0, k, s2)));
  }
}

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

void VFmaakF32Vop2::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(vsrc1.read_lane(wf, lane));
    float k = std::bit_cast<float>(simm32_);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(std::fma(s0, s1, k)));
  }
}

VAddCoU32Vop2::VAddCoU32Vop2(const MachineInst *inst)
    : Vop2("v_add_co_u32", reinterpret_cast<const OpEncoding *>(inst)),
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

void VAddCoU32Vop2::execute(amdgpu::Wavefront &wf) {
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
    : Vop2("v_sub_co_u32", reinterpret_cast<const OpEncoding *>(inst)),
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

void VSubCoU32Vop2::execute(amdgpu::Wavefront &wf) {
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
    : Vop2("v_subrev_co_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_NOLDS, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VSubrevCoU32Vop2::execute(amdgpu::Wavefront &wf) {
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
    : Vop2("v_addc_co_u32", reinterpret_cast<const OpEncoding *>(inst)),
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

void VAddcCoU32Vop2::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  uint64_t old_vcc = vcc;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane);
    uint32_t sv1 = vsrc1.read_lane(wf, lane);
    uint32_t cin = (old_vcc & (1ULL << lane)) ? 1u : 0u;
    uint64_t wide = static_cast<uint64_t>(sv0) + static_cast<uint64_t>(sv1) + cin;
    vdst.write_lane(wf, lane, static_cast<uint32_t>(wide));
    if (wide > 0xFFFFFFFFULL)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VSubbCoU32Vop2::VSubbCoU32Vop2(const MachineInst *inst)
    : Vop2("v_subb_co_u32", reinterpret_cast<const OpEncoding *>(inst)),
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

void VSubbCoU32Vop2::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  uint64_t old_vcc = vcc;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane);
    uint32_t sv1 = vsrc1.read_lane(wf, lane);
    uint32_t cin = (old_vcc & (1ULL << lane)) ? 1u : 0u;
    uint64_t wide = static_cast<uint64_t>(sv0) - static_cast<uint64_t>(sv1) - cin;
    bool borrow = static_cast<uint64_t>(sv0) < static_cast<uint64_t>(sv1) + cin;
    vdst.write_lane(wf, lane, static_cast<uint32_t>(wide));
    if (borrow)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VSubbrevCoU32Vop2::VSubbrevCoU32Vop2(const MachineInst *inst)
    : Vop2("v_subbrev_co_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_NOLDS, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VSubbrevCoU32Vop2::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  uint64_t old_vcc = vcc;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane);
    uint32_t sv1 = vsrc1.read_lane(wf, lane);
    uint32_t cin = (old_vcc & (1ULL << lane)) ? 1u : 0u;
    uint64_t wide = static_cast<uint64_t>(sv1) - static_cast<uint64_t>(sv0) - cin;
    bool borrow = static_cast<uint64_t>(sv1) < static_cast<uint64_t>(sv0) + cin;
    vdst.write_lane(wf, lane, static_cast<uint32_t>(wide));
    if (borrow)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
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

void VAddF16Vop2::execute(amdgpu::Wavefront &wf) {
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

void VSubF16Vop2::execute(amdgpu::Wavefront &wf) {
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
    : Vop2("v_subrev_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC_NOLDS, reinterpret_cast<const OpEncoding *>(inst)->src0),
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

void VMulF16Vop2::execute(amdgpu::Wavefront &wf) {
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
    : Vop2("v_mac_f16", reinterpret_cast<const OpEncoding *>(inst)),
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

void VMacF16Vop2::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float sv0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float sv1 = util::f16_to_f32(static_cast<uint16_t>(vsrc1.read_lane(wf, lane)));
    vdst.write_lane(wf, lane,
                    util::f32_to_f16(sv0 * sv1 + util::f16_to_f32(static_cast<uint16_t>(
                                                     vdst.read_lane(wf, lane)))));
  }
}

VMadmkF16Vop2::VMadmkF16Vop2(const MachineInst *inst)
    : Vop2("v_madmk_f16", reinterpret_cast<const OpEncoding *>(inst)),
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

void VMadmkF16Vop2::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float k = util::f16_to_f32(static_cast<uint16_t>(simm32_));
    float s2 = util::f16_to_f32(static_cast<uint16_t>(vsrc1.read_lane(wf, lane)));
    vdst.write_lane(wf, lane, util::f32_to_f16(std::fma(s0, k, s2)));
  }
}

VMadakF16Vop2::VMadakF16Vop2(const MachineInst *inst)
    : Vop2("v_madak_f16", reinterpret_cast<const OpEncoding *>(inst)),
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

void VMadakF16Vop2::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(vsrc1.read_lane(wf, lane)));
    float k = util::f16_to_f32(static_cast<uint16_t>(simm32_));
    vdst.write_lane(wf, lane, util::f32_to_f16(std::fma(s0, s1, k)));
  }
}

VAddU16Vop2::VAddU16Vop2(const MachineInst *inst)
    : Vop2("v_add_u16", reinterpret_cast<const OpEncoding *>(inst)),
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

void VAddU16Vop2::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t sv0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t sv1 = static_cast<uint16_t>(vsrc1.read_lane(wf, lane));
    vdst.write_lane(wf, lane, static_cast<uint32_t>(static_cast<uint16_t>(sv0 + sv1)));
  }
}

VSubU16Vop2::VSubU16Vop2(const MachineInst *inst)
    : Vop2("v_sub_u16", reinterpret_cast<const OpEncoding *>(inst)),
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

void VSubU16Vop2::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t sv0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t sv1 = static_cast<uint16_t>(vsrc1.read_lane(wf, lane));
    vdst.write_lane(wf, lane, static_cast<uint32_t>(static_cast<uint16_t>(sv0 - sv1)));
  }
}

VSubrevU16Vop2::VSubrevU16Vop2(const MachineInst *inst)
    : Vop2("v_subrev_u16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC_NOLDS, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VSubrevU16Vop2::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t sv0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t sv1 = static_cast<uint16_t>(vsrc1.read_lane(wf, lane));
    vdst.write_lane(wf, lane, static_cast<uint32_t>(static_cast<uint16_t>(sv1 - sv0)));
  }
}

VMulLoU16Vop2::VMulLoU16Vop2(const MachineInst *inst)
    : Vop2("v_mul_lo_u16", reinterpret_cast<const OpEncoding *>(inst)),
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

void VMulLoU16Vop2::execute(amdgpu::Wavefront &wf) {
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
    : Vop2("v_lshlrev_b16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC_NOLDS, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VLshlrevB16Vop2::execute(amdgpu::Wavefront &wf) {
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
    : Vop2("v_lshrrev_b16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC_NOLDS, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VLshrrevB16Vop2::execute(amdgpu::Wavefront &wf) {
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
    : Vop2("v_ashrrev_i16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC_NOLDS, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VAshrrevI16Vop2::execute(amdgpu::Wavefront &wf) {
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

void VMaxF16Vop2::execute(amdgpu::Wavefront &wf) {
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

void VMinF16Vop2::execute(amdgpu::Wavefront &wf) {
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
    : Vop2("v_max_u16", reinterpret_cast<const OpEncoding *>(inst)),
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

void VMaxU16Vop2::execute(amdgpu::Wavefront &wf) {
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
    : Vop2("v_max_i16", reinterpret_cast<const OpEncoding *>(inst)),
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

void VMaxI16Vop2::execute(amdgpu::Wavefront &wf) {
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
    : Vop2("v_min_u16", reinterpret_cast<const OpEncoding *>(inst)),
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

void VMinU16Vop2::execute(amdgpu::Wavefront &wf) {
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
    : Vop2("v_min_i16", reinterpret_cast<const OpEncoding *>(inst)),
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

void VMinI16Vop2::execute(amdgpu::Wavefront &wf) {
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

void VLdexpF16Vop2::execute(amdgpu::Wavefront &wf) {
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
    : Vop2("v_add_u32", reinterpret_cast<const OpEncoding *>(inst)),
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

void VAddU32Vop2::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane);
    uint32_t sv1 = vsrc1.read_lane(wf, lane);
    vdst.write_lane(wf, lane, sv0 + sv1);
  }
}

VSubU32Vop2::VSubU32Vop2(const MachineInst *inst)
    : Vop2("v_sub_u32", reinterpret_cast<const OpEncoding *>(inst)),
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

void VSubU32Vop2::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane);
    uint32_t sv1 = vsrc1.read_lane(wf, lane);
    vdst.write_lane(wf, lane, sv0 - sv1);
  }
}

VSubrevU32Vop2::VSubrevU32Vop2(const MachineInst *inst)
    : Vop2("v_subrev_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_NOLDS, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const Vop2InstLiteralMachineInst *>(inst)->simm32));
}

void VSubrevU32Vop2::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane);
    uint32_t sv1 = vsrc1.read_lane(wf, lane);
    vdst.write_lane(wf, lane, sv1 - sv0);
  }
}

VDot2cF32F16Vop2::VDot2cF32F16Vop2(const MachineInst *inst)
    : Vop2("v_dot2c_f32_f16", reinterpret_cast<const OpEncoding *>(inst)),
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

void VDot2cF32F16Vop2::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a = src0.read_lane(wf, lane);
    uint32_t b = vsrc1.read_lane(wf, lane);
    int32_t acc = static_cast<int32_t>(vdst.read_lane(wf, lane));
    (void)a;
    (void)b; // dot2c needs F16/I16 support
    vdst.write_lane(wf, lane, static_cast<uint32_t>(acc));
  }
}

VDot2cI32I16Vop2::VDot2cI32I16Vop2(const MachineInst *inst)
    : Vop2("v_dot2c_i32_i16", reinterpret_cast<const OpEncoding *>(inst)),
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

void VDot2cI32I16Vop2::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a = src0.read_lane(wf, lane);
    uint32_t b = vsrc1.read_lane(wf, lane);
    int32_t acc = static_cast<int32_t>(vdst.read_lane(wf, lane));
    (void)a;
    (void)b; // dot2c needs F16/I16 support
    vdst.write_lane(wf, lane, static_cast<uint32_t>(acc));
  }
}

VDot4cI32I8Vop2::VDot4cI32I8Vop2(const MachineInst *inst)
    : Vop2("v_dot4c_i32_i8", reinterpret_cast<const OpEncoding *>(inst)),
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

void VDot4cI32I8Vop2::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a = src0.read_lane(wf, lane);
    uint32_t b = vsrc1.read_lane(wf, lane);
    int32_t acc = static_cast<int32_t>(vdst.read_lane(wf, lane));
    for (int i = 0; i < 4; ++i) {
      int8_t ea = static_cast<int8_t>((a >> (i * 8)) & 0xFF);
      int8_t eb = static_cast<int8_t>((b >> (i * 8)) & 0xFF);
      acc += static_cast<int32_t>(ea) * static_cast<int32_t>(eb);
    }
    vdst.write_lane(wf, lane, static_cast<uint32_t>(acc));
  }
}

VDot8cI32I4Vop2::VDot8cI32I4Vop2(const MachineInst *inst)
    : Vop2("v_dot8c_i32_i4", reinterpret_cast<const OpEncoding *>(inst)),
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

void VDot8cI32I4Vop2::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a = src0.read_lane(wf, lane);
    uint32_t b = vsrc1.read_lane(wf, lane);
    int32_t acc = static_cast<int32_t>(vdst.read_lane(wf, lane));
    for (int i = 0; i < 8; ++i) {
      int32_t ea = static_cast<int32_t>((a >> (i * 4)) & 0xF);
      if (ea & 8)
        ea |= ~0xF;
      int32_t eb = static_cast<int32_t>((b >> (i * 4)) & 0xF);
      if (eb & 8)
        eb |= ~0xF;
      acc += ea * eb;
    }
    vdst.write_lane(wf, lane, static_cast<uint32_t>(acc));
  }
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

void VFmacF32Vop2::execute(amdgpu::Wavefront &wf) {
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
    : Vop2("v_pk_fmac_f16", reinterpret_cast<const OpEncoding *>(inst)),
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

void VPkFmacF16Vop2::execute(amdgpu::Wavefront &wf) { (void)wf; }

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

void VXnorB32Vop2::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane);
    uint32_t sv1 = vsrc1.read_lane(wf, lane);
    vdst.write_lane(wf, lane, ~(sv0 ^ sv1));
  }
}

} // namespace cdna4
} // namespace rocjitsu

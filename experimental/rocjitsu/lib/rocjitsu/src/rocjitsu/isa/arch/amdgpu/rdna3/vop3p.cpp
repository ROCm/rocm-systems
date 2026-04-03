// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/rdna3/vop3p.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"
#include "util/except.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

namespace rocjitsu {
namespace rdna3 {

VPkMadI16Vop3p::VPkMadI16Vop3p(const MachineInst *inst)
    : Vop3p("v_pk_mad_i16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VPkMadI16Vop3p::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t raw0 = src0.read_lane(wf, lane);
    uint32_t raw1 = src1.read_lane(wf, lane);
    uint32_t raw2 = src2.read_lane(wf, lane);
    bool sel0_lo = (inst_.op_sel >> 0) & 1;
    bool sel1_lo = (inst_.op_sel >> 1) & 1;
    bool sel2_lo = (inst_.op_sel >> 2) & 1;
    bool sel0_hi = (inst_.op_sel_hi >> 0) & 1;
    bool sel1_hi = (inst_.op_sel_hi >> 1) & 1;
    bool sel2_hi = inst_.op_sel_hi_2;
    int16_t a_lo = static_cast<int16_t>(sel0_lo ? (raw0 >> 16) : raw0);
    int16_t b_lo = static_cast<int16_t>(sel1_lo ? (raw1 >> 16) : raw1);
    int16_t c_lo = static_cast<int16_t>(sel2_lo ? (raw2 >> 16) : raw2);
    int16_t a_hi = static_cast<int16_t>(sel0_hi ? (raw0 >> 16) : raw0);
    int16_t b_hi = static_cast<int16_t>(sel1_hi ? (raw1 >> 16) : raw1);
    int16_t c_hi = static_cast<int16_t>(sel2_hi ? (raw2 >> 16) : raw2);
    uint16_t rlo = static_cast<uint16_t>(a_lo * b_lo + c_lo);
    uint16_t rhi = static_cast<uint16_t>(a_hi * b_hi + c_hi);
    vdst.write_lane(wf, lane, static_cast<uint32_t>(rlo) | (static_cast<uint32_t>(rhi) << 16));
  }
}

VPkMulLoU16Vop3p::VPkMulLoU16Vop3p(const MachineInst *inst)
    : Vop3p("v_pk_mul_lo_u16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VPkMulLoU16Vop3p::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t raw0 = src0.read_lane(wf, lane);
    uint32_t raw1 = src1.read_lane(wf, lane);
    bool sel0_lo = (inst_.op_sel >> 0) & 1;
    bool sel1_lo = (inst_.op_sel >> 1) & 1;
    bool sel0_hi = (inst_.op_sel_hi >> 0) & 1;
    bool sel1_hi = (inst_.op_sel_hi >> 1) & 1;
    uint16_t a_lo = static_cast<uint16_t>(sel0_lo ? (raw0 >> 16) : raw0);
    uint16_t b_lo = static_cast<uint16_t>(sel1_lo ? (raw1 >> 16) : raw1);
    uint16_t a_hi = static_cast<uint16_t>(sel0_hi ? (raw0 >> 16) : raw0);
    uint16_t b_hi = static_cast<uint16_t>(sel1_hi ? (raw1 >> 16) : raw1);
    uint16_t rlo = static_cast<uint16_t>(a_lo * b_lo);
    uint16_t rhi = static_cast<uint16_t>(a_hi * b_hi);
    vdst.write_lane(wf, lane, static_cast<uint32_t>(rlo) | (static_cast<uint32_t>(rhi) << 16));
  }
}

VPkAddI16Vop3p::VPkAddI16Vop3p(const MachineInst *inst)
    : Vop3p("v_pk_add_i16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VPkAddI16Vop3p::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t raw0 = src0.read_lane(wf, lane);
    uint32_t raw1 = src1.read_lane(wf, lane);
    bool sel0_lo = (inst_.op_sel >> 0) & 1;
    bool sel1_lo = (inst_.op_sel >> 1) & 1;
    bool sel0_hi = (inst_.op_sel_hi >> 0) & 1;
    bool sel1_hi = (inst_.op_sel_hi >> 1) & 1;
    int16_t a_lo = static_cast<int16_t>(sel0_lo ? (raw0 >> 16) : raw0);
    int16_t b_lo = static_cast<int16_t>(sel1_lo ? (raw1 >> 16) : raw1);
    int16_t a_hi = static_cast<int16_t>(sel0_hi ? (raw0 >> 16) : raw0);
    int16_t b_hi = static_cast<int16_t>(sel1_hi ? (raw1 >> 16) : raw1);
    uint16_t rlo = static_cast<uint16_t>(a_lo + b_lo);
    uint16_t rhi = static_cast<uint16_t>(a_hi + b_hi);
    vdst.write_lane(wf, lane, static_cast<uint32_t>(rlo) | (static_cast<uint32_t>(rhi) << 16));
  }
}

VPkSubI16Vop3p::VPkSubI16Vop3p(const MachineInst *inst)
    : Vop3p("v_pk_sub_i16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VPkSubI16Vop3p::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t raw0 = src0.read_lane(wf, lane);
    uint32_t raw1 = src1.read_lane(wf, lane);
    bool sel0_lo = (inst_.op_sel >> 0) & 1;
    bool sel1_lo = (inst_.op_sel >> 1) & 1;
    bool sel0_hi = (inst_.op_sel_hi >> 0) & 1;
    bool sel1_hi = (inst_.op_sel_hi >> 1) & 1;
    int16_t a_lo = static_cast<int16_t>(sel0_lo ? (raw0 >> 16) : raw0);
    int16_t b_lo = static_cast<int16_t>(sel1_lo ? (raw1 >> 16) : raw1);
    int16_t a_hi = static_cast<int16_t>(sel0_hi ? (raw0 >> 16) : raw0);
    int16_t b_hi = static_cast<int16_t>(sel1_hi ? (raw1 >> 16) : raw1);
    uint16_t rlo = static_cast<uint16_t>(a_lo - b_lo);
    uint16_t rhi = static_cast<uint16_t>(a_hi - b_hi);
    vdst.write_lane(wf, lane, static_cast<uint32_t>(rlo) | (static_cast<uint32_t>(rhi) << 16));
  }
}

VPkLshlrevB16Vop3p::VPkLshlrevB16Vop3p(const MachineInst *inst)
    : Vop3p("v_pk_lshlrev_b16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VPkLshlrevB16Vop3p::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t raw0 = src0.read_lane(wf, lane);
    uint32_t raw1 = src1.read_lane(wf, lane);
    bool sel0_lo = (inst_.op_sel >> 0) & 1;
    bool sel1_lo = (inst_.op_sel >> 1) & 1;
    bool sel0_hi = (inst_.op_sel_hi >> 0) & 1;
    bool sel1_hi = (inst_.op_sel_hi >> 1) & 1;
    uint16_t a_lo = static_cast<uint16_t>(sel0_lo ? (raw0 >> 16) : raw0);
    uint16_t b_lo = static_cast<uint16_t>(sel1_lo ? (raw1 >> 16) : raw1);
    uint16_t a_hi = static_cast<uint16_t>(sel0_hi ? (raw0 >> 16) : raw0);
    uint16_t b_hi = static_cast<uint16_t>(sel1_hi ? (raw1 >> 16) : raw1);
    uint16_t rlo = static_cast<uint16_t>(static_cast<uint16_t>(b_lo << (a_lo & 15u)));
    uint16_t rhi = static_cast<uint16_t>(static_cast<uint16_t>(b_hi << (a_hi & 15u)));
    vdst.write_lane(wf, lane, static_cast<uint32_t>(rlo) | (static_cast<uint32_t>(rhi) << 16));
  }
}

VPkLshrrevB16Vop3p::VPkLshrrevB16Vop3p(const MachineInst *inst)
    : Vop3p("v_pk_lshrrev_b16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VPkLshrrevB16Vop3p::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t raw0 = src0.read_lane(wf, lane);
    uint32_t raw1 = src1.read_lane(wf, lane);
    bool sel0_lo = (inst_.op_sel >> 0) & 1;
    bool sel1_lo = (inst_.op_sel >> 1) & 1;
    bool sel0_hi = (inst_.op_sel_hi >> 0) & 1;
    bool sel1_hi = (inst_.op_sel_hi >> 1) & 1;
    uint16_t a_lo = static_cast<uint16_t>(sel0_lo ? (raw0 >> 16) : raw0);
    uint16_t b_lo = static_cast<uint16_t>(sel1_lo ? (raw1 >> 16) : raw1);
    uint16_t a_hi = static_cast<uint16_t>(sel0_hi ? (raw0 >> 16) : raw0);
    uint16_t b_hi = static_cast<uint16_t>(sel1_hi ? (raw1 >> 16) : raw1);
    uint16_t rlo = static_cast<uint16_t>(static_cast<uint16_t>(b_lo >> (a_lo & 15u)));
    uint16_t rhi = static_cast<uint16_t>(static_cast<uint16_t>(b_hi >> (a_hi & 15u)));
    vdst.write_lane(wf, lane, static_cast<uint32_t>(rlo) | (static_cast<uint32_t>(rhi) << 16));
  }
}

VPkAshrrevI16Vop3p::VPkAshrrevI16Vop3p(const MachineInst *inst)
    : Vop3p("v_pk_ashrrev_i16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VPkAshrrevI16Vop3p::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t raw0 = src0.read_lane(wf, lane);
    uint32_t raw1 = src1.read_lane(wf, lane);
    bool sel0_lo = (inst_.op_sel >> 0) & 1;
    bool sel1_lo = (inst_.op_sel >> 1) & 1;
    bool sel0_hi = (inst_.op_sel_hi >> 0) & 1;
    bool sel1_hi = (inst_.op_sel_hi >> 1) & 1;
    int16_t a_lo = static_cast<int16_t>(sel0_lo ? (raw0 >> 16) : raw0);
    int16_t b_lo = static_cast<int16_t>(sel1_lo ? (raw1 >> 16) : raw1);
    int16_t a_hi = static_cast<int16_t>(sel0_hi ? (raw0 >> 16) : raw0);
    int16_t b_hi = static_cast<int16_t>(sel1_hi ? (raw1 >> 16) : raw1);
    uint16_t rlo = static_cast<uint16_t>(static_cast<int16_t>(b_lo >> (a_lo & 15)));
    uint16_t rhi = static_cast<uint16_t>(static_cast<int16_t>(b_hi >> (a_hi & 15)));
    vdst.write_lane(wf, lane, static_cast<uint32_t>(rlo) | (static_cast<uint32_t>(rhi) << 16));
  }
}

VPkMaxI16Vop3p::VPkMaxI16Vop3p(const MachineInst *inst)
    : Vop3p("v_pk_max_i16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VPkMaxI16Vop3p::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t raw0 = src0.read_lane(wf, lane);
    uint32_t raw1 = src1.read_lane(wf, lane);
    bool sel0_lo = (inst_.op_sel >> 0) & 1;
    bool sel1_lo = (inst_.op_sel >> 1) & 1;
    bool sel0_hi = (inst_.op_sel_hi >> 0) & 1;
    bool sel1_hi = (inst_.op_sel_hi >> 1) & 1;
    int16_t a_lo = static_cast<int16_t>(sel0_lo ? (raw0 >> 16) : raw0);
    int16_t b_lo = static_cast<int16_t>(sel1_lo ? (raw1 >> 16) : raw1);
    int16_t a_hi = static_cast<int16_t>(sel0_hi ? (raw0 >> 16) : raw0);
    int16_t b_hi = static_cast<int16_t>(sel1_hi ? (raw1 >> 16) : raw1);
    uint16_t rlo = static_cast<uint16_t>(a_lo > b_lo ? a_lo : b_lo);
    uint16_t rhi = static_cast<uint16_t>(a_hi > b_hi ? a_hi : b_hi);
    vdst.write_lane(wf, lane, static_cast<uint32_t>(rlo) | (static_cast<uint32_t>(rhi) << 16));
  }
}

VPkMinI16Vop3p::VPkMinI16Vop3p(const MachineInst *inst)
    : Vop3p("v_pk_min_i16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VPkMinI16Vop3p::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t raw0 = src0.read_lane(wf, lane);
    uint32_t raw1 = src1.read_lane(wf, lane);
    bool sel0_lo = (inst_.op_sel >> 0) & 1;
    bool sel1_lo = (inst_.op_sel >> 1) & 1;
    bool sel0_hi = (inst_.op_sel_hi >> 0) & 1;
    bool sel1_hi = (inst_.op_sel_hi >> 1) & 1;
    int16_t a_lo = static_cast<int16_t>(sel0_lo ? (raw0 >> 16) : raw0);
    int16_t b_lo = static_cast<int16_t>(sel1_lo ? (raw1 >> 16) : raw1);
    int16_t a_hi = static_cast<int16_t>(sel0_hi ? (raw0 >> 16) : raw0);
    int16_t b_hi = static_cast<int16_t>(sel1_hi ? (raw1 >> 16) : raw1);
    uint16_t rlo = static_cast<uint16_t>(a_lo < b_lo ? a_lo : b_lo);
    uint16_t rhi = static_cast<uint16_t>(a_hi < b_hi ? a_hi : b_hi);
    vdst.write_lane(wf, lane, static_cast<uint32_t>(rlo) | (static_cast<uint32_t>(rhi) << 16));
  }
}

VPkMadU16Vop3p::VPkMadU16Vop3p(const MachineInst *inst)
    : Vop3p("v_pk_mad_u16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VPkMadU16Vop3p::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t raw0 = src0.read_lane(wf, lane);
    uint32_t raw1 = src1.read_lane(wf, lane);
    uint32_t raw2 = src2.read_lane(wf, lane);
    bool sel0_lo = (inst_.op_sel >> 0) & 1;
    bool sel1_lo = (inst_.op_sel >> 1) & 1;
    bool sel2_lo = (inst_.op_sel >> 2) & 1;
    bool sel0_hi = (inst_.op_sel_hi >> 0) & 1;
    bool sel1_hi = (inst_.op_sel_hi >> 1) & 1;
    bool sel2_hi = inst_.op_sel_hi_2;
    uint16_t a_lo = static_cast<uint16_t>(sel0_lo ? (raw0 >> 16) : raw0);
    uint16_t b_lo = static_cast<uint16_t>(sel1_lo ? (raw1 >> 16) : raw1);
    uint16_t c_lo = static_cast<uint16_t>(sel2_lo ? (raw2 >> 16) : raw2);
    uint16_t a_hi = static_cast<uint16_t>(sel0_hi ? (raw0 >> 16) : raw0);
    uint16_t b_hi = static_cast<uint16_t>(sel1_hi ? (raw1 >> 16) : raw1);
    uint16_t c_hi = static_cast<uint16_t>(sel2_hi ? (raw2 >> 16) : raw2);
    uint16_t rlo = static_cast<uint16_t>(a_lo * b_lo + c_lo);
    uint16_t rhi = static_cast<uint16_t>(a_hi * b_hi + c_hi);
    vdst.write_lane(wf, lane, static_cast<uint32_t>(rlo) | (static_cast<uint32_t>(rhi) << 16));
  }
}

VPkAddU16Vop3p::VPkAddU16Vop3p(const MachineInst *inst)
    : Vop3p("v_pk_add_u16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VPkAddU16Vop3p::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t raw0 = src0.read_lane(wf, lane);
    uint32_t raw1 = src1.read_lane(wf, lane);
    bool sel0_lo = (inst_.op_sel >> 0) & 1;
    bool sel1_lo = (inst_.op_sel >> 1) & 1;
    bool sel0_hi = (inst_.op_sel_hi >> 0) & 1;
    bool sel1_hi = (inst_.op_sel_hi >> 1) & 1;
    uint16_t a_lo = static_cast<uint16_t>(sel0_lo ? (raw0 >> 16) : raw0);
    uint16_t b_lo = static_cast<uint16_t>(sel1_lo ? (raw1 >> 16) : raw1);
    uint16_t a_hi = static_cast<uint16_t>(sel0_hi ? (raw0 >> 16) : raw0);
    uint16_t b_hi = static_cast<uint16_t>(sel1_hi ? (raw1 >> 16) : raw1);
    uint16_t rlo = static_cast<uint16_t>(a_lo + b_lo);
    uint16_t rhi = static_cast<uint16_t>(a_hi + b_hi);
    vdst.write_lane(wf, lane, static_cast<uint32_t>(rlo) | (static_cast<uint32_t>(rhi) << 16));
  }
}

VPkSubU16Vop3p::VPkSubU16Vop3p(const MachineInst *inst)
    : Vop3p("v_pk_sub_u16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VPkSubU16Vop3p::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t raw0 = src0.read_lane(wf, lane);
    uint32_t raw1 = src1.read_lane(wf, lane);
    bool sel0_lo = (inst_.op_sel >> 0) & 1;
    bool sel1_lo = (inst_.op_sel >> 1) & 1;
    bool sel0_hi = (inst_.op_sel_hi >> 0) & 1;
    bool sel1_hi = (inst_.op_sel_hi >> 1) & 1;
    uint16_t a_lo = static_cast<uint16_t>(sel0_lo ? (raw0 >> 16) : raw0);
    uint16_t b_lo = static_cast<uint16_t>(sel1_lo ? (raw1 >> 16) : raw1);
    uint16_t a_hi = static_cast<uint16_t>(sel0_hi ? (raw0 >> 16) : raw0);
    uint16_t b_hi = static_cast<uint16_t>(sel1_hi ? (raw1 >> 16) : raw1);
    uint16_t rlo = static_cast<uint16_t>(a_lo - b_lo);
    uint16_t rhi = static_cast<uint16_t>(a_hi - b_hi);
    vdst.write_lane(wf, lane, static_cast<uint32_t>(rlo) | (static_cast<uint32_t>(rhi) << 16));
  }
}

VPkMaxU16Vop3p::VPkMaxU16Vop3p(const MachineInst *inst)
    : Vop3p("v_pk_max_u16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VPkMaxU16Vop3p::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t raw0 = src0.read_lane(wf, lane);
    uint32_t raw1 = src1.read_lane(wf, lane);
    bool sel0_lo = (inst_.op_sel >> 0) & 1;
    bool sel1_lo = (inst_.op_sel >> 1) & 1;
    bool sel0_hi = (inst_.op_sel_hi >> 0) & 1;
    bool sel1_hi = (inst_.op_sel_hi >> 1) & 1;
    uint16_t a_lo = static_cast<uint16_t>(sel0_lo ? (raw0 >> 16) : raw0);
    uint16_t b_lo = static_cast<uint16_t>(sel1_lo ? (raw1 >> 16) : raw1);
    uint16_t a_hi = static_cast<uint16_t>(sel0_hi ? (raw0 >> 16) : raw0);
    uint16_t b_hi = static_cast<uint16_t>(sel1_hi ? (raw1 >> 16) : raw1);
    uint16_t rlo = static_cast<uint16_t>(a_lo > b_lo ? a_lo : b_lo);
    uint16_t rhi = static_cast<uint16_t>(a_hi > b_hi ? a_hi : b_hi);
    vdst.write_lane(wf, lane, static_cast<uint32_t>(rlo) | (static_cast<uint32_t>(rhi) << 16));
  }
}

VPkMinU16Vop3p::VPkMinU16Vop3p(const MachineInst *inst)
    : Vop3p("v_pk_min_u16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VPkMinU16Vop3p::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t raw0 = src0.read_lane(wf, lane);
    uint32_t raw1 = src1.read_lane(wf, lane);
    bool sel0_lo = (inst_.op_sel >> 0) & 1;
    bool sel1_lo = (inst_.op_sel >> 1) & 1;
    bool sel0_hi = (inst_.op_sel_hi >> 0) & 1;
    bool sel1_hi = (inst_.op_sel_hi >> 1) & 1;
    uint16_t a_lo = static_cast<uint16_t>(sel0_lo ? (raw0 >> 16) : raw0);
    uint16_t b_lo = static_cast<uint16_t>(sel1_lo ? (raw1 >> 16) : raw1);
    uint16_t a_hi = static_cast<uint16_t>(sel0_hi ? (raw0 >> 16) : raw0);
    uint16_t b_hi = static_cast<uint16_t>(sel1_hi ? (raw1 >> 16) : raw1);
    uint16_t rlo = static_cast<uint16_t>(a_lo < b_lo ? a_lo : b_lo);
    uint16_t rhi = static_cast<uint16_t>(a_hi < b_hi ? a_hi : b_hi);
    vdst.write_lane(wf, lane, static_cast<uint32_t>(rlo) | (static_cast<uint32_t>(rhi) << 16));
  }
}

VPkFmaF16Vop3p::VPkFmaF16Vop3p(const MachineInst *inst)
    : Vop3p("v_pk_fma_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VPkFmaF16Vop3p::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t raw0 = src0.read_lane(wf, lane);
    uint32_t raw1 = src1.read_lane(wf, lane);
    uint32_t raw2 = src2.read_lane(wf, lane);
    bool sel0_lo = (inst_.op_sel >> 0) & 1;
    bool sel1_lo = (inst_.op_sel >> 1) & 1;
    bool sel2_lo = (inst_.op_sel >> 2) & 1;
    bool sel0_hi = (inst_.op_sel_hi >> 0) & 1;
    bool sel1_hi = (inst_.op_sel_hi >> 1) & 1;
    bool sel2_hi = inst_.op_sel_hi_2;
    float a_lo = util::f16_to_f32(static_cast<uint16_t>(sel0_lo ? (raw0 >> 16) : raw0));
    float b_lo = util::f16_to_f32(static_cast<uint16_t>(sel1_lo ? (raw1 >> 16) : raw1));
    float c_lo = util::f16_to_f32(static_cast<uint16_t>(sel2_lo ? (raw2 >> 16) : raw2));
    float a_hi = util::f16_to_f32(static_cast<uint16_t>(sel0_hi ? (raw0 >> 16) : raw0));
    float b_hi = util::f16_to_f32(static_cast<uint16_t>(sel1_hi ? (raw1 >> 16) : raw1));
    float c_hi = util::f16_to_f32(static_cast<uint16_t>(sel2_hi ? (raw2 >> 16) : raw2));
    if (inst_.neg & 1) {
      a_lo = -a_lo;
    }
    if (inst_.neg & 2) {
      b_lo = -b_lo;
    }
    if (inst_.neg & 4) {
      c_lo = -c_lo;
    }
    if (inst_.neg_hi & 1) {
      a_hi = -a_hi;
    }
    if (inst_.neg_hi & 2) {
      b_hi = -b_hi;
    }
    if (inst_.neg_hi & 4) {
      c_hi = -c_hi;
    }
    float rlo = std::fma(a_lo, b_lo, c_lo);
    float rhi = std::fma(a_hi, b_hi, c_hi);
    vdst.write_lane(wf, lane,
                    util::f32_to_f16(rlo) | (static_cast<uint32_t>(util::f32_to_f16(rhi)) << 16));
  }
}

VPkAddF16Vop3p::VPkAddF16Vop3p(const MachineInst *inst)
    : Vop3p("v_pk_add_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VPkAddF16Vop3p::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t raw0 = src0.read_lane(wf, lane);
    uint32_t raw1 = src1.read_lane(wf, lane);
    bool sel0_lo = (inst_.op_sel >> 0) & 1;
    bool sel1_lo = (inst_.op_sel >> 1) & 1;
    bool sel0_hi = (inst_.op_sel_hi >> 0) & 1;
    bool sel1_hi = (inst_.op_sel_hi >> 1) & 1;
    float a_lo = util::f16_to_f32(static_cast<uint16_t>(sel0_lo ? (raw0 >> 16) : raw0));
    float b_lo = util::f16_to_f32(static_cast<uint16_t>(sel1_lo ? (raw1 >> 16) : raw1));
    float a_hi = util::f16_to_f32(static_cast<uint16_t>(sel0_hi ? (raw0 >> 16) : raw0));
    float b_hi = util::f16_to_f32(static_cast<uint16_t>(sel1_hi ? (raw1 >> 16) : raw1));
    if (inst_.neg & 1) {
      a_lo = -a_lo;
    }
    if (inst_.neg & 2) {
      b_lo = -b_lo;
    }
    if (inst_.neg_hi & 1) {
      a_hi = -a_hi;
    }
    if (inst_.neg_hi & 2) {
      b_hi = -b_hi;
    }
    float rlo = a_lo + b_lo;
    float rhi = a_hi + b_hi;
    vdst.write_lane(wf, lane,
                    util::f32_to_f16(rlo) | (static_cast<uint32_t>(util::f32_to_f16(rhi)) << 16));
  }
}

VPkMulF16Vop3p::VPkMulF16Vop3p(const MachineInst *inst)
    : Vop3p("v_pk_mul_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VPkMulF16Vop3p::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t raw0 = src0.read_lane(wf, lane);
    uint32_t raw1 = src1.read_lane(wf, lane);
    bool sel0_lo = (inst_.op_sel >> 0) & 1;
    bool sel1_lo = (inst_.op_sel >> 1) & 1;
    bool sel0_hi = (inst_.op_sel_hi >> 0) & 1;
    bool sel1_hi = (inst_.op_sel_hi >> 1) & 1;
    float a_lo = util::f16_to_f32(static_cast<uint16_t>(sel0_lo ? (raw0 >> 16) : raw0));
    float b_lo = util::f16_to_f32(static_cast<uint16_t>(sel1_lo ? (raw1 >> 16) : raw1));
    float a_hi = util::f16_to_f32(static_cast<uint16_t>(sel0_hi ? (raw0 >> 16) : raw0));
    float b_hi = util::f16_to_f32(static_cast<uint16_t>(sel1_hi ? (raw1 >> 16) : raw1));
    if (inst_.neg & 1) {
      a_lo = -a_lo;
    }
    if (inst_.neg & 2) {
      b_lo = -b_lo;
    }
    if (inst_.neg_hi & 1) {
      a_hi = -a_hi;
    }
    if (inst_.neg_hi & 2) {
      b_hi = -b_hi;
    }
    float rlo = a_lo * b_lo;
    float rhi = a_hi * b_hi;
    vdst.write_lane(wf, lane,
                    util::f32_to_f16(rlo) | (static_cast<uint32_t>(util::f32_to_f16(rhi)) << 16));
  }
}

VPkMinF16Vop3p::VPkMinF16Vop3p(const MachineInst *inst)
    : Vop3p("v_pk_min_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VPkMinF16Vop3p::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t raw0 = src0.read_lane(wf, lane);
    uint32_t raw1 = src1.read_lane(wf, lane);
    bool sel0_lo = (inst_.op_sel >> 0) & 1;
    bool sel1_lo = (inst_.op_sel >> 1) & 1;
    bool sel0_hi = (inst_.op_sel_hi >> 0) & 1;
    bool sel1_hi = (inst_.op_sel_hi >> 1) & 1;
    float a_lo = util::f16_to_f32(static_cast<uint16_t>(sel0_lo ? (raw0 >> 16) : raw0));
    float b_lo = util::f16_to_f32(static_cast<uint16_t>(sel1_lo ? (raw1 >> 16) : raw1));
    float a_hi = util::f16_to_f32(static_cast<uint16_t>(sel0_hi ? (raw0 >> 16) : raw0));
    float b_hi = util::f16_to_f32(static_cast<uint16_t>(sel1_hi ? (raw1 >> 16) : raw1));
    if (inst_.neg & 1) {
      a_lo = -a_lo;
    }
    if (inst_.neg & 2) {
      b_lo = -b_lo;
    }
    if (inst_.neg_hi & 1) {
      a_hi = -a_hi;
    }
    if (inst_.neg_hi & 2) {
      b_hi = -b_hi;
    }
    float rlo = std::fmin(a_lo, b_lo);
    float rhi = std::fmin(a_hi, b_hi);
    vdst.write_lane(wf, lane,
                    util::f32_to_f16(rlo) | (static_cast<uint32_t>(util::f32_to_f16(rhi)) << 16));
  }
}

VPkMaxF16Vop3p::VPkMaxF16Vop3p(const MachineInst *inst)
    : Vop3p("v_pk_max_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VPkMaxF16Vop3p::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t raw0 = src0.read_lane(wf, lane);
    uint32_t raw1 = src1.read_lane(wf, lane);
    bool sel0_lo = (inst_.op_sel >> 0) & 1;
    bool sel1_lo = (inst_.op_sel >> 1) & 1;
    bool sel0_hi = (inst_.op_sel_hi >> 0) & 1;
    bool sel1_hi = (inst_.op_sel_hi >> 1) & 1;
    float a_lo = util::f16_to_f32(static_cast<uint16_t>(sel0_lo ? (raw0 >> 16) : raw0));
    float b_lo = util::f16_to_f32(static_cast<uint16_t>(sel1_lo ? (raw1 >> 16) : raw1));
    float a_hi = util::f16_to_f32(static_cast<uint16_t>(sel0_hi ? (raw0 >> 16) : raw0));
    float b_hi = util::f16_to_f32(static_cast<uint16_t>(sel1_hi ? (raw1 >> 16) : raw1));
    if (inst_.neg & 1) {
      a_lo = -a_lo;
    }
    if (inst_.neg & 2) {
      b_lo = -b_lo;
    }
    if (inst_.neg_hi & 1) {
      a_hi = -a_hi;
    }
    if (inst_.neg_hi & 2) {
      b_hi = -b_hi;
    }
    float rlo = std::fmax(a_lo, b_lo);
    float rhi = std::fmax(a_hi, b_hi);
    vdst.write_lane(wf, lane,
                    util::f32_to_f16(rlo) | (static_cast<uint32_t>(util::f32_to_f16(rhi)) << 16));
  }
}

VDot2F32F16Vop3p::VDot2F32F16Vop3p(const MachineInst *inst)
    : Vop3p("v_dot2_f32_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VDot2F32F16Vop3p::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t raw0 = src0.read_lane(wf, lane);
    uint32_t raw1 = src1.read_lane(wf, lane);
    float a0 = util::f16_to_f32(static_cast<uint16_t>(raw0));
    float a1 = util::f16_to_f32(static_cast<uint16_t>(raw0 >> 16));
    float b0 = util::f16_to_f32(static_cast<uint16_t>(raw1));
    float b1 = util::f16_to_f32(static_cast<uint16_t>(raw1 >> 16));
    if (inst_.neg & 1) {
      a0 = -a0;
      a1 = -a1;
    }
    if (inst_.neg & 2) {
      b0 = -b0;
      b1 = -b1;
    }
    float acc = std::bit_cast<float>(src2.read_lane(wf, lane));
    if (inst_.neg & 4)
      acc = -acc;
    float result = a0 * b0 + a1 * b1 + acc;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(result));
  }
}

VDot4I32Iu8Vop3p::VDot4I32Iu8Vop3p(const MachineInst *inst)
    : Vop3p("v_dot4_i32_iu8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VDot4I32Iu8Vop3p::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VDot4U32U8Vop3p::VDot4U32U8Vop3p(const MachineInst *inst)
    : Vop3p("v_dot4_u32_u8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VDot4U32U8Vop3p::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t raw0 = src0.read_lane(wf, lane);
    uint32_t raw1 = src1.read_lane(wf, lane);
    uint32_t acc = src2.read_lane(wf, lane);
    uint32_t sum = acc;
    for (int i = 0; i < 4; ++i) {
      uint8_t a = static_cast<uint8_t>((raw0 >> (i * 8)) & 0xFF);
      uint8_t b = static_cast<uint8_t>((raw1 >> (i * 8)) & 0xFF);
      sum += static_cast<uint32_t>(a) * b;
    }
    vdst.write_lane(wf, lane, sum);
  }
}

VDot8I32Iu4Vop3p::VDot8I32Iu4Vop3p(const MachineInst *inst)
    : Vop3p("v_dot8_i32_iu4", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VDot8I32Iu4Vop3p::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VDot8U32U4Vop3p::VDot8U32U4Vop3p(const MachineInst *inst)
    : Vop3p("v_dot8_u32_u4", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VDot8U32U4Vop3p::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t raw0 = src0.read_lane(wf, lane);
    uint32_t raw1 = src1.read_lane(wf, lane);
    uint32_t acc = src2.read_lane(wf, lane);
    uint32_t sum = acc;
    for (int i = 0; i < 8; ++i) {
      uint32_t a = (raw0 >> (i * 4)) & 0xF;
      uint32_t b = (raw1 >> (i * 4)) & 0xF;
      sum += a * b;
    }
    vdst.write_lane(wf, lane, sum);
  }
}

VDot2F32Bf16Vop3p::VDot2F32Bf16Vop3p(const MachineInst *inst)
    : Vop3p("v_dot2_f32_bf16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VDot2F32Bf16Vop3p::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VFmaMixF32Vop3p::VFmaMixF32Vop3p(const MachineInst *inst)
    : Vop3p("v_fma_mix_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VFmaMixF32Vop3p::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VFmaMixloF16Vop3p::VFmaMixloF16Vop3p(const MachineInst *inst)
    : Vop3p("v_fma_mixlo_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VFmaMixloF16Vop3p::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VFmaMixhiF16Vop3p::VFmaMixhiF16Vop3p(const MachineInst *inst)
    : Vop3p("v_fma_mixhi_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VFmaMixhiF16Vop3p::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VWmmaF3216x16x16F16Vop3p::VWmmaF3216x16x16F16Vop3p(const MachineInst *inst)
    : Vop3p("v_wmma_f32_16x16x16_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(256, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(256, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(256, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(256, OperandType::OPR_SRC_VGPR_OR_INLINE,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VWmmaF3216x16x16F16Vop3p::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VWmmaF3216x16x16Bf16Vop3p::VWmmaF3216x16x16Bf16Vop3p(const MachineInst *inst)
    : Vop3p("v_wmma_f32_16x16x16_bf16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(256, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(256, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(256, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(256, OperandType::OPR_SRC_VGPR_OR_INLINE,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VWmmaF3216x16x16Bf16Vop3p::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VWmmaF1616x16x16F16Vop3p::VWmmaF1616x16x16F16Vop3p(const MachineInst *inst)
    : Vop3p("v_wmma_f16_16x16x16_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(256, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(256, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(256, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(256, OperandType::OPR_SRC_VGPR_OR_INLINE,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VWmmaF1616x16x16F16Vop3p::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VWmmaBf1616x16x16Bf16Vop3p::VWmmaBf1616x16x16Bf16Vop3p(const MachineInst *inst)
    : Vop3p("v_wmma_bf16_16x16x16_bf16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(256, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(256, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(256, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(256, OperandType::OPR_SRC_VGPR_OR_INLINE,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VWmmaBf1616x16x16Bf16Vop3p::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VWmmaI3216x16x16Iu8Vop3p::VWmmaI3216x16x16Iu8Vop3p(const MachineInst *inst)
    : Vop3p("v_wmma_i32_16x16x16_iu8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(256, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(128, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(128, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(256, OperandType::OPR_SRC_VGPR_OR_INLINE,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VWmmaI3216x16x16Iu8Vop3p::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VWmmaI3216x16x16Iu4Vop3p::VWmmaI3216x16x16Iu4Vop3p(const MachineInst *inst)
    : Vop3p("v_wmma_i32_16x16x16_iu4", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(256, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(256, OperandType::OPR_SRC_VGPR_OR_INLINE,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VWmmaI3216x16x16Iu4Vop3p::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

} // namespace rdna3
} // namespace rocjitsu

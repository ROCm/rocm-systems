// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/cdna1/vop3p.h"
#include "rocjitsu/isa/arch/amdgpu/cdna1/mfma_exec.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"
#include "util/except.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

namespace rocjitsu {
namespace cdna1 {

VPkMadI16Vop3p::VPkMadI16Vop3p(const MachineInst *inst)
    : Vop3p("v_pk_mad_i16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_NOLIT, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src2) {
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
      src0(32, OperandType::OPR_SRC_NOLIT, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src1) {
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
      src0(32, OperandType::OPR_SRC_NOLIT, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src1) {
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
      src0(32, OperandType::OPR_SRC_NOLIT, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src1) {
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
      src0(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src1) {
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
      src0(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src1) {
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
      src0(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src1) {
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
      src0(32, OperandType::OPR_SRC_NOLIT, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src1) {
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
      src0(32, OperandType::OPR_SRC_NOLIT, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src1) {
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
      src0(32, OperandType::OPR_SRC_NOLIT, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src2) {
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
      src0(32, OperandType::OPR_SRC_NOLIT, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src1) {
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
      src0(32, OperandType::OPR_SRC_NOLIT, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src1) {
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
      src0(32, OperandType::OPR_SRC_NOLIT, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src1) {
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
      src0(32, OperandType::OPR_SRC_NOLIT, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src1) {
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
      src0(32, OperandType::OPR_SRC_NOLIT, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src2) {
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
      src0(32, OperandType::OPR_SRC_NOLIT, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src1) {
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
      src0(32, OperandType::OPR_SRC_NOLIT, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src1) {
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
      src0(32, OperandType::OPR_SRC_NOLIT, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src1) {
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
      src0(32, OperandType::OPR_SRC_NOLIT, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src1) {
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

VMadMixF32Vop3p::VMadMixF32Vop3p(const MachineInst *inst)
    : Vop3p("v_mad_mix_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_NOLIT, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMadMixF32Vop3p::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t raw0 = src0.read_lane(wf, lane);
    uint32_t raw1 = src1.read_lane(wf, lane);
    uint32_t raw2 = src2.read_lane(wf, lane);
    float a, b, c;
    if (inst_.op_sel_hi & 1)
      a = util::f16_to_f32(static_cast<uint16_t>((inst_.op_sel & 1) ? (raw0 >> 16) : raw0));
    else
      a = std::bit_cast<float>(raw0);
    if (inst_.op_sel_hi & 2)
      b = util::f16_to_f32(static_cast<uint16_t>((inst_.op_sel & 2) ? (raw1 >> 16) : raw1));
    else
      b = std::bit_cast<float>(raw1);
    if (inst_.op_sel_hi_2)
      c = util::f16_to_f32(static_cast<uint16_t>((inst_.op_sel & 4) ? (raw2 >> 16) : raw2));
    else
      c = std::bit_cast<float>(raw2);
    if (inst_.neg & 1)
      a = -a;
    if (inst_.neg & 2)
      b = -b;
    if (inst_.neg & 4)
      c = -c;
    float result = a * b + c;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(result));
  }
}

VMadMixloF16Vop3p::VMadMixloF16Vop3p(const MachineInst *inst)
    : Vop3p("v_mad_mixlo_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_NOLIT, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  src_operands_.emplace_back(&vdst);
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMadMixloF16Vop3p::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t raw0 = src0.read_lane(wf, lane);
    uint32_t raw1 = src1.read_lane(wf, lane);
    uint32_t raw2 = src2.read_lane(wf, lane);
    float a, b, c;
    if (inst_.op_sel_hi & 1)
      a = util::f16_to_f32(static_cast<uint16_t>((inst_.op_sel & 1) ? (raw0 >> 16) : raw0));
    else
      a = std::bit_cast<float>(raw0);
    if (inst_.op_sel_hi & 2)
      b = util::f16_to_f32(static_cast<uint16_t>((inst_.op_sel & 2) ? (raw1 >> 16) : raw1));
    else
      b = std::bit_cast<float>(raw1);
    if (inst_.op_sel_hi_2)
      c = util::f16_to_f32(static_cast<uint16_t>((inst_.op_sel & 4) ? (raw2 >> 16) : raw2));
    else
      c = std::bit_cast<float>(raw2);
    if (inst_.neg & 1)
      a = -a;
    if (inst_.neg & 2)
      b = -b;
    if (inst_.neg & 4)
      c = -c;
    float result = a * b + c;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    uint16_t h = util::f32_to_f16(result);
    uint32_t prev = vdst.read_lane(wf, lane);
    vdst.write_lane(wf, lane, (prev & 0xFFFF0000u) | h);
  }
}

VMadMixhiF16Vop3p::VMadMixhiF16Vop3p(const MachineInst *inst)
    : Vop3p("v_mad_mixhi_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_NOLIT, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  src_operands_.emplace_back(&vdst);
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMadMixhiF16Vop3p::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t raw0 = src0.read_lane(wf, lane);
    uint32_t raw1 = src1.read_lane(wf, lane);
    uint32_t raw2 = src2.read_lane(wf, lane);
    float a, b, c;
    if (inst_.op_sel_hi & 1)
      a = util::f16_to_f32(static_cast<uint16_t>((inst_.op_sel & 1) ? (raw0 >> 16) : raw0));
    else
      a = std::bit_cast<float>(raw0);
    if (inst_.op_sel_hi & 2)
      b = util::f16_to_f32(static_cast<uint16_t>((inst_.op_sel & 2) ? (raw1 >> 16) : raw1));
    else
      b = std::bit_cast<float>(raw1);
    if (inst_.op_sel_hi_2)
      c = util::f16_to_f32(static_cast<uint16_t>((inst_.op_sel & 4) ? (raw2 >> 16) : raw2));
    else
      c = std::bit_cast<float>(raw2);
    if (inst_.neg & 1)
      a = -a;
    if (inst_.neg & 2)
      b = -b;
    if (inst_.neg & 4)
      c = -c;
    float result = a * b + c;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    uint16_t h = util::f32_to_f16(result);
    uint32_t prev = vdst.read_lane(wf, lane);
    vdst.write_lane(wf, lane, (prev & 0x0000FFFFu) | (static_cast<uint32_t>(h) << 16));
  }
}

VDot2F32F16Vop3p::VDot2F32F16Vop3p(const MachineInst *inst)
    : Vop3p("v_dot2_f32_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_NOLIT, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src2) {
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

VDot2I32I16Vop3p::VDot2I32I16Vop3p(const MachineInst *inst)
    : Vop3p("v_dot2_i32_i16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_NOLIT, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VDot2I32I16Vop3p::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t raw0 = src0.read_lane(wf, lane);
    uint32_t raw1 = src1.read_lane(wf, lane);
    int16_t a0 = static_cast<int16_t>(raw0);
    int16_t a1 = static_cast<int16_t>(raw0 >> 16);
    int16_t b0 = static_cast<int16_t>(raw1);
    int16_t b1 = static_cast<int16_t>(raw1 >> 16);
    int32_t acc = static_cast<int32_t>(src2.read_lane(wf, lane));
    int32_t result = static_cast<int32_t>(a0) * b0 + static_cast<int32_t>(a1) * b1 + acc;
    if (inst_.clamp)
      result = std::clamp(result, static_cast<int32_t>(0), std::numeric_limits<int32_t>::max());
    vdst.write_lane(wf, lane, static_cast<uint32_t>(result));
  }
}

VDot2U32U16Vop3p::VDot2U32U16Vop3p(const MachineInst *inst)
    : Vop3p("v_dot2_u32_u16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_NOLIT, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VDot2U32U16Vop3p::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t raw0 = src0.read_lane(wf, lane);
    uint32_t raw1 = src1.read_lane(wf, lane);
    uint16_t a0 = static_cast<uint16_t>(raw0);
    uint16_t a1 = static_cast<uint16_t>(raw0 >> 16);
    uint16_t b0 = static_cast<uint16_t>(raw1);
    uint16_t b1 = static_cast<uint16_t>(raw1 >> 16);
    uint32_t acc = src2.read_lane(wf, lane);
    uint32_t result = static_cast<uint32_t>(a0) * b0 + static_cast<uint32_t>(a1) * b1 + acc;
    vdst.write_lane(wf, lane, result);
  }
}

VDot4I32I8Vop3p::VDot4I32I8Vop3p(const MachineInst *inst)
    : Vop3p("v_dot4_i32_i8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_NOLIT, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VDot4I32I8Vop3p::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t raw0 = src0.read_lane(wf, lane);
    uint32_t raw1 = src1.read_lane(wf, lane);
    int32_t acc = static_cast<int32_t>(src2.read_lane(wf, lane));
    int32_t sum = acc;
    for (int i = 0; i < 4; ++i) {
      int8_t a = static_cast<int8_t>((raw0 >> (i * 8)) & 0xFF);
      int8_t b = static_cast<int8_t>((raw1 >> (i * 8)) & 0xFF);
      sum += static_cast<int32_t>(a) * b;
    }
    if (inst_.clamp)
      sum = std::clamp(sum, static_cast<int32_t>(0), std::numeric_limits<int32_t>::max());
    vdst.write_lane(wf, lane, static_cast<uint32_t>(sum));
  }
}

VDot4U32U8Vop3p::VDot4U32U8Vop3p(const MachineInst *inst)
    : Vop3p("v_dot4_u32_u8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_NOLIT, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src2) {
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

VDot8I32I4Vop3p::VDot8I32I4Vop3p(const MachineInst *inst)
    : Vop3p("v_dot8_i32_i4", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_NOLIT, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VDot8I32I4Vop3p::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t raw0 = src0.read_lane(wf, lane);
    uint32_t raw1 = src1.read_lane(wf, lane);
    int32_t acc = static_cast<int32_t>(src2.read_lane(wf, lane));
    int32_t sum = acc;
    for (int i = 0; i < 8; ++i) {
      int32_t a = static_cast<int32_t>((raw0 >> (i * 4)) & 0xF);
      if (a & 0x8)
        a |= ~0xF;
      int32_t b = static_cast<int32_t>((raw1 >> (i * 4)) & 0xF);
      if (b & 0x8)
        b |= ~0xF;
      sum += a * b;
    }
    if (inst_.clamp)
      sum = std::clamp(sum, static_cast<int32_t>(0), std::numeric_limits<int32_t>::max());
    vdst.write_lane(wf, lane, static_cast<uint32_t>(sum));
  }
}

VDot8U32U4Vop3p::VDot8U32U4Vop3p(const MachineInst *inst)
    : Vop3p("v_dot8_u32_u4", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_NOLIT, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src2) {
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

VMfmaF3232x32x1f32Vop3pMfma::VMfmaF3232x32x1f32Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_32x32x1f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(1024, OperandType::OPR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(1024, OperandType::OPR_SRC_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF3232x32x1f32Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA stub: V_MFMA_F32_32X32X1F32
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VMfmaF3216x16x1f32Vop3pMfma::VMfmaF3216x16x1f32Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_16x16x1f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(512, OperandType::OPR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(512, OperandType::OPR_SRC_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF3216x16x1f32Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA stub: V_MFMA_F32_16X16X1F32
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VMfmaF324x4x1f32Vop3pMfma::VMfmaF324x4x1f32Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_4x4x1f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(128, OperandType::OPR_SRC_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF324x4x1f32Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA stub: V_MFMA_F32_4X4X1F32
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VMfmaF3232x32x2f32Vop3pMfma::VMfmaF3232x32x2f32Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_32x32x2f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(512, OperandType::OPR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(512, OperandType::OPR_SRC_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF3232x32x2f32Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA stub: V_MFMA_F32_32X32X2F32
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VMfmaF3216x16x4f32Vop3pMfma::VMfmaF3216x16x4f32Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_16x16x4f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(128, OperandType::OPR_SRC_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF3216x16x4f32Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA stub: V_MFMA_F32_16X16X4F32
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VMfmaF3232x32x4f16Vop3pMfma::VMfmaF3232x32x4f16Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_32x32x4f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(1024, OperandType::OPR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(1024, OperandType::OPR_SRC_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF3232x32x4f16Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA stub: V_MFMA_F32_32X32X4F16
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VMfmaF3216x16x4f16Vop3pMfma::VMfmaF3216x16x4f16Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_16x16x4f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(512, OperandType::OPR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(512, OperandType::OPR_SRC_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF3216x16x4f16Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA stub: V_MFMA_F32_16X16X4F16
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VMfmaF324x4x4f16Vop3pMfma::VMfmaF324x4x4f16Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_4x4x4f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(128, OperandType::OPR_SRC_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF324x4x4f16Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA stub: V_MFMA_F32_4X4X4F16
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VMfmaF3232x32x8f16Vop3pMfma::VMfmaF3232x32x8f16Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_32x32x8f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(512, OperandType::OPR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(512, OperandType::OPR_SRC_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF3232x32x8f16Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA stub: V_MFMA_F32_32X32X8F16
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VMfmaF3216x16x16f16Vop3pMfma::VMfmaF3216x16x16f16Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_16x16x16f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(128, OperandType::OPR_SRC_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF3216x16x16f16Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA stub: V_MFMA_F32_16X16X16F16
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VMfmaI3232x32x4i8Vop3pMfma::VMfmaI3232x32x4i8Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_i32_32x32x4i8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(1024, OperandType::OPR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(1024, OperandType::OPR_SRC_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaI3232x32x4i8Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA stub: V_MFMA_I32_32X32X4I8
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VMfmaI3216x16x4i8Vop3pMfma::VMfmaI3216x16x4i8Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_i32_16x16x4i8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(512, OperandType::OPR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(512, OperandType::OPR_SRC_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaI3216x16x4i8Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA stub: V_MFMA_I32_16X16X4I8
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VMfmaI324x4x4i8Vop3pMfma::VMfmaI324x4x4i8Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_i32_4x4x4i8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(128, OperandType::OPR_SRC_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaI324x4x4i8Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA stub: V_MFMA_I32_4X4X4I8
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VMfmaI3232x32x8i8Vop3pMfma::VMfmaI3232x32x8i8Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_i32_32x32x8i8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(512, OperandType::OPR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(512, OperandType::OPR_SRC_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaI3232x32x8i8Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA stub: V_MFMA_I32_32X32X8I8
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VMfmaI3216x16x16i8Vop3pMfma::VMfmaI3216x16x16i8Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_i32_16x16x16i8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(128, OperandType::OPR_SRC_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaI3216x16x16i8Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA stub: V_MFMA_I32_16X16X16I8
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VAccvgprReadVop3p::VAccvgprReadVop3p(const MachineInst *inst)
    : Vop3p("v_accvgpr_read", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VAccvgprReadVop3p::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    vdst.write_lane(wf, lane, src0.read_lane(wf, lane));
  }
}

VAccvgprWriteVop3p::VAccvgprWriteVop3p(const MachineInst *inst)
    : Vop3p("v_accvgpr_write", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_NOLIT, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VAccvgprWriteVop3p::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    vdst.write_lane(wf, lane, src0.read_lane(wf, lane));
  }
}

VMfmaF3232x32x2bf16Vop3pMfma::VMfmaF3232x32x2bf16Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_32x32x2bf16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(1024, OperandType::OPR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(1024, OperandType::OPR_SRC_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF3232x32x2bf16Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA stub: V_MFMA_F32_32X32X2BF16
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VMfmaF3216x16x2bf16Vop3pMfma::VMfmaF3216x16x2bf16Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_16x16x2bf16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(512, OperandType::OPR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(512, OperandType::OPR_SRC_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF3216x16x2bf16Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA stub: V_MFMA_F32_16X16X2BF16
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VMfmaF324x4x2bf16Vop3pMfma::VMfmaF324x4x2bf16Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_4x4x2bf16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(128, OperandType::OPR_SRC_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF324x4x2bf16Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA stub: V_MFMA_F32_4X4X2BF16
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VMfmaF3232x32x4bf16Vop3pMfma::VMfmaF3232x32x4bf16Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_32x32x4bf16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(512, OperandType::OPR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(512, OperandType::OPR_SRC_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF3232x32x4bf16Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA stub: V_MFMA_F32_32X32X4BF16
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VMfmaF3216x16x8bf16Vop3pMfma::VMfmaF3216x16x8bf16Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_16x16x8bf16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(128, OperandType::OPR_SRC_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF3216x16x8bf16Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA stub: V_MFMA_F32_16X16X8BF16
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

} // namespace cdna1
} // namespace rocjitsu

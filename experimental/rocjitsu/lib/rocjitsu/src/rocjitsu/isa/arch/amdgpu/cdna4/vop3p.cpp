// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/cdna4/vop3p.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/mfma_exec.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"
#include "util/except.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

namespace rocjitsu {
namespace cdna4 {

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
    bool sel2_hi = inst_.pad_14;
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
    bool sel2_hi = inst_.pad_14;
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
    bool sel2_hi = inst_.pad_14;
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

VDot2F32Bf16Vop3p::VDot2F32Bf16Vop3p(const MachineInst *inst)
    : Vop3p("v_dot2_f32_bf16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_NOLIT, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VDot2F32Bf16Vop3p::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VPkMinimum3F16Vop3p::VPkMinimum3F16Vop3p(const MachineInst *inst)
    : Vop3p("v_pk_minimum3_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_NOLIT, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VPkMinimum3F16Vop3p::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VPkMaximum3F16Vop3p::VPkMaximum3F16Vop3p(const MachineInst *inst)
    : Vop3p("v_pk_maximum3_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_NOLIT, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VPkMaximum3F16Vop3p::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
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
    if (inst_.pad_14)
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
    if (inst_.pad_14)
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
    if (inst_.pad_14)
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

VMfmaF3216x16x128F8f6f4Vop3pMfma::VMfmaF3216x16x128F8f6f4Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_16x16x128_f8f6f4", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(256, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(256, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF3216x16x128F8f6f4Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA stub: V_MFMA_F32_16X16X128_F8F6F4
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VMfmaF3232x32x64F8f6f4Vop3pMfma::VMfmaF3232x32x64F8f6f4Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_32x32x64_f8f6f4", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(512, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(256, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(256, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(512, OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF3232x32x64F8f6f4Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA stub: V_MFMA_F32_32X32X64_F8F6F4
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VPkFmaF32Vop3p::VPkFmaF32Vop3p(const MachineInst *inst)
    : Vop3p("v_pk_fma_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_NOLIT, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(64, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VPkFmaF32Vop3p::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint64_t raw0 = src0.read_lane64(wf, lane);
    uint64_t raw1 = src1.read_lane64(wf, lane);
    uint64_t raw2 = src2.read_lane64(wf, lane);
    float a_lo = std::bit_cast<float>(static_cast<uint32_t>(raw0));
    float a_hi = std::bit_cast<float>(static_cast<uint32_t>(raw0 >> 32));
    float b_lo = std::bit_cast<float>(static_cast<uint32_t>(raw1));
    float b_hi = std::bit_cast<float>(static_cast<uint32_t>(raw1 >> 32));
    float c_lo = std::bit_cast<float>(static_cast<uint32_t>(raw2));
    float c_hi = std::bit_cast<float>(static_cast<uint32_t>(raw2 >> 32));
    if (inst_.neg & 1) {
      a_lo = -a_lo;
      a_hi = -a_hi;
    }
    if (inst_.neg & 2) {
      b_lo = -b_lo;
      b_hi = -b_hi;
    }
    if (inst_.neg & 4) {
      c_lo = -c_lo;
      c_hi = -c_hi;
    }
    uint32_t rlo = std::bit_cast<uint32_t>(std::fma(a_lo, b_lo, c_lo));
    uint32_t rhi = std::bit_cast<uint32_t>(std::fma(a_hi, b_hi, c_hi));
    vdst.write_lane64(wf, lane, static_cast<uint64_t>(rlo) | (static_cast<uint64_t>(rhi) << 32));
  }
}

VPkMulF32Vop3p::VPkMulF32Vop3p(const MachineInst *inst)
    : Vop3p("v_pk_mul_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_NOLIT, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VPkMulF32Vop3p::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint64_t raw0 = src0.read_lane64(wf, lane);
    uint64_t raw1 = src1.read_lane64(wf, lane);
    float a_lo = std::bit_cast<float>(static_cast<uint32_t>(raw0));
    float a_hi = std::bit_cast<float>(static_cast<uint32_t>(raw0 >> 32));
    float b_lo = std::bit_cast<float>(static_cast<uint32_t>(raw1));
    float b_hi = std::bit_cast<float>(static_cast<uint32_t>(raw1 >> 32));
    if (inst_.neg & 1) {
      a_lo = -a_lo;
      a_hi = -a_hi;
    }
    if (inst_.neg & 2) {
      b_lo = -b_lo;
      b_hi = -b_hi;
    }
    uint32_t rlo = std::bit_cast<uint32_t>(a_lo * b_lo);
    uint32_t rhi = std::bit_cast<uint32_t>(a_hi * b_hi);
    vdst.write_lane64(wf, lane, static_cast<uint64_t>(rlo) | (static_cast<uint64_t>(rhi) << 32));
  }
}

VPkAddF32Vop3p::VPkAddF32Vop3p(const MachineInst *inst)
    : Vop3p("v_pk_add_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_NOLIT, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VPkAddF32Vop3p::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint64_t raw0 = src0.read_lane64(wf, lane);
    uint64_t raw1 = src1.read_lane64(wf, lane);
    float a_lo = std::bit_cast<float>(static_cast<uint32_t>(raw0));
    float a_hi = std::bit_cast<float>(static_cast<uint32_t>(raw0 >> 32));
    float b_lo = std::bit_cast<float>(static_cast<uint32_t>(raw1));
    float b_hi = std::bit_cast<float>(static_cast<uint32_t>(raw1 >> 32));
    if (inst_.neg & 1) {
      a_lo = -a_lo;
      a_hi = -a_hi;
    }
    if (inst_.neg & 2) {
      b_lo = -b_lo;
      b_hi = -b_hi;
    }
    uint32_t rlo = std::bit_cast<uint32_t>(a_lo + b_lo);
    uint32_t rhi = std::bit_cast<uint32_t>(a_hi + b_hi);
    vdst.write_lane64(wf, lane, static_cast<uint64_t>(rlo) | (static_cast<uint64_t>(rhi) << 32));
  }
}

VPkMovB32Vop3p::VPkMovB32Vop3p(const MachineInst *inst)
    : Vop3p("v_pk_mov_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_NOLIT, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC_SIMPLE, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VPkMovB32Vop3p::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint64_t raw0 = src0.read_lane64(wf, lane);
    uint64_t raw1 = src1.read_lane64(wf, lane);
    uint32_t lo =
        (inst_.op_sel & 1) ? static_cast<uint32_t>(raw0 >> 32) : static_cast<uint32_t>(raw0);
    uint32_t hi =
        (inst_.op_sel & 2) ? static_cast<uint32_t>(raw1 >> 32) : static_cast<uint32_t>(raw1);
    vdst.write_lane64(wf, lane, static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32));
  }
}

VMfmaF3216x16x32Bf16Vop3pMfma::VMfmaF3216x16x32Bf16Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_16x16x32_bf16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF3216x16x32Bf16Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t vb = wf.vgpr_alloc().base;
  uint32_t dst = mfma::dst_base(vb, vdst.encoding_value_);
  uint32_t const_acc;
  uint32_t s2 = mfma::resolve_acc(vb, dst, src2.encoding_value_, const_acc,
                                  [&] { return src2.read_scalar(wf); });
  mfma::exec_f32(cu, 16, 16, 32, 1, 16, dst, mfma::src_base(vb, src0.encoding_value_),
                 mfma::src_base(vb, src1.encoding_value_), s2, mfma::extract_bf16,
                 mfma::extract_bf16, const_acc);
}

VMfmaI3216x16x64I8Vop3pMfma::VMfmaI3216x16x64I8Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_i32_16x16x64_i8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaI3216x16x64I8Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t vb = wf.vgpr_alloc().base;
  uint32_t dst = mfma::dst_base(vb, vdst.encoding_value_);
  uint32_t const_acc;
  uint32_t s2 = mfma::resolve_acc(vb, dst, src2.encoding_value_, const_acc,
                                  [&] { return src2.read_scalar(wf); });
  mfma::exec_i32_i8(cu, 16, 16, 64, 1, dst, mfma::src_base(vb, src0.encoding_value_),
                    mfma::src_base(vb, src1.encoding_value_), s2, const_acc);
}

VMfmaF3232x32x16Bf16Vop3pMfma::VMfmaF3232x32x16Bf16Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_32x32x16_bf16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(512, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(512, OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF3232x32x16Bf16Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t vb = wf.vgpr_alloc().base;
  uint32_t dst = mfma::dst_base(vb, vdst.encoding_value_);
  uint32_t const_acc;
  uint32_t s2 = mfma::resolve_acc(vb, dst, src2.encoding_value_, const_acc,
                                  [&] { return src2.read_scalar(wf); });
  mfma::exec_f32(cu, 32, 32, 16, 1, 16, dst, mfma::src_base(vb, src0.encoding_value_),
                 mfma::src_base(vb, src1.encoding_value_), s2, mfma::extract_bf16,
                 mfma::extract_bf16, const_acc);
}

VMfmaI3232x32x32I8Vop3pMfma::VMfmaI3232x32x32I8Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_i32_32x32x32_i8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(512, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(512, OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaI3232x32x32I8Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t vb = wf.vgpr_alloc().base;
  uint32_t dst = mfma::dst_base(vb, vdst.encoding_value_);
  uint32_t const_acc;
  uint32_t s2 = mfma::resolve_acc(vb, dst, src2.encoding_value_, const_acc,
                                  [&] { return src2.read_scalar(wf); });
  mfma::exec_i32_i8(cu, 32, 32, 32, 1, dst, mfma::src_base(vb, src0.encoding_value_),
                    mfma::src_base(vb, src1.encoding_value_), s2, const_acc);
}

VSmfmacF3216x16x64Bf16Vop3pMfma::VSmfmacF3216x16x64Bf16Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_smfmac_f32_16x16x64_bf16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(256, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VSmfmacF3216x16x64Bf16Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA: V_SMFMAC_F32_16X16X64_BF16 — 16x16x64 BF16→F32
  // D(4 regs/lane) += A * B, functional model
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a_raw = src0.read_lane(wf, lane);
    uint32_t b_raw = src1.read_lane(wf, lane);
    float a0 = util::bf16_to_f32(static_cast<uint16_t>(a_raw));
    float a1 = util::bf16_to_f32(static_cast<uint16_t>(a_raw >> 16));
    float b0 = util::bf16_to_f32(static_cast<uint16_t>(b_raw));
    float b1 = util::bf16_to_f32(static_cast<uint16_t>(b_raw >> 16));
    float dot = a0 * b0 + a1 * b1;
    float acc = std::bit_cast<float>(src2.read_lane(wf, lane));
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(acc + dot));
  }
}

VSmfmacI3216x16x128I8Vop3pMfma::VSmfmacI3216x16x128I8Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_smfmac_i32_16x16x128_i8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(256, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VSmfmacI3216x16x128I8Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA: V_SMFMAC_I32_16X16X128_I8 — 16x16x128 I8→I32
  // D(4 regs/lane) += A * B, functional model
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a_raw = src0.read_lane(wf, lane);
    uint32_t b_raw = src1.read_lane(wf, lane);
    int32_t dot = 0;
    for (int k = 0; k < 8; ++k) {
      int8_t ae = static_cast<int8_t>((a_raw >> (k * 8)) & 0xFF);
      int8_t be = static_cast<int8_t>((b_raw >> (k * 8)) & 0xFF);
      dot += static_cast<int32_t>(ae) * be;
    }
    int32_t acc0 = static_cast<int32_t>(src2.read_lane(wf, lane));
    vdst.write_lane(wf, lane, static_cast<uint32_t>(acc0 + dot));
    // Additional result registers would require cross-lane data
  }
}

VSmfmacF3216x16x128Bf8Bf8Vop3pMfma::VSmfmacF3216x16x128Bf8Bf8Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_smfmac_f32_16x16x128_bf8_bf8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(256, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VSmfmacF3216x16x128Bf8Bf8Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA: V_SMFMAC_F32_16X16X128_BF8_BF8 — 16x16x128 BF8_BF8→F32
  // D(4 regs/lane) += A * B, functional model
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a_raw = src0.read_lane(wf, lane);
    uint32_t b_raw = src1.read_lane(wf, lane);
    float dot = 0.0f;
    for (int k = 0; k < 4; ++k) {
      float ae = util::bf8_e5m2_to_f32(static_cast<uint8_t>((a_raw >> (k * 8)) & 0xFF));
      float be = util::bf8_e5m2_to_f32(static_cast<uint8_t>((b_raw >> (k * 8)) & 0xFF));
      dot += ae * be;
    }
    float acc = std::bit_cast<float>(src2.read_lane(wf, lane));
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(acc + dot));
  }
}

VSmfmacF3216x16x128Bf8Fp8Vop3pMfma::VSmfmacF3216x16x128Bf8Fp8Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_smfmac_f32_16x16x128_bf8_fp8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(256, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VSmfmacF3216x16x128Bf8Fp8Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA: V_SMFMAC_F32_16X16X128_BF8_FP8 — 16x16x128 BF8_FP8→F32
  // D(4 regs/lane) += A * B, functional model
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a_raw = src0.read_lane(wf, lane);
    uint32_t b_raw = src1.read_lane(wf, lane);
    float dot = 0.0f;
    for (int k = 0; k < 4; ++k) {
      float ae = util::bf8_e5m2_to_f32(static_cast<uint8_t>((a_raw >> (k * 8)) & 0xFF));
      float be = util::fp8_e4m3_to_f32(static_cast<uint8_t>((b_raw >> (k * 8)) & 0xFF));
      dot += ae * be;
    }
    float acc = std::bit_cast<float>(src2.read_lane(wf, lane));
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(acc + dot));
  }
}

VSmfmacF3216x16x128Fp8Bf8Vop3pMfma::VSmfmacF3216x16x128Fp8Bf8Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_smfmac_f32_16x16x128_fp8_bf8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(256, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VSmfmacF3216x16x128Fp8Bf8Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA: V_SMFMAC_F32_16X16X128_FP8_BF8 — 16x16x128 FP8_BF8→F32
  // D(4 regs/lane) += A * B, functional model
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a_raw = src0.read_lane(wf, lane);
    uint32_t b_raw = src1.read_lane(wf, lane);
    float dot = 0.0f;
    for (int k = 0; k < 4; ++k) {
      float ae = util::fp8_e4m3_to_f32(static_cast<uint8_t>((a_raw >> (k * 8)) & 0xFF));
      float be = util::bf8_e5m2_to_f32(static_cast<uint8_t>((b_raw >> (k * 8)) & 0xFF));
      dot += ae * be;
    }
    float acc = std::bit_cast<float>(src2.read_lane(wf, lane));
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(acc + dot));
  }
}

VMfmaF3232x32x12bF32Vop3pMfma::VMfmaF3232x32x12bF32Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_32x32x1_2b_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(1024, OperandType::OPR_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(1024, OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF3232x32x12bF32Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t vb = wf.vgpr_alloc().base;
  uint32_t dst = mfma::dst_base(vb, vdst.encoding_value_);
  uint32_t const_acc;
  uint32_t s2 = mfma::resolve_acc(vb, dst, src2.encoding_value_, const_acc,
                                  [&] { return src2.read_scalar(wf); });
  mfma::exec_f32(cu, 32, 32, 1, 2, 32, dst, mfma::src_base(vb, src0.encoding_value_),
                 mfma::src_base(vb, src1.encoding_value_), s2, mfma::extract_f32, mfma::extract_f32,
                 const_acc);
}

VMfmaF3216x16x14bF32Vop3pMfma::VMfmaF3216x16x14bF32Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_16x16x1_4b_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(512, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(512, OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF3216x16x14bF32Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t vb = wf.vgpr_alloc().base;
  uint32_t dst = mfma::dst_base(vb, vdst.encoding_value_);
  uint32_t const_acc;
  uint32_t s2 = mfma::resolve_acc(vb, dst, src2.encoding_value_, const_acc,
                                  [&] { return src2.read_scalar(wf); });
  mfma::exec_f32(cu, 16, 16, 1, 4, 32, dst, mfma::src_base(vb, src0.encoding_value_),
                 mfma::src_base(vb, src1.encoding_value_), s2, mfma::extract_f32, mfma::extract_f32,
                 const_acc);
}

VMfmaF324x4x116bF32Vop3pMfma::VMfmaF324x4x116bF32Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_4x4x1_16b_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF324x4x116bF32Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t vb = wf.vgpr_alloc().base;
  uint32_t dst = mfma::dst_base(vb, vdst.encoding_value_);
  uint32_t const_acc;
  uint32_t s2 = mfma::resolve_acc(vb, dst, src2.encoding_value_, const_acc,
                                  [&] { return src2.read_scalar(wf); });
  mfma::exec_f32(cu, 4, 4, 1, 16, 32, dst, mfma::src_base(vb, src0.encoding_value_),
                 mfma::src_base(vb, src1.encoding_value_), s2, mfma::extract_f32, mfma::extract_f32,
                 const_acc);
}

VSmfmacF3216x16x128Fp8Fp8Vop3pMfma::VSmfmacF3216x16x128Fp8Fp8Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_smfmac_f32_16x16x128_fp8_fp8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(256, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VSmfmacF3216x16x128Fp8Fp8Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA: V_SMFMAC_F32_16X16X128_FP8_FP8 — 16x16x128 FP8_FP8→F32
  // D(4 regs/lane) += A * B, functional model
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a_raw = src0.read_lane(wf, lane);
    uint32_t b_raw = src1.read_lane(wf, lane);
    float dot = 0.0f;
    for (int k = 0; k < 4; ++k) {
      float ae = util::fp8_e4m3_to_f32(static_cast<uint8_t>((a_raw >> (k * 8)) & 0xFF));
      float be = util::fp8_e4m3_to_f32(static_cast<uint8_t>((b_raw >> (k * 8)) & 0xFF));
      dot += ae * be;
    }
    float acc = std::bit_cast<float>(src2.read_lane(wf, lane));
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(acc + dot));
  }
}

VMfmaF3232x32x2F32Vop3pMfma::VMfmaF3232x32x2F32Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_32x32x2_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(512, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(512, OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF3232x32x2F32Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t vb = wf.vgpr_alloc().base;
  uint32_t dst = mfma::dst_base(vb, vdst.encoding_value_);
  uint32_t const_acc;
  uint32_t s2 = mfma::resolve_acc(vb, dst, src2.encoding_value_, const_acc,
                                  [&] { return src2.read_scalar(wf); });
  mfma::exec_f32(cu, 32, 32, 2, 1, 32, dst, mfma::src_base(vb, src0.encoding_value_),
                 mfma::src_base(vb, src1.encoding_value_), s2, mfma::extract_f32, mfma::extract_f32,
                 const_acc);
}

VMfmaF3216x16x4F32Vop3pMfma::VMfmaF3216x16x4F32Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_16x16x4_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF3216x16x4F32Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t vb = wf.vgpr_alloc().base;
  uint32_t dst = mfma::dst_base(vb, vdst.encoding_value_);
  uint32_t const_acc;
  uint32_t s2 = mfma::resolve_acc(vb, dst, src2.encoding_value_, const_acc,
                                  [&] { return src2.read_scalar(wf); });
  mfma::exec_f32(cu, 16, 16, 4, 1, 32, dst, mfma::src_base(vb, src0.encoding_value_),
                 mfma::src_base(vb, src1.encoding_value_), s2, mfma::extract_f32, mfma::extract_f32,
                 const_acc);
}

VSmfmacF3232x32x32Bf16Vop3pMfma::VSmfmacF3232x32x32Bf16Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_smfmac_f32_32x32x32_bf16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(512, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(256, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VSmfmacF3232x32x32Bf16Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA: V_SMFMAC_F32_32X32X32_BF16 — 32x32x32 BF16→F32
  // D(16 regs/lane) += A * B, functional model
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a_raw = src0.read_lane(wf, lane);
    uint32_t b_raw = src1.read_lane(wf, lane);
    float a0 = util::bf16_to_f32(static_cast<uint16_t>(a_raw));
    float a1 = util::bf16_to_f32(static_cast<uint16_t>(a_raw >> 16));
    float b0 = util::bf16_to_f32(static_cast<uint16_t>(b_raw));
    float b1 = util::bf16_to_f32(static_cast<uint16_t>(b_raw >> 16));
    float dot = a0 * b0 + a1 * b1;
    float acc = std::bit_cast<float>(src2.read_lane(wf, lane));
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(acc + dot));
  }
}

VSmfmacI3232x32x64I8Vop3pMfma::VSmfmacI3232x32x64I8Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_smfmac_i32_32x32x64_i8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(512, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(256, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VSmfmacI3232x32x64I8Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA: V_SMFMAC_I32_32X32X64_I8 — 32x32x64 I8→I32
  // D(16 regs/lane) += A * B, functional model
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a_raw = src0.read_lane(wf, lane);
    uint32_t b_raw = src1.read_lane(wf, lane);
    int32_t dot = 0;
    for (int k = 0; k < 8; ++k) {
      int8_t ae = static_cast<int8_t>((a_raw >> (k * 8)) & 0xFF);
      int8_t be = static_cast<int8_t>((b_raw >> (k * 8)) & 0xFF);
      dot += static_cast<int32_t>(ae) * be;
    }
    int32_t acc0 = static_cast<int32_t>(src2.read_lane(wf, lane));
    vdst.write_lane(wf, lane, static_cast<uint32_t>(acc0 + dot));
    // Additional result registers would require cross-lane data
  }
}

VMfmaF3232x32x42bF16Vop3pMfma::VMfmaF3232x32x42bF16Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_32x32x4_2b_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(1024, OperandType::OPR_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(1024, OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF3232x32x42bF16Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t vb = wf.vgpr_alloc().base;
  uint32_t dst = mfma::dst_base(vb, vdst.encoding_value_);
  uint32_t const_acc;
  uint32_t s2 = mfma::resolve_acc(vb, dst, src2.encoding_value_, const_acc,
                                  [&] { return src2.read_scalar(wf); });
  mfma::exec_f32(cu, 32, 32, 4, 2, 16, dst, mfma::src_base(vb, src0.encoding_value_),
                 mfma::src_base(vb, src1.encoding_value_), s2, mfma::extract_f16, mfma::extract_f16,
                 const_acc);
}

VMfmaF3216x16x44bF16Vop3pMfma::VMfmaF3216x16x44bF16Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_16x16x4_4b_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(512, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(512, OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF3216x16x44bF16Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t vb = wf.vgpr_alloc().base;
  uint32_t dst = mfma::dst_base(vb, vdst.encoding_value_);
  uint32_t const_acc;
  uint32_t s2 = mfma::resolve_acc(vb, dst, src2.encoding_value_, const_acc,
                                  [&] { return src2.read_scalar(wf); });
  mfma::exec_f32(cu, 16, 16, 4, 4, 16, dst, mfma::src_base(vb, src0.encoding_value_),
                 mfma::src_base(vb, src1.encoding_value_), s2, mfma::extract_f16, mfma::extract_f16,
                 const_acc);
}

VMfmaF324x4x416bF16Vop3pMfma::VMfmaF324x4x416bF16Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_4x4x4_16b_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF324x4x416bF16Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t vb = wf.vgpr_alloc().base;
  uint32_t dst = mfma::dst_base(vb, vdst.encoding_value_);
  uint32_t const_acc;
  uint32_t s2 = mfma::resolve_acc(vb, dst, src2.encoding_value_, const_acc,
                                  [&] { return src2.read_scalar(wf); });
  mfma::exec_f32(cu, 4, 4, 4, 16, 16, dst, mfma::src_base(vb, src0.encoding_value_),
                 mfma::src_base(vb, src1.encoding_value_), s2, mfma::extract_f16, mfma::extract_f16,
                 const_acc);
}

VSmfmacF3232x32x64Bf8Bf8Vop3pMfma::VSmfmacF3232x32x64Bf8Bf8Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_smfmac_f32_32x32x64_bf8_bf8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(512, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(256, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VSmfmacF3232x32x64Bf8Bf8Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA: V_SMFMAC_F32_32X32X64_BF8_BF8 — 32x32x64 BF8_BF8→F32
  // D(16 regs/lane) += A * B, functional model
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a_raw = src0.read_lane(wf, lane);
    uint32_t b_raw = src1.read_lane(wf, lane);
    float dot = 0.0f;
    for (int k = 0; k < 4; ++k) {
      float ae = util::bf8_e5m2_to_f32(static_cast<uint8_t>((a_raw >> (k * 8)) & 0xFF));
      float be = util::bf8_e5m2_to_f32(static_cast<uint8_t>((b_raw >> (k * 8)) & 0xFF));
      dot += ae * be;
    }
    float acc = std::bit_cast<float>(src2.read_lane(wf, lane));
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(acc + dot));
  }
}

VMfmaF3232x32x8F16Vop3pMfma::VMfmaF3232x32x8F16Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_32x32x8_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(512, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(512, OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF3232x32x8F16Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t vb = wf.vgpr_alloc().base;
  uint32_t dst = mfma::dst_base(vb, vdst.encoding_value_);
  uint32_t const_acc;
  uint32_t s2 = mfma::resolve_acc(vb, dst, src2.encoding_value_, const_acc,
                                  [&] { return src2.read_scalar(wf); });
  mfma::exec_f32(cu, 32, 32, 8, 1, 16, dst, mfma::src_base(vb, src0.encoding_value_),
                 mfma::src_base(vb, src1.encoding_value_), s2, mfma::extract_f16, mfma::extract_f16,
                 const_acc);
}

VMfmaF3216x16x16F16Vop3pMfma::VMfmaF3216x16x16F16Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_16x16x16_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF3216x16x16F16Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t vb = wf.vgpr_alloc().base;
  uint32_t dst = mfma::dst_base(vb, vdst.encoding_value_);
  uint32_t const_acc;
  uint32_t s2 = mfma::resolve_acc(vb, dst, src2.encoding_value_, const_acc,
                                  [&] { return src2.read_scalar(wf); });
  mfma::exec_f32(cu, 16, 16, 16, 1, 16, dst, mfma::src_base(vb, src0.encoding_value_),
                 mfma::src_base(vb, src1.encoding_value_), s2, mfma::extract_f16, mfma::extract_f16,
                 const_acc);
}

VSmfmacF3232x32x64Bf8Fp8Vop3pMfma::VSmfmacF3232x32x64Bf8Fp8Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_smfmac_f32_32x32x64_bf8_fp8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(512, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(256, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VSmfmacF3232x32x64Bf8Fp8Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA: V_SMFMAC_F32_32X32X64_BF8_FP8 — 32x32x64 BF8_FP8→F32
  // D(16 regs/lane) += A * B, functional model
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a_raw = src0.read_lane(wf, lane);
    uint32_t b_raw = src1.read_lane(wf, lane);
    float dot = 0.0f;
    for (int k = 0; k < 4; ++k) {
      float ae = util::bf8_e5m2_to_f32(static_cast<uint8_t>((a_raw >> (k * 8)) & 0xFF));
      float be = util::fp8_e4m3_to_f32(static_cast<uint8_t>((b_raw >> (k * 8)) & 0xFF));
      dot += ae * be;
    }
    float acc = std::bit_cast<float>(src2.read_lane(wf, lane));
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(acc + dot));
  }
}

VSmfmacF3232x32x64Fp8Bf8Vop3pMfma::VSmfmacF3232x32x64Fp8Bf8Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_smfmac_f32_32x32x64_fp8_bf8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(512, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(256, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VSmfmacF3232x32x64Fp8Bf8Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA: V_SMFMAC_F32_32X32X64_FP8_BF8 — 32x32x64 FP8_BF8→F32
  // D(16 regs/lane) += A * B, functional model
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a_raw = src0.read_lane(wf, lane);
    uint32_t b_raw = src1.read_lane(wf, lane);
    float dot = 0.0f;
    for (int k = 0; k < 4; ++k) {
      float ae = util::fp8_e4m3_to_f32(static_cast<uint8_t>((a_raw >> (k * 8)) & 0xFF));
      float be = util::bf8_e5m2_to_f32(static_cast<uint8_t>((b_raw >> (k * 8)) & 0xFF));
      dot += ae * be;
    }
    float acc = std::bit_cast<float>(src2.read_lane(wf, lane));
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(acc + dot));
  }
}

VMfmaI3232x32x42bI8Vop3pMfma::VMfmaI3232x32x42bI8Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_i32_32x32x4_2b_i8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(1024, OperandType::OPR_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(1024, OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaI3232x32x42bI8Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t vb = wf.vgpr_alloc().base;
  uint32_t dst = mfma::dst_base(vb, vdst.encoding_value_);
  uint32_t const_acc;
  uint32_t s2 = mfma::resolve_acc(vb, dst, src2.encoding_value_, const_acc,
                                  [&] { return src2.read_scalar(wf); });
  mfma::exec_i32_i8(cu, 32, 32, 4, 2, dst, mfma::src_base(vb, src0.encoding_value_),
                    mfma::src_base(vb, src1.encoding_value_), s2, const_acc);
}

VMfmaI3216x16x44bI8Vop3pMfma::VMfmaI3216x16x44bI8Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_i32_16x16x4_4b_i8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(512, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(512, OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaI3216x16x44bI8Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t vb = wf.vgpr_alloc().base;
  uint32_t dst = mfma::dst_base(vb, vdst.encoding_value_);
  uint32_t const_acc;
  uint32_t s2 = mfma::resolve_acc(vb, dst, src2.encoding_value_, const_acc,
                                  [&] { return src2.read_scalar(wf); });
  mfma::exec_i32_i8(cu, 16, 16, 4, 4, dst, mfma::src_base(vb, src0.encoding_value_),
                    mfma::src_base(vb, src1.encoding_value_), s2, const_acc);
}

VMfmaI324x4x416bI8Vop3pMfma::VMfmaI324x4x416bI8Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_i32_4x4x4_16b_i8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaI324x4x416bI8Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t vb = wf.vgpr_alloc().base;
  uint32_t dst = mfma::dst_base(vb, vdst.encoding_value_);
  uint32_t const_acc;
  uint32_t s2 = mfma::resolve_acc(vb, dst, src2.encoding_value_, const_acc,
                                  [&] { return src2.read_scalar(wf); });
  mfma::exec_i32_i8(cu, 4, 4, 4, 16, dst, mfma::src_base(vb, src0.encoding_value_),
                    mfma::src_base(vb, src1.encoding_value_), s2, const_acc);
}

VSmfmacF3232x32x64Fp8Fp8Vop3pMfma::VSmfmacF3232x32x64Fp8Fp8Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_smfmac_f32_32x32x64_fp8_fp8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(512, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(256, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VSmfmacF3232x32x64Fp8Fp8Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA: V_SMFMAC_F32_32X32X64_FP8_FP8 — 32x32x64 FP8_FP8→F32
  // D(16 regs/lane) += A * B, functional model
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a_raw = src0.read_lane(wf, lane);
    uint32_t b_raw = src1.read_lane(wf, lane);
    float dot = 0.0f;
    for (int k = 0; k < 4; ++k) {
      float ae = util::fp8_e4m3_to_f32(static_cast<uint8_t>((a_raw >> (k * 8)) & 0xFF));
      float be = util::fp8_e4m3_to_f32(static_cast<uint8_t>((b_raw >> (k * 8)) & 0xFF));
      dot += ae * be;
    }
    float acc = std::bit_cast<float>(src2.read_lane(wf, lane));
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(acc + dot));
  }
}

VMfmaF3216x16x32F16Vop3pMfma::VMfmaF3216x16x32F16Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_16x16x32_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF3216x16x32F16Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t vb = wf.vgpr_alloc().base;
  uint32_t dst = mfma::dst_base(vb, vdst.encoding_value_);
  uint32_t const_acc;
  uint32_t s2 = mfma::resolve_acc(vb, dst, src2.encoding_value_, const_acc,
                                  [&] { return src2.read_scalar(wf); });
  mfma::exec_f32(cu, 16, 16, 32, 1, 16, dst, mfma::src_base(vb, src0.encoding_value_),
                 mfma::src_base(vb, src1.encoding_value_), s2, mfma::extract_f16, mfma::extract_f16,
                 const_acc);
}

VMfmaF3232x32x16F16Vop3pMfma::VMfmaF3232x32x16F16Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_32x32x16_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(512, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(512, OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF3232x32x16F16Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t vb = wf.vgpr_alloc().base;
  uint32_t dst = mfma::dst_base(vb, vdst.encoding_value_);
  uint32_t const_acc;
  uint32_t s2 = mfma::resolve_acc(vb, dst, src2.encoding_value_, const_acc,
                                  [&] { return src2.read_scalar(wf); });
  mfma::exec_f32(cu, 32, 32, 16, 1, 16, dst, mfma::src_base(vb, src0.encoding_value_),
                 mfma::src_base(vb, src1.encoding_value_), s2, mfma::extract_f16, mfma::extract_f16,
                 const_acc);
}

VMfmaI3232x32x16I8Vop3pMfma::VMfmaI3232x32x16I8Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_i32_32x32x16_i8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(512, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(512, OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaI3232x32x16I8Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t vb = wf.vgpr_alloc().base;
  uint32_t dst = mfma::dst_base(vb, vdst.encoding_value_);
  uint32_t const_acc;
  uint32_t s2 = mfma::resolve_acc(vb, dst, src2.encoding_value_, const_acc,
                                  [&] { return src2.read_scalar(wf); });
  mfma::exec_i32_i8(cu, 32, 32, 16, 1, dst, mfma::src_base(vb, src0.encoding_value_),
                    mfma::src_base(vb, src1.encoding_value_), s2, const_acc);
}

VMfmaI3216x16x32I8Vop3pMfma::VMfmaI3216x16x32I8Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_i32_16x16x32_i8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaI3216x16x32I8Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t vb = wf.vgpr_alloc().base;
  uint32_t dst = mfma::dst_base(vb, vdst.encoding_value_);
  uint32_t const_acc;
  uint32_t s2 = mfma::resolve_acc(vb, dst, src2.encoding_value_, const_acc,
                                  [&] { return src2.read_scalar(wf); });
  mfma::exec_i32_i8(cu, 16, 16, 32, 1, dst, mfma::src_base(vb, src0.encoding_value_),
                    mfma::src_base(vb, src1.encoding_value_), s2, const_acc);
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

VSmfmacF3216x16x64F16Vop3pMfma::VSmfmacF3216x16x64F16Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_smfmac_f32_16x16x64_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(256, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VSmfmacF3216x16x64F16Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA: V_SMFMAC_F32_16X16X64_F16 — 16x16x64 F16→F32
  // D(4 regs/lane) += A * B, functional model
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a_raw = src0.read_lane(wf, lane);
    uint32_t b_raw = src1.read_lane(wf, lane);
    float a0 = util::f16_to_f32(static_cast<uint16_t>(a_raw));
    float a1 = util::f16_to_f32(static_cast<uint16_t>(a_raw >> 16));
    float b0 = util::f16_to_f32(static_cast<uint16_t>(b_raw));
    float b1 = util::f16_to_f32(static_cast<uint16_t>(b_raw >> 16));
    float dot = a0 * b0 + a1 * b1;
    float acc = std::bit_cast<float>(src2.read_lane(wf, lane));
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(acc + dot));
  }
}

VSmfmacF3232x32x32F16Vop3pMfma::VSmfmacF3232x32x32F16Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_smfmac_f32_32x32x32_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(512, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(256, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VSmfmacF3232x32x32F16Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA: V_SMFMAC_F32_32X32X32_F16 — 32x32x32 F16→F32
  // D(16 regs/lane) += A * B, functional model
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a_raw = src0.read_lane(wf, lane);
    uint32_t b_raw = src1.read_lane(wf, lane);
    float a0 = util::f16_to_f32(static_cast<uint16_t>(a_raw));
    float a1 = util::f16_to_f32(static_cast<uint16_t>(a_raw >> 16));
    float b0 = util::f16_to_f32(static_cast<uint16_t>(b_raw));
    float b1 = util::f16_to_f32(static_cast<uint16_t>(b_raw >> 16));
    float dot = a0 * b0 + a1 * b1;
    float acc = std::bit_cast<float>(src2.read_lane(wf, lane));
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(acc + dot));
  }
}

VMfmaF3232x32x42bBf16Vop3pMfma::VMfmaF3232x32x42bBf16Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_32x32x4_2b_bf16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(1024, OperandType::OPR_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(1024, OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF3232x32x42bBf16Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t vb = wf.vgpr_alloc().base;
  uint32_t dst = mfma::dst_base(vb, vdst.encoding_value_);
  uint32_t const_acc;
  uint32_t s2 = mfma::resolve_acc(vb, dst, src2.encoding_value_, const_acc,
                                  [&] { return src2.read_scalar(wf); });
  mfma::exec_f32(cu, 32, 32, 4, 2, 16, dst, mfma::src_base(vb, src0.encoding_value_),
                 mfma::src_base(vb, src1.encoding_value_), s2, mfma::extract_bf16,
                 mfma::extract_bf16, const_acc);
}

VMfmaF3216x16x44bBf16Vop3pMfma::VMfmaF3216x16x44bBf16Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_16x16x4_4b_bf16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(512, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(512, OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF3216x16x44bBf16Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t vb = wf.vgpr_alloc().base;
  uint32_t dst = mfma::dst_base(vb, vdst.encoding_value_);
  uint32_t const_acc;
  uint32_t s2 = mfma::resolve_acc(vb, dst, src2.encoding_value_, const_acc,
                                  [&] { return src2.read_scalar(wf); });
  mfma::exec_f32(cu, 16, 16, 4, 4, 16, dst, mfma::src_base(vb, src0.encoding_value_),
                 mfma::src_base(vb, src1.encoding_value_), s2, mfma::extract_bf16,
                 mfma::extract_bf16, const_acc);
}

VMfmaF324x4x416bBf16Vop3pMfma::VMfmaF324x4x416bBf16Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_4x4x4_16b_bf16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF324x4x416bBf16Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t vb = wf.vgpr_alloc().base;
  uint32_t dst = mfma::dst_base(vb, vdst.encoding_value_);
  uint32_t const_acc;
  uint32_t s2 = mfma::resolve_acc(vb, dst, src2.encoding_value_, const_acc,
                                  [&] { return src2.read_scalar(wf); });
  mfma::exec_f32(cu, 4, 4, 4, 16, 16, dst, mfma::src_base(vb, src0.encoding_value_),
                 mfma::src_base(vb, src1.encoding_value_), s2, mfma::extract_bf16,
                 mfma::extract_bf16, const_acc);
}

VMfmaF3232x32x8Bf16Vop3pMfma::VMfmaF3232x32x8Bf16Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_32x32x8_bf16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(512, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(512, OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF3232x32x8Bf16Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t vb = wf.vgpr_alloc().base;
  uint32_t dst = mfma::dst_base(vb, vdst.encoding_value_);
  uint32_t const_acc;
  uint32_t s2 = mfma::resolve_acc(vb, dst, src2.encoding_value_, const_acc,
                                  [&] { return src2.read_scalar(wf); });
  mfma::exec_f32(cu, 32, 32, 8, 1, 16, dst, mfma::src_base(vb, src0.encoding_value_),
                 mfma::src_base(vb, src1.encoding_value_), s2, mfma::extract_bf16,
                 mfma::extract_bf16, const_acc);
}

VMfmaF3216x16x16Bf16Vop3pMfma::VMfmaF3216x16x16Bf16Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_16x16x16_bf16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF3216x16x16Bf16Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t vb = wf.vgpr_alloc().base;
  uint32_t dst = mfma::dst_base(vb, vdst.encoding_value_);
  uint32_t const_acc;
  uint32_t s2 = mfma::resolve_acc(vb, dst, src2.encoding_value_, const_acc,
                                  [&] { return src2.read_scalar(wf); });
  mfma::exec_f32(cu, 16, 16, 16, 1, 16, dst, mfma::src_base(vb, src0.encoding_value_),
                 mfma::src_base(vb, src1.encoding_value_), s2, mfma::extract_bf16,
                 mfma::extract_bf16, const_acc);
}

VSmfmacF3216x16x32F16Vop3pMfma::VSmfmacF3216x16x32F16Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_smfmac_f32_16x16x32_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VSmfmacF3216x16x32F16Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA: V_SMFMAC_F32_16X16X32_F16 — 16x16x32 F16→F32
  // D(4 regs/lane) += A * B, functional model
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a_raw = src0.read_lane(wf, lane);
    uint32_t b_raw = src1.read_lane(wf, lane);
    float a0 = util::f16_to_f32(static_cast<uint16_t>(a_raw));
    float a1 = util::f16_to_f32(static_cast<uint16_t>(a_raw >> 16));
    float b0 = util::f16_to_f32(static_cast<uint16_t>(b_raw));
    float b1 = util::f16_to_f32(static_cast<uint16_t>(b_raw >> 16));
    float dot = a0 * b0 + a1 * b1;
    float acc = std::bit_cast<float>(src2.read_lane(wf, lane));
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(acc + dot));
  }
}

VSmfmacF3232x32x16F16Vop3pMfma::VSmfmacF3232x32x16F16Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_smfmac_f32_32x32x16_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(512, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VSmfmacF3232x32x16F16Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA: V_SMFMAC_F32_32X32X16_F16 — 32x32x16 F16→F32
  // D(16 regs/lane) += A * B, functional model
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a_raw = src0.read_lane(wf, lane);
    uint32_t b_raw = src1.read_lane(wf, lane);
    float a0 = util::f16_to_f32(static_cast<uint16_t>(a_raw));
    float a1 = util::f16_to_f32(static_cast<uint16_t>(a_raw >> 16));
    float b0 = util::f16_to_f32(static_cast<uint16_t>(b_raw));
    float b1 = util::f16_to_f32(static_cast<uint16_t>(b_raw >> 16));
    float dot = a0 * b0 + a1 * b1;
    float acc = std::bit_cast<float>(src2.read_lane(wf, lane));
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(acc + dot));
  }
}

VSmfmacF3216x16x32Bf16Vop3pMfma::VSmfmacF3216x16x32Bf16Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_smfmac_f32_16x16x32_bf16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VSmfmacF3216x16x32Bf16Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA: V_SMFMAC_F32_16X16X32_BF16 — 16x16x32 BF16→F32
  // D(4 regs/lane) += A * B, functional model
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a_raw = src0.read_lane(wf, lane);
    uint32_t b_raw = src1.read_lane(wf, lane);
    float a0 = util::bf16_to_f32(static_cast<uint16_t>(a_raw));
    float a1 = util::bf16_to_f32(static_cast<uint16_t>(a_raw >> 16));
    float b0 = util::bf16_to_f32(static_cast<uint16_t>(b_raw));
    float b1 = util::bf16_to_f32(static_cast<uint16_t>(b_raw >> 16));
    float dot = a0 * b0 + a1 * b1;
    float acc = std::bit_cast<float>(src2.read_lane(wf, lane));
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(acc + dot));
  }
}

VSmfmacF3232x32x16Bf16Vop3pMfma::VSmfmacF3232x32x16Bf16Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_smfmac_f32_32x32x16_bf16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(512, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VSmfmacF3232x32x16Bf16Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA: V_SMFMAC_F32_32X32X16_BF16 — 32x32x16 BF16→F32
  // D(16 regs/lane) += A * B, functional model
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a_raw = src0.read_lane(wf, lane);
    uint32_t b_raw = src1.read_lane(wf, lane);
    float a0 = util::bf16_to_f32(static_cast<uint16_t>(a_raw));
    float a1 = util::bf16_to_f32(static_cast<uint16_t>(a_raw >> 16));
    float b0 = util::bf16_to_f32(static_cast<uint16_t>(b_raw));
    float b1 = util::bf16_to_f32(static_cast<uint16_t>(b_raw >> 16));
    float dot = a0 * b0 + a1 * b1;
    float acc = std::bit_cast<float>(src2.read_lane(wf, lane));
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(acc + dot));
  }
}

VSmfmacI3216x16x64I8Vop3pMfma::VSmfmacI3216x16x64I8Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_smfmac_i32_16x16x64_i8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VSmfmacI3216x16x64I8Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA: V_SMFMAC_I32_16X16X64_I8 — 16x16x64 I8→I32
  // D(4 regs/lane) += A * B, functional model
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a_raw = src0.read_lane(wf, lane);
    uint32_t b_raw = src1.read_lane(wf, lane);
    int32_t dot = 0;
    for (int k = 0; k < 8; ++k) {
      int8_t ae = static_cast<int8_t>((a_raw >> (k * 8)) & 0xFF);
      int8_t be = static_cast<int8_t>((b_raw >> (k * 8)) & 0xFF);
      dot += static_cast<int32_t>(ae) * be;
    }
    int32_t acc0 = static_cast<int32_t>(src2.read_lane(wf, lane));
    vdst.write_lane(wf, lane, static_cast<uint32_t>(acc0 + dot));
    // Additional result registers would require cross-lane data
  }
}

VSmfmacI3232x32x32I8Vop3pMfma::VSmfmacI3232x32x32I8Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_smfmac_i32_32x32x32_i8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(512, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VSmfmacI3232x32x32I8Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA: V_SMFMAC_I32_32X32X32_I8 — 32x32x32 I8→I32
  // D(16 regs/lane) += A * B, functional model
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a_raw = src0.read_lane(wf, lane);
    uint32_t b_raw = src1.read_lane(wf, lane);
    int32_t dot = 0;
    for (int k = 0; k < 8; ++k) {
      int8_t ae = static_cast<int8_t>((a_raw >> (k * 8)) & 0xFF);
      int8_t be = static_cast<int8_t>((b_raw >> (k * 8)) & 0xFF);
      dot += static_cast<int32_t>(ae) * be;
    }
    int32_t acc0 = static_cast<int32_t>(src2.read_lane(wf, lane));
    vdst.write_lane(wf, lane, static_cast<uint32_t>(acc0 + dot));
    // Additional result registers would require cross-lane data
  }
}

VMfmaF6416x16x4F64Vop3pMfma::VMfmaF6416x16x4F64Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f64_16x16x4_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(256, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(256, OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF6416x16x4F64Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t vb = wf.vgpr_alloc().base;
  uint32_t dst = mfma::dst_base(vb, vdst.encoding_value_);
  uint32_t const_acc;
  uint32_t s2 = mfma::resolve_acc(vb, dst, src2.encoding_value_, const_acc,
                                  [&] { return src2.read_scalar(wf); });
  mfma::exec_f64(cu, 16, 16, 4, 1, dst, mfma::src_base(vb, src0.encoding_value_),
                 mfma::src_base(vb, src1.encoding_value_), s2, const_acc);
}

VMfmaF644x4x44bF64Vop3pMfma::VMfmaF644x4x44bF64Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f64_4x4x4_4b_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF644x4x44bF64Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t vb = wf.vgpr_alloc().base;
  uint32_t dst = mfma::dst_base(vb, vdst.encoding_value_);
  uint32_t const_acc;
  uint32_t s2 = mfma::resolve_acc(vb, dst, src2.encoding_value_, const_acc,
                                  [&] { return src2.read_scalar(wf); });
  mfma::exec_f64(cu, 4, 4, 4, 4, dst, mfma::src_base(vb, src0.encoding_value_),
                 mfma::src_base(vb, src1.encoding_value_), s2, const_acc);
}

VMfmaF3216x16x32Bf8Bf8Vop3pMfma::VMfmaF3216x16x32Bf8Bf8Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_16x16x32_bf8_bf8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF3216x16x32Bf8Bf8Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t vb = wf.vgpr_alloc().base;
  uint32_t dst = mfma::dst_base(vb, vdst.encoding_value_);
  uint32_t const_acc;
  uint32_t s2 = mfma::resolve_acc(vb, dst, src2.encoding_value_, const_acc,
                                  [&] { return src2.read_scalar(wf); });
  mfma::exec_f32(cu, 16, 16, 32, 1, 8, dst, mfma::src_base(vb, src0.encoding_value_),
                 mfma::src_base(vb, src1.encoding_value_), s2, mfma::extract_bf8, mfma::extract_bf8,
                 const_acc);
}

VMfmaF3216x16x32Bf8Fp8Vop3pMfma::VMfmaF3216x16x32Bf8Fp8Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_16x16x32_bf8_fp8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF3216x16x32Bf8Fp8Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t vb = wf.vgpr_alloc().base;
  uint32_t dst = mfma::dst_base(vb, vdst.encoding_value_);
  uint32_t const_acc;
  uint32_t s2 = mfma::resolve_acc(vb, dst, src2.encoding_value_, const_acc,
                                  [&] { return src2.read_scalar(wf); });
  mfma::exec_f32(cu, 16, 16, 32, 1, 8, dst, mfma::src_base(vb, src0.encoding_value_),
                 mfma::src_base(vb, src1.encoding_value_), s2, mfma::extract_bf8, mfma::extract_fp8,
                 const_acc);
}

VMfmaF3216x16x32Fp8Bf8Vop3pMfma::VMfmaF3216x16x32Fp8Bf8Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_16x16x32_fp8_bf8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF3216x16x32Fp8Bf8Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t vb = wf.vgpr_alloc().base;
  uint32_t dst = mfma::dst_base(vb, vdst.encoding_value_);
  uint32_t const_acc;
  uint32_t s2 = mfma::resolve_acc(vb, dst, src2.encoding_value_, const_acc,
                                  [&] { return src2.read_scalar(wf); });
  mfma::exec_f32(cu, 16, 16, 32, 1, 8, dst, mfma::src_base(vb, src0.encoding_value_),
                 mfma::src_base(vb, src1.encoding_value_), s2, mfma::extract_fp8, mfma::extract_bf8,
                 const_acc);
}

VMfmaF3216x16x32Fp8Fp8Vop3pMfma::VMfmaF3216x16x32Fp8Fp8Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_16x16x32_fp8_fp8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF3216x16x32Fp8Fp8Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t vb = wf.vgpr_alloc().base;
  uint32_t dst = mfma::dst_base(vb, vdst.encoding_value_);
  uint32_t const_acc;
  uint32_t s2 = mfma::resolve_acc(vb, dst, src2.encoding_value_, const_acc,
                                  [&] { return src2.read_scalar(wf); });
  mfma::exec_f32(cu, 16, 16, 32, 1, 8, dst, mfma::src_base(vb, src0.encoding_value_),
                 mfma::src_base(vb, src1.encoding_value_), s2, mfma::extract_fp8, mfma::extract_fp8,
                 const_acc);
}

VMfmaF3232x32x16Bf8Bf8Vop3pMfma::VMfmaF3232x32x16Bf8Bf8Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_32x32x16_bf8_bf8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(512, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(512, OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF3232x32x16Bf8Bf8Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t vb = wf.vgpr_alloc().base;
  uint32_t dst = mfma::dst_base(vb, vdst.encoding_value_);
  uint32_t const_acc;
  uint32_t s2 = mfma::resolve_acc(vb, dst, src2.encoding_value_, const_acc,
                                  [&] { return src2.read_scalar(wf); });
  mfma::exec_f32(cu, 32, 32, 16, 1, 8, dst, mfma::src_base(vb, src0.encoding_value_),
                 mfma::src_base(vb, src1.encoding_value_), s2, mfma::extract_bf8, mfma::extract_bf8,
                 const_acc);
}

VMfmaF3232x32x16Bf8Fp8Vop3pMfma::VMfmaF3232x32x16Bf8Fp8Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_32x32x16_bf8_fp8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(512, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(512, OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF3232x32x16Bf8Fp8Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t vb = wf.vgpr_alloc().base;
  uint32_t dst = mfma::dst_base(vb, vdst.encoding_value_);
  uint32_t const_acc;
  uint32_t s2 = mfma::resolve_acc(vb, dst, src2.encoding_value_, const_acc,
                                  [&] { return src2.read_scalar(wf); });
  mfma::exec_f32(cu, 32, 32, 16, 1, 8, dst, mfma::src_base(vb, src0.encoding_value_),
                 mfma::src_base(vb, src1.encoding_value_), s2, mfma::extract_bf8, mfma::extract_fp8,
                 const_acc);
}

VMfmaF3232x32x16Fp8Bf8Vop3pMfma::VMfmaF3232x32x16Fp8Bf8Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_32x32x16_fp8_bf8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(512, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(512, OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF3232x32x16Fp8Bf8Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t vb = wf.vgpr_alloc().base;
  uint32_t dst = mfma::dst_base(vb, vdst.encoding_value_);
  uint32_t const_acc;
  uint32_t s2 = mfma::resolve_acc(vb, dst, src2.encoding_value_, const_acc,
                                  [&] { return src2.read_scalar(wf); });
  mfma::exec_f32(cu, 32, 32, 16, 1, 8, dst, mfma::src_base(vb, src0.encoding_value_),
                 mfma::src_base(vb, src1.encoding_value_), s2, mfma::extract_fp8, mfma::extract_bf8,
                 const_acc);
}

VMfmaF3232x32x16Fp8Fp8Vop3pMfma::VMfmaF3232x32x16Fp8Fp8Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_mfma_f32_32x32x16_fp8_fp8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(512, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(512, OperandType::OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST,
           reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMfmaF3232x32x16Fp8Fp8Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t vb = wf.vgpr_alloc().base;
  uint32_t dst = mfma::dst_base(vb, vdst.encoding_value_);
  uint32_t const_acc;
  uint32_t s2 = mfma::resolve_acc(vb, dst, src2.encoding_value_, const_acc,
                                  [&] { return src2.read_scalar(wf); });
  mfma::exec_f32(cu, 32, 32, 16, 1, 8, dst, mfma::src_base(vb, src0.encoding_value_),
                 mfma::src_base(vb, src1.encoding_value_), s2, mfma::extract_fp8, mfma::extract_fp8,
                 const_acc);
}

VSmfmacF3216x16x64Bf8Bf8Vop3pMfma::VSmfmacF3216x16x64Bf8Bf8Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_smfmac_f32_16x16x64_bf8_bf8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VSmfmacF3216x16x64Bf8Bf8Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA: V_SMFMAC_F32_16X16X64_BF8_BF8 — 16x16x64 BF8_BF8→F32
  // D(4 regs/lane) += A * B, functional model
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a_raw = src0.read_lane(wf, lane);
    uint32_t b_raw = src1.read_lane(wf, lane);
    float dot = 0.0f;
    for (int k = 0; k < 4; ++k) {
      float ae = util::bf8_e5m2_to_f32(static_cast<uint8_t>((a_raw >> (k * 8)) & 0xFF));
      float be = util::bf8_e5m2_to_f32(static_cast<uint8_t>((b_raw >> (k * 8)) & 0xFF));
      dot += ae * be;
    }
    float acc = std::bit_cast<float>(src2.read_lane(wf, lane));
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(acc + dot));
  }
}

VSmfmacF3216x16x64Bf8Fp8Vop3pMfma::VSmfmacF3216x16x64Bf8Fp8Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_smfmac_f32_16x16x64_bf8_fp8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VSmfmacF3216x16x64Bf8Fp8Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA: V_SMFMAC_F32_16X16X64_BF8_FP8 — 16x16x64 BF8_FP8→F32
  // D(4 regs/lane) += A * B, functional model
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a_raw = src0.read_lane(wf, lane);
    uint32_t b_raw = src1.read_lane(wf, lane);
    float dot = 0.0f;
    for (int k = 0; k < 4; ++k) {
      float ae = util::bf8_e5m2_to_f32(static_cast<uint8_t>((a_raw >> (k * 8)) & 0xFF));
      float be = util::fp8_e4m3_to_f32(static_cast<uint8_t>((b_raw >> (k * 8)) & 0xFF));
      dot += ae * be;
    }
    float acc = std::bit_cast<float>(src2.read_lane(wf, lane));
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(acc + dot));
  }
}

VSmfmacF3216x16x64Fp8Bf8Vop3pMfma::VSmfmacF3216x16x64Fp8Bf8Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_smfmac_f32_16x16x64_fp8_bf8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VSmfmacF3216x16x64Fp8Bf8Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA: V_SMFMAC_F32_16X16X64_FP8_BF8 — 16x16x64 FP8_BF8→F32
  // D(4 regs/lane) += A * B, functional model
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a_raw = src0.read_lane(wf, lane);
    uint32_t b_raw = src1.read_lane(wf, lane);
    float dot = 0.0f;
    for (int k = 0; k < 4; ++k) {
      float ae = util::fp8_e4m3_to_f32(static_cast<uint8_t>((a_raw >> (k * 8)) & 0xFF));
      float be = util::bf8_e5m2_to_f32(static_cast<uint8_t>((b_raw >> (k * 8)) & 0xFF));
      dot += ae * be;
    }
    float acc = std::bit_cast<float>(src2.read_lane(wf, lane));
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(acc + dot));
  }
}

VSmfmacF3216x16x64Fp8Fp8Vop3pMfma::VSmfmacF3216x16x64Fp8Fp8Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_smfmac_f32_16x16x64_fp8_fp8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VSmfmacF3216x16x64Fp8Fp8Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA: V_SMFMAC_F32_16X16X64_FP8_FP8 — 16x16x64 FP8_FP8→F32
  // D(4 regs/lane) += A * B, functional model
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a_raw = src0.read_lane(wf, lane);
    uint32_t b_raw = src1.read_lane(wf, lane);
    float dot = 0.0f;
    for (int k = 0; k < 4; ++k) {
      float ae = util::fp8_e4m3_to_f32(static_cast<uint8_t>((a_raw >> (k * 8)) & 0xFF));
      float be = util::fp8_e4m3_to_f32(static_cast<uint8_t>((b_raw >> (k * 8)) & 0xFF));
      dot += ae * be;
    }
    float acc = std::bit_cast<float>(src2.read_lane(wf, lane));
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(acc + dot));
  }
}

VSmfmacF3232x32x32Bf8Bf8Vop3pMfma::VSmfmacF3232x32x32Bf8Bf8Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_smfmac_f32_32x32x32_bf8_bf8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(512, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VSmfmacF3232x32x32Bf8Bf8Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA: V_SMFMAC_F32_32X32X32_BF8_BF8 — 32x32x32 BF8_BF8→F32
  // D(16 regs/lane) += A * B, functional model
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a_raw = src0.read_lane(wf, lane);
    uint32_t b_raw = src1.read_lane(wf, lane);
    float dot = 0.0f;
    for (int k = 0; k < 4; ++k) {
      float ae = util::bf8_e5m2_to_f32(static_cast<uint8_t>((a_raw >> (k * 8)) & 0xFF));
      float be = util::bf8_e5m2_to_f32(static_cast<uint8_t>((b_raw >> (k * 8)) & 0xFF));
      dot += ae * be;
    }
    float acc = std::bit_cast<float>(src2.read_lane(wf, lane));
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(acc + dot));
  }
}

VSmfmacF3232x32x32Bf8Fp8Vop3pMfma::VSmfmacF3232x32x32Bf8Fp8Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_smfmac_f32_32x32x32_bf8_fp8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(512, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VSmfmacF3232x32x32Bf8Fp8Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA: V_SMFMAC_F32_32X32X32_BF8_FP8 — 32x32x32 BF8_FP8→F32
  // D(16 regs/lane) += A * B, functional model
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a_raw = src0.read_lane(wf, lane);
    uint32_t b_raw = src1.read_lane(wf, lane);
    float dot = 0.0f;
    for (int k = 0; k < 4; ++k) {
      float ae = util::bf8_e5m2_to_f32(static_cast<uint8_t>((a_raw >> (k * 8)) & 0xFF));
      float be = util::fp8_e4m3_to_f32(static_cast<uint8_t>((b_raw >> (k * 8)) & 0xFF));
      dot += ae * be;
    }
    float acc = std::bit_cast<float>(src2.read_lane(wf, lane));
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(acc + dot));
  }
}

VSmfmacF3232x32x32Fp8Bf8Vop3pMfma::VSmfmacF3232x32x32Fp8Bf8Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_smfmac_f32_32x32x32_fp8_bf8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(512, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VSmfmacF3232x32x32Fp8Bf8Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA: V_SMFMAC_F32_32X32X32_FP8_BF8 — 32x32x32 FP8_BF8→F32
  // D(16 regs/lane) += A * B, functional model
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a_raw = src0.read_lane(wf, lane);
    uint32_t b_raw = src1.read_lane(wf, lane);
    float dot = 0.0f;
    for (int k = 0; k < 4; ++k) {
      float ae = util::fp8_e4m3_to_f32(static_cast<uint8_t>((a_raw >> (k * 8)) & 0xFF));
      float be = util::bf8_e5m2_to_f32(static_cast<uint8_t>((b_raw >> (k * 8)) & 0xFF));
      dot += ae * be;
    }
    float acc = std::bit_cast<float>(src2.read_lane(wf, lane));
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(acc + dot));
  }
}

VSmfmacF3232x32x32Fp8Fp8Vop3pMfma::VSmfmacF3232x32x32Fp8Fp8Vop3pMfma(const MachineInst *inst)
    : Vop3pMfma("v_smfmac_f32_32x32x32_fp8_fp8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(512, OperandType::OPR_VGPR_OR_ACCVGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(128, OperandType::OPR_SRC_VGPR_OR_ACCVGPR,
           reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VSmfmacF3232x32x32Fp8Fp8Vop3pMfma::execute(amdgpu::Wavefront &wf) {
  // MFMA: V_SMFMAC_F32_32X32X32_FP8_FP8 — 32x32x32 FP8_FP8→F32
  // D(16 regs/lane) += A * B, functional model
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a_raw = src0.read_lane(wf, lane);
    uint32_t b_raw = src1.read_lane(wf, lane);
    float dot = 0.0f;
    for (int k = 0; k < 4; ++k) {
      float ae = util::fp8_e4m3_to_f32(static_cast<uint8_t>((a_raw >> (k * 8)) & 0xFF));
      float be = util::fp8_e4m3_to_f32(static_cast<uint8_t>((b_raw >> (k * 8)) & 0xFF));
      dot += ae * be;
    }
    float acc = std::bit_cast<float>(src2.read_lane(wf, lane));
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(acc + dot));
  }
}

} // namespace cdna4
} // namespace rocjitsu
